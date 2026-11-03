"""连续视频流水线：接收、写盘、解码、语义识别各自独立，GUI 只轮询最新结果。"""

from __future__ import annotations

import faulthandler
import json
import queue
import threading
from copy import deepcopy
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from time import monotonic
from uuid import uuid4

import av

from doudizhu_lab import __version__
from doudizhu_lab.devices.adb import AdbClient
from doudizhu_lab.devices.video import TIME_BASE, SegmentWriter, setup_video
from doudizhu_lab.domain.models import Frame
from doudizhu_lab.runtime.analysis import MediaSampler, OrderedAnalyzer
from doudizhu_lab.runtime.capture import CaptureProcess, foreground
from doudizhu_lab.runtime.media_source import VideoSource
from doudizhu_lab.storage.config import Store
from doudizhu_lab.storage.evidence import EvidenceWriter
from doudizhu_lab.storage.retention import (
    RecordingResult,
    RetentionLedger,
    StopReason,
    atomic_json,
    finalize_retention,
)
from doudizhu_lab.storage.session import SessionStore, reserve_acceptance, snapshot_template_dirs
from doudizhu_lab.vision.factory import create_recognizer
from doudizhu_lab.vision.table import MarkerLibrary, TableRecognizer


def game_is_foreground(window_dump: str, power_dump: str) -> bool:
    """只接受清醒手机上明确属于游戏的焦点窗口；兼容 HyperOS 的多个显示器条目。"""
    return foreground(window_dump, power_dump)


class RecordingPipeline:
    """一条手机视频会话；对象本身不创建 Qt 对象，也不发送任何输入。

    start 在后台调用，poll 可从 GUI 调用，stop 等待录像尾部落盘。
    raw 编码队列有界但不静默丢录像；实时解码落后可跳至下一关键帧，原录像用于离线补算。
    """

    def __init__(
        self,
        store: Store,
        serial: str,
        *,
        analyze: bool = True,
        acceptance: bool = False,
        stream: VideoSource | None = None,
        automatic: bool = False,
    ):
        """创建流水线；显式replay媒体源不准备手机服务，也不创建输入客户端。

        真机保持固定scrcpy校验。注入流仅用于独立离线子类，必须使用replay标识；
        其画面transport为replay，输入闸门会拒绝。离线子类需提供不查询设备的_guard。
        automatic仅供日常自动跟踪；显式录制、离线回放及正式验收完整保留。
        构造不启动线程，start负责启动，stop负责关闭自己持有的流。
        """
        if stream is not None and (serial != "replay" or acceptance):
            raise ValueError("注入媒体源仅供离线回放，不能标为真机验收")
        self.store, self.serial, self.analyze = store, serial, analyze
        self.profile = deepcopy(store.profile)
        server_path = setup_video(store.root, download=False) if stream is None else None
        self.session = SessionStore.create(
            store.root,
            store.profile,
            snapshot_template_dirs(store),
            serial,
            model_root=store.root,
        )
        if acceptance:
            try:
                self.session.set_meta("acceptance", reserve_acceptance(store, self.session))
            except Exception as exc:
                self.session.set_meta("status", "failed")
                self.session.add_issue({"code": "startup", "message": str(exc)})
                self.session.close()
                raise
        self.client = AdbClient(store.adb_path, serial) if stream is None else None
        # 只读dumpsys不占用输入客户端的锁；所有真实输入仍由全局InputGate串行授权。
        self.control_client = AdbClient(store.adb_path, serial, timeout=3) if stream is None else None
        self.remote_capture = stream is None
        self.acceptance = acceptance
        self.automatic = automatic and not acceptance and self.remote_capture
        self.stream = (
            CaptureProcess(self.session.folder, store.adb_path, serial, server_path, automatic=self.automatic)
            if self.remote_capture
            else stream
        )
        self.writer = (
            None
            if self.remote_capture
            else SegmentWriter(self.session.folder / "video", quota_root=store.root / "sessions")
        )
        self.retention = None if self.remote_capture else RetentionLedger(self.session.folder)
        self._health_stop = threading.Event()
        self._health_lock = threading.Lock()
        self._diagnostic_lock = threading.Lock()
        self._gui_at = None
        self._stages = {}
        self._faults = {}
        self._fault_details = []
        self._completed_games = set()
        self._last_health = {}
        self._dumped = False
        self._stack_file = None
        self.result = None
        self.stop_event = threading.Event()
        self.ready = threading.Event()
        self.record_ready = threading.Event()
        self.closed = False
        self.failed = False
        self.threads = []
        self.write_queue = queue.Queue(maxsize=256)
        self.decode_queue = queue.Queue(maxsize=32)
        self.wake_analysis = threading.Event()
        self.messages = queue.Queue()
        self.ui_events = queue.Queue(maxsize=1000)
        self.gaps = queue.Queue()
        self.frame = self.observation = self.recognition = self.state = None
        self.allowed = False
        self._sequence = 0
        self._codec_epoch = 0
        self._pts = 0
        self._epoch = uuid4().hex[:8]
        self._offset = None
        self._started = monotonic()
        self._guard_reason = "等待确认游戏处于前台"
        self._gap_lock = threading.Lock()
        self._open_gaps = {}
        self.pending_candidates = []
        self._snapshot_lock = threading.Lock()
        self._bot_snapshot = None
        self._flow_snapshot = None
        self._flow_started = False
        self._last_card_enabled = False
        from doudizhu_lab.runtime.operational import OperationalObserver

        self.operational = (
            OperationalObserver(self.profile) if self.profile.continuous_policy != "strict" else None
        )
        self.sampler = MediaSampler(10)
        # 两条识别线程各保留一个在途任务，不再积压旧帧；忙时跳分析，原录像完整保存。
        self.analysis_pool = OrderedAnalyzer(self._recognizer_factory, capacity=2) if analyze else None
        self.evidence = EvidenceWriter(self.session.folder / "evidence", on_error=self._evidence_error)
        self.metrics = {
            "received_packets": 0,
            "decoded_frames": 0,
            "analyzed_frames": 0,
            "encoded_bytes": 0,
            "decode_resets": 0,
            "max_write_queue": 0,
            "max_decode_queue": 0,
            "analysis_ms_total": 0.0,
            "analysis_latency_ms_max": 0.0,
            "analysis_skipped_busy": 0,
            "max_analysis_queue": 0,
            "queue_samples": [],
        }

    def _recognizer_factory(self):
        """每个工作线程持有独立模板缓存，所有线程使用同一冻结快照和只读配置。"""
        recognizer = TableRecognizer(
            create_recognizer(
                self.session.folder / "templates" / "templates",
                self.profile,
                model_root=self.session.folder,
                include_bundled=False,
            ),
            MarkerLibrary(self.session.folder / "templates" / "markers"),
        )
        return lambda frame: recognizer.observe(frame, self.profile)

    def stage_started(self, name: str) -> None:
        """登记有实际工作在途的阶段；重复输入不会刷新最早等待时间，正常等待不调用。"""
        with self._health_lock:
            stage = self._stages.setdefault(name, {"pending_at": None, "output_at": None})
            if stage["pending_at"] is None:
                stage["pending_at"] = monotonic()

    def stage_finished(self, name: str, *, pending=False) -> None:
        """登记一次完整处理结果；仍有排队工作时从本次进度继续计时，保留初始化状态。"""
        with self._health_lock:
            self._stages[name] = {"pending_at": monotonic() if pending else None, "output_at": monotonic()}

    def health_snapshot(self) -> dict:
        """复制不含原始像素的诊断信息；不取得SQLite锁，避免主账本阻塞拖住独立监控。"""
        with self._health_lock:
            stages = {k: dict(v) for k, v in self._stages.items()}
        state = self.state
        return {
            "reported_at": monotonic(),
            "gui_at": self._gui_at,
            "stages": stages,
            "pts_us": self._pts,
            "metrics": {k: v for k, v in self.metrics.items() if k != "queue_samples"},
            "state": asdict(state) if is_dataclass(state) else state,
            "queues": {
                "decode": self.decode_queue.qsize(),
                "analysis": self.analysis_pool.size if self.analysis_pool else 0,
            },
            "channel_warnings": self.stream.channel.snapshot() if self.remote_capture else {},
            "capture_lifecycle": self._last_health.get("lifecycle"),
        }

    def report_failure(self, code: str, message: str, *, occurred_at: float | None = None) -> None:
        """先固定故障及最近状态，再尝试截图和堆栈；用户停止不能撤销故障。

        occurred_at仅用于传递本机录制进程已记录的单调故障时间，默认取当前时刻；
        按发生时间展示首因，不能用跨进程交付顺序覆盖更早的真实故障。不发送输入。
        """
        with self._health_lock:
            if code in self._faults:
                return
            detail = {
                "code": code,
                "message": message,
                "occurred_at": monotonic() if occurred_at is None else occurred_at,
                "pts_us": self._pts,
            }
            self._faults[code] = message
            self._fault_details.append(detail)
            self._fault_details.sort(key=lambda f: f["occurred_at"])
            self.failed = True
        health = self.health_snapshot()
        if self.remote_capture:
            self.stream.command(
                "fault", code=code, message=message, health=health, occurred_at=detail["occurred_at"]
            )
        else:
            self.retention.last_pts = self._pts
            self.retention.fault(code, message, health=health, occurred_at=detail["occurred_at"])
        self.messages.put(message)
        try:
            # 发布单独串行化，不能持有健康锁等待磁盘；每次实际发布前重新取完整
            # 故障集，防止较早回调迟到覆盖已经登记的首因或后续故障。
            with self._diagnostic_lock:
                with self._health_lock:
                    faults, details = dict(self._faults), list(self._fault_details)
                atomic_json(
                    self.session.folder / "故障诊断.json",
                    {
                        "faults": faults,
                        "primary_fault": details[0],
                        "secondary_faults": details[1:],
                        "health": health,
                    },
                )
            if self.frame is not None:
                self.evidence.submit(self.frame, ["runtime-fault"])
            if not self._dumped:
                self._dumped = True
                with (self.session.folder / "线程堆栈.txt").open("a", encoding="utf-8") as output:
                    faulthandler.dump_traceback(file=output, all_threads=True)
        except (OSError, ValueError):
            pass  # 独立录制进程仍保存自己的故障账本，不能让截图失败掩盖原故障。

    def begin_game(self, game_id: str) -> None:
        """绑定当前临时录像窗口；独立录制进程保留此前的大厅、匹配和叫抢上下文。"""
        if self.remote_capture:
            self.stream.command("game_start", game_id=game_id)

    def enable_automatic_retention(self) -> None:
        """连续任务启动时启用日常留存策略；正式验收和离线源始终保留完整媒体。"""
        if self.remote_capture and not self.acceptance and not self.automatic:
            self.automatic = True
            self.stream.command("automatic")

    def complete_game(self, game_id: str, pts_us: int) -> None:
        """按正常跟踪完成登记清理边界，与输赢及任务局数门槛无关；重复调用幂等。"""
        if game_id in self._completed_games or self.failed:
            return
        self._completed_games.add(game_id)
        if self.remote_capture:
            self.stream.command("game_complete", game_id=game_id, pts_us=pts_us)

    def abort_recording_window(self) -> None:
        """仅供显式用户暂停清理未完成窗口；系统异常暂停必须走故障上报接口。"""
        if self.remote_capture:
            self.stream.command("user_abort")

    def _health_report(self):
        """每秒跨进程报告健康信息；独立采集故障立即反映为失败，GUI据此撤销输入。"""
        last_gui = None
        while not self._health_stop.is_set():
            try:
                if self._gui_at is not None and self._gui_at != last_gui and not self.failed:
                    # 原生超时转储不依赖Python线程取得GIL；主界面或全部Python线程阻塞
                    # 时，仍有机会在独立录像进程保存的画面之外留下调用栈。
                    last_gui = self._gui_at
                    faulthandler.dump_traceback_later(3, file=self._stack_file)
                self.stream.heartbeat(self.health_snapshot())
                status = self.stream.status()
                self._last_health = status
                if status.get("started") and monotonic() - status.get("updated_at", monotonic()) > 3:
                    self.report_failure("capture_unresponsive", "独立录像进程超过三秒未报告状态")
                self.allowed = status.get("allowed", False)
                self.record_ready.set() if status.get("record_ready") else self.record_ready.clear()
                for fault in status.get("faults", []):
                    self.report_failure(fault["code"], fault["message"], occurred_at=fault.get("occurred_at"))
                if self.stream.process and self.stream.process.poll() is not None and not self.closed:
                    self.report_failure("capture_exit", "独立录像进程提前退出")
            except Exception as exc:
                self.report_failure("health_report", f"录像健康报告失败：{exc}")
            self._health_stop.wait(1)

    def enable_flow(self):
        """显式启动独立、最多2FPS的流程识别；全屏OCR不阻塞原有牌桌识别和录像线程。"""
        from doudizhu_lab.vision.flow import FlowRecognizer, select_flow_source

        with self._snapshot_lock:
            if self._flow_started:
                return
            self._flow_started = True
        self.session.snapshot_vision_models(self.store.root)
        recognizer = FlowRecognizer(self.session.folder)
        self.session.set_meta("flow_enabled", True)
        # 快速“不出”有时只稳定约300ms。旧固定刻度可能选中两个相隔95ms的帧，
        # 第三帧提示已消失；连续模式使用至少100ms的采样间隔，不放宽确认规则。
        self.sampler = MediaSampler(10, minimum_spacing=True)
        self.session.set_meta("analysis_sampling", {"fps_limit": 10, "minimum_spacing": True})
        self._last_card_enabled = True

        def observe_flow():
            last = ""
            last_pts = -1
            try:
                while not self.stop_event.wait(0.05):
                    snapshot = self.bot_snapshot()
                    frame, table, ended, deal_id = select_flow_source(snapshot, self.frame)
                    if (
                        frame is None
                        or frame.frame_id == last
                        or frame.media_pts_us is None
                        or frame.media_pts_us <= last_pts
                    ):
                        continue
                    last, last_pts = frame.frame_id, frame.media_pts_us
                    self.stage_started("流程识别")
                    result = recognizer.observe(
                        frame,
                        table,
                        deal_id=deal_id,
                        ended=ended,
                        tracked_phase=(snapshot.frame.frame_id, snapshot.state["phase"])
                        if snapshot
                        else None,
                    )
                    with self._snapshot_lock:
                        self._flow_snapshot = result
                    self.session.set_meta(
                        "flow_metrics",
                        {
                            "frame_id": frame.frame_id,
                            "result_age_ms": frame.age * 1000,
                            "table_same_frame": table is not None,
                        },
                    )
                    self.stage_finished("流程识别")
                    self.stop_event.wait(0.45)
            except Exception as exc:
                self.messages.put(f"连续流程识别失败：{exc}")
                self._issue("flow_analysis", f"流程识别中断：{exc}")

        self._spawn("连续页面识别", observe_flow)

    def flow_snapshot(self):
        """返回同帧流程观察；不把最新原图与较早页面OCR重新拼成伪新帧。"""
        with self._snapshot_lock:
            return self._flow_snapshot

    def _spawn(self, name, target):
        thread = threading.Thread(target=target, name=name, daemon=True)
        self.threads.append(thread)
        thread.start()

    def start(self):
        """开启后台流水线；启动失败关闭本会话并向调用者返回明确错误。"""
        try:
            self.stream.start()
            self.evidence.start()
            if self.remote_capture:
                self._stack_file = (self.session.folder / "线程堆栈.txt").open("a", encoding="utf-8")
                self._spawn("录像健康报告", self._health_report)
            else:
                self._spawn("录像写盘", self._write)
            self._spawn("视频解码", self._decode)
            if not self.remote_capture:
                self._spawn("游戏前台检查", self._guard)
            if self.analyze:
                self._spawn("牌桌识别", self._analyze)
            self._spawn("视频接收", self._receive)
            self.ready.set()
        except Exception as exc:
            self._issue("startup", str(exc), fatal=True)
            self.stop()
            raise

    def _issue(self, code, message, *, fatal=False):
        """记录可审计缺口，不能把掉线/写盘失败归类为识别成功。"""
        if fatal or code in ("analysis", "flow_analysis", "device", "foreground", "orientation"):
            self.report_failure(code, message)
        issue = {
            "issue_id": uuid4().hex,
            "game_id": getattr(self.state, "game_id", ""),
            "pts_us": self._pts,
            "code": code,
            "message": message,
            "start_us": self._pts,
            "end_us": None,
            "scope": "analysis" if code in ("analysis", "decode_backlog", "analysis_decode") else "capture",
        }
        with self._gap_lock:
            previous = self._open_gaps.get(code)
            if previous is None:
                self._open_gaps[code] = issue["issue_id"]
        if previous is not None:
            self.messages.put(message)
            if fatal:
                self.failed = True
                self.stop_event.set()
            return
        try:
            self.session.add_issue(issue)
        except Exception:
            pass
        self.messages.put(message)
        self.gaps.put((issue["start_us"], message, issue["scope"] == "capture"))
        if fatal:
            self.failed = True
            self.stop_event.set()

    def _close_gaps(self, codes, pts_us):
        """用追加记录关闭明确恢复的缺口；原疑点及其开始时间保持不变。"""
        with self._gap_lock:
            closed = [(code, self._open_gaps.pop(code)) for code in codes if code in self._open_gaps]
        for code, identity in closed:
            self.session.add_issue(
                {
                    "code": "gap_closed",
                    "gap_id": identity,
                    "pts_us": pts_us,
                    "end_us": pts_us,
                    "scope": "info",
                    "message": f"{code} 缺口结束",
                }
            )

    def _evidence_error(self, frame_id, message):
        """截图故障独立于录像缺口；源视频仍可用于核对，不能因此伪造保存成功。"""
        self.messages.put(message)
        try:
            self.session.add_issue(
                {
                    "code": "evidence_failed",
                    "scope": "evidence",
                    "frame_id": frame_id,
                    "pts_us": self._pts,
                    "message": message,
                }
            )
        except Exception:
            pass

    def _receive(self):
        reset_decoder = False
        last_session = None
        config = None
        try:
            for packet in self.stream.packets():
                if self.stop_event.is_set():
                    break
                if packet.session:
                    if last_session is not None and (
                        not self.remote_capture
                        or (last_session.width, last_session.height) != (packet.width, packet.height)
                    ):
                        self._issue("orientation", "视频尺寸/编码会话变化，已分段并标记缺口")
                    last_session, config = packet, None
                elif packet.config:
                    config = packet
                else:
                    self._pts = packet.pts_us
                self.metrics["received_packets"] += 1
                self.metrics["encoded_bytes"] += len(packet.data)
                if not self.remote_capture:
                    try:
                        self.write_queue.put((packet, self.allowed), timeout=1)
                    except queue.Full:
                        self._issue("disk_backlog", "录像写盘积压，已停止完整记录", fatal=True)
                        break
                if reset_decoder:
                    if packet.keyframe and last_session and config:
                        for item in (last_session, config, packet):
                            self.decode_queue.put(item, timeout=1)
                        reset_decoder = False
                else:
                    try:
                        self.decode_queue.put_nowait(packet)
                    except queue.Full:
                        while True:
                            try:
                                self.decode_queue.get_nowait()
                            except queue.Empty:
                                break
                        reset_decoder = True
                        self.metrics["decode_resets"] += 1
                        self._issue("decode_backlog", "实时解码落后，等待关键帧；请用录像重新分析该区间")
                self.metrics["max_write_queue"] = max(
                    self.metrics["max_write_queue"], self.write_queue.qsize()
                )
                self.metrics["max_decode_queue"] = max(
                    self.metrics["max_decode_queue"], self.decode_queue.qsize()
                )
        except Exception as exc:
            if not self.stop_event.is_set():
                self._issue("transport", f"视频传输中断：{exc}", fatal=True)
        finally:
            for target in (self.write_queue, self.decode_queue):
                try:
                    target.put(None, timeout=2)
                except queue.Full:
                    pass
            self.stop_event.set()
            self.wake_analysis.set()

    def _write(self):
        previous = False
        try:
            while True:
                try:
                    item = self.write_queue.get(timeout=0.5)
                except queue.Empty:
                    if self.stop_event.is_set():
                        break
                    continue
                if item is None:
                    break
                packet, allowed = item
                if previous and not allowed:
                    self.record_ready.clear()
                    self.writer.close_segment()
                previous = allowed
                if packet.session or packet.config or allowed:
                    if packet.session:
                        self.record_ready.clear()
                    self.writer.write(packet)
                    if (
                        allowed
                        and not packet.session
                        and not packet.config
                        and self.writer.container is not None
                        and self.writer.segments[-1].frames
                    ):
                        # 首个可回放关键帧真正写入之后才允许连续任务匹配，
                        # 不能把解码已开始误当成录像已覆盖开局。
                        self.record_ready.set()
                    if allowed and packet.keyframe:
                        self._close_gaps(("foreground", "device", "orientation"), packet.pts_us)
        except Exception as exc:
            self._issue("recording", f"录像停止：{exc}", fatal=True)
        finally:
            try:
                self.writer.close()
            except Exception as exc:
                self._issue("finalize", f"录像收尾失败：{exc}", fatal=True)

    def _decode(self):
        decoder, config = None, b""
        try:
            while True:
                try:
                    packet = self.decode_queue.get(timeout=0.5)
                except queue.Empty:
                    if self.stop_event.is_set():
                        break
                    continue
                if packet is None:
                    break
                if packet.session:
                    self._codec_epoch += 1
                    self.frame = None
                    decoder = av.CodecContext.create("h264", "r")
                    decoder.thread_count = 2
                    config = b""
                    continue
                if packet.config:
                    config = packet.data
                    continue
                if decoder is None:
                    raise ValueError("视频包先于尺寸会话")
                data = av.Packet((config if packet.keyframe else b"") + packet.data)
                data.pts = data.dts = packet.pts_us
                data.time_base = TIME_BASE
                self.stage_started("视频解码")
                for video_frame in decoder.decode(data):
                    pts = int(video_frame.pts * video_frame.time_base * 1_000_000)
                    now = monotonic()
                    # 采用观测到的最小传输偏移，不因后续解码积压把旧帧刷新成新帧。
                    offset = now - pts / 1_000_000
                    self._offset = min(self._offset, offset) if self._offset is not None else offset
                    self._sequence += 1
                    frame = Frame(
                        self.serial,
                        video_frame.to_ndarray(format="rgb24"),
                        pts / 1_000_000 + self._offset,
                        datetime.now(timezone.utc).isoformat(),
                        max(0, (offset - self._offset) * 1000),
                        "video" if self.client is not None else "replay",
                        f"{self._epoch}-{self._sequence}",
                        f"{self._epoch}-codec{self._codec_epoch}",
                        pts,
                    )
                    self.frame = frame
                    self._close_gaps(("decode_backlog",), pts)
                    self.metrics["decoded_frames"] += 1
                    if self.analysis_pool and self.allowed and self.sampler.accept(frame):
                        if not self.analysis_pool.submit(frame):
                            self.metrics["analysis_skipped_busy"] += 1
                        else:
                            self.stage_started("牌桌识别与状态处理")
                        self.metrics["max_analysis_queue"] = max(
                            self.metrics["max_analysis_queue"], self.analysis_pool.size
                        )
                    self.wake_analysis.set()
                self.stage_finished("视频解码")
        except Exception as exc:
            if not self.stop_event.is_set():
                self._issue("analysis_decode", f"视频解码失败：{exc}", fatal=True)
        finally:
            if self.analysis_pool:
                self.analysis_pool.close()

    def _guard(self):
        previous = None
        next_sample = monotonic()
        while not self.stop_event.is_set():
            try:
                output = self.client._run(["shell", "dumpsys", "window"]).decode("utf-8", errors="replace")
                # HyperOS 的 windows 子命令不输出焦点；完整输出可包含无焦点的虚拟屏幕。
                # 必须检查所有显示器的焦点，不能把首个 null 当成当前游戏已离开。
                power = self.client._run(["shell", "dumpsys", "power"]).decode("utf-8", errors="replace")
                allowed = game_is_foreground(output, power)
                if allowed != previous:
                    if previous is not None and not allowed:
                        self._issue(
                            "foreground",
                            "游戏不在前台，已暂停录像并保留缺口",
                        )
                    self.allowed, previous = allowed, allowed
                    self.messages.put("已确认游戏前台" if allowed else "等待游戏前台；当前画面不写入录像")
            except Exception as exc:
                self.allowed = False
                if previous is not False:
                    self._issue("device", f"设备状态不可确认：{exc}")
                previous = False
            if monotonic() >= next_sample:
                self.metrics["queue_samples"].append(
                    {
                        "elapsed_seconds": monotonic() - self._started,
                        "write": self.write_queue.qsize(),
                        "decode": self.decode_queue.qsize(),
                        "evidence": self.evidence.queue.qsize(),
                        "analysis": self.analysis_pool.size if self.analysis_pool else 0,
                        "decoded": self.metrics["decoded_frames"],
                        "analyzed": self.metrics["analyzed_frames"],
                    }
                )
                next_sample = monotonic() + 5
            self.stop_event.wait(0.75)

    def _analyze(self):
        from doudizhu_lab.domain.bot_types import ObservationSnapshot, SnapshotSequence
        from doudizhu_lab.domain.tracker import GameTracker
        from doudizhu_lab.vision.controls import ControlRecognizer

        tracker = GameTracker(self.profile.seat_order, config=self.profile.tracker_settings)
        history = []
        sequence = SnapshotSequence()
        pending_gaps = []
        last_card_ocr = None
        try:
            controls = ControlRecognizer(self.session.folder / "templates/controls")
            while True:
                result = self.analysis_pool.take()
                if result is None:
                    if self.analysis_pool.closed and self.analysis_pool.size == 0:
                        break
                    if self.stop_event.is_set():
                        with self._health_lock:
                            stage = self._stages.get("牌桌识别与状态处理", {})
                            pending = stage.get("pending_at")
                        if pending is not None and monotonic() - pending > (
                            3 if stage.get("output_at") else 20
                        ):
                            raise TimeoutError("停止时识别任务仍未完成")
                    continue
                start = monotonic()
                frame, (observation, recognition) = result
                controls.augment_observation(frame, observation, self.profile)
                if self._last_card_enabled:
                    if last_card_ocr is None:
                        from doudizhu_lab.vision.last_card import LastCardOCR

                        last_card_ocr = LastCardOCR(self.session.folder)
                    # 普通回合也须调用：同一服务同时补充完整短提示，不能只在末手调用。
                    # 王类分支在服务内部仍严格要求末手标记和零剩牌。
                    observation, recognition = last_card_ocr.enrich(
                        frame, observation, recognition, self.profile
                    )
                while not self.gaps.empty():
                    pending_gaps.append(self.gaps.get_nowait())
                # 后帧识别可先完成，但缺口也必须按源时间应用，不能污染中断前的在途帧。
                for gap_pts, message, lost in list(pending_gaps):
                    if gap_pts > observation.pts_us:
                        continue
                    gap = tracker.mark_gap(observation, message, capture_lost=lost)
                    for event in gap.events:
                        self.session.add_event(event)
                    pending_gaps.remove((gap_pts, message, lost))
                update = tracker.observe(observation)
                for event in update.events:
                    if event.kind == "game_start":
                        history = []
                        self.begin_game(event.game_id)
                    if event.confirmed:
                        from dataclasses import asdict

                        history.append(asdict(event))
                self.session.add_observation(observation)
                for event in update.events:
                    if event.kind in ("play", "pass") and event.confirmed:
                        event.data["live_confirmation_delay_us"] = max(
                            0,
                            int(frame.age * 1_000_000) + observation.pts_us - event.data["first_evidence_us"],
                        )
                        event.data["processing_finished_utc"] = datetime.now(timezone.utc).isoformat()
                    self.session.add_event(event)
                    try:
                        self.ui_events.put_nowait(event)
                    except queue.Full:
                        pass  # 仅跳过界面通知，原始事件已事务落盘。
                for issue in update.issues:
                    self.session.add_issue(issue)
                if update.events or update.issues:
                    if not self.evidence.submit(frame, [e.event_id for e in update.events]):
                        self.session.add_issue(
                            {
                                "code": "evidence_deferred",
                                "scope": "evidence",
                                "pts_us": frame.media_pts_us,
                                "frame_id": frame.frame_id,
                                "message": "截图写盘繁忙，已登记录像源帧，收尾后补采",
                            }
                        )
                self.observation, self.recognition, self.state = observation, recognition, update.state
                self.pending_candidates = update.pending_candidates
                snapshot = ObservationSnapshot.create(
                    frame,
                    recognition,
                    observation,
                    update.state,
                    history,
                    pending=bool(update.pending_candidates),
                    allowed=self.allowed,
                )
                snapshot = sequence.publish(controls.enrich(snapshot, self.profile))
                if self.operational:
                    snapshot = self.operational.transform(snapshot)
                    operating_events, notices = self.operational.drain()
                    for event in operating_events:
                        self.session.add_operational_event(event)
                    for notice in notices:
                        self.session.add_recovery(notice)
                with self._snapshot_lock:
                    self._bot_snapshot = snapshot
                self.metrics["analyzed_frames"] += 1
                self.stage_finished("牌桌识别与状态处理", pending=self.analysis_pool.size > 0)
                for event in update.events:
                    if (
                        event.kind == "game_end"
                        and event.confirmed
                        and update.state.complete
                        and not self._flow_started
                    ):
                        self.complete_game(event.game_id, event.pts_us)
                self.metrics["analysis_ms_total"] += observation.elapsed_ms + (monotonic() - start) * 1000
                self.metrics["analysis_latency_ms_max"] = max(
                    self.metrics["analysis_latency_ms_max"], frame.age * 1000
                )
        except Exception as exc:
            self._issue("analysis", f"实时识别中断：{exc}；录像保留供重算")
        finally:
            self.analysis_pool.shutdown(wait=not self.failed)

    def poll(self) -> dict:
        """无阻塞获取快照；返回的帧不在后台修改，GUI 可安全复制到 QImage。"""
        self._gui_at = monotonic()
        messages = []
        events = []
        while not self.ui_events.empty():
            events.append(self.ui_events.get_nowait())
        while True:
            try:
                messages.append(self.messages.get_nowait())
            except queue.Empty:
                break
        return {
            "frame": self.frame,
            "recognition": self.recognition,
            "observation": self.observation,
            "state": self.state,
            "messages": messages,
            "stopped": self.stop_event.is_set() or self.failed,
            "health": self._last_health,
            "failed": self.failed,
            "primary_fault": dict(self._fault_details[0]) if self._fault_details else None,
            "metrics": dict(self.metrics),
            "allowed": self.allowed,
            "events": events,
            "codec_epoch": self._codec_epoch,
            "pending_candidates": self.pending_candidates,
            "bot_snapshot": self.bot_snapshot(),
        }

    def bot_snapshot(self):
        """返回一次原子发布的同帧快照，不从独立变化的最新图像及状态字段拼接。"""
        with self._snapshot_lock:
            return self._bot_snapshot

    def stop(self, reason: StopReason = StopReason.USER):
        """结束本次录制并等待尾录、分析与截图收尾；数据库关闭后再调用最终清理接口。

        reason只表示调用者意图；任意已登记故障都优先保留。该接口可重复调用，
        不删除仍被数据库或工作线程使用的目录，显式录像和验收始终保留完整内容。
        """
        if self.closed:
            return self.result
        self.closed = True
        self._health_stop.set()
        for thread in self.threads:
            if thread.name == "录像健康报告":
                thread.join(timeout=3)
        if self.remote_capture:
            faulthandler.cancel_dump_traceback_later()
            if self._stack_file:
                self._stack_file.close()
        self.stop_event.set()
        self.wake_analysis.set()
        try:
            if self.remote_capture:
                self.stream.close(StopReason.FAULT if self.failed else reason)
            else:
                self.stream.close()
        except Exception as exc:
            self.report_failure("capture_shutdown", str(exc))
        if self.analysis_pool and not any(t.name == "视频解码" for t in self.threads):
            self.analysis_pool.close()
        deadline = monotonic() + 12
        for thread in self.threads:
            if thread is not threading.current_thread():
                thread.join(timeout=max(0, deadline - monotonic()))
        remaining = [
            thread.name
            for thread in self.threads
            if thread.is_alive() and thread is not threading.current_thread()
        ]
        if self.analysis_pool and self.analysis_pool.unfinished:
            remaining.append("识别工作线程")
        if remaining:
            self._issue("shutdown", "后台收尾超时：" + "、".join(remaining), fatal=True)
        try:
            if not self.evidence.stop(rebuild=not remaining):
                remaining.append("证据截图写盘")
                self._issue("shutdown", "证据截图写盘线程未结束", fatal=True)
        except Exception as exc:
            self._evidence_error("", f"证据截图收尾未完成：{exc}")
            self.report_failure("evidence_shutdown", str(exc))
        elapsed = monotonic() - self._started
        metrics = dict(self.metrics) | {
            "elapsed_seconds": elapsed,
            "decoded_fps": self.metrics["decoded_frames"] / max(elapsed, 0.01),
            "analyzed_fps": self.metrics["analyzed_frames"] / max(elapsed, 0.01),
            "pending_write": self.write_queue.qsize(),
            "pending_decode": self.decode_queue.qsize(),
            "pending_analysis": self.analysis_pool.size if self.analysis_pool else 0,
            "unfinished_threads": remaining,
            "evidence": self.evidence.statistics(),
        }
        self._final_metrics = metrics
        try:
            self.session.set_meta("metrics", metrics)
            self.session.set_meta("status", "interrupted" if self.failed else "stopped")
            (self.session.folder / "运行统计.json").write_text(
                json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            # 自动会话以SQLite为原始记录，避免收尾时重放、导出整局再立即删除。
            # 故障会话可在离线界面或命令行按需导出，显式录像继续兼容旧导出。
            if not self.automatic:
                self.session.export()
        except ImportError:
            pass
        except Exception as exc:
            self.report_failure("session_finalize", f"会话资料收尾失败：{exc}")
        if self.remote_capture:
            path = self.session.folder / "留存记录.json"
            if path.exists() and self.failed:
                value = json.loads(path.read_text(encoding="utf-8"))
                value.update(reason=StopReason.FAULT)
                value["faults"] = value.get("faults", []) + [
                    detail
                    for detail in self._fault_details
                    if not any(f["code"] == detail["code"] for f in value.get("faults", []))
                ]
                value["primary_fault"] = min(
                    value["faults"], key=lambda f: f.get("occurred_at", float("inf"))
                )
                atomic_json(path, value)
        else:
            self.retention.last_pts = self._pts
            self.retention.finish(StopReason.FAULT if self.failed else reason)

    def finalize_recording(self) -> RecordingResult:
        """调用方关闭会话数据库后执行处置；只有全部正常收尾的自动会话才可删除。"""
        if not self.closed:
            raise RuntimeError("录像尚未停止，不能执行清理")
        self.result = finalize_retention(
            self.session.folder,
            self.store.root,
            runtime={
                "application_version": __version__,
                "device": self.serial,
                "metrics": getattr(self, "_final_metrics", {}),
            },
        )
        return self.result
