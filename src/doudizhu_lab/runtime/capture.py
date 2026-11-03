"""独立录像进程及其只读媒体代理；识别、Qt 或主进程阻塞均不阻塞录像写盘。"""

from __future__ import annotations

import json
import os
import queue
import struct
import subprocess
import sys
import threading
from copy import deepcopy
from pathlib import Path
from time import monotonic, sleep

from doudizhu_lab.devices.adb import AdbClient
from doudizhu_lab.devices.video import (
    CONFIG_FLAG,
    KEY_FLAG,
    SESSION_FLAG,
    ScrcpyStream,
    SegmentWriter,
    read_packet,
)
from doudizhu_lab.runtime.channel import RuntimeChannel
from doudizhu_lab.storage.retention import RetentionLedger, StopReason, atomic_json


def foreground(window: str, power: str) -> bool:
    """只读确认游戏焦点与亮屏状态；不发送触摸、按键或唤醒命令。"""
    return (
        any("com.tencent.tmgp.sgame/" in line for line in window.splitlines() if "mCurrentFocus=" in line)
        and "mWakefulness=Awake" in power
    )


def packet_bytes(packet) -> bytes:
    """沿用已校验的视频包协议传递分析副本，独立于 MKV 保存和控制通道。"""
    if packet.session:
        return struct.pack(">III", SESSION_FLAG >> 32, packet.width, packet.height)
    flags = packet.pts_us | (CONFIG_FLAG if packet.config else 0) | (KEY_FLAG if packet.keyframe else 0)
    return struct.pack(">QI", flags, len(packet.data)) + packet.data


def health_timeout(health: dict, now: float) -> str | None:
    """判断主界面或有在途工作的处理阶段是否超时；静态画面、等待对手不触发超时。

    首次输出允许二十秒预热；已有输出的阶段在途三秒无完成即报告。时间来自同机
    单调时钟，不能用不同进程的墙上时间或牌局输赢代替。
    """
    if health.get("gui_at") is not None and now - health["gui_at"] > 3:
        return "界面心跳超过三秒未更新"
    for name, stage in health.get("stages", {}).items():
        pending = stage.get("pending_at")
        limit = 3 if stage.get("output_at") else 20
        if pending is not None and now - pending > limit:
            return f"{name}阶段有在途工作但超过{limit}秒未完成"
    return None


class CaptureProcess:
    """主进程持有的媒体代理；控制和健康记录走独立文件，视频副本走有界管道。

    start 在后台等待握手，packets 仅供接收线程读取。command 只入队，heartbeat
    由唯一报告线程发布文件；close 只终止本实例创建的进程，不触碰其他服务。
    """

    def __init__(self, folder: Path, adb: str, serial: str, server: Path, *, automatic=False):
        """准备当前会话的代理配置；不启动设备服务，automatic只决定正常资料清理策略。"""
        self.folder = folder
        self.process = None
        self.stderr = None
        self.automatic = automatic
        self.commands = queue.SimpleQueue()
        self.health = {}
        self._sequence = 0
        self.channel = RuntimeChannel()
        self._last_status = {}
        self._health_written_at = None
        self._health_started_at = monotonic()
        self._pending_command = None
        self._command_blocked_at = None
        self.config = {
            "folder": str(folder.resolve()),
            "adb": adb,
            "serial": serial,
            "server": str(server.resolve()),
            "automatic": automatic,
        }

    def start(self) -> None:
        """启动无控制台子进程并等待采集握手；失败保留配置和进程日志供诊断。"""
        atomic_json(self.folder / "采集配置.json", self.config)
        (self.folder / "采集命令").mkdir(exist_ok=True)
        self.stderr = (self.folder / "采集进程日志.txt").open("wb")
        self.process = subprocess.Popen(
            [sys.executable, "-u", "-m", __name__, str(self.folder / "采集配置.json")],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=self.stderr,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            bufsize=0,
        )
        deadline = monotonic() + 20
        while monotonic() < deadline:
            status = self.status()
            if status.get("started"):
                return
            if self.process.poll() is not None:
                raise RuntimeError("独立录像进程启动失败，请查看采集进程日志")
            sleep(0.05)
        raise TimeoutError("独立录像进程二十秒内未完成握手")

    def packets(self):
        """读取分析用编码副本；主程序消费缓慢不会反向阻塞子进程的完整录像。"""
        while True:
            yield read_packet(self.process.stdout)

    def command(self, kind: str, **value) -> None:
        """无阻塞登记低频处置命令；命令携带当前会话路径，不向手机发送任何输入。"""
        self.commands.put({"kind": kind, **value})

    def heartbeat(self, value: dict) -> None:
        """发布健康信息及有序命令；短暂访问冲突只记告警，旧报告超过有效期才抛错。

        首份报告允许二十秒，之后三秒无成功发布即故障；缓存中的源时间不会刷新。
        健康文件暂不可写时仍尝试发送控制命令，避免一条通道拖住另一条通道。
        """
        self.health = value
        if self.channel.write(
            self.folder / "主程序健康.json", value | {"channel_warnings": self.channel.snapshot()}
        ):
            self._health_written_at = monotonic()
        self.flush_commands()
        limit = 3 if self._health_written_at is not None else 20
        if monotonic() - (self._health_written_at or self._health_started_at) > limit:
            raise TimeoutError(f"主程序健康报告连续{limit}秒无法发布")

    def flush_commands(self) -> bool:
        """按提交顺序发布命令；未发布的队首保持原编号，连续三秒阻塞明确抛错。

        返回 True 表示当前队列已全部发布。只能由唯一报告线程或已结束报告线程的
        收尾方调用；False 时不消费后续命令，下一次从同一个队首继续。
        """
        while True:
            if self._pending_command is None:
                try:
                    self._pending_command = self.commands.get_nowait()
                except queue.Empty:
                    return True
            path = self.folder / "采集命令" / f"{self._sequence + 1:08d}.json"
            if not self.channel.write(path, self._pending_command):
                if self._command_blocked_at is None:
                    self._command_blocked_at = monotonic()
                if monotonic() - self._command_blocked_at >= 3:
                    raise TimeoutError("录像控制命令连续三秒无法发布")
                return False
            self._sequence += 1
            self._pending_command = self._command_blocked_at = None

    def status(self) -> dict:
        """返回最近有效状态；读取冲突沿用原快照，绝不刷新源 updated_at 时间。

        尚无有效消息时返回空字典，由启动预热处理；真实磁盘或路径错误仍上抛。
        返回副本避免调用方修改缓存，只能由握手线程或报告线程顺序调用。
        """
        value = self.channel.read(self.folder / "采集状态.json", timestamp="updated_at")
        if value is not None:
            self._last_status = value
        return deepcopy(self._last_status)

    def close(self, reason: StopReason = StopReason.USER) -> None:
        """发送有类型的停止命令并等待尾录及收尾；三十秒后只终止本实例的失控子进程。

        必须先结束健康报告线程。强制结束或异常退出明确抛错，由上层按故障留存，
        不会将未封口录像误标为正常清理。
        """
        if self.process is None:
            if self.stderr:
                self.stderr.close()
            return
        try:
            if self.process.poll() is None:
                self.command("stop", reason=str(reason))
                control_error = None
                try:
                    while not self.flush_commands() and self.process.poll() is None:
                        sleep(0.05)
                except Exception as exc:
                    # 不能提前关闭分析管道或遗留正在写入的子进程；即使停止命令未发出，
                    # 子进程也会按旧健康报告失效独立尾录，仍等待其封口再报告错误。
                    control_error = exc
                try:
                    self.process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=5)
                    raise TimeoutError("独立录像进程收尾超时，已保留现有片段") from None
                if control_error is not None:
                    raise control_error
            if self.process.returncode:
                raise RuntimeError(f"独立录像进程异常退出：{self.process.returncode}")
        finally:
            if self.process.stdout:
                self.process.stdout.close()
            if self.stderr:
                self.stderr.close()


class CaptureWorker:
    """子进程内部的录像、只读前台检查与监控；仅写本次会话，完全不打开主数据库。"""

    def __init__(self, config: dict, *, stream=None, output=None, tail_seconds=10):
        """创建独占的封装器与处置账本；注入媒体和输出仅用于隔离测试，不改变设备权限。

        默认尾录十秒。构造不准备手机服务；所有文件均写入config指定的本次会话，
        run负责启动、监控和关闭，不把主进程SQLite连接或Qt对象传入子进程。
        """
        self.folder = Path(config["folder"])
        self.ledger = RetentionLedger(self.folder, automatic=config.get("automatic", False))
        self.client = AdbClient(config.get("adb", "adb"), config.get("serial", ""), timeout=3)
        self.stream = stream if stream is not None else ScrcpyStream(self.client, Path(config["server"]))
        self.output = output
        self.writer = SegmentWriter(
            self.folder / "video",
            segment_seconds=30 if self.ledger.automatic else 120,
            quota_root=self.folder.parent,
        )
        self.lock = threading.RLock()
        self.stopped = threading.Event()
        self.forward = queue.Queue(maxsize=32)
        self.tail_seconds = tail_seconds
        self.fault_deadline = None
        self.stop_reason = StopReason.USER
        self.allowed = False
        self.started = False
        self.closed = False
        self.record_ready = False
        self.last_packet_at = monotonic()
        self.started_at = self.last_packet_at
        self.health_seen = False
        self.packets_count = self.dropped = 0
        self._previous_allowed = False
        self.channel = RuntimeChannel()
        self.phase = "starting"
        self._last_health = {}
        self._status_written_at = None
        self._command_blocked_at = None
        self._applied_command = None

    def fault(self, code: str, message: str, *, health=None, occurred_at: float | None = None) -> None:
        """先持久化故障再开始最多十秒尾录；同类故障不会不断延长录制截止时间。"""
        with self.lock:
            self.ledger.fault(code, message, health=health, occurred_at=occurred_at)
            if self.fault_deadline is None:
                self.fault_deadline = monotonic() + self.tail_seconds
                if not self.closed:
                    self.phase = "tail_recording"

    def _guard(self):
        """独立检查前台，主界面卡住时仍能停止记录其他应用；只读命令各有三秒超时。"""
        previous = None
        while not self.stopped.is_set():
            try:
                window = self.client._run(["shell", "dumpsys", "window"]).decode("utf-8", errors="replace")
                power = self.client._run(["shell", "dumpsys", "power"]).decode("utf-8", errors="replace")
                self.allowed = foreground(window, power)
                if previous is True and not self.allowed:
                    self.fault("foreground", "游戏前台状态失效")
                previous = self.allowed
            except Exception as exc:
                self.allowed = False
                self.fault("device", f"设备状态检查失败：{exc}")
            self.stopped.wait(0.75)

    def _send(self):
        """分析副本使用独立发送线程；管道满时仅该线程等待，不持有录像写入锁。"""
        try:
            while not self.stopped.is_set():
                try:
                    packet = self.forward.get(timeout=0.2)
                except queue.Empty:
                    continue
                data = memoryview(packet_bytes(packet))
                if self.output is not None:
                    self.output(data)
                else:
                    while data:
                        data = data[os.write(sys.stdout.fileno(), data) :]
        except (OSError, ValueError):
            if not self.stopped.is_set():
                self.fault("analysis_pipe", "主程序分析通道已关闭")

    def _capture(self):
        session = config = None
        resync = False
        try:
            for packet in self.stream.packets():
                if self.stopped.is_set():
                    break
                if self.fault_deadline is not None and monotonic() >= self.fault_deadline:
                    self.stopped.set()
                    break
                with self.lock:
                    self.last_packet_at = monotonic()
                    self.packets_count += 1
                    if packet.session:
                        session, config = packet, None
                        self.record_ready = False
                    elif packet.config:
                        config = packet
                    else:
                        self.ledger.last_pts = packet.pts_us
                    if self._previous_allowed and not self.allowed:
                        self.writer.close_segment()
                        self.record_ready = False
                    self._previous_allowed = self.allowed
                    if packet.session or packet.config or self.allowed:
                        self.writer.write(packet)
                        self.record_ready = bool(self.allowed and self.writer.container is not None)
                        self.ledger.prune(self.writer)
                if resync:
                    if not (packet.keyframe and session and config and self.forward.qsize() <= 29):
                        continue
                    for item in (session, config, packet):
                        self.forward.put_nowait(item)
                    resync = False
                else:
                    try:
                        self.forward.put_nowait(packet)
                    except queue.Full:
                        self.dropped += 1
                        resync = True
        except Exception as exc:
            if not self.stopped.is_set():
                self.fault(getattr(exc, "code", "capture"), f"录像采集或写盘失败：{exc}")
                self.stopped.set()

    def _commands(self):
        for path in sorted((self.folder / "采集命令").glob("*.json")):
            if self._applied_command == path:
                if not self.channel.remove(path):
                    self._command_blocked()
                    break
                self._applied_command = self._command_blocked_at = None
                continue
            value = self.channel.read(path)
            if value is None:
                self._command_blocked()
                break
            with self.lock:
                kind = value["kind"]
                if kind == "automatic":
                    self.ledger.automatic = True
                    self.writer.segment_us = 30_000_000
                    self.ledger.persist()
                elif kind == "game_start":
                    self.ledger.begin_game(value["game_id"])
                elif kind == "game_complete":
                    self.ledger.complete_game(value["game_id"], value["pts_us"])
                    self.ledger.prune(self.writer)
                elif kind == "game_fault":
                    self.ledger.pin_game(value["game_id"], value["code"], value["message"])
                elif kind == "game_skipped":
                    self.ledger.skip_game(value["game_id"], value["pts_us"])
                    self.ledger.prune(self.writer)
                elif kind == "rotate_media":
                    names = set(value["names"])
                    candidates = [
                        s
                        for s in self.writer.segments
                        if s.file in names and s.closed and s.global_end_us < self.ledger.active["start_us"]
                    ]
                    if any(Path(s.file).name != s.file for s in candidates):
                        raise ValueError("轮转媒体路径越界")
                    for segment in candidates:
                        if not any(r["file"] == segment.file for r in self.ledger.removed):
                            self.ledger.removed.append(
                                {
                                    "file": segment.file,
                                    "start_us": segment.global_start_us,
                                    "end_us": segment.global_end_us,
                                    "reason": "task_media_rotation",
                                    "status": "planned",
                                }
                            )
                    self.ledger.persist()
                    pending = [
                        r
                        for r in self.ledger.removed
                        if r["file"] in names
                        and r["reason"] == "task_media_rotation"
                        and r["end_us"] < self.ledger.active["start_us"]
                    ]
                    accepted = {r["file"] for r in pending}
                    self.writer.segments[:] = [s for s in self.writer.segments if s.file not in accepted]
                    self.writer._index()
                    for record in pending:
                        try:
                            (self.writer.folder / record["file"]).unlink(missing_ok=True)
                            record["status"] = "removed"
                        except OSError as exc:
                            record["cleanup_error"] = str(exc)
                    self.ledger.persist()
                elif kind == "user_abort":
                    self.ledger.abort_window()
                    self.ledger.prune(self.writer)
                elif kind == "fault":
                    self.fault(
                        value["code"],
                        value["message"],
                        health=value.get("health"),
                        occurred_at=value.get("occurred_at"),
                    )
                elif kind == "stop":
                    self.stop_reason = StopReason(value["reason"])
                    if self.stop_reason == StopReason.FAULT and not self.ledger.faults:
                        self.fault("application", "主程序请求按异常保存")
                    if self.fault_deadline is None:
                        self.stopped.set()
                else:
                    raise ValueError(f"未知录像处置命令：{kind}")
            self._applied_command = path
            if not self.channel.remove(path):
                self._command_blocked()
                break
            self._applied_command = self._command_blocked_at = None

    def _command_blocked(self) -> None:
        """队首读取或确认删除失败时保序等待；三秒后固定控制通道故障，仍独立尾录。"""
        if self._command_blocked_at is None:
            self._command_blocked_at = monotonic()
        if monotonic() - self._command_blocked_at >= 3:
            self.fault("control_channel", "录像控制命令连续三秒无法读取或确认")

    def _monitor_health(self, now: float) -> None:
        """运行阶段检查源健康时间；进入故障尾录后不再将主动停止上报误记为失联。"""
        if self.phase != "running":
            return
        health = self.channel.read(self.folder / "主程序健康.json", timestamp="reported_at")
        if health is not None:
            self._last_health = health
            self.health_seen = True
        health = self._last_health
        if self.health_seen:
            reason = health_timeout(health, now)
            if now - health["reported_at"] > 3:
                reason = "主程序健康报告超过三秒未更新"
            if reason:
                self.fault("health_timeout", reason, health=health)
        elif now - self.started_at > 20:
            self.fault("health_startup", "二十秒内未收到主程序健康报告")

    def _status(self):
        """锁内复制媒体指标，锁外发布；状态文件占用不会阻塞完整录像写盘。"""
        with self.lock:
            value = deepcopy(
                {
                    "updated_at": monotonic(),
                    "started": self.started,
                    "closed": self.closed,
                    "allowed": self.allowed,
                    "record_ready": self.record_ready,
                    "packets": self.packets_count,
                    "analysis_dropped": self.dropped,
                    "last_packet_at": self.last_packet_at,
                    "last_pts_us": self.ledger.last_pts,
                    "faults": self.ledger.faults,
                    "primary_fault": self.ledger.value()["primary_fault"],
                    "lifecycle": self.phase,
                    "channel_warnings": self.channel.snapshot(),
                }
            )
        if self.channel.write(self.folder / "采集状态.json", value):
            self._status_written_at = monotonic()
        else:
            limit = 3 if self._status_written_at is not None else 20
            if monotonic() - (self._status_written_at or self.started_at) > limit:
                self.fault("status_channel", f"采集状态连续{limit}秒无法发布")

    def run(self, *, guard=True) -> None:
        """运行直到正常停止或尾录截止；即使父进程消失，也独立封口并留下异常标记。"""
        threads = []
        try:
            self.stream.start()
            self.started = True
            self.phase = "running"
            for name, target in [
                ("录像采集", self._capture),
                ("分析副本发送", self._send),
                *([("只读前台检查", self._guard)] if guard else []),
            ]:
                thread = threading.Thread(target=target, name=name, daemon=True)
                thread.start()
                threads.append(thread)
            while not self.stopped.is_set():
                self._commands()
                if self.stopped.is_set():
                    break
                now = monotonic()
                self._monitor_health(now)
                if self.fault_deadline is not None and now >= self.fault_deadline:
                    self.stopped.set()
                self._status()
                self.stopped.wait(0.2)
        except Exception as exc:
            self.fault("worker", f"独立录像进程异常：{exc}")
            raise
        finally:
            self.stopped.set()
            self.stream.close()
            for thread in threads:
                if thread.name != "分析副本发送":
                    thread.join(timeout=7)
            with self.lock:
                try:
                    self.writer.close()
                    self.ledger.finish(self.stop_reason)
                    # 当前尾部在父进程的截图、分析及数据库也正常关闭后再统一清理。
                    if self.ledger.faults:
                        self.ledger.prune(self.writer)
                except Exception as exc:
                    self.ledger.fault("finalize", f"录像收尾失败：{exc}")
                    raise
                finally:
                    self.closed = True
                    self.phase = "closed"
            self._status()


if __name__ == "__main__":
    CaptureWorker(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))).run()
