"""用旧录像的编码包按原速复测生产流水线；本模块不连接或操作手机。"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from time import monotonic

import av

from doudizhu_lab.devices.video import VideoPacket
from doudizhu_lab.runtime.recording import RecordingPipeline
from doudizhu_lab.storage.config import Store
from doudizhu_lab.storage.replay import VideoReplay
from doudizhu_lab.storage.truth import media_manifest


def annexb_config(extra: bytes) -> bytes:
    """把 MKV 的 AVC 配置转换为生产解码器需要的 Annex B SPS/PPS。

    只接受完整的 AVCDecoderConfigurationRecord 或已有起始码的配置。逐个检查
    长度，截断时明确失败；不能把损坏配置作为有效视频包继续分析。
    """
    if extra.startswith((b"\x00\x00\x00\x01", b"\x00\x00\x01")):
        return extra
    if len(extra) < 7 or extra[0] != 1:
        raise ValueError("录像缺少有效的 H.264 编码配置")
    result, offset = [], 6
    counts = [extra[5] & 31]
    for group in range(2):
        if group:
            if offset >= len(extra):
                raise ValueError("录像 PPS 配置被截断")
            counts.append(extra[offset])
            offset += 1
        for _ in range(counts[group]):
            if offset + 2 > len(extra):
                raise ValueError("录像编码配置长度被截断")
            size = int.from_bytes(extra[offset : offset + 2], "big")
            offset += 2
            if not size or offset + size > len(extra):
                raise ValueError("录像编码配置内容被截断")
            result.append(b"\x00\x00\x00\x01" + extra[offset : offset + size])
            offset += size
    if not all(counts):
        raise ValueError("录像缺少 SPS 或 PPS")
    return b"".join(result)


class EncodedReplayStream:
    """以媒体时钟发送原录像 H.264 包，替代 USB 源而复用其后的生产代码。

    不解码、不预先识别、不丢包，也不调整播放速度来追赶识别。文件解复用和
    AVC 到 Annex B 的封装转换不是 scrcpy 线协议测试，报告必须注明这一边界。
    close 只唤醒本对象的等待，不调用 ADB 或停止其他进程。
    """

    def __init__(self, folder: Path):
        self.video = VideoReplay(folder)
        self.stop_event = threading.Event()
        self.started = None
        self.sent = 0
        self.lateness_ms = []

    def start(self):
        """检查所有片段正常收尾后建立一次原速播放时钟；不创建手机服务。"""
        if not self.video.segments or any(not s.get("closed") for s in self.video.segments):
            raise ValueError("原速诊断需要已停止且全部片段已收尾的录像")
        self.started = monotonic()

    def packets(self):
        """按源时间等待后返回协议数据类，供原有接收、写盘和解码线程消费。"""
        previous_config = None
        for segment in self.video.segments:
            with av.open(str(self.video.video / segment["file"])) as container:
                stream = container.streams.video[0]
                config = annexb_config(stream.codec_context.extradata)
                identity = (stream.width, stream.height, config)
                if identity != previous_config:
                    yield VideoPacket(session=True, width=stream.width, height=stream.height)
                    yield VideoPacket(config=True, data=config)
                    previous_config = identity
                conversion = av.bitstream.BitStreamFilterContext("h264_mp4toannexb", stream)
                for packet in container.demux(stream):
                    if not packet.size:
                        continue
                    for item in conversion.filter(packet):
                        if item.pts is None or item.dts != item.pts:
                            raise ValueError("原速诊断只支持当前无 B 帧的 scrcpy H.264 录像")
                        pts = segment["global_start_us"] + int(item.pts * item.time_base * 1_000_000)
                        due = self.started + (pts - self.video.start_us) / 1_000_000
                        if self.stop_event.wait(max(0, due - monotonic())):
                            return
                        self.lateness_ms.append(max(0, (monotonic() - due) * 1000))
                        self.sent += 1
                        yield VideoPacket(pts, bytes(item), keyframe=item.is_keyframe)

    def close(self):
        """中断本地播放等待；容器由迭代器所属接收线程退出时关闭。"""
        self.stop_event.set()


class PacedReplayPipeline(RecordingPipeline):
    """保留生产队列和消费者，只用录像替换 USB 源及手机前台查询。

    快照、双识别线程、六任务上限、10 FPS 采样、证据 PNG 和状态推演全部沿用
    RecordingPipeline。此会话永远标为 paced_replay，不能成为正式实时验收。
    """

    def __init__(self, store: Store, source: Path):
        super().__init__(store, "replay", stream=EncodedReplayStream(source))
        self.allowed = True
        self.session.set_meta("analysis_mode", "paced_replay")
        self.session.set_meta("video_source", os.path.relpath(self.stream.video.folder, self.session.folder))
        self.session.set_meta(
            "diagnostic_scope", "旧录像原速编码包回放；未检查 USB、scrcpy 线协议、手机前台与原生界面"
        )
        self.session.set_meta("source_video_manifest", media_manifest(self.stream.video.folder))
        self.session.set_meta("formal_acceptance", False)

    def _guard(self):
        """不查询设备；每五秒采样实际队列，并保证停止信号能够立即结束等待。"""
        while not self.stop_event.is_set():
            self.metrics["queue_samples"].append(
                {
                    "elapsed_seconds": monotonic() - self._started,
                    "write": self.write_queue.qsize(),
                    "decode": self.decode_queue.qsize(),
                    "evidence": self.evidence.queue.qsize(),
                    "analysis": self.analysis_pool.size,
                    "decoded": self.metrics["decoded_frames"],
                    "analyzed": self.metrics["analyzed_frames"],
                }
            )
            self.stop_event.wait(5)


def paced_replay(source: Path, store: Store, *, progress=None) -> Path:
    """完整原速重放一条旧会话，并在独立诊断目录保留观察、事件和写盘副本。

    source 为已停止录像目录；store 决定本次配置和隔离保存目录。progress 接收
    独立进度字典，约每五秒调用一次。返回诊断会话目录；任何中断都保留已落盘
    结果和缺口，不更改原会话。此函数会阻塞至原录像时长结束，供后台命令运行。
    """
    pipe = PacedReplayPipeline(store, source)
    interrupted = False
    try:
        pipe.start()
        while not pipe.stop_event.wait(5):
            sample = {
                "folder": str(pipe.session.folder.resolve()),
                "elapsed_seconds": monotonic() - pipe._started,
                "source_seconds": (pipe._pts - pipe.stream.video.start_us) / 1_000_000,
                "decoded": pipe.metrics["decoded_frames"],
                "analyzed": pipe.metrics["analyzed_frames"],
                "analysis_queue": pipe.analysis_pool.size,
                "skipped": pipe.metrics["analysis_skipped_busy"],
                "max_latency_ms": pipe.metrics["analysis_latency_ms_max"],
            }
            (pipe.session.folder / "原速进度.json").write_text(
                json.dumps(sample, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            if progress:
                progress(sample)
            pipe.poll()  # 沿用界面的轮询消费，界面通知不作为动作成功依据。
    except BaseException:
        interrupted = True
        pipe._issue("shutdown", "本地原速回放被中断", fatal=True)
        raise
    finally:
        pipe.stop()
        lateness = sorted(pipe.stream.lateness_ms)
        report = pipe.session.metadata()["metrics"] | {
            "mode": "旧录像原速生产队列诊断，非真机实时验收",
            "formal_acceptance": False,
            "source": str(source.resolve()),
            "source_seconds": (pipe.stream.video.end_us - pipe.stream.video.start_us) / 1_000_000,
            "sent_media_packets": pipe.stream.sent,
            "source_lateness_ms_p95": lateness[int((len(lateness) - 1) * 0.95)] if lateness else None,
            "source_lateness_ms_max": max(lateness, default=None),
            "interrupted": interrupted or pipe.failed,
            "pending_candidates": pipe.pending_candidates,
        }
        pipe.session.set_meta("paced_report", report)
        (pipe.session.folder / "原速诊断.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        pipe.session.close()
    return pipe.session.folder
