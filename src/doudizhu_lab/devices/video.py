"""固定 scrcpy 5.0.1 视频协议、只读接收与无音频 MKV 封装。

本模块不建立控制通道；ADB 的人工点击仍由独立设备接口负责。
协议依据同版本 Streamer.java：会话、配置、关键帧分别占最高三个标志位。
"""

from __future__ import annotations

import hashlib
import io
import json
import shlex
import shutil
import socket
import struct
import subprocess
import threading
import urllib.request
from collections import deque
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path
from time import monotonic, sleep
from uuid import uuid4

import av

from doudizhu_lab import __version__
from doudizhu_lab.devices.adb import AdbClient, AdbError

SERVER_VERSION = "5.0.1"
SERVER_SHA256 = "764eb6f79811d5211fe9df341120882ba9994c7a61b897d7bf3fb662e53bc536"
SERVER_URL = f"https://github.com/Genymobile/scrcpy/releases/download/v{SERVER_VERSION}/scrcpy-server-v{SERVER_VERSION}"
TIME_BASE = Fraction(1, 1_000_000)
SESSION_FLAG, CONFIG_FLAG, KEY_FLAG = 1 << 63, 1 << 62, 1 << 61


def setup_video(root: Path, *, download: bool = True) -> Path:
    """获取固定版本服务器并核对官方发行物 SHA-256，校验失败不使用该文件。

    root 为本机数据目录；download=False 只检查，不联网或创建文件。
    下载先写临时文件，完整校验后替换，避免断网留下可执行的半文件。
    """
    path = root / "tools" / f"scrcpy-server-v{SERVER_VERSION}"
    if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == SERVER_SHA256:
        return path
    if not download:
        raise AdbError("整局记录不可用：视频组件缺失或校验失败，请运行 ddz-check setup-video")
    request = urllib.request.Request(SERVER_URL, headers={"User-Agent": f"doudizhu-lab/{__version__}"})
    with urllib.request.urlopen(request, timeout=45) as response:
        data = response.read(4 * 1024 * 1024)
    if hashlib.sha256(data).hexdigest() != SERVER_SHA256:
        raise AdbError("视频组件 SHA-256 不符，已拒绝使用")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_bytes(data)
    temporary.replace(path)
    return path


@dataclass(frozen=True)
class VideoPacket:
    """一个编码包或尺寸会话通知；pts_us 是设备媒体时钟，不能当成墙上时间。

    session=True 时 data 为空且 width/height 有效；config=True 时 data 为 SPS/PPS。
    每个媒体包保持原始顺序，不能为了刷新预览而丢弃录像编码包。
    """

    pts_us: int = 0
    data: bytes = b""
    config: bool = False
    keyframe: bool = False
    session: bool = False
    width: int = 0
    height: int = 0


def read_exact(reader, length: int) -> bytes:
    """从 socket 或测试用文件读取精确字节数；EOF 与截断明确失败，不拼成伪帧。"""
    result = bytearray()
    while len(result) < length:
        part = (
            reader.recv(length - len(result))
            if hasattr(reader, "recv")
            else reader.read(length - len(result))
        )
        if not part:
            raise EOFError("视频流断开或数据被截断")
        result.extend(part)
    return bytes(result)


def read_packet(reader) -> VideoPacket:
    """解析 5.0.1 的 12 字节头；限制尺寸与包长，避免损坏数据造成无界分配。"""
    header = read_exact(reader, 12)
    flags, length = struct.unpack(">QI", header)
    if flags & SESSION_FLAG:
        width, height = struct.unpack(">II", header[4:])
        if not (0 < width <= 10000 and 0 < height <= 10000 and width * height <= 40_000_000):
            raise ValueError("视频会话尺寸无效")
        return VideoPacket(session=True, width=width, height=height)
    if not 0 < length <= 16 * 1024 * 1024:
        raise ValueError("视频编码包长度无效")
    return VideoPacket(
        flags & (KEY_FLAG - 1), read_exact(reader, length), bool(flags & CONFIG_FLAG), bool(flags & KEY_FLAG)
    )


class ScrcpyStream:
    """仅接收视频的 ADB 会话；拥有自己的随机转发和临时服务器文件。

    start/packets 在接收线程调用；close 可由退出线程调用以唤醒阻塞读取。
    不调用 adb kill-server，不停止其他应用创建的转发或设备进程。
    """

    def __init__(self, client: AdbClient, server: Path, *, max_fps: int = 15, bit_rate: int = 8_000_000):
        self.client, self.server = client, server
        self.max_fps, self.bit_rate = max_fps, bit_rate
        self.scid = int(uuid4().hex[:7], 16)
        self.remote = f"/data/local/tmp/ddz-video-{self.scid:08x}.jar"
        self.port = None
        self.socket = None
        self.process = None
        self.pid = None
        self.logs = deque(maxlen=80)
        self.stopped = threading.Event()
        self.started = monotonic()

    def start(self) -> None:
        """推送校验过的组件、启动视频服务并握手；所有设备命令绑定明确序列号。"""
        if hashlib.sha256(self.server.read_bytes()).hexdigest() != SERVER_SHA256:
            raise AdbError("视频组件校验失败")
        try:
            self.client._run(["push", str(self.server), self.remote])
            port = self.client._run(["forward", "tcp:0", f"localabstract:scrcpy_{self.scid:08x}"])
            self.port = int(port.strip())
            arguments = [
                "app_process",
                "/",
                "com.genymobile.scrcpy.Server",
                SERVER_VERSION,
                f"scid={self.scid:08x}",
                "log_level=info",
                "audio=false",
                "control=false",
                "clipboard_autosync=false",
                "power_on=false",
                "cleanup=false",
                "tunnel_forward=true",
                "send_device_meta=false",
                "video_codec=h264",
                "max_size=0",
                f"max_fps={self.max_fps}",
                f"video_bit_rate={self.bit_rate}",
                "video_codec_options=i-frame-interval:int=2",
            ]
            # 手机 shell 脚本仅包含程序生成的路径和固定选项；不拼接用户文本。
            script = f"echo DDZ_PID=$$; export CLASSPATH={shlex.quote(self.remote)}; exec " + " ".join(
                map(shlex.quote, arguments)
            )
            self.process = subprocess.Popen(
                [self.client.executable, "-s", self.client.serial, "shell", script],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            threading.Thread(target=self._read_logs, daemon=True, name="视频服务日志").start()
            deadline = monotonic() + 10
            while not self.stopped.is_set() and monotonic() < deadline:
                candidate = None
                try:
                    candidate = socket.create_connection(("127.0.0.1", self.port), timeout=1)
                    if read_exact(candidate, 1) != b"\0":
                        raise ValueError("视频握手字节无效")
                    candidate.settimeout(8)
                    if read_exact(candidate, 4) != b"h264":
                        raise ValueError("视频服务器未返回 H.264")
                    self.socket = candidate
                    self.started = monotonic()
                    return
                except (OSError, EOFError):
                    if candidate:
                        candidate.close()
                    if self.process.poll() is not None:
                        break
                    sleep(0.1)
            raise AdbError("整局记录不可用：视频服务启动失败\n" + "\n".join(self.logs))
        except Exception:
            self.close()
            raise

    def _read_logs(self) -> None:
        for raw in self.process.stdout:
            line = raw.decode("utf-8", errors="replace").strip()
            if line.startswith("DDZ_PID="):
                try:
                    self.pid = int(line.split("=", 1)[1])
                except ValueError:
                    pass
            else:
                self.logs.append(line)

    def packets(self):
        """顺序迭代编码数据；正常停止以外的超时或断流交给调用者登记缺口。"""
        while not self.stopped.is_set():
            yield read_packet(self.socket)

    def close(self) -> None:
        """关闭本次 socket、ADB 子进程和对应转发；操作幂等，不修改手机权限。"""
        self.stopped.set()
        if self.socket:
            try:
                self.socket.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            self.socket.close()
            self.socket = None
        if self.process:
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                if self.pid:
                    try:
                        self.client._run(["shell", "kill", "-TERM", str(self.pid)])
                    except AdbError:
                        pass
                self.process.terminate()
                self.process.wait(timeout=3)
            self.process.stdout.close()
            self.process = None
        if self.port:
            try:
                self.client._run(["forward", "--remove", f"tcp:{self.port}"])
            except AdbError:
                pass
            self.port = None
        try:
            self.client._run(["shell", "rm", "-f", self.remote])
        except AdbError:
            pass


@dataclass
class Segment:
    """录像片段索引；global_start_us 对应首帧原始 PTS，文件内从零开始。"""

    file: str
    global_start_us: int
    global_end_us: int
    width: int
    height: int
    frames: int = 0
    closed: bool = False


class SegmentWriter:
    """把原始编码包封装到 MKV，不重编码；只在关键帧切段。

    一个写盘线程独占实例。quota_root 统计所有会话录像，达到限额立即失败；不删资料。
    index.json 在开段、周期检查及关闭时原子更新，异常退出可找到已写片段。
    """

    def __init__(
        self,
        folder: Path,
        *,
        segment_seconds: float = 120,
        quota_bytes: int = 20 * 1024**3,
        min_free_bytes: int = 2 * 1024**3,
        quota_root: Path | None = None,
    ):
        self.folder = folder
        folder.mkdir(parents=True, exist_ok=True)
        self.segment_us = int(segment_seconds * 1_000_000)
        self.quota_bytes, self.min_free_bytes = quota_bytes, min_free_bytes
        self.quota_root = quota_root or folder.parent
        self.segments: list[Segment] = []
        self.container = self.stream = self.template_container = None
        self.config = b""
        self.size = (0, 0)
        self.last_check = 0.0
        # 已清理片段会从索引移除，文件编号必须独立递增，避免覆盖仍被故障局引用的片段。
        self._segment_sequence = 0

    def _index(self):
        """原子发布索引；对 Windows 短暂共享占用最多等待半秒，持续失败仍报告录像中断。"""
        temporary = self.folder / f"index-{uuid4().hex}.tmp"
        temporary.write_text(
            json.dumps([asdict(s) for s in self.segments], ensure_ascii=False, indent=2), encoding="utf-8"
        )
        for attempt in range(5):
            try:
                temporary.replace(self.folder / "index.json")
                return
            except PermissionError as exc:
                if getattr(exc, "winerror", None) not in (5, 32, 33) or attempt == 4:
                    raise
                sleep(0.05 * (attempt + 1))

    def check_space(self):
        """检查真实空闲空间及会话录像总额；调用方记录停止原因，不继续生成完整状态。"""
        if shutil.disk_usage(self.folder).free < self.min_free_bytes:
            raise OSError("磁盘剩余空间低于 2 GiB，已停止录像")
        used = sum(path.stat().st_size for path in self.quota_root.rglob("*.mkv"))
        if used >= self.quota_bytes:
            raise OSError("录像总额达到配置上限，已停止录像；请手动整理资料")

    def write(self, packet: VideoPacket) -> None:
        """消费一个完整协议包；会话变化先关闭片段，等待新配置和新关键帧。"""
        if packet.session:
            self.close_segment()
            self.size = packet.width, packet.height
            self.config = b""
            return
        if packet.config:
            self.config = packet.data
            return
        if monotonic() - self.last_check >= 2:
            self.check_space()
            self.last_check = monotonic()
            self._index()
        if (
            self.container
            and packet.keyframe
            and packet.pts_us - self.segments[-1].global_start_us >= self.segment_us
        ):
            self.close_segment()
        if self.container is None:
            if not packet.keyframe:
                return
            if not self.config or not all(self.size):
                raise ValueError("录像缺少编码配置或尺寸会话")
            name = f"segment-{self._segment_sequence:04d}.mkv"
            self._segment_sequence += 1
            self.template_container = av.open(io.BytesIO(self.config + packet.data), format="h264", mode="r")
            self.container = av.open(str(self.folder / name), mode="w", format="matroska")
            self.stream = self.container.add_stream_from_template(self.template_container.streams.video[0])
            self.stream.time_base = TIME_BASE
            self.segments.append(Segment(name, packet.pts_us, packet.pts_us, *self.size))
            self._index()
        segment = self.segments[-1]
        data = av.Packet(packet.data)
        data.pts = data.dts = packet.pts_us - segment.global_start_us
        data.time_base, data.stream = TIME_BASE, self.stream
        data.is_keyframe = packet.keyframe
        self.container.mux(data)
        segment.global_end_us = packet.pts_us
        segment.frames += 1

    def close_segment(self):
        """写入容器尾部并刷新索引；关闭失败不把片段标记为已完整关闭。"""
        if self.container is not None:
            self.container.close()
            self.container = None
            self.template_container.close()
            self.template_container = None
            self.segments[-1].closed = True
        self._index()

    def close(self):
        self.close_segment()
