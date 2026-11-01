"""通过独立参数列表调用 ADB；不使用 shell 拼接，不改变手机系统配置。"""

from __future__ import annotations

import os
import shutil
import struct
import subprocess
import threading
from datetime import datetime, timezone
from time import monotonic

import cv2
import numpy as np

from doudizhu_lab.domain.models import Device, Frame


class AdbError(RuntimeError):
    """中文诊断及原始 ADB 返回信息，避免排查时只剩下笼统的权限提示。

    command 保存实际参数列表，returncode 保存子进程退出码；stdout/stderr 保留原始字节。
    截图输出可能为二进制，因此不自动把完整 stdout 写入界面或日志。
    """

    def __init__(
        self,
        message: str,
        *,
        command: list[str] | None = None,
        returncode: int | None = None,
        stdout: bytes = b"",
        stderr: bytes = b"",
    ):
        super().__init__(message)
        self.command = tuple(command or ())
        self.returncode = returncode
        self.stdout, self.stderr = stdout, stderr


class InputRejected(AdbError):
    """输入在本地校验阶段被拒绝，ADB尚未调用；与发送后的超时或结果未知严格区分。"""


class RawFormatError(ValueError):
    """像素格式不支持时可回退 PNG；长度错误、截断等数据损坏不属于此类。"""


def discover_adb() -> str:
    """读取显式ADB_PATH或PATH；调用者提供的设备配置优先，不依赖任何本机固定盘符。"""
    configured = os.environ.get("ADB_PATH")
    return configured or shutil.which("adb") or "adb"


def decode_raw(data: bytes) -> np.ndarray:
    """解析 Android screencap 原始帧，支持 12/16 字节头的 RGBA/RGBX/RGB。

    通过完整长度校验区分头格式，拒绝截断、额外字节和异常大尺寸。
    返回独立连续 RGB 数组，避免引用子进程输出缓冲区导致生命周期问题。
    """
    if len(data) < 12:
        raise ValueError("原始截图头不完整")
    width, height, pixel_format = struct.unpack_from("<III", data)
    if not (0 < width <= 10000 and 0 < height <= 10000 and width * height <= 40_000_000):
        raise ValueError("截图尺寸异常")
    channels = {1: 4, 2: 4, 3: 3}.get(pixel_format)
    if channels is None:
        raise RawFormatError(f"尚不支持原始像素格式 {pixel_format}")
    payload = width * height * channels
    header = len(data) - payload
    if header not in (12, 16):
        raise ValueError("原始截图长度不符或数据被截断")
    if (
        header == 12
        and pixel_format in (1, 2)
        and len(data) >= 16
        and struct.unpack_from("<I", data, 12)[0] in (0, 1, 2)
    ):
        raise ValueError("疑似带色彩空间头的截图被截断了 4 字节")
    return np.frombuffer(data, np.uint8, offset=header).reshape(height, width, channels)[:, :, :3].copy()


class AdbClient:
    """设备传输层。serial 明确绑定设备；单实例互斥执行命令，防止点击和截图竞争。"""

    def __init__(self, executable: str | None = None, serial: str = "", timeout: float = 8):
        self.executable = executable or discover_adb()
        self.serial = serial
        self.timeout = timeout
        self._lock = threading.RLock()
        self._png_only = False

    def _run(self, args: list[str], *, device: bool = True) -> bytes:
        """执行一次受超时限制的命令；Windows 隐藏控制台，stderr 转换为中文诊断。"""
        command = [self.executable]
        if device:
            if not self.serial:
                raise AdbError("尚未选择手机")
            command += ["-s", self.serial]
        command += args
        try:
            with self._lock:
                result = subprocess.run(
                    command,
                    capture_output=True,
                    timeout=self.timeout,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
        except subprocess.TimeoutExpired as exc:
            raise AdbError(f"ADB 操作超过 {self.timeout:g} 秒，请检查 USB 连接") from exc
        except OSError as exc:
            raise AdbError(f"无法启动 ADB，请检查路径：{self.executable}") from exc
        error = result.stderr.decode("utf-8", errors="replace").strip()
        # 部分 Android 版本把 input 异常写到 stdout，退出码仍为 0。
        # 只对已知的文本输入命令检查 stdout，不解码二进制截图。
        is_input = args[:2] == ["shell", "input"]
        input_output = result.stdout.decode("utf-8", errors="replace").strip() if is_input else ""
        diagnostic = "\n".join(part for part in (error, input_output) if part)
        if (
            result.returncode
            or "INJECT_EVENTS" in diagnostic
            or "SecurityException" in diagnostic
            or (is_input and "Exception" in diagnostic)
        ):
            if "INJECT_EVENTS" in diagnostic:
                message = "手机拒绝输入（INJECT_EVENTS），请检查小米“USB 调试（安全设置）”并重新验证"
            elif "SecurityException" in diagnostic:
                message = "手机拒绝此 ADB 操作，具体权限见原始错误"
            elif "unauthorized" in diagnostic:
                message = "手机尚未授权，请在手机端允许这台电脑进行 USB 调试"
            elif "offline" in diagnostic:
                message = "设备离线，请重新连接 USB 并保持手机解锁"
            else:
                message = "ADB 操作失败"
            lines = diagnostic.splitlines()
            original = next(
                (line for line in lines if "Exception" in line or "INJECT_EVENTS" in line),
                lines[0] if lines else "设备未响应",
            )
            raise AdbError(
                f"{message}\n原始错误：{original[:1000]}",
                command=command,
                returncode=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
            )
        return result.stdout

    def list_devices(self) -> list[Device]:
        """读取所有 ADB 设备及状态，不擅自选择多台设备中的某一台。"""
        output = self._run(["devices", "-l"], device=False).decode("utf-8", errors="replace")
        devices = []
        for line in output.splitlines():
            parts = line.split()
            if len(parts) < 2 or parts[0] == "List" or parts[0].startswith("*"):
                continue
            model = next((item[6:] for item in parts[2:] if item.startswith("model:")), "")
            devices.append(Device(parts[0], parts[1], model))
        return devices

    def capture(self) -> Frame:
        """获取当前画面，优先使用原始帧；仅格式解析失败时回退 PNG，连接错误直接上报。"""
        start = monotonic()
        transport = "raw"
        with self._lock:
            if not self._png_only:
                raw = self._run(["exec-out", "screencap"])
                try:
                    rgb = decode_raw(raw)
                except RawFormatError:
                    self._png_only = True
                except ValueError as exc:
                    raise AdbError(f"原始截图无效：{exc}") from exc
            if self._png_only:
                transport = "png"
                data = self._run(["exec-out", "screencap", "-p"])
                if not data.startswith(b"\x89PNG\r\n\x1a\n"):
                    raise AdbError("手机返回的 PNG 截图不完整")
                bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                if bgr is None:
                    raise AdbError("无法解码手机截图")
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return Frame(
            self.serial,
            rgb,
            start,
            datetime.now(timezone.utc).isoformat(),
            (monotonic() - start) * 1000,
            transport,
        )

    def tap(self, x: int, y: int, frame: Frame, *, max_age: float = 3.0) -> None:
        """在与当前设备关联的新鲜画面上点击。

        x/y 为截图原始像素；拒绝设备不匹配、越界和超过 max_age 秒的旧画面。
        返回仅表示输入命令未报错，调用方仍须刷新画面验证实际操作结果。
        """
        with self._lock:
            if frame.transport not in ("raw", "png", "video"):
                raise InputRejected("离线截图和回放画面禁止发送手机输入")
            if frame.serial != self.serial:
                raise InputRejected("画面所属设备已变化，请重新获取画面")
            if frame.age > max_age:
                raise InputRejected("画面已过期，请等待新截图后再点击")
            if (
                isinstance(x, bool)
                or isinstance(y, bool)
                or not isinstance(x, int)
                or not isinstance(y, int)
                or not (0 <= x < frame.width and 0 <= y < frame.height)
            ):
                raise InputRejected("点击坐标必须是原图范围内的整数像素")
            self._run(["shell", "input", "tap", str(x), str(y)])
