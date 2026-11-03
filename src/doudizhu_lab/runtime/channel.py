"""运行时 JSON 通道的暂态容错；健康有效期由消息源时间决定，不能由读取时间刷新。"""

from __future__ import annotations

import json
import math
from pathlib import Path
from time import monotonic

from doudizhu_lab.storage.json_io import atomic_json, retry_file_io, temporary_access_error


class RuntimeChannel:
    """单个报告线程独占的文件通道；统计暂态告警，不持有数据库或录像写入锁。"""

    def __init__(self) -> None:
        """初始化有界诊断统计；只保留最近错误，不累积逐次异常对象。"""
        self.retries = self.read_failures = self.write_failures = 0
        self.last_error = ""
        self.last_error_at = None

    def _warning(self, exc: BaseException) -> None:
        self.retries += 1
        self.last_error, self.last_error_at = str(exc), monotonic()

    def snapshot(self) -> dict:
        """返回可写入健康记录的告警统计；告警自身不表示已触发故障。"""
        return {
            "retries": self.retries,
            "read_failures": self.read_failures,
            "write_failures": self.write_failures,
            "last_error": self.last_error,
            "last_error_at": self.last_error_at,
        }

    def read(self, path: Path, *, timestamp: str | None = None) -> dict | None:
        """读取对象消息，暂态冲突或无有效消息时返回 None；其他 I/O 错误原样上抛。

        timestamp 指定必须存在的源单调时钟字段，拒绝缺失、非数值和非有限时间。
        本接口不缓存消息；调用方只能沿用原始时间戳缓存，不得将读取成功当作进度。
        """

        def read() -> dict:
            value = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                raise ValueError("运行时消息必须是对象")
            if timestamp is not None:
                stamp = value.get(timestamp)
                if isinstance(stamp, bool) or not isinstance(stamp, (int, float)) or not math.isfinite(stamp):
                    raise ValueError(f"运行时消息缺少有效源时间：{timestamp}")
            return value

        try:
            return retry_file_io(read, invalid_json=True, on_retry=self._warning)
        except FileNotFoundError:
            return None
        except (OSError, ValueError) as exc:
            if not temporary_access_error(exc) and not isinstance(exc, (ValueError, UnicodeError)):
                raise
            self.read_failures += 1
            self.last_error, self.last_error_at = str(exc), monotonic()
            return None

    def write(self, path: Path, value: dict) -> bool:
        """原子发布消息；短暂访问冲突耗尽预算返回 False，磁盘不足等错误立即上抛。

        False 表示消息尚未发布，调用方须继续检查旧消息有效期或保留待发命令，不能
        据此认定录像、健康报告或控制命令已经成功。
        """
        try:
            atomic_json(path, value, on_retry=self._warning)
            return True
        except OSError as exc:
            if not temporary_access_error(exc):
                raise
            self.write_failures += 1
            self.last_error, self.last_error_at = str(exc), monotonic()
            return False

    def remove(self, path: Path) -> bool:
        """确认命令已应用后删除消息；占用时返回 False，调用方不得重复执行该命令。"""
        try:
            retry_file_io(lambda: path.unlink(missing_ok=True), on_retry=self._warning)
            return True
        except OSError as exc:
            if not temporary_access_error(exc):
                raise
            self.write_failures += 1
            self.last_error, self.last_error_at = str(exc), monotonic()
            return False
