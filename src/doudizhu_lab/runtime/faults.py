"""连续任务故障的结构化分类；恢复策略不依赖中文提示内容。"""

from dataclasses import dataclass
from enum import StrEnum


class FaultScope(StrEnum):
    """单局隔离、执行会话重建、外部条件等待三种恢复边界。"""

    ROUND = "round"
    SESSION = "session"
    EXTERNAL = "external"


class StorageUnavailable(OSError):
    """空间或录像配额不可用；稳定 code 用于自动等待，不解析异常中文文案。"""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class RuntimeFault:
    """只读故障事实；code 为稳定代码，message 仅展示，不授予设备输入。"""

    code: str
    message: str
    scope: FaultScope = FaultScope.SESSION


@dataclass(frozen=True)
class RecoveryPolicy:
    """一次任务冻结的恢复参数；秒数使用单调时钟，不限制总尝试次数。"""

    backoff: tuple[int, ...] = (2, 5, 10, 30, 60)
    startup_seconds: float = 90
    shutdown_seconds: float = 45
    external_seconds: float = 60
    media_games: int = 20
    media_bytes: int = 8 * 1024**3
    cleanup_free_bytes: int = 3 * 1024**3
