"""设备、画面和识别结果的数据边界；坐标一律指原始截图像素。"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from time import monotonic

import numpy as np

REGION_NAMES = {
    "hand": "自己的手牌",
    "bottom": "公共底牌",
    "left": "左侧对手出牌",
    "right": "右侧对手出牌",
    "self_play": "自己的出牌",
}
RANKS = ["3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A", "2", "SJ", "BJ"]
RANK_LABELS = {rank: rank for rank in RANKS} | {"SJ": "小王", "BJ": "大王"}


@dataclass(frozen=True)
class TrackerSettings:
    """跨帧确认参数，单位均为微秒；随会话快照保存，离线重算不读取全局新值。

    stable_us 是两次独立观察的最小间隔；pass_gap_us 仅用于相同轮次不出字样的短暂漏检。
    candidate_wait_us 限制动作证据等待时间，超时必须保留疑点，不允许静默补造历史。
    """

    stable_us: int = 100_000
    pass_gap_us: int = 300_000
    candidate_wait_us: int = 3_000_000

    def __post_init__(self):
        if any(
            not isinstance(v, int) or isinstance(v, bool)
            for v in (self.stable_us, self.pass_gap_us, self.candidate_wait_us)
        ):
            raise ValueError("事件确认时间必须是整数微秒")
        if (
            not 100_000 <= self.stable_us <= self.pass_gap_us <= 300_000
            or not self.pass_gap_us < self.candidate_wait_us <= 3_000_000
        ):
            raise ValueError("事件确认参数超出支持范围")


@dataclass(frozen=True)
class Device:
    """ADB 列表中的设备；state 保留 device、unauthorized 或 offline 等原始状态。"""

    serial: str
    state: str
    model: str = ""


@dataclass(frozen=True)
class Frame:
    """一次截图。图像为 RGB uint8；started 是采集开始的单调时钟，用于防止旧帧点击。"""

    serial: str
    rgb: np.ndarray = field(repr=False)
    started: float
    captured_at: str
    capture_ms: float
    transport: str = "raw"
    frame_id: str = ""
    stream_id: str = ""
    media_pts_us: int | None = None

    def __post_init__(self) -> None:
        """在跨模块边界验证像素格式；UI、识别和导出只接收非空三通道 RGB uint8。"""
        if (
            not isinstance(self.rgb, np.ndarray)
            or self.rgb.dtype != np.uint8
            or self.rgb.ndim != 3
            or self.rgb.shape[2] != 3
            or min(self.rgb.shape[:2]) <= 0
        ):
            raise ValueError("Frame 图像必须是非空 RGB uint8 数组")
        if (
            not self.serial
            or not self.captured_at
            or not np.isfinite(self.started)
            or not np.isfinite(self.capture_ms)
            or self.capture_ms < 0
        ):
            raise ValueError("Frame 的设备、时间或耗时无效")

    @property
    def width(self) -> int:
        return int(self.rgb.shape[1])

    @property
    def height(self) -> int:
        return int(self.rgb.shape[0])

    @property
    def age(self) -> float:
        return monotonic() - self.started


@dataclass(frozen=True)
class PixelRect:
    """左上角和宽高组成的原图矩形，右、下边界不包含在矩形中。"""

    x: int
    y: int
    width: int
    height: int

    def crop(self, rgb: np.ndarray) -> np.ndarray:
        return rgb[self.y : self.y + self.height, self.x : self.x + self.width]


@dataclass(frozen=True)
class NormalizedRect:
    """相对原图的区域，四个数值均使用 0～1 比例。保存时不依赖窗口尺寸。"""

    x: float
    y: float
    width: float
    height: float

    def __post_init__(self) -> None:
        values = [self.x, self.y, self.width, self.height]
        if (
            not all(np.isfinite(v) for v in values)
            or self.x < 0
            or self.y < 0
            or self.width <= 0
            or self.height <= 0
            or self.x + self.width > 1.00001
            or self.y + self.height > 1.00001
        ):
            raise ValueError("识别区域必须位于图像内，且宽高大于零")

    def pixels(self, width: int, height: int) -> PixelRect:
        x, y = int(self.x * width), int(self.y * height)
        right = min(width, max(x + 1, round((self.x + self.width) * width)))
        bottom = min(height, max(y + 1, round((self.y + self.height) * height)))
        return PixelRect(x, y, right - x, bottom - y)


class RegionStatus(StrEnum):
    OK = "已识别"
    EMPTY = "区域无牌"
    UNCERTAIN = "无法确认"
    NO_TEMPLATES = "缺少模板"
    NOT_TABLE = "非牌桌或布局不匹配"


@dataclass(frozen=True)
class CardObservation:
    """一张可见牌的识别结果；score 为分类器原始分数，来源见 reason。

    模板相似度和OCR分数不可互相比较，也都不是实际正确概率或对局胜率。
    """

    rank: str | None
    box: PixelRect
    score: float
    reason: str = ""


@dataclass
class RegionResult:
    region: str
    status: RegionStatus
    cards: list[CardObservation] = field(default_factory=list)
    message: str = ""


@dataclass
class Recognition:
    """与单帧绑定的识别结果，供显示及 JSON 导出；不会产生设备操作。"""

    regions: list[RegionResult]
    frame_time: str
    elapsed_ms: float
