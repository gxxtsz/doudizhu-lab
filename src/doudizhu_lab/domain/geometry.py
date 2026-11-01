"""显示窗口与原始截图间的纯坐标换算，不依赖系统物理 DPI。"""

import math
from dataclasses import dataclass

from doudizhu_lab.domain.models import NormalizedRect


@dataclass(frozen=True)
class Viewport:
    """Qt 逻辑像素中的等比显示区域；Qt 已处理设备像素比，不再额外乘 DPI。"""

    x: float
    y: float
    width: float
    height: float
    image_width: int
    image_height: int

    @classmethod
    def fit(cls, view_width: float, view_height: float, image_width: int, image_height: int) -> "Viewport":
        """计算居中等比缩放后的图像区域；拒绝零尺寸以免错误映射点击。"""
        if min(view_width, view_height, image_width, image_height) <= 0:
            raise ValueError("窗口和图像尺寸必须大于零")
        scale = min(view_width / image_width, view_height / image_height)
        width, height = image_width * scale, image_height * scale
        return cls(
            (view_width - width) / 2, (view_height - height) / 2, width, height, image_width, image_height
        )

    def to_image(self, x: float, y: float) -> tuple[int, int] | None:
        """将窗口逻辑坐标转为原图整数像素；落在黑边或右下排除边界时返回 None。"""
        if (
            not math.isfinite(x)
            or not math.isfinite(y)
            or not (self.x <= x < self.x + self.width and self.y <= y < self.y + self.height)
        ):
            return None
        return (
            min(self.image_width - 1, int((x - self.x) / self.width * self.image_width)),
            min(self.image_height - 1, int((y - self.y) / self.height * self.image_height)),
        )

    def selection(self, start: tuple[float, float], end: tuple[float, float]) -> NormalizedRect | None:
        """拖拽生成归一化区域；起点和终点都必须在图像内，至少覆盖 3×3 原图像素。"""
        a, b = self.to_image(*start), self.to_image(*end)
        if a is None or b is None:
            return None
        left, right = sorted((a[0], b[0]))
        top, bottom = sorted((a[1], b[1]))
        if right - left < 3 or bottom - top < 3:
            return None
        return NormalizedRect(
            left / self.image_width,
            top / self.image_height,
            (right - left + 1) / self.image_width,
            (bottom - top + 1) / self.image_height,
        )
