"""手机画面控件：等比预览、区域框选和原图像素点击。"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QMouseEvent, QPainter, QPen
from PySide6.QtWidgets import QWidget

from doudizhu_lab.domain.geometry import Viewport
from doudizhu_lab.domain.models import RANK_LABELS, Frame, NormalizedRect, Recognition
from doudizhu_lab.storage.config import Profile


class FrameCanvas(QWidget):
    """所有鼠标位置使用 Qt 逻辑坐标；仅在点击控制模式发射 tapped 信号。"""

    tapped = Signal(int, int)
    selected = Signal(object)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setMinimumSize(500, 300)
        self.setMouseTracking(True)
        self.setObjectName("phoneCanvas")
        self.frame: Frame | None = None
        self.recognition: Recognition | None = None
        self.profile = Profile()
        self.mode = "view"
        self.control_enabled = False
        self._drag_start: QPointF | None = None
        self._drag_end: QPointF | None = None
        self._image: QImage | None = None

    def set_frame(self, frame: Frame, recognition: Recognition | None = None) -> None:
        """规范化后复制 RGB，兼容视频行尾填充、旋转视图及负步长。

        PyAV 的 RGB 数组可能每行带对齐填充，memoryview 并非 C 连续；Qt 的
        Python 缓冲接口拒绝这种输入。只在显示边界规范化，不修改原始帧及坐标。
        QImage 自持像素副本，后台释放或重用源数组不会影响当前画面。
        """
        rgb = np.ascontiguousarray(frame.rgb)
        image = QImage(
            rgb.data, frame.width, frame.height, int(rgb.strides[0]), QImage.Format.Format_RGB888
        ).copy()
        self.frame, self.recognition, self._image = frame, recognition, image
        self.update()

    def clear(self) -> None:
        """切换设备时清空旧画面和旧识别结果，防止跨设备点击或导出。"""
        self.frame = None
        self.recognition = None
        self._image = None
        self.control_enabled = False
        self.update()

    def viewport(self) -> Viewport | None:
        if self.frame is None or min(self.width(), self.height()) <= 24:
            return None
        fit = Viewport.fit(self.width() - 24, self.height() - 24, self.frame.width, self.frame.height)
        return Viewport(fit.x + 12, fit.y + 12, fit.width, fit.height, fit.image_width, fit.image_height)

    def _rect(self, area: NormalizedRect, vp: Viewport) -> QRectF:
        return QRectF(
            vp.x + area.x * vp.width,
            vp.y + area.y * vp.height,
            area.width * vp.width,
            area.height * vp.height,
        )

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#0c1522"))
        vp = self.viewport()
        if vp is None or self._image is None:
            painter.setPen(QColor("#a4b4c8"))
            painter.drawText(
                self.rect(),
                Qt.AlignmentFlag.AlignCenter,
                "连接手机并开始预览\n\n也可以导入本地截图进行识别与校准",
            )
            return
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        painter.drawImage(QRectF(vp.x, vp.y, vp.width, vp.height), self._image)
        if self.mode in ("region", "template"):
            painter.setPen(QPen(QColor("#56d8c5"), 1, Qt.PenStyle.DashLine))
            for area in self.profile.regions.values():
                painter.drawRect(self._rect(area, vp))
        if self.recognition and self.recognition.frame_time == self.frame.captured_at:
            for result in self.recognition.regions:
                for card in result.cards:
                    painter.setPen(QPen(QColor("#55e0bc" if card.rank else "#ffb85c"), 1.5))
                    box = card.box
                    rect = QRectF(
                        vp.x + box.x / self.frame.width * vp.width,
                        vp.y + box.y / self.frame.height * vp.height,
                        box.width / self.frame.width * vp.width,
                        box.height / self.frame.height * vp.height,
                    )
                    painter.drawRect(rect)
                    label = RANK_LABELS.get(card.rank, "?")
                    painter.drawText(rect.topLeft() + QPointF(0, -4), label)
        if self._drag_start is not None and self._drag_end is not None:
            painter.setPen(QPen(QColor("#80ffe7"), 2))
            painter.setBrush(QColor(38, 190, 166, 45))
            painter.drawRect(QRectF(self._drag_start, self._drag_end).normalized())

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return
        vp = self.viewport()
        if vp is None:
            return
        pixel = vp.to_image(event.position().x(), event.position().y())
        if pixel is None:
            return
        if self.mode == "control" and self.control_enabled:
            self.tapped.emit(*pixel)
        elif self.mode in ("region", "template"):
            self._drag_start = event.position()
            self._drag_end = event.position()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._drag_start is not None:
            self._drag_end = event.position()
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self._drag_start is None:
            return
        vp = self.viewport()
        area = (
            vp.selection(
                (self._drag_start.x(), self._drag_start.y()), (event.position().x(), event.position().y())
            )
            if vp
            else None
        )
        self._drag_start = None
        self._drag_end = None
        self.update()
        if area is not None and self.mode in ("region", "template"):
            self.selected.emit(area)
