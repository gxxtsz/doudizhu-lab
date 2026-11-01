"""使用 Qt 事件循环检验模式互斥、旧帧淘汰、点击取消与单任务调度。"""

from dataclasses import replace
from threading import Event
from time import monotonic
from unittest.mock import Mock

import numpy as np
import pytest
from PySide6.QtCore import QPoint, Qt

from doudizhu_lab.domain.models import Device, NormalizedRect, Recognition
from doudizhu_lab.storage.config import Store
from doudizhu_lab.ui.window import MainWindow


@pytest.fixture
def window(qtbot, tmp_path, frame):
    window = MainWindow(Store(tmp_path), auto_refresh=False)
    window.timer.stop()
    qtbot.addWidget(window)
    window.devices.addItem("模拟手机", Device(frame.serial, "device", "测试"))
    window.frame = frame
    window.canvas.set_frame(frame)
    window._direction_ready = 2
    window._last_size = (frame.width, frame.height)
    window.mode.setCurrentIndex(1)
    window.running = True
    window._next_capture = monotonic() + 100
    window.show()
    window._tick()
    # pytest-qt 在自己的 teardown 中关闭并 deleteLater；不能再由 fixture 二次释放。
    return window


@pytest.mark.gui
def test_view_and_calibration_cannot_send_tap(window, qtbot):
    received = []
    window.canvas.tapped.connect(lambda x, y: received.append((x, y)))
    window.mode.setCurrentIndex(0)
    qtbot.mouseClick(window.canvas, Qt.MouseButton.LeftButton, pos=window.canvas.rect().center())
    window.mode.setCurrentIndex(2)
    assert not window.running
    assert not window.canvas.control_enabled
    qtbot.mouseClick(window.canvas, Qt.MouseButton.LeftButton, pos=window.canvas.rect().center())
    assert received == []


@pytest.mark.gui
def test_canvas_black_border_and_scaled_center(window, qtbot):
    # 断开窗口发命令的槽，只检验控件发出的原图像素，测试不会连接真机。
    window.canvas.tapped.disconnect(window.request_tap)
    received = []
    window.canvas.tapped.connect(lambda x, y: received.append((x, y)))
    qtbot.mouseClick(window.canvas, Qt.MouseButton.LeftButton, pos=QPoint(1, 1))
    assert not received
    vp = window.canvas.viewport()
    qtbot.mouseClick(
        window.canvas,
        Qt.MouseButton.LeftButton,
        pos=QPoint(round(vp.x + vp.width / 2), round(vp.y + vp.height / 2)),
    )
    assert abs(received[0][0] - 1200) < 4 and abs(received[0][1] - 540) < 4


def test_stale_pause_and_direction_change_disable_input(window, frame):
    window.frame = replace(frame, started=monotonic() - 10)
    window._tick()
    assert not window.canvas.control_enabled
    result = Recognition([], "portrait", 0)
    portrait = replace(
        frame, rgb=np.zeros((2400, 1080, 3), np.uint8), captured_at="portrait", started=monotonic()
    )
    window._done(window.generation, "capture", (portrait, result), "")
    window._tick()
    assert window._direction_ready == 1
    assert not window.canvas.control_enabled
    window.pause()
    assert not window.tap_button.isEnabled()


@pytest.mark.gui
def test_cancel_during_preflight_never_taps(window, frame, qtbot):
    started, release = Event(), Event()

    def capture():
        started.set()
        release.wait(2)
        return replace(frame, started=monotonic())

    window.client.capture = capture
    window.client.tap = Mock()
    window.request_tap(100, 100)
    qtbot.waitUntil(started.is_set, timeout=1000)
    window.pause()
    release.set()
    qtbot.waitUntil(lambda: not window.busy, timeout=2000)
    window.client.tap.assert_not_called()


@pytest.mark.gui
def test_no_capture_queue_and_old_results_are_discarded(window, frame, qtbot):
    release = Event()
    assert window._submit("test", lambda: release.wait(2))
    for _ in range(20):
        assert not window._submit("test", lambda: None)
    old_generation = window.generation
    window.pause()
    old = window.frame
    window._done(
        old_generation, "capture", (replace(frame, captured_at="old"), Recognition([], "old", 0)), ""
    )
    assert window.frame is old
    release.set()
    qtbot.waitUntil(lambda: window.pool.activeThreadCount() == 0, timeout=2000)


def test_only_one_user_tap_waits_for_capture(window):
    window.busy = True
    window._kind = "capture"
    window._tick()
    assert window.canvas.control_enabled
    window.request_tap(100, 200)
    window.request_tap(300, 400)
    assert window._pending_tap[:2] == (100, 200)
    window.pause()
    assert window._pending_tap is None
    window.busy = False


@pytest.mark.gui
def test_calibration_keeps_glyph_scale_and_never_taps(window, frame, qtbot):
    window.mode.setCurrentIndex(2)
    window.client.tap = Mock()
    window.frame = replace(frame, rgb=np.zeros((720, 1600, 3), np.uint8))
    window.selection_done(NormalizedRect(0.1, 0.65, 0.8, 0.3))
    qtbot.waitUntil(lambda: not window.busy, timeout=2000)
    assert window.store.profile.base_height == 720
    assert window.store.profile.glyph_heights["hand"] == pytest.approx(74 * 720 / 1080)
    assert window.store.settings_path.exists()
    window.client.tap.assert_not_called()


@pytest.mark.gui
@pytest.mark.parametrize("layout", ["padded", "rotated", "negative"])
def test_canvas_noncontiguous_rgb_preserves_pixels_and_coordinates(qtbot, frame, layout):
    from doudizhu_lab.ui.canvas import FrameCanvas

    pixels = np.arange(5 * 9 * 3, dtype=np.uint8).reshape(5, 9, 3)
    rgb = {"padded": pixels[:, :7], "rotated": pixels.transpose(1, 0, 2), "negative": pixels[:, ::-1]}[layout]
    assert not rgb.flags.c_contiguous
    canvas = FrameCanvas()
    qtbot.addWidget(canvas)
    source = replace(frame, rgb=rgb)
    expected = rgb.copy()
    canvas.set_frame(source)
    assert canvas.frame is source
    assert (canvas._image.width(), canvas._image.height()) == (source.width, source.height)
    for x, y in ((0, 0), (source.width - 1, source.height - 1)):
        assert canvas._image.pixelColor(x, y).getRgb()[:3] == tuple(expected[y, x])
    rgb[:] = 0
    assert canvas._image.pixelColor(source.width - 1, source.height - 1).getRgb()[:3] == tuple(
        expected[-1, -1]
    )


def test_video_render_failure_revokes_input_without_stopping_capture(window, monkeypatch):
    token = window.input_gate.grant("bot")
    heartbeat = window.input_gate.heartbeat_at
    recording = object()
    window.recording = recording
    window._pending_tap = (100, 100, window.frame)
    monkeypatch.setattr(window, "_poll_video", Mock(side_effect=BufferError("测试行对齐异常")))
    try:
        window._tick()
        assert window.recording is recording
        assert window.input_gate.generation > token and window.input_gate.owner is None
        assert window.input_gate.heartbeat_at == heartbeat
        assert window._pending_tap is None and window._cancel.is_set()
        assert not window.canvas.control_enabled and not window.tap_button.isEnabled()
        assert "输入已停止" in window.status_label.text()
    finally:
        window.recording = None
