"""使用 Qt 事件循环检验模式互斥、旧帧淘汰、点击取消与单任务调度。"""

from dataclasses import replace
from threading import Event
from time import monotonic
from types import SimpleNamespace
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


@pytest.fixture
def continuous_backend(window, monkeypatch, tmp_path):
    """替换连续启动的模型、录像及执行后端，测试只运行真实的 GUI 调度流程。"""
    pipeline = Mock()
    pipeline.session.folder = tmp_path / "模拟录像"
    runtime = Mock(status="running", store=SimpleNamespace(folder=tmp_path / "模拟连续任务"))
    recording_factory = Mock(return_value=pipeline)
    runtime_factory = Mock(return_value=runtime)
    monkeypatch.setattr("doudizhu_lab.vision.assets.model_path", Mock())
    monkeypatch.setattr("doudizhu_lab.ui.window.RecordingPipeline", recording_factory)
    monkeypatch.setattr("doudizhu_lab.runtime.continuous.ContinuousRuntime", runtime_factory)
    monkeypatch.setattr(window, "load_bot", Mock())
    monkeypatch.setattr(window.client, "capture", Mock())
    monkeypatch.setattr(window.client, "tap", Mock())
    yield SimpleNamespace(
        pipeline=pipeline,
        runtime=runtime,
        recording_factory=recording_factory,
        runtime_factory=runtime_factory,
    )
    window.recording = None
    window.continuous = None


@pytest.mark.gui
@pytest.mark.parametrize("kind", ["capture", "recognize"])
def test_continuous_start_waits_for_preview_and_starts_once(window, continuous_backend, frame, qtbot, kind):
    release = Event()
    previous = window.frame

    def read_frame():
        release.wait(5)
        return replace(frame, captured_at="暂停前的迟到帧"), Recognition([], "旧画面", 0)

    assert window._submit(kind, read_frame)
    window._pending_tap = (100, 200, frame)
    window._pending_recognition = True
    try:
        window.continuous_start.click()
        window.continuous_start.click()
        assert window.busy and window._continuous_requested and not window.running
        assert window._pending_tap is None and not window._pending_recognition
        assert "等待当前画面" in window.continuous_label.text()
        window._tick()
        continuous_backend.recording_factory.assert_not_called()
        window.load_bot.assert_not_called()
        assert window.pool.activeThreadCount() == 1
    finally:
        release.set()
    qtbot.waitUntil(lambda: continuous_backend.runtime.start.called, timeout=3000)
    assert window.recording is continuous_backend.pipeline
    assert window.continuous is continuous_backend.runtime
    assert window.frame is previous
    assert not window._continuous_requested and not window.busy
    window.load_bot.assert_called_once_with()
    continuous_backend.recording_factory.assert_called_once()
    continuous_backend.pipeline.start.assert_called_once_with()
    continuous_backend.runtime_factory.assert_called_once()
    window.client.capture.assert_not_called()
    window.client.tap.assert_not_called()


@pytest.mark.gui
@pytest.mark.parametrize("cancel", ["stop", "pause", "device", "close"])
def test_waiting_continuous_start_can_be_cancelled(window, continuous_backend, frame, qtbot, cancel):
    release = Event()

    def read_frame():
        release.wait(5)
        return frame, Recognition([], "旧画面", 0)

    assert window._submit("capture", read_frame)
    try:
        window.continuous_start.click()
        assert window._continuous_requested
        if cancel == "close":
            # 任务已返回但尚未由 GUI 收取时关闭，也不能消费待启动请求。
            release.set()
            window.close()
        else:
            {"stop": window.stop_bot, "pause": window.pause, "device": window._device_changed}[cancel]()
    finally:
        release.set()
    if cancel != "close":
        qtbot.waitUntil(lambda: not window.busy, timeout=2000)
    assert not window._continuous_requested
    assert window.input_gate.owner is None
    continuous_backend.recording_factory.assert_not_called()
    continuous_backend.runtime_factory.assert_not_called()
    window.load_bot.assert_not_called()


@pytest.mark.gui
@pytest.mark.parametrize("kind", ["devices", "tap", "record_start", "record_stop"])
def test_continuous_start_does_not_interrupt_other_tasks(window, continuous_backend, qtbot, kind):
    release = Event()
    assert window._submit(kind, lambda: release.wait(5))
    try:
        window.continuous_start.click()
        assert not window._continuous_requested
        assert "正在启动或收尾" in window.log_view.toPlainText()
        continuous_backend.recording_factory.assert_not_called()
    finally:
        window.pause()
        release.set()
    qtbot.waitUntil(lambda: not window.busy, timeout=2000)
    continuous_backend.runtime_factory.assert_not_called()


@pytest.mark.gui
def test_stop_cancels_continuous_launch_already_queued_by_video(window, continuous_backend, qtbot):
    window._continuous_requested = True
    window._done(window.generation, "record_start", continuous_backend.pipeline, "")
    window.bot_stop.click()
    qtbot.wait(50)
    assert window.recording is continuous_backend.pipeline
    assert not window._continuous_requested and window.input_gate.owner is None
    continuous_backend.runtime_factory.assert_not_called()


@pytest.mark.gui
@pytest.mark.parametrize("failure", ["assets", "recording"])
def test_deferred_continuous_start_reports_failure_without_retry(
    window, continuous_backend, frame, qtbot, monkeypatch, failure
):
    if failure == "assets":
        monkeypatch.setattr(
            "doudizhu_lab.vision.assets.model_path", Mock(side_effect=ValueError("测试模型不可用"))
        )
    else:
        continuous_backend.pipeline.start.side_effect = ValueError("测试视频不可用")
    release = Event()

    def read_frame():
        release.wait(5)
        return frame, Recognition([], "旧画面", 0)

    assert window._submit("capture", read_frame)
    try:
        window.continuous_start.click()
        assert window._continuous_requested
    finally:
        release.set()
    qtbot.waitUntil(lambda: not window.busy and not window._continuous_requested, timeout=3000)
    assert "启动失败" in window.continuous_label.text()
    assert window.recording is None and window.input_gate.owner is None
    continuous_backend.runtime_factory.assert_not_called()
    assert continuous_backend.pipeline.start.call_count == int(failure == "recording")


@pytest.mark.gui
def test_continuous_start_reuses_current_video_session(window, continuous_backend):
    window.recording = continuous_backend.pipeline
    window.continuous_start.click()
    assert window.continuous is continuous_backend.runtime
    assert not window._continuous_requested
    continuous_backend.runtime.start.assert_called_once_with()
    continuous_backend.recording_factory.assert_not_called()


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
