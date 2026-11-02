"""工作台真实 Qt 调度回归；只用模拟模型和隔离设备，禁止连接手机。"""

from copy import deepcopy
from dataclasses import replace
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from PySide6.QtCore import QPoint, Qt, QTime
from PySide6.QtWidgets import QDialogButtonBox, QLabel, QPushButton

from doudizhu_lab.domain.models import Device
from doudizhu_lab.runtime.bot import BotRuntime, ModelStatus
from doudizhu_lab.runtime.continuous import ContinuousProgress
from doudizhu_lab.storage.config import Store
from doudizhu_lab.ui.settings import SettingsDialog
from doudizhu_lab.ui.window import MainWindow


class LocalAgent:
    """不读取权重也不创建进程的加载替身；推理能力由生产模型测试覆盖。"""

    def start(self):
        pass

    def close(self):
        pass

    def interrupt(self):
        pass


@pytest.fixture
def window(qtbot, tmp_path, frame):
    window = MainWindow(Store(tmp_path), auto_refresh=False, auto_load_model=False, auto_preview=False)
    qtbot.addWidget(window)
    window.timer.stop()
    window.available_devices = [Device(frame.serial, "device", "模拟手机")]
    window.select_device(frame.serial)
    window.client.capture = Mock(return_value=frame)
    window.client.tap = Mock()
    window.bot = BotRuntime(tmp_path, window.input_gate, agent=LocalAgent(), threaded=False)
    window.bot.start()
    window.frame = frame
    window.canvas.set_frame(frame)
    window.show()
    window.render_dashboard()
    return window


class FakeContinuous:
    """任务替身仍使用真实输入闸门，便于确定性验证 GUI 停止与恢复。"""

    def __init__(self, root, gate, bot, pipeline):
        self.gate, self.bot = gate, bot
        self.status, self.reason = "idle", ""
        self.close_window_requested = self.stop_recording_requested = False
        self.completed = 0
        self.closed = 0

    def progress(self):
        return ContinuousProgress(self.status, "lobby", self.reason, self.completed, 5, 12, 1800)

    def start(self):
        self.status = "running"
        self.gate.grant("flow")

    def stop(self):
        self.gate.revoke()
        if self.status != "completed":
            self.status = "stopped"

    def request_stop(self, reason="用户停止连续任务"):
        self.stop()

    def close(self):
        self.stop()
        self.closed += 1

    def resume(self):
        if self.reason == "真实历史缺口":
            raise ValueError(self.reason)
        self.status, self.reason = "running", ""

    def pause(self, reason):
        self.gate.revoke()
        self.status, self.reason = "paused", reason

    def poll_stop_request(self):
        pass


@pytest.fixture
def backend(window, monkeypatch, tmp_path):
    pipeline = SimpleNamespace(
        profile=deepcopy(window.store.profile),
        session=SimpleNamespace(folder=tmp_path / "录像", close=Mock()),
        start=Mock(),
        stop=Mock(),
        bot_snapshot=Mock(return_value=None),
        poll=Mock(return_value={"frame": None, "state": None, "allowed": True, "stopped": False}),
    )
    factory = Mock(return_value=pipeline)
    runtime = Mock(side_effect=FakeContinuous)
    monkeypatch.setattr("doudizhu_lab.ui.window.RecordingPipeline", factory)
    monkeypatch.setattr("doudizhu_lab.ui.window.ContinuousRuntime", runtime)
    return SimpleNamespace(pipeline=pipeline, factory=factory, runtime=runtime)


@pytest.mark.gui
def test_workspace_is_readonly_and_has_no_legacy_controls(window, qtbot):
    hits = []
    window.canvas.tapped.connect(lambda *xy: hits.append(xy))
    qtbot.mouseClick(window.canvas, Qt.MouseButton.LeftButton, pos=window.canvas.rect().center())
    assert not hits and not window.canvas.control_enabled and window.canvas.recognition is None
    assert not window.log_view.isVisible()
    for name in ("tap_button", "bot_mode", "devices", "result_tree", "x_input", "import_button"):
        assert not hasattr(window, name)
    assert window.acceptance is False
    buttons = [b.text() for b in window.findChildren(QPushButton)]
    assert "开始连续游戏" in buttons and "停止并保存" in buttons
    assert "加载 WP 模型" not in buttons and "原图坐标点击" not in buttons
    window.log("中文状态 <b>也作为纯文本</b>")
    assert "<b>也作为纯文本</b>" in window.log_view.toPlainText()
    window.log_toggle.click()
    assert window.log_view.isVisible()


@pytest.mark.gui
def test_continuous_waits_for_old_capture_and_starts_once(window, backend, frame, qtbot):
    release = Event()
    old_frame = window.frame
    window.running = True
    assert window._submit("capture", lambda: (release.wait(5), replace(frame, captured_at="迟到"))[1])
    try:
        window.action_button.click()
        window.start_continuous()
        assert window._requested_action == "continuous" and not window.running
        assert not window.action_button.isEnabled() and window.stop_button.isEnabled()
        assert "等待读取" in window.message or "等待当前画面" in window.message
        backend.factory.assert_not_called()
    finally:
        release.set()
    qtbot.waitUntil(lambda: window.continuous is not None, timeout=3000)
    assert window.frame is old_frame
    backend.factory.assert_called_once()
    backend.runtime.assert_called_once()
    backend.pipeline.start.assert_called_once()
    window.client.tap.assert_not_called()
    window.stop_and_save()
    qtbot.waitUntil(lambda: window.recording is None and not window.busy)


@pytest.mark.gui
@pytest.mark.parametrize("closing", [False, True])
def test_stop_or_close_during_video_start_cleans_late_pipeline(window, backend, qtbot, closing):
    started, release = Event(), Event()

    def start():
        started.set()
        release.wait(5)

    backend.pipeline.start.side_effect = start
    window.start_continuous()
    qtbot.waitUntil(started.is_set)
    try:
        window.close() if closing else window.stop_button.click()
        window.stop_and_save()
        assert window.input_gate.owner is None and not window._requested_action
        assert not window.action_button.isEnabled()
    finally:
        release.set()
    qtbot.waitUntil(lambda: not window.busy and window.recording is None, timeout=3000)
    backend.runtime.assert_not_called()
    backend.pipeline.start.assert_called_once()
    backend.pipeline.stop.assert_called_once()
    backend.pipeline.session.close.assert_called_once()
    if closing:
        qtbot.waitUntil(lambda: not window.isVisible())
    else:
        assert window.task_status == "stopped" and window.saved_replay_button.isVisible()


@pytest.mark.gui
def test_cancel_while_waiting_for_capture_does_not_create_recording(window, backend, frame, qtbot):
    release = Event()
    assert window._submit("capture", lambda: (release.wait(5), frame)[1])
    try:
        window.start_continuous()
        window.stop_button.click()
        assert not window._requested_action and window.input_gate.owner is None
    finally:
        release.set()
    qtbot.waitUntil(lambda: not window.busy)
    assert window.task_status == "stopped"
    backend.factory.assert_not_called()


@pytest.mark.gui
def test_stop_revokes_input_before_async_save_and_retains_summary(window, backend, qtbot):
    window.start_continuous()
    qtbot.waitUntil(lambda: window.continuous is not None)
    runtime = window.continuous
    runtime.completed = 2
    started, release = Event(), Event()
    backend.pipeline.stop.side_effect = lambda: (started.set(), release.wait(5))
    try:
        window.stop_button.click()
        assert window.input_gate.owner is None
        qtbot.waitUntil(started.is_set)
        window.stop_button.click()
        assert window.task_status == "stopping" and not window.replay_button.isEnabled()
    finally:
        release.set()
    qtbot.waitUntil(lambda: window.recording is None and not window.busy)
    assert runtime.closed == 1
    backend.pipeline.stop.assert_called_once()
    assert window.dashboard_state().completed == 2
    assert window.dashboard_state().elapsed == 12
    assert window.replay_button.isEnabled() and window.bot.ready.is_set()


@pytest.mark.gui
def test_pause_reason_and_rejected_resume_are_visible_with_log_folded(window, backend, qtbot):
    window.start_continuous()
    qtbot.waitUntil(lambda: window.continuous is not None)
    window.continuous.pause("真实历史缺口")
    window.render_dashboard()
    assert "真实历史缺口" in window.notice_label.text()
    assert window.action_button.text() == "恢复任务" and not window.log_view.isVisible()
    window.action_button.click()
    assert "恢复被拒绝" in window.notice_label.text()
    window.continuous.reason = "短暂等待"
    window.action_button.click()
    assert window.continuous.status == "running"
    window.stop_and_save()
    qtbot.waitUntil(lambda: window.recording is None and not window.busy)


@pytest.mark.gui
@pytest.mark.parametrize("failure", ["启动失败", "保存失败"])
def test_recording_failures_remain_actionable(window, backend, qtbot, failure):
    if failure == "启动失败":
        backend.pipeline.start.side_effect = ValueError(failure)
    else:
        backend.pipeline.stop.side_effect = ValueError(failure)
    window.start_continuous()
    if failure == "保存失败":
        qtbot.waitUntil(lambda: window.continuous is not None)
        window.stop_and_save()
    qtbot.waitUntil(lambda: window.task_status == "failed" and not window.busy)
    assert failure in window.notice_label.text()
    assert not window.settings_locked and window.input_gate.owner is None
    assert not window.saved_replay_button.isVisible()
    backend.pipeline.session.close.assert_called_once()
    backend.pipeline.stop.assert_called_once()


@pytest.mark.gui
@pytest.mark.parametrize("finish", ["completed", "interrupted"])
def test_automatic_finish_saves_and_resumes_readonly_preview(window, backend, qtbot, finish):
    window.auto_preview = True
    window.start_continuous()
    qtbot.waitUntil(lambda: window.continuous is not None)
    if finish == "completed":
        window.continuous.completed = 5
        window.continuous.status, window.continuous.reason = "completed", "已达局数上限"
    else:
        backend.pipeline.poll.return_value["stopped"] = True
    window._tick()
    qtbot.waitUntil(lambda: window.recording is None and not window.busy)
    assert window.running and window.input_gate.owner is None
    assert window.task_status == ("completed" if finish == "completed" else "failed")
    assert window.dashboard_state().saved


@pytest.mark.gui
def test_settings_roundtrip_preserves_calibration_and_seconds(window, qtbot):
    before = deepcopy(window.store.profile.bot_controls)
    dialog = SettingsDialog(window)
    qtbot.addWidget(dialog)
    dialog.show()
    dialog.room.setCurrentIndex(dialog.room.findData("高级"))
    dialog.policy.setCurrentIndex(dialog.policy.findData("balanced"))
    dialog.bidding.setCurrentIndex(dialog.bidding.findData("model"))
    dialog.games.setValue(7)
    dialog.time_limit.setTime(QTime(0, 12, 34))
    dialog.save()
    saved = Store(window.store.root)
    assert saved.settings_version == 4 and saved.profile.bot_controls == before
    assert saved.profile.flow_room == "高级" and saved.profile.continuous_settings == {
        "max_games": 7,
        "time_limit": 754,
    }
    assert saved.profile.continuous_policy == "balanced" and saved.profile.bid_settings["mode"] == "model"


@pytest.mark.gui
def test_settings_are_readonly_for_every_recording_phase(window, backend, qtbot):
    window.start_continuous()
    dialog = SettingsDialog(window)
    qtbot.addWidget(dialog)
    for status in ("preparing", "running", "paused", "stopping"):
        window.task_status = status
        dialog.sync()
        assert not dialog.task.isEnabled() and not dialog.connection.isEnabled()
        assert not dialog.buttons.button(QDialogButtonBox.StandardButton.Save).isEnabled()
    qtbot.waitUntil(lambda: window.continuous is not None)
    with pytest.raises(ValueError):
        window.select_device("")
    window.stop_and_save()
    qtbot.waitUntil(lambda: window.recording is None and not window.busy)


@pytest.mark.gui
def test_failed_settings_save_rolls_back_memory(window, monkeypatch, qtbot):
    original = window.store.profile
    monkeypatch.setattr(window.store, "save", Mock(side_effect=OSError("磁盘不可写")))
    dialog = SettingsDialog(window)
    qtbot.addWidget(dialog)
    dialog.games.setValue(20)
    dialog.save()
    assert window.store.profile is original and "磁盘不可写" in dialog.error.text()


@pytest.mark.gui
@pytest.mark.parametrize(
    "devices,expected",
    [
        ([], ""),
        ([Device("u", "unauthorized")], ""),
        ([Device("a", "device"), Device("u", "unauthorized")], "a"),
        ([Device("a", "device"), Device("b", "device")], ""),
    ],
)
def test_discovery_only_auto_selects_unique_authorized_device(window, devices, expected):
    window.auto_preview = True
    window.selected_device = None
    window._done(window.generation, "devices", devices, "")
    assert (window.selected_device.serial if window.selected_device else "") == expected
    assert window.running == bool(expected)
    assert window.input_gate.owner is None


@pytest.mark.gui
def test_explicit_recording_precedes_default_preview(window, backend, qtbot):
    window.auto_preview = True
    window._startup_action = "record"
    devices = window.available_devices
    window.selected_device = None
    window._done(window.generation, "devices", devices, "")
    assert not window.running
    window._tick()
    qtbot.waitUntil(lambda: window.recording is not None)
    backend.runtime.assert_not_called()
    assert window.input_gate.owner is None
    window.stop_and_save()
    qtbot.waitUntil(lambda: window.recording is None and not window.busy)


@pytest.mark.gui
def test_explicit_continuous_waits_for_model_and_consumes_startup_once(window, backend, qtbot):
    window._startup_action = "continuous"
    window.bot._model_status = ModelStatus("loading")
    window._tick()
    backend.factory.assert_not_called()
    window.client.capture.assert_not_called()
    assert window._startup_action == "continuous"
    window.bot._model_status = ModelStatus("ready")
    window._tick()
    qtbot.waitUntil(lambda: window.continuous is not None)
    window._tick()
    backend.factory.assert_called_once()
    backend.runtime.assert_called_once()
    assert not window._startup_action
    window.stop_and_save()
    qtbot.waitUntil(lambda: window.recording is None and not window.busy)


@pytest.mark.gui
def test_no_capture_queue_and_device_change_discards_old_frame(window, frame, qtbot):
    release = Event()
    assert window._submit("capture", lambda: (release.wait(5), frame)[1])
    try:
        for _ in range(20):
            assert not window._submit("capture", window.client.capture)
        window.available_devices.append(Device("second", "device", "第二台模拟手机"))
        window.select_device("second")
    finally:
        release.set()
    qtbot.waitUntil(lambda: not window.busy)
    assert window.frame is None and window.canvas.frame is None
    assert window.selected_device.serial == "second" and window.pool.maxThreadCount() == 1


@pytest.mark.gui
def test_video_render_error_revokes_input_without_stopping_capture(window, backend, monkeypatch, qtbot):
    window.start_continuous()
    qtbot.waitUntil(lambda: window.continuous is not None)
    heartbeat = window.input_gate.heartbeat_at
    monkeypatch.setattr(window, "_poll_video", Mock(side_effect=BufferError("测试行尾填充")))
    window._tick()
    assert window.input_gate.owner is None and window.input_gate.heartbeat_at == heartbeat
    assert window.recording is backend.pipeline and window.continuous.status == "paused"
    assert "输入已停止" in window.notice_label.text()
    window.stop_and_save()
    qtbot.waitUntil(lambda: window.recording is None and not window.busy)


@pytest.mark.gui
@pytest.mark.parametrize("status", ["idle", "preparing", "running", "paused", "failed", "stopping"])
def test_small_window_keeps_primary_controls_visible(window, status, qtbot):
    window.resize(1100, 720)
    window.task_status = status
    window.message = (
        "暂停原因：等待新的可靠画面，确认手机连接及前台状态后恢复。" if status == "paused" else ""
    )
    window.render_dashboard()
    qtbot.wait(10)
    for widget in (window.action_button, window.stop_button, window.settings_button, window.replay_button):
        point = widget.mapTo(window, QPoint(0, 0))
        assert window.rect().contains(point) and widget.width() > 50
    assert window.canvas.width() >= 440 and window.canvas.height() >= 260
    assert not any("非牌桌" in item.text() for item in window.findChildren(QLabel))


@pytest.mark.gui
@pytest.mark.parametrize("layout", ["padded", "rotated", "negative"])
def test_canvas_noncontiguous_rgb_preserves_pixels_and_coordinates(qtbot, frame, layout):
    from doudizhu_lab.ui.canvas import FrameCanvas

    pixels = np.arange(5 * 9 * 3, dtype=np.uint8).reshape(5, 9, 3)
    rgb = {"padded": pixels[:, :7], "rotated": pixels.transpose(1, 0, 2), "negative": pixels[:, ::-1]}[layout]
    canvas = FrameCanvas()
    qtbot.addWidget(canvas)
    canvas.set_frame(replace(frame, rgb=rgb))
    expected = rgb.copy()
    rgb[:] = 0
    assert canvas._image.pixelColor(0, 0).getRgb()[:3] == tuple(expected[0, 0])


@pytest.mark.gui
def test_background_model_load_failure_retry_and_deduplication(qtbot, tmp_path, monkeypatch):
    release = Event()
    calls = []

    class LoadingAgent(LocalAgent):
        def __init__(self, *args):
            pass

        def start(self):
            calls.append(1)
            release.wait(5)
            if len(calls) == 1:
                raise ValueError("本地权重缺失")

    monkeypatch.setattr("doudizhu_lab.runtime.bot.Agent", LoadingAgent)
    window = MainWindow(Store(tmp_path), auto_refresh=False, auto_preview=False)
    qtbot.addWidget(window)
    window.show()
    qtbot.waitUntil(lambda: window.bot is not None)
    window.load_bot()
    assert window.bot.model_status().state == "loading" and len(calls) == 1
    assert not window.action_button.isEnabled() and window.input_gate.owner is None
    release.set()
    qtbot.waitUntil(lambda: window.bot.model_status().state == "failed")
    qtbot.waitUntil(lambda: window.retry_button.isVisible())
    assert "本地权重缺失" in window.notice_label.text()
    qtbot.wait(150)
    assert len(calls) == 1
    window.retry_button.click()
    qtbot.waitUntil(lambda: window.bot.ready.is_set())
    window.render_dashboard()
    assert window.bot.model_status().state == "ready" and len(calls) == 2
    assert window.input_gate.owner is None


@pytest.mark.gui
def test_close_while_model_loading_never_restarts(qtbot, tmp_path, monkeypatch):
    from time import sleep

    starts = []

    class SlowAgent(LocalAgent):
        def __init__(self, *args):
            pass

        def start(self):
            starts.append(1)
            sleep(0.1)

    monkeypatch.setattr("doudizhu_lab.runtime.bot.Agent", SlowAgent)
    window = MainWindow(Store(tmp_path), auto_refresh=False, auto_preview=False)
    qtbot.addWidget(window)
    window.show()
    qtbot.waitUntil(lambda: window.bot is not None and bool(starts))
    window.close()
    assert not window.bot.thread.is_alive() and len(starts) == 1
    assert window.input_gate.owner is None


def test_zero_elapsed_clock_close_remains_safe(qtbot, tmp_path, monkeypatch):
    import json

    monkeypatch.setattr("doudizhu_lab.ui.window.monotonic", lambda: 123.0)
    window = MainWindow(Store(tmp_path), auto_refresh=False, auto_load_model=False, auto_preview=False)
    qtbot.addWidget(window)
    assert window.close()
    metrics = json.loads((tmp_path / "界面运行统计.json").read_text(encoding="utf-8"))
    assert metrics["elapsed_seconds"] == 0 and metrics["average_fps"] == 0


def test_progress_and_stop_intent_do_not_wait_for_device_lock(tmp_path):
    """真实流程锁被设备线程持有时，GUI 仍能读取缓存并立即阻止新的授权。"""
    from threading import Thread
    from time import monotonic

    from doudizhu_lab.runtime.bot import InputGate
    from doudizhu_lab.runtime.continuous import ContinuousRuntime

    gate = InputGate()
    bot = BotRuntime(tmp_path, gate, agent=LocalAgent(), threaded=False)
    runtime = ContinuousRuntime(
        tmp_path, gate, bot, SimpleNamespace(profile=Store(tmp_path).profile), threaded=False
    )
    runtime.status = "running"
    runtime.store = Mock()
    runtime.progress()
    locked, release = Event(), Event()

    def hold_device_operation():
        with runtime.lock:
            locked.set()
            release.wait(5)

    thread = Thread(target=hold_device_operation)
    thread.start()
    assert locked.wait(1)
    try:
        started = monotonic()
        assert runtime.progress().status == "running"
        gate.suspend()
        runtime.request_stop("用户停止测试")
        gate.grant("flow")
        assert monotonic() - started < 0.2
        assert gate.owner is None and not runtime.active
    finally:
        release.set()
        thread.join(2)
    runtime.stop()
    assert runtime.status == "stopped"
    assert runtime.store.add.call_args.kwargs["reason"] == "用户停止测试"
    gate.allow()
    assert gate.owner is None and not gate.blocked
