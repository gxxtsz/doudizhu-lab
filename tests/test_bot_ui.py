"""复用生产窗口、后台协调和模拟反馈，验证 GUI 生命周期与取消；无手机输入。"""

import gc
from threading import Thread

import pytest
from PySide6.QtCore import Qt

from doudizhu_lab.agents.douzero import TO_RANK, context
from doudizhu_lab.domain.bot_types import Decision
from doudizhu_lab.runtime.bot import BotRuntime
from doudizhu_lab.storage.config import Store
from doudizhu_lab.ui.demo import DemoWindow
from doudizhu_lab.ui.lifetime import ensure_gui_lifetime


class FastAgent:
    """GUI 测试专用合法策略；模型权重与特征在独立真实模型测试中验证。"""

    model_id = "模拟界面测试策略"

    def __init__(self, *args):
        pass

    def start(self):
        pass

    def close(self):
        pass

    def interrupt(self):
        pass

    def decide(self, snapshot):
        data = context(snapshot)
        cards = data["legal_actions"][0]
        s = snapshot.state
        return Decision(
            tuple(TO_RANK[n] for n in cards),
            snapshot.version,
            s["game_id"],
            s["turn"],
            data["player_position"],
            self.model_id,
            1,
            0,
        )


@pytest.mark.gui
def test_native_simulator_step_stop_record_close(qtbot, tmp_path, monkeypatch):
    monkeypatch.setattr("doudizhu_lab.runtime.bot.Agent", FastAgent)
    window = DemoWindow(Store(tmp_path))
    qtbot.addWidget(window)
    window.show()
    assert window.bot_mode.currentData() == "suggest"
    qtbot.mouseClick(window.record_button, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: window.recording is not None, timeout=5000)
    pipe = window.recording
    qtbot.mouseClick(window.bot_load, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: window.bot is not None and window.bot.suggestion is not None, timeout=5000)
    assert pipe.device.taps == 0
    qtbot.mouseClick(window.bot_execute, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: window.bot.executor.stage == "已确认", timeout=5000)
    qtbot.waitUntil(lambda: window.bot.mode == "suggest")
    assert pipe.device.taps > 0 and not pipe.device.invalid
    window.bot_mode.setCurrentIndex(2)
    qtbot.mouseClick(window.bot_stop, Qt.MouseButton.LeftButton)
    before = pipe.device.taps
    qtbot.wait(350)
    assert pipe.device.taps == before and window.bot.mode == "stop"
    qtbot.mouseClick(window.record_button, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: window.recording is None, timeout=5000)
    window.close()
    qtbot.waitUntil(lambda: not window.isVisible())
    assert not window.bot.thread.is_alive() and not pipe.thread.is_alive()
    assert window.pool.activeThreadCount() == 0


@pytest.mark.gui
def test_cancel_calibration_does_not_publish_region(qtbot, tmp_path, frame):
    from copy import deepcopy

    from doudizhu_lab.domain.models import NormalizedRect
    from doudizhu_lab.ui.calibration import BotCalibrationDialog

    store = Store(tmp_path)
    original = deepcopy(store.profile.bot_controls)
    dialog = BotCalibrationDialog(store, frame)
    qtbot.addWidget(dialog)
    dialog.selected(NormalizedRect(0.2, 0.5, 0.3, 0.1))
    dialog.reject()
    assert store.profile.bot_controls == original


@pytest.mark.gui
def test_gui_collection_thread_and_model_load_shutdown(qtbot, tmp_path, monkeypatch):
    import threading
    from time import sleep

    controller = ensure_gui_lifetime()
    assert not gc.isenabled()
    errors = []

    def wrong_thread():
        try:
            controller.collect()
        except RuntimeError as exc:
            errors.append(str(exc))

    thread = Thread(target=wrong_thread)
    thread.start()
    thread.join()
    assert errors

    class LoadingAgent(FastAgent):
        def start(self):
            sleep(0.15)

    monkeypatch.setattr("doudizhu_lab.runtime.bot.Agent", LoadingAgent)
    runtime = BotRuntime(tmp_path, __import__("doudizhu_lab.runtime.bot", fromlist=["InputGate"]).InputGate())
    runtime.start()
    runtime.close()
    assert not runtime.thread.is_alive()
    assert threading.current_thread() is threading.main_thread()


@pytest.mark.gui
def test_close_with_zero_elapsed_clock_does_not_raise_in_qt(qtbot, tmp_path, monkeypatch):
    """原生崩溃的确定性回归：同刻度关闭必须正常返回，不能靠重跑碰运气。"""
    import json

    from doudizhu_lab.ui.window import MainWindow

    monkeypatch.setattr("doudizhu_lab.ui.window.monotonic", lambda: 123.0)
    window = MainWindow(Store(tmp_path), auto_refresh=False)
    qtbot.addWidget(window)
    window.show()
    assert window.close()
    metrics = json.loads((tmp_path / "界面运行统计.json").read_text(encoding="utf8"))
    assert metrics["elapsed_seconds"] == 0 and metrics["average_fps"] == 0
    assert not window.timer.isActive()


def test_settlement_checks_last_submission_before_stopping(tmp_path):
    from unittest.mock import Mock

    from doudizhu_lab.runtime.bot import InputGate
    from doudizhu_lab.simulation.bot import SimulatedDevice

    runtime = BotRuntime(tmp_path, InputGate())
    runtime.mode = "auto"
    executor = Mock(active=True, stage="等待反馈")

    def confirmed(snapshot):
        executor.active = False
        executor.stage = "已确认"

    executor.advance.side_effect = confirmed
    runtime.executor = executor
    snapshot = SimulatedDevice(73100, "landlord").last
    runtime._finish_game(snapshot)
    executor.advance.assert_called_once_with(snapshot)
    executor.cancel.assert_not_called()
    assert runtime.mode == "suggest" and executor.stage == "已确认"
