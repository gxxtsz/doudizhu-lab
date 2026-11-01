"""第二阶段的 Qt 隔离、录制配置只读和离线编辑验证，测试不会连接手机。"""

from dataclasses import replace
from types import SimpleNamespace

import pytest
from PySide6.QtCore import Qt

from doudizhu_lab.domain.models import Device
from doudizhu_lab.runtime.recording import game_is_foreground
from doudizhu_lab.storage.config import Store
from doudizhu_lab.ui.replay import EventEditor, SessionDialog, parse_cards
from doudizhu_lab.ui.window import MainWindow


def test_hyperos_multiple_display_focus_and_sleep():
    output = "mCurrentFocus=null\nmCurrentFocus=Window{x com.tencent.tmgp.sgame/.SGameActivity}"
    assert game_is_foreground(output, "mWakefulness=Awake")
    assert not game_is_foreground(output, "mWakefulness=Asleep")
    assert not game_is_foreground(
        "mCurrentFocus=Window{x com.android.settings/.Settings}", "mWakefulness=Awake"
    )


@pytest.mark.gui
def test_recording_freezes_calibration_and_cannot_start_screenshot_tasks(qtbot, tmp_path, frame):
    window = MainWindow(Store(tmp_path), auto_refresh=False)
    window.timer.stop()
    qtbot.addWidget(window)
    window.devices.addItem("测试手机", Device(frame.serial, "device"))
    video_frame = replace(frame, transport="video", frame_id="video-1")
    pipeline = SimpleNamespace(
        allowed=True, _started=frame.started, write_queue=SimpleNamespace(qsize=lambda: 0)
    )
    pipeline.poll = lambda: {
        "frame": video_frame,
        "observation": None,
        "recognition": None,
        "state": None,
        "messages": [],
        "events": [],
        "stopped": False,
        "allowed": True,
        "metrics": {"decoded_frames": 1, "analyzed_frames": 0, "encoded_bytes": 0},
    }
    window.recording = pipeline
    window.running = True
    window._lock_recording(True)
    try:
        window.mode.setCurrentIndex(2)
        assert window.mode.currentData() == "view" and window.running
        assert not window.import_button.isEnabled() and not window.reload_button.isEnabled()
        window._tick()
        assert not window.busy and window.frame.frame_id == "video-1"
    finally:
        window.recording = None
        window._lock_recording(False)


@pytest.mark.gui
def test_replay_canvas_has_no_mobile_control(qtbot, tmp_path, frame):
    dialog = SessionDialog(Store(tmp_path))
    qtbot.addWidget(dialog)
    dialog.show()
    dialog.canvas.set_frame(replace(frame, serial="replay", transport="replay"))
    hits = []
    dialog.canvas.tapped.connect(lambda *x: hits.append(x))
    for index in range(dialog.mode.count()):
        dialog.mode.setCurrentIndex(index)
        qtbot.mouseClick(dialog.canvas, Qt.MouseButton.LeftButton, pos=dialog.canvas.rect().center())
        assert not dialog.canvas.control_enabled
    assert hits == []


@pytest.mark.gui
def test_event_editor_validates_labels_and_preserves_repeated_cards(qtbot):
    event = {
        "event_id": "e",
        "game_id": "g",
        "pts_us": 1234567,
        "kind": "play",
        "seat": "self",
        "cards": ["3", "3"],
    }
    dialog = EventEditor(event)
    qtbot.addWidget(dialog)
    assert dialog.value()["cards"] == ["3", "3"] and dialog.value()["pts_us"] == 1234567
    dialog.cards.setText("小王 大王 10 10")
    assert dialog.value()["cards"] == ["SJ", "BJ", "10", "10"]
    with pytest.raises(ValueError):
        parse_cards("猜测")


@pytest.mark.gui
def test_same_size_video_session_change_cancels_old_tap(qtbot, tmp_path, frame):
    window = MainWindow(Store(tmp_path), auto_refresh=False)
    window.timer.stop()
    qtbot.addWidget(window)
    window.show()
    qtbot.waitUntil(window.isVisible)
    data = {
        "codec_epoch": 1,
        "frame": replace(frame, transport="video", frame_id="one"),
        "observation": None,
        "recognition": None,
        "state": None,
        "messages": [],
        "events": [],
        "stopped": False,
        "allowed": True,
        "metrics": {"decoded_frames": 2, "analyzed_frames": 0, "encoded_bytes": 0},
    }
    window.recording = SimpleNamespace(
        poll=lambda: data, _started=frame.started, write_queue=SimpleNamespace(qsize=lambda: 0)
    )
    try:
        window._poll_video()
        data["frame"] = replace(data["frame"], frame_id="two")
        window._poll_video()
        assert window._direction_ready == 2
        old_cancellation = window._cancel
        data.update(codec_epoch=2, frame=None)
        window._poll_video()
        assert old_cancellation.is_set() and window._direction_ready == 0
        assert not window.canvas.control_enabled
    finally:
        window.recording = None
        window.close()
