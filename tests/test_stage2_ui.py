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
    window = MainWindow(Store(tmp_path), auto_refresh=False, auto_load_model=False, auto_preview=False)
    window.timer.stop()
    qtbot.addWidget(window)
    window.available_devices = [Device(frame.serial, "device")]
    window.select_device(frame.serial)
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
    try:
        from doudizhu_lab.ui.settings import SettingsDialog

        settings = SettingsDialog(window)
        qtbot.addWidget(settings)
        assert not settings.connection.isEnabled() and not settings.task.isEnabled()
        window._tick()
        assert not window.busy and window.frame.frame_id == "video-1"
    finally:
        window.recording = None


@pytest.mark.gui
def test_replay_canvas_has_no_mobile_control(qtbot, tmp_path, frame):
    dialog = SessionDialog(Store(tmp_path))
    qtbot.addWidget(dialog)
    dialog.show()
    dialog.canvas.set_frame(replace(frame, serial="replay", transport="replay"))
    hits = []
    dialog.canvas.tapped.connect(lambda *x: hits.append(x))
    qtbot.mouseClick(dialog.canvas, Qt.MouseButton.LeftButton, pos=dialog.canvas.rect().center())
    assert not dialog.canvas.control_enabled and dialog.canvas.mode == "view"
    assert not hasattr(dialog, "mode") and not hasattr(dialog, "target")
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
    window = MainWindow(Store(tmp_path), auto_refresh=False, auto_load_model=False, auto_preview=False)
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
        assert window._video_identity == "two"
        data.update(codec_epoch=2, frame=None)
        window._poll_video()
        assert window._video_identity == "" and window._video_epoch == 2
        assert not window.canvas.control_enabled
    finally:
        window.recording = None
        window.close()


@pytest.mark.gui
def test_replay_real_video_seek_and_correction_reopen(qtbot, tmp_path, monkeypatch):
    """真实短 MKV 验证事件跳转、逐帧和修订重开，不使用手机或旧私人录像。"""
    import shutil

    from PySide6.QtWidgets import QDialog

    from doudizhu_lab.storage.session import SessionStore
    from tests.test_paced_replay import tiny_recording

    store = Store(tmp_path / "数据")
    session = SessionStore.create(store.root, store.profile, [], "test")
    folder = session.folder
    shutil.copytree(tiny_recording(tmp_path / "媒体") / "video", folder / "video")
    original = {
        "event_id": "play-1",
        "game_id": "g1",
        "pts_us": 1_133_000,
        "kind": "play",
        "seat": "self",
        "cards": ["3", "3"],
        "data": {},
    }
    session.add_event(
        {"event_id": "start-1", "game_id": "g1", "pts_us": 1_000_000, "kind": "game_start", "data": {}}
    )
    session.add_event(original)
    session.add_observation({"frame_id": "original", "pts_us": 1_133_000, "hand": ["3", "3"]})
    session.close()
    dialog = SessionDialog(store, initial_folder=folder)
    qtbot.addWidget(dialog)
    dialog.show()
    qtbot.waitUntil(lambda: dialog.frame is not None)
    item = next(
        dialog.events.topLevelItem(i)
        for i in range(dialog.events.topLevelItemCount())
        if dialog.events.topLevelItem(i).data(0, Qt.ItemDataRole.UserRole).get("event_id") == "play-1"
    )
    dialog.events.setCurrentItem(item)
    dialog.jump(item)
    qtbot.waitUntil(lambda: dialog.frame.media_pts_us == 1_133_000)
    dialog.step(1)
    qtbot.waitUntil(lambda: dialog.frame.media_pts_us == 1_200_000)
    dialog.step(-1)
    qtbot.waitUntil(lambda: dialog.frame.media_pts_us == 1_133_000)
    dialog.toggle_play()
    assert dialog.play_timer.isActive()
    dialog.toggle_play()
    assert not dialog.play_timer.isActive()
    assert dialog.edit_button.isEnabled() and not dialog.canvas.control_enabled

    class CorrectedEditor:
        def __init__(self, event, parent):
            self.event = event

        def exec(self):
            return QDialog.DialogCode.Accepted

        def value(self):
            return self.event | {"cards": ["4", "4"]}

    monkeypatch.setattr("doudizhu_lab.ui.replay.EventEditor", CorrectedEditor)
    dialog.edit()
    assert dialog.session.effective_events()[1]["cards"] == ["4", "4"]
    dialog.close()
    reopened = SessionDialog(store, initial_folder=folder)
    qtbot.addWidget(reopened)
    qtbot.waitUntil(lambda: reopened.frame is not None)
    assert reopened.session.rows("events")[1] == original
    assert reopened.session.rows("observations")[0]["hand"] == ["3", "3"]
    assert reopened.session.effective_events()[1]["cards"] == ["4", "4"]
    reopened.events.setCurrentItem(reopened.events.topLevelItem(1))
    reopened.delete()
    assert len(reopened.session.effective_events()) == 1
    reopened.undo()
    assert reopened.session.effective_events()[1]["cards"] == ["4", "4"]
    reopened.insert()
    assert len(reopened.session.effective_events()) == 3
    reopened.close()
