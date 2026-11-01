"""有界证据写盘及独立真值存储，验证故障不会污染原观察或把草稿发布成真值。"""

import hashlib
import json
import threading
from dataclasses import replace
from types import SimpleNamespace

import pytest

from doudizhu_lab.domain.tracker import GameState, apply_event
from doudizhu_lab.storage.config import Profile
from doudizhu_lab.storage.evidence import EvidenceWriter
from doudizhu_lab.storage.session import SessionStore
from doudizhu_lab.storage.truth import TruthStore, media_manifest, truth_events, validate_truth


def labelled_game():
    hand = ["3"] * 4 + ["4"] * 4 + ["5"] * 4 + ["6"] * 4 + ["7"] * 4
    game = {
        "game_id": "manual",
        "start_us": 0,
        "end_us": 1_900_000,
        "start_frame": "start",
        "end_frame": "end",
        "initial_hand": hand,
        "bottom": ["3", "3", "3"],
        "landlord": "self",
        "seat_order": ["self", "right", "left"],
        "reviewed": True,
    }
    document = {"games": [game], "events": []}
    start = truth_events(document)[0]
    state = apply_event(GameState(), start)
    n = 0
    for position, card in enumerate(hand):
        actions = [("self", "play", [card])]
        if position < len(hand) - 1:
            actions += [("right", "pass", []), ("left", "pass", [])]
        for seat, kind, cards in actions:
            n += 1
            action = {
                "event_id": f"m{n}",
                "game_id": "manual",
                "pts_us": n * 30_000,
                "frame_id": f"frame-{n}",
                "kind": kind,
                "seat": seat,
                "cards": cards,
                "reviewed": True,
            }
            state = apply_event(state, action)
            action["checkpoint"] = {"hand": state.hand, "remaining": state.remaining}
            document["events"].append(action)
    return document


def fake_session(tmp_path):
    session = SessionStore.create(tmp_path, Profile(), [], "offline")
    video = session.folder / "video"
    video.mkdir()
    (video / "segment.mkv").write_bytes(b"test-video-placeholder")
    (video / "index.json").write_text(
        json.dumps(
            [{"file": "segment.mkv", "global_start_us": 0, "global_end_us": 2_000_000, "closed": True}]
        )
    )
    return session


def test_bounded_evidence_does_not_block_or_drop_save_status(tmp_path, frame, monkeypatch):
    entered, release = threading.Event(), threading.Event()

    def slow_write(path, rgb):
        entered.set()
        assert release.wait(3)
        path.write_bytes(b"png")

    monkeypatch.setattr("doudizhu_lab.storage.evidence.write_rgb", slow_write)
    writer = EvidenceWriter(tmp_path / "evidence", capacity=1)
    writer.start()
    assert writer.submit(replace(frame, frame_id="a", media_pts_us=1), ["e1"])
    assert entered.wait(2)
    assert writer.submit(replace(frame, frame_id="b", media_pts_us=2), ["e2"])
    assert not writer.submit(replace(frame, frame_id="c", media_pts_us=3), ["e3"])
    assert writer.entries["c"]["status"] == "deferred"
    release.set()
    assert writer.stop(rebuild=False)
    result = writer.statistics()
    assert result["maximum_queue"] == 1 and result["saved"] == 2 and result["deferred"] == 1
    assert json.loads((writer.folder / "index.json").read_text())["c"]["event_ids"] == ["e3"]


def test_evidence_disk_failure_is_explicit(tmp_path, frame, monkeypatch):
    monkeypatch.setattr("doudizhu_lab.storage.evidence.shutil.disk_usage", lambda p: SimpleNamespace(free=1))
    errors = []
    writer = EvidenceWriter(tmp_path / "evidence", on_error=lambda *v: errors.append(v))
    writer.start()
    writer.submit(replace(frame, frame_id="disk-failure"), ["event"])
    assert writer.stop(rebuild=False)
    assert errors and writer.statistics()["failed"] == 1
    assert not (writer.folder / "disk-failure.png").exists()


def test_truth_state_checks_each_action_and_detects_wrong_remaining():
    doc = labelled_game()
    assert validate_truth(doc, require_review=True)["valid"]
    doc["events"][3]["checkpoint"]["remaining"] = {"self": 20, "left": 17, "right": 17}
    result = validate_truth(doc, require_review=True)
    assert not result["valid"] and any("剩牌数" in p for p in result["problems"])


def test_independent_store_never_reads_predictions_and_keeps_history(tmp_path, monkeypatch):
    session = fake_session(tmp_path)
    session.add_observation({"frame_id": "raw", "pts_us": 100, "hand": ["3"]})
    original = hashlib.sha256(json.dumps(session.rows("observations"), sort_keys=True).encode()).hexdigest()
    folder = session.folder
    session.close()
    with monkeypatch.context() as patch:
        patch.setattr(SessionStore, "rows", lambda *a: pytest.fail("独立标注不得读取自动结果"))
        labels = TruthStore(folder)
    assert labels.document["games"] == [] and labels.document["events"] == []
    labels.select_range(0, 2_000_000)
    document = labelled_game()
    labels.document.update(document)
    labels.save(publish=True)
    published = json.loads(labels.path.read_text(encoding="utf-8"))
    assert published["independent_review"]
    assert published["source"]["video_manifest"] == media_manifest(folder)
    changed = document["events"][0] | {"reviewed": False}
    labels.put_event(changed)
    assert not labels.document["independent_review"] and not labels.document["games"][0]["reviewed"]
    assert list((labels.path.parent / "history").glob("*.json"))
    with pytest.raises(ValueError, match="不能发布"):
        labels.save(publish=True)
    reopened = SessionStore(folder)
    assert (
        hashlib.sha256(json.dumps(reopened.rows("observations"), sort_keys=True).encode()).hexdigest()
        == original
    )
    reopened.close()


def test_unclosed_segment_is_gap_even_without_shutdown_handler(tmp_path):
    session = fake_session(tmp_path)
    index = session.folder / "video/index.json"
    value = json.loads(index.read_text())
    value[0]["closed"] = False
    index.write_text(json.dumps(value))
    assert session.gap_intervals()[0]["code"] == "unclosed_segment"
    assert not session.export().read_text(encoding="utf-8").endswith('"complete_trusted": true\n}')
    session.close()


@pytest.mark.gui
def test_annotation_window_starts_blank_and_cannot_tap(qtbot, tmp_path, monkeypatch, frame):
    from PySide6.QtCore import Qt

    from doudizhu_lab.storage.replay import VideoReplay
    from doudizhu_lab.ui.truth import TruthDialog

    session = fake_session(tmp_path)
    session.add_event({"event_id": "machine", "game_id": "g", "pts_us": 0, "kind": "game_start"})
    folder = session.folder
    session.close()
    replay = replace(frame, serial="replay", transport="replay", frame_id="media", media_pts_us=0)
    monkeypatch.setattr(VideoReplay, "at", lambda *args, **kwargs: replay)
    dialog = TruthDialog(folder)
    qtbot.addWidget(dialog)
    dialog.show()
    qtbot.waitUntil(lambda: dialog.frame is not None)
    assert dialog.events.topLevelItemCount() == 0 and dialog.games.count() == 0
    assert dialog.hand.text() == "" and not dialog.canvas.control_enabled
    taps = []
    dialog.canvas.tapped.connect(lambda *v: taps.append(v))
    qtbot.mouseClick(dialog.canvas, Qt.MouseButton.LeftButton, pos=dialog.canvas.rect().center())
    assert taps == []
    dialog.close()
