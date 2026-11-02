"""SQLite 原始观察不可变、修订撤销及关闭重开后的确定性验证。"""

import json
from dataclasses import asdict

import pytest

from doudizhu_lab.domain.models import NormalizedRect
from doudizhu_lab.storage.config import Profile, Store
from doudizhu_lab.storage.replay import evaluate_session
from doudizhu_lab.storage.session import SessionStore


def test_revision_reopen_and_undo_preserve_original(tmp_path):
    session = SessionStore.create(tmp_path, Profile(), [], "test")
    original = {
        "event_id": "e1",
        "game_id": "g1",
        "pts_us": 100,
        "kind": "game_start",
        "frame_id": "f1",
        "data": {"initial_hand": ["3"]},
    }
    session.add_event(original)
    session.add_observation({"frame_id": "f1", "pts_us": 100, "hand": ["3"]})
    revision = session.revise("replace", "e1", {"data": {"initial_hand": ["4"]}})
    folder = session.folder
    session.close()
    session = SessionStore(folder)
    assert session.effective_events()[0]["data"]["initial_hand"] == ["4"]
    session.revise("undo", str(revision), {})
    assert session.effective_events() == [original]
    assert session.rows("observations")[0]["hand"] == ["3"]
    a = session.export().read_bytes()
    b = session.export().read_bytes()
    assert a == b
    with pytest.raises(ValueError):
        session.rows("events; DROP TABLE events")
    session.close()


def test_no_self_evaluation_without_manual_truth(tmp_path):
    session = SessionStore.create(tmp_path, Profile(), [], "test")
    folder = session.folder
    session.close()
    result = evaluate_session(folder)
    assert not result["labelled"] and result["automatic"]["play"]["precision"] is None


def test_legacy_profile_backup_and_region_preservation(tmp_path):
    profile = asdict(Profile())
    profile["regions"].pop("self_play")
    profile["glyph_heights"].pop("self_play")
    profile["regions"]["hand"] = asdict(NormalizedRect(0.1, 0.7, 0.8, 0.29))
    raw = json.dumps({"version": 1, "profile": profile})
    (tmp_path / "settings.json").write_text(raw)
    store = Store(tmp_path)
    assert store.profile.regions["hand"].y == 0.7
    assert "self_play" in store.profile.regions
    store.save()
    assert (tmp_path / "settings.v1.bak.json").read_text() == raw
    assert json.loads((tmp_path / "settings.json").read_text(encoding="utf-8"))["version"] == 6


def test_boundary_revision_reassigns_events_without_rewriting_raw(tmp_path):
    session = SessionStore.create(tmp_path, Profile(), [], "test")
    for event in [
        {"event_id": "a", "game_id": "g1", "pts_us": 0, "kind": "game_start"},
        {"event_id": "p", "game_id": "g1", "pts_us": 150, "kind": "play", "seat": "self", "cards": ["3"]},
        {"event_id": "b", "game_id": "g2", "pts_us": 200, "kind": "game_start"},
    ]:
        session.add_event(event)
    revision = session.revise("boundary", "b", {"pts_us": 100})
    assert next(e for e in session.effective_events() if e["event_id"] == "p")["game_id"] == "g2"
    assert session.rows("events")[1]["game_id"] == "g1"
    session.revise("undo", str(revision), {})
    assert next(e for e in session.effective_events() if e["event_id"] == "p")["game_id"] == "g1"
    session.close()
