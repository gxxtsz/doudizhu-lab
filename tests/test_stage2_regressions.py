"""真实验收暴露的边界、快过牌、拒识和独立评测问题的回归。"""

import json

import numpy as np
import pytest

from doudizhu_lab.domain.models import REGION_NAMES
from doudizhu_lab.domain.tracker import GameState, GameTracker, apply_event
from doudizhu_lab.evaluation.cards import evaluate
from doudizhu_lab.storage.config import Profile, Store, write_rgb
from doudizhu_lab.storage.replay import evaluate_session
from doudizhu_lab.storage.session import SessionStore
from doudizhu_lab.vision.cards import Recognizer
from doudizhu_lab.vision.table import TableObservation


def test_closed_game_rejects_late_action_and_cannot_remain_complete():
    state = GameState(game_id="g", ended=True, complete=True, hand=["3"])
    result = apply_event(state, {"game_id": "g", "kind": "play", "seat": "self", "cards": ["3"]})
    assert result.pending and not result.complete and result.hand == state.hand


def test_invalid_setup_keeps_last_trusted_fields():
    state = GameState(game_id="g", landlord="self", hand=["3"], initial_hand=["3"])
    changed = apply_event(
        state, {"game_id": "g", "kind": "setup", "data": {"landlord": "unknown", "initial_hand": ["4"]}}
    )
    assert changed.pending and changed.landlord == "self" and changed.hand == ["3"]
    assert state.pending is False


def test_zero_badge_alone_does_not_certify_complete_history():
    state = GameState(
        game_id="g",
        initial_hand=["3"] * 4 + ["4"] * 4 + ["5"] * 4 + ["6"] * 4 + ["7"],
        landlord="left",
        bottom=["A", "2", "SJ"],
        hand=["3"],
        remaining={"self": 1, "left": 0, "right": 17},
        seat_order=["self", "right", "left"],
    )
    assert not apply_event(state, {"game_id": "g", "kind": "game_end"}).complete


def test_two_fast_passes_follow_verified_order_without_clock_frames():
    tracker = GameTracker(["self", "right", "left"])
    tracker.state = GameState(
        game_id="g",
        phase="playing",
        current_actor="right",
        last_play=["2"],
        last_player="self",
        seat_order=["self", "right", "left"],
    )
    tracker._phase = "playing"
    events = []
    for pts in (1000, 201000, 401000, 601000):
        obs = TableObservation(
            str(pts),
            pts,
            phase="playing",
            passes=["left", "right"],
            pass_labels={"left": "过", "right": "要不起"},
        )
        events.extend(tracker.observe(obs).events)
    assert [(e.kind, e.seat) for e in events] == [("pass", "right"), ("pass", "left")]
    assert tracker.state.current_actor == "self" and not tracker.state.pending


def test_dealing_before_hand_stability_retains_new_game_boundary():
    tracker = GameTracker()
    tracker.state = GameState(game_id="old", phase="playing", turn=4)
    tracker._phase = "playing"
    for pts in (1000, 201000):
        tracker.observe(TableObservation(str(pts), pts, phase="dealing"))
    for pts in (401000, 601000):
        tracker.observe(TableObservation(str(pts), pts, phase="bidding", hand=["3", "4"]))
    assert tracker.state.game_id != "old" and tracker.state.initial_hand == ["3", "4"]


def test_first_play_on_phase_transition_keeps_original_counts():
    tracker = GameTracker(["self", "right", "left"])
    for pts in (1000, 201000):
        tracker.observe(
            TableObservation(
                str(pts),
                pts,
                phase="doubling",
                landlord="left",
                hand=["3", "4"],
                remaining={"self": 2, "left": 20, "right": 17},
            )
        )
    events = []
    for pts in (401000, 601000, 801000):
        events.extend(
            tracker.observe(
                TableObservation(
                    str(pts),
                    pts,
                    phase="playing",
                    landlord="left",
                    hand=["3", "4"],
                    remaining={"self": 2, "left": 19, "right": 17},
                    plays={"left": ["5"]},
                )
            ).events
        )
    assert [(e.kind, e.seat, e.cards) for e in events if e.kind == "play"] == [("play", "left", ["5"])]
    assert tracker.state.remaining["left"] == 19 and not tracker.state.pending


def test_validation_null_regions_and_whole_game_source_isolation(tmp_path):
    store = Store(tmp_path / "store")
    rgb = np.full((120, 240, 3), 100, np.uint8)
    write_rgb(tmp_path / "sample.png", rgb)
    manifest = tmp_path / "labels.json"
    data = {"version": 2, "samples": [{"image": "sample.png", "regions": dict.fromkeys(REGION_NAMES)}]}
    manifest.write_text(json.dumps(data))
    recognizer = Recognizer(store.template_dir, include_bundled=False)
    report = evaluate(manifest, recognizer, store.profile)
    assert not report["target_passed"] and report["by_region"]["hand"]["labelled_samples"] == 0
    from types import SimpleNamespace

    recognizer.templates = [SimpleNamespace(source={"session": "s", "game": "different-id", "pts_us": 150})]
    data["validation_spans"] = [{"session": "s", "start_us": 100, "end_us": 200}]
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="整局|牌局时间"):
        evaluate(manifest, recognizer, store.profile)


def test_boundary_metrics_detect_wrong_times_even_with_equal_counts(tmp_path):
    session = SessionStore.create(tmp_path, Profile(), [], "offline")
    for kind, pts in (("game_start", 0), ("game_end", 10_000_000)):
        session.add_event({"event_id": kind, "game_id": "g", "kind": kind, "pts_us": pts})
    labels = tmp_path / "truth.json"
    labels.write_text(json.dumps({"events": [], "games": [{"start_us": 5_000_000, "end_us": 15_000_000}]}))
    report = evaluate_session(session.folder, labels)
    assert not report["boundary_check"]["automatic"]["passed"]
    assert report["boundary_check"]["automatic"]["game_start"]["missed"] == 1
    session.add_issue({"code": "disk_backlog", "message": "磁盘堵塞"})
    report = evaluate_session(session.folder)
    assert report["gaps"][0]["code"] == "disk_backlog" and not report["target_passed"]
    session.close()


def test_session_keeps_executable_engine_snapshot_and_detects_modification(tmp_path):
    session = SessionStore.create(tmp_path, Profile(), [], "offline")
    assert (session.folder / "engine" / "uv.lock").is_file()
    session.verify_snapshot()
    source = session.folder / "engine" / "src" / "doudizhu_lab" / "domain" / "tracker.py"
    source.write_text("# 被修改的内容\n", encoding="utf-8")
    with pytest.raises(ValueError, match="快照校验失败"):
        session.verify_snapshot()
    session.close()
