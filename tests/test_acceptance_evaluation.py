"""正式验收的匹配、缺口归属和通过门槛，避免人工纠错或片段演示被当成整局通过。"""

import json
import sys

import pytest

from doudizhu_lab.cli.check import main
from doudizhu_lab.evaluation.session import _score, evaluate_session
from doudizhu_lab.storage.config import Profile, Store
from doudizhu_lab.storage.session import (
    SessionStore,
    freeze_release,
    reserve_acceptance,
    snapshot_template_dirs,
)


def event(ident, time, seat="self", cards=None, game="g", kind="play"):
    return {
        "event_id": ident,
        "pts_us": time,
        "game_id": game,
        "kind": kind,
        "seat": seat,
        "cards": cards or ["3"],
        "confirmed": True,
    }


def test_ordered_matching_does_not_swap_actions_or_reuse_truth():
    a, b = event("a", 1_000_000, cards=["3"]), event("b", 1_100_000, cards=["4"])
    predicted = [a | {"cards": ["4"]}, b | {"cards": ["3"]}]
    metrics, _, _ = _score(predicted, {"events": [a, b]}, 2_000_000)
    assert metrics["play"]["correct"] == 1
    assert metrics["play"]["out_of_order"] == 1
    assert metrics["play"]["missed"] == 1
    metrics, _, _ = _score([a, a | {"event_id": "repeat", "pts_us": 1_300_000}], {"events": [a]}, 2_000_000)
    assert metrics["play"]["correct"] == 1 and metrics["play"]["duplicates"] == 1


def test_failed_acceptance_batch_cannot_be_promoted_after_development(tmp_path):
    session = SessionStore.create(tmp_path, Profile(), [], "phone")
    session.set_meta("acceptance", {"batch_id": "test"})
    with pytest.raises(ValueError, match="停止"):
        session.reject_acceptance("短提示漏记")
    session.set_meta("status", "stopped")
    outcome = session.reject_acceptance("短提示漏记")
    assert session.reject_acceptance("重复操作") == outcome
    folder = session.folder
    session.close()
    report = evaluate_session(folder)
    assert not report["target_passed"]
    assert any("已失败并转为开发" in reason for reason in report["unmet"])


def test_games_matched_one_to_one_before_matching_events():
    predicted = [
        {"event_id": "s1", "game_id": "g1", "kind": "game_start", "pts_us": 0},
        event("p1", 5_000_000, game="g1"),
        {"event_id": "e1", "game_id": "g1", "kind": "game_end", "pts_us": 10_000_000},
        {"event_id": "s2", "game_id": "g2", "kind": "game_start", "pts_us": 11_000_000},
        event("p2", 15_000_000, game="g2"),
    ]
    truth = {
        "games": [
            {"game_id": "t1", "start_us": 0, "end_us": 10_000_000},
            {"game_id": "t2", "start_us": 11_000_000, "end_us": 20_000_000},
        ],
        "events": [event("tp", 15_000_000, game="t1")],
    }
    metrics, boundary, details = _score(predicted, truth, 20_000_000)
    assert metrics["play"]["correct"] == 0
    assert len(details["game_pairs"]) == 2 and not boundary["passed"]


def test_default_one_second_tolerance_and_card_multisets():
    truth = {"events": [event("t", 1_000_000, cards=["3", "3", "4"])]}
    metrics, _, _ = _score([event("p", 2_000_000, cards=["4", "3", "3"])], truth, 3_000_000)
    assert metrics["play"]["correct"] == 1
    metrics, _, _ = _score([event("p", 2_000_001, cards=["3", "3", "4"])], truth, 3_000_000)
    assert metrics["play"]["missed"] == 1


def test_abandoned_deal_is_separate_but_cannot_hide_played_game():
    abandoned = [
        {"game_id": "aborted", "kind": "game_start", "pts_us": 0},
        {"game_id": "aborted", "kind": "game_end", "pts_us": 5_000_000, "data": {"abandoned_deal": True}},
    ]
    played = [
        {"game_id": "g", "kind": "game_start", "pts_us": 5_000_000},
        event("p", 7_000_000),
        {"game_id": "g", "kind": "game_end", "pts_us": 10_000_000},
    ]
    truth = {
        "games": [{"game_id": "g", "start_us": 5_000_000, "end_us": 10_000_000}],
        "events": [event("t", 7_000_000)],
    }
    metrics, boundary, _ = _score(abandoned + played, truth, 10_000_000)
    assert metrics["play"]["correct"] == 1 and boundary["passed"]
    # 即使人为附上流局标签，已记录动作的局仍计入边界和误记检查。
    metrics, boundary, _ = _score(
        [abandoned[0], event("hidden", 3_000_000, game="aborted"), abandoned[1]] + played,
        truth,
        10_000_000,
    )
    assert metrics["play"]["false_positive"] == 1 and not boundary["passed"]


def test_pending_event_counts_as_miss_and_offline_cannot_pass(tmp_path):
    session = SessionStore.create(tmp_path, Profile(), [], "offline")
    session.add_event(event("e", 1_000_000) | {"confirmed": False})
    labels = tmp_path / "truth.json"
    labels.write_text(json.dumps({"events": [event("e", 1_000_000)]}))
    result = evaluate_session(session.folder, labels)
    assert result["automatic"]["play"]["missed"] == 1
    assert result["offline"] is not None and result["live"] is None and not result["target_passed"]
    session.close()


def test_gap_interval_survives_boundary_edits_without_poisoning_later_games(tmp_path):
    session = SessionStore.create(tmp_path, Profile(), [], "offline")
    for ident, start, end in (("a", 0, 100), ("b", 200, 300)):
        session.add_event(
            {
                "event_id": ident + "s",
                "game_id": ident,
                "kind": "game_start",
                "pts_us": start,
                "data": {"capture_incomplete": False},
            }
        )
        session.add_event({"event_id": ident + "e", "game_id": ident, "kind": "game_end", "pts_us": end})
    session.add_issue({"issue_id": "gap", "code": "transport", "game_id": "b", "start_us": 20, "end_us": 40})
    states = session.game_states()
    assert states["a"]["capture_incomplete"]
    assert not states["b"]["capture_incomplete"]
    # 初始手牌未知可使状态不完整；缺口原因必须只附着于物理相交的第一局。
    assert [g["start_us"] for g in session.gap_intervals()] == [20]
    session.revise("boundary", "as", {"pts_us": 10})
    assert session.gap_intervals()[0]["end_us"] == 40
    session.add_issue({"issue_id": "open", "code": "analysis", "scope": "analysis", "start_us": 210})
    session.add_issue({"code": "gap_closed", "gap_id": "open", "end_us": 230})
    assert len(session.gap_intervals()) == 1
    assert session.gap_intervals(include_analysis=True)[1]["end_us"] == 230
    session.close()


def test_freeze_fingerprint_rejects_modified_config(tmp_path):
    store = Store(tmp_path)
    freeze_release(store)
    session = SessionStore.create(store.root, store.profile, snapshot_template_dirs(store), "phone")
    assert reserve_acceptance(store, session)["fingerprint"]
    session.close()
    store.profile.threshold += 0.01
    changed = SessionStore.create(store.root, store.profile, snapshot_template_dirs(store), "phone")
    with pytest.raises(ValueError, match="冻结"):
        reserve_acceptance(store, changed)
    changed.close()


def test_require_pass_saves_report_before_nonzero_exit(tmp_path, monkeypatch):
    session = SessionStore.create(tmp_path, Profile(), [], "offline")
    target = tmp_path / "报告" / "评测.json"
    folder = session.folder
    session.close()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ddz-check",
            "--data-dir",
            str(tmp_path),
            "evaluate-session",
            str(folder),
            "--output",
            str(target),
            "--require-pass",
        ],
    )
    with pytest.raises(SystemExit) as outcome:
        main()
    assert outcome.value.code == 2 and not json.loads(target.read_text(encoding="utf-8"))["target_passed"]
