"""首轮阶段迟到、稳定差分及同帧流程提示回归；只构造公开牌点与合成页面。"""

from dataclasses import replace

import numpy as np
import pytest

from doudizhu_lab.domain.models import Frame, PixelRect
from doudizhu_lab.domain.table import TableObservation
from doudizhu_lab.domain.tracker import GameTracker
from doudizhu_lab.vision.flow import FlowRecognizer

HAND = ["3", "4", "4", "5", "6", "6", "7", "9", "10", "J", "Q", "Q", "Q", "K", "A", "A", "SJ"]
STRAIGHT = ["3", "4", "5", "6", "7"]


def sample(ms, *, phase="unknown", landlord="right", hand=None, right=20, left=17, plays=None):
    return TableObservation(
        frame_id=f"synthetic-{ms}",
        pts_us=ms * 1000,
        phase=phase,
        landlord=landlord,
        hand=list(HAND if hand is None else hand),
        remaining={"self": len(HAND if hand is None else hand), "left": left, "right": right},
        plays=plays or {},
    )


def prepared():
    tracker = GameTracker(["self", "right", "left"])
    for ms in (0, 120):
        tracker.observe(sample(ms, phase="bidding", landlord=None, right=17))
    for ms in (260, 380):
        tracker.observe(sample(ms, phase="doubling"))
    assert not tracker.state.capture_incomplete and tracker.state.phase == "doubling"
    return tracker


@pytest.mark.parametrize("raw_phase", ["unknown", "doubling", "dealing", "bidding"])
def test_first_two_plays_are_confirmed_before_own_buttons_appear(raw_phase):
    tracker = prepared()
    original = sample(1000, phase=raw_phase, right=15, plays={"right": ["3"]})
    tracker.observe(original)
    tracker.observe(sample(1120, phase=raw_phase, right=15, plays={"right": STRAIGHT}))
    first = tracker.observe(sample(1240, phase=raw_phase, right=15, plays={"right": STRAIGHT}))
    assert tracker.state.phase == "playing" and tracker.state.turn == 1
    phase = next(e for e in first.events if e.kind == "phase")
    assert phase.data["basis"] == "stable_opening_play" and len(phase.data["evidence_frames"]) >= 2
    assert original.phase == raw_phase
    tracker.observe(sample(1400, right=15, plays={"right": None}))
    second_cards = ["4", "5", "6", "7", "8"]
    tracker.observe(sample(1600, right=15, left=12, plays={"left": second_cards}))
    second = tracker.observe(sample(1720, right=15, left=12, plays={"left": second_cards}))
    plays = [e for e in first.events + second.events if e.kind == "play"]
    assert [(e.seat, e.cards, e.confirmed) for e in plays] == [
        ("right", STRAIGHT, True),
        ("left", second_cards, True),
    ]
    assert plays[0].data["first_evidence_us"] == 1_000_000
    assert tracker.state.turn == 2 and not tracker.state.pending


@pytest.mark.parametrize(
    "cards,remaining", [(["3", "4", "5", "6"], 15), (["3", "4", "5"], 17), (["8", "8", "8", "8", "5"], 15)]
)
def test_inconsistent_count_or_illegal_shape_does_not_infer_playing(cards, remaining):
    tracker = prepared()
    for ms in (1000, 1120, 1240):
        tracker.observe(sample(ms, right=remaining, plays={"right": cards}))
    assert tracker.state.phase == "doubling" and tracker.state.turn == 0


def test_single_frame_evidence_cannot_change_phase_and_unknown_breaks_stability():
    tracker = prepared()
    tracker.observe(sample(1000, right=15, plays={"right": STRAIGHT}))
    tracker.observe(sample(1060, right=15, plays={"right": None}))
    tracker.observe(sample(1120, right=15, plays={"right": STRAIGHT}))
    assert tracker.state.phase == "doubling"
    tracker.observe(sample(1240, right=15, plays={"right": STRAIGHT}))
    assert tracker.state.phase == "playing" and tracker.state.turn == 1


def test_other_player_cannot_lead_before_known_landlord():
    tracker = prepared()
    for ms in (1000, 1120):
        tracker.observe(sample(ms, left=12, plays={"left": STRAIGHT}))
    assert tracker.state.phase == "doubling" and tracker.state.turn == 0


@pytest.mark.parametrize("matching_hand", [True, False])
def test_self_opening_requires_matching_hand_difference(matching_hand):
    from collections import Counter

    tracker = GameTracker(["self", "right", "left"])
    for ms in (0, 120):
        tracker.observe(sample(ms, phase="bidding", landlord=None, right=17))
    initial = HAND + ["8", "10", "K"]
    for ms in (260, 380):
        tracker.observe(sample(ms, phase="doubling", landlord="self", hand=initial, right=17))
    removed = STRAIGHT if matching_hand else ["3", "4", "5", "6", "9"]
    after = list((Counter(initial) - Counter(removed)).elements())
    for ms in (1000, 1120):
        tracker.observe(sample(ms, landlord="self", hand=after, right=17, plays={"self": STRAIGHT}))
    assert tracker.state.turn == (1 if matching_hand else 0)
    assert tracker.state.phase == ("playing" if matching_hand else "doubling")


def test_incomplete_start_or_existing_gap_cannot_be_repaired_by_phase_hint():
    tracker = prepared()
    tracker.mark_gap(sample(500), "模拟真实断流", capture_lost=True)
    for ms in (1000, 1120):
        tracker.observe(sample(ms, right=15, plays={"right": STRAIGHT}))
    assert tracker.state.pending and tracker.state.capture_incomplete
    assert tracker.state.turn == 0


def test_preparation_animation_does_not_restart_confirmed_playing_game():
    tracker = prepared()
    for ms in (1000, 1120):
        tracker.observe(sample(ms, right=15, plays={"right": STRAIGHT}))
    game = tracker.state.game_id
    for phase, start in (("doubling", 1300), ("dealing", 1700), ("bidding", 2100)):
        for ms in (start, start + 120):
            tracker.observe(sample(ms, phase=phase, right=15))
    assert tracker.state.game_id == game and tracker.state.phase == "playing"
    assert tracker.state.turn == 1 and not tracker.state.pending


@pytest.fixture
def flow(monkeypatch):
    recognizer = FlowRecognizer.__new__(FlowRecognizer)
    monkeypatch.setattr(recognizer, "_text", lambda *args: [])
    monkeypatch.setattr(recognizer, "_words", lambda *args: [])
    monkeypatch.setattr(recognizer, "_colored", lambda *args: True)
    frame = Frame("test", np.zeros((1000, 2000, 3), np.uint8), 0, "same", 0)
    return recognizer, frame


def test_flow_uses_only_same_frame_tracked_phase(flow):
    recognizer, frame = flow
    table = {"frame_id": frame.frame_id, "phase": "unknown"}
    assert recognizer.observe(frame, table, tracked_phase=("old", "playing")).scene == "unknown"
    assert recognizer.observe(frame, table, tracked_phase=(frame.frame_id, "playing")).scene == "playing"
    assert table["phase"] == "unknown"
    assert (
        recognizer.observe(
            replace(frame, frame_id="new"), table, tracked_phase=(frame.frame_id, "playing")
        ).scene
        == "unknown"
    )


def test_tracked_playing_does_not_override_blocking_or_settlement(flow, monkeypatch):
    recognizer, frame = flow
    table = {"frame_id": frame.frame_id, "phase": "unknown"}
    hint = (frame.frame_id, "playing")
    monkeypatch.setattr(
        recognizer, "_words", lambda *args: [("健康系统请您休息一下", 0.99, PixelRect(600, 500, 400, 80))]
    )
    assert recognizer.observe(frame, table, tracked_phase=hint).blocked_reason
    monkeypatch.setattr(
        recognizer,
        "_words",
        lambda *args: [
            ("返回大厅", 0.99, PixelRect(400, 800, 200, 60)),
            ("继续游戏", 0.99, PixelRect(1400, 800, 200, 60)),
        ],
    )
    result = recognizer.observe(frame, table, tracked_phase=hint)
    assert result.scene == "settlement" and {b.kind for b in result.buttons} == {"return_lobby", "continue"}
