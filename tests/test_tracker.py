"""公开观察的稳定确认、相同牌再次出牌、不出轮转和保守冻结。"""

from dataclasses import asdict

from doudizhu_lab.domain.tracker import GameState, GameTracker, apply_event, replay_events
from doudizhu_lab.vision.table import TableObservation


def observation(ms, hand=None, actor="self", left=17, right=17, plays=None, passes=None, phase="playing"):
    hand = ["3", "3", "4"] if hand is None else hand
    return TableObservation(
        str(ms),
        ms * 1000,
        phase=phase,
        landlord="self",
        actor=actor,
        hand=hand,
        bottom=["3", "4", "5"],
        plays=plays or {},
        remaining={"self": len(hand), "left": left, "right": right},
        passes=passes or [],
    )


def feed(tracker, start, **kwargs):
    a = tracker.observe(observation(start, **kwargs))
    b = tracker.observe(observation(start + 120, **kwargs))
    return a.events + b.events


def test_stability_duplicate_frames_and_multiset_reorder():
    tracker = GameTracker(["self", "left", "right"])
    assert not tracker.observe(observation(0)).events
    assert not tracker.observe(observation(50)).events
    events = tracker.observe(observation(120)).events
    assert any(e.kind == "game_start" for e in events)
    assert not tracker.observe(observation(120)).events
    assert not [e for e in feed(tracker, 400, hand=["4", "3", "3"]) if e.kind == "play"]


def test_same_cards_next_trick_are_new_event_and_pass_persistence_is_not():
    tracker = GameTracker(["self", "left", "right"])
    events = feed(tracker, 0)
    events += feed(tracker, 1000, hand=["3", "4"], actor="left", plays={"self": ["3"]})
    events += feed(tracker, 2000, hand=["3", "4"], actor="left", plays={"self": ["3"]}, passes=["left"])
    events += feed(tracker, 2400, hand=["3", "4"], actor="left", plays={"self": ["3"]}, passes=["left"])
    events += feed(tracker, 3000, hand=["3", "4"], actor="right", plays={"self": ["3"]}, passes=["right"])
    events += feed(tracker, 3500, hand=["3", "4"], actor="self", plays={"self": ["3"]})
    events += feed(tracker, 4000, hand=["4"], actor="left", plays={"self": ["3"]})
    assert [(e.kind, e.seat) for e in events if e.kind in ("play", "pass")] == [
        ("play", "self"),
        ("pass", "left"),
        ("pass", "right"),
        ("play", "self"),
    ]
    assert tracker.state.hand == ["4"] and not tracker.state.pending
    assert replay_events([asdict(e) for e in events])[tracker.state.game_id] == asdict(tracker.state)


def test_empty_regions_do_not_infer_pass_and_missing_play_freezes():
    tracker = GameTracker()
    feed(tracker, 0)
    assert not [e for e in feed(tracker, 500, plays={"left": [], "right": []}) if e.kind == "pass"]
    feed(tracker, 1000, hand=["3", "4"], plays={"self": None})
    update = tracker.observe(observation(2500, hand=["3", "4"], plays={"self": None}))
    assert not update.state.pending and update.pending_candidates
    # 新版保留三秒的证据窗口，不能沿用旧版 1.2 秒的提前冻结行为。
    update = tracker.observe(observation(4500, hand=["3", "4"], plays={"self": None}))
    assert update.state.pending and update.state.hand == ["3", "3", "4"]
    assert any(i["code"] == "missing_play" for i in update.issues)


def test_gap_cannot_be_marked_complete():
    tracker = GameTracker()
    feed(tracker, 0)
    tracker.mark_gap(observation(500), "USB 断线", capture_lost=True)
    feed(tracker, 1000, phase="settlement")
    assert tracker.state.capture_incomplete and tracker.state.ended and not tracker.state.complete


def test_settlement_then_new_deal_is_separate_game():
    tracker = GameTracker()
    feed(tracker, 0)
    first = tracker.state.game_id
    feed(tracker, 1000, phase="settlement")
    feed(tracker, 2000, phase="dealing")
    assert tracker.state.game_id != first and not tracker.state.ended


def test_invalid_manual_initial_hand_and_conservation_are_rejected():
    event = {"game_id": "g", "kind": "game_start", "data": {"initial_hand": ["BOGUS"]}}
    state = apply_event(GameState(), event)
    assert state.pending and state.hand is None
    state = GameState(
        game_id="g", hand=["3"], initial_hand=["3"] * 4, remaining={"self": 1, "left": 17, "right": 17}
    )
    result = apply_event(state, {"game_id": "g", "kind": "play", "seat": "left", "cards": ["3"]})
    assert result.pending and result.remaining == state.remaining and result.played == state.played


def test_unknown_breaks_consecutive_stability():
    tracker = GameTracker()
    tracker.observe(observation(0))
    unknown = observation(60)
    unknown.hand = None
    unknown.remaining["self"] = None
    tracker.observe(unknown)
    assert not tracker.observe(observation(120)).events
    assert tracker.observe(observation(240)).state.game_id


def test_bidding_back_to_hidden_bottom_does_not_split_same_deal():
    tracker = GameTracker()
    feed(tracker, 0, phase="bidding")
    identity = tracker.state.game_id
    feed(tracker, 500, phase="dealing")
    assert tracker.state.game_id == identity


def test_two_passes_return_lead_to_last_valid_player():
    state = GameState(
        game_id="g",
        last_play=["A"],
        last_player="self",
        current_actor="left",
        seat_order=["self", "left", "right"],
    )
    for seat in ("left", "right"):
        state = apply_event(state, {"kind": "pass", "game_id": "g", "seat": seat})
    assert state.current_actor == "self" and state.last_play == [] and state.passes == 0
