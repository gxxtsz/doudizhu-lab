"""动作证据窗口、短暂漏检、轮次和初始化的针对性回归；不依赖设备或 Qt。"""

from dataclasses import asdict

from doudizhu_lab.domain.tracker import GameTracker, replay_events
from doudizhu_lab.vision.table import TableObservation


def obs(ms, hand=None, **fields):
    hand = ["3", "3", "4", "5", "6"] if hand is None else hand
    value = TableObservation(
        str(ms),
        ms * 1000,
        phase="playing",
        landlord="self",
        actor="self",
        hand=hand,
        bottom=["3", "4", "5"],
        remaining={"self": len(hand), "left": 17, "right": 17},
    )
    for key, content in fields.items():
        setattr(value, key, content)
    return value


def stable(tracker, time, **values):
    return [tracker.observe(obs(time, **values)), tracker.observe(obs(time + 120, **values))]


def initialized():
    tracker = GameTracker(["self", "left", "right"])
    stable(tracker, 0)
    stable(tracker, 1000, hand=["3", "4", "5", "6"], plays={"self": ["3"]})
    return tracker


def test_delayed_card_evidence_waits_and_preserves_first_timestamp():
    tracker = GameTracker(["self", "left", "right"])
    stable(tracker, 0)
    updates = stable(tracker, 1000, hand=["3", "4", "5", "6"])
    assert updates[-1].pending_candidates and not tracker.state.pending
    updates += stable(tracker, 2400, hand=["3", "4", "5", "6"], plays={"self": ["3"]})
    events = [e for u in updates for e in u.events if e.kind == "play"]
    assert len(events) == 1 and events[0].confirmed
    assert events[0].data["first_evidence_us"] == 1_000_000
    assert events[0].data["confirmed_at_us"] == 2_520_000
    assert tracker.state.hand == ["3", "4", "5", "6"]


def test_short_pass_miss_bridges_only_within_300ms():
    tracker = initialized()
    tracker.observe(obs(2000, hand=["3", "4", "5", "6"], passes=["left"]))
    tracker.observe(obs(2070, hand=["3", "4", "5", "6"]))
    update = tracker.observe(obs(2200, hand=["3", "4", "5", "6"], passes=["left"]))
    assert [(e.kind, e.seat, e.confirmed) for e in update.events] == [("pass", "left", True)]
    assert not tracker.state.pending


def test_99ms_is_not_rounded_into_100ms():
    tracker = initialized()
    tracker.observe(obs(2000, hand=["3", "4", "5", "6"], passes=["left"]))
    update = tracker.observe(obs(2099, hand=["3", "4", "5", "6"], passes=["left"]))
    assert not update.events and not update.pending_candidates[0]["ready"]
    update = tracker.observe(obs(2200, hand=["3", "4", "5", "6"], passes=["left"]))
    assert update.events[0].confirmed


def test_long_miss_restarts_stability_without_duplicate_candidates():
    tracker = initialized()
    tracker.observe(obs(2000, hand=["3", "4", "5", "6"], passes=["left"]))
    tracker.observe(obs(2400, hand=["3", "4", "5", "6"]))
    update = tracker.observe(obs(2500, hand=["3", "4", "5", "6"], passes=["left"]))
    assert len(update.pending_candidates) == 1 and not update.events
    update = tracker.observe(obs(2620, hand=["3", "4", "5", "6"], passes=["left"]))
    assert len(update.events) == 1 and update.events[0].confirmed
    assert not tracker.observe(obs(6000, hand=["3", "4", "5", "6"])).state.pending


def test_simultaneous_evidence_commits_in_verified_seat_order():
    tracker = initialized()
    updates = stable(tracker, 2000, hand=["3", "4", "5", "6"], passes=["right", "left"])
    events = [e for u in updates for e in u.events if e.kind == "pass"]
    assert [e.seat for e in events] == ["left", "right"]
    assert all(e.confirmed for e in events) and tracker.state.current_actor == "self"


def test_consumed_pass_flicker_cannot_create_duplicate():
    tracker = initialized()
    stable(tracker, 2000, hand=["3", "4", "5", "6"], passes=["left"])
    stable(tracker, 2600, hand=["3", "4", "5", "6"])
    updates = stable(tracker, 3200, hand=["3", "4", "5", "6"], passes=["left"])
    assert not [e for u in updates for e in u.events]
    assert not tracker.state.pending


def test_bidding_text_is_never_a_pass():
    tracker = initialized()
    updates = stable(tracker, 2000, hand=["3", "4", "5", "6"], passes=["left"], pass_labels={"left": "不抢"})
    assert not [e for u in updates for e in u.events] and not tracker._candidates


def test_bomb_occlusion_recovery_does_not_forge_hand_loss():
    tracker = initialized()
    stable(tracker, 1500, hand=["3"])
    assert tracker._candidates and tracker.state.hand == ["3", "4", "5", "6"]
    updates = stable(tracker, 2100, hand=["3", "4", "5", "6"])
    assert not tracker._candidates and not tracker.state.pending
    assert any(i["code"] == "occlusion_recovered" for u in updates for i in u.issues)
    stable(tracker, 5500, hand=["3", "4", "5", "6"])
    assert not tracker.state.pending and tracker.state.turn == 1


def test_gap_invalidates_pending_and_prevents_cross_gap_stability():
    tracker = initialized()
    tracker.observe(obs(2000, hand=["3", "4", "5", "6"], passes=["left"]))
    tracker.mark_gap(obs(2050), "测试断线", capture_lost=True)
    update = tracker.observe(obs(2200, hand=["3", "4", "5", "6"], passes=["left"]))
    assert update.state.pending and update.state.capture_incomplete
    assert not any(e.confirmed and e.kind == "pass" for e in update.events)


def test_occlusion_recovery_restores_pass_round_before_next_real_pass():
    tracker = initialized()
    hand = ["3", "4", "5", "6"]
    counts = {"self": 4, "left": 17, "right": 17}
    for time, seat, rank in ((2000, "left", "7"), (2500, "right", "8")):
        counts[seat] -= 1
        stable(tracker, time, hand=hand, remaining=dict(counts), plays={seat: [rank]})
    stable(tracker, 3000, hand=hand, remaining=dict(counts), passes=["self"])
    for time, seat, rank in ((3500, "left", "9"), (4000, "right", "10")):
        counts[seat] -= 1
        stable(tracker, time, hand=hand, remaining=dict(counts), plays={seat: [rank]})
    stable(tracker, 4500, hand=["3"], remaining=counts | {"self": 1})
    stable(tracker, 5000, hand=hand, remaining=dict(counts))
    updates = stable(tracker, 5500, hand=hand, remaining=dict(counts), passes=["self"])
    assert [(e.kind, e.seat, e.confirmed) for u in updates for e in u.events] == [("pass", "self", True)]
    assert not tracker.state.pending and tracker.state.turn == 7


def test_landlord_bottom_added_once_and_first_play_not_initialization():
    deal = ["3"] * 3 + ["4"] * 3 + ["5"] * 3 + ["6"] * 3 + ["7"] * 3 + ["8", "9"]
    bottom = ["A", "K", "2"]
    tracker = GameTracker(["self", "left", "right"])
    events = []
    for time, fields in (
        (0, {"phase": "dealing", "hand": deal, "bottom": None, "landlord": None}),
        (500, {"phase": "bidding", "hand": deal, "bottom": None}),
        (1000, {"phase": "doubling", "hand": deal + bottom, "bottom": bottom}),
        (1600, {"hand": deal + bottom, "bottom": bottom}),
        (2200, {"hand": deal + ["A", "K"], "bottom": bottom, "plays": {"self": ["2"]}}),
    ):
        for update in stable(tracker, time, **fields):
            events.extend(asdict(e) for e in update.events)
    assert len(tracker.state.initial_hand) == 20 and len(tracker.state.hand) == 19
    assert tracker.state.played["self"] == ["2"] and not tracker.state.pending
    assert replay_events(events)[tracker.state.game_id] == asdict(tracker.state)


def test_last_card_requires_explicit_zero_observation_and_card_evidence():
    tracker = GameTracker(["self", "left", "right"])
    stable(tracker, 0, hand=["3"])
    stable(tracker, 1000, hand=["3"], plays={"self": []})
    assert tracker.state.remaining["self"] == 1
    updates = stable(tracker, 2000, hand=[], plays={"self": ["3"]})
    assert any(e.kind == "play" and e.confirmed for u in updates for e in u.events)
    assert tracker.state.remaining["self"] == 0


def test_redeal_uses_growing_deal_evidence_even_without_phase_transition():
    old = ["3"] * 4 + ["4"] * 4 + ["5"] * 4 + ["6"] * 4 + ["7"]
    new = ["8"] * 4 + ["9"] * 4 + ["10"] * 4 + ["J"] * 4 + ["Q"]
    tracker = GameTracker(["self", "right", "left"])
    base = {"phase": "dealing", "bottom": None, "landlord": None, "actor": None}
    stable(tracker, 0, hand=old, **base)
    first = tracker.state.game_id
    # 背面底牌让阶段一直是 dealing；不能依赖 bidding -> dealing 的切换。
    for ms, length in ((1000, 2), (1200, 5), (1400, 9), (1600, 14)):
        tracker.observe(obs(ms, hand=new[:length], **base))
    updates = stable(tracker, 1800, hand=new, **base)
    boundaries = [e for u in updates for e in u.events if e.kind in ("game_start", "game_end")]
    assert tracker.state.game_id != first
    assert [e.kind for e in boundaries] == ["game_end", "game_start"]
    assert all(e.pts_us == 1_000_000 for e in boundaries)
    assert boundaries[0].data["abandoned_deal"] and boundaries[1].frame_id == "1000"
    assert tracker.state.initial_hand == sorted(new, key=lambda x: ["8", "9", "10", "J", "Q"].index(x))


def test_reordering_occlusion_or_single_wrong_hand_cannot_split_deal():
    old = ["3"] * 4 + ["4"] * 4 + ["5"] * 4 + ["6"] * 4 + ["7"]
    tracker = GameTracker()
    base = {"phase": "dealing", "bottom": None, "landlord": None}
    stable(tracker, 0, hand=old, **base)
    first = tracker.state.game_id
    stable(tracker, 300, hand=list(reversed(old)), **base)
    for ms, length in ((1000, 2), (1200, 5), (1400, 9), (1600, 14)):
        tracker.observe(obs(ms, hand=old[:length], **base))
    stable(tracker, 1800, hand=old, **base)
    # 完整手牌只变了一张、没有新发牌序列时不允许假造新局。
    stable(tracker, 5000, hand=old[:-1] + ["8"], **base)
    assert tracker.state.game_id == first


def test_gap_prevents_redeal_candidate_from_crossing_missing_video():
    old = ["3"] * 4 + ["4"] * 4 + ["5"] * 4 + ["6"] * 4 + ["7"]
    new = ["8"] * 4 + ["9"] * 4 + ["10"] * 4 + ["J"] * 4 + ["Q"]
    tracker = GameTracker()
    base = {"phase": "dealing", "bottom": None, "landlord": None}
    stable(tracker, 0, hand=old, **base)
    first = tracker.state.game_id
    for ms, length in ((1000, 2), (1200, 5), (1400, 9)):
        tracker.observe(obs(ms, hand=new[:length], **base))
    tracker.mark_gap(obs(1500, **base), "录像中断", capture_lost=True)
    stable(tracker, 1800, hand=new, **base)
    assert tracker.state.game_id == first and tracker.state.capture_incomplete
