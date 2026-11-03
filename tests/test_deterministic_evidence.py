"""确定性证明的合成正反例；原始观察和机器人提交账本相互独立。"""

from collections import Counter
from copy import deepcopy

import pytest

from doudizhu_lab.agents.douzero import DECK, TO_RANK
from doudizhu_lab.domain.inference import deductions, make_proof, verify_proof
from doudizhu_lab.domain.tracker import GameState, apply_event


def state():
    hand = ["3"] * 4 + ["4"] * 4 + ["5"] * 4 + ["6"] * 4 + ["7"]
    return GameState(
        game_id="g",
        phase="playing",
        landlord="right",
        initial_hand=hand,
        hand=hand,
        bottom=["J", "Q", "K"],
        remaining={"self": 17, "right": 20, "left": 17},
        current_actor="self",
        seat_order=["self", "right", "left"],
    )


def pair(before, *, hand=None, actor="right", counts=None, phase="playing", plays=None, markers=None):
    hand = before.hand if hand is None else hand
    return [
        dict(
            frame_id=f"f{i}",
            pts_us=1_000_000 + i * 120_000,
            phase=phase,
            landlord=before.landlord,
            hand=hand,
            actor=actor,
            remaining=counts or before.remaining | {"self": len(hand)},
            plays=plays or dict.fromkeys(("self", "right", "left")),
            marker_evidence=markers or {},
            passes=[],
            frame_context={"serial": "device", "stream_id": "stream", "width": 600, "height": 100},
        )
        for i in range(2)
    ]


def event_for(before, claim, values, *, baseline=()):
    event = {
        "event_id": "e",
        "game_id": before.game_id,
        "pts_us": values[0]["pts_us"],
        **{k: v for k, v in claim.items() if k != "after_hand"},
        "source": "operational_inferred",
    }
    event["data"] = dict(
        event.get("data", {}), inference=make_proof(before, claim, values, deal_values=baseline)
    )
    return event


def test_exact_hand_difference_confirms_missing_play_without_intent():
    before = state()
    values = pair(before, hand=before.hand[1:])
    untouched = deepcopy(values)
    claim = deductions(before, values)[0]
    assert claim["cards"] == ["3"] and claim["after_hand"] == before.hand[1:]
    assert verify_proof(event_for(before, claim, values), before, {o["frame_id"]: o for o in values})
    assert values == untouched and before.turn == 0


@pytest.mark.parametrize(
    "fault",
    [
        "duplicate",
        "stream",
        "short",
        "missing_hand",
        "added_card",
        "contradiction",
        "preparation",
        "gap",
        "illegal",
    ],
)
def test_incomplete_or_conflicting_anchors_never_prove_play(fault):
    before = state()
    values = pair(before, hand=before.hand[1:])
    if fault == "duplicate":
        values[1]["frame_id"] = values[0]["frame_id"]
    elif fault == "stream":
        values[1]["frame_context"]["stream_id"] = "other"
    elif fault == "short":
        values[1]["pts_us"] = values[0]["pts_us"] + 90_000
    elif fault == "gap":
        before.capture_incomplete = True
    else:
        for obs in values:
            if fault == "missing_hand":
                obs["hand"] = None
            elif fault == "added_card":
                obs["hand"] = before.hand[2:] + ["BJ"]
            elif fault == "contradiction":
                obs["plays"]["self"] = ["4"]
            elif fault == "preparation":
                obs["phase"] = "bidding"
            elif fault == "illegal":
                obs["hand"] = before.hand[1:8] + before.hand[9:]
                obs["remaining"]["self"] = len(obs["hand"])
    assert not any(c["kind"] == "play" for c in deductions(before, values))


@pytest.mark.parametrize("fault", ["hash", "conclusion", "game", "timestamp", "anchor"])
def test_independent_verifier_rejects_tampering(fault):
    before = state()
    values = pair(before, hand=before.hand[1:])
    event = event_for(before, deductions(before, values)[0], values)
    originals = {o["frame_id"]: o for o in values}
    proof = event["data"]["inference"]
    if fault == "hash":
        proof["before_hash"] = "invented"
    elif fault == "conclusion":
        proof["conclusion"]["cards"] = ["BJ"]
    elif fault == "game":
        event["game_id"] = "next"
    elif fault == "timestamp":
        proof["from_us"] -= 1
    else:
        originals["f1"]["remaining"]["right"] -= 1
    assert not verify_proof(event, before, originals)


def test_verifier_binds_anchors_to_the_observed_game_start():
    before = state()
    values = pair(before, hand=before.hand[1:])
    for obs in values:
        obs["frame_context"]["round_id"] = "another-round"
    event = event_for(before, deductions(before, values)[0], values)
    expected = values[0]["frame_context"] | {"round_id": "trusted-round"}
    assert not verify_proof(event, before, {o["frame_id"]: o for o in values}, expected_context=expected)


def test_two_missing_passes_have_separately_verifiable_prefixes():
    before = state()
    before.current_actor, before.last_play, before.last_player = "right", ["8"], "self"
    before.turn = 1
    values = pair(before, actor="self")
    raw = {o["frame_id"]: o for o in values}
    claims = deductions(before, values)
    assert [c["seat"] for c in claims] == ["right", "left"]
    for claim in claims:
        event = event_for(before, claim, values)
        assert verify_proof(event, before, raw)
        before = apply_event(before, event)
    assert not before.pending and before.current_actor == "self" and not before.last_play


def test_final_j_requires_positive_zero_and_known_remaining_hand():
    before = state()
    before.hand, before.remaining["self"] = ["J"], 1
    values = pair(before, hand=[], actor=None, markers={"winner_self": {"accepted": True}})
    claim = deductions(before, values)[0]
    assert claim["cards"] == ["J"]
    assert verify_proof(event_for(before, claim, values), before, {o["frame_id"]: o for o in values})
    for obs in values:
        obs["marker_evidence"] = {}
    assert not deductions(before, values)


def test_bottom_and_landlord_are_proved_from_seventeen_to_twenty():
    before = state()
    before.phase, before.landlord, before.bottom = "bidding", None, None
    before.remaining = {s: 17 for s in ("self", "right", "left")}
    bottom = ["BJ", "8", "9"]
    values = pair(before, hand=before.hand + bottom, actor=None, phase="doubling")
    claim = deductions(before, values)[0]
    assert Counter(claim["data"]["bottom"]) == Counter(bottom)
    assert claim["data"]["landlord"] == "self"
    assert verify_proof(event_for(before, claim, values), before, {o["frame_id"]: o for o in values})
    baseline = pair(before, actor=None, phase="bidding")
    for o in baseline:
        o["pts_us"] -= 500_000
        o["frame_id"] = "baseline-" + o["frame_id"]
    before.initial_hand = before.hand = values[0]["hand"]
    claim = deductions(before, values, deal_values=baseline)[0]
    event = event_for(before, claim, values, baseline=baseline)
    assert verify_proof(event, before, {o["frame_id"]: o for o in values + baseline})
    assert not verify_proof(event, before, {o["frame_id"]: o for o in values})


@pytest.mark.parametrize("fault", ["bottom", "landlord"])
def test_setup_rejects_independently_visible_conflicting_facts(fault):
    before = state()
    before.phase, before.landlord, before.bottom = "bidding", None, None
    before.remaining = dict.fromkeys(("self", "left", "right"), 17)
    values = pair(before, hand=before.hand + ["BJ", "8", "9"], actor=None, phase="doubling")
    for obs in values:
        if fault == "bottom":
            obs["bottom"] = ["J", "Q", "K"]
        else:
            obs["landlord"] = "right"
    assert not any("bottom" in c.get("data", {}) for c in deductions(before, values))


def test_high_frame_rate_uses_current_frame_and_an_older_independent_anchor():
    from tests.test_balanced import inference_observer, snapshot

    observer, before = inference_observer("strict")
    for ms in range(1000, 1700, 33):
        raw = snapshot(ms, before, actor="self", plays={"self": ["3"], "right": [], "left": []})
        result = observer.transform(raw)
        assert not raw.observation["passes"]
    inferred = [e for e in result.history if e["source"] == "operational_inferred"]
    assert len(inferred) == 2
    assert all(
        e["data"]["inference"]["to_us"] - e["data"]["inference"]["from_us"] >= 100_000 for e in inferred
    )


def test_final_hand_is_verified_before_direct_settlement_flushes_history():
    from dataclasses import replace

    from doudizhu_lab.domain.bot_types import encode
    from tests.test_balanced import inference_observer, snapshot

    observer, _ = inference_observer("strict")
    before = state()
    before.hand, before.remaining["self"] = ["J"], 1
    before.initial_hand = ["3"] * 4 + ["4"] * 4 + ["5"] * 4 + ["6"] * 4 + ["J"]
    before.played["self"] = before.initial_hand[:-1]
    before.last_play, before.last_player, before.passes, before.current_actor = [], None, 0, "self"
    observer.tracker.state = deepcopy(before)
    observer.tracker._counts, observer.tracker._hand = dict(before.remaining), ["J"]
    originals = {}
    for ms in (1000, 1120, 1240, 1360):
        raw = snapshot(ms, before, hand=[], actor=None, plays=dict.fromkeys(("self", "right", "left")))
        obs = raw.observation
        obs["phase"] = "settlement"
        obs["marker_evidence"] = {"winner_self": {"accepted": True}}
        raw = replace(raw, observation_json=encode(obs))
        originals[raw.frame.frame_id] = raw.observation
        result = observer.transform(raw)
        assert raw.observation["phase"] == "settlement" and raw.observation["plays"]["self"] is None
    assert result.state["ended"] and result.state["complete"] and not result.state["pending"]
    event = next(e for e in result.history if e["source"] == "operational_inferred")
    assert event["cards"] == ["J"] and verify_proof(event, before, originals)


def test_complete_hand_difference_replaces_an_unproved_animation_candidate():
    from tests.test_balanced import inference_observer, snapshot

    observer, _ = inference_observer("strict")
    before = state()
    observer.tracker.state = deepcopy(before)
    observer.tracker._counts, observer.tracker._hand = dict(before.remaining), list(before.hand)
    for ms in (1000, 1120):
        observer.transform(
            snapshot(ms, before, hand=["7"], actor=None, plays=dict.fromkeys(("self", "right", "left")))
        )
    assert observer.tracker._candidates and not observer.tracker.state.pending
    originals = {}
    for ms in (1240, 1360, 1480, 1600):
        raw = snapshot(
            ms, before, hand=before.hand[1:], actor="right", plays=dict.fromkeys(("self", "right", "left"))
        )
        originals[raw.frame.frame_id] = raw.observation
        result = observer.transform(raw)
    assert not result.state["pending"] and result.state["hand"] == before.hand[1:]
    event = next(e for e in result.history if e["kind"] == "play")
    assert event["cards"] == ["3"] and event["source"] == "operational_inferred"
    assert verify_proof(event, before, originals)


def test_submitted_candidate_can_wait_read_only_past_three_seconds_but_not_eight():
    from doudizhu_lab.domain.table import TableObservation
    from doudizhu_lab.domain.tracker import GameTracker

    tracker = GameTracker(["self", "right", "left"])
    before = state()
    tracker.state = before
    tracker._phase = "playing"
    tracker._role_initialized = tracker._active_started = True
    tracker._counts, tracker._hand = dict(before.remaining), list(before.hand)
    pending = TableObservation("first", 1_000_000, phase="playing")
    tracker._candidate(pending, "play", "self", before_count=17, after_count=16)
    tracker.observe(TableObservation("later", 4_500_000), action_deadline_us=9_000_000)
    assert not tracker.state.pending and tracker._candidates
    tracker.observe(TableObservation("limit", 9_100_000), action_deadline_us=9_000_000)
    assert tracker.state.pending


@pytest.mark.parametrize("fault", [None, "duplicate", "stream", "same_round", "hand", "expired", "tampered"])
def test_independent_continue_proof_requires_a_stable_new_round(fault):
    from doudizhu_lab.storage.runs import verify_flow_transition

    values = pair(state(), actor=None, phase="bidding")
    for obs in values:
        obs["frame_context"]["round_id"] = "new-raw"
    sent = {
        "kind": "continue",
        "origin_scene": "settlement",
        "origin_game_id": "old",
        "origin_round_id": "old-raw",
        "pts_us": 500_000,
        "frame_context": {k: v for k, v in values[0]["frame_context"].items() if k != "round_id"},
    }
    confirmed = {
        "pts_us": values[-1]["pts_us"],
        "proof": {
            "version": 2,
            "frames": [o["frame_id"] for o in values],
            "game_id": "new",
            "origin_game_id": "old",
            "raw_round_id": "new-raw",
        },
    }
    if fault == "duplicate":
        confirmed["proof"]["frames"] = [values[0]["frame_id"]] * 2
    elif fault == "stream":
        values[1]["frame_context"]["stream_id"] = "other"
    elif fault == "same_round":
        sent["origin_round_id"] = "new-raw"
    elif fault == "hand":
        values[1]["hand"] = values[1]["hand"][1:] + ["BJ"]
    elif fault == "expired":
        sent["pts_us"] -= 8_000_000
    elif fault == "tampered":
        confirmed["proof"]["raw_round_id"] = "invented"
    assert verify_flow_transition(sent, confirmed, {o["frame_id"]: o for o in values}) is (fault is None)


@pytest.mark.parametrize("available,expected", [(["3", "3"], ["3"]), (["3", "5"], None)])
def test_opponent_uses_only_unique_public_pool(available, expected):
    before = state()
    before.hand = ["4"]
    before.remaining = {"self": 1, "right": 1, "left": 1}
    before.current_actor = "right"
    used = list(
        (
            Counter({TO_RANK[n]: n_cards for n, n_cards in DECK.items()})
            - Counter(before.hand)
            - Counter(available)
        ).elements()
    )
    before.played = {"self": used[:16], "right": used[16:35], "left": used[35:]}
    before.initial_hand = before.played["self"] + before.hand
    values = pair(
        before,
        counts=before.remaining | {"right": 0},
        actor=None,
        markers={"winner_right": {"accepted": True}},
    )
    claims = deductions(before, values)
    assert (claims[0]["cards"] if claims else None) == expected
    if claims:
        assert verify_proof(event_for(before, claims[0], values), before, {o["frame_id"]: o for o in values})


def test_opponent_search_budget_does_not_claim_partial_uniqueness(monkeypatch):
    import doudizhu_lab.domain.inference as module

    before = state()
    before.current_actor = "right"
    calls = []
    monkeypatch.setattr(module, "_legal", lambda *args: calls.append(1) and False)
    values = pair(before, actor="left", counts=before.remaining | {"right": 14})
    assert not deductions(before, values) and len(calls) <= 4096


@pytest.mark.parametrize("policy", ["strict", "infer", "balanced"])
def test_every_policy_records_verified_inference_and_keeps_raw(policy):
    from tests.test_balanced import inference_observer, snapshot

    observer, before = inference_observer(policy)
    values = {}
    for ms in (1000, 1120, 1240, 1360):
        raw = snapshot(ms, before, actor="self", plays={"self": ["3"], "right": [], "left": []})
        values[raw.frame.frame_id] = raw.observation
        result = observer.transform(raw)
        assert not raw.observation["passes"]
    inferred = [e for e in result.history if e["source"] == "operational_inferred"]
    assert len(inferred) == 2
    cursor = deepcopy(before)
    for event in result.history:
        if event["source"] == "operational_inferred":
            assert verify_proof(event, cursor, values)
        cursor = apply_event(cursor, event)
    assert not cursor.pending
