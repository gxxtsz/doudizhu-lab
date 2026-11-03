"""分局确认迟到、两套编号不同与完成前独立复核的合成回归。"""

from copy import deepcopy
from dataclasses import asdict

import numpy as np
import pytest

from doudizhu_lab.agents.readiness import assess
from doudizhu_lab.domain.bot_types import ObservationSnapshot
from doudizhu_lab.domain.models import Frame
from doudizhu_lab.domain.round_evidence import resolve_round_binding
from doudizhu_lab.domain.table import TableObservation
from doudizhu_lab.domain.tracker import GameState
from doudizhu_lab.runtime.operational import OperationalObserver
from doudizhu_lab.storage.config import Profile
from tests.test_continuous import run as run
from tests.test_continuous import startup_frame


def boundary():
    context = {"serial": "phone", "stream_id": "video", "width": 600, "height": 100}
    observations = {
        "early": dict(
            frame_id="early", pts_us=1_000_000, phase="dealing", frame_context=context | {"round_id": "old"}
        ),
        "confirmed": dict(
            frame_id="confirmed",
            pts_us=2_000_000,
            phase="bidding",
            frame_context=context | {"round_id": "raw-new"},
        ),
    }
    start = dict(
        event_id="op-start",
        game_id="op-new",
        frame_id="early",
        pts_us=1_000_000,
        kind="game_start",
        confirmed=True,
        data=dict(
            phase="dealing", first_evidence_us=1_000_000, confirmed_at_us=1_120_000, capture_incomplete=False
        ),
    )
    original = deepcopy(start) | {"event_id": "raw-start", "game_id": "raw-new", "frame_id": "confirmed"}
    original["data"]["confirmed_at_us"] = 2_000_000
    return start, [original], observations


def test_binding_uses_confirmed_frame_and_allows_different_game_ids():
    start, raw, observations = boundary()
    untouched = deepcopy((start, raw, observations))
    binding = resolve_round_binding(start, raw, observations)
    assert binding.game_id == "op-new" and binding.raw_game_id == "raw-new"
    assert binding.confirmed_frame_id == "confirmed" and binding.frame_context["round_id"] == "raw-new"
    assert (start, raw, observations) == untouched


@pytest.mark.parametrize(
    "fault",
    [
        "duplicate",
        "missing",
        "stream",
        "device",
        "boundary",
        "capture",
        "phase",
        "wrong_round",
        "ambiguous_frame",
        "forged_frame",
    ],
)
def test_binding_rejects_missing_ambiguous_or_cross_source_evidence(fault):
    start, raw, observations = boundary()
    if fault == "duplicate":
        raw.append(deepcopy(raw[0]) | {"event_id": "another"})
    elif fault == "missing":
        observations.pop("confirmed")
    elif fault in ("stream", "device", "wrong_round"):
        key = {"stream": "stream_id", "device": "serial", "wrong_round": "round_id"}[fault]
        observations["confirmed"]["frame_context"][key] = "other"
    elif fault == "boundary":
        raw[0]["data"]["first_evidence_us"] += 1
    elif fault == "capture":
        raw[0]["data"]["capture_incomplete"] = True
    elif fault == "phase":
        observations["confirmed"]["phase"] = "playing"
    elif fault == "ambiguous_frame":
        observations["second"] = deepcopy(observations["confirmed"]) | {"frame_id": "second"}
    elif fault == "forged_frame":
        raw[0]["data"]["confirmed_frame_id"] = "early"
    assert resolve_round_binding(start, raw, observations) is None


def opening_snapshot(ms, *, raw_start=None, context_round="old", phase="bidding"):
    hand = ["3"] * 4 + ["4"] * 4 + ["5"] * 4 + ["6"] * 4 + ["7"]
    frame = Frame(
        "phone",
        np.zeros((100, 600, 3), np.uint8),
        ms / 1000,
        str(ms),
        0,
        "simulation",
        f"f{ms}",
        "video",
        ms * 1000,
    )
    obs = TableObservation(
        frame.frame_id,
        frame.media_pts_us,
        phase=phase,
        hand=hand,
        remaining={"self": 17, "left": 17, "right": 20},
        plays={"self": [], "left": [], "right": []},
    )
    obs.frame_context = {
        "serial": "phone",
        "stream_id": "video",
        "width": 600,
        "height": 100,
        "round_id": context_round,
    }
    state = GameState(game_id=context_round, phase=phase)
    return ObservationSnapshot.create(frame, None, obs, state, [raw_start] if raw_start else [])


def delayed_binding():
    observer = OperationalObserver(Profile(continuous_policy="strict"))
    first = observer.transform(opening_snapshot(1000))
    assert first.state["round_binding_ready"] is False
    pending = observer.transform(opening_snapshot(1120))
    assert not pending.state["round_binding_ready"] and assess(pending).status == "wait"
    start = deepcopy(observer.opening) | {"game_id": "raw-new", "event_id": "raw-start", "frame_id": "f1360"}
    start["data"]["confirmed_at_us"] = 1_360_000
    start["data"]["confirmed_frame_id"] = "f1360"
    bound = observer.transform(opening_snapshot(1360, raw_start=start, context_round="raw-new"))
    return observer, start, bound


def test_runtime_waits_for_original_boundary_and_discards_old_anchors():
    observer, _, bound = delayed_binding()
    assert bound.state["round_binding_ready"]
    assert observer.round_binding.game_id != observer.round_binding.raw_game_id
    assert all(o["frame_context"]["round_id"] == "raw-new" for o in observer.raw_anchors)
    assert not any(e["source"] == "operational_inferred" for e in bound.history)


def test_bound_runtime_never_rebinds_mid_game_to_another_round():
    observer, start, _ = delayed_binding()
    for ms in (1480, 1600):
        observer.transform(opening_snapshot(ms, raw_start=start, context_round="raw-new", phase="playing"))
    before = asdict(observer.round_binding)
    value = observer.transform(
        opening_snapshot(1720, raw_start=start, context_round="another", phase="playing")
    )
    assert asdict(observer.round_binding) == before
    assert value.state["pending"] and assess(value).status == "halt"


def test_unbound_runtime_cannot_bind_after_play_has_started():
    observer = OperationalObserver(Profile(continuous_policy="strict"))
    for ms, phase in (
        (1000, "bidding"),
        (1120, "bidding"),
        (1240, "playing"),
        (1360, "playing"),
        (1480, "playing"),
    ):
        value = observer.transform(opening_snapshot(ms, phase=phase))
    assert value.state["round_binding_error"] and assess(value).status == "halt"


def test_explicit_confirmation_frame_is_checked_against_original_timestamp():
    start, raw, observations = boundary()
    raw[0]["data"]["confirmed_frame_id"] = "confirmed"
    assert resolve_round_binding(start, raw, observations)
    raw[0]["data"]["confirmed_at_us"] += 100_000
    assert resolve_round_binding(start, raw, observations) is None


@pytest.mark.parametrize("fault", ["proof", "binding", "missing_result", "duplicate_submit"])
def test_completion_rejects_bad_evidence_before_counting_and_pins_media(run, monkeypatch, fault):
    """在生产完成入口注入证据破坏；不是把评测返回值替换成一个预设失败。"""
    run.runtime.unattended = True
    original = run.pipeline.session.game_evidence
    pinned, cleaned = [], []
    monkeypatch.setattr(run.pipeline, "pin_game", lambda *args: pinned.append(args), raising=False)
    monkeypatch.setattr(run.pipeline, "complete_game", lambda *args: cleaned.append(args), raising=False)

    def corrupted(*args):
        data = original(*args)
        if fault == "proof":
            move = next(e for e in data["events"] if e["kind"] == "play")
            move["source"] = "operational_inferred"
            move.setdefault("data", {})["inference"] = {"version": 2, "before_hash": "forged"}
        elif fault == "binding":
            start = next(e for e in data["events"] if e["kind"] == "game_start")
            start["data"]["first_evidence_us"] += 1
        else:
            result = next(
                a for a in data["actions"] if a["stage"] == "命令已返回" and a.get("kind") in ("play", "pass")
            )
            if fault == "missing_result":
                data["actions"].remove(result)
            else:
                prepared = next(
                    a
                    for a in data["actions"]
                    if a["action_id"] == result["action_id"]
                    and a["stage"] == "准备输入"
                    and a.get("kind") in ("play", "pass")
                )
                data["actions"].append(deepcopy(prepared))
        return data

    monkeypatch.setattr(run.pipeline.session, "game_evidence", corrupted)
    for _ in range(3000):
        run.tick()
        if run.runtime.quarantined:
            break
    assert run.runtime.quarantined and not run.runtime.completed and not cleaned
    assert len(pinned) == 1 and pinned[0][1] == "proof_error"
    assert not any(
        kind == "game_complete" for (kind,) in run.runtime.store.db.execute("SELECT kind FROM journal")
    )
    assert sum(d.duplicate + d.invalid for d in run.pipeline.devices) == 0


def test_stop_during_read_only_audit_prevents_late_completion_or_cleanup(run, monkeypatch):
    original = run.pipeline.session.game_evidence
    cleaned = []
    monkeypatch.setattr(run.pipeline, "complete_game", lambda *args: cleaned.append(args), raising=False)

    def stop_during_read(*args):
        result = original(*args)
        run.runtime.request_stop()
        return result

    monkeypatch.setattr(run.pipeline.session, "game_evidence", stop_during_read)
    for _ in range(3000):
        run.tick()
        if run.runtime.stop_event.is_set():
            break
    before = list(run.pipeline.flow_commands)
    for _ in range(10):
        run.tick()
    assert not run.runtime.completed and not cleaned
    assert run.pipeline.flow_commands == before


@pytest.mark.parametrize("field", ["raw_game_id", "confirmed_frame_id"])
def test_independent_audit_rejects_only_tampered_binding_on_an_otherwise_valid_proof(run, field):
    from doudizhu_lab.storage.runs import audit_game_evidence

    for _ in range(3000):
        if run.pipeline.device:
            run.pipeline.device.hide_passes = True
        run.tick()
        if run.runtime.completed:
            break
    gid = run.runtime.completed[0]
    start = next(
        e
        for e in run.pipeline.session.rows("operational_events")
        if e["game_id"] == gid and e["kind"] == "game_start"
    )
    data = run.pipeline.session.game_evidence(gid, start["data"]["first_evidence_us"], run.pipeline.pts)
    valid = audit_game_evidence({"game_id": gid}, **data, policy="balanced", source="simulation_bot")
    assert valid["passed"]
    inferred = next(e for e in data["events"] if e["source"] == "operational_inferred")
    inferred["data"]["round_binding"][field] = "forged"
    invalid = audit_game_evidence({"game_id": gid}, **data, policy="balanced", source="simulation_bot")
    assert not invalid["passed"] and invalid["inference_proof_errors"] == [inferred["event_id"]]


@pytest.mark.parametrize("scene", ["bidding", "doubling", "playing"])
def test_unconfirmed_round_blocks_all_game_inputs_and_has_a_read_only_deadline(run, scene):
    from dataclasses import replace

    from doudizhu_lab.domain.bot_types import encode
    from doudizhu_lab.domain.models import PixelRect
    from doudizhu_lab.vision.flow import FlowButton

    run.runtime.startup_phase, run.runtime.lobby_confirmed = "ready", True
    pending = opening_snapshot(1000)
    pending = replace(pending, state_json=encode(pending.state | {"round_binding_ready": False}))
    buttons = (
        FlowButton("decline_bid", "不叫", PixelRect(450, 65, 100, 25), True),
        FlowButton("decline_double", "不加倍", PixelRect(450, 65, 100, 25), True),
    )
    for _ in range(3):
        startup_frame(run, scene, buttons=buttons, snapshot=pending)
    assert run.runtime.active and not run.pipeline.flow_commands and not run.runtime.armed_game
    startup_frame(run, scene, buttons=buttons, snapshot=pending, elapsed=8.1)
    assert run.runtime.status == "paused" and not run.pipeline.flow_commands
