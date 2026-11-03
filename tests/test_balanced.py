"""平衡模式只补唯一不出、只执行明确局部回合；不修改原始事实。"""

from collections import Counter
from dataclasses import asdict, replace

import numpy as np
import pytest

from doudizhu_lab.agents.readiness import ObservationWait, assess
from doudizhu_lab.agents.rules import RuleRound, context, decide, visible_target
from doudizhu_lab.domain.bot_types import ActionButton, HandCard, ObservationSnapshot, encode
from doudizhu_lab.domain.models import Frame, PixelRect
from doudizhu_lab.domain.tracker import GameState
from doudizhu_lab.runtime.executor import ActionExecutor
from doudizhu_lab.runtime.operational import OperationalObserver
from doudizhu_lab.storage.config import Profile
from doudizhu_lab.vision.table import TableObservation


def snapshot(ms, state, *, actor="self", hand=None, plays=None, passes=(), selected=(), lead=False):
    hand = list(state.hand if hand is None else hand)
    obs = TableObservation(
        f"frame-{ms}",
        ms * 1000,
        phase="playing",
        landlord=state.landlord,
        bottom=state.bottom,
        hand=hand,
        actor=actor,
        remaining=state.remaining | {"self": len(hand)},
        plays=plays if plays is not None else {"self": [], "left": [], "right": ["3"]},
        passes=list(passes),
    )
    frame = Frame(
        "simulation",
        np.zeros((100, 600, 3), np.uint8),
        ms / 1000,
        str(ms),
        0,
        "simulation",
        obs.frame_id,
        "balanced-test",
        obs.pts_us,
    )
    value = ObservationSnapshot.create(frame, None, obs, state, [], allowed=True)
    buttons = (ActionButton("play", PixelRect(420, 10, 70, 30), True, "出牌"),)
    if not lead:
        buttons += (ActionButton("pass", PixelRect(510, 10, 70, 30), True, "不出"),)
    layout = tuple(
        HandCard(c, i, PixelRect(i * 25 + 5, 65, 15, 20), i in selected) for i, c in enumerate(hand)
    )
    return replace(value, buttons=buttons, hand=layout, control_reason="")


def local_state():
    return GameState(
        game_id="g",
        phase="playing",
        landlord="right",
        initial_hand=["4", "8", "J"],
        hand=["4", "8", "J"],
        bottom=["3", "5", "6"],
        remaining={"self": 3, "right": 8, "left": 7},
        seat_order=["self", "right", "left"],
        pending=True,
        current_actor="self",
        last_play=["3"],
        turn=5,
    )


def test_rules_context_has_no_fabricated_history_or_opponent_hands():
    original = local_state()
    rules = RuleRound(asdict(original))
    rules.project(snapshot(1000, original))
    value = rules.project(snapshot(1120, original))
    data = context(value)
    assert data["strategy"] == "rules" and not data["history_complete"]
    assert not set(data) & {"other_hand_cards", "all_handcards", "card_play_action_seq"}
    choice = decide(value)
    assert choice.strategy == "rules" and choice.cards == ("4",)
    assert original.pending and value.state["history_incomplete"]


def test_leading_requires_positive_calibration_not_empty_regions():
    state = local_state()
    value = snapshot(1000, state, lead=True, plays={"self": [], "left": [], "right": []})
    with pytest.raises(ObservationWait, match="领出"):
        visible_target(value)
    assert visible_target(value, {"center": [455 / 600, 0.25]})[0] == []
    with pytest.raises(ObservationWait):
        visible_target(replace(value, buttons=()), {"center": [455 / 600, 0.25]})


def test_unknown_or_incomparable_targets_do_not_trigger_rules():
    state = local_state()
    with pytest.raises(ObservationWait, match="遮挡"):
        visible_target(snapshot(1000, state, plays={"self": [], "left": None, "right": ["3"]}))
    with pytest.raises(ObservationWait, match="唯一"):
        visible_target(snapshot(1000, state, plays={"self": [], "left": ["4", "4"], "right": ["3"]}))


def test_rules_action_uses_production_selection_submit_and_two_frame_feedback():
    state, now = local_state(), [1.12]
    rules = RuleRound(asdict(state))
    rules.project(snapshot(1000, state))
    initial = rules.project(snapshot(1120, state))
    decision = decide(initial)
    selected, commands, records = set(), [], []
    actual_hand, actual_actor = list(state.hand), ["self"]
    actual_plays = {"self": [], "left": [], "right": ["3"]}

    def send(kind, x, y, snap, ident):
        commands.append(kind)
        if kind == "select":
            selected.symmetric_difference_update({(x - 5) // 25})
        else:
            rules.submitted(decision, snap.frame.media_pts_us)
            cards = [actual_hand[i] for i in selected]
            actual_hand[:] = list((Counter(actual_hand) - Counter(cards)).elements())
            actual_plays["self"], actual_actor[0] = cards, "right"
            selected.clear()

    executor = ActionExecutor(send, records.append, clock=lambda: now[0])
    executor.start(decision, initial)
    for ms in range(1240, 3000, 120):
        now[0] = ms / 1000
        value = rules.project(
            snapshot(
                ms, state, actor=actual_actor[0], hand=actual_hand, plays=actual_plays, selected=selected
            )
        )
        executor.advance(value)
        if not executor.active:
            break
    assert executor.stage == "已确认" and commands == ["select", "play"]
    assert records[-1]["feedback"]["basis"] == "hand_play_and_actor"
    assert len(records[-1]["feedback"]["frames"]) == 2
    rules.confirmed(decision)
    assert rules.intent is None


def test_rule_uncommanded_hand_change_requires_takeover():
    state = local_state()
    rules = RuleRound(asdict(state))
    rules.project(snapshot(1000, state))
    rules.project(snapshot(1120, state))
    rules.project(snapshot(1240, state, hand=["8", "J"], actor="right"))
    value = rules.project(snapshot(1360, state, hand=["8", "J"], actor="right"))
    assert assess(value).status == "halt"


def inference_observer(policy="infer"):
    profile = Profile(continuous_policy=policy, seat_order=["self", "right", "left"])
    observer = OperationalObserver(profile)
    initial = [rank for rank in ("3", "4", "5", "6", "7") for _ in range(4)]
    hand = initial.copy()
    hand.remove("3")
    state = GameState(
        game_id="inferred",
        phase="playing",
        landlord="self",
        initial_hand=initial,
        hand=hand,
        bottom=["3", "4", "5"],
        remaining={"self": 19, "right": 17, "left": 17},
        seat_order=profile.seat_order,
        current_actor="right",
        last_play=["3"],
        last_player="self",
        played={"self": ["3"], "right": [], "left": []},
        turn=1,
    )
    observer.tracker.state = state
    observer.tracker._role_initialized = observer.tracker._active_started = True
    observer.tracker._phase = "playing"
    return observer, state


def test_unique_missing_passes_are_operational_and_raw_stays_unchanged():
    observer, state = inference_observer()
    for ms in (1000, 1120, 1240, 1360):
        raw = snapshot(ms, state, actor="self", plays={"self": ["3"], "left": [], "right": []})
        result = observer.transform(raw)
        assert raw.observation["passes"] == [] and raw.state["turn"] == 1
    assert result.state["turn"] == 3 and not result.state["last_play"]
    inferred = [e for e in result.history if e["source"] == "operational_inferred"]
    assert len(inferred) == 2 and {e["seat"] for e in inferred} == {"right", "left"}
    assert all(e["data"]["inference"]["basis"] for e in inferred)


def test_count_drop_cannot_be_inferred_as_pass():
    observer, state = inference_observer()
    for ms in (1000, 1120, 1240):
        raw = snapshot(ms, state, actor="left")
        obs = raw.observation
        obs["remaining"]["right"] -= 1
        observer.transform(replace(raw, observation_json=encode(obs)))
    assert not any(e["source"] == "operational_inferred" for e in observer.history)


@pytest.mark.parametrize("cards,expected", [((), True), (("3",), False)])
def test_pending_own_pass_can_use_unchanged_hand_counts_and_unique_actor(cards, expected):
    from doudizhu_lab.domain.bot_types import Decision
    from doudizhu_lab.vision.table import TableObservation

    observer, state = inference_observer("balanced")
    state.current_actor, state.turn, state.last_player = "self", 3, "left"
    decision = Decision(cards, "v", state.game_id, state.turn, "landlord", "model", 0, 0)
    observer.note_submission(decision, 1_000_000)
    for ms in (1500, 1620):
        obs = TableObservation(
            **snapshot(ms, state, actor="right", plays={"self": None, "right": [], "left": ["3"]}).observation
        )
        observer._infer_passes(obs)
    assert ("self" in obs.passes) == expected
    if expected:
        proof = observer.inferred["self"]
        assert proof["own_pass_after_submission"] and proof["submission_pts_us"] == 1_000_000
        assert len(proof["frames"]) == 2 and proof["to_us"] - proof["from_us"] >= 100_000


@pytest.mark.parametrize("cards,expected", [(None, True), (["8"], False)])
def test_unique_counts_allow_unknown_region_but_reject_new_visible_cards(cards, expected):
    observer, state = inference_observer()
    for ms in (1000, 1120, 1240):
        raw = snapshot(ms, state, actor="left", plays={"self": ["3"], "right": cards, "left": []})
        observer.transform(raw)
    assert any(e["source"] == "operational_inferred" for e in observer.history) == expected


def test_unknown_submission_and_capture_gap_cannot_activate_fallback():
    observer, state = inference_observer("balanced")
    state.pending = True
    observer.open_submission = object()
    result = observer.transform(snapshot(1000, state))
    assert observer.rule is None and result.state["pending"]
    observer.open_submission = None
    state.capture_incomplete = True
    result = observer.transform(snapshot(1120, state))
    assert observer.rule is None and result.state["capture_incomplete"]


@pytest.fixture(params=[4000.0, 9000.0])
def balanced_run(tmp_path, request):
    from types import SimpleNamespace

    from doudizhu_lab.agents.douzero import TO_RANK
    from doudizhu_lab.agents.douzero import context as model_context
    from doudizhu_lab.domain.bot_types import Decision
    from doudizhu_lab.runtime.bot import BotRuntime, InputGate
    from doudizhu_lab.runtime.continuous import ContinuousRuntime
    from doudizhu_lab.simulation.continuous import ContinuousSimPipeline
    from doudizhu_lab.storage.config import Store

    class LegalAgent:
        model_id = "仅用于故障测试"

        def start(self):
            pass

        def decide(self, snap):
            data = model_context(snap)
            return Decision(
                tuple(TO_RANK[n] for n in data["legal_actions"][0]),
                snap.version,
                snap.state["game_id"],
                snap.state["turn"],
                data["player_position"],
                self.model_id,
                0,
                0,
            )

    # 两种量级下 0.12 的浮点累加方向相反；模拟调度都必须保留足够的确认帧。
    now = [request.param]
    store = Store(tmp_path)
    store.profile.continuous_policy = "balanced"
    store.profile.continuous_settings["max_games"] = 1
    pipeline = ContinuousSimPipeline(store, clock=lambda: now[0])
    pipeline.start()
    gate = InputGate(clock=lambda: now[0], transport="simulation")
    bot = BotRuntime(tmp_path, gate, clock=lambda: now[0], agent=LegalAgent(), threaded=False)
    runtime = ContinuousRuntime(tmp_path, gate, bot, pipeline, clock=lambda: now[0], threaded=False)
    runtime.start()

    def tick():
        now[0] += 0.12
        pipeline.tick()
        gate.heartbeat()
        bot.pulse()
        runtime.advance(pipeline.flow_snapshot(), pipeline.bot_snapshot())

    yield SimpleNamespace(tick=tick, pipeline=pipeline, bot=bot, runtime=runtime, now=now)
    runtime.close()
    bot.close()
    pipeline.stop()
    pipeline.session.close()


def test_missing_pass_text_recovers_with_wp_and_keeps_original_failure(balanced_run):
    run = balanced_run
    for _ in range(6000):
        if run.pipeline.device:
            run.pipeline.device.hide_passes = True
        run.tick()
        if not run.runtime.active:
            break
    assert run.runtime.status == "completed", run.runtime.reason
    assert not run.pipeline.device.invalid and not run.pipeline.device.duplicate
    assert any(e["source"] == "operational_inferred" for e in run.pipeline.session.rows("operational_events"))
    assert run.pipeline.device.tracker.state.pending
    assert not any(a.get("strategy") == "rules" for a in run.pipeline.session.rows("bot_actions"))
    from doudizhu_lab.storage.runs import evaluate_run

    report = evaluate_run(run.runtime.store.folder)
    assert all(g["passed"] for g in report["game_audits"])
    assert not report["inference_proof_errors"]


def test_unknown_play_can_switch_to_rules_and_finish_without_hidden_hands(balanced_run):
    run = balanced_run
    run.runtime.settings["max_games"] = 2
    injected = False
    for _ in range(6000):
        if run.runtime.armed_game and not injected:
            run.pipeline.fault = "hidden_play"
            injected = True
        run.tick()
        if not run.runtime.active:
            break
    assert run.runtime.status == "completed", run.runtime.reason
    assert not run.pipeline.device.invalid and not run.pipeline.device.duplicate
    assert any(r["kind"] == "rule_takeover" for r in run.pipeline.session.rows("recoveries"))
    rules = [
        a
        for a in run.pipeline.session.rows("bot_actions")
        if a["stage"] == "开始" and a.get("strategy") == "rules"
    ]
    assert rules
    assert all(a["input_context"]["history_complete"] is False for a in rules)
    assert all(not set(a["input_context"]) & {"all_handcards", "other_hand_cards"} for a in rules)
    second = run.runtime.completed[1]
    assert all(
        a.get("strategy") == "wp" for a in run.pipeline.session.rows("bot_actions") if a["game_id"] == second
    )
    from doudizhu_lab.storage.runs import evaluate_run

    assert all(g["passed"] for g in evaluate_run(run.runtime.store.folder)["game_audits"])


def test_unknown_submission_still_pauses_whole_task_without_retry(balanced_run):
    run = balanced_run
    gap_injected = False
    for _ in range(6000):
        if run.runtime.armed_game and not gap_injected:
            run.pipeline.fault = "hidden_play"
            gap_injected = True
        if run.pipeline.operational.rule is not None:
            run.pipeline.device.fault = "lost_submission"
        run.tick()
        if not run.runtime.active:
            break
    assert run.runtime.status == "paused" and "提交后无法确认" in run.runtime.reason
    taps = run.pipeline.device.taps
    for _ in range(30):
        run.tick()
    assert run.pipeline.device.taps == taps
    assert not run.pipeline.device.invalid and not run.pipeline.device.duplicate
