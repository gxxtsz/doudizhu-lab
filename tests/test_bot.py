"""机器人公开信息、同帧快照、坐标反馈及故障停止测试；没有真实设备输入。"""

from collections import Counter
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from doudizhu_lab.agents.douzero import DECK, ROLES, TO_NUMBER, Agent, context, legal_actions
from doudizhu_lab.domain.bot_types import Decision, ObservationSnapshot, encode
from doudizhu_lab.runtime.bot import InputGate
from doudizhu_lab.runtime.executor import ActionExecutor
from doudizhu_lab.simulation.bot import SimulatedDevice


@pytest.fixture
def device():
    return SimulatedDevice(73100, "landlord")


def decision(device, cards=("3",)):
    s = device.last.state
    return Decision(cards, device.last.version, s["game_id"], s["turn"], "landlord", "测试模型", 1, 0.0)


@pytest.mark.parametrize(
    "source,frames,hand,expected",
    [
        ("operational_observed", ["one", "two"], ["3", "4"], True),
        ("operational_observed", ["one"], ["3", "4"], False),
        ("operational_inferred", ["one", "two"], ["3", "4"], False),
        ("operational_observed", ["one", "two"], ["3"], False),
    ],
)
def test_confirmed_pass_remains_evidence_after_prompt_disappears(source, frames, hand, expected):
    """复现三十局首批故障：两帧要不起已确认，当前提示消失，手牌与本轮仍须守恒。"""
    executor = ActionExecutor(lambda *a: None)
    executor.decision = Decision((), "v", "game", 0, "landlord_up", "test", 0, 0)
    executor.before_hand = Counter(["3", "4"])
    executor.submitted_pts = 100_000
    event = {
        "seat": "self",
        "kind": "pass",
        "cards": [],
        "confirmed": True,
        "pts_us": 200_000,
        "source": source,
        "data": {"evidence_frames": frames},
    }
    snapshot = SimpleNamespace(
        state={"pending": False, "game_id": "game", "turn": 1, "hand": ["3", "4"]},
        observation={"passes": [], "hand": hand},
        history=[event],
    )
    assert executor._confirmed(snapshot) == expected


def test_bomb_feedback_waits_for_delayed_analysis_without_any_new_input():
    import numpy as np

    from doudizhu_lab.domain.models import Frame

    now = [1.2]
    commands = []
    executor = ActionExecutor(lambda *args: commands.append(args), clock=lambda: now[0])
    executor.decision = Decision(("8",) * 4, "v", "game", 0, "landlord", "test", 0, 0)
    frame = Frame("sim", np.zeros((3, 3, 3), np.uint8), 0, "old", 0, "simulation", "old", "stream", 0)
    executor.identity = frame.serial, frame.stream_id, frame.width, frame.height
    executor.stage, executor.submitted_at = "等待反馈", 0
    snap = SimpleNamespace(
        frame=frame,
        allowed=True,
        state={"game_id": "game"},
        observation={},
        history=[],
        version="v",
        revision=0,
    )
    for stamp in (1.2, 3.4):
        now[0] = stamp
        executor.advance(snap)
        assert executor.active and not commands
    now[0] = 5.1
    executor.advance(snap)
    assert executor.stage == "暂停" and not commands


def run_action(device, fault="", cards=("3",)):
    device.fault = fault
    executor = ActionExecutor(device.send, device.audits.append, clock=lambda: device.now)
    executor.start(decision(device, cards), device.last)
    for _ in range(150):
        executor.advance(device.render())
        if not executor.active:
            break
    return executor


def test_snapshot_immutable_and_semantic_version_stable(device):
    snap = device.last
    with pytest.raises(ValueError):
        snap.frame.rgb[0, 0, 0] = 4
    with pytest.raises(ValueError):
        snap.frame.rgb.setflags(write=True)
    state = snap.state
    state["hand"].clear()
    assert snap.state["hand"]
    device.selected.add(0)
    assert device.render().version == snap.version
    with pytest.raises(ValueError, match="不是同一帧"):
        ObservationSnapshot.create(
            snap.frame, None, snap.observation | {"frame_id": "wrong"}, snap.state, snap.history
        )


def test_context_excludes_individual_opponent_hands_and_counts_bottom_once(device):
    data = context(device.last)
    assert "all_handcards" not in data
    assert Counter(data["three_landlord_cards"]) == Counter(device.bottom)
    assert len(data["player_hand_cards"]) == 20 and len(data["other_hand_cards"]) == 34
    assert Counter(data["player_hand_cards"] + data["other_hand_cards"]) == DECK
    assert set(data) == {
        "player_position",
        "player_hand_cards",
        "other_hand_cards",
        "num_cards_left_dict",
        "played_cards",
        "last_move",
        "last_move_dict",
        "card_play_action_seq",
        "bomb_num",
        "legal_actions",
        "three_landlord_cards",
    }


@pytest.mark.parametrize("flag", ["pending", "capture_incomplete", "ended"])
def test_context_rejects_unsafe_state(device, flag):
    changed = replace(device.last, state_json=encode(device.last.state | {flag: True}))
    with pytest.raises(ValueError):
        context(changed)


def test_missing_event_cannot_be_filled_as_pass(device):
    changed = replace(device.last, state_json=encode(device.last.state | {"turn": 1}))
    with pytest.raises(ValueError, match="历史不连续"):
        context(changed)


@pytest.mark.parametrize(
    "hand,previous,wanted",
    [
        ([3, 3, 4, 4], [3, 3], [4, 4]),
        ([3, 3, 3, 4, 4], [], [3, 3, 3, 4, 4]),
        ([3, 4, 5, 6, 7, 8], [3, 4, 5, 6, 7], [4, 5, 6, 7, 8]),
        ([3, 3, 3, 3, 20, 30], [14, 14, 14, 14], [20, 30]),
        ([3, 3, 3, 4, 4, 4, 5, 6], [], [3, 3, 3, 4, 4, 4, 5, 6]),
        ([3, 3, 3, 3, 4, 5], [], [3, 3, 3, 3, 4, 5]),
        ([3, 3, 3, 3, 4, 4, 5, 5], [], [3, 3, 3, 3, 4, 4, 5, 5]),
    ],
)
def test_classic_legal_combinations(hand, previous, wanted):
    assert wanted in legal_actions(hand, previous)
    assert ([] in legal_actions(hand, previous)) == bool(previous)
    assert legal_actions(hand, [20, 30]) == [[]]


@pytest.mark.parametrize("fault", ["", "wrong_card"])
def test_visual_feedback_closed_loop_and_wrong_selection_repair(device, fault):
    before = Counter(device.last.state["hand"])
    executor = run_action(device, fault)
    assert executor.stage == "已确认"
    assert Counter(device.tracker.state.hand) == before - Counter(["3"])
    assert len(device.submissions) == 1
    assert sum(a["stage"] == "命令已返回" and a["kind"] == "play" for a in device.audits) == 1
    with pytest.raises(ValueError):
        executor.start(executor.decision, device.last)


@pytest.mark.parametrize("fault", ["missed_click", "disconnect", "lost_submission"])
def test_uncertain_command_or_feedback_never_retries_submit(device, fault):
    executor = run_action(device, fault)
    assert executor.stage == "暂停"
    assert not any(a["stage"] == "已确认" for a in device.audits)
    assert sum(a["stage"] == "准备输入" and a["kind"] == "play" for a in device.audits) <= 1


def test_same_frame_does_not_count_as_two_confirmations(device):
    device.selected.add(device.last.state["hand"].index("3"))
    snapshot = device.render()
    executor = ActionExecutor(device.send, clock=lambda: device.now)
    executor.start(decision(device), snapshot)
    for _ in range(10):
        executor.advance(snapshot)
    assert executor.stage == "核对" and not device.submissions
    executor.advance(device.render())
    assert len(device.submissions) == 1


@pytest.mark.parametrize("fault", ["stale", "orientation", "stream", "device", "uncertain", "gap", "manual"])
def test_execution_stops_on_invalid_observation(device, fault):
    d = decision(device)
    executor = ActionExecutor(device.send, clock=lambda: device.now)
    executor.start(d, device.last)
    snapshot = device.render()
    if fault == "stale":
        snapshot = replace(snapshot, frame=replace(snapshot.frame, started=device.now - 2))
    if fault == "stream":
        snapshot = replace(snapshot, frame=replace(snapshot.frame, stream_id="new"))
    if fault == "device":
        snapshot = replace(snapshot, frame=replace(snapshot.frame, serial="other"))
    if fault == "orientation":
        snapshot = replace(snapshot, frame=replace(snapshot.frame, rgb=snapshot.frame.rgb.transpose(1, 0, 2)))
    if fault == "uncertain":
        snapshot = replace(snapshot, hand=tuple(replace(c, selected=None) for c in snapshot.hand))
    if fault == "gap":
        # pending候选现在允许短暂等待；本用例继续验证真正的可信历史缺口立即停止。
        snapshot = replace(snapshot, pending=True, state_json=encode(snapshot.state | {"pending": True}))
    if fault == "manual":
        executor.cancel("人工介入")
    executor.advance(snapshot)
    if fault == "uncertain":
        assert executor.active and device.taps == 0
        device.now += 3.1
        # 动画期间允许只读等待；持续未知仍须暂停，绝不发送试探性点击。
        snapshot = replace(
            snapshot, frame=replace(snapshot.frame, started=device.now, frame_id="uncertain-later")
        )
        executor.advance(snapshot)
    assert executor.stage == "暂停" and device.taps == 0


def test_temporary_selection_animation_recovers_without_extra_click(device):
    executor = ActionExecutor(device.send, clock=lambda: device.now)
    executor.start(decision(device), device.last)
    unknown = replace(device.render(), hand=tuple(replace(c, selected=None) for c in device.last.hand))
    executor.advance(unknown)
    assert executor.active and device.taps == 0
    for _ in range(80):
        executor.advance(device.render())
        if not executor.active:
            break
    assert executor.stage == "已确认" and device.invalid == device.duplicate == 0


def test_manual_selection_change_without_command_pauses(device):
    device.selected.add(device.last.state["hand"].index("3"))
    snap = device.render()
    ex = ActionExecutor(device.send, clock=lambda: device.now)
    ex.start(decision(device), snap)
    ex.advance(snap)
    device.selected.add(1)
    ex.advance(device.render())
    assert ex.stage == "暂停" and device.taps == 0


def test_cancel_is_immediate_and_no_new_command(device):
    ex = ActionExecutor(device.send, clock=lambda: device.now)
    ex.start(decision(device), device.last)
    ex.advance(device.render())
    before = device.taps
    ex.cancel("用户立即停止")
    for _ in range(10):
        ex.advance(device.render())
    assert device.taps == before


def test_input_gate_revocation_heartbeat_and_replay(device):
    gate = InputGate(clock=lambda: device.now)
    client = Mock()
    token = gate.grant("bot")
    gate.heartbeat()
    frame = replace(device.last.frame, transport="video")
    gate.revoke()
    with pytest.raises(ValueError):
        gate.tap("bot", token, client, 1, 1, frame)
    token = gate.grant("bot")
    with pytest.raises(ValueError):
        gate.tap("bot", token, client, 1, 1, replace(frame, transport="replay"))
    device.now += 2
    with pytest.raises(ValueError):
        gate.tap("bot", token, client, 1, 1, frame)
    gate.heartbeat()
    with pytest.raises(ValueError):
        gate.tap("bot", token, client, 1, 1, frame, guard=lambda: False)
    client.tap.assert_not_called()


def test_pinned_models_missing_or_changed_rejected(tmp_path):
    agent = Agent(tmp_path)
    with pytest.raises(OSError):
        agent.start()
    assert agent.process is None


def test_real_model_features_match_official_public_view():
    from types import SimpleNamespace

    import numpy as np

    from doudizhu_lab._vendor.douzero.env.env import get_obs

    for role in ROLES:
        device = SimulatedDevice(73102, role)
        while device.env.acting_player_position != role:
            device.act(device.env.game_infoset.legal_actions[0])
            for _ in range(3):
                device.render()
        adapted = get_obs(SimpleNamespace(**context(device.last)))
        official = get_obs(device.env.game_infoset)
        assert adapted["legal_actions"] == official["legal_actions"]
        assert np.array_equal(adapted["z_batch"], official["z_batch"])
        assert np.array_equal(adapted["x_batch"], official["x_batch"])


@pytest.mark.model
def test_trained_agent_all_roles_and_process_exit():
    root = Path(__file__).resolve().parents[1] / "data/models/douzero-wp"
    if not root.exists():
        pytest.skip("本机未安装模型")
    agent = Agent(root)
    try:
        agent.start()
        for role in ROLES:
            device = SimulatedDevice(73102, role)
            while device.env.acting_player_position != role:
                device.act(device.env.game_infoset.legal_actions[0])
                for _ in range(3):
                    device.render()
            d = agent.decide(device.last)
            assert [TO_NUMBER[c] for c in d.cards] in device.env.game_infoset.legal_actions
            assert d.role == role
        agent.process.terminate()
        agent.process.wait()
        with pytest.raises(ValueError):
            agent.decide(device.last)
    finally:
        agent.close()


def test_moving_button_uses_latest_frame_and_disabled_never_submits(device):
    device.selected.add(0)
    device.render()
    sent = []

    def send(kind, x, y, snapshot, action_id):
        sent.append((kind, x, y))
        device.send(kind, x, y, snapshot, action_id)

    ex = ActionExecutor(send, clock=lambda: device.now)
    ex.start(decision(device), device.last)
    ex.advance(device.last)
    device.button_dx = -100
    ex.advance(device.render())
    assert sent[-1][0:2] == ("play", 355)
    for _ in range(3):
        ex.advance(device.render())
    assert ex.stage == "已确认"
    fresh = SimulatedDevice(73100, "landlord")
    fresh.selected.add(0)
    fresh.buttons_enabled = False
    fresh.render()
    ex = ActionExecutor(fresh.send, clock=lambda: fresh.now)
    ex.start(decision(fresh), fresh.last)
    for _ in range(105):
        ex.advance(fresh.render())
    assert ex.stage == "暂停" and not fresh.submissions


def test_expired_decision_and_wrong_button_coordinate_cannot_submit(device):
    ex = ActionExecutor(device.send, clock=lambda: device.now)
    with pytest.raises(ValueError):
        ex.start(replace(decision(device), version="旧版本"), device.last)
    device.selected.add(0)
    device.send("play", 1, 1, device.last, "误点")
    assert not device.submissions and device.tracker.state.turn == 0


def test_unresponsive_process_stops_at_two_seconds(device, tmp_path):
    import subprocess
    import sys
    from time import perf_counter

    agent = Agent(tmp_path)
    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    agent.process = child
    started = perf_counter()
    try:
        with pytest.raises(ValueError, match="超时"):
            agent.decide(device.last)
        assert 1.8 < perf_counter() - started < 4
        assert child.poll() is not None and agent.process is None
    finally:
        agent.close()


def test_semantic_revision_only_changes_for_game_facts(device):
    initial = device.last.revision
    device.selected.add(0)
    assert device.render().revision == initial
    assert device.render().revision == initial
    device.act([3])
    for _ in range(3):
        device.render()
    assert device.last.revision > initial
