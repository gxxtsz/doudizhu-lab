"""从大厅到多局结算的生产协调器回归；模拟策略仅用于测试，绝不连接真实手机。"""

import json
from dataclasses import asdict, replace
from threading import Event, Thread
from time import monotonic
from types import SimpleNamespace

import numpy as np
import pytest

from doudizhu_lab.agents.douzero import TO_RANK, context
from doudizhu_lab.agents.readiness import assess
from doudizhu_lab.domain.bot_types import Decision, encode
from doudizhu_lab.domain.models import Frame, PixelRect
from doudizhu_lab.domain.settings import FLOW_ROOMS
from doudizhu_lab.runtime.bot import BotRuntime, InputGate
from doudizhu_lab.runtime.continuous import ContinuousRuntime
from doudizhu_lab.simulation.continuous import ContinuousSimPipeline
from doudizhu_lab.storage.config import Profile, Store
from doudizhu_lab.storage.runs import evaluate_run, uncovered_video_intervals
from doudizhu_lab.vision.flow import FlowRecognizer


class LegalTestAgent:
    """固定合法策略使故障测试快速可重现；正式模拟另用真实WP权重，禁止替代。"""

    model_id = "仅用于测试的合法策略"

    def start(self):
        pass

    def close(self):
        pass

    def decide(self, snapshot):
        value = context(snapshot)
        cards = value["legal_actions"][0]
        s = snapshot.state
        return Decision(
            tuple(TO_RANK[n] for n in cards),
            snapshot.version,
            s["game_id"],
            s["turn"],
            value["player_position"],
            self.model_id,
            1,
            0,
        )


@pytest.fixture
def run(tmp_path):
    now = [monotonic()]

    def clock():
        return now[0]

    store = Store(tmp_path)
    pipeline = ContinuousSimPipeline(store, clock=clock)
    pipeline.start()
    gate = InputGate(clock=clock, transport="simulation")
    bot = BotRuntime(tmp_path, gate, clock=clock, agent=LegalTestAgent(), threaded=False)
    runtime = ContinuousRuntime(tmp_path, gate, bot, pipeline, clock=clock, threaded=False)
    runtime.start()

    def tick(count=1):
        for _ in range(count):
            now[0] += 0.12
            pipeline.tick()
            gate.heartbeat()
            bot.pulse()
            runtime.advance(pipeline.flow_snapshot(), pipeline.bot_snapshot())

    value = SimpleNamespace(
        now=now, clock=clock, store=store, pipeline=pipeline, gate=gate, bot=bot, runtime=runtime, tick=tick
    )
    yield value
    runtime.close()
    bot.close()
    pipeline.stop()
    pipeline.session.close()


def playing(run):
    for _ in range(100):
        run.tick()
        if run.runtime.armed_game:
            return
    raise AssertionError(run.runtime.description)


def test_five_games_share_production_pipeline_and_audit_every_own_action(run):
    for _ in range(12000):
        run.tick()
        if not run.runtime.active:
            break
    report = evaluate_run(run.runtime.store.folder)
    assert report["passed"] and not report["live_passed"]
    assert report["completed_games"] == 5 and report["transitions"] == 4
    assert all(g["own_actions"] == g["confirmed_actions"] for g in report["games"])
    assert not sum(d.invalid + d.duplicate for d in run.pipeline.devices)
    assert sum(k == "match" for _, k in run.pipeline.flow_commands) == 5
    assert sum(k == "continue" for _, k in run.pipeline.flow_commands) == 4


def test_thirty_game_target_cannot_pass_at_five_and_stops_exactly_at_thirty(run):
    """局数上限与验收目标共用同一快照，验证29次衔接及到达30局后不再匹配。"""
    run.runtime.settings.update(max_games=30, time_limit=14400)
    run.pipeline.profile.continuous_settings.update(max_games=30, time_limit=14400)
    run.runtime.store.set("profile", asdict(run.pipeline.profile))
    checked_five = False
    for _ in range(72000):
        run.tick()
        if len(run.runtime.completed) == 5 and not checked_five:
            result = evaluate_run(run.runtime.store.folder)
            assert result["required_games"] == 30 and not result["passed"]
            checked_five = True
        if not run.runtime.active:
            break
    result = evaluate_run(run.runtime.store.folder)
    assert checked_five and result["passed"]
    assert result["completed_games"] == 30 and result["transitions"] == 29
    assert sum(k == "match" for _, k in run.pipeline.flow_commands) == 30
    before = len(run.pipeline.flow_commands)
    run.tick(10)
    assert len(run.pipeline.flow_commands) == before


def test_identical_frame_cannot_stabilize_or_duplicate_ui_command(run):
    run.tick()
    frame = run.pipeline.flow_snapshot()
    for _ in range(30):
        run.runtime.advance(frame, None)
    assert not run.pipeline.flow_commands
    run.tick()
    assert len(run.pipeline.flow_commands) == 1
    for _ in range(10):
        run.runtime.advance(run.pipeline.flow_snapshot(), None)
    assert len(run.pipeline.flow_commands) == 1


def test_delayed_blocking_popup_stops_without_using_old_frame_for_input(run):
    """阻断证据允许撤销输入；OCR延迟不能掩盖健康系统，也不能放宽旧帧点击限制。"""
    run.tick()
    old = run.pipeline.flow_snapshot()
    blocked = replace(old, scene="unknown", blocked_reason="游戏健康系统要求休息")
    run.now[0] += 1.5
    run.gate.heartbeat()
    before = len(run.pipeline.flow_commands)
    run.runtime.advance(blocked, None)
    assert run.runtime.status == "paused" and "健康系统" in run.runtime.reason
    run.tick(40)
    assert len(run.pipeline.flow_commands) == before and run.gate.owner is None


def test_ui_no_feedback_pauses_without_retry_and_resume_refuses_unknown_command(run):
    run.pipeline.fault = "no_feedback"
    run.tick(40)
    assert run.runtime.status == "paused" and len(run.pipeline.flow_commands) == 1
    before = len(run.pipeline.flow_commands)
    run.tick(20)
    assert len(run.pipeline.flow_commands) == before
    with pytest.raises(ValueError, match="未确认"):
        run.runtime.resume()
    assert evaluate_run(run.runtime.store.folder)["unconfirmed_flow_actions"]


def test_balanced_missing_page_waits_for_unique_progress_without_reclick(run):
    run.pipeline.profile.continuous_policy = "balanced"
    run.pipeline.fault = "no_feedback"
    run.tick(2)
    pending = run.runtime.pending_flow
    assert pending
    start = pending["sent_at"]
    original = run.pipeline.flow_snapshot()
    for index, (elapsed, scene) in enumerate(((3.5, "unknown"), (4.2, "dealing"))):
        run.now[0] = start + elapsed
        frame = replace(
            original.frame,
            started=run.clock(),
            frame_id=f"delayed-page-{index}",
            media_pts_us=original.frame.media_pts_us + round(elapsed * 1_000_000),
        )
        observed = replace(original, frame=frame, scene=scene, buttons=())
        run.gate.heartbeat()
        run.runtime.advance(observed, None)
        assert run.runtime.active and len(run.pipeline.flow_commands) == 1
    assert run.runtime.pending_flow is None


def test_matching_waits_for_first_archived_keyframe_and_times_out_without_input(run):
    run.pipeline.record_ready = Event()
    run.tick(40)
    assert run.runtime.active and not run.pipeline.flow_commands
    run.pipeline.record_ready.set()
    run.tick(3)
    assert len(run.pipeline.flow_commands) == 1


def test_cold_foreground_probe_waits_without_input_then_requires_valid_evidence(run):
    run.pipeline.allowed = False
    run.tick(60)
    assert run.runtime.active and not run.pipeline.flow_commands
    run.pipeline.allowed = True
    run.tick(4)
    assert run.runtime.active and len(run.pipeline.flow_commands) == 1


def test_confirmed_ending_animation_keeps_observing_without_input_and_has_bound(run):
    run.tick()
    initial = run.pipeline.flow_snapshot()
    start = run.clock()
    for index, elapsed in enumerate((0.1, 9.0, 10.3)):
        run.now[0] = start + elapsed
        fresh = replace(
            initial.frame,
            frame_id=f"ending-{index}",
            started=run.clock(),
            media_pts_us=initial.frame.media_pts_us + round(elapsed * 1_000_000),
        )
        flow = replace(initial, frame=fresh, scene="ending", buttons=())
        run.gate.heartbeat()
        run.runtime.advance(flow, None)
        assert not run.pipeline.flow_commands
        if elapsed < 10:
            assert run.runtime.active
    assert run.runtime.status == "paused" and "十秒" in run.runtime.reason


def test_recording_without_keyframe_cannot_wait_forever_or_click(run):
    run.pipeline.record_ready = Event()
    run.tick(180)
    assert run.runtime.status == "paused" and not run.pipeline.flow_commands
    assert "关键帧" in run.runtime.reason


def test_entering_midgame_without_verified_lobby_cannot_arm_bot(run):
    playing(run)
    run.runtime.mode_verified = False
    run.runtime.armed_game = ""
    run.tick()
    assert run.runtime.status == "paused" and "大厅" in run.runtime.reason


def test_video_coverage_distinguishes_closed_containers_from_whole_game():
    segments = [
        {"global_start_us": 1_500_000, "global_end_us": 3_000_000},
        {"global_start_us": 3_066_667, "global_end_us": 5_000_000},
    ]
    assert uncovered_video_intervals(segments, 1_000_000, 4_000_000) == [(1_000_000, 1_500_000)]
    assert not uncovered_video_intervals(segments, 2_000_000, 4_000_000)
    assert uncovered_video_intervals(segments, 2_000_000, 6_000_000) == [(5_000_000, 6_000_000)]
    assert uncovered_video_intervals([], 2_000_000, 6_000_000) == [(2_000_000, 6_000_000)]


def test_double_button_disappearance_needs_two_frames_when_phase_temporarily_regresses(run):
    for _ in range(60):
        run.tick()
        if run.runtime.pending_flow and run.runtime.pending_flow["kind"] == "decline_double":
            break
    assert run.runtime.pending_flow["kind"] == "decline_double"
    # 真实加倍动画有时只剩牌桌外壳，OCR反馈短暂漏检；不把它当作阶段前进。
    for index in range(2):
        run.now[0] += 0.2
        frame = replace(
            run.pipeline.frame,
            frame_id=f"doubling-animation-{index}",
            media_pts_us=run.pipeline.pts + (index + 1) * 200_000,
            started=run.clock(),
        )
        flow = replace(run.pipeline.flow, frame=frame, scene="dealing", buttons=(), feedbacks=())
        run.gate.heartbeat()
        run.runtime.advance(flow, None)
        if index == 0:
            assert run.runtime.pending_flow is not None
    assert run.runtime.pending_flow is None and run.runtime.active
    assert sum(k == "decline_double" for _, k in run.pipeline.flow_commands) == 1


def test_bot_worker_failure_clears_previously_ready_authorization(run, monkeypatch):
    def fail():
        raise RuntimeError("模拟模型协调线程退出")

    monkeypatch.setattr(run.bot, "pulse", fail)
    run.bot._run()
    assert not run.bot.ready.is_set() and run.bot.mode == "stop"
    run.bot.thread = SimpleNamespace(is_alive=lambda: False)
    run.gate.heartbeat()
    run.runtime.advance(run.pipeline.flow_snapshot(), None)
    assert run.runtime.status == "paused" and not run.pipeline.flow_commands
    run.bot.thread = None


def test_model_exit_is_detected_before_any_page_action(tmp_path):
    from doudizhu_lab.agents.douzero import Agent

    agent = Agent(tmp_path)
    agent.process = SimpleNamespace(poll=lambda: 1)
    with pytest.raises(RuntimeError, match="模型子进程已经退出"):
        agent.check_running()
    agent.process = SimpleNamespace(poll=lambda: None)
    agent.check_running()


def test_stop_does_not_wait_for_inflight_usb_command(run, monkeypatch):
    started, release = Event(), Event()

    def tap(*a, **kw):
        started.set()
        assert release.wait(3)

    monkeypatch.setattr(run.pipeline, "tap", tap)
    run.tick()
    thread = Thread(target=run.tick)
    thread.start()
    try:
        assert started.wait(2)
        before = monotonic()
        run.runtime.stop()
        assert monotonic() - before < 0.5 and run.gate.owner is None
    finally:
        release.set()
        thread.join(3)
    assert not thread.is_alive() and run.runtime.status == "stopped"


@pytest.mark.parametrize("kind", ["stream", "orientation", "device"])
def test_source_changes_revoke_future_input(run, kind):
    run.tick()
    flow = run.pipeline.flow_snapshot()
    frame = flow.frame
    if kind == "stream":
        frame = replace(frame, stream_id="other")
    elif kind == "device":
        frame = replace(frame, serial="other")
    else:
        frame = replace(frame, rgb=frame.rgb.transpose(1, 0, 2))
    run.gate.heartbeat()
    run.runtime.advance(replace(flow, frame=replace(frame, frame_id="changed")), None)
    assert run.runtime.status == "paused" and not run.pipeline.flow_commands


def test_unknown_page_and_match_timeouts(run):
    for scene, delay in (("unknown", 5.1), ("matching", 120.1)):
        run.runtime.status = "running"
        run.runtime.last_id = ""
        run.pipeline.tick()
        run.gate.heartbeat()
        run.runtime.advance(replace(run.pipeline.flow_snapshot(), scene=scene, buttons=()), None)
        run.now[0] += delay
        run.pipeline.tick()
        run.gate.heartbeat()
        run.runtime.advance(replace(run.pipeline.flow_snapshot(), scene=scene, buttons=()), None)
        assert run.runtime.status == "paused"
    assert not run.pipeline.flow_commands


def test_true_gap_pauses_and_cannot_resume_this_game(run):
    playing(run)
    run.pipeline.device.tracker.state.pending = True
    run.tick(2)
    assert run.runtime.status == "paused"
    before = run.pipeline.device.taps
    with pytest.raises(ValueError, match="历史缺口"):
        run.runtime.resume()
    run.tick(10)
    assert run.pipeline.device.taps == before


def test_transient_candidate_waits_but_true_history_gap_halts(run):
    playing(run)
    snapshot = run.pipeline.bot_snapshot()
    state = snapshot.state | {"current_actor": "self"}
    obs = snapshot.observation | {"actor": "self"}
    snapshot = replace(snapshot, state_json=encode(state), observation_json=encode(obs), pending=True)
    result = assess(snapshot)
    assert result.status == "wait" and result.bounded
    snapshot = replace(snapshot, state_json=encode(state | {"pending": True}))
    assert assess(snapshot).status == "halt"


def test_time_limit_finishes_current_game_and_does_not_match_again(run):
    playing(run)
    run.runtime.settings["time_limit"] = 1
    for _ in range(3000):
        run.tick()
        if not run.runtime.active:
            break
    assert run.runtime.status == "completed" and len(run.runtime.completed) == 1
    assert sum(k == "match" for _, k in run.pipeline.flow_commands) == 1
    assert not evaluate_run(run.runtime.store.folder)["passed"]  # 少于五局必须如实未通过。


def test_redeal_is_not_counted_as_completed_game(run):
    for _ in range(30):
        if run.pipeline.phase == "bidding":
            run.pipeline.fault = "redeal"
            break
        run.tick()
    run.tick(5)
    assert not run.runtime.completed
    assert run.pipeline.phase in ("dealing", "bidding")


def test_model_failure_or_heartbeat_loss_never_rearms_automatically(run):
    run.bot.ready.clear()
    run.bot.thread = SimpleNamespace(is_alive=lambda: False)
    run.bot.status = "模型子进程退出"
    run.gate.heartbeat()
    run.runtime.advance(run.pipeline.flow_snapshot(), None)
    assert run.runtime.status == "paused" and not run.pipeline.flow_commands
    run.bot.thread = None
    run.bot.ready.set()
    run.tick(5)
    assert not run.pipeline.flow_commands


def test_version_four_preserves_version_three_and_never_persists_authorization(tmp_path):
    original = json.dumps({"version": 3, "profile": asdict(Profile())}, ensure_ascii=False)
    (tmp_path / "settings.json").write_text(original, encoding="utf8")
    store = Store(tmp_path)
    store.save()
    assert (tmp_path / "settings.v3.bak.json").read_text(encoding="utf8") == original
    data = json.loads(store.settings_path.read_text(encoding="utf8"))
    assert data["version"] == 4 and data["profile"]["continuous_settings"] == {
        "max_games": 5,
        "time_limit": 1800,
    }
    assert not any(k in data["profile"] for k in ("auto", "armed", "run_id", "running"))


def test_late_bottom_and_landlord_can_complete_setup_only_before_first_action():
    from collections import Counter

    from doudizhu_lab.domain.tracker import GameTracker
    from doudizhu_lab.vision.table import TableObservation

    tracker = GameTracker(["self", "right", "left"])
    hand = ["3", "3", "3", "4", "4", "4", "5", "5", "5", "6", "6", "7", "8", "9", "10", "J", "Q"]
    bottom = ["A", "2", "SJ"]

    def feed(t, phase, shown, **values):
        return tracker.observe(TableObservation(str(t), t * 1000, phase=phase, hand=shown, **values))

    for t in (0, 120, 240):
        feed(t, "bidding", hand)
    for t in (360, 480):
        feed(t, "playing", hand)
    assert tracker.state.initial_hand == hand
    for t in (600, 720):
        feed(
            t,
            "playing",
            hand + bottom,
            bottom=bottom,
            landlord="self",
            actor="self",
            remaining={"self": 20, "left": 17, "right": 17},
        )
    assert Counter(tracker.state.initial_hand) == Counter(hand + bottom)
    assert tracker.state.current_actor == "self" and not tracker.state.pending
    # 已知20张开局后进入第一手，初始手牌必须保留；底牌也不能再加入一次。
    for t in (840, 960, 1080):
        feed(
            t,
            "playing",
            hand + bottom[1:],
            bottom=bottom,
            landlord="self",
            actor="right",
            remaining={"self": 19, "left": 17, "right": 17},
            plays={"self": ["A"]},
        )
    assert Counter(tracker.state.initial_hand) == Counter(hand + bottom)
    assert len(tracker.state.initial_hand) == 20


@pytest.mark.parametrize("lobby_room", FLOW_ROOMS)
@pytest.mark.parametrize("use_current", [False, True])
def test_flow_ocr_requires_button_color_region_and_known_mode(monkeypatch, lobby_room, use_current):
    recognizer = FlowRecognizer.__new__(FlowRecognizer)
    rgb = np.full((1080, 2400, 3), 80, np.uint8)
    # 用固定OCR输出隔离测试几何及背景规则；不声称这张合成图片能被真实OCR读出。
    words = [
        ("开始匹配", 0.99, PixelRect(1900, 930, 250, 80)),
        (f"洗牌{lobby_room}场", 0.99, PixelRect(1850, 790, 330, 60)),
    ]
    calls = []

    def text(*args):
        calls.append(True)
        return words

    monkeypatch.setattr(recognizer, "_text", text)
    frame = Frame("offline", rgb, 0, "one", 0, "file", frame_id="one")
    room = "当前" if use_current else lobby_room
    assert not recognizer.observe(frame, room=room).buttons
    rgb[900:1040, 1850:2220] = (240, 170, 30)
    calls.clear()
    result = recognizer.observe(frame, room=room)
    assert result.scene == "lobby" and result.mode_verified and result.buttons[0].kind == "match"
    assert len(calls) == 1  # 三个场次均走右下角快速扫描，不额外识别角色或昵称。
    monkeypatch.setattr(recognizer, "_text", lambda *a: words[:1])
    assert recognizer.observe(frame, room=room).scene == "unknown"


@pytest.mark.parametrize(
    ("room", "mode", "label", "score", "box", "colored"),
    [
        ("初级", "洗牌高级场", "开始匹配", 0.99, PixelRect(1900, 930, 250, 80), True),
        ("高级", "洗牌中级场", "开始匹配", 0.99, PixelRect(1900, 930, 250, 80), True),
        ("当前", "洗牌专家场", "开始匹配", 0.99, PixelRect(1900, 930, 250, 80), True),
        ("专家", "洗牌专家场", "开始匹配", 0.99, PixelRect(1900, 930, 250, 80), True),
        ("当前", "经典高级场", "开始匹配", 0.99, PixelRect(1900, 930, 250, 80), True),
        ("当前", "不洗牌高级场", "开始匹配", 0.99, PixelRect(1900, 930, 250, 80), True),
        ("初级", "洗牌高级场无法进入初级场", "开始匹配", 0.99, PixelRect(1900, 930, 250, 80), True),
        ("当前", "洗牌中级场洗牌高级场", "开始匹配", 0.99, PixelRect(1900, 930, 250, 80), True),
        ("当前", "洗牌高级场", "开始匹配", 0.84, PixelRect(1900, 930, 250, 80), True),
        ("当前", "洗牌高级场", "开始匹配", 0.99, PixelRect(900, 930, 250, 80), True),
        ("当前", "洗牌高级场", "开始匹配", 0.99, PixelRect(1900, 930, 250, 80), False),
        ("当前", "洗牌高级场", "开始", 0.99, PixelRect(1900, 930, 250, 80), True),
    ],
)
def test_lobby_cannot_match_without_selected_room_and_button_evidence(
    monkeypatch, room, mode, label, score, box, colored
):
    recognizer = FlowRecognizer.__new__(FlowRecognizer)
    rgb = np.full((1080, 2400, 3), (240, 170, 30) if colored else (80, 80, 80), np.uint8)
    words = [(label, score, box), (mode, 0.99, PixelRect(1850, 790, 330, 60))]
    monkeypatch.setattr(recognizer, "_text", lambda *args: words)
    frame = Frame("offline", rgb, 0, "one", 0, "file", frame_id="one")
    result = recognizer.observe(frame, room=room)
    assert result.scene == "unknown" and not result.buttons


@pytest.mark.parametrize("lobby_room", FLOW_ROOMS)
def test_recognized_lobby_starts_and_continues_games_without_duplicate_match(run, monkeypatch, lobby_room):
    """合成 OCR 经生产识别器进入模拟设备，验证两帧授权、匹配及下一局衔接。"""
    run.pipeline.profile.flow_room = "当前"
    run.pipeline.profile.continuous_settings["max_games"] = run.runtime.settings["max_games"] = 2
    recognizer = FlowRecognizer.__new__(FlowRecognizer)
    words = [
        (f"洗牌{lobby_room}场", 0.99, PixelRect(450, 70, 100, 8)),
        ("开始匹配", 0.99, PixelRect(460, 84, 90, 10)),
    ]
    monkeypatch.setattr(recognizer, "_text", lambda *args: words)
    for index in range(2):
        run.now[0] += 0.12
        run.pipeline.tick()
        frame = replace(run.pipeline.frame, rgb=np.full((100, 600, 3), (240, 170, 30), np.uint8))
        run.pipeline.frame = frame
        run.pipeline.flow = recognizer.observe(frame, room=run.pipeline.profile.flow_room)
        run.gate.heartbeat()
        run.bot.pulse()
        run.runtime.advance(run.pipeline.flow, None)
        assert len(run.pipeline.flow_commands) == index
        # 同一观察再消费多次，不能增加稳定证据或再次发送匹配。
        for _ in range(3):
            run.runtime.advance(run.pipeline.flow, None)
        assert len(run.pipeline.flow_commands) == index
    assert run.runtime.mode_verified and run.pipeline.phase == "matching"
    for _ in range(6000):
        run.tick()
        if not run.runtime.active:
            break
    assert run.runtime.status == "completed" and len(run.runtime.completed) == 2
    assert sum(kind == "match" for _, kind in run.pipeline.flow_commands) == 2
    assert sum(kind == "continue" for _, kind in run.pipeline.flow_commands) == 1
    assert not sum(device.invalid + device.duplicate for device in run.pipeline.devices)
    before = len(run.pipeline.flow_commands)
    run.tick(10)
    assert len(run.pipeline.flow_commands) == before


@pytest.mark.parametrize("text", ["呵护双眼请您休息一下", "健康系统您今日的累计在线时长已经很长时间了哦"])
def test_health_popup_has_explicit_stop_reason_and_no_buttons(monkeypatch, text):
    from doudizhu_lab.domain.models import PixelRect

    recognizer = FlowRecognizer.__new__(FlowRecognizer)
    frame = Frame("offline", np.zeros((1080, 2400, 3), np.uint8), 0, "one", 0, "file", frame_id="one")
    monkeypatch.setattr(recognizer, "_text", lambda *a: [(text, 0.99, PixelRect(1120, 330, 700, 70))])
    result = recognizer.observe(frame)
    assert "健康系统" in result.blocked_reason and result.scene == "unknown" and not result.buttons


@pytest.mark.parametrize(("source", "target"), [("中级", "初级"), ("初级", "中级"), ("中级", "高级")])
@pytest.mark.parametrize("use_current", [False, True])
def test_room_upgrade_requires_explicit_selection_and_verified_message(
    monkeypatch, source, target, use_current
):
    recognizer = FlowRecognizer.__new__(FlowRecognizer)
    rgb = np.full((1080, 2400, 3), 80, np.uint8)
    rgb[720:810, 1030:1370] = (230, 175, 65)
    words = [
        (f"当前福豆不符合要求无法进入{source}场已为您切换", 0.99, PixelRect(700, 470, 900, 50)),
        (f"至{target}场", 0.99, PixelRect(1100, 525, 200, 50)),
        ("确定", 0.99, PixelRect(1160, 750, 80, 40)),
    ]
    monkeypatch.setattr(recognizer, "_text", lambda *a: words)
    frame = Frame("offline", rgb, 0, "one", 0, "file", frame_id="one")
    refused = recognizer.observe(frame, room=source)
    assert refused.scene == "unknown" and not refused.buttons and f"切换{target}场" in refused.blocked_reason
    selected = recognizer.observe(frame, room="当前" if use_current else target)
    assert selected.scene == "room_confirmation" and selected.buttons[0].kind == "confirm_room"
    monkeypatch.setattr(recognizer, "_text", lambda *a: words[-1:])
    assert not recognizer.observe(frame, room=target).buttons


@pytest.mark.parametrize(
    "message",
    [
        "已为您切换",
        "已为您切换至专家场",
        "已为您切换至中级场已为您切换至高级场",
        "中级场已为您切换请确认",
        "已为您切换至高级场已为您切换",
    ],
)
def test_room_switch_unknown_or_conflicting_target_blocks_confirmation(monkeypatch, message):
    recognizer = FlowRecognizer.__new__(FlowRecognizer)
    rgb = np.full((1080, 2400, 3), (230, 175, 65), np.uint8)
    words = [
        (message, 0.99, PixelRect(700, 470, 900, 50)),
        ("确定", 0.99, PixelRect(1160, 750, 80, 40)),
    ]
    monkeypatch.setattr(recognizer, "_text", lambda *args: words)
    frame = Frame("offline", rgb, 0, "one", 0, "file", frame_id="one")
    result = recognizer.observe(frame, room="当前")
    assert result.scene == "unknown" and not result.buttons and "无法确认" in result.blocked_reason


@pytest.mark.parametrize("target", FLOW_ROOMS)
def test_room_switch_accepts_target_split_across_ocr_words(monkeypatch, target):
    recognizer = FlowRecognizer.__new__(FlowRecognizer)
    rgb = np.full((1080, 2400, 3), (230, 175, 65), np.uint8)
    words = [
        ("已为您", 0.99, PixelRect(700, 470, 150, 50)),
        (f"切换到洗牌{target[0]}", 0.99, PixelRect(850, 470, 350, 50)),
        (f"{target[1:]}场", 0.99, PixelRect(1200, 470, 150, 50)),
        ("确定", 0.99, PixelRect(1160, 750, 80, 40)),
    ]
    monkeypatch.setattr(recognizer, "_text", lambda *args: words)
    frame = Frame("offline", rgb, 0, "one", 0, "file", frame_id="one")
    result = recognizer.observe(frame, room="当前")
    assert result.scene == "room_confirmation" and [button.kind for button in result.buttons] == [
        "confirm_room"
    ]


def test_matching_loading_text_and_back_cards_do_not_hide_bidding_button(monkeypatch):
    from doudizhu_lab.domain.models import PixelRect

    recognizer = FlowRecognizer.__new__(FlowRecognizer)
    rgb = np.full((1080, 2400, 3), 80, np.uint8)
    frame = Frame("offline", rgb, 0, "one", 0, "file", frame_id="one")
    monkeypatch.setattr(recognizer, "_text", lambda *a: [("对局开启中", 0.99, PixelRect(1050, 500, 300, 60))])
    assert recognizer.observe(frame).scene == "matching"
    rgb[600:710, 1300:1650] = (90, 210, 240)
    monkeypatch.setattr(recognizer, "_text", lambda *a: [("不抢", 0.99, PixelRect(1420, 640, 130, 45))])
    result = recognizer.observe(frame, {"frame_id": "one", "phase": "dealing"})
    assert result.scene == "bidding" and result.buttons[0].label == "不抢"


def test_stop_request_can_close_completed_window_without_renewing_input(run):
    run.runtime._complete("测试结束")
    (run.runtime.store.folder / "停止请求.json").write_text(
        json.dumps({"close_window": True}), encoding="utf8"
    )
    run.runtime.poll_stop_request()
    assert run.runtime.close_window_requested and run.runtime.stop_recording_requested
    assert run.gate.owner is None and not run.pipeline.flow_commands


def test_colored_readonly_bidding_feedback_is_not_a_button():
    from doudizhu_lab.domain.models import PixelRect

    image = np.full((1080, 2400, 3), (60, 110, 190), np.uint8)
    image[610:700, 1140:1270] = (230, 180, 20)
    frame = Frame("offline", image, 0, "one", 0, "file")
    assert not FlowRecognizer._colored(frame, PixelRect(1130, 600, 160, 110))
    image[600:710, 1300:1650] = (100, 220, 245)
    assert FlowRecognizer._colored(frame, PixelRect(1420, 640, 130, 45))


def test_own_readonly_feedback_confirms_preparation_but_is_not_clickable(monkeypatch):
    from doudizhu_lab.domain.models import PixelRect

    recognizer = FlowRecognizer.__new__(FlowRecognizer)
    frame = Frame(
        "offline", np.full((1080, 2400, 3), (60, 110, 190), np.uint8), 0, "one", 0, "file", frame_id="one"
    )
    monkeypatch.setattr(recognizer, "_text", lambda *a: [("不加倍", 0.99, PixelRect(1090, 610, 230, 100))])
    result = recognizer.observe(frame, {"frame_id": "one", "phase": "unknown"})
    assert result.scene == "doubling" and not result.buttons and result.feedbacks == ("decline_double",)


def test_entering_table_shell_confirms_scene_without_inventing_cards(monkeypatch):
    from doudizhu_lab.domain.models import PixelRect

    recognizer = FlowRecognizer.__new__(FlowRecognizer)
    frame = Frame("offline", np.full((1080, 2400, 3), 80, np.uint8), 0, "one", 0, "file", frame_id="one")
    words = [("斗地主规则", 0.99, PixelRect(250, 25, 250, 60)), ("托管", 0.99, PixelRect(2160, 25, 130, 60))]
    monkeypatch.setattr(recognizer, "_text", lambda *a: words)
    observed = recognizer.observe(frame)
    assert observed.scene == "dealing" and not observed.buttons
    monkeypatch.setattr(recognizer, "_text", lambda *a: words[1:])
    assert recognizer.observe(frame).scene == "unknown"


def test_winner_animation_scene_requires_same_frame_explicit_end_evidence(monkeypatch):
    recognizer = FlowRecognizer.__new__(FlowRecognizer)
    monkeypatch.setattr(recognizer, "_text", lambda *a: [])
    frame = Frame("offline", np.zeros((1080, 2400, 3), np.uint8), 0, "one", 0, "file", "one")
    table = {
        "frame_id": "one",
        "phase": "playing",
        "remaining": {"right": 0},
        "marker_evidence": {"winner_right": {"accepted": True}},
    }
    result = recognizer.observe(frame, table)
    assert result.scene == "settlement" and not result.buttons
    assert recognizer.observe(frame, table | {"frame_id": "old"}).scene == "unknown"
    assert recognizer.observe(frame, table | {"remaining": {"right": None}}).scene == "playing"


@pytest.mark.gui
def test_gui_continuous_pause_resume_stop_and_shutdown(qtbot, tmp_path, monkeypatch):
    from doudizhu_lab.ui.demo import ContinuousDemoWindow

    monkeypatch.setattr("doudizhu_lab.runtime.bot.Agent", lambda *a: LegalTestAgent())
    window = ContinuousDemoWindow(Store(tmp_path))
    qtbot.addWidget(window)
    window.show()
    qtbot.waitUntil(lambda: window.bot is not None and window.bot.ready.is_set(), timeout=5000)
    window.action_button.click()
    qtbot.waitUntil(lambda: window.continuous is not None and window.continuous.active, timeout=5000)
    runtime, pipeline = window.continuous, window.recording
    qtbot.waitUntil(lambda: len(pipeline.flow_commands) >= 1, timeout=5000)
    # 在匹配的无在途命令阶段暂停，确保恢复不靠清除未知提交。
    qtbot.waitUntil(lambda: runtime.pending_flow is None, timeout=5000)
    runtime.pause("测试暂停")
    before = len(pipeline.flow_commands)
    qtbot.wait(300)
    assert len(pipeline.flow_commands) == before
    qtbot.waitUntil(lambda: window.action_button.text() == "恢复任务")
    window.action_button.click()
    assert runtime.active
    window.stop_button.click()
    before = len(pipeline.flow_commands)
    qtbot.wait(300)
    assert len(pipeline.flow_commands) == before and runtime.status == "stopped"
    window.close()
    qtbot.waitUntil(lambda: not window.isVisible(), timeout=10000)
    assert not runtime.thread.is_alive() and not pipeline.thread.is_alive()
