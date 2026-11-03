"""叫抢决策、同帧按钮、回退及真实规则闭环；测试不连接手机。"""

import hashlib
import json
from dataclasses import asdict, replace
from time import monotonic
from types import SimpleNamespace

import numpy as np
import pytest

from doudizhu_lab.agents.bidding import (
    BID_MODEL_ID,
    BID_SHA256,
    BidAgent,
    BiddingController,
    default_bid_settings,
    encode_hand,
    verify_bid_model,
)
from doudizhu_lab.domain.models import Frame, PixelRect
from doudizhu_lab.domain.settings import parse_bid_threshold
from doudizhu_lab.runtime.bot import BotRuntime, InputGate
from doudizhu_lab.runtime.continuous import ContinuousRuntime
from doudizhu_lab.simulation.continuous import ContinuousSimPipeline
from doudizhu_lab.storage.config import Store, validate_bot_profile
from doudizhu_lab.storage.runs import evaluate_run
from doudizhu_lab.vision.flow import FlowButton, FlowObservation, FlowRecognizer, select_flow_source

HAND = ("3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A", "2", "2", "2", "SJ", "BJ")


class ScoreAgent:
    """仅供故障模拟的定值评分；正式离线报告使用固定真实权重。"""

    model_id = BID_MODEL_ID

    def __init__(self, score=1, error=""):
        self.value, self.error, self.inputs = score, error, []

    def start(self):
        if self.error:
            raise ValueError(self.error)

    def score(self, hand):
        self.inputs.append(tuple(hand))
        encode_hand(hand)
        return self.value, 1

    def interrupt(self):
        pass

    def close(self):
        pass


def flow(ms, *, prompt="call", hand=HAND, deal="deal-one"):
    frame = Frame(
        "simulation",
        np.zeros((100, 600, 3), np.uint8),
        ms / 1000,
        str(ms),
        0,
        "simulation",
        f"f-{ms}",
        "video",
        ms * 1000,
    )
    return FlowObservation(
        frame,
        "bidding",
        (
            FlowButton(
                "decline_bid", "不叫" if prompt == "call" else "不抢", PixelRect(100, 45, 90, 25), True
            ),
            FlowButton(
                "call_landlord" if prompt == "call" else "rob_landlord",
                "叫地主" if prompt == "call" else "抢地主",
                PixelRect(350, 45, 100, 25),
                True,
            ),
        ),
        bid_hand=hand,
        bid_deal_id=deal,
    )


def controller(agent=None):
    settings = default_bid_settings() | {"mode": "model"}
    return BiddingController(settings, agent or ScoreAgent(), threaded=False)


def test_encoder_matches_upstream_row_major_counts_and_jokers():
    data = np.array(encode_hand(HAND)).reshape(4, 15)
    assert data.sum() == 17 and list(data[:, 12]) == [1, 1, 1, 0]
    assert list(data[:, 13]) == [1, 0, 0, 0]
    assert list(data[:, 14]) == [1, 0, 0, 0]


@pytest.mark.parametrize(
    "bad", [HAND[:-1], HAND + ("3",), ("3",) * 17, ("SJ",) * 2 + HAND[2:], ("unknown",) + HAND[1:]]
)
def test_bidding_rejects_incomplete_or_impossible_hands(bad):
    with pytest.raises(ValueError):
        encode_hand(bad)


@pytest.mark.model
def test_fixed_model_scores_match_published_net2_and_never_randomly_fallback(tmp_path):
    folder = Store().root / "models/bidding"
    assert hashlib.sha256((folder / "bid_weights.pkl").read_bytes()).hexdigest() == BID_SHA256
    agent = BidAgent(folder)
    try:
        agent.start()
        score, elapsed = agent.score(HAND)
        assert score == pytest.approx(2.845994472503662, abs=1e-5) and elapsed < 1000
        assert len(agent.ready_info["parameter_shapes"]) == 12
    finally:
        agent.close()
    (tmp_path / "bid_weights.pkl").write_bytes(b"not weights")
    (tmp_path / "manifest.json").write_text(
        (folder / "manifest.json").read_text(encoding="utf8"), encoding="utf8"
    )
    with pytest.raises(ValueError, match="哈希"):
        verify_bid_model(tmp_path)


def test_only_distinct_consistent_frames_allow_call_and_reclaim_threshold():
    agent = ScoreAgent(0.55)
    # 显式保留旧版分阶段配置，验证默认值统一后仍兼容不同阈值。
    settings = default_bid_settings() | {"mode": "model", "call_threshold": 0.2, "reclaim_threshold": 0.6}
    c = BiddingController(settings, agent, threaded=False)
    assert c.choose(flow(0), "one") is None
    assert c.choose(flow(0), "one") is None
    call = c.choose(flow(120), "one")
    assert call.kind == "call_landlord" and call.threshold == 0.2
    c.committed(call, "one")
    assert c.choose(flow(240, prompt="rob"), "one") is None
    rob = c.choose(flow(360, prompt="rob"), "one")
    assert rob.kind == "decline_bid" and rob.stage == "reclaim" and rob.threshold == 0.6
    assert agent.inputs == [HAND, HAND]
    # 已选动作锁定，迟到评分不能在同一个机会改点另一个按钮。
    agent.value = 9
    assert c.choose(flow(480, prompt="rob"), "one") == rob
    c.close()


@pytest.mark.parametrize("stage", ["call", "rob", "reclaim"])
@pytest.mark.parametrize("score,positive", [(0.7, False), (0.75, False), (0.8, True)])
def test_custom_shared_threshold_controls_all_stages_with_strict_comparison(stage, score, positive):
    """真实决策器按自定义统一阈值比较，评分相等也拒绝；回抢经首叫提交反馈进入。"""
    settings = {"mode": "model", "call_threshold": 0.75, "rob_threshold": 0.75, "reclaim_threshold": 0.75}
    agent = ScoreAgent(1)
    c = BiddingController(settings, agent, threaded=False)
    try:
        if stage == "reclaim":
            assert c.choose(flow(0), "one") is None
            call = c.choose(flow(120), "one")
            assert call.kind == "call_landlord"
            c.committed(call, "one")
        agent.value = score
        prompt = "call" if stage == "call" else "rob"
        assert c.choose(flow(240, prompt=prompt), "one") is None
        decision = c.choose(flow(360, prompt=prompt), "one")
        assert decision.stage == stage and decision.threshold == 0.75 and decision.score == score
        expected = ("call_landlord" if stage == "call" else "rob_landlord") if positive else "decline_bid"
        assert decision.kind == expected
    finally:
        c.close()


@pytest.mark.parametrize(
    "text,expected",
    [
        ("-10", -10),
        ("10", 10),
        ("0", 0),
        (" 0.5 ", 0.5),
        ("1e-7", 0.0000001),
        ("0.12345678901234567", 0.12345678901234567),
    ],
)
def test_shared_threshold_parser_accepts_range_and_preserves_precision(text, expected):
    assert parse_bid_threshold(text) == expected


@pytest.mark.parametrize(
    "text", ["", " ", "abc", "零", "nan", "NaN", "inf", "-inf", "1e309", "-10.001", "10.001"]
)
def test_shared_threshold_parser_rejects_invalid_text_with_chinese_reason(text):
    with pytest.raises(ValueError, match="叫抢阈值必须为 -10～10 之间的有限数值"):
        parse_bid_threshold(text)


def test_changed_hand_resets_stability_and_waits_before_scoring():
    agent = ScoreAgent()
    c = controller(agent)
    c.choose(flow(0), "one")
    changed = ("3",) + HAND[1:-1] + ("3",)
    assert c.choose(flow(120, hand=changed), "one") is None and not agent.inputs
    assert c.choose(flow(240, hand=changed), "one").kind == "call_landlord"
    assert agent.inputs == [changed]
    c.close()


def test_unknown_hand_or_failed_model_declines_without_positive_click():
    c = controller()
    assert c.choose(flow(0, hand=None), "one") is None
    decision = c.choose(flow(3120, hand=None), "one")
    assert decision.kind == "decline_bid" and decision.score is None and "回退" in decision.reason
    c.close()
    c = controller(ScoreAgent(error="参数不兼容"))
    assert c.choose(flow(0), "one").kind == "decline_bid"
    c.close()


def test_positive_button_missing_cannot_be_guessed_from_model():
    c = controller()
    c.choose(flow(0), "one")
    missing = replace(flow(120), buttons=flow(120).buttons[:1])
    decision = c.choose(missing, "one")
    assert decision.kind == "decline_bid" and "未确认" in decision.reason
    c.close()


def test_cached_positive_decision_is_revoked_when_current_hand_changes():
    c = controller()
    c.choose(flow(0), "one")
    assert c.choose(flow(120), "one").kind == "call_landlord"
    refused = c.choose(flow(240, hand=HAND[:-1]), "one")
    assert refused.kind == "decline_bid" and refused.score is None
    c.close()


def test_flow_positive_buttons_require_background_and_same_frame_hand(monkeypatch):
    recognizer = FlowRecognizer.__new__(FlowRecognizer)
    frame = flow(0).frame
    rgb = np.full((1080, 2400, 3), 80, np.uint8)
    rgb[580:720, 800:1700] = (100, 210, 240)
    frame = replace(frame, rgb=rgb)
    words = [("不叫", 0.99, PixelRect(880, 620, 130, 45)), ("叫地主", 0.99, PixelRect(1400, 620, 180, 45))]
    monkeypatch.setattr(recognizer, "_words", lambda *a: words)
    observed = recognizer.observe(frame, {"frame_id": frame.frame_id, "phase": "bidding", "hand": list(HAND)})
    assert {b.kind for b in observed.buttons} == {"call_landlord", "decline_bid"}
    assert observed.bid_hand == HAND
    assert recognizer.observe(frame, {"frame_id": "old", "hand": list(HAND)}).bid_hand is None
    rgb[:] = 80
    assert not recognizer.observe(frame).buttons


def test_showcase_continue_closes_overlay_instead_of_starting_next_game(monkeypatch):
    recognizer = FlowRecognizer.__new__(FlowRecognizer)
    rgb = np.full((1080, 2400, 3), 80, np.uint8)
    rgb[940:1060, 1550:2280] = (100, 210, 240)
    frame = replace(flow(0).frame, rgb=rgb)
    words = [("继续", 0.99, PixelRect(1671, 977, 102, 57)), ("分享", 0.99, PixelRect(2000, 977, 100, 57))]
    monkeypatch.setattr(recognizer, "_words", lambda *a: words)
    result = recognizer.observe(frame, {"frame_id": frame.frame_id, "phase": "settlement"})
    assert result.scene == "showcase" and [b.kind for b in result.buttons] == ["close_showcase"]
    assert all(b.label != "分享" for b in result.buttons)
    monkeypatch.setattr(
        recognizer,
        "_words",
        lambda *a: [
            ("退出", 0.99, PixelRect(1697, 947, 101, 58)),
            ("继续游戏", 0.99, PixelRect(2004, 947, 191, 57)),
        ],
    )
    result = recognizer.observe(frame, {"frame_id": frame.frame_id, "phase": "settlement"})
    assert result.scene == "settlement" and {b.kind for b in result.buttons} == {"continue", "return_lobby"}


def test_known_footer_does_not_scan_large_showcase_or_need_phase_template(monkeypatch):
    recognizer = FlowRecognizer.__new__(FlowRecognizer)
    rgb = np.full((1080, 2400, 3), (100, 210, 240), np.uint8)
    frame = replace(flow(0).frame, rgb=rgb)
    words = [("继续", 0.99, PixelRect(1668, 973, 107, 64)), ("分享", 0.99, PixelRect(2000, 977, 100, 57))]
    calls = []

    def text(*args):
        calls.append(True)
        assert len(calls) == 1, "已确认按钮后不应继续扫描大海报"
        return words

    monkeypatch.setattr(recognizer, "_text", text)
    assert recognizer.observe(frame).scene == "showcase" and len(calls) == 1
    calls.clear()
    words[:] = [
        ("退出", 0.99, PixelRect(1697, 947, 101, 58)),
        ("继续游戏", 0.99, PixelRect(2004, 947, 191, 57)),
    ]
    assert recognizer.observe(frame).scene == "settlement" and len(calls) == 1


def test_confirmed_game_end_waits_for_buttons_without_inventing_page_progress(monkeypatch):
    recognizer = FlowRecognizer.__new__(FlowRecognizer)
    frame = replace(flow(0).frame, rgb=np.full((1080, 2400, 3), (100, 210, 240), np.uint8))
    calls, words = [], []

    def text(*args):
        calls.append(True)
        return words

    monkeypatch.setattr(recognizer, "_text", text)
    observed = recognizer.observe(frame, ended=True, deal_id="one")
    assert observed.scene == "ending" and not observed.buttons and len(calls) == 1
    # 展示页真实按钮出现后才确认showcase，期间没有伪造“结算页已打开”。
    words[:] = [("继续", 0.99, PixelRect(1668, 973, 107, 64)), ("分享", 0.99, PixelRect(2000, 977, 100, 57))]
    assert recognizer.observe(frame, ended=True, deal_id="one").scene == "showcase"
    words.clear()
    assert recognizer.observe(frame, ended=True, deal_id="one").scene == "ending"
    # 海报里的残留内容触发一个不完整的新局候选时，仍保留已确认的结算等待。
    calls.clear()
    assert recognizer.observe(frame, ended=False, deal_id="phantom-two").scene == "ending"
    assert len(calls) == 1
    calls.clear()
    assert (
        recognizer.observe(
            frame, {"frame_id": frame.frame_id, "phase": "playing"}, ended=False, deal_id="phantom-two"
        ).scene
        == "ending"
    )
    assert len(calls) == 1
    words[:] = [
        ("退出", 0.99, PixelRect(1697, 947, 101, 58)),
        ("继续游戏", 0.99, PixelRect(2004, 947, 191, 57)),
    ]
    assert recognizer.observe(frame, ended=True, deal_id="one").scene == "settlement"
    # 结算按钮已见过，随后可能进入大厅或匹配，不能继续限制为只扫描底部。
    words.clear()
    calls.clear()
    assert recognizer.observe(frame, ended=True, deal_id="one").scene == "unknown"
    assert len(calls) > 1


def source_snapshot(*, ended=False):
    """为源选择测试提供独立时间轴；未创建真实视频或手机接口。"""
    frame = flow(100_000).frame
    return SimpleNamespace(
        frame=frame,
        state={"game_id": "one", "ended": ended},
        observation={"frame_id": frame.frame_id, "hand": list(HAND)},
    )


def test_live_bidding_prefers_fresh_same_frame_table_for_seventeen_cards():
    snap = source_snapshot()
    decoded = replace(snap.frame, frame_id="decoded", started=100.15, media_pts_us=100_150_000)
    frame, table, ended, gid = select_flow_source(snap, decoded, now=100.2)
    assert frame is snap.frame and table is snap.observation
    assert table["frame_id"] == frame.frame_id and not ended and gid == "one"


def test_slow_table_analysis_uses_actual_decoded_frame_without_rebinding_old_hand():
    snap = source_snapshot()
    decoded = replace(snap.frame, frame_id="decoded", started=101.0, media_pts_us=101_000_000)
    frame, table, ended, _ = select_flow_source(snap, decoded, now=101.1)
    assert frame is decoded and table is None and not ended
    assert frame.started == 101.0 and snap.frame.started == 100.0


def test_game_end_page_reads_latest_decoded_frame_even_when_table_is_recent():
    snap = source_snapshot(ended=True)
    decoded = replace(snap.frame, frame_id="decoded", started=100.2, media_pts_us=100_200_000)
    frame, table, ended, gid = select_flow_source(snap, decoded, now=100.3)
    assert frame is decoded and table is None and ended and gid == "one"


def test_source_switch_drops_previous_device_game_hints_and_keeps_real_time():
    snap = source_snapshot(ended=True)
    decoded = replace(snap.frame, serial="other", stream_id="other", frame_id="decoded", started=101.0)
    frame, table, ended, gid = select_flow_source(snap, decoded, now=101.1)
    assert frame is decoded and table is None and not ended and not gid


@pytest.mark.parametrize(
    "key,value", [("call_threshold", float("nan")), ("rob_threshold", -1), ("mode", "random")]
)
def test_invalid_bid_settings_are_not_saved(tmp_path, key, value):
    profile = Store(tmp_path).profile
    profile.bid_settings[key] = value
    with pytest.raises(ValueError):
        validate_bot_profile(profile)


def test_old_configuration_remains_decline_and_is_backed_up_before_upgrade(tmp_path):
    store = Store(tmp_path)
    old = asdict(store.profile)
    old.pop("bid_settings")
    text = json.dumps({"version": 4, "profile": old})
    store.settings_path.write_text(text, encoding="utf8")
    loaded = Store(tmp_path)
    assert loaded.profile.bid_settings == {
        "mode": "disabled",
        "call_threshold": 0.5,
        "rob_threshold": 0.5,
        "reclaim_threshold": 0.5,
    }
    assert loaded.settings_path.read_text(encoding="utf8") == text
    loaded.save()
    assert (tmp_path / "settings.before-bidding.bak.json").read_text(encoding="utf8") == text


def test_actual_call_then_reclaim_assigns_landlord_and_finishes_via_production_executor(tmp_path):
    from tests.test_continuous import LegalTestAgent

    now = [monotonic()]

    def clock():
        return now[0]

    store = Store(tmp_path)
    store.profile.bid_settings["mode"] = "model"
    store.profile.continuous_policy = "balanced"
    store.profile.continuous_settings["max_games"] = 1
    pipeline = ContinuousSimPipeline(store, clock=clock)
    pipeline.start()
    gate = InputGate(clock=clock, transport="simulation")
    bot = BotRuntime(tmp_path, gate, clock=clock, agent=LegalTestAgent(), threaded=False)
    runtime = ContinuousRuntime(
        tmp_path, gate, bot, pipeline, clock=clock, threaded=False, bid_agent=ScoreAgent(2)
    )
    runtime.start()
    try:
        for _ in range(5000):
            now[0] += 0.12
            pipeline.tick()
            gate.heartbeat()
            bot.pulse()
            runtime.advance(pipeline.flow_snapshot(), pipeline.bot_snapshot())
            if not runtime.active:
                break
        assert runtime.status == "completed", runtime.reason
        assert pipeline.devices[0].roles["self"] == "landlord"
        assert len(pipeline.devices[0].initial["landlord"]) == 20
        assert not pipeline.device.invalid and not pipeline.device.duplicate
        commands = [kind for _, kind in pipeline.flow_commands]
        assert commands == ["match", "call_landlord", "rob_landlord", "decline_double"]
        report = evaluate_run(runtime.store.folder)
        assert not report["bidding_proof_errors"] and not report["unconfirmed_flow_actions"]
        before = list(pipeline.flow_commands)
        runtime.stop()
        runtime.advance(pipeline.flow_snapshot(), pipeline.bot_snapshot())
        assert pipeline.flow_commands == before
    finally:
        runtime.close()
        bot.close()
        pipeline.stop()
        pipeline.session.close()
