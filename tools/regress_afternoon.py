"""只读重放11月3日下午故障动作；使用会话冻结模板，结果单独保存，不连接手机。"""

import argparse
import json
import sqlite3
from collections import Counter
from contextlib import closing
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from xml.etree import ElementTree

from doudizhu_lab.agents.douzero import TO_RANK
from doudizhu_lab.domain.bot_types import Decision, ObservationSnapshot
from doudizhu_lab.domain.inference import frame_context, verify_proof
from doudizhu_lab.domain.models import Frame
from doudizhu_lab.domain.tracker import GameState, GameTracker, apply_event, replay_events
from doudizhu_lab.runtime.executor import ActionExecutor
from doudizhu_lab.runtime.operational import OperationalObserver
from doudizhu_lab.storage.config import read_rgb
from doudizhu_lab.storage.replay import VideoReplay, profile_from_dict
from doudizhu_lab.vision.cards import Recognizer
from doudizhu_lab.vision.controls import ControlRecognizer
from doudizhu_lab.vision.flow import FlowRecognizer
from doudizhu_lab.vision.table import MarkerLibrary, TableRecognizer


def read_session(folder):
    """只读打开现有SQLite；所有连接显式关闭，不触发建表、迁移或原报告更新。"""
    with closing(
        sqlite3.connect((folder / "session.sqlite3").resolve().as_uri() + "?mode=ro", uri=True)
    ) as db:
        profile = profile_from_dict(
            json.loads(db.execute("SELECT value FROM meta WHERE key='profile'").fetchone()[0])
        )
        actions = [json.loads(p) for (p,) in db.execute("SELECT payload FROM bot_actions ORDER BY seq")]
        events = [json.loads(p) for (p,) in db.execute("SELECT payload FROM operational_events ORDER BY seq")]
    return profile, actions, events


def vision(folder, profile):
    """仅加载该会话冻结的牌点、标记与按钮模板，禁止回退到当前用户模板。"""
    return (
        TableRecognizer(
            Recognizer(folder / "templates/templates", include_bundled=False),
            MarkerLibrary(folder / "templates/markers"),
        ),
        ControlRecognizer(folder / "templates/controls"),
    )


def replay_action(folder, ident, *, check_following_history=False):
    """以已确认的公开前缀为起点，按原媒体时间重放提交后的八秒反馈窗口。

    只调用生产识别、运行观察器及等待反馈分支。发送接口恒抛异常，离线帧保持replay来源；
    模型建议仅用于核对原提交意图，不参与推导。此检查不是整局重新执行或真机成绩。
    """
    profile, actions, events = read_session(folder)
    records = [a for a in actions if a["action_id"] == ident]
    start = next(a for a in records if a["stage"] == "开始")
    submitted = next(a for a in records if a["stage"] == "准备输入" and a.get("kind") in ("play", "pass"))
    decision = Decision(**start["decision"])
    prefix = [
        e
        for e in events
        if e["game_id"] == decision.game_id
        and (e.get("data", {}).get("confirmed_at_us") or e["pts_us"]) <= start["pts_us"]
    ]
    before = GameState()
    for event in prefix:
        before = apply_event(before, event)
    observer = OperationalObserver(profile)
    tracker = observer.tracker
    tracker.state = deepcopy(before)
    tracker._role_initialized = tracker._active_started = True
    tracker._phase, tracker._actor = before.phase, before.current_actor
    tracker._counts, tracker._hand = dict(before.remaining), deepcopy(before.hand)
    tracker._last_pts = submitted["pts_us"] - 1
    tracker._last_event_pts = prefix[-1]["pts_us"] if prefix else -1
    tracker._event_number = max((int(e["event_id"].rsplit("-e", 1)[1]) for e in prefix), default=0)
    # 恢复可信前缀中已消费的不出提示。窗口回放不能把仍留在屏幕上的旧字样
    # 当成一个新的越轮动作，否则会人为制造原运行状态里不存在的三秒超时。
    moves = [e for e in prefix if e["kind"] in ("play", "pass")]
    for seat in ("self", "left", "right"):
        previous = next(((i, e) for i, e in reversed(list(enumerate(moves))) if e.get("seat") == seat), None)
        if previous and previous[1]["kind"] == "pass":
            index, event = previous
            tracker._presence[seat] = {
                "first_us": event["pts_us"],
                "last_us": submitted["pts_us"],
                "seen": 2,
                "consumed": True,
                "candidate": None,
                "label": event.get("data", {}).get("label", "pass"),
                "consumed_turn": index + 1,
                "absent": False,
            }
    observer.history = deepcopy(prefix)
    table, controls = vision(folder, profile)
    now = [submitted["pts_us"] / 1_000_000]
    audit = []

    def no_input(*args):
        raise AssertionError("故障回放禁止发送任何输入")

    executor = ActionExecutor(no_input, audit.append, clock=lambda: now[0])
    executor.decision, executor.action_id = decision, ident
    executor.stage, executor.submitted_at = "等待反馈", now[0]
    executor.submitted_pts = submitted["pts_us"]
    data = start["input_context"]
    executor.before_hand = Counter(TO_RANK[n] for n in data["player_hand_cards"])
    executor.before_counts = dict(data.get("remaining", before.remaining))
    executor.before_order = list(profile.seat_order)
    executor.rule_feedback = executor.confirmation_evidence = None
    originals, inferred, seen = {}, [], 0
    cursor = deepcopy(before)
    proof_errors = []
    frames = VideoReplay(folder).frames(
        fps=10, start_us=submitted["pts_us"] + 1, end_us=submitted["pts_us"] + 8_000_000, minimum_spacing=True
    )
    try:
        for frame in frames:
            now[0] = frame.media_pts_us / 1_000_000
            frame = replace(frame, started=now[0])
            observation, recognition = table.observe(frame, profile)
            controls.augment_observation(frame, observation, profile)
            observation.frame_context = frame_context(frame) | {"round_id": before.game_id}
            originals[frame.frame_id] = json.loads(json.dumps(observation.__dict__, ensure_ascii=False))
            snap = ObservationSnapshot.create(frame, recognition, observation, before, prefix, allowed=True)
            if decision.strategy != "rules":
                snap = observer.transform(snap)
                new, _ = observer.drain()
                for event in new:
                    if event["source"] == "operational_inferred":
                        inferred.append(event)
                        if not verify_proof(event, cursor, originals):
                            proof_errors.append(event["event_id"])
                    cursor = apply_event(cursor, event)
            if not seen:
                executor.identity = frame.serial, frame.stream_id, frame.width, frame.height
            executor.advance(snap)
            seen += 1
            if not executor.active and not check_following_history:
                break
    finally:
        frames.close()
    return {
        "动作编号": ident,
        "原决策": list(decision.cards),
        "策略": decision.strategy,
        "通过": executor.stage == "已确认"
        and not proof_errors
        and (not check_following_history or not cursor.pending),
        "阶段": executor.stage,
        "说明": executor.reason,
        "媒体帧数": seen,
        "确认耗时秒": round(now[0] - executor.submitted_at, 3),
        "反馈依据": executor.confirmation_evidence,
        "推断事件数": len(inferred),
        "证明错误": proof_errors,
        "后续历史检查": list(cursor.problems) if check_following_history else None,
        "检查类型": "原媒体时序的提交后反馈窗口回放",
    }


def image_case(root, session, fid, *, bottom=None, hand=None, selection=False):
    """同一故障源图独立重识别；不从预设答案生成模板或改变识别输入。"""
    folder = root / "sessions" / session
    profile, _, _ = read_session(folder)
    table, controls = vision(folder, profile)
    frame = Frame(
        "replay", read_rgb(folder / "evidence" / (fid + ".png")), 0, fid, 0, "replay", fid, session, 0
    )
    obs, recognition = table.observe(frame, profile)
    passed = (bottom is None or obs.bottom == bottom) and (hand is None or obs.hand == hand)
    reason = ""
    if selection:
        snap = controls.enrich(ObservationSnapshot.create(frame, recognition, obs, GameState(), []), profile)
        passed = (
            passed
            and not snap.control_reason
            and len(snap.hand) == 20
            and all(c.selected is False for c in snap.hand)
        )
        reason = snap.control_reason
    return {
        "会话": session,
        "源帧": fid,
        "通过": bool(passed),
        "底牌": obs.bottom,
        "手牌": obs.hand,
        "说明": reason,
        "检查类型": "冻结模板与原始截图",
    }


def replay_continue(root):
    """复核15:58续局后出现的普通切场确认；只读OCR不执行确认按钮。"""
    folder = root / "sessions/20261103_153036_3918af"
    recognizer = FlowRecognizer(folder)
    submitted = 74743116617
    frames = VideoReplay(folder).frames(fps=2, start_us=submitted + 1, end_us=submitted + 8_000_000)
    result = {"通过": False, "说明": "八秒内未看到可信新页面"}
    try:
        for frame in frames:
            flow = recognizer.observe(frame)
            if flow.blocked_reason:
                result["说明"] = flow.blocked_reason
                break
            if flow.scene == "room_confirmation" and any(
                b.enabled and b.kind == "confirm_room" for b in flow.buttons
            ):
                result = {
                    "通过": True,
                    "阶段": flow.scene,
                    "确认耗时秒": round((frame.media_pts_us - submitted) / 1e6, 3),
                    "说明": "出现明确普通切场确认页；页面协调器推进由对应故障注入用例复核",
                }
                break
    finally:
        frames.close()
    return result | {"检查类型": "冻结OCR与原续局视频窗口", "动作编号": "01984a8ac0854998b675667149706df9"}


def replay_earlier_opening(root):
    """从12:51原录像的叫抢阶段重建开局，不采用原运行时补全结果或当前模板。"""
    folder = root / "sessions/20261103_124911_608316"
    profile, _, _ = read_session(folder)
    table, controls = vision(folder, profile)
    raw_tracker = GameTracker(profile.seat_order, config=profile.tracker_settings)
    observer = OperationalObserver(profile)
    originals, errors, inferred = {}, [], []
    cursor, result, seen = GameState(), None, 0
    frames = VideoReplay(folder).frames(
        fps=10, start_us=63478000000, end_us=63494000000, minimum_spacing=True
    )
    try:
        for frame in frames:
            observation, recognition = table.observe(frame, profile)
            controls.augment_observation(frame, observation, profile)
            raw = raw_tracker.observe(observation)
            observation.frame_context = frame_context(frame) | {"round_id": raw.state.game_id}
            originals[frame.frame_id] = json.loads(json.dumps(observation.__dict__, ensure_ascii=False))
            snap = ObservationSnapshot.create(frame, recognition, observation, raw.state, [])
            result = observer.transform(snap)
            events, _ = observer.drain()
            for event in events:
                if event["source"] == "operational_inferred":
                    inferred.append(event)
                    if not verify_proof(event, cursor, originals, expected_context=observer.round_context):
                        errors.append(event["event_id"])
                cursor = apply_event(cursor, event)
            seen += 1
            if (
                result.state["bottom"]
                and len(result.state["hand"] or []) == 20
                and result.state["landlord"] == "self"
            ):
                break
    finally:
        frames.close()
    state = result.state if result else {}
    return {
        "会话": folder.name,
        "通过": bool(
            state.get("bottom")
            and Counter(state["bottom"]) == Counter(["J", "BJ", "2"])
            and not state.get("pending")
            and not errors
        ),
        "底牌": state.get("bottom"),
        "媒体帧数": seen,
        "推断事件数": len(inferred),
        "证明错误": errors,
        "检查类型": "更早底牌漏王的冻结模板与原媒体开局重建",
    }


def verify_legacy_gaps(root):
    """早期窗口已按原留存策略清理，只读重放原事件确认旧缺口不会被新版口径改成成功。"""
    folder = root / "sessions/20261103_123813_604b40"
    with closing(sqlite3.connect((folder / "session.sqlite3").as_uri() + "?mode=ro", uri=True)) as db:
        events = [json.loads(p) for (p,) in db.execute("SELECT payload FROM events ORDER BY seq")]
    states = replay_events(events)
    expected = ("g001-8467c889", "g002-fccbcfce")
    return {
        "会话": folder.name,
        "通过": all(states.get(gid, {}).get("pending") and not states[gid]["complete"] for gid in expected),
        "检查类型": "早期历史缺口的只读原事件回放",
        "说明": "缺口仍隔离，旧成绩不升级；部分原视频已被原留存策略清理，此项不算像素重识别。",
        "未完成牌局": list(expected),
    }


def coverage(root, checks, public_tests):
    """逐项关联原父账本18次异常；媒体回放和故障注入分别标注，不混称整局重跑。"""
    source = root / "runs/20261103_133054_66f554/run.sqlite3"
    with closing(sqlite3.connect(source.as_uri() + "?mode=ro", uri=True)) as db:
        skipped = [
            json.loads(p) for (p,) in db.execute("SELECT payload FROM task_events WHERE kind='game_skipped'")
        ]
    tests = (
        list(ElementTree.parse(public_tests).iter("testcase"))
        if public_tests and public_tests.is_file()
        else []
    )

    def passed(name):
        matches = [t for t in tests if t.get("name", "").split("[", 1)[0] == name]
        return bool(matches) and all(
            not list(t) or all(c.tag not in ("failure", "error", "skipped") for c in t) for t in matches
        )

    by_action = {c.get("动作编号"): c for c in checks if c.get("动作编号")}
    by_frame = {c.get("源帧"): c for c in checks if c.get("源帧")}
    media = {
        "g012-5ee59021": by_frame.get("3017c974-22943"),
        "g004-1be63bca": by_frame.get("a972d490-5124"),
        "g006-09437466": by_frame.get("a972d490-9251"),
        "g001-287939c3": by_frame.get("4d7a26b0-3004"),
        "g011-b49b4105": by_action.get("efffa79f20ca406fad5af3fc52a7fd7f"),
        "g011-0101387f": by_action.get("f753bc580ef24b33ba9eb90fc232c5a0"),
        "g014-f670eb70": by_action.get("b27562c32d5a42a0800e15b4f4452c08"),
        "g001-a23a1412": by_action.get("b346e3e0abb441a09e4bce1671af1bba"),
        "g006-cb4f56c1": by_action.get("8b05e400ec6b447987058d3e443ad13c"),
        "g016-3664f2f2": by_frame.get("4d7a26b0-25398"),
        "g017-7d33838e": by_frame.get("4d7a26b0-29189"),
    }
    result = []
    for item in skipped:
        proof = media.get(item.get("game_id"))
        if proof is not None:
            evidence, ok = proof["检查类型"], proof["通过"]
        else:
            name = (
                "test_page_stale_rejection_reacquires_evidence_without_rebuilding"
                if "输入未发送" in item["reason"]
                else "test_stale_flow_results_wait_eight_seconds_while_video_is_fresh"
                if item["code"] == "runtime_failure"
                else "test_external_wait_uses_light_probes_and_resumes_same_device"
            )
            evidence, ok = "故障注入用例：" + name, passed(name)
        result.append({"原异常": item, "验证方式": evidence, "通过": bool(ok)})
    return result


def main():
    """显式指定本机数据根和独立输出文件；不会扫描其他数据目录或更新历史任务报告。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True, help="包含私人会话的本机数据根")
    parser.add_argument("--output", type=Path, required=True, help="独立回归JSON报告")
    parser.add_argument(
        "--public-tests", type=Path, help="公共测试的JUnit XML，用于关联时间及外部条件故障注入"
    )
    args = parser.parse_args()
    root = args.data_root.resolve()
    if any(args.output.resolve().is_relative_to(root / category) for category in ("sessions", "runs")):
        parser.error("输出必须位于历史sessions/runs目录之外")
    checks = []
    images = [
        ("20261103_133552_594545", "3017c974-22943", {"bottom": ["J", "9", "BJ"]}),
        ("20261103_143626_6cbfcd", "a972d490-5124", {"bottom": ["5", "BJ", "9"]}),
        ("20261103_143626_6cbfcd", "a972d490-9251", {"bottom": ["8", "BJ", "4"]}),
        ("20261103_145717_5db1ff", "4d7a26b0-3004", {"bottom": ["Q", "8", "BJ"]}),
        ("20261103_145717_5db1ff", "4d7a26b0-25398", {"selection": True}),
        ("20261103_153036_3918af", "569e43a5-9727", {"hand": ["9"]}),
        ("20261103_145717_5db1ff", "4d7a26b0-29189", {"hand": ["10"]}),
    ]
    for session, fid, expected in images:
        result = image_case(root, session, fid, **expected)
        checks.append(result)
        print(json.dumps({"源帧": fid, "通过": result["通过"]}, ensure_ascii=False), flush=True)
    action_cases = [
        ("20261103_133552_594545", "efffa79f20ca406fad5af3fc52a7fd7f"),
        ("20261103_145717_5db1ff", "f753bc580ef24b33ba9eb90fc232c5a0"),
        ("20261103_145717_5db1ff", "b27562c32d5a42a0800e15b4f4452c08"),
        ("20261103_153036_3918af", "b346e3e0abb441a09e4bce1671af1bba"),
        ("20261103_153036_3918af", "8b05e400ec6b447987058d3e443ad13c"),
        ("20261103_130703_987f1c", "973a7260b03e492890860a027c5b4514"),
    ]
    for session, ident in action_cases:
        try:
            result = replay_action(root / "sessions" / session, ident)
        except (OSError, ValueError, KeyError, AssertionError) as exc:
            result = {"动作编号": ident, "通过": False, "说明": str(exc)}
        checks.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    checks.append(replay_continue(root))
    checks.append(replay_earlier_opening(root))
    checks.append(verify_legacy_gaps(root))
    rows = coverage(root, checks, args.public_tests)
    report = {
        "日期": "2026-11-03",
        "源资料只读": True,
        "通过": all(c["通过"] for c in checks) and all(r["通过"] for r in rows),
        "检查": checks,
        "原18次异常覆盖": rows,
        "边界": "仅验证历史故障图与提交反馈窗口，不是整局重新执行或真机连续验收。",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf8")
    return 0 if report["通过"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
