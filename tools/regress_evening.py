"""只读回放11月3日晚间故障窗口；原始语义前缀与冻结模板录像分别核验。"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from collections import Counter
from contextlib import closing
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np

from doudizhu_lab.agents.readiness import assess
from doudizhu_lab.domain.bot_types import Decision, ObservationSnapshot
from doudizhu_lab.domain.inference import frame_context, verify_proof
from doudizhu_lab.domain.models import Frame
from doudizhu_lab.domain.round_evidence import resolve_round_binding
from doudizhu_lab.domain.table import TableObservation
from doudizhu_lab.domain.tracker import GameState, GameTracker, apply_event
from doudizhu_lab.runtime.executor import ActionExecutor
from doudizhu_lab.runtime.operational import OperationalObserver
from doudizhu_lab.storage.replay import VideoReplay, profile_from_dict
from doudizhu_lab.storage.runs import audit_game_evidence
from tools.regress_afternoon import vision

SESSION = "20261103_205940_49d71a"


def read_source(folder: Path) -> dict:
    """原数据库只读打开并显式关闭；不使用会建表或更新评测文件的存储入口。"""
    with closing(
        sqlite3.connect((folder / "session.sqlite3").resolve().as_uri() + "?mode=ro", uri=True)
    ) as db:
        values = {
            table: [json.loads(r[0]) for r in db.execute(f"SELECT payload FROM {table} ORDER BY seq")]
            for table in ("events", "operational_events", "observations", "bot_actions")
        }
        values["profile"] = profile_from_dict(
            json.loads(db.execute("SELECT value FROM meta WHERE key='profile'").fetchone()[0])
        )
    return values


def _restore_tracker(profile, state, history, last_pts):
    tracker = GameTracker(profile.seat_order, config=profile.tracker_settings)
    tracker.state = deepcopy(state)
    tracker._role_initialized = tracker._active_started = True
    tracker._phase, tracker._actor = state.phase, state.current_actor
    tracker._counts, tracker._hand = dict(state.remaining), deepcopy(state.hand)
    tracker._last_pts = last_pts
    tracker._last_event_pts = history[-1]["pts_us"] if history else -1
    tracker._event_number = max((int(e["event_id"].rsplit("-e", 1)[1]) for e in history), default=0)
    moves = [e for e in history if e["kind"] in ("play", "pass")]
    for seat in state.seat_order:
        previous = next(((i, e) for i, e in reversed(list(enumerate(moves))) if e.get("seat") == seat), None)
        if previous and previous[1]["kind"] == "pass":
            index, event = previous
            tracker._presence[seat] = dict(
                first_us=event["pts_us"],
                last_us=last_pts,
                seen=2,
                consumed=True,
                candidate=None,
                label=event.get("data", {}).get("label", "pass"),
                consumed_turn=index + 1,
                absent=False,
            )
    return tracker


def replay_window(folder: Path, source: dict, gid: str, action_id: str, *, raw_only: bool = False) -> dict:
    """先重放原始开局至提交，再用实际录像重识别后八秒；任何发送入口都直接抛错。

    开局使用原 observations 和原 events 的实际确认时序，使两套牌局编号及原始确认
    迟到都真实进入生产 OperationalObserver。录像始终保留 replay 来源；仅恢复已记录
    的设备与编码流标识以核对原始证据归属，绝不把回放包装成实时输入授权。
    """
    profile = source["profile"]
    records = [a for a in source["bot_actions"] if a["action_id"] == action_id]
    decision_record = next(a for a in records if a["stage"] == "开始")
    submitted = next(a for a in records if a["stage"] == "准备输入" and a.get("kind") in ("play", "pass"))
    opening = next(
        e for e in source["operational_events"] if e["game_id"] == gid and e["kind"] == "game_start"
    )
    first = opening["data"]["first_evidence_us"]
    originals = {o["frame_id"]: o for o in source["observations"]}
    observer = OperationalObserver(profile)
    observer.tracker._game_number = int(gid[1:4]) - 1
    raw_state, raw_history, index = GameState(), [], 0
    raw_events = source["events"]
    pixels = np.zeros((profile.base_height, profile.base_width, 3), np.uint8)
    latest = None
    for obs in source["observations"]:
        if obs["pts_us"] > submitted["pts_us"]:
            break
        while index < len(raw_events):
            event = raw_events[index]
            confirmed = event.get("data", {}).get("confirmed_at_us") or event["pts_us"]
            if confirmed > obs["pts_us"]:
                break
            index += 1
            raw_state = apply_event(raw_state, event)
            if event["kind"] == "game_start":
                raw_history = []
            if event.get("confirmed", True):
                raw_history.append(event)
        if obs["pts_us"] < first:
            continue
        context = obs["frame_context"]
        frame = Frame(
            context["serial"],
            pixels,
            obs["pts_us"] / 1_000_000,
            str(obs["pts_us"]),
            0,
            "replay",
            obs["frame_id"],
            context["stream_id"],
            obs["pts_us"],
        )
        latest = observer.transform(ObservationSnapshot.create(frame, None, obs, raw_state, raw_history))
        observer.drain()
    if latest is None or observer.round_binding is None or latest.state["pending"]:
        raise ValueError("原始开局前缀未能建立唯一绑定或可信历史")
    decision = Decision(**decision_record["decision"])
    if decision.turn != latest.state["turn"]:
        raise ValueError("重放提交前的行动轮次与原实际提交不一致")
    decision = replace(decision, game_id=latest.state["game_id"])
    observer.note_submission(decision, submitted["pts_us"])
    tracker = _restore_tracker(profile, raw_state, raw_history, submitted["pts_us"])
    context = dict(observer.round_context)
    table, controls = vision(folder, profile)
    now = [submitted["pts_us"] / 1_000_000]
    audit = []

    def no_input(*args):
        raise AssertionError("晚间故障回放禁止发送任何设备输入")

    executor = ActionExecutor(no_input, audit.append, clock=lambda: now[0])
    executor.decision, executor.action_id = decision, action_id
    executor.stage, executor.submitted_at, executor.submitted_pts = "等待反馈", now[0], submitted["pts_us"]
    executor.before_hand = Counter(latest.state["hand"])
    executor.before_counts, executor.before_order = dict(latest.state["remaining"]), list(profile.seat_order)
    executor.identity = context["serial"], context["stream_id"], context["width"], context["height"]
    cursor = deepcopy(observer.tracker.state)
    proof_errors, inferred, seen, confirmation_time = [], [], 0, None
    ready_after, last_state = False, latest.state
    right_joker = False
    if raw_only:
        frames = (
            Frame(
                context["serial"],
                pixels,
                o["pts_us"] / 1_000_000,
                str(o["pts_us"]),
                0,
                "replay",
                o["frame_id"],
                context["stream_id"],
                o["pts_us"],
            )
            for o in source["observations"]
            if submitted["pts_us"] < o["pts_us"] <= submitted["pts_us"] + 8_000_000
        )
    else:
        frames = VideoReplay(folder).frames(
            fps=10,
            start_us=submitted["pts_us"] + 1,
            end_us=submitted["pts_us"] + 8_000_000,
            minimum_spacing=True,
        )
    try:
        for frame in frames:
            now[0] = frame.media_pts_us / 1_000_000
            frame = replace(frame, serial=context["serial"], stream_id=context["stream_id"], started=now[0])
            if raw_only:
                observation, recognition = TableObservation(**deepcopy(originals[frame.frame_id])), None
            else:
                observation, recognition = table.observe(frame, profile)
                controls.augment_observation(frame, observation, profile)
            right_joker |= observation.plays.get("right") == ["BJ"]
            update = tracker.observe(observation)
            for event in update.events:
                if event.kind == "game_start":
                    raw_history = []
                if event.confirmed:
                    raw_history.append(json.loads(json.dumps(event.__dict__)))
            observation.frame_context = frame_context(frame) | {"round_id": update.state.game_id}
            originals[frame.frame_id] = deepcopy(observation.__dict__)
            # allowed 仅表示本次反馈测试没有注入断流；执行器固定在只读等待反馈，
            # transport 仍是 replay，发送函数恒抛错，不存在真实设备输入权限。
            snap = ObservationSnapshot.create(
                frame, recognition, observation, update.state, raw_history, allowed=True
            )
            if recognition is not None:
                snap = controls.enrich(snap, profile)
            snap = observer.transform(snap)
            last_state = snap.state
            ready_after |= executor.stage == "已确认" and assess(snap).status == "ready"
            events, _ = observer.drain()
            for event in events:
                if event["source"] == "operational_inferred":
                    inferred.append(event)
                    if not verify_proof(event, cursor, originals, expected_context=context):
                        proof_errors.append(event["event_id"])
                cursor = apply_event(cursor, event)
            executor.advance(snap)
            if executor.stage == "已确认" and confirmation_time is None:
                confirmation_time = round(now[0] - executor.submitted_at, 3)
                observer.note_confirmation(decision)
            seen += 1
    finally:
        frames.close()
    binding = resolve_round_binding(observer.opening, source["events"], originals)
    sent_count = sum(
        a["stage"] in ("命令已返回", "命令结果未知")
        and a.get("kind") == submitted["kind"]
        and a.get("frame_id") == submitted["frame_id"]
        for a in records
    )
    return {
        "原牌局": gid,
        "实际提交编号": action_id,
        "检查类型": "原始观察完整时序重放" if raw_only else "原始开局语义时序及冻结模板录像故障窗口",
        "通过": executor.stage == "已确认"
        and not cursor.pending
        and not last_state["pending"]
        and not proof_errors
        and binding == observer.round_binding
        and sent_count == 1
        and ready_after
        and (right_joker or gid.startswith("g009")),
        "提交反馈": executor.stage,
        "反馈说明": executor.reason,
        "反馈耗时秒": confirmation_time,
        "窗口帧数": seen,
        "窗口后历史疑点": list(cursor.problems),
        "后续本人回合恢复就绪": ready_after,
        "原账本实际提交次数": sent_count,
        "证明错误": proof_errors,
        "有效推断数": len(inferred) - len(proof_errors),
        "大王直接识别": right_joker,
        "设备输入数": 0,
        "原始分局": observer.round_binding.raw_game_id,
        "运行分局": observer.round_binding.game_id,
    }


def main() -> int:
    """结果只写独立路径；原数据库、原报告、媒体索引及留存决定用哈希确认未改写。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True, help="本机原始资料目录，只读")
    parser.add_argument("--output", type=Path, required=True, help="独立中文回归报告")
    args = parser.parse_args()
    root, output = args.data_root.resolve(), args.output.resolve()
    if any(output.is_relative_to(root / name) for name in ("sessions", "runs")):
        parser.error("回归输出不得指向原会话或任务目录")
    folder = root / "sessions" / SESSION
    protected = [folder / name for name in ("session.sqlite3", "video/index.json", "留存记录.json")]
    protected += [
        root / "runs" / name / item
        for name in ("20261103_205940_df3d5c", "20261103_205942_5a10b4")
        for item in ("run.sqlite3", "连续任务报告.json")
    ]

    def hashes():
        return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}

    before = hashes()
    source = read_source(folder)
    checks = [
        replay_window(folder, source, gid, ident)
        for gid, ident in (
            ("g003-f557ec2d", "e7944d050d5249feb6353c8fa6ba2f2b"),
            ("g009-c18e781d", "c6a2031c45bf49fc818bdbf1c8b8dc44"),
        )
    ]
    checks.append(
        replay_window(folder, source, "g009-c18e781d", "c6a2031c45bf49fc818bdbf1c8b8dc44", raw_only=True)
    )
    observations = {o["frame_id"]: o for o in source["observations"]}
    for gid in ("g008-ee3ccc40", "g028-5a8c430f"):
        audit = audit_game_evidence(
            {"game_id": gid},
            source["operational_events"],
            source["events"],
            observations,
            source["bot_actions"],
            policy="balanced",
        )
        checks.append(
            {
                "原牌局": gid,
                "检查类型": "原始观察与账本的独立证明复核，非录像重识别",
                "通过": audit["passed"],
                "证明错误": audit["inference_proof_errors"],
                "提交错误": audit["submission_proof_errors"],
            }
        )
    report = {
        "通过": all(c["通过"] for c in checks) and before == hashes(),
        "原始资料未改写": before == hashes(),
        "检查": checks,
        "源文件哈希": before,
        "真机实测": "本轮未启动真机对局",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return 0 if report["通过"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
