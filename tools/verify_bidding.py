"""只读核对已停止的真机叫抢任务；重算评分并验证本人地主的17→20、首次领出及结算。

运行：uv run python tools/verify_bidding.py data/runs/<任务> --require-pass
本工具不创建ADB或Qt对象、不补写观察和出牌事件，只在任务目录导出独立核验报告。
"""

import argparse
import hashlib
import json
import sqlite3
from collections import Counter
from pathlib import Path

from doudizhu_lab.agents.bidding import BidAgent
from doudizhu_lab.agents.douzero import TO_RANK
from doudizhu_lab.domain.tracker import replay_events
from doudizhu_lab.storage.runs import evaluate_run


def rows(path, table):
    """以SQLite只读连接取得既有JSON行；表名只由本文件中的常量提供。"""
    with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as database:
        return [json.loads(v[0]) for v in database.execute(f"SELECT payload FROM {table} ORDER BY seq")]


def verify(folder):
    """使用会话快照重新评分，独立核对已完成本人地主局；缺少证据时如实返回未通过。"""
    folder = Path(folder)
    path = folder / "run.sqlite3"
    with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as database:
        metadata = {k: json.loads(v) for k, v in database.execute("SELECT key,payload FROM meta")}
        journal = [
            (k, json.loads(v)) for k, v in database.execute("SELECT kind,payload FROM journal ORDER BY seq")
        ]
    session = Path(metadata["session"])
    observations = rows(session / "session.sqlite3", "observations")
    original_hash = hashlib.sha256(json.dumps(observations, sort_keys=True).encode()).hexdigest()
    events = rows(session / "session.sqlite3", "events")
    actions = rows(session / "session.sqlite3", "bot_actions")
    checks = evaluate_run(folder)
    errors, scores = [], []
    decisions = [v for k, v in journal if k == "bidding_decision"]
    agent = BidAgent(session / "models/bidding")
    try:
        agent.start()
        for decision in decisions:
            if decision["score"] is None:
                continue
            actual, _ = agent.score(decision["hand"])
            difference = abs(actual - decision["score"])
            scores.append(
                {
                    "opportunity": decision["opportunity"],
                    "kind": decision["kind"],
                    "stage": decision["stage"],
                    "score": actual,
                    "recorded_score": decision["score"],
                    "difference": difference,
                    "threshold": decision["threshold"],
                    "elapsed_ms": decision["elapsed_ms"],
                    "frame_id": decision["frame_id"],
                    "pts_us": decision["pts_us"],
                }
            )
            if difference > 1e-5 or decision["model_id"] != agent.model_id:
                errors.append("评分无法使用本会话权重复现：" + decision["opportunity"])
    except (ValueError, OSError, RuntimeError) as exc:
        errors.append(f"会话叫抢模型无法重算：{exc}")
    finally:
        agent.close()
    states = replay_events(events)
    complete = {v["game_id"]: v for k, v in journal if k == "game_complete"}
    audits = {v["game_id"]: v["passed"] for v in checks["game_audits"]}
    starts = {v["action_id"]: v for v in actions if v["stage"] == "开始"}
    confirmed = {v["action_id"] for v in actions if v["stage"] == "已确认"}
    own_games = []
    for gid, state in states.items():
        if state["landlord"] != "self" or gid not in complete:
            continue
        initial, bottom = state["initial_hand"] or [], state["bottom"] or []
        bid = next(
            (
                v
                for v in reversed(decisions)
                if v["opportunity"].startswith(gid + ":") and v["kind"] in ("call_landlord", "rob_landlord")
            ),
            None,
        )
        first = next((v for v in starts.values() if v["game_id"] == gid and v["decision"]["turn"] == 0), None)
        moves = [e for e in events if e["game_id"] == gid and e["kind"] in ("play", "pass")]
        full_frames = [
            o
            for o in observations
            if o["landlord"] == "self"
            and Counter(o["hand"] or []) == Counter(initial)
            and Counter(o["bottom"] or []) == Counter(bottom)
            and complete[gid]["start_us"] <= o["pts_us"] <= complete[gid]["end_us"]
        ]
        full_stable = (
            len(full_frames) >= 2 and full_frames[-1]["pts_us"] - full_frames[0]["pts_us"] >= 100_000
        )
        first_ok = bool(
            first
            and moves
            and moves[0]["kind"] == "play"
            and moves[0]["seat"] == "self"
            and Counter(moves[0]["cards"]) == Counter(first["decision"]["cards"])
            and first["decision"]["role"] == "landlord"
            and first["action_id"] in confirmed
            and Counter(TO_RANK[n] for n in first["input_context"]["player_hand_cards"]) == Counter(initial)
        )
        conservation = bool(
            bid
            and len(initial) == 20
            and len(bottom) == 3
            and Counter(bid["hand"]) + Counter(bottom) == Counter(initial)
        )
        item = {
            "game_id": gid,
            "landlord": "self",
            "bid_kind": bid["kind"] if bid else None,
            "initial_hand": initial,
            "bottom": bottom,
            "hand_conservation": conservation,
            "twenty_card_frames": [o["frame_id"] for o in full_frames[:2]],
            "twenty_card_pts_us": [o["pts_us"] for o in full_frames[:2]],
            "twenty_cards_stable": full_stable,
            "first_play": moves[0]["cards"] if moves else None,
            "first_play_confirmed": first_ok,
            "first_action_id": first["action_id"] if first else None,
            "ended": state["ended"],
            "complete": state["complete"],
            "own_actions": complete[gid]["own_actions"],
            "confirmed_actions": complete[gid]["confirmed_actions"],
            "start_us": complete[gid]["start_us"],
            "end_us": complete[gid]["end_us"],
            "passed": conservation
            and full_stable
            and first_ok
            and state["ended"]
            and state["complete"]
            and audits.get(gid, False),
        }
        own_games.append(item)
    after = rows(session / "session.sqlite3", "observations")
    unchanged = original_hash == hashlib.sha256(json.dumps(after, sort_keys=True).encode()).hexdigest()
    passed = bool(
        metadata["source"] == "live_continuous"
        and scores
        and not errors
        and own_games
        and all(g["passed"] for g in own_games)
        and not checks["bidding_proof_errors"]
        and not checks["unconfirmed_flow_actions"]
        and not checks["duplicate_submissions"]
        and not checks["capture_issues"]
        and checks["video_closed"]
        and unchanged
    )
    result = {
        "source": "live_bidding_audit",
        "scope": "叫抢及本人地主单局链路；连续任务整体结果单列",
        "run_status": metadata["status"],
        "run": str(folder.resolve()),
        "session": str(session),
        "passed": passed,
        "scores": scores,
        "score_errors": errors,
        "actions": checks["bidding_actions"],
        "fallbacks": checks["bidding_fallbacks"],
        "own_landlord_games": own_games,
        "whole_run_passed": checks["passed"],
        "raw_observations_unchanged": unchanged,
        "raw_observation_hash": original_hash,
        "note": "验证现成模型评分和本人地主执行链路；不作为叫抢最优收益或对战胜率结论。",
    }
    target = folder / "叫抢及本人地主核验.json"
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf8")
    return result


def main():
    """核验不足返回2；无论是否通过均保存报告，不覆盖原始观察或人工修订。"""
    parser = argparse.ArgumentParser(description="只读核对真机叫抢和本人地主链路")
    parser.add_argument("folder", type=Path, help="已停止的任务目录")
    parser.add_argument("--require-pass", action="store_true")
    args = parser.parse_args()
    result = verify(args.folder)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.require_pass and not result["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
