"""连续任务账本和来源明确的完成评测；流程意图与牌局事实使用不同数据库。"""

import json
import sqlite3
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from types import SimpleNamespace
from uuid import uuid4


class RunStore:
    """线程安全的追加账本；当前状态可更新，操作、暂停与完成证据只追加。"""

    def __init__(self, folder):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.lock = RLock()
        self.db = sqlite3.connect(self.folder / "run.sqlite3", check_same_thread=False)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY,payload TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS journal(seq INTEGER PRIMARY KEY,kind TEXT,payload TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS observations(seq INTEGER PRIMARY KEY,frame_id TEXT UNIQUE,payload TEXT NOT NULL);
        """)
        self.db.commit()

    @classmethod
    def create(cls, root, pipeline):
        """从已启动录像创建任务；保存对应会话、策略及版本，不保存可恢复的自动授权。"""
        ident = datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid4().hex[:6]
        result = cls(Path(root) / "runs" / ident)
        source = (
            "simulation_continuous"
            if getattr(pipeline, "bot_source", "") == "simulation_bot"
            else "live_continuous"
        )
        for key, value in {
            "run_id": ident,
            "source": source,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "session": str(pipeline.session.folder.resolve()),
            "profile": asdict(pipeline.profile),
            "policy": "模型叫抢、不加倍"
            if pipeline.profile.bid_settings["mode"] == "model"
            else "不叫、不抢、不加倍",
            "status": "running",
            "configuration_version": 4,
        }.items():
            result.set(key, value)
        return result

    def set(self, key, value):
        """写入任务状态或配置；不得把状态更新当作已确认的出牌证据。"""
        with self.lock, self.db:
            self.db.execute(
                "INSERT OR REPLACE INTO meta VALUES (?,?)", (key, json.dumps(value, ensure_ascii=False))
            )

    def add(self, event_kind, **value):
        """追加一次流程证据，保留源帧、动作机会、发送及反馈时间供独立核对。"""
        with self.lock, self.db:
            self.db.execute(
                "INSERT INTO journal(kind,payload) VALUES (?,?)",
                (event_kind, json.dumps(value, ensure_ascii=False)),
            )

    def observation(self, flow):
        """保存语义观察，不把RGB写进SQLite；源画面由会话录像和证据队列保留。"""
        value = {
            "frame_id": flow.frame.frame_id,
            "pts_us": flow.frame.media_pts_us,
            "scene": flow.scene,
            "buttons": [asdict(b) for b in flow.buttons],
            "texts": flow.texts,
            "blocked_reason": flow.blocked_reason,
            "mode_verified": flow.mode_verified,
            "feedbacks": flow.feedbacks,
            "bid_hand": flow.bid_hand,
            "bid_markers": flow.bid_markers,
            "bid_deal_id": flow.bid_deal_id,
        }
        with self.lock, self.db:
            self.db.execute(
                "INSERT OR IGNORE INTO observations(frame_id,payload) VALUES (?,?)",
                (flow.frame.frame_id, json.dumps(value, ensure_ascii=False)),
            )

    def close(self):
        """在所有输入与后台线程停止后关闭；任务授权始终不会从数据库恢复。"""
        with self.lock:
            self.db.close()


def uncovered_video_intervals(segments, start_us, end_us):
    """返回牌局时间范围内未被录像覆盖的区间，容忍正常分段之间不超过250ms的帧间隔。

    首尾允许2ms的MKV时间基量化误差。容器能完整解码不代表开局以前的画面曾被保存，
    因此评测必须另外核对这个时间覆盖；缺少首个关键帧不能标成完整记录。
    """
    cursor, gaps = start_us, []
    seen = False
    for segment in sorted(segments, key=lambda s: s["global_start_us"]):
        left, right = segment["global_start_us"], segment["global_end_us"]
        if right < start_us or left > end_us:
            continue
        tolerance = 250_000 if seen else 2_000
        if left - cursor > tolerance:
            gaps.append((cursor, min(left, end_us)))
        cursor = max(cursor, right)
        seen = True
    if end_us - cursor > 2_000:
        gaps.append((cursor, end_us))
    return gaps


def verify_rule_feedback(start, submitted, confirmed, observations):
    """独立按原始观察复核规则动作，不直接相信动作账本的“已确认”文字。"""
    from doudizhu_lab.agents.douzero import TO_NUMBER, TO_RANK, legal_actions
    from doudizhu_lab.agents.rules import feedback

    decision, data = start.get("decision", {}), start.get("input_context", {})
    numbers = sorted(TO_NUMBER[c] for c in decision.get("cards", []))
    try:
        if numbers not in legal_actions(data["player_hand_cards"], data["target"]):
            return False
        frames = confirmed.get("feedback", {}).get("frames", [])
        values = [observations[f] for f in frames]
        if len(set(frames)) != 2 or values[-1]["pts_us"] - values[0]["pts_us"] < 100_000:
            return False
        before = Counter(TO_RANK[n] for n in data["player_hand_cards"])
        intent = SimpleNamespace(cards=decision["cards"])
        for observation in values:
            snap = SimpleNamespace(
                observation=observation, frame=SimpleNamespace(media_pts_us=observation["pts_us"])
            )
            result = feedback(snap, intent, before, data["remaining"], submitted["pts_us"])
            if result is None or result["basis"] != confirmed["feedback"]["basis"]:
                return False
    except (KeyError, TypeError, ValueError):
        return False
    return True


def verify_inference(event, before, observations):
    """核对推断不出的两个原始行动者/计数锚点；有牌数变化或轮转不唯一则拒绝。"""
    proof = event.get("data", {}).get("inference", {})
    try:
        values = [observations[f] for f in proof["frames"]]
        if len(values) != 2 or values[-1]["pts_us"] - values[0]["pts_us"] < 100_000 or not before.last_play:
            return False
        actor = values[0]["actor"]
        distance = (before.seat_order.index(actor) - before.seat_order.index(before.current_actor)) % 3
        if not 1 <= distance <= 2 - before.passes:
            return False
        for obs in values:
            if obs["actor"] != actor or Counter(obs.get("hand") or []) != Counter(before.hand or []):
                return False
            if any(obs["remaining"].get(s) != before.remaining[s] for s in before.seat_order):
                return False
    except (KeyError, ValueError, TypeError):
        return False
    return True


def evaluate_run(folder, *, output=None):
    """只读评测任务；模拟结果单独标记，实时完成还要求录像正常收尾及无采集缺口。"""
    folder = Path(folder)
    with sqlite3.connect((folder / "run.sqlite3").resolve().as_uri() + "?mode=ro", uri=True) as db:
        meta = {k: json.loads(v) for k, v in db.execute("SELECT key,payload FROM meta")}
        journal = [(k, json.loads(v)) for k, v in db.execute("SELECT kind,payload FROM journal ORDER BY seq")]
        flow_frames = {
            fid: json.loads(v) for fid, v in db.execute("SELECT frame_id,payload FROM observations")
        }
    games = [v for k, v in journal if k == "game_complete"]
    not_sent = {v["action_id"] for k, v in journal if k == "flow_not_sent"}
    sends = [v for k, v in journal if k == "flow_sent" and v["action_id"] not in not_sent]
    confirmed = {v["action_id"] for k, v in journal if k == "flow_confirmed"}
    unresolved = [v for v in sends if v["action_id"] not in confirmed]
    duplicates = sum(n - 1 for n in Counter(v["opportunity"] for v in sends).values() if n > 1)
    bidding_errors = []
    bid_sends = [v for v in sends if v.get("bidding")]
    for sent in bid_sends:
        decision = sent["bidding"]
        try:
            source = flow_frames[sent["frame_id"]]
            if decision["kind"] != sent["kind"] or decision["opportunity"] != sent["opportunity"]:
                raise ValueError("叫抢意图与实际点击不一致")
            if not any(b["kind"] == sent["kind"] and b["enabled"] for b in source["buttons"]):
                raise ValueError("源帧没有可用叫抢按钮")
            if sent["kind"] in ("call_landlord", "rob_landlord"):
                from doudizhu_lab.agents.bidding import BID_MODEL_ID, encode_hand

                encode_hand(decision["hand"])
                evidence = [flow_frames[f] for f in decision["evidence_frames"]]
                if (
                    decision["model_id"] != BID_MODEL_ID
                    or decision["score"] is None
                    or decision["score"] <= decision["threshold"]
                    or len(evidence) != 2
                    or evidence[-1]["pts_us"] - evidence[0]["pts_us"] < 100_000
                    or len({f["frame_id"] for f in evidence}) != 2
                    or any(
                        Counter(f.get("bid_hand") or []) != Counter(decision["hand"])
                        for f in [source, *evidence]
                    )
                ):
                    raise ValueError("肯定叫抢缺少固定模型、评分或两帧17张牌证据")
        except (KeyError, TypeError, ValueError) as exc:
            bidding_errors.append({"action_id": sent["action_id"], "reason": str(exc)})
    pauses = [v for k, v in journal if k == "paused"]
    manual = [v for k, v in journal if k == "manual_takeover"]
    simulation = meta["source"] == "simulation_continuous"
    session = Path(meta["session"])
    video_closed, capture_issues, submissions, illegal = simulation, [], [], []
    game_audits = []
    policy = meta.get("profile", {}).get("continuous_policy", "strict")
    operational_events, actions = [], []
    inference_errors = []
    db_path = session / "session.sqlite3"
    if db_path.exists():
        with sqlite3.connect(db_path.resolve().as_uri() + "?mode=ro", uri=True) as db:
            actions = [json.loads(v[0]) for v in db.execute("SELECT payload FROM bot_actions ORDER BY seq")]
            all_issues = [json.loads(v[0]) for v in db.execute("SELECT payload FROM issues")]
            events = [json.loads(v[0]) for v in db.execute("SELECT payload FROM events ORDER BY seq")]
            tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if policy != "strict" and "operational_events" in tables:
                operational_events = [
                    json.loads(v[0])
                    for v in db.execute("SELECT payload FROM operational_events ORDER BY seq")
                ]
            observation_map = (
                {
                    value["frame_id"]: value
                    for (payload,) in db.execute("SELECT payload FROM observations")
                    for value in [json.loads(payload)]
                }
                if policy != "strict"
                else {}
            )
        if operational_events:
            events = operational_events
        completed_ids = {g["game_id"] for g in games}
        starts = {a["action_id"]: a.get("decision", {}) for a in actions if a["stage"] == "开始"}
        rejected = {a["action_id"] for a in actions if a["stage"] == "输入未发送"}
        for a in actions:
            if a.get("game_id") not in completed_ids:
                continue
            if (
                a["stage"] == "准备输入"
                and a.get("kind") in ("play", "pass")
                and a["action_id"] not in rejected
            ):
                d = starts.get(a["action_id"], {})
                submissions.append((a["game_id"], d.get("turn")))
            if a["stage"] == "命令结果未知":
                illegal.append(a)
        # 只有与完成牌局时间重叠的物理缺口影响这些牌局；大厅前后的正常等待不串局。
        from doudizhu_lab.storage.session import ANALYSIS_GAP_CODES, HARD_GAP_CODES

        closures = {i.get("gap_id"): i["end_us"] for i in all_issues if i.get("code") == "gap_closed"}
        end = max((g["end_us"] for g in games), default=0)
        capture_issues = [
            i
            for i in all_issues
            if i.get("code") in HARD_GAP_CODES | ANALYSIS_GAP_CODES
            and any(
                i.get("start_us", i.get("pts_us", 0)) <= g["end_us"]
                and closures.get(i.get("issue_id"), i.get("end_us") or end) >= g["start_us"]
                for g in games
            )
        ]
        # 独立重放原事件并核对每一次自己的动作，不能只相信game_complete声称的verified。
        from doudizhu_lab.domain.tracker import GameState, apply_event, replay_events

        before = GameState()
        for event in events:
            if event.get("source") == "operational_inferred" and not verify_inference(
                event, before, observation_map
            ):
                inference_errors.append(event["event_id"])
            before = apply_event(before, event)

        states = replay_events(events)
        expected_source = "simulation_bot" if simulation else "live_bot"
        for game in games:
            gid = game["game_id"]
            moves = [
                e
                for e in events
                if e["game_id"] == gid and e["kind"] in ("play", "pass") and e.get("confirmed", True)
            ]
            expected = {n: e for n, e in enumerate(moves) if e["seat"] == "self"}
            good = set()
            for action in actions:
                if action.get("game_id") != gid or action["stage"] != "已确认":
                    continue
                decision = starts.get(action["action_id"], {})
                turn = decision.get("turn")
                if (
                    turn in expected
                    and Counter(decision.get("cards", [])) == Counter(expected[turn]["cards"])
                    and action.get("source") == expected_source
                ):
                    good.add(turn)
            actual = {turn for game_id, turn in submissions if game_id == gid}
            valid = bool(expected and set(expected) == good == actual and states.get(gid, {}).get("complete"))
            if game.get("degraded") and policy == "balanced":
                game_starts = {
                    a["action_id"]: a for a in actions if a.get("game_id") == gid and a["stage"] == "开始"
                }
                confirmations = {
                    a["action_id"]: a for a in actions if a.get("game_id") == gid and a["stage"] == "已确认"
                }
                sent = {
                    a["action_id"]: a
                    for a in actions
                    if a.get("game_id") == gid
                    and a["stage"] == "准备输入"
                    and a.get("kind") in ("play", "pass")
                }
                valid = bool(game_starts and set(game_starts) == set(confirmations) == set(sent))
                for ident, start in game_starts.items():
                    if ident not in confirmations or ident not in sent:
                        valid = False
                        continue
                    decision = start["decision"]
                    if decision.get("strategy", "wp") == "rules":
                        valid = valid and verify_rule_feedback(
                            start, sent[ident], confirmations[ident], observation_map
                        )
                    else:
                        valid = valid and decision["turn"] in good
                valid = valid and all(a.get("source") == expected_source for a in confirmations.values())
            game_audits.append(
                {
                    "game_id": gid,
                    "passed": valid,
                    "own_events": len(expected),
                    "confirmed_submissions": len(good),
                }
            )
    if not simulation:
        index = session / "video/index.json"
        if index.exists():
            data = json.loads(index.read_text(encoding="utf8"))
            segments = data.get("segments", []) if isinstance(data, dict) else data
            video_closed = bool(segments) and all(v.get("closed") for v in segments)
            for game in games:
                for left, right in uncovered_video_intervals(segments, game["start_us"], game["end_us"]):
                    capture_issues.append(
                        {
                            "code": "video_coverage",
                            "game_id": game["game_id"],
                            "start_us": left,
                            "end_us": right,
                            "message": "完成牌局的部分时间不在已保存录像范围内",
                        }
                    )
    duplicate_submissions = sum(n - 1 for n in Counter(submissions).values() if n > 1)
    # 保留旧报告的五局门槛；较长任务必须达到声明目标，不能30局任务只跑5局就通过。
    required_games = max(5, meta.get("profile", {}).get("continuous_settings", {}).get("max_games", 5))
    passed = bool(
        meta.get("status") == "completed"
        and len(games) >= required_games
        and len({g["game_id"] for g in games}) == len(games)
        and all(g["verified"] for g in games)
        and not unresolved
        and not duplicates
        and not duplicate_submissions
        and not pauses
        and not manual
        and not capture_issues
        and not illegal
        and not inference_errors
        and not bidding_errors
        and video_closed
        and len(game_audits) == len(games)
        and all(g["passed"] for g in game_audits)
    )
    result = {
        "source": meta["source"],
        "policy": policy,
        "status": meta.get("status"),
        "passed": passed,
        "live_passed": passed and not simulation,
        "strict_live_passed": passed
        and not simulation
        and not any(g.get("degraded") or g.get("inferred_events") for g in games),
        "rules_actions": sum(a["stage"] == "已确认" and a.get("strategy") == "rules" for a in actions),
        "inferred_events": sum(e.get("source") == "operational_inferred" for e in operational_events),
        "inference_proof_errors": inference_errors,
        "completed_games": len(games),
        "required_games": required_games,
        "transitions": max(0, len(games) - 1),
        "games": games,
        "game_audits": game_audits,
        "flow_commands": len(sends),
        "unconfirmed_flow_actions": unresolved,
        "inputs_rejected_before_send": len(not_sent),
        "duplicate_flow_commands": duplicates,
        "bidding_mode": meta.get("profile", {}).get("bid_settings", {}).get("mode", "disabled"),
        "bidding_actions": dict(Counter(v["kind"] for v in bid_sends)),
        "bidding_fallbacks": sum(
            v["bidding"].get("score") is None
            for v in bid_sends
            if v["bidding"].get("reason") != "已选择固定不叫不抢"
        ),
        "bidding_proof_errors": bidding_errors,
        "duplicate_submissions": duplicate_submissions,
        "input_errors": illegal,
        "pauses": pauses,
        "manual_interventions": len(manual),
        "capture_issues": capture_issues,
        "video_closed": video_closed,
        "note": "连续任务功能检查；不替代事件识别正式验收，不报告对战胜率。",
    }
    target = Path(output) if output else folder / "连续任务报告.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf8")
    return result
