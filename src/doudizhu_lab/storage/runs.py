"""连续任务账本和来源明确的完成评测；流程意图与牌局事实使用不同数据库。"""

import json
import sqlite3
from collections import Counter
from contextlib import closing
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from types import SimpleNamespace
from uuid import uuid4

from doudizhu_lab.domain.settings import CONFIGURATION_VERSION


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
            "configuration_version": CONFIGURATION_VERSION,
            "evidence_policy_version": 2,
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
            # 仅保存旧观察字段以兼容历史格式，不代表新任务要求或执行过玩法校验。
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

    @classmethod
    def create_task(cls, root, profile, serial, *, simulation=False):
        """在父进程创建无人值守任务账本，不创建设备会话或恢复任何旧输入授权。

        profile 为本次配置快照，serial 固定设备；simulation 只供隔离模拟标记来源。
        返回可追加的账本，I/O 异常交由看护器进入自动等待，调用者负责关闭。
        """
        ident = datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid4().hex[:6]
        result = cls(Path(root) / "runs" / ident)
        for key, value in {
            "version": 2,
            "evidence_policy_version": 2,
            "run_id": ident,
            "profile": asdict(profile),
            "device": serial,
            "source": "simulation_continuous" if simulation else "live_continuous",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "running",
        }.items():
            result.set(key, value)
        result.db.execute(
            "CREATE TABLE IF NOT EXISTS task_events(event_id TEXT PRIMARY KEY,kind TEXT,payload TEXT)"
        )
        result.db.commit()
        return result

    def commit_event(self, event_id, kind, value):
        """按稳定事件编号事务提交一次事实；重复交付返回 False，不增加完成局数。

        仅父任务写入线程调用；返回 True 表示已经持久提交，可以回复执行进程。
        不发送手机输入，写入失败抛出 sqlite3/OSError，不能先回复成功。
        """
        with self.lock, self.db:
            cursor = self.db.execute(
                "INSERT OR IGNORE INTO task_events VALUES (?,?,?)",
                (event_id, kind, json.dumps(value, ensure_ascii=False)),
            )
            return bool(cursor.rowcount)

    def task_outcomes(self) -> dict[str, str]:
        """父账本写入异常恢复后只读核对已提交结果，避免把提交结果未知的有效局改成跳过。

        仅任务看护线程调用，返回稳定牌局编号到结果类型的映射；不恢复任何输入授权。
        SQLite 错误上抛，调用者继续等待存储恢复，不能凭内存猜测提交成功或失败。
        """
        with self.lock:
            return {
                json.loads(payload)["round_id"]: kind
                for kind, payload in self.db.execute(
                    "SELECT kind,payload FROM task_events WHERE kind IN ('game_complete','game_skipped')"
                )
            }


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
        if data.get("seat_order") and (
            not values[0].get("frame_context") or values[0]["frame_context"] != values[1].get("frame_context")
        ):
            return False
        before = Counter(TO_RANK[n] for n in data["player_hand_cards"])
        intent = SimpleNamespace(cards=decision["cards"])
        for observation in values:
            snap = SimpleNamespace(
                observation=observation, frame=SimpleNamespace(media_pts_us=observation["pts_us"])
            )
            result = feedback(
                snap,
                intent,
                before,
                data["remaining"],
                submitted["pts_us"],
                seat_order=data.get("seat_order", ()),
            )
            if result is None or result["basis"] != confirmed["feedback"]["basis"]:
                return False
    except (KeyError, TypeError, ValueError):
        return False
    return True


def verify_inference(event, before, observations, *, expected_context=None):
    """按证明版本独立核对原始锚点；第二版使用前置状态指纹和唯一约束重新求解。

    第一版旧记录保留原不出检查口径；expected_context由本局原始开局帧提供，
    只参与第二版校验，不用新的验收规则倒改历史成绩。返回False保留明确证明错误。
    """
    proof = event.get("data", {}).get("inference", {})
    if proof.get("version") == 2:
        from doudizhu_lab.domain.inference import verify_proof

        return verify_proof(event, before, observations, expected_context=expected_context)
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


def verify_flow_transition(sent: dict, confirmed: dict, observations: dict[str, dict]) -> bool:
    """独立检查续局的新牌局双帧证据；仅点击返回、按钮消失或结果标记不构成证明。

    observations必须来自会话原始观察表，不能使用运行时替换后的视图。旧版本未使用
    此证明形状，不经本接口改写历史页面反馈结论。
    """
    try:
        proof = confirmed["proof"]
        a, b = [observations[f] for f in proof["frames"]]
        return bool(
            proof["version"] == 2
            and sent["kind"] == "continue"
            and sent["origin_scene"] == "settlement"
            and proof["game_id"] != proof.get("origin_game_id")
            and proof.get("origin_game_id") == sent.get("origin_game_id")
            and a["frame_id"] != b["frame_id"]
            and sent["pts_us"] < a["pts_us"] < b["pts_us"] <= confirmed["pts_us"]
            and b["pts_us"] - a["pts_us"] >= 100_000
            and b["pts_us"] - sent["pts_us"] <= 8_000_000
            and a.get("frame_context")
            and a["frame_context"] == b.get("frame_context")
            and bool(proof.get("raw_round_id"))
            and a["frame_context"].get("round_id") == proof["raw_round_id"]
            and proof["raw_round_id"] != sent.get("origin_round_id")
            and sent.get("frame_context")
            and all(
                a["frame_context"].get(key) == sent["frame_context"].get(key)
                for key in ("serial", "stream_id", "width", "height")
            )
            and all(
                o.get("phase") in ("dealing", "bidding", "doubling", "playing")
                and len(o.get("hand") or []) in (17, 20)
                for o in (a, b)
            )
            and Counter(a["hand"]) == Counter(b["hand"])
        )
    except (KeyError, TypeError, ValueError):
        return False


def audit_game_evidence(
    game: dict,
    events: list[dict],
    raw_events: list[dict],
    observations: dict[str, dict],
    actions: list[dict],
    *,
    evidence_version: int = 2,
    policy: str = "strict",
    source: str = "live_bot",
) -> dict:
    """独立复核一局的分局、推断和单次实际提交，运行完成门槛与离线报告共用。

    game 只提供 game_id/degraded，不信任其 verified 或 strict_complete。events 为
    独立运行事件，raw_events/observations 必须来自不可变原始账本；actions 是实际
    执行账本。函数不读写数据库、不操作手机、不修改参数，失败返回按本局归属的错误。
    第二版要求唯一原始分局绑定；缺少版本的旧资料由调用方传入 1，保留旧确认口径。
    """
    from dataclasses import asdict

    from doudizhu_lab.domain.round_evidence import resolve_round_binding
    from doudizhu_lab.domain.tracker import GameState, apply_event

    gid = game["game_id"]
    history = [e for e in events if e["game_id"] == gid]
    records = [a for a in actions if a.get("game_id") == gid]
    openings = [e for e in history if e["kind"] == "game_start"]
    binding = resolve_round_binding(openings[0], raw_events, observations) if len(openings) == 1 else None
    binding_errors = [gid] if evidence_version >= 2 and binding is None else []
    inference_errors = []
    before = GameState()
    for event in history:
        if event.get("source") == "operational_inferred":
            declared = event.get("data", {}).get("round_binding")
            if (
                (evidence_version >= 2 and binding is None)
                or (declared is not None and (binding is None or declared != asdict(binding)))
                or not verify_inference(
                    event, before, observations, expected_context=binding.frame_context if binding else None
                )
            ):
                inference_errors.append(event["event_id"])
        before = apply_event(before, event)
    moves = [e for e in history if e["kind"] in ("play", "pass") and e.get("confirmed", True)]
    expected = {n: e for n, e in enumerate(moves) if e["seat"] == "self"}
    starts = {a["action_id"]: a for a in records if a["stage"] == "开始"}
    confirmations = [a for a in records if a["stage"] == "已确认"]
    rejected = {
        (a["action_id"], a.get("frame_id"), a.get("kind")) for a in records if a["stage"] == "输入未发送"
    }
    attempts = Counter(
        (a["action_id"], a.get("frame_id"), a.get("kind"))
        for a in records
        if a["stage"] in ("命令已返回", "命令结果未知") and a.get("kind") in ("play", "pass")
    )
    submitted = {}
    turns = []
    submission_errors = []
    for record in records:
        key = record["action_id"], record.get("frame_id"), record.get("kind")
        if record["stage"] != "准备输入" or record.get("kind") not in ("play", "pass") or key in rejected:
            continue
        ident = record["action_id"]
        submitted.setdefault(ident, []).append(record)
        turns.append(starts.get(ident, {}).get("decision", {}).get("turn"))
        if evidence_version >= 2 and attempts[key] != 1:
            submission_errors.append(ident)
    good = set()
    for confirmation in confirmations:
        ident = confirmation["action_id"]
        decision = starts.get(ident, {}).get("decision", {})
        turn = decision.get("turn")
        sends = submitted.get(ident, [])
        sent = sends[0] if len(sends) == 1 else {}
        actual = evidence_version < 2 or bool(
            sent
            and ident not in submission_errors
            and turn in expected
            and sent.get("kind") == expected[turn]["kind"] == ("play" if decision.get("cards") else "pass")
            and expected[turn]["pts_us"] > sent["pts_us"]
            and confirmation["pts_us"]
            >= (expected[turn].get("data", {}).get("confirmed_at_us") or expected[turn]["pts_us"])
        )
        if (
            turn in expected
            and Counter(decision.get("cards", [])) == Counter(expected[turn]["cards"])
            and confirmation.get("source") == source
            and actual
        ):
            good.add(turn)
        elif decision.get("strategy", "wp") != "rules":
            submission_errors.append(ident)
    duplicates = sum(n - 1 for n in Counter(turns).values() if n > 1)
    valid = bool(expected and set(expected) == good == set(turns) and before.complete and not duplicates)
    if game.get("degraded") and policy == "balanced":
        confirmed = {a["action_id"]: a for a in confirmations}
        valid = bool(starts and set(starts) == set(confirmed) == set(submitted) and not duplicates)
        for ident, start in starts.items():
            if ident not in confirmed or len(submitted.get(ident, [])) != 1:
                valid = False
                continue
            decision = start["decision"]
            sent = submitted[ident][0]
            actual = evidence_version < 2 or ident not in submission_errors
            if decision.get("strategy", "wp") == "rules":
                actual = actual and verify_rule_feedback(start, sent, confirmed[ident], observations)
                if evidence_version >= 2 and binding is not None:
                    actual = actual and all(
                        observations.get(f, {}).get("frame_context") == binding.frame_context
                        for f in confirmed[ident].get("feedback", {}).get("frames", [])
                    )
            else:
                actual = actual and decision["turn"] in good
            valid = valid and actual and confirmed[ident].get("source") == source
    valid = bool(valid and not inference_errors and not binding_errors and not submission_errors)
    return {
        "game_id": gid,
        "passed": valid,
        "own_events": len(expected),
        "confirmed_submissions": len(good),
        "inference_proof_errors": inference_errors,
        "submission_proof_errors": sorted(set(submission_errors)),
        "binding_errors": binding_errors,
        "duplicate_submissions": duplicates,
        "round_binding": asdict(binding) if binding else None,
    }


def evaluate_run(folder, *, output=None):
    """只读评测任务；模拟结果单独标记，实时完成还要求录像正常收尾及无采集缺口。"""
    folder = Path(folder)
    with closing(sqlite3.connect((folder / "run.sqlite3").resolve().as_uri() + "?mode=ro", uri=True)) as db:
        meta = {k: json.loads(v) for k, v in db.execute("SELECT key,payload FROM meta")}
        if meta.get("version") == 2:
            events = [
                (k, json.loads(v))
                for k, v in db.execute("SELECT kind,payload FROM task_events ORDER BY rowid")
            ]
            return evaluate_task(folder, meta, events, output=output)
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
    summary_path = folder.parent.parent / "recording-summaries" / f"{session.name}.json"
    retention_path = session / "留存记录.json"
    retention = {}
    for path in (summary_path, retention_path):
        if path.is_file():
            retention.update(json.loads(path.read_text(encoding="utf-8")))
    policy_cleaned = retention.get("media_status") == "cleaned" or bool(retention.get("removed_segments"))
    video_closed, capture_issues, submissions, illegal = simulation, [], [], []
    game_audits = []
    policy = meta.get("profile", {}).get("continuous_policy", "strict")
    operational_events, actions, events = [], [], []
    inference_errors, flow_proof_errors, submission_errors, binding_errors = [], [], [], []
    db_path = session / "session.sqlite3"
    if db_path.exists():
        with closing(sqlite3.connect(db_path.resolve().as_uri() + "?mode=ro", uri=True)) as db:
            actions = [json.loads(v[0]) for v in db.execute("SELECT payload FROM bot_actions ORDER BY seq")]
            all_issues = [json.loads(v[0]) for v in db.execute("SELECT payload FROM issues")]
            events = [json.loads(v[0]) for v in db.execute("SELECT payload FROM events ORDER BY seq")]
            tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if (
                policy != "strict" or meta.get("evidence_policy_version", 1) >= 2
            ) and "operational_events" in tables:
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
                if policy != "strict" or meta.get("evidence_policy_version", 1) >= 2
                else {}
            )
        raw_events = events
        if operational_events:
            events = operational_events
        sends_by_id = {s["action_id"]: s for s in sends}
        for kind, record in journal:
            if kind == "flow_confirmed" and record.get("evidence") == "new_round":
                if not verify_flow_transition(
                    sends_by_id.get(record["action_id"], {}), record, observation_map
                ):
                    flow_proof_errors.append(record["action_id"])
        completed_ids = {g["game_id"] for g in games}
        starts = {a["action_id"]: a.get("decision", {}) for a in actions if a["stage"] == "开始"}
        rejected = {
            (a["action_id"], a.get("frame_id"), a.get("kind")) for a in actions if a["stage"] == "输入未发送"
        }
        resolved = {a["action_id"] for a in actions if a["stage"] == "已确认"}
        # 第二版必须同时存在一次准备记录和一次真实调用结果。准备写入本身只是意图，
        # 明确未发送不能算提交；超时后的未知结果允许由原始画面证明，但不能重复发送。
        attempted = Counter(
            (a["action_id"], a.get("frame_id"), a.get("kind"))
            for a in actions
            if a["stage"] in ("命令已返回", "命令结果未知") and a.get("kind") in ("play", "pass")
        )
        submitted_records = {}
        for a in actions:
            if a.get("game_id") not in completed_ids:
                continue
            if (
                a["stage"] == "准备输入"
                and a.get("kind") in ("play", "pass")
                and (a["action_id"], a.get("frame_id"), a.get("kind")) not in rejected
            ):
                d = starts.get(a["action_id"], {})
                submissions.append((a["game_id"], d.get("turn")))
                submitted_records.setdefault(a["action_id"], []).append(a)
                if (
                    meta.get("evidence_policy_version", 1) >= 2
                    and attempted[a["action_id"], a.get("frame_id"), a.get("kind")] != 1
                ):
                    submission_errors.append(a["action_id"])
            if a["stage"] == "命令结果未知" and a["action_id"] not in resolved:
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
        # 完成前与收尾后都使用同一纯函数，错误直接归属到具体牌局。
        game_by_id = {g["game_id"]: g for g in games}
        audits = {
            gid: audit_game_evidence(
                game_by_id.get(gid, {"game_id": gid}),
                events,
                raw_events,
                observation_map,
                actions,
                evidence_version=meta.get("evidence_policy_version", 1),
                policy=policy,
                source="simulation_bot" if simulation else "live_bot",
            )
            for gid in dict.fromkeys(e["game_id"] for e in events if e["kind"] == "game_start")
        }
        inference_errors = [error for a in audits.values() for error in a["inference_proof_errors"]]
        for game in games:
            audit = audits.get(game["game_id"], {"game_id": game["game_id"], "passed": False})
            game_audits.append(audit)
            binding_errors.extend(audit.get("binding_errors", []))
            submission_errors.extend(audit.get("submission_proof_errors", []))
        submission_errors = sorted(set(submission_errors))
    if not simulation:
        index = session / "video/index.json"
        if index.exists():
            data = json.loads(index.read_text(encoding="utf8"))
            segments = data.get("segments", []) if isinstance(data, dict) else data
            video_closed = bool(segments) and all(v.get("closed") for v in segments)
            # 仅用已登记清理范围解释对应缺口；同会话其他区间仍须检查真实采集丢失。
            coverage = segments + [
                {"global_start_us": s["start_us"], "global_end_us": s["end_us"]}
                for s in retention.get("removed_segments", [])
            ]
            for game in games:
                missing = uncovered_video_intervals(coverage, game["start_us"], game["end_us"])
                for left, right in missing:
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
        and not flow_proof_errors
        and not submission_errors
        and not bidding_errors
        and video_closed
        and not policy_cleaned
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
        and not any(
            g.get("degraded") or (g.get("inferred_events") and meta.get("evidence_policy_version", 1) < 2)
            for g in games
        ),
        "evidence_policy_version": meta.get("evidence_policy_version", 1),
        "directly_observed_events": sum(e.get("source") != "operational_inferred" for e in events),
        "inferred_events_by_kind": dict(
            Counter(e["kind"] for e in operational_events if e.get("source") == "operational_inferred")
        ),
        "rules_actions": sum(a["stage"] == "已确认" and a.get("strategy") == "rules" for a in actions),
        "inferred_events": sum(e.get("source") == "operational_inferred" for e in operational_events),
        "valid_inferred_events": sum(e.get("source") == "operational_inferred" for e in operational_events)
        - len(inference_errors),
        "inference_proof_errors": inference_errors,
        "binding_errors": binding_errors,
        "flow_proof_errors": flow_proof_errors,
        "submission_proof_errors": submission_errors,
        "completed_games": len(games),
        "target_reached": len(games)
        >= meta.get("profile", {}).get("continuous_settings", {}).get("max_games", 5),
        "tracking_completed_games": max(
            len(games), sum(w.get("outcome") == "completed" for w in retention.get("windows", []))
        ),
        "media_status": "policy_cleaned" if policy_cleaned else "retained",
        "video_audit_available": video_closed and not policy_cleaned,
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


def evaluate_task(folder, meta, events, *, output=None):
    """汇总多会话任务；目标达成与严格验收分开，未知提交和异常不会被恢复次数冲掉。

    仅读取子执行账本；缺失或未封口的子报告记为未通过。旧会话走原 evaluate_run。
    输出中文报告文件，不修改原始观察、旧会话或完成事实。
    """
    sessions = [v for k, v in events if k == "session"]
    games = [v for k, v in events if k == "game_complete"]
    skipped = [v for k, v in events if k == "game_skipped"]
    recoveries = [v for k, v in events if k == "recovery"]
    audits = []
    for session in sessions:
        path = Path(session["run_folder"]) / "连续任务报告.json"
        try:
            child = json.loads(path.read_text(encoding="utf8"))
            audits.append({"session": session, "passed": child.get("passed", False), "report": child})
        except (OSError, ValueError):
            audits.append({"session": session, "passed": False})
    total = meta["profile"]["continuous_settings"]["max_games"]
    reached = len(games) >= total
    passed = bool(
        meta.get("status") == "completed"
        and reached
        and not skipped
        and not recoveries
        and audits
        and all(a["passed"] for a in audits)
    )
    result = {
        "source": meta["source"],
        "evidence_policy_version": meta.get("evidence_policy_version", 1),
        "directly_observed_events": sum(
            a.get("report", {}).get("directly_observed_events", 0) for a in audits
        ),
        "valid_inferred_events": sum(a.get("report", {}).get("valid_inferred_events", 0) for a in audits),
        "rules_actions": sum(a.get("report", {}).get("rules_actions", 0) for a in audits),
        "proof_errors": [
            {"session": a["session"].get("folder", ""), "kind": kind, "error": error}
            for a in audits
            for kind in (
                "inference_proof_errors",
                "flow_proof_errors",
                "submission_proof_errors",
                "binding_errors",
            )
            for error in a.get("report", {}).get(kind, [])
        ],
        "status": meta["status"],
        "target_reached": reached,
        "completed_games": len(games),
        "required_games": total,
        "skipped_games": len(skipped),
        "games": games,
        "skipped": skipped,
        "recoveries": recoveries,
        "sessions": audits,
        "passed": passed,
        "live_passed": passed and meta["source"] == "live_continuous",
        "note": "任务目标按有效局数统计；恢复后完成不代表严格连续运行或录像验收通过。",
    }
    from doudizhu_lab.storage.json_io import atomic_json

    atomic_json(Path(output) if output else Path(folder) / "连续任务报告.json", result)
    return result


def refresh_run_retention(folder: str | Path) -> dict:
    """在关闭并评测全部数据库、完成媒体处置后，只更新报告的留存状态。

    不重新打开已删除的原始数据库，不用缺失源文件覆盖先前独立核验的计数和证明结果。
    已按策略清理媒体仍不能声明完整录像验收通过；清理等待不改变目标完成事实。
    """
    from doudizhu_lab.storage.json_io import atomic_json

    folder = Path(folder)
    report_path = folder / "连续任务报告.json"
    report = json.loads(report_path.read_text(encoding="utf8"))
    with closing(sqlite3.connect((folder / "run.sqlite3").resolve().as_uri() + "?mode=ro", uri=True)) as db:
        row = db.execute("SELECT payload FROM meta WHERE key='session'").fetchone()
    if row is None:
        return report
    session = Path(json.loads(row[0]))
    summary = folder.parent.parent / "recording-summaries" / f"{session.name}.json"
    if summary.is_file():
        value = json.loads(summary.read_text(encoding="utf8"))
        status = value.get("media_status", "retained")
        report["cleanup_pending"] = status == "cleanup_pending"
        if status == "cleaned":
            report.update(
                media_status="policy_cleaned",
                video_audit_available=False,
                passed=False,
                live_passed=False,
                strict_live_passed=False,
            )
        elif status in ("cleanup_pending", "cleanup_failed"):
            report["media_status"] = status
        atomic_json(report_path, report)
    return report
