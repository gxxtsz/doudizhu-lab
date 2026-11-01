"""按牌局和动作顺序评测；实时自动、离线重算与人工修订的口径明确分离。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from doudizhu_lab.domain.tracker import GameState, apply_event, canonical
from doudizhu_lab.storage.corrections import observation_hash
from doudizhu_lab.storage.session import SessionStore, game_intervals, release_fingerprint


def ordered_matches(left, right, eligible, distance):
    """保持两侧顺序的一对一匹配：先最大化命中数，再最小化时间误差。

    返回索引对，不按时间最近贪心复用事件，因此连续相同牌点和错序不会互相抵消。
    规模按单局动作计算，时间和空间复杂度均为 O(n*m)。
    """
    n, m = len(left), len(right)
    scores = [[(0, 0) for _ in range(m + 1)] for _ in range(n + 1)]
    moves = [[0 for _ in range(m + 1)] for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            options = [(scores[i - 1][j], 1), (scores[i][j - 1], 2)]
            if eligible(left[i - 1], right[j - 1]):
                hit, cost = scores[i - 1][j - 1]
                options.append(((hit + 1, cost - distance(left[i - 1], right[j - 1])), 3))
            scores[i][j], moves[i][j] = max(options, key=lambda v: (v[0], v[1]))
    pairs = []
    i, j = n, m
    while i and j:
        move = moves[i][j]
        if move == 3:
            pairs.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif move == 1:
            i -= 1
        else:
            j -= 1
    return list(reversed(pairs))


def _signature(event):
    return event.get("seat"), event["kind"], tuple(sorted(event.get("cards", [])))


def _predicted_games(events, end):
    intervals = game_intervals(events, end)
    abandoned = _abandoned_games(events)
    return [
        {
            "game_id": ident,
            "start_us": span[0],
            "end_us": next(
                (e["pts_us"] for e in events if e["kind"] == "game_end" and e["game_id"] == ident), None
            ),
        }
        for ident, span in intervals.items()
        if ident not in abandoned
    ]


def _abandoned_games(events):
    """将已确认的无地主流局单列，不把已发生动作的牌局用结束标签排除评分。

    状态机必须保存重新发牌证据，且这一段没有出牌、不出或已确定地主的记录。
    只有这些条件同时成立才排除于正式出牌局的边界匹配；原事件仍完整保留。
    """
    marked = {
        e["game_id"] for e in events if e["kind"] == "game_end" and e.get("data", {}).get("abandoned_deal")
    }
    active = {
        e["game_id"] for e in events if e["kind"] in ("play", "pass") or e.get("data", {}).get("landlord")
    }
    return marked - active


def _boundaries(predicted, expected, tolerance):
    checks = {}
    for kind, key in (("game_start", "start_us"), ("game_end", "end_us")):
        left = [g[key] for g in predicted if g.get(key) is not None]
        right = [g[key] for g in expected if g.get(key) is not None]
        pairs = ordered_matches(left, right, lambda a, b: abs(a - b) <= tolerance, lambda a, b: abs(a - b))
        checks[kind] = {
            "expected": len(right),
            "observed": len(left),
            "matched": len(pairs),
            "missed": len(right) - len(pairs),
            "extra": len(left) - len(pairs),
            "max_offset_us": max((abs(left[i] - right[j]) for i, j in pairs), default=None),
        }
    checks["passed"] = all(
        not checks[k]["missed"] and not checks[k]["extra"] for k in ("game_start", "game_end")
    )
    return checks


def _score(events, truth, end):
    expected_games = [
        g | {"game_id": g.get("game_id", f"truth-{i}")} for i, g in enumerate(truth.get("games", []))
    ]
    predicted_games = _predicted_games(events, end)
    tolerance = truth.get("tolerance_us", 1_000_000)
    boundary_tolerance = truth.get("boundary_tolerance_us", 2_000_000)
    predicted = [e for e in events if e["kind"] in ("play", "pass") and e.get("confirmed", True)]
    expected = [e for e in truth["events"] if e["kind"] in ("play", "pass")]
    pairs = []

    def eligible(a, b):
        return _signature(a) == _signature(b) and abs(a["pts_us"] - b["pts_us"]) <= tolerance

    def distance(a, b):
        return abs(a["pts_us"] - b["pts_us"])

    game_pairs = []
    if expected_games:

        def game_close(a, b):
            return any(
                a.get(k) is not None and b.get(k) is not None and abs(a[k] - b[k]) <= boundary_tolerance
                for k in ("start_us", "end_us")
            )

        game_pairs = ordered_matches(
            predicted_games,
            expected_games,
            game_close,
            lambda a, b: sum(
                abs(a[k] - b[k])
                for k in ("start_us", "end_us")
                if a.get(k) is not None and b.get(k) is not None
            ),
        )
        groups = []
        for i, j in game_pairs:
            p, t = predicted_games[i], expected_games[j]
            pi = [n for n, e in enumerate(predicted) if e.get("game_id") == p["game_id"]]
            ti = [
                n
                for n, e in enumerate(expected)
                if (
                    e.get("game_id") == t["game_id"]
                    if e.get("game_id")
                    else t["start_us"] <= e["pts_us"] <= (t.get("end_us") or end)
                )
            ]
            groups.append((pi, ti, p["game_id"], t["game_id"]))
    else:
        # 兼容旧版片段标签；缺少完整牌局边界永远不能通过正式验收。
        groups = [(list(range(len(predicted))), list(range(len(expected))), None, None)]
    candidates = {}
    for pi, ti, _, _ in groups:
        local = ordered_matches([predicted[i] for i in pi], [expected[j] for j in ti], eligible, distance)
        pairs.extend((pi[i], ti[j]) for i, j in local)
        for i in pi:
            candidates[i] = [j for j in ti if eligible(predicted[i], expected[j])]
    used_pred, used_truth = {i for i, _ in pairs}, {j for _, j in pairs}
    metrics = {}
    for kind in ("play", "pass"):
        ps = [i for i, e in enumerate(predicted) if e["kind"] == kind]
        ts = [i for i, e in enumerate(expected) if e["kind"] == kind]
        correct = sum(i in used_pred for i in ps)
        duplicate = sum(
            i not in used_pred and any(j in used_truth for j in candidates.get(i, [])) for i in ps
        )
        unordered = sum(
            i not in used_pred and any(j not in used_truth for j in candidates.get(i, [])) for i in ps
        )
        metrics[kind] = {
            "truth": len(ts),
            "events": len(ps),
            "correct": correct,
            "missed": len(ts) - correct,
            "false_positive": len(ps) - correct,
            "duplicates": duplicate,
            "out_of_order": unordered,
            "precision": correct / len(ps) if ps else (0.0 if ts else 1.0),
            "recall": correct / len(ts) if ts else 1.0,
        }
    details = {
        "matched": [
            {
                "event_id": predicted[i].get("event_id"),
                "truth_event_id": expected[j].get("event_id"),
                "offset_us": predicted[i]["pts_us"] - expected[j]["pts_us"],
            }
            for i, j in pairs
        ],
        "missed": [e for j, e in enumerate(expected) if j not in used_truth],
        "extra": [e for i, e in enumerate(predicted) if i not in used_pred],
        "game_pairs": [
            {"predicted": predicted_games[i]["game_id"], "truth": expected_games[j]["game_id"]}
            for i, j in game_pairs
        ],
    }
    return metrics, _boundaries(predicted_games, expected_games, boundary_tolerance), details


def _state_checks(events, truth, details):
    """将匹配动作之后的手牌、剩牌与独立检查点逐项比较，不只比较动作总数。"""
    states, after, problems = {}, {}, []
    for event in events:
        ident = event.get("game_id", "")
        state = apply_event(states.get(ident, GameState()), event)
        states[ident] = state
        after[event.get("event_id")] = state
    expected = {e.get("event_id"): e for e in truth.get("events", [])}
    for pair in details.get("matched", []):
        label = expected.get(pair["truth_event_id"], {})
        check, state = label.get("checkpoint", {}), after.get(pair["event_id"])
        if state is None or state.pending:
            problems.append({"event_id": pair["event_id"], "message": "动作之后没有可信状态"})
            continue
        if check.get("hand") is not None and canonical(check["hand"]) != state.hand:
            problems.append({"event_id": pair["event_id"], "message": "手牌与独立检查点不一致"})
        if check.get("remaining") is not None and check["remaining"] != state.remaining:
            problems.append({"event_id": pair["event_id"], "message": "剩牌与独立检查点不一致"})
    expected_games = {g.get("game_id", f"truth-{i}"): g for i, g in enumerate(truth.get("games", []))}
    for pair in details.get("game_pairs", []):
        state, label = states.get(pair["predicted"]), expected_games[pair["truth"]]
        if state is None:
            continue
        for field in ("landlord", "initial_hand", "bottom", "seat_order"):
            if field in label:
                actual, value = getattr(state, field), label[field]
                if field in ("initial_hand", "bottom"):
                    actual, value = canonical(actual), canonical(value)
                if actual != value:
                    problems.append(
                        {"game_id": pair["predicted"], "message": f"开局字段 {field} 与独立标注不一致"}
                    )
    return problems


def _provenance(session, truth):
    """验收来源核验：真实冻结快照、录像哈希和模板来源隔离缺一不可。"""
    from doudizhu_lab.storage.truth import media_manifest, validate_truth

    problems = []
    meta = session.metadata()
    acceptance = meta.get("acceptance", {})
    source = truth.get("source", {})
    if not acceptance or acceptance.get("fingerprint") != release_fingerprint(meta):
        problems.append("会话源码、配置或模板与冻结版本不一致")
    try:
        session.verify_snapshot()
        actual_media = media_manifest(session.folder)
        if not source.get("video_manifest") or source["video_manifest"] != actual_media:
            problems.append("独立标注未绑定当前完整录像哈希")
        audit_path = session.folder / "media-audit.json"
        audit = json.loads(audit_path.read_text(encoding="utf-8")) if audit_path.exists() else {}
        if not audit.get("passed") or audit.get("video_manifest") != actual_media:
            problems.append("录像尚未通过对应文件哈希的逐帧完整性审计")
    except (OSError, ValueError, KeyError) as exc:
        problems.append(f"来源文件核验失败：{exc}")
    if source.get("session_id") != meta.get("session_id"):
        problems.append("独立标注与被评测会话编号不一致")
    check = validate_truth(truth, require_review=True)
    if not truth.get("independent_review") or not check["valid"]:
        problems.append("独立标注尚未完成逐动作校验")
    selection = truth.get("selection")
    if not selection or not selection.get("locked_at"):
        problems.append("缺少预先登记的连续验收区间")
    # 来源时间采用手机媒体时钟；即使另一会话同时录到同局，也不能绕过整局隔离。
    for entry in meta.get("template_manifest", []):
        if not entry["path"].endswith(".json"):
            continue
        try:
            item = json.loads((session.folder / entry["path"]).read_text(encoding="utf-8"))
            origin = item.get("source", {})
            if not isinstance(origin, dict):
                continue
            timestamp = origin.get("pts_us")
            if origin.get("session", origin.get("session_id")) == meta.get("session_id") or (
                timestamp is not None and source.get("start_us", 0) <= timestamp <= source.get("end_us", -1)
            ):
                problems.append("验收录像与模板来源整局不隔离")
                break
        except (OSError, ValueError, TypeError):
            problems.append("模板来源记录无法核验")
            break
    return {"passed": not problems, "problems": problems, "truth_validation": check}


def evaluate_session(folder: Path, labels: Path | None = None) -> dict:
    """评测一个会话，旧标签可评分，但正式通过还要求新批次来源、实时模式和完整核对。

    预测候选不算正确命中，疑点因此保留为漏记。人工修订只进入 corrected 栏；
    离线版本只有 offline 成绩，不会被当作实时自动结果用于通过标志。
    """
    session = SessionStore(folder)
    try:
        meta = session.metadata()
        mode = meta.get("analysis_mode", "offline" if meta.get("device") == "replay" else "legacy_live")
        truth = json.loads(labels.read_text(encoding="utf-8")) if labels else None
        original_truth = truth
        if truth and truth.get("version") == 1 and truth.get("selection"):
            # 首尾不完整局保留在独立文件中，但单列；不能稀释完整牌局事件指标。
            complete_games = [g for g in truth.get("games", []) if not g.get("incomplete")]
            identifiers = {g["game_id"] for g in complete_games}
            truth = truth | {
                "games": complete_games,
                "events": [e for e in truth["events"] if e.get("game_id") in identifiers],
                "ranges": [
                    {"start_us": g["start_us"] - 2_000_000, "end_us": g["end_us"] + 2_000_000}
                    for g in complete_games
                    if g.get("end_us") is not None
                ],
            }
        ranges = truth.get("ranges", []) if truth else []

        def scoped(e):
            return not ranges or any(r["start_us"] <= e["pts_us"] <= r["end_us"] for r in ranges)

        automatic = [e for e in session.rows("events") if scoped(e)]
        effective = [e for e in session.effective_events() if scoped(e)]
        if truth:
            truth = truth | {"events": [e for e in truth["events"] if scoped(e)]}
        intervals = ranges or (
            [
                {"start_us": g["start_us"], "end_us": g["end_us"]}
                for g in truth.get("games", [])
                if g.get("end_us") is not None
            ]
            if truth
            else []
        )
        gaps = [
            g
            for g in session.gap_intervals(include_analysis=True)
            if not intervals
            or any(g["start_us"] <= r["end_us"] and g["end_us"] >= r["start_us"] for r in intervals)
        ]
        report = {
            "version": 3,
            "labelled": truth is not None,
            "analysis_mode": mode,
            "ranges": ranges,
            "scope": "人工指定区间" if ranges else "整个会话",
            "unconfirmed_events": sum(
                not e.get("confirmed", True) for e in automatic if e["kind"] in ("play", "pass")
            ),
            "issues": len([i for i in session.rows("issues") if i["code"] != "gap_closed"]),
            "gaps": gaps,
            "automatic_games": session.game_states(automatic),
            "abandoned_deals": sorted(_abandoned_games(automatic)),
            "corrected_games": session.game_states(effective),
            "boundary_check": {
                "labelled": bool(truth and truth.get("games")),
                "observed_starts": sum(e["kind"] == "game_start" for e in automatic),
                "observed_ends": sum(e["kind"] == "game_end" for e in automatic),
            },
        }
        for name, events in (("automatic", automatic), ("corrected", effective)):
            if truth is None:
                report[name] = {
                    k: {
                        "events": sum(e["kind"] == k and e.get("confirmed", True) for e in events),
                        "precision": None,
                        "recall": None,
                    }
                    for k in ("play", "pass")
                }
            else:
                report[name], boundary, details = _score(events, truth, session.media_end_us())
                report["boundary_check"][name] = boundary
                report[name + "_details"] = details
                report[name + "_state_mismatches"] = _state_checks(events, truth, details)
                report["boundary_check"]["expected_games"] = len(truth.get("games", []))
        latencies = sorted(
            e.get("data", {}).get(
                "live_confirmation_delay_us",
                e.get("data", {}).get("confirmed_at_us", e["pts_us"])
                - e.get("data", {}).get("first_evidence_us", e["pts_us"]),
            )
            for e in automatic
            if e["kind"] in ("play", "pass")
            and e.get("confirmed", True)
            and e.get("data", {}).get("confirmed_at_us") is not None
        )
        report["confirmation_delay"] = {
            "basis": (
                "实时处理完成时间估计"
                if mode == "live"
                else "原速回放处理完成时间估计（非手机实时）"
                if mode == "paced_replay"
                else "源帧证据跨度"
            ),
            "samples": len(latencies),
            "p50_us": latencies[(len(latencies) - 1) // 2] if latencies else None,
            "p95_us": latencies[min(len(latencies) - 1, int(len(latencies) * 0.95))] if latencies else None,
            "maximum_us": max(latencies, default=None),
        }
        report["live"] = report["automatic"] if mode == "live" else None
        report["offline"] = report["automatic"] if mode == "offline" else None
        # 原速本地回放有实际墙钟耗时，但不经过手机传输，必须保留独立成绩列。
        report["paced"] = report["automatic"] if mode == "paced_replay" else None
        report["manual"] = report["corrected"]
        report["labels_sha256"] = hashlib.sha256(labels.read_bytes()).hexdigest() if labels else None
        reasons = []
        if meta.get("acceptance_outcome", {}).get("disposition") == "development":
            reasons.append("该验收批次已失败并转为开发回归资料，不能再次作为独立验收")
        if mode != "live":
            reasons.append("该版本不是实时自动记录")
        games = truth.get("games", []) if truth else []
        if len(games) < 5 or not {"self", "left", "right"}.issubset({g.get("landlord") for g in games}):
            reasons.append("不足五个完整牌局或未覆盖三个地主座位")
        if not truth or sum(v.get("truth", 0) for v in report["automatic"].values()) < 200:
            reasons.append("独立标注事件不足 200 个")
        if not truth or not all(
            v.get("precision") is not None and v["precision"] >= 0.99 and v["recall"] >= 0.95
            for v in report["automatic"].values()
        ):
            reasons.append("实时出牌或不出准确率未达到目标")
        if not report["boundary_check"].get("automatic", {}).get("passed"):
            reasons.append("分局边界未通过核对")
        if gaps:
            reasons.append("验收牌局存在采集或实时分析缺口")
        provenance = truth.get("source", {}) if truth else {}
        if (
            not meta.get("acceptance")
            or provenance.get("session_id") != meta.get("session_id")
            or not truth
            or not truth.get("independent_review")
        ):
            reasons.append("缺少冻结版本的新批次及独立标注来源证明")
        if original_truth and original_truth.get("version") == 1:
            report["provenance"] = _provenance(session, original_truth)
            if not report["provenance"]["passed"]:
                reasons.extend(report["provenance"]["problems"])
        else:
            reasons.append("旧片段标签缺少完整独立标注与来源核验")
        if report.get("automatic_state_mismatches"):
            reasons.append("实时状态与独立逐动作检查点不一致")
        if any(
            s.get("complete")
            for ident, s in report["automatic_games"].items()
            if ident
            not in {p["predicted"] for p in report.get("automatic_details", {}).get("game_pairs", [])}
        ):
            reasons.append("存在未核对的完整可信牌局标记")
        report["automatic_target_passed"] = not reasons
        metrics = meta.get("metrics", {})
        report["capture_target_passed"] = bool(
            meta.get("status") == "stopped"
            and metrics.get("elapsed_seconds", 0) >= 1800
            and metrics.get("decoded_fps", 0) >= 10
            and metrics.get("analyzed_fps", 0) >= 5
            and not metrics.get("unfinished_threads", [])
            and metrics.get("pending_write", 1) == 0
            and metrics.get("pending_decode", 1) == 0
            and metrics.get("pending_analysis", 0) == 0
        )
        samples = metrics.get("queue_samples", [])
        growing = any(
            all(b.get(key, 0) > a.get(key, 0) for a, b in zip(samples[i : i + 6], samples[i + 1 : i + 7]))
            for key in ("write", "decode", "evidence", "analysis")
            for i in range(max(0, len(samples) - 6))
        )
        report["queue_check"] = {
            "samples": len(samples),
            "sustained_growth": growing,
            "max_write": metrics.get("max_write_queue"),
            "max_decode": metrics.get("max_decode_queue"),
            "max_analysis": metrics.get("max_analysis_queue"),
            "analysis_skipped_busy": metrics.get("analysis_skipped_busy", 0),
        }
        if len(samples) < 12 or growing:
            report["capture_target_passed"] = False
        if not report["capture_target_passed"]:
            reasons.append("最终版本连续采集性能尚未通过验收")
        corrected_states = report["corrected_games"]
        corrected_pairs = report.get("corrected_details", {}).get("game_pairs", [])
        report["corrected_complete_games"] = sum(
            bool(corrected_states.get(p["predicted"], {}).get("complete")) for p in corrected_pairs
        )
        report["correction_target_passed"] = bool(
            truth
            and report["corrected_complete_games"] >= 5
            and all(v["missed"] == 0 and v["false_positive"] == 0 for v in report["corrected"].values())
            and report["boundary_check"].get("corrected", {}).get("passed")
            and not report.get("corrected_state_mismatches")
        )
        if not report["correction_target_passed"]:
            reasons.append("五局完整纠错的一致性尚未通过核对")
        verification = meta.get("correction_verification", {})
        effective_digest = hashlib.sha256(
            json.dumps(session.effective_events(), sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest()
        report["reproducibility_check"] = verification
        if (
            not verification.get("passed")
            or verification.get("effective_events_sha256") != effective_digest
            or verification.get("raw_observations_sha256") != observation_hash(session)
        ):
            reasons.append("当前修订版本尚未通过关闭重开、撤销及相同观察重放检查")
            report["correction_target_passed"] = False
        report["target_passed"] = not reasons
        report["unmet"] = reasons
        return report
    finally:
        session.close()
