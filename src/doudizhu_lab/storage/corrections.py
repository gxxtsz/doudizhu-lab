"""使用已独立核对的完整标注追加纠错，并验证关闭重开、撤销及确定性重放。"""

import hashlib
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from doudizhu_lab.domain.table import TableObservation
from doudizhu_lab.domain.tracker import GameTracker
from doudizhu_lab.paths import engine_hashes
from doudizhu_lab.storage.session import SessionStore, game_intervals
from doudizhu_lab.storage.truth import media_manifest, truth_events, validate_truth


def observation_hash(session: SessionStore) -> str:
    """按原始追加顺序计算观察内容哈希，排除 SQLite 页布局、修订和导出文件的变化。"""
    digest = hashlib.sha256()
    for observation in session.rows("observations"):
        digest.update(
            json.dumps(observation, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        )
        digest.update(b"\n")
    return digest.hexdigest()


def apply_label_corrections(folder: Path, labels: Path) -> dict:
    """将已发布的独立标注作为人工修订追加，原观察、原事件和物理采集缺口均保留。

    此方向是“独立真值 → 纠错”，不会导出或复制机器预测作为真值。每条修订记录
    引用标签文件哈希；尚未整局复核的草稿、不同来源录像或跨区间分局均拒绝应用。
    """
    truth = json.loads(labels.read_text(encoding="utf-8"))
    checked = validate_truth(truth, require_review=True)
    if not truth.get("independent_review") or not checked["valid"]:
        raise ValueError("请先逐动作核对并发布独立标注，草稿不能批量应用为纠错")
    selected = [g for g in truth["games"] if not g.get("incomplete")]
    if not selected:
        raise ValueError("没有可应用的完整标注牌局")
    session = SessionStore(folder)
    try:
        if truth.get("source", {}).get("session_id") != session.metadata().get("session_id"):
            raise ValueError("标注与会话编号不一致")
        if truth["source"].get("video_manifest") != media_manifest(folder):
            raise ValueError("标注与录像文件哈希不一致")
        current = session.effective_events()
        spans = game_intervals(current, session.media_end_us())
        owners = set()
        for ident, (begin, end) in spans.items():
            overlap = [g for g in selected if g["start_us"] <= end and begin <= g["end_us"]]
            if not overlap:
                continue
            if (
                begin < min(g["start_us"] for g in overlap) - 2_000_000
                or end > max(g["end_us"] for g in overlap) + 2_000_000
            ):
                raise ValueError("自动分局跨越未核对范围，请先在回放页调整边界")
            owners.add(ident)
        batch = {
            "batch_id": uuid4().hex,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "labels_sha256": hashlib.sha256(labels.read_bytes()).hexdigest(),
            "raw_observations_sha256": observation_hash(session),
            "revisions": [],
            "games": [g["game_id"] for g in selected],
            "status": "applying",
        }
        batches = session.metadata().get("correction_batches", []) + [batch]
        session.set_meta("correction_batches", batches)
        for event in current:
            if event["game_id"] in owners:
                batch["revisions"].append(
                    session.revise(
                        "delete",
                        event["event_id"],
                        {"reason": "已按独立完整标注重新核对", "labels_sha256": batch["labels_sha256"]},
                    )
                )
        identifiers = {g["game_id"] for g in selected}
        for event in truth_events(truth):
            if event["game_id"] not in identifiers:
                continue
            event.setdefault(
                "frame_id",
                next(g.get("start_frame", "") for g in selected if g["game_id"] == event["game_id"]),
            )
            event["data"] = event.get("data", {}) | {
                "labels_sha256": batch["labels_sha256"],
                "independent_event_id": event["event_id"],
            }
            batch["revisions"].append(session.revise("insert", "", event))
        batch["status"] = "applied"
        session.set_meta("correction_batches", batches)
        session.export()
        return batch
    finally:
        session.close()


def verify_correction_replay(folder: Path, batch_id: str) -> dict:
    """实际关闭重开、撤销整批、再次恢复并比较状态；所有撤销也作为追加修订保留。

    同时从原始观察重新执行当前快照的状态机。源码版本不同则明确拒绝该项一致性，
    调用者须在会话保存的 engine 中运行相同命令，而不能宣称新算法复现了旧算法。
    """
    session = SessionStore(folder)
    try:
        meta = session.metadata()
        batch = next((b for b in meta.get("correction_batches", []) if b["batch_id"] == batch_id), None)
        if batch is None or batch["status"] != "applied":
            raise ValueError("找不到已完成的纠错批次")
        raw = observation_hash(session)
        expected_events, expected_states = session.effective_events(), session.game_states()
        session.close()
        session = SessionStore(folder)
        reopened = session.effective_events() == expected_events and session.game_states() == expected_states
        undo_ids = [
            session.revise("undo", str(seq), {"verification": batch_id})
            for seq in reversed(batch["revisions"])
        ]
        reverted = session.effective_events()
        for seq in reversed(undo_ids):
            session.revise("undo", str(seq), {"verification": batch_id})
        restored = session.effective_events() == expected_events and session.game_states() == expected_states
        unchanged = raw == observation_hash(session) == batch["raw_observations_sha256"]
        hashes = meta.get("engine_sha256", {})
        same_engine = bool(hashes) and all(
            engine_hashes().get(name) == digest for name, digest in hashes.items()
        )
        automatic_equal = False
        if same_engine:
            tracker = GameTracker(
                meta["profile"].get("seat_order", []), config=meta["profile"].get("tracker_settings")
            )
            actual = []
            source_events = session.rows("events")
            for value in session.rows("observations"):
                observation = TableObservation(**value)
                for gap in source_events:
                    if (
                        gap["kind"] == "gap"
                        and "capture_lost" in gap.get("data", {})
                        and gap.get("frame_id") == observation.frame_id
                    ):
                        update = tracker.mark_gap(
                            observation, gap["data"]["message"], capture_lost=gap["data"]["capture_lost"]
                        )
                        actual.extend(asdict(e) for e in update.events)
                actual.extend(asdict(e) for e in tracker.observe(observation).events)

            def semantic(events):
                return [
                    e
                    | {
                        "data": {
                            k: v
                            for k, v in e.get("data", {}).items()
                            if k not in ("live_confirmation_delay_us", "processing_finished_utc")
                        }
                    }
                    for e in events
                ]

            automatic_equal = semantic(actual) == semantic(source_events)
        result = {
            "batch_id": batch_id,
            "closed_reopened_equal": reopened,
            "undo_changed_events": reverted != expected_events,
            "undo_redo_equal": restored,
            "raw_observations_unchanged": unchanged,
            "same_engine": same_engine,
            "automatic_observation_replay_equal": automatic_equal,
            "effective_events_sha256": hashlib.sha256(
                json.dumps(expected_events, sort_keys=True, ensure_ascii=False).encode()
            ).hexdigest(),
            "raw_observations_sha256": raw,
            "passed": reopened and restored and unchanged and same_engine and automatic_equal,
        }
        session.set_meta("correction_verification", result)
        session.export()
        return result
    finally:
        session.close()
