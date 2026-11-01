"""人工标注截图的牌张级评测；模板来源与验证来源严格分离。"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from time import monotonic

from doudizhu_lab.domain.models import RANKS, REGION_NAMES, Frame
from doudizhu_lab.storage.config import Profile, read_rgb
from doudizhu_lab.vision.cards import Recognizer


def evaluate(manifest_path: Path, recognizer: Recognizer, profile: Profile) -> dict:
    """按每个区域的牌点多重集合计算精确率与召回率，保留重复张数。

    manifest 的 samples 包含 image、regions；第一版四区、第二版五区。
    第二版允许某区域标为 null（未标注），该区域不计分；空列表专指人工确认无牌。
    validation_spans 用源会话与媒体时间区间隔离整局，模板源帧落入区间即拒绝评测。
    未确认输出不算预测正确，自然计入漏识别。错标牌同时记一次误识别和一次漏识别。
    不使用顶部记牌器推断标签，也不把同一截图同时用作模板与验证集。
    """
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    totals = {key: Counter(tp=0, fp=0, fn=0, unknown=0) for key in REGION_NAMES}
    details, labels_seen, hashes = [], set(), set()
    template_sources = {t.source for t in recognizer.templates if isinstance(t.source, str)}
    source_hashes = {t.source.get("image_sha256") for t in recognizer.templates if isinstance(t.source, dict)}
    source_games = {
        (t.source.get("session"), t.source.get("game"))
        for t in recognizer.templates
        if isinstance(t.source, dict) and t.source.get("game")
    }
    for template in recognizer.templates:
        source = template.source
        if not isinstance(source, dict) or source.get("pts_us") is None:
            continue
        for span in manifest.get("validation_spans", []):
            if (
                source.get("session") == span["session"]
                and span["start_us"] <= source["pts_us"] <= span["end_us"]
            ):
                raise ValueError("验证牌局时间区间含有模板来源帧，不能作为独立验收集")
    labelled_samples = Counter()
    region_ranks = {key: set() for key in REGION_NAMES}
    for sample in manifest["samples"]:
        path = manifest_path.parent / sample["image"]
        if path.stem in template_sources:
            raise ValueError(f"验证截图与模板来源重复：{path.name}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest in hashes:
            raise ValueError(f"验证集存在完全重复截图：{path.name}")
        hashes.add(digest)
        required = set(REGION_NAMES) - {"self_play"} if manifest.get("version", 1) == 1 else set(REGION_NAMES)
        if set(sample["regions"]) != required:
            raise ValueError(f"必须标注本版本要求的区域：{path.name}")
        frame = Frame("offline", read_rgb(path), monotonic(), path.name, 0, "file")
        if (
            hashlib.sha256(frame.rgb.tobytes()).hexdigest() in source_hashes
            or (sample.get("source_session"), sample.get("source_game")) in source_games
        ):
            raise ValueError(f"验证截图与模板来自同帧或同局：{path.name}")
        result = recognizer.recognize(frame, profile)
        row = {"image": sample["image"], "sha256": digest, "regions": {}}
        for region in result.regions:
            if region.region not in sample["regions"]:
                continue
            labels = sample["regions"][region.region]
            if labels is None and manifest.get("version", 1) >= 2:
                continue
            if not isinstance(labels, list):
                raise ValueError(f"人工标签必须为列表或未标注的 null：{path.name}")
            if any(rank not in RANKS for rank in labels):
                raise ValueError(f"验证集牌点无效：{path.name}")
            labelled_samples[region.region] += 1
            region_ranks[region.region].update(labels)
            labels_seen.update(labels)
            expected = Counter(labels)
            predicted = Counter(c.rank for c in region.cards if c.rank)
            matched = sum((expected & predicted).values())
            counts = Counter(
                tp=matched,
                fp=sum(predicted.values()) - matched,
                fn=sum(expected.values()) - matched,
                unknown=sum(c.rank is None for c in region.cards),
            )
            totals[region.region].update(counts)
            row["regions"][region.region] = {
                "expected": labels,
                "predicted": list(predicted.elements()),
                "counts": dict(counts),
                "status": region.status.value,
                "cards": [asdict(c) for c in region.cards],
            }
        details.append(row)
    total = sum(totals.values(), Counter())

    def metrics(counts):
        tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
        return {key: int(counts[key]) for key in ("tp", "fp", "fn", "unknown")} | {
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
        }

    combined = metrics(total)
    by_region = {
        key: metrics(value)
        | {
            "labelled_samples": labelled_samples[key],
            "labelled_cards": value["tp"] + value["fn"],
            "ranks_covered": [rank for rank in RANKS if rank in region_ranks[key]],
        }
        for key, value in totals.items()
    }
    stage2 = manifest.get("version", 1) >= 2
    passed = (
        len(details) >= 60
        and len(labels_seen) == 15
        and all(v["precision"] >= 0.99 and v["recall"] >= 0.95 for v in by_region.values())
        and all(by_region[key]["labelled_cards"] >= 100 for key in ("left", "right", "self_play"))
        if stage2
        else len(details) >= 30
        and len(labels_seen) == 15
        and combined["precision"] >= 0.99
        and combined["recall"] >= 0.95
    )
    return {
        "version": manifest.get("version", 1),
        "validation_spans": manifest.get("validation_spans", []),
        "samples": len(details),
        "ranks_covered": [r for r in RANKS if r in labels_seen],
        "total": combined,
        "by_region": by_region,
        "target_passed": passed,
        "details": details,
    }
