"""免训练识别的统一离线评测与可追溯样例导出，不创建手机客户端。"""

import hashlib
import json
from collections import Counter
from dataclasses import asdict
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from doudizhu_lab.domain.models import Frame
from doudizhu_lab.evaluation.cards import evaluate
from doudizhu_lab.paths import engine_hashes
from doudizhu_lab.storage.config import read_rgb, write_rgb
from doudizhu_lab.vision.assets import ASSETS
from doudizhu_lab.vision.ocr import build_backend
from doudizhu_lab.vision.table import TableRecognizer


class MeasuredRecognizer:
    """给所有方法提供相同区域和计时边界；不将标签或预期张数传给识别器。"""

    def __init__(self, backend):
        self.backend, self.templates, self.times = backend, backend.templates, []

    def recognize(self, frame, profile):
        profile = TableRecognizer._bottom_layout(frame, profile)
        result = self.backend.recognize(frame, profile)
        self.times.append(result.elapsed_ms)
        return result


def compare_vision(store, manifests, *, backend, output, limit=None):
    """按已有人工标签比较相同像素，不以旧模型预测或记牌器数字作为真值。

    limit 只用于开发冒烟，截取的清单不修改原文件；任何冒烟结果都不算达标。
    分数同时报告牌张多重集合、整组完全正确率和CPU耗时，避免单一检测指标掩盖漏牌。
    """
    start = perf_counter()
    recognizer = MeasuredRecognizer(build_backend(backend, store.root))
    load_ms = (perf_counter() - start) * 1000
    reports = []
    for manifest in manifests:
        manifest = Path(manifest).resolve()
        if limit:
            # evaluate 的路径解析基于清单目录，临时清单保留原图的绝对路径。
            data = json.loads(manifest.read_text(encoding="utf8"))
            data["samples"] = data["samples"][:limit]
            for sample in data["samples"]:
                sample["image"] = str((manifest.parent / sample["image"]).resolve())
            temporary = output.parent / f"{backend}-subset-{manifest.parent.name}.json"
            temporary.parent.mkdir(parents=True, exist_ok=True)
            temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf8")
            used = temporary
        else:
            used = manifest
        print(f"开始：{backend} · {manifest.parent.name}", flush=True)
        timing_start = len(recognizer.times)
        report = evaluate(used, recognizer, store.profile)
        report["manifest"] = str(manifest)
        report["manifest_sha256"] = hashlib.sha256(manifest.read_bytes()).hexdigest()
        # evaluate 的每张输出包含原图哈希、原标签及完整位置，额外补充整组指标。
        exact = Counter()
        latencies = recognizer.times[timing_start:]
        for row in report["details"]:
            for key, region in row["regions"].items():
                good = Counter(region["expected"]) == Counter(region["predicted"])
                exact[key + "_groups"] += 1
                exact[key + "_exact"] += good
            exact["labelled_frames"] += 1
            exact["labelled_frames_exact"] += all(
                Counter(r["expected"]) == Counter(r["predicted"]) for r in row["regions"].values()
            )
            for region in row["regions"].values():
                expected, predicted = Counter(region["expected"]), Counter(region["predicted"])
                extra = predicted - expected
                exact["excess_existing_ranks"] += sum(n for rank, n in extra.items() if expected[rank])
                exact["absent_rank_false_positives"] += sum(
                    n for rank, n in extra.items() if not expected[rank]
                )
        # 测量单帧耗时由包装器记录，包含每个区域推理，排除读盘。
        report["exact_sets"] = dict(exact)
        warm = latencies[1:] or latencies
        report["cpu_timing"] = {
            "first_frame_ms": latencies[0],
            "mean_ms": float(np.mean(warm)),
            "median_ms": float(np.median(warm)),
            "p95_ms": float(np.percentile(warm, 95)),
            "recognizer_only_fps": 1000 / float(np.mean(warm)),
            "samples": len(latencies),
        }
        reports.append(report)
    dependencies = {}
    for name in ("numpy", "opencv-python-headless", "onnxruntime", "rapidocr"):
        try:
            dependencies[name] = version(name)
        except PackageNotFoundError:
            dependencies[name] = None
    keys = [] if backend == "template" else ["ocr_det", "ocr_en", "ocr_zh"]
    result = {
        "backend": backend,
        "source": "offline_pretrained_comparison",
        "trained": False,
        "limited": bool(limit),
        "load_ms": load_ms,
        "datasets": reports,
        "rank_metrics_passed": not limit and all(r["target_passed"] for r in reports),
        "live_adopted": False,
        "available_in_gui": backend == "hand-ocr",
        "dependencies": dependencies,
        "model_assets": {k: ASSETS[k] for k in keys},
        "engine_sha256": engine_hashes(),
        "note": "开发回归；不代表真机自动出牌或胜率；模型无训练。",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf8")
    print(f"完成：{backend} · {output}", flush=True)
    return result


def export_prediction(store, image, *, backend, output):
    """导出源图、神经网络框和原始输出说明；结果仅供查看，不产生点击坐标指令。"""
    recognizer = build_backend(backend, store.root)
    image = Path(image)
    frame = Frame("offline", read_rgb(image), 0, image.name, 0, "file")
    result = recognizer.recognize(frame, store.profile)
    output.mkdir(parents=True, exist_ok=True)
    overlay = frame.rgb.copy()
    for region in result.regions:
        for card in region.cards:
            box = card.box
            color = (30, 200, 60) if card.rank else (255, 160, 40)
            cv2.rectangle(overlay, (box.x, box.y), (box.x + box.width, box.y + box.height), color, 2)
            cv2.putText(
                overlay,
                f"{card.rank or '?'} {card.score:.2f}",
                (box.x, max(14, box.y - 3)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
            )
    write_rgb(output / "检测结果.png", overlay)
    payload = {
        "backend": backend,
        "source": str(image.resolve()),
        "image_sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
        "recognition": asdict(result),
        "offline_only": True,
    }
    (output / "检测结果.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf8")
    return payload
