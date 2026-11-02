"""免训练识别、拒识、区域路由与可重放快照；所有设备接口均不参与本文件。"""

import hashlib
import importlib.util
import json
from collections import Counter
from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from doudizhu_lab.domain.models import (
    CardObservation,
    Frame,
    PixelRect,
    Recognition,
    RegionResult,
    RegionStatus,
)
from doudizhu_lab.evaluation.vision import compare_vision
from doudizhu_lab.storage.config import Profile, Store, read_rgb
from doudizhu_lab.storage.session import SessionStore, release_fingerprint
from doudizhu_lab.vision.assets import ASSETS, model_path
from doudizhu_lab.vision.cards import Recognizer
from doudizhu_lab.vision.factory import create_recognizer
from doudizhu_lab.vision.ocr import MixedCards, OcrGlyphs, parse_ranks
from tests.support import local_data_root

DATA = local_data_root()
READY = importlib.util.find_spec("rapidocr") is not None and all(
    (DATA / "models/pretrained-vision" / a["file"]).exists() for a in ASSETS.values()
)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("10 10 AA2", ["10", "10", "A", "A", "2"]),
        ("jQk", ["J", "Q", "K"]),
        ("O", None),
        ("I0", None),
        ("JOKER", None),
        ("23abc", None),
        ("", None),
    ],
)
def test_rank_parser_preserves_duplicates_without_guessing(text, expected):
    assert parse_ranks(text) == expected


def test_model_hash_missing_and_corrupt_are_explicit_errors(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="setup-vision"):
        model_path(tmp_path, "ocr_en")
    target = tmp_path / "models/pretrained-vision" / ASSETS["ocr_en"]["file"]
    target.parent.mkdir(parents=True)
    target.write_bytes(b"test model")
    with pytest.raises(ValueError, match="校验失败"):
        model_path(tmp_path, "ocr_en")
    monkeypatch.setitem(ASSETS["ocr_en"], "sha256", hashlib.sha256(target.read_bytes()).hexdigest())
    assert model_path(tmp_path, "ocr_en") == target


def test_ocr_rejects_ambiguous_characters_and_preserves_unknown():
    recognizer = OcrGlyphs.__new__(OcrGlyphs)
    recognizer.engine = Mock(return_value=SimpleNamespace(txts=["O"], scores=[0.99]))
    crop = np.full((60, 35, 3), 255, np.uint8)
    assert recognizer.classify(crop, "hand", Profile())[0] is None
    recognizer.engine.return_value = SimpleNamespace(txts=["10"], scores=[0.49])
    assert recognizer.classify(crop, "hand", Profile())[0] is None
    recognizer.engine.return_value = SimpleNamespace(txts=["10"], scores=[0.99])
    assert recognizer.classify(crop, "hand", Profile())[0] == "10"
    assert recognizer.engine.call_args.kwargs == {"use_det": False, "use_cls": False}


def test_mixed_backend_routes_once_and_does_not_fill_unknown_hand_with_template(frame):
    mixed = MixedCards.__new__(MixedCards)
    mixed.hand, mixed.other = Mock(), Mock()
    unknown = CardObservation(None, PixelRect(1, 2, 3, 4), 0.4, "OCR无法确认")
    mixed.hand.recognize.return_value = Recognition(
        [RegionResult("hand", RegionStatus.UNCERTAIN, [unknown])], frame.captured_at, 1
    )
    profile = Profile()
    mixed.other.recognize.return_value = Recognition(
        [RegionResult(k, RegionStatus.EMPTY) for k in profile.regions if k != "hand"], frame.captured_at, 2
    )
    result = mixed.recognize(frame, profile, stabilize=True)
    assert [r.region for r in result.regions] == list(profile.regions)
    assert result.regions[0].cards == [unknown] and result.regions[0].status == RegionStatus.UNCERTAIN
    assert mixed.hand.recognize.call_args.kwargs["regions"] == {"hand"}
    assert mixed.other.recognize.call_args.kwargs["regions"] == set(profile.regions) - {"hand"}
    assert result.frame_time == frame.captured_at


def test_config_legacy_default_and_explicit_factory(tmp_path, monkeypatch):
    store = Store(tmp_path)
    assert store.profile.hand_classifier == "template"
    assert isinstance(create_recognizer(store.template_dir, store.profile, model_root=tmp_path), Recognizer)
    store.profile.hand_classifier = "pretrained-ocr"
    store.save()
    assert Store(tmp_path).profile.hand_classifier == "pretrained-ocr"
    constructor = Mock(return_value="仅测试的OCR")
    monkeypatch.setattr("doudizhu_lab.vision.ocr.MixedCards", constructor)
    assert create_recognizer(store.template_dir, store.profile, model_root=tmp_path) == "仅测试的OCR"
    store.profile.hand_classifier = "unknown"
    with pytest.raises(ValueError, match="配置无效"):
        store.save()
    raw = {"version": 3, "profile": asdict(Profile())}
    raw["profile"].pop("hand_classifier")
    store.settings_path.write_text(json.dumps(raw), encoding="utf8")
    assert Store(tmp_path).profile.hand_classifier == "template"


def test_ocr_session_snapshots_models_and_detects_later_tampering(tmp_path, monkeypatch):
    root = tmp_path / "models/pretrained-vision"
    for key in ("ocr_det", "ocr_en", "ocr_zh"):
        target = root / ASSETS[key]["file"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(key.encode())
        monkeypatch.setitem(ASSETS[key], "sha256", hashlib.sha256(target.read_bytes()).hexdigest())
    session = SessionStore.create(
        tmp_path, Profile(hand_classifier="pretrained-ocr"), [], "test", model_root=tmp_path
    )
    try:
        session.verify_snapshot()
        meta = session.metadata()
        assert len(meta["vision_model_manifest"]) == 3
        fingerprint = release_fingerprint(meta)
        target = session.folder / meta["vision_model_manifest"][0]["path"]
        target.write_bytes(b"changed")
        assert release_fingerprint(meta) == fingerprint  # 声称的哈希不变仍须检验真实文件。
        with pytest.raises(ValueError, match="校验失败"):
            session.verify_snapshot()
    finally:
        session.close()


def test_smoke_comparison_cannot_pass_and_counts_duplicates(tmp_path, monkeypatch):
    fake = Mock(templates=[])
    fake.recognize.return_value = Recognition([], "time", 10)
    monkeypatch.setattr("doudizhu_lab.evaluation.vision.build_backend", lambda *a: fake)

    def evaluate(*args):
        args[1].recognize(Mock(), Profile())
        return {
            "target_passed": True,
            "details": [
                {"regions": {"hand": {"expected": ["3", "3", "A"], "predicted": ["3", "3", "3", "K"]}}}
            ],
        }

    monkeypatch.setattr("doudizhu_lab.evaluation.vision.evaluate", evaluate)
    monkeypatch.setattr("doudizhu_lab.evaluation.vision.TableRecognizer._bottom_layout", lambda f, p: p)
    manifest = tmp_path / "labels.json"
    manifest.write_text(json.dumps({"samples": [{"image": "one.png"}]}), encoding="utf8")
    output = tmp_path / "report.json"
    result = compare_vision(Store(tmp_path), [manifest], backend="hand-ocr", output=output, limit=1)
    assert not result["rank_metrics_passed"] and output.exists()
    counts = result["datasets"][0]["exact_sets"]
    assert counts["hand_exact"] == 0 and counts["excess_existing_ranks"] == 1
    assert counts["absent_rank_false_positives"] == 1


@pytest.mark.local_data
@pytest.mark.skipif(not READY, reason="可选vision依赖及本地模型未安装")
def test_real_ocr_large_hand_center_bottom_and_selected_joker_are_compatible():
    recognizer = MixedCards(DATA, DATA / "templates")
    expected = Counter(["SJ", "2", "2", "A", "A", "A", "J", "9", "8", "8", "7", "6", "6", "5", "4", "4", "3"])
    for name in ("live-cancelled.png", "live-selected.png"):
        image = DATA / "bot-evaluation" / name
        if not image.exists():
            pytest.skip("真机手牌开发截图未随源码分发")
        frame = Frame("offline", read_rgb(image), 0, name, 0, "file")
        result = {r.region: r for r in recognizer.recognize(frame, Profile()).regions}
        assert Counter(c.rank for c in result["hand"].cards) == expected
        assert Counter(c.rank for c in result["bottom"].cards) == Counter(["2", "A", "9"])
        assert not recognizer.hand.templates


@pytest.mark.gui
def test_gui_preserves_explicit_ocr_mode_without_phone_or_implicit_execution(qtbot, tmp_path):
    from doudizhu_lab.ui.window import MainWindow

    store = Store(tmp_path)
    store.profile.hand_classifier = "pretrained-ocr"
    window = MainWindow(store, auto_refresh=False, auto_load_model=False, auto_preview=False)
    qtbot.addWidget(window)
    window.show()
    assert window.store.profile.hand_classifier == "pretrained-ocr"
    assert window.canvas.recognition is None and not window.canvas.control_enabled
    assert not window.running and window.bot is None
    window.close()
    qtbot.waitUntil(lambda: not window.isVisible())
