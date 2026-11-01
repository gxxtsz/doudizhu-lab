"""覆盖本地文件、保守拒识、牌点颜色、重复计数和真实截图回归。"""

import json
from dataclasses import replace
from time import monotonic

import cv2
import numpy as np
import pytest

from doudizhu_lab.domain.models import RANKS, Frame, Recognition, RegionStatus
from doudizhu_lab.evaluation.cards import evaluate
from doudizhu_lab.storage.config import Store, read_rgb, write_rgb
from doudizhu_lab.vision.cards import Recognizer, glyph_color, ink_mask, normalize_glyph, similarity
from tests.support import local_data_root

DATA = local_data_root() / "validation"


def test_chinese_paths_and_export_are_bound_to_same_frame(tmp_path, frame):
    store = Store(tmp_path / "中文配置")
    store.save()
    assert Store(store.root).profile.regions == store.profile.regions
    result = Recognition([], frame.captured_at, 2)
    folder = store.export(frame, result)
    assert read_rgb(folder / "画面.png").shape == frame.rgb.shape
    payload = json.loads((folder / "识别结果.json").read_text(encoding="utf8"))
    assert payload["device"] == frame.serial
    assert payload["recognition"]["frame_time"] == frame.captured_at
    with pytest.raises(ValueError, match="同一帧"):
        store.export(frame, Recognition([], "wrong", 0))


def test_corrupt_config_and_empty_image_do_not_succeed(tmp_path):
    (tmp_path / "settings.json").write_text('{"profile":{"glyph_heights":{"hand":-1}}}')
    with pytest.raises(ValueError, match="配置文件损坏"):
        Store(tmp_path)
    (tmp_path / "空图.png").touch()
    with pytest.raises(ValueError, match="为空"):
        read_rgb(tmp_path / "空图.png")


def test_template_colors_and_rejection(tmp_path):
    store = Store(tmp_path)
    glyph = np.full((64, 48, 3), 220, np.uint8)
    cv2.putText(glyph, "J", (8, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.7, (180, 30, 30), 4)
    assert glyph_color(glyph) == "red"
    store.add_template(glyph, "BJ", "hand", "template_frame")
    recognizer = Recognizer(store.template_dir, include_bundled=False)
    assert recognizer.classify(glyph, "hand", store.profile)[0] == "BJ"
    black = glyph.copy()
    black[ink_mask(glyph) > 0] = (20, 20, 20)
    assert recognizer.classify(black, "hand", store.profile)[0] is None
    assert recognizer.classify(glyph, "bottom", store.profile)[0] is None
    assert recognizer.classify(np.full_like(glyph, 255), "hand", store.profile)[0] is None
    norm = normalize_glyph(ink_mask(glyph))
    assert similarity(norm, norm) == pytest.approx(1, abs=1e-5)


def test_missing_or_invalid_templates_and_non_table(tmp_path, frame):
    store = Store(tmp_path)
    (store.template_dir / "bad.json").write_text('{"rank":"X","group":"hand"}')
    recognizer = Recognizer(store.template_dir, include_bundled=False)
    assert recognizer.load_errors
    assert recognizer.coverage()["hand"] == []
    assert all(r.status == RegionStatus.NOT_TABLE for r in recognizer.recognize(frame, store.profile).regions)


@pytest.mark.local_data
@pytest.mark.skipif(not DATA.exists(), reason="真实手机样本属于本地忽略数据，分发代码时不携带")
def test_real_holdout_accuracy_and_all_ranks(tmp_path):
    store = Store(tmp_path)
    report = evaluate(DATA / "labels.json", Recognizer(store.template_dir), store.profile)
    assert report["samples"] >= 30
    assert report["ranks_covered"] == RANKS
    assert report["target_passed"]


@pytest.mark.local_data
@pytest.mark.skipif(not DATA.exists(), reason="本地真实截图回归")
def test_missing_templates_empty_regions_and_temporal_changes(tmp_path):
    store = Store(tmp_path)
    image = read_rgb(DATA / "images/frame_0006.png")
    frame = Frame("mock", image, monotonic(), "one", 0)
    missing = Recognizer(store.template_dir, include_bundled=False)
    results = missing.recognize(frame, store.profile).regions
    assert results[0].status == RegionStatus.NO_TEMPLATES
    assert not any(c.rank for c in results[0].cards)
    assert results[2].status == RegionStatus.EMPTY
    live = Recognizer(store.template_dir)
    assert live.recognize(frame, store.profile, stabilize=True).regions[0].status == RegionStatus.UNCERTAIN
    stable = live.recognize(
        replace(frame, started=frame.started + 0.5, captured_at="two"), store.profile, stabilize=True
    )
    assert stable.regions[0].status == RegionStatus.OK
    changed = read_rgb(DATA / "images/frame_0024.png")
    moving = live.recognize(
        replace(frame, rgb=changed, started=frame.started + 1, captured_at="three"),
        store.profile,
        stabilize=True,
    )
    assert moving.regions[0].status == RegionStatus.UNCERTAIN
    assert not any(c.rank for c in moving.regions[0].cards)


def test_validation_rejects_template_source_overlap(tmp_path):
    store = Store(tmp_path / "store")
    glyph = np.full((60, 50, 3), 220, np.uint8)
    cv2.putText(glyph, "3", (4, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (20, 20, 20), 3)
    store.add_template(glyph, "3", "hand", "source")
    write_rgb(tmp_path / "source.png", glyph)
    manifest = tmp_path / "labels.json"
    manifest.write_text(json.dumps({"samples": [{"image": "source.png", "regions": {}}]}))
    with pytest.raises(ValueError, match="来源重复"):
        evaluate(manifest, Recognizer(store.template_dir, include_bundled=False), store.profile)


@pytest.mark.local_data
@pytest.mark.skipif(not (DATA.parent / "negative").exists(), reason="本地异常场景截图")
def test_settlement_covered_bottom_and_animation_are_not_success(tmp_path):
    store = Store(tmp_path)
    recognizer = Recognizer(store.template_dir)

    def load(name):
        return Frame("mock", read_rgb(DATA.parent / "negative" / f"{name}.png"), monotonic(), name, 0)

    assert all(
        r.status == RegionStatus.NOT_TABLE
        for r in recognizer.recognize(load("settlement"), store.profile).regions
    )
    covered = recognizer.recognize(load("covered"), store.profile)
    assert covered.regions[1].status == RegionStatus.UNCERTAIN
    assert covered.regions[1].cards == []
    moving = recognizer.recognize(load("animation"), store.profile, stabilize=True)
    assert moving.regions[0].status != RegionStatus.OK
