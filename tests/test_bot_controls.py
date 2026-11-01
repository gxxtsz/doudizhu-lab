"""实际旧录像操作对象回归及合成禁用、动画、全选；全部为离线检查。"""

import json
from collections import Counter
from dataclasses import asdict, replace

import cv2
import pytest

from doudizhu_lab.domain.bot_types import ObservationSnapshot
from doudizhu_lab.domain.models import Frame, PixelRect
from doudizhu_lab.domain.tracker import GameState
from doudizhu_lab.storage.config import Store, read_rgb
from doudizhu_lab.vision.cards import Recognizer
from doudizhu_lab.vision.controls import ControlRecognizer
from doudizhu_lab.vision.table import MarkerLibrary, TableRecognizer
from tests.support import local_data_root


@pytest.fixture
def controls():
    root = local_data_root()
    if not (root / "bot-validation/818.png").exists():
        pytest.skip("缺少本机旧录像开发样本")
    store = Store(root)
    return (
        store,
        TableRecognizer(Recognizer(store.template_dir), MarkerLibrary(root / "markers")),
        ControlRecognizer(root / "controls"),
    )


def snapshot(controls, name):
    store, table, control = controls
    f = Frame(
        "replay", read_rgb(store.root / f"bot-validation/{name}.png"), 0, name, 0, "replay", name, "old", 1
    )
    obs, recognition = table.observe(f, store.profile)
    snap = ObservationSnapshot.create(f, recognition, obs, GameState(), [])
    return control.enrich(snap, store.profile), recognition


@pytest.mark.local_data
@pytest.mark.parametrize(
    "name,count,selected",
    [
        ("816", 20, []),
        ("817.8", 20, []),
        ("818", 20, list("66778899")),
        ("873.7", 4, []),
        ("small-joker", 17, []),
    ],
)
def test_old_video_layout_independent_frames(controls, name, count, selected):
    snap, _ = snapshot(controls, name)
    assert not snap.control_reason and len(snap.hand) == count
    assert Counter(c.rank for c in snap.hand if c.selected) == Counter(selected)
    assert all(c.selected is not None for c in snap.hand)


@pytest.mark.local_data
def test_all_selected_and_animation_height(controls):
    store, _, control = controls
    snap, recognition = snapshot(controls, "816")
    for shift, expected in ((30, True), (15, None)):
        rgb = snap.frame.rgb.copy()
        rgb[:-shift] = rgb[shift:].copy()
        changed = replace(snap.frame, rgb=rgb)
        result = asdict(recognition)
        for region in result["regions"]:
            if region["region"] == "hand":
                for card in region["cards"]:
                    card["box"]["y"] -= shift
        out = control.enrich(replace(snap, frame=changed, recognition_json=json.dumps(result)), store.profile)
        assert all(c.selected is expected for c in out.hand)


@pytest.mark.local_data
def test_button_moved_disabled_and_wrong_phase(controls):
    store, _, control = controls
    snap, _ = snapshot(controls, "816")
    obs = {"phase": "playing", "actor": "self"}
    buttons = control.detect_buttons(snap.frame, obs, store.profile)
    play = next(b for b in buttons if b.kind == "play")
    rgb = snap.frame.rgb.copy()
    box = play.hit
    crop = box.crop(rgb).copy()
    rgb[box.y : box.y + box.height, box.x : box.x + box.width] = 0
    moved = PixelRect(box.x + 70, box.y, box.width, box.height)
    rgb[moved.y : moved.y + moved.height, moved.x : moved.x + moved.width] = crop
    found = control.detect_buttons(replace(snap.frame, rgb=rgb), obs, store.profile)
    assert next(b for b in found if b.kind == "play").hit.x == moved.x
    gray = cv2.cvtColor(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)
    assert not control.detect_buttons(replace(snap.frame, rgb=gray), obs, store.profile)
    assert not control.detect_buttons(snap.frame, {"phase": "bidding", "actor": "self"}, store.profile)


@pytest.mark.local_data
def test_cannot_beat_is_own_turn_evidence(controls):
    store, table, control = controls
    snap, _ = snapshot(controls, "cannot-beat")
    obs, _ = table.observe(snap.frame, store.profile)
    control.augment_observation(snap.frame, obs, store.profile)
    assert obs.actor == "self" and obs.phase == "playing"
    assert any(
        b.label == "要不起" and b.enabled
        for b in control.detect_buttons(snap.frame, asdict(obs), store.profile)
    )


def test_v2_config_backed_up_and_bot_bounds(tmp_path):
    store = Store(tmp_path)
    store.save()
    data = json.loads(store.settings_path.read_text(encoding="utf8"))
    data["version"] = 2
    data["profile"].pop("bot_controls")
    data["profile"].pop("bot_settings")
    original = json.dumps(data, ensure_ascii=False)
    store.settings_path.write_text(original, encoding="utf8")
    upgraded = Store(tmp_path)
    assert not upgraded.profile.bot_controls["verified"]
    upgraded.save()
    assert (tmp_path / "settings.v2.bak.json").read_text(encoding="utf8") == original
    upgraded.profile.bot_settings["max_frame_age"] = 10
    with pytest.raises(ValueError):
        upgraded.save()
