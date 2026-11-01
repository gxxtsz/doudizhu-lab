"""真机新增的记牌器关闭布局回归；不把顶部记牌器数字当作底牌。"""

import numpy as np
import pytest

from doudizhu_lab.domain.models import Frame, NormalizedRect
from doudizhu_lab.storage.config import Profile, Store, read_rgb
from doudizhu_lab.vision.cards import Recognizer
from doudizhu_lab.vision.table import MarkerLibrary, TableRecognizer
from tests.support import local_data_root


@pytest.mark.local_data
def test_live_center_bottom_cards():
    store = Store(local_data_root())
    path = store.root / "bot-evaluation/live-cancelled.png"
    if not path.exists():
        pytest.skip("未附带本机真机布局样本")
    frame = Frame("replay", read_rgb(path), 0, "center", 0, "replay")
    before = store.profile.regions["bottom"]
    obs, _ = TableRecognizer(Recognizer(store.template_dir), MarkerLibrary(store.root / "markers")).observe(
        frame, store.profile
    )
    assert obs.bottom == ["2", "A", "9"]
    assert store.profile.regions["bottom"] == before


def test_memory_bar_and_custom_region_are_not_overridden():
    profile = Profile()
    rgb = np.zeros((1080, 2400, 3), np.uint8)
    rgb[0:100, 1100:1280] = 240
    frame = Frame("test", rgb, 0, "layout", 0)
    assert TableRecognizer._bottom_layout(frame, profile).regions["bottom"].x == 0.462
    rgb[10:95, 1320:1710] = 220
    assert TableRecognizer._bottom_layout(frame, profile) is profile
    profile.regions["bottom"] = NormalizedRect(0.4, 0.01, 0.08, 0.09)
    rgb[:] = 0
    rgb[0:100, 1100:1280] = 240
    assert TableRecognizer._bottom_layout(frame, profile) is profile
