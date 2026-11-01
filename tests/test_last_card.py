"""末手王的独立视觉证据；拒绝普通J、三字母、低分及缺少末手标记的场景。"""

import importlib.util
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from doudizhu_lab.domain.models import Frame, Recognition, RegionResult, RegionStatus
from doudizhu_lab.storage.config import Profile, Store, read_rgb
from doudizhu_lab.vision.cards import Recognizer
from doudizhu_lab.vision.last_card import LastCardOCR
from doudizhu_lab.vision.table import MarkerLibrary, TableObservation, TableRecognizer
from tests.support import local_data_root


@pytest.mark.parametrize(
    ("text", "score", "winner", "count", "expected"),
    [
        ("JOKE", 0.99, True, 0, "BJ"),
        ("JOKER", 0.99, True, 0, "BJ"),
        ("JOK", 0.99, True, 0, None),
        ("J", 0.99, True, 0, None),
        ("JOKE", 0.94, True, 0, None),
        ("JOKE", 0.99, False, 0, None),
        ("JOKE", 0.99, True, 1, None),
    ],
)
def test_partial_word_requires_four_letters_color_and_explicit_last_card(
    text, score, winner, count, expected
):
    image = np.full((300, 600, 3), 240, np.uint8)
    image[60:120:3, 107:118] = (190, 20, 20)
    frame = Frame("offline", image, 0, "one", 0, "file", frame_id="one", media_pts_us=1)
    observation = TableObservation(
        "one",
        1,
        remaining={"left": count},
        plays={"left": None},
        marker_evidence={"winner_left": {"accepted": winner, "box": [100, 130, 50, 20]}},
    )
    result = Recognition([RegionResult(k, RegionStatus.UNCERTAIN) for k in Profile().regions], "one", 0)
    recognizer = LastCardOCR.__new__(LastCardOCR)
    recognizer.engine = Mock(
        return_value=SimpleNamespace(
            boxes=np.array([[[12, 5], [26, 5], [26, 65], [12, 65]]]), txts=[text], scores=[score]
        )
    )
    recognizer.enrich(frame, observation, result, Profile())
    assert observation.plays["left"] == ([expected] if expected else None)


@pytest.mark.local_data
def test_real_last_joker_covered_by_last_play_label():
    root = local_data_root()
    path = root / "sessions/20261101_141157_793962/evidence/8e0fa480-1574.png"
    if not path.exists() or importlib.util.find_spec("rapidocr") is None:
        pytest.skip("本机开发截图或可选OCR依赖未分发")
    store = Store(root)
    frame = Frame("offline", read_rgb(path), 0, "last", 0, "file", frame_id="last", media_pts_us=1)
    obs, result = TableRecognizer(Recognizer(store.template_dir), MarkerLibrary(root / "markers")).observe(
        frame, store.profile
    )
    assert obs.remaining["left"] == 0 and obs.plays["left"] is None
    LastCardOCR(root).enrich(frame, obs, result, store.profile)
    assert obs.plays["left"] == ["BJ"]
    assert obs.marker_evidence["last_card_left"]["frame_id"] == frame.frame_id
