"""完全合成的 JOKER 字形；私人故障图不进入公共测试和发行包。"""

import json

import cv2
import numpy as np
import pytest

from doudizhu_lab.storage.config import Profile, write_rgb
from doudizhu_lab.vision.cards import Recognizer


def joker(color, *, size=0.7, spacing=29):
    rgb = np.full((190, 100, 3), 235, np.uint8)
    for i, letter in enumerate("JOKER"):
        cv2.putText(
            rgb, letter, (32, 35 + i * spacing), cv2.FONT_HERSHEY_SIMPLEX, size, color, 2, cv2.LINE_AA
        )
    return rgb


@pytest.mark.parametrize("color,rank", [((185, 30, 30), "BJ"), ((25, 25, 25), "SJ")])
def test_split_o_keeps_five_complete_letters_and_existing_threshold(tmp_path, color, rank):
    normal = joker(color)
    write_rgb(tmp_path / "joker.png", normal[18:154, 30:52])
    (tmp_path / "joker.json").write_text(json.dumps({"rank": rank, "group": "table"}), encoding="utf-8")
    damaged = normal.copy()
    damaged[45:68, 40:42] = 235
    recognizer = Recognizer(tmp_path, include_bundled=False)
    boxes = recognizer.candidates(damaged, 50)
    assert len(boxes) == 1 and boxes[0].height >= 130
    box = boxes[0]
    result = recognizer.classify(
        damaged[box.y : box.y + box.height, box.x : box.x + box.width], "table", Profile()
    )
    assert result[0] == rank and result[1] >= 0.84


@pytest.mark.parametrize("color,rank", [((185, 30, 30), "BJ"), ((25, 25, 25), "SJ")])
def test_small_joker_joined_k_e_remains_a_complete_five_letter_glyph(tmp_path, color, rank):
    normal = joker(color, size=0.5, spacing=18)
    write_rgb(tmp_path / "joker.png", normal)
    (tmp_path / "joker.json").write_text(json.dumps({"rank": rank, "group": "bottom"}), encoding="utf-8")
    joined = normal.copy()
    cv2.line(joined, (33, 70), (33, 78), color, 1)
    recognizer = Recognizer(tmp_path, include_bundled=False)
    boxes = recognizer.candidates(joined, 40)
    assert len(boxes) == 1 and boxes[0].height >= 80
    box = boxes[0]
    assert (
        recognizer.classify(
            joined[box.y : box.y + box.height, box.x : box.x + box.width], "bottom", Profile()
        )[0]
        == rank
    )


@pytest.mark.parametrize("fault", ["missing_r", "ordinary_j", "noise", "wrong_color"])
def test_incomplete_or_unrelated_shapes_are_not_accepted_as_big_joker(tmp_path, fault):
    normal = joker((185, 30, 30))
    write_rgb(tmp_path / "joker.png", normal[18:154, 30:52])
    (tmp_path / "joker.json").write_text(json.dumps({"rank": "BJ", "group": "table"}), encoding="utf-8")
    if fault == "missing_r":
        normal[133:] = 235
    elif fault == "ordinary_j":
        normal[:] = 235
        cv2.putText(normal, "J", (32, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (185, 30, 30), 3)
    elif fault == "noise":
        normal[:] = 235
        for i in range(5):
            cv2.rectangle(normal, (35, 20 + i * 29), (48, 35 + i * 29), (185, 30, 30), -1)
    else:
        normal = joker((25, 25, 25))
    recognizer = Recognizer(tmp_path, include_bundled=False)
    for box in recognizer.candidates(normal, 50):
        assert (
            recognizer.classify(
                normal[box.y : box.y + box.height, box.x : box.x + box.width], "table", Profile()
            )[0]
            != "BJ"
        )
