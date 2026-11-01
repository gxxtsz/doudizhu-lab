"""仅明确的操作按钮用途可避免金色色块搜索，普通阶段标记不能压住动作证据。"""

from types import SimpleNamespace

import pytest

from doudizhu_lab.domain.models import Recognition
from doudizhu_lab.storage.config import Profile
from doudizhu_lab.vision.table import TableRecognizer


@pytest.mark.parametrize("explicit_controls", [False, True])
def test_self_pass_search_requires_explicit_control_semantics(frame, explicit_controls):
    profile = Profile()
    profile.indicator_regions = {key: profile.indicator_regions[key] for key in ("pass_self", "phase")}
    calls = []

    def detect(rgb, key, scale, **kwargs):
        calls.append(key)
        label = "playing" if key == "phase" else "过"
        evidence = {"box": [0, 0, 10, 10], "label": label, "score": 1.0}
        if key == "phase" and explicit_controls:
            evidence["semantic"] = "self_action_buttons"
        return label, 1.0, evidence

    cards = SimpleNamespace(recognize=lambda *args, **kwargs: Recognition([], frame.captured_at, 0))
    recognizer = TableRecognizer(cards, SimpleNamespace(detect=detect))
    observation, _ = recognizer.observe(frame, profile)
    assert observation.phase == "playing"
    assert observation.passes == ([] if explicit_controls else ["self"])
    assert ("pass_self" in calls) is not explicit_controls
