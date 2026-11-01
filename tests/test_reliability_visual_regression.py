"""已人工核对的动画、昵称和末手问题回归；不冒充独立验收集。"""

import hashlib
import json

import pytest

from doudizhu_lab.domain.models import Frame
from doudizhu_lab.domain.tracker import GameState, GameTracker, canonical
from doudizhu_lab.runtime.analysis import MediaSampler
from doudizhu_lab.storage.config import Store, read_rgb
from doudizhu_lab.vision.cards import Recognizer
from doudizhu_lab.vision.table import MarkerLibrary, TableObservation, TableRecognizer
from tests.support import local_data_root


@pytest.mark.local_data
def test_reviewed_development_frames():
    root = local_data_root()
    labels = root / "reliability-regression/labels.json"
    if not labels.exists():
        pytest.skip("本机开发录像回归样本未附带在源码仓库")
    store = Store(root)
    recognizer = TableRecognizer(Recognizer(store.template_dir), MarkerLibrary(root / "markers"))
    for case in json.loads(labels.read_text(encoding="utf-8"))["cases"]:
        image = labels.parent / case["image"]
        assert hashlib.sha256(image.read_bytes()).hexdigest() == case["image_sha256"]
        obs, _ = recognizer.observe(Frame("offline", read_rgb(image), 0, case["image"], 0), store.profile)
        for key, expected in case["checks"].items():
            actual = getattr(obs, key, None)
            if key == "absent_pass":
                assert not set(expected) & set(obs.passes), case["image"]
            elif isinstance(expected, dict):
                for seat, value in expected.items():
                    assert (canonical(actual[seat]) if isinstance(value, list) else actual[seat]) == (
                        canonical(value) if isinstance(value, list) else value
                    ), (case["image"], key, seat, actual)
            elif key == "hand":
                assert canonical(actual) == canonical(expected), case["image"]
            else:
                assert actual == expected, (case["image"], key, actual)


@pytest.mark.local_data
@pytest.mark.parametrize("sequence_index", range(7))
def test_short_animation_is_confirmed_for_each_sampling_phase(sequence_index):
    root = local_data_root()
    manifest = root / "reliability-regression/short-prompts/sequences.json"
    if not manifest.exists():
        pytest.skip("本机短提示开发回归样本未附带在源码仓库")
    sequence = json.loads(manifest.read_text(encoding="utf-8"))[sequence_index]
    library = MarkerLibrary(root / "markers")
    samples = []
    for entry in sequence["frames"]:
        path = manifest.parent / entry["file"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]
        rgb = read_rgb(path)
        label, _, _ = library.detect(rgb, "pass_" + sequence["seat"])
        samples.append(
            (
                Frame(
                    "replay",
                    rgb,
                    0,
                    entry["frame_id"],
                    0,
                    "replay",
                    entry["frame_id"],
                    "sequence",
                    entry["pts_us"],
                ),
                label,
            )
        )
    for phase_us in range(0, 100_000, 10_000):
        sampler = MediaSampler()
        first = samples[0][0]
        from dataclasses import replace

        sampler.accept(replace(first, media_pts_us=first.media_pts_us + phase_us - 1_000_000))
        tracker = GameTracker(["self", "right", "left"])
        tracker.state = GameState(
            game_id="test",
            phase="playing",
            current_actor=sequence["seat"],
            last_play=["A"],
            last_player=next(s for s in ("self", "left", "right") if s != sequence["seat"]),
            seat_order=tracker.order,
        )
        tracker._phase, tracker._active_started = "playing", True
        events = []
        for frame, label in samples:
            if sampler.accept(frame):
                observation = TableObservation(
                    frame.frame_id,
                    frame.media_pts_us,
                    phase="playing",
                    passes=[sequence["seat"]] if label else [],
                    pass_labels={sequence["seat"]: label} if label else {},
                )
                events.extend(tracker.observe(observation).events)
        passes = [e for e in events if e.kind == "pass"]
        assert len(passes) == 1 and passes[0].confirmed, (sequence_index, phase_us)
        assert not tracker.state.pending
