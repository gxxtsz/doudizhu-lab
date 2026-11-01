"""短提示OCR的严格文字、颜色和来源检查；合成图只验证边界，不冒充真实识别。"""

from types import SimpleNamespace

import numpy as np
import pytest

from doudizhu_lab.domain.models import Frame
from doudizhu_lab.storage.config import Profile
from doudizhu_lab.vision.action_text import PassTextOCR
from doudizhu_lab.vision.table import TableObservation
from tests.support import local_data_root


def sample(text="不要", score=0.999):
    profile = Profile()
    rgb = np.full((1080, 2400, 3), (60, 110, 190), np.uint8)
    rect = profile.indicator_regions["pass_self"].pixels(2400, 1080)
    x, y = rect.x + 300, rect.y + 70
    for dx in (0, 65):
        rgb[y : y + 80, x + dx : x + dx + 14] = (240, 200, 20)
        rgb[y + 66 : y + 80, x + dx : x + dx + 50] = (240, 200, 20)
    frame = Frame("offline", rgb, 0, "test", 0, "file", "same-frame", "video", 123456)
    obs = TableObservation("same-frame", 123456, phase="playing", landlord="left", bottom=["3", "4", "5"])
    calls = []

    def engine(*a, **kw):
        calls.append(kw)
        return SimpleNamespace(txts=[text], scores=[score])

    return frame, obs, profile, PassTextOCR(engine=engine), calls


def test_complete_high_score_text_keeps_same_frame_provenance():
    frame, obs, profile, recognizer, calls = sample()
    recognizer.enrich(frame, obs, profile)
    assert obs.passes == ["self"] and obs.pass_labels == {"self": "不要"}
    evidence = obs.marker_evidence["pass_self"]
    assert evidence["frame_id"] == frame.frame_id and evidence["pts_us"] == frame.media_pts_us
    assert calls == [{"use_det": False, "use_cls": False}]


@pytest.mark.parametrize("text,score", [("不", 1), ("不抢", 1), ("不加倍", 1), ("不要", 0.979), ("要不", 1)])
def test_partial_bidding_or_low_score_is_not_pass(text, score):
    frame, obs, profile, recognizer, _ = sample(text, score)
    recognizer.enrich(frame, obs, profile)
    assert not obs.passes


@pytest.mark.parametrize("phase", ["bidding", "doubling", "settlement"])
def test_non_playing_stage_never_adds_action(phase):
    frame, obs, profile, recognizer, calls = sample()
    obs.phase = phase
    recognizer.enrich(frame, obs, profile)
    assert not obs.passes and not calls


def test_own_buttons_and_existing_template_evidence_are_not_overwritten():
    frame, obs, profile, recognizer, calls = sample()
    obs.marker_evidence["phase"] = {"accepted": True, "semantic": "self_action_buttons"}
    recognizer.enrich(frame, obs, profile)
    assert not obs.passes and not calls
    obs.marker_evidence.clear()
    obs.passes.append("self")
    recognizer.enrich(frame, obs, profile)
    assert obs.passes == ["self"] and not calls


def test_solid_gold_button_background_is_not_action_text():
    frame, obs, profile, recognizer, calls = sample()
    rect = profile.indicator_regions["pass_self"].pixels(2400, 1080)
    frame.rgb[rect.y + 70 : rect.y + 150, rect.x + 300 : rect.x + 450] = (240, 200, 20)
    recognizer.enrich(frame, obs, profile)
    assert not obs.passes and not calls


def test_different_source_frame_cannot_borrow_current_pixels():
    frame, obs, profile, recognizer, calls = sample()
    obs.frame_id = "old-frame"
    recognizer.enrich(frame, obs, profile)
    assert not obs.passes and not calls


@pytest.mark.local_data
def test_real_warped_own_pass_keeps_complete_text_without_lowering_template_threshold():
    from doudizhu_lab.storage.replay import VideoReplay, profile_from_dict
    from doudizhu_lab.storage.session import SessionStore
    from doudizhu_lab.vision.cards import Recognizer
    from doudizhu_lab.vision.ocr import make_ocr
    from doudizhu_lab.vision.table import MarkerLibrary, TableRecognizer

    source = local_data_root() / "sessions/20261101_150056_800b64"
    if not (source / "video/index.json").exists():
        pytest.skip("本机真实短提示开发录像不随源码提交")
    pytest.importorskip("rapidocr", reason="真实OCR回归需安装vision可选依赖")
    pytest.importorskip("onnxruntime", reason="真实OCR回归需安装vision可选依赖")
    session = SessionStore(source, readonly=True)
    try:
        profile = profile_from_dict(session.metadata()["profile"])
    finally:
        session.close()
    frame = VideoReplay(source).at(869072200000)
    observation, _ = TableRecognizer(
        Recognizer(source / "templates/templates", include_bundled=False),
        MarkerLibrary(source / "templates/markers"),
    ).observe(frame, profile)
    assert "self" not in observation.passes
    PassTextOCR(engine=make_ocr(source)).enrich(frame, observation, profile)
    assert "self" in observation.passes and observation.pass_labels["self"] == "不要"
    assert observation.marker_evidence["pass_self"]["score"] >= 0.98


def test_production_analysis_saves_ordinary_turn_ocr_without_winner_marker(tmp_path, monkeypatch):
    from doudizhu_lab.domain.models import Recognition
    from doudizhu_lab.runtime.recording import RecordingPipeline
    from doudizhu_lab.storage.config import Store

    frame, observation, _, _, _ = sample()
    recognition = Recognition([], frame.captured_at, 0)
    calls = []

    class TextService:
        def __init__(self, root):
            pass

        def enrich(self, frame, obs, cards, profile):
            calls.append(frame.frame_id)
            obs.passes.append("self")
            obs.pass_labels["self"] = "不要"
            return obs, cards

    monkeypatch.setattr(
        "doudizhu_lab.runtime.recording.setup_video", lambda *a, **kw: tmp_path / "unused.jar"
    )
    monkeypatch.setattr("doudizhu_lab.vision.last_card.LastCardOCR", TextService)
    pipeline = RecordingPipeline(Store(tmp_path), "test")
    pipeline.analysis_pool.shutdown()
    results = iter([(frame, (observation, recognition)), None])
    pipeline.analysis_pool = SimpleNamespace(take=lambda: next(results), closed=True, shutdown=lambda: None)
    pipeline.evidence = SimpleNamespace(submit=lambda *a: True)
    pipeline._last_card_enabled = True
    try:
        pipeline._analyze()
        rows = pipeline.session.rows("observations")
        assert calls == [frame.frame_id] and rows[0]["passes"] == ["self"]
        assert pipeline.bot_snapshot().observation["pass_labels"]["self"] == "不要"
        assert not pipeline.failed
    finally:
        pipeline.session.close()
