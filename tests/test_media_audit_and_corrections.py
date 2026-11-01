"""真实小型录像的截断审计，以及独立标注到追加纠错的往返检查。"""

import json
from fractions import Fraction

import av
import numpy as np

from doudizhu_lab.domain.tracker import GameTracker
from doudizhu_lab.storage.config import Profile
from doudizhu_lab.storage.corrections import (
    apply_label_corrections,
    observation_hash,
    verify_correction_replay,
)
from doudizhu_lab.storage.replay import VideoReplay, reanalyze
from doudizhu_lab.storage.session import SessionStore
from doudizhu_lab.storage.truth import TruthStore
from doudizhu_lab.vision.table import TableObservation
from tests.test_evidence_truth import fake_session, labelled_game


def tiny_video(folder):
    video = folder / "video"
    video.mkdir(exist_ok=True)
    with av.open(str(video / "sample.mkv"), "w", format="matroska") as container:
        stream = container.add_stream("libx264", rate=10)
        stream.width, stream.height, stream.pix_fmt = 64, 48, "yuv420p"
        stream.options = {"preset": "ultrafast", "crf": "30"}
        for i in range(20):
            frame = av.VideoFrame.from_ndarray(np.full((48, 64, 3), i * 10, np.uint8), format="rgb24")
            frame.pts, frame.time_base = i, Fraction(1, 10)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    (video / "index.json").write_text(
        json.dumps(
            [
                {
                    "file": "sample.mkv",
                    "global_start_us": 1_000_000,
                    "global_end_us": 2_900_000,
                    "frames": 20,
                    "width": 64,
                    "height": 48,
                    "closed": True,
                }
            ]
        )
    )
    return video / "sample.mkv"


def test_media_audit_reads_every_frame_and_rejects_truncation(tmp_path):
    path = tiny_video(tmp_path)
    report = VideoReplay(tmp_path).audit()
    assert report["passed"] and report["segments"][0]["decoded_frames"] == 20
    path.write_bytes(path.read_bytes()[: len(path.read_bytes()) // 2])
    failed = VideoReplay(tmp_path).audit()
    assert not failed["passed"]


def test_bounded_frames_seek_without_repeating_prior_frames(tmp_path):
    tiny_video(tmp_path)
    video = VideoReplay(tmp_path)
    frames = list(video.frames(fps=0, start_us=1_450_000, end_us=1_700_000))
    assert [f.media_pts_us for f in frames] == [1_500_000, 1_600_000, 1_700_000]
    assert all(f.width == 64 and f.height == 48 for f in frames)


def test_parallel_reanalysis_is_ordered_and_keeps_original_observations(tmp_path):
    session = SessionStore.create(tmp_path, Profile(), [], "offline")
    tiny_video(session.folder)
    before = observation_hash(session)
    sequential = reanalyze(session.folder, fps=10, workers=1)
    parallel = reanalyze(session.folder, fps=10, workers=2)
    a, b = SessionStore(sequential), SessionStore(parallel)
    try:
        assert a.rows("events") == b.rows("events")
        assert [o["pts_us"] for o in a.rows("observations")] == [o["pts_us"] for o in b.rows("observations")]
        assert before == observation_hash(session)
        assert a.metadata()["analysis_mode"] == b.metadata()["analysis_mode"] == "offline"
    finally:
        a.close()
        b.close()
        session.close()


def test_published_truth_appends_correction_and_real_undo_redo(tmp_path):
    session = fake_session(tmp_path)
    # 原始流由相同状态机产生，确保复现检查比较的是实际自动处理路径。
    tracker = GameTracker()
    for pts in (0, 120_000):
        observation = TableObservation(str(pts), pts, phase="dealing")
        session.add_observation(observation)
        for event in tracker.observe(observation).events:
            session.add_event(event)
    folder = session.folder
    original_hash = observation_hash(session)
    session.close()
    labels = TruthStore(folder)
    labels.select_range(0, 2_000_000)
    labels.document.update(labelled_game())
    labels.save(publish=True)
    batch = apply_label_corrections(folder, labels.path)
    check = verify_correction_replay(folder, batch["batch_id"])
    assert check["passed"] and check["undo_changed_events"]
    reopened = SessionStore(folder)
    assert observation_hash(reopened) == original_hash
    assert len(reopened.rows("observations")) == 2
    assert reopened.game_states()["manual"]["complete"]
    assert len(reopened.rows("revisions")) == len(batch["revisions"]) * 3
    reopened.close()
