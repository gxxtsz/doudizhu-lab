"""异常录像留存、真实编码、阻塞隔离及限定范围清理的回归验证，不连接手机。"""

import json
import sqlite3
import subprocess
import sys
import threading
from contextlib import closing
from dataclasses import replace
from pathlib import Path
from time import monotonic, sleep

import av
import pytest

from doudizhu_lab.devices.video import SegmentWriter, read_packet
from doudizhu_lab.evaluation.paced_replay import EncodedReplayStream
from doudizhu_lab.runtime.analysis import OrderedAnalyzer
from doudizhu_lab.runtime.capture import CaptureWorker, health_timeout, packet_bytes
from doudizhu_lab.runtime.recording import RecordingPipeline
from doudizhu_lab.storage.cleanup import apply_legacy_cleanup, legacy_inventory
from doudizhu_lab.storage.config import Store
from doudizhu_lab.storage.retention import (
    RecordingResult,
    RetentionLedger,
    StopReason,
    atomic_json,
    checked_tree,
    finalize_retention,
    playable_video,
)
from doudizhu_lab.storage.runs import RunStore, evaluate_run
from tests.test_paced_replay import tiny_recording


def media_packets(folder):
    """从真实小型MKV提取协议包，避免用不可解码的占位数据证明录像成功。"""
    stream = EncodedReplayStream(tiny_recording(folder))
    stream.start()
    return list(stream.packets())


@pytest.mark.parametrize("reason", [StopReason.COMPLETED, StopReason.USER])
@pytest.mark.parametrize("automatic", [True, False])
def test_finalize_policy_and_repeat(tmp_path, reason, automatic):
    folder = tmp_path / "sessions" / "one"
    folder.mkdir(parents=True)
    ledger = RetentionLedger(folder, automatic=automatic)
    ledger.finish(reason)
    (folder / "会话快照").mkdir()
    (folder / "会话快照" / "old.bin").write_bytes(b"snapshot")
    result = finalize_retention(folder, tmp_path)
    assert isinstance(result, RecordingResult) and result.reason == reason
    assert result.summary_path.is_file()
    assert folder.exists() is not automatic
    assert result == finalize_retention(folder, tmp_path)


def test_fault_wins_over_manual_stop_and_keeps_valid_video(tmp_path):
    folder = tmp_path / "sessions" / "failure"
    folder.mkdir(parents=True)
    packets = media_packets(tmp_path / "input")
    writer = SegmentWriter(folder / "video")
    for packet in packets:
        writer.write(packet)
    writer.close()
    ledger = RetentionLedger(folder, automatic=True)
    ledger.begin_game("bad-game")
    assert ledger.fault("recognition_timeout", "识别阻塞")
    assert not ledger.fault("recognition_timeout", "重复通知")
    ledger.finish(StopReason.USER)
    result = finalize_retention(folder, tmp_path)
    assert result.reason == StopReason.FAULT and result.replay_folder == folder
    assert playable_video(folder)
    with av.open(str(folder / "video" / writer.segments[0].file)) as video:
        assert len(list(video.decode(video=0))) == 5


def test_pruning_preserves_boundary_and_live_segment_and_unique_names(tmp_path):
    packets = media_packets(tmp_path / "input")
    folder = tmp_path / "sessions" / "mixed"
    folder.mkdir(parents=True)
    ledger = RetentionLedger(folder, automatic=True)
    writer = SegmentWriter(folder / "video", segment_seconds=1)
    writer.write(packets[0])
    writer.write(packets[1])
    keyframe = packets[2]
    for second in range(3):
        writer.write(replace(keyframe, pts_us=second * 1_000_000))
    ledger.complete_game("good", 1_000_000)
    ledger.prune(writer)
    assert [s.file for s in writer.segments] == ["segment-0001.mkv", "segment-0002.mkv"]
    assert not (writer.folder / "segment-0000.mkv").exists()
    ledger.fault("tracking", "下一局未推进")
    ledger.complete_game("bad", 3_000_000)
    writer.write(replace(keyframe, pts_us=3_000_000))
    ledger.prune(writer)
    writer.close()
    assert [s.file for s in writer.segments] == ["segment-0001.mkv", "segment-0002.mkv", "segment-0003.mkv"]
    assert len(json.loads((writer.folder / "index.json").read_text(encoding="utf-8"))) == 3
    for segment in writer.segments:
        with av.open(str(writer.folder / segment.file)) as video:
            assert len(list(video.decode(video=0))) == 1


def test_user_pause_discards_only_unpinned_window(tmp_path):
    ledger = RetentionLedger(tmp_path, automatic=True)
    ledger.begin_game("one")
    ledger.last_pts = 100
    ledger.abort_window()
    assert ledger.windows[0]["outcome"] == "user_aborted"
    assert ledger.active["started_midgame"]
    ledger.fault("fault", "恢复后异常")
    before = ledger.value()
    ledger.abort_window()
    assert ledger.value() == before


def test_missing_final_decision_preserves_evidence(tmp_path):
    folder = tmp_path / "sessions" / "unfinished"
    folder.mkdir(parents=True)
    RetentionLedger(folder, automatic=True)
    result = finalize_retention(folder, tmp_path)
    assert result.reason == StopReason.FAULT and folder.exists()


def test_cleanup_failure_is_not_reported_as_cleaned(tmp_path, monkeypatch):
    folder = tmp_path / "sessions" / "occupied"
    folder.mkdir(parents=True)
    ledger = RetentionLedger(folder, automatic=True)
    ledger.finish(StopReason.USER)
    monkeypatch.setattr(
        "doudizhu_lab.storage.retention.remove_tree",
        lambda *args: (_ for _ in ()).throw(PermissionError("文件占用")),
    )
    with pytest.raises(PermissionError):
        finalize_retention(folder, tmp_path)
    summary = tmp_path / "recording-summaries" / "occupied.json"
    assert json.loads(summary.read_text(encoding="utf-8"))["media_status"] == "cleanup_failed"
    assert folder.exists()


def test_health_timeouts_distinguish_work_and_waiting():
    assert health_timeout({"gui_at": 99, "stages": {}}, 100) is None
    assert health_timeout({"gui_at": 90}, 100)
    assert health_timeout({"stages": {"识别": {"pending_at": None, "output_at": 1}}}, 100) is None
    assert health_timeout({"stages": {"识别": {"pending_at": 96, "output_at": 95}}}, 100)
    assert health_timeout({"stages": {"识别": {"pending_at": 81, "output_at": None}}}, 100) is None
    assert health_timeout({"stages": {"识别": {"pending_at": 79, "output_at": None}}}, 100)


def test_blocked_recognizer_has_bounded_wait_and_preserves_order(frame):
    release = threading.Event()
    pool = OrderedAnalyzer(lambda: lambda value: (release.wait(3), value)[1], capacity=2)
    try:
        assert pool.submit(frame)
        before = monotonic()
        assert pool.take(timeout=0.02) is None
        assert monotonic() - before < 0.5 and pool.size == 1 and pool.unfinished == 1
        release.set()
        assert pool.take(timeout=1)[0] == frame
    finally:
        release.set()
        pool.shutdown()


def test_recognizer_own_timeout_is_not_swallowed(frame):
    def fail(_):
        raise TimeoutError("识别函数自身超时")

    pool = OrderedAnalyzer(lambda: fail)
    try:
        pool.submit(frame)
        with pytest.raises(TimeoutError, match="自身"):
            pool.take(timeout=1)
    finally:
        pool.shutdown()


def make_database(path, values):
    """建立最小历史数据库，以验证清理范围和运行状态，不实例化模型或手机服务。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(path)) as db:
        db.execute("CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT)")
        db.executemany("INSERT INTO meta VALUES (?,?)", [(k, json.dumps(v)) for k, v in values.items()])
        db.commit()


def test_legacy_cleanup_preserves_globals_unrelated_runs_and_new_sessions(tmp_path):
    old = tmp_path / "sessions" / "old"
    make_database(old / "session.sqlite3", {"status": "stopped"})
    (old / "video.mkv").write_bytes(b"old-video")
    run = tmp_path / "runs" / "related"
    make_database(run / "run.sqlite3", {"status": "stopped", "session": str(old)})
    unrelated = tmp_path / "runs" / "simulation"
    make_database(unrelated / "run.sqlite3", {"status": "completed", "session": str(tmp_path / "elsewhere")})
    for name in ("models", "templates", "controls", "markers"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "保留.bin").write_bytes(b"keep")
    (tmp_path / "settings.json").write_text("{}", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    atomic_json(manifest, legacy_inventory(tmp_path))
    new = tmp_path / "sessions" / "new"
    make_database(new / "session.sqlite3", {"status": "recording"})
    result = apply_legacy_cleanup(tmp_path, manifest)
    assert not old.exists() and not run.exists()
    assert new.exists() and unrelated.exists()
    assert result["protected_unchanged"] and not result["remaining"]
    assert result["freed_bytes"] > 0
    assert apply_legacy_cleanup(tmp_path, manifest)["freed_bytes"] == result["freed_bytes"]


def test_active_or_outside_tree_is_rejected_before_any_deletion(tmp_path):
    old = tmp_path / "sessions" / "old"
    active = tmp_path / "sessions" / "active"
    make_database(old / "session.sqlite3", {"status": "stopped"})
    make_database(active / "session.sqlite3", {"status": "recording"})
    manifest = tmp_path / "manifest.json"
    atomic_json(manifest, legacy_inventory(tmp_path))
    with pytest.raises(RuntimeError, match="仍在使用"):
        apply_legacy_cleanup(tmp_path, manifest)
    assert old.exists() and active.exists()
    with pytest.raises(ValueError, match="越界"):
        checked_tree(tmp_path / "models", tmp_path / "sessions")


class RepeatingStream:
    """持续输出真实关键帧的替身，可阻塞分析发送而不影响视频生产。"""

    def __init__(self, packets):
        self.initial = packets[:2]
        self.keyframe = packets[2]
        self.closed = threading.Event()

    def start(self):
        pass

    def packets(self):
        yield from self.initial
        for sequence in range(10000):
            if self.closed.wait(0.002):
                return
            yield replace(self.keyframe, pts_us=sequence * 66_667)

    def close(self):
        self.closed.set()


def synthetic_worker(folder, source):
    """独立子进程测试入口；只读取测试MKV，禁止准备任何真机服务。"""
    source_stream = EncodedReplayStream(Path(source))
    source_stream.start()
    worker = CaptureWorker(
        {"folder": str(folder), "automatic": True},
        stream=RepeatingStream(list(source_stream.packets())),
        tail_seconds=0.25,
    )
    worker.allowed = True
    worker.run(guard=False)


def test_independent_process_survives_stale_parent_and_leaves_playable_video(tmp_path):
    source = tiny_recording(tmp_path / "input")
    folder = tmp_path / "sessions" / "child"
    folder.mkdir(parents=True)
    atomic_json(folder / "主程序健康.json", {"reported_at": monotonic() - 5})
    code = (
        "from pathlib import Path; from tests.test_recording_retention import synthetic_worker; "
        "import sys; synthetic_worker(Path(sys.argv[1]), Path(sys.argv[2]))"
    )
    process = subprocess.Popen(
        [sys.executable, "-c", code, str(folder), str(source)], stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    try:
        # 故意不消费stdout，让子进程自身完成留存与退出，不依靠主程序轮询。
        process.wait(timeout=20)
        assert process.returncode == 0, process.stderr.read().decode(errors="replace")
        ledger = json.loads((folder / "留存记录.json").read_text(encoding="utf-8"))
        assert ledger["reason"] == "fault" and ledger["faults"][0]["code"] == "health_timeout"
        assert playable_video(folder)
        for path in (folder / "video").glob("*.mkv"):
            with av.open(str(path)) as video:
                assert list(video.decode(video=0))
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
        process.stdout.close()
        process.stderr.close()


def test_blocked_analysis_pipe_does_not_block_recording(tmp_path):
    packets = media_packets(tmp_path / "input")
    folder = tmp_path / "sessions" / "backpressure"
    folder.mkdir(parents=True)
    release = threading.Event()
    stream = RepeatingStream(packets)
    worker = CaptureWorker(
        {"folder": str(folder), "automatic": True},
        stream=stream,
        output=lambda data: release.wait(5),
        tail_seconds=0.1,
    )
    worker.allowed = True
    thread = threading.Thread(target=worker.run, kwargs={"guard": False})
    thread.start()
    try:
        deadline = monotonic() + 4
        while worker.dropped == 0 and monotonic() < deadline:
            sleep(0.01)
        assert worker.dropped > 0 and worker.packets_count > 32
        worker.fault("test", "模拟识别消费停止")
        thread.join(timeout=5)
        assert not thread.is_alive() and playable_video(folder)
    finally:
        release.set()
        worker.stopped.set()
        stream.close()
        thread.join(timeout=5)


def test_disk_failure_preserves_closed_video_and_fault_decision(tmp_path, monkeypatch):
    """写盘中途失败时保留已取得的可解码片段，随后手动停止也不能改成正常清理。"""
    folder = tmp_path / "sessions" / "disk-failure"
    folder.mkdir(parents=True)
    worker = CaptureWorker(
        {"folder": str(folder), "automatic": True},
        stream=RepeatingStream(media_packets(tmp_path / "input")),
        output=lambda data: None,
    )
    worker.allowed = True
    original = worker.writer.write

    def write_until_disk_failure(packet):
        if worker.packets_count > 10:
            raise OSError("模拟磁盘空间不足")
        original(packet)

    monkeypatch.setattr(worker.writer, "write", write_until_disk_failure)
    worker.run(guard=False)
    result = finalize_retention(folder, tmp_path)
    assert result.reason == StopReason.FAULT and result.replay_folder == folder
    assert "磁盘空间不足" in worker.ledger.faults[0]["message"]
    for path in (folder / "video").glob("*.mkv"):
        with av.open(str(path)) as video:
            assert list(video.decode(video=0))


def test_forwarded_protocol_is_compatible(tmp_path):
    import io

    for packet in media_packets(tmp_path / "input"):
        assert read_packet(io.BytesIO(packet_bytes(packet))) == packet


@pytest.mark.parametrize("shutdown_failure", [False, True])
def test_production_pipeline_independent_capture_and_manual_cleanup(tmp_path, monkeypatch, shutdown_failure):
    source = tiny_recording(tmp_path / "source")
    real_popen = subprocess.Popen

    def launch_fake_media(arguments, **kwargs):
        config = Path(arguments[-1])
        code = (
            "import json,sys; from pathlib import Path; "
            "from tests.test_recording_retention import synthetic_worker; "
            "synthetic_worker(Path(json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))['folder']), "
            "Path(sys.argv[2]))"
        )
        return real_popen([sys.executable, "-c", code, str(config), str(source)], **kwargs)

    monkeypatch.setattr("doudizhu_lab.runtime.capture.subprocess.Popen", launch_fake_media)
    monkeypatch.setattr(
        "doudizhu_lab.runtime.recording.setup_video", lambda *a, **kw: tmp_path / "unused.jar"
    )
    store = Store(tmp_path / "data")
    pipeline = RecordingPipeline(store, "isolated-test", analyze=False, automatic=True)
    if shutdown_failure:
        original_stop = pipeline.evidence.stop

        def fail_after_stop(**kwargs):
            original_stop(**kwargs)
            raise OSError("模拟截图收尾失败")

        monkeypatch.setattr(pipeline.evidence, "stop", fail_after_stop)
    try:
        pipeline.start()
        deadline = monotonic() + 5
        while pipeline.frame is None and monotonic() < deadline:
            pipeline.poll()
            sleep(0.01)
        assert pipeline.frame is not None and pipeline.frame.transport == "video"
        assert pipeline.stream.process.pid != __import__("os").getpid()
        pipeline.stop(StopReason.USER)
        assert pipeline.failed is shutdown_failure
    finally:
        pipeline.stop()
        pipeline.session.close()
    result = pipeline.finalize_recording()
    if shutdown_failure:
        assert result.reason == StopReason.FAULT and result.replay_folder == pipeline.session.folder
        assert playable_video(pipeline.session.folder)
    else:
        assert result.reason == StopReason.USER and result.replay_folder is None
        assert not pipeline.session.folder.exists()
    assert result.summary_path.is_file()
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert summary["runtime"]["application_version"]
    assert summary["runtime"]["metrics"]["received_packets"] > 0


def test_recording_process_crash_is_reported(tmp_path):
    from doudizhu_lab.runtime.capture import CaptureProcess

    proxy = CaptureProcess(tmp_path, "unused-adb", "test", tmp_path / "unused.jar")
    proxy.process = subprocess.Popen(
        [sys.executable, "-c", "raise SystemExit(7)"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
    )
    proxy.process.wait(timeout=10)
    with pytest.raises(RuntimeError, match="异常退出"):
        proxy.close()


def test_cleanup_rejects_link_without_touching_destination(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "keep.txt").write_text("保留", encoding="utf-8")
    root = tmp_path / "sessions"
    root.mkdir()
    link = root / "link"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("系统未开放创建符号链接权限")
    try:
        with pytest.raises(ValueError, match="链接"):
            checked_tree(link, root)
        assert (outside / "keep.txt").read_text(encoding="utf-8") == "保留"
    finally:
        link.unlink()


def test_policy_cleanup_is_not_capture_loss_or_formal_acceptance(tmp_path):
    from types import SimpleNamespace

    store = Store(tmp_path)
    folder = tmp_path / "sessions" / "cleaned"
    folder.mkdir(parents=True)
    ledger = RetentionLedger(folder, automatic=True)
    ledger.complete_game("one", 10)
    ledger.finish(StopReason.COMPLETED)
    run = RunStore.create(
        tmp_path, SimpleNamespace(profile=store.profile, session=SimpleNamespace(folder=folder))
    )
    run.set("status", "completed")
    run.add("game_complete", game_id="one", start_us=0, end_us=10, verified=True)
    run.close()
    finalize_retention(folder, tmp_path)
    report = evaluate_run(run.folder)
    assert report["tracking_completed_games"] == 1
    assert report["media_status"] == "policy_cleaned"
    assert not report["video_audit_available"] and not report["live_passed"]
    assert not report["capture_issues"]


def test_policy_cleanup_does_not_hide_other_capture_gaps(tmp_path):
    """同会话中只豁免明确登记的清理范围，其他丢失画面仍报告真实覆盖缺口。"""
    from types import SimpleNamespace

    store = Store(tmp_path)
    folder = tmp_path / "sessions" / "partly-cleaned"
    (folder / "video").mkdir(parents=True)
    ledger = RetentionLedger(folder, automatic=True)
    ledger.removed.append({"start_us": 0, "end_us": 4_000_000, "file": "old.mkv"})
    ledger.fault("test", "保留其余故障区间")
    atomic_json(
        folder / "video/index.json",
        [{"global_start_us": 10_000_000, "global_end_us": 12_000_000, "closed": True}],
    )
    run = RunStore.create(
        tmp_path, SimpleNamespace(profile=store.profile, session=SimpleNamespace(folder=folder))
    )
    run.add("game_complete", game_id="one", start_us=0, end_us=12_000_000, verified=True)
    run.close()
    report = evaluate_run(run.folder)
    assert [(gap["start_us"], gap["end_us"]) for gap in report["capture_issues"]] == [(4_000_000, 10_000_000)]
