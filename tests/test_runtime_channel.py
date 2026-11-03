"""跨进程状态访问冲突、源时间有效期及有序控制回归；只使用临时文件和合成媒体。"""

import errno
import json
import threading
from pathlib import Path
from time import monotonic, sleep
from types import SimpleNamespace

import pytest

from doudizhu_lab.runtime.capture import CaptureProcess, CaptureWorker
from doudizhu_lab.runtime.channel import RuntimeChannel
from doudizhu_lab.storage.retention import atomic_json
from tests.test_recording_retention import RepeatingStream, media_packets


@pytest.fixture
def clock(monkeypatch):
    """推进独立单调时钟验证二百毫秒预算及三秒边界，不让测试真实等待超时。"""
    now = [1.0]
    monkeypatch.setattr("doudizhu_lab.storage.json_io.monotonic", lambda: now[0])
    monkeypatch.setattr(
        "doudizhu_lab.storage.json_io.sleep", lambda delay: now.__setitem__(0, now[0] + delay)
    )
    monkeypatch.setattr("doudizhu_lab.runtime.capture.monotonic", lambda: now[0])
    return now


def proxy(folder):
    return CaptureProcess(folder, "unused-adb", "isolated", folder / "unused.jar")


def worker(folder):
    return CaptureWorker({"folder": str(folder), "automatic": True}, stream=SimpleNamespace())


def test_status_read_conflict_keeps_original_source_time(tmp_path, monkeypatch, clock):
    channel = proxy(tmp_path)
    path = tmp_path / "采集状态.json"
    atomic_json(path, {"updated_at": 1.0, "started": True})
    assert channel.status()["updated_at"] == 1
    original = Path.read_text

    def denied(p, *args, **kwargs):
        if p == path:
            raise PermissionError(errno.EACCES, "模拟仅有errno13的读取占用")
        return original(p, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", denied)
    value = channel.status()
    assert value["updated_at"] == 1
    assert 1.19 <= clock[0] <= 1.201
    assert channel.channel.read_failures == 1 and channel.channel.retries > 0
    value["updated_at"] = 99
    monkeypatch.setattr(Path, "read_text", original)
    assert channel.status()["updated_at"] == 1
    atomic_json(path, {"updated_at": 1.5, "started": True})
    assert channel.status()["updated_at"] == 1.5


def test_atomic_publish_recovers_from_errno13_and_persistent_failure_raises(tmp_path, monkeypatch, clock):
    path = tmp_path / "状态.json"
    original = Path.replace
    attempts = [0]

    def conflict(p, target):
        attempts[0] += 1
        if attempts[0] <= 2:
            raise PermissionError(errno.EACCES, "模拟替换占用")
        return original(p, target)

    monkeypatch.setattr(Path, "replace", conflict)
    channel = RuntimeChannel()
    assert channel.write(path, {"value": 1})
    assert channel.retries == 2

    def always_denied(*args):
        raise PermissionError(errno.EACCES, "持续占用")

    monkeypatch.setattr(Path, "replace", always_denied)
    before = clock[0]
    assert not channel.write(path, {"value": 2})
    assert clock[0] - before <= 0.201
    assert json.loads(path.read_text(encoding="utf-8")) == {"value": 1}
    with pytest.raises(PermissionError):
        atomic_json(tmp_path / "留存决定.json", {"delete": True})
    assert not list(tmp_path.glob("*.tmp"))


def test_disk_full_is_not_treated_as_temporary_access_error(tmp_path, monkeypatch):
    def full(*args, **kwargs):
        raise OSError(errno.ENOSPC, "模拟磁盘不足")

    monkeypatch.setattr(Path, "write_text", full)
    with pytest.raises(OSError) as result:
        RuntimeChannel().write(tmp_path / "状态.json", {})
    assert result.value.errno == errno.ENOSPC


def test_outgoing_commands_keep_blocked_head_and_original_sequence(tmp_path, monkeypatch, clock):
    (tmp_path / "采集命令").mkdir()
    channel = proxy(tmp_path)
    channel.command("game_start", game_id="one")
    channel.command("game_complete", game_id="one", pts_us=100)
    original = channel.channel.write
    monkeypatch.setattr(channel.channel, "write", lambda *args: False)
    assert not channel.flush_commands()
    assert not list((tmp_path / "采集命令").glob("*.json"))
    monkeypatch.setattr(channel.channel, "write", original)
    assert channel.flush_commands()
    files = sorted((tmp_path / "采集命令").glob("*.json"))
    assert [p.name for p in files] == ["00000001.json", "00000002.json"]
    assert [json.loads(p.read_text(encoding="utf-8"))["kind"] for p in files] == [
        "game_start",
        "game_complete",
    ]


def test_outgoing_command_blocking_is_bounded(tmp_path, monkeypatch, clock):
    channel = proxy(tmp_path)
    channel.command("stop", reason="fault")
    monkeypatch.setattr(channel.channel, "write", lambda *args: False)
    assert not channel.flush_commands()
    clock[0] += 3.01
    with pytest.raises(TimeoutError, match="控制命令"):
        channel.flush_commands()
    assert channel._pending_command["kind"] == "stop"


def test_health_publish_conflict_does_not_drop_commands_or_refresh_success(tmp_path, monkeypatch, clock):
    (tmp_path / "采集命令").mkdir()
    channel = proxy(tmp_path)
    channel.heartbeat({"reported_at": clock[0]})
    written = channel._health_written_at
    original = channel.channel.write
    monkeypatch.setattr(
        channel.channel,
        "write",
        lambda path, value: False if path.name == "主程序健康.json" else original(path, value),
    )
    channel.command("fault", code="test", message="模拟原始故障")
    clock[0] += 0.5
    channel.heartbeat({"reported_at": clock[0]})
    assert channel._health_written_at == written
    assert (tmp_path / "采集命令/00000001.json").is_file()
    clock[0] += 3
    with pytest.raises(TimeoutError, match="健康报告"):
        channel.heartbeat({"reported_at": clock[0]})


def test_health_checks_source_progress_instead_of_repeated_successful_reads(tmp_path, clock):
    capture = worker(tmp_path)
    capture.phase = "running"
    atomic_json(tmp_path / "主程序健康.json", {"reported_at": clock[0]})
    capture._monitor_health(clock[0])
    clock[0] += 2.9
    capture._monitor_health(clock[0])
    assert not capture.ledger.faults
    clock[0] += 0.2
    capture._monitor_health(clock[0])
    assert capture.phase == "tail_recording"
    assert capture.ledger.faults[0]["code"] == "health_timeout"
    capture.writer.close()


def test_health_first_report_uses_twenty_second_warmup(tmp_path, clock):
    capture = worker(tmp_path)
    capture.phase = "running"
    clock[0] += 19.9
    capture._monitor_health(clock[0])
    assert not capture.ledger.faults
    clock[0] += 0.2
    capture._monitor_health(clock[0])
    assert capture.ledger.faults[0]["code"] == "health_startup"
    capture.writer.close()


def test_tail_recording_preserves_first_fault_and_deadline_without_heartbeat_noise(tmp_path, clock):
    capture = worker(tmp_path)
    capture.phase = "running"
    capture.fault("analysis", "首个故障", occurred_at=1.0)
    deadline = capture.fault_deadline
    atomic_json(tmp_path / "主程序健康.json", {"reported_at": 0, "gui_at": 0})
    clock[0] += 5
    capture._monitor_health(clock[0])
    capture.fault("analysis", "重复故障")
    capture.fault("capture", "后续真实写盘故障")
    assert capture.fault_deadline == deadline
    assert [f["code"] for f in capture.ledger.faults] == ["analysis", "capture"]
    assert capture.ledger.value()["primary_fault"]["message"] == "首个故障"
    capture.writer.close()


def test_incoming_command_read_blocking_does_not_overtake_head(tmp_path, monkeypatch, clock):
    capture = worker(tmp_path)
    commands = tmp_path / "采集命令"
    commands.mkdir()
    first, second = commands / "00000001.json", commands / "00000002.json"
    atomic_json(first, {"kind": "game_start", "game_id": "one"})
    atomic_json(second, {"kind": "user_abort"})
    original = capture.channel.read
    monkeypatch.setattr(capture.channel, "read", lambda p: None if p == first else original(p))
    capture._commands()
    assert not capture.ledger.windows and capture.ledger.active["game_id"] == ""
    clock[0] += 3.1
    capture._commands()
    assert capture.ledger.faults[0]["code"] == "control_channel"
    assert first.exists() and second.exists()
    capture.writer.close()


def test_applied_command_is_not_repeated_when_ack_file_is_occupied(tmp_path, monkeypatch):
    capture = worker(tmp_path)
    commands = tmp_path / "采集命令"
    commands.mkdir()
    path = commands / "00000001.json"
    atomic_json(path, {"kind": "user_abort"})
    original = capture.channel.remove
    monkeypatch.setattr(capture.channel, "remove", lambda p: False)
    capture._commands()
    capture._commands()
    assert len(capture.ledger.windows) == 1 and path.exists()
    monkeypatch.setattr(capture.channel, "remove", original)
    capture._commands()
    assert len(capture.ledger.windows) == 1 and not path.exists()
    capture.writer.close()


def test_status_file_occupied_does_not_hold_recording_lock_or_stop_video(tmp_path, monkeypatch):
    folder = tmp_path / "session"
    folder.mkdir()
    capture = CaptureWorker(
        {"folder": str(folder), "automatic": True},
        stream=RepeatingStream(media_packets(tmp_path / "source")),
        output=lambda data: None,
        tail_seconds=0.05,
    )
    capture.allowed = True
    entered, release = threading.Event(), threading.Event()
    original = capture.channel.write

    def occupied(path, value):
        if path.name == "采集状态.json" and not entered.is_set():
            entered.set()
            assert release.wait(3)
        return original(path, value)

    monkeypatch.setattr(capture.channel, "write", occupied)
    thread = threading.Thread(target=capture.run, kwargs={"guard": False})
    thread.start()
    try:
        assert entered.wait(2)
        with capture.lock:
            before = capture.packets_count
        deadline = monotonic() + 1
        while capture.packets_count <= before + 10 and monotonic() < deadline:
            sleep(0.01)
        assert capture.packets_count > before + 10
        capture.fault("test", "结束隔离测试")
    finally:
        release.set()
        thread.join(5)
        if thread.is_alive():
            capture.stopped.set()
            capture.stream.close()
            thread.join(3)
    assert not thread.is_alive() and capture.phase == "closed"


def test_first_failure_time_is_not_replaced_by_later_delivery_order(tmp_path):
    capture = worker(tmp_path)
    capture.fault("later", "先收到的后续故障", occurred_at=10)
    deadline = capture.fault_deadline
    capture.fault("earlier", "后收到的原始故障", occurred_at=5)
    assert capture.ledger.value()["primary_fault"]["code"] == "earlier"
    assert capture.fault_deadline == deadline
    capture.writer.close()


def test_concurrent_diagnostic_writers_cannot_overwrite_first_fault(tmp_path, monkeypatch):
    from doudizhu_lab.evaluation.paced_replay import EncodedReplayStream
    from doudizhu_lab.runtime.recording import RecordingPipeline
    from doudizhu_lab.storage.config import Store
    from tests.test_recording_retention import tiny_recording

    source = EncodedReplayStream(tiny_recording(tmp_path / "source"))
    pipeline = RecordingPipeline(Store(tmp_path / "data"), "replay", analyze=False, stream=source)
    pipeline._dumped = True
    entered, release = threading.Event(), threading.Event()

    def publish(path, value):
        if path.name == "故障诊断.json" and not entered.is_set():
            entered.set()
            assert release.wait(3)
        atomic_json(path, value)

    monkeypatch.setattr("doudizhu_lab.runtime.recording.atomic_json", publish)
    later = threading.Thread(
        target=pipeline.report_failure, args=("later", "后续错误"), kwargs={"occurred_at": 10}
    )
    earlier = threading.Thread(
        target=pipeline.report_failure, args=("earlier", "原始错误"), kwargs={"occurred_at": 5}
    )
    try:
        later.start()
        assert entered.wait(2)
        earlier.start()
        deadline = monotonic() + 2
        while "earlier" not in pipeline._faults and monotonic() < deadline:
            sleep(0.01)
        assert "earlier" in pipeline._faults
    finally:
        release.set()
        later.join(3)
        if earlier.ident:
            earlier.join(3)
        pipeline.writer.close()
        source.close()
        pipeline.session.close()
    assert not later.is_alive() and not earlier.is_alive()
    value = json.loads((pipeline.session.folder / "故障诊断.json").read_text(encoding="utf-8"))
    assert set(value["faults"]) == {"earlier", "later"}
    assert value["primary_fault"]["code"] == "earlier"
