"""固定协议的边界、截断、磁盘限制及真实录像回放检查。"""

import io
import json
import struct
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from doudizhu_lab.devices.adb import AdbClient, AdbError
from doudizhu_lab.devices.video import (
    CONFIG_FLAG,
    KEY_FLAG,
    SESSION_FLAG,
    SegmentWriter,
    read_packet,
    setup_video,
)
from doudizhu_lab.storage.replay import VideoReplay
from tests.support import local_data_root


class Fragmented(io.BytesIO):
    def read(self, n=-1):
        return super().read(min(n, 2))


def test_protocol_flags_and_fragmented_socket():
    data = struct.pack(">III", SESSION_FLAG >> 32, 2400, 1080)
    data += struct.pack(">QI", CONFIG_FLAG, 4) + b"sps!"
    data += struct.pack(">QI", KEY_FLAG | 1234567, 5) + b"frame"
    source = Fragmented(data)
    session, config, frame = (read_packet(source) for _ in range(3))
    assert (session.width, session.height, session.session) == (2400, 1080, True)
    assert config.config and config.data == b"sps!"
    assert frame.keyframe and frame.pts_us == 1234567 and frame.data == b"frame"


@pytest.mark.parametrize("data", [b"", b"123", struct.pack(">QI", 42, 5) + b"abc"])
def test_truncation_never_becomes_valid_packet(data):
    with pytest.raises(EOFError):
        read_packet(io.BytesIO(data))


@pytest.mark.parametrize(
    "data",
    [
        struct.pack(">QI", 0, 0),
        struct.pack(">QI", 0, 20 * 1024**2),
        struct.pack(">III", SESSION_FLAG >> 32, 0, 1080),
    ],
)
def test_corrupt_lengths_rejected(data):
    with pytest.raises(ValueError):
        read_packet(io.BytesIO(data))


def test_bad_server_hash_is_not_executed(tmp_path):
    folder = tmp_path / "tools"
    folder.mkdir()
    (folder / "scrcpy-server-v5.0.1").write_bytes(b"bad")
    with pytest.raises(AdbError, match="校验失败"):
        setup_video(tmp_path, download=False)


def test_disk_and_quota_do_not_delete_old_recordings(tmp_path, monkeypatch):
    existing = tmp_path / "old.mkv"
    existing.write_bytes(b"saved")
    writer = SegmentWriter(tmp_path / "video", quota_root=tmp_path, quota_bytes=5)
    with pytest.raises(OSError, match="上限"):
        writer.check_space()
    monkeypatch.setattr("doudizhu_lab.devices.video.shutil.disk_usage", lambda p: SimpleNamespace(free=1))
    with pytest.raises(OSError, match="2 GiB"):
        writer.check_space()
    assert existing.read_bytes() == b"saved"


def test_replay_cannot_send_device_input_even_with_matching_serial(frame):
    with pytest.raises(AdbError, match="回放"):
        AdbClient(serial=frame.serial).tap(1, 1, replace(frame, transport="replay"))


def test_replay_gap_is_explicit(tmp_path):
    video = tmp_path / "video"
    video.mkdir()
    (video / "index.json").write_text(
        json.dumps([{"file": "x.mkv", "global_start_us": 1000, "global_end_us": 2000}])
    )
    with pytest.raises(ValueError, match="没有录像"):
        VideoReplay(tmp_path).at(500)


@pytest.mark.local_data
@pytest.mark.skipif(
    not (local_data_root() / "video-probe/index.json").exists(), reason="仅本机保留真机验收录像"
)
def test_actual_mkv_seeking_roundtrip(tmp_path):
    import shutil

    shutil.copytree(local_data_root() / "video-probe", tmp_path / "video")
    video = VideoReplay(tmp_path)
    first = video.at(video.start_us)
    second = video.at(first.media_pts_us + 1, after=True)
    assert (first.width, first.height) == (2400, 1080)
    assert second.media_pts_us > first.media_pts_us
    assert video.at(second.media_pts_us - 1).media_pts_us == first.media_pts_us
    assert first.transport == "replay"


@pytest.mark.parametrize("persistent", [False, True])
def test_index_windows_sharing_violation_is_bounded_and_preserves_previous(tmp_path, monkeypatch, persistent):
    writer = SegmentWriter(tmp_path)
    index = tmp_path / "index.json"
    index.write_text('["previous"]')
    original = Path.replace
    attempts = []
    waits = []

    def replace_file(path, target):
        attempts.append(path)
        assert index.read_text() == '["previous"]'
        if persistent or len(attempts) < 3:
            error = PermissionError("模拟 Windows 短暂文件占用")
            error.winerror = 5
            raise error
        return original(path, target)

    monkeypatch.setattr(Path, "replace", replace_file)
    monkeypatch.setattr("doudizhu_lab.devices.video.sleep", waits.append)
    if persistent:
        with pytest.raises(PermissionError):
            writer._index()
        assert len(attempts) == 5 and index.read_text() == '["previous"]'
    else:
        writer._index()
        assert len(attempts) == 3 and json.loads(index.read_text()) == []
    assert sum(waits) <= 0.51
