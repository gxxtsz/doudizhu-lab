"""旧录像原速诊断的编码转换、停止、生产队列及验收口径回归。"""

import json
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import pytest

from doudizhu_lab.devices.adb import AdbClient
from doudizhu_lab.evaluation.paced_replay import (
    EncodedReplayStream,
    PacedReplayPipeline,
    annexb_config,
    paced_replay,
)
from doudizhu_lab.evaluation.session import evaluate_session
from doudizhu_lab.storage.config import Store
from doudizhu_lab.storage.session import SessionStore
from doudizhu_lab.vision.table import TableObservation


def tiny_recording(tmp_path):
    """生成真正的短 H.264/MKV，以验证字节转换后仍可被生产解码器读取。"""
    folder = tmp_path / "source"
    video = folder / "video"
    video.mkdir(parents=True)
    with av.open(str(video / "one.mkv"), "w") as container:
        stream = container.add_stream("libx264", rate=15)
        stream.width = stream.height = 32
        stream.pix_fmt = "yuv420p"
        stream.options = {"preset": "ultrafast", "tune": "zerolatency", "bf": "0"}
        for index in range(5):
            frame = av.VideoFrame.from_ndarray(np.full((32, 32, 3), index * 30, np.uint8), format="rgb24")
            frame.pts, frame.time_base = index, Fraction(1, 15)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    (video / "index.json").write_text(
        json.dumps(
            [
                {
                    "file": "one.mkv",
                    "global_start_us": 1_000_000,
                    "global_end_us": 1_267_000,
                    "width": 32,
                    "height": 32,
                    "frames": 5,
                    "closed": True,
                }
            ]
        ),
        encoding="utf-8",
    )
    return folder


@pytest.mark.parametrize(
    "data", [b"", b"\x01" * 6, b"\x01\x64\x00\x28\xff\xe1\x00\x05\x67", b"\x01\x64\x00\x28\xff\xe0\x00"]
)
def test_truncated_or_empty_configuration_is_rejected(data):
    with pytest.raises(ValueError):
        annexb_config(data)


def check_encoded_source(folder):
    source = EncodedReplayStream(folder)
    source.start()
    packets = list(source.packets())
    assert packets[0].session and packets[1].config
    media = packets[2:]
    assert len(media) == 5
    assert [p.pts_us for p in media] == [1_000_000, 1_067_000, 1_133_000, 1_200_000, 1_267_000]
    decoder = av.CodecContext.create("h264", "r")
    decoded = []
    for item in media:
        decoded.extend(decoder.decode(av.Packet(item.data)))
    assert len(decoded) == 5
    source = EncodedReplayStream(folder)
    source.start()
    source.close()
    assert not [p for p in source.packets() if not p.session and not p.config]


def check_production_pipeline(folder, tmp_path):
    def forbidden(*args, **kwargs):
        raise AssertionError("原速回放不得访问手机")

    AdbClient._run = forbidden
    PacedReplayPipeline._recognizer_factory = lambda self: (
        lambda f: (TableObservation(f.frame_id, f.media_pts_us), None)
    )
    output = paced_replay(folder, Store(tmp_path / "diagnostic"))
    session = SessionStore(output)
    meta = session.metadata()
    report = meta["paced_report"]
    # 回放不得持有真实输入客户端，也不需要下载或准备手机视频服务器。
    assert meta["analysis_mode"] == "paced_replay" and not meta["formal_acceptance"]
    assert report["decoded_frames"] == report["sent_media_packets"] == 5
    assert 2 <= report["analyzed_frames"] <= 3
    assert report["pending_analysis"] == report["pending_decode"] == report["pending_write"] == 0
    assert not report["interrupted"] and not report["unfinished_threads"]
    session.close()
    segments = json.loads((output / "video/index.json").read_text())
    assert all(s["closed"] for s in segments) and sum(s["frames"] for s in segments) == 5
    with av.open(str(output / "video" / segments[0]["file"])) as container:
        assert len(list(container.decode(video=0))) == 5
    assert Path(meta["video_source"]).parts
    evaluation = evaluate_session(output)
    assert evaluation["live"] is None and evaluation["offline"] is None
    assert evaluation["paced"] == evaluation["automatic"]
    assert "非手机实时" in evaluation["confirmation_delay"]["basis"]
    assert not evaluation["target_passed"]


def run_cli_probe(tmp_path, mode):
    """沿用实际 CLI 的独立进程边界，子进程任何异常或原生崩溃都必须使测试失败。

    将 PyAV 编码集成与 pytest-qt 的常驻 QApplication 混在同一测试进程时，Windows
    出现过随后关闭 QWidget 的访问异常。尚未定位其原生根因，不以重试或忽略退出码
    消除失败；诊断命令实际在独立进程运行，因此在该真实边界验证全部原断言。
    """
    result = subprocess.run(
        [sys.executable, "-X", "utf8", str(Path(__file__).resolve()), str(tmp_path), mode],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_encoded_source_preserves_all_frames_pts_and_can_stop(tmp_path):
    run_cli_probe(tmp_path, "source")


def test_production_pipeline_writes_decodes_and_never_calls_adb(tmp_path):
    run_cli_probe(tmp_path, "pipeline")


if __name__ == "__main__":
    directory = Path(sys.argv[1])
    recording = tiny_recording(directory)
    if sys.argv[2] == "source":
        check_encoded_source(recording)
    else:
        check_production_pipeline(recording, directory)
