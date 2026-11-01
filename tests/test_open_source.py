"""开源分发边界、安装路径及领域隔离的行为回归，不访问真实设备。"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from doudizhu_lab import __version__
from doudizhu_lab.domain import Frame, GameTracker, TableObservation
from doudizhu_lab.evaluation.scope import load_scope
from doudizhu_lab.paths import PACKAGE_ROOT, resolve_data_dir
from doudizhu_lab.storage.config import Store


def test_domain_import_does_not_load_device_gui_or_model_process():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import doudizhu_lab.domain; assert not any(n.startswith(('PySide6', 'torch', 'doudizhu_lab.devices', 'doudizhu_lab.agents', 'doudizhu_lab.vision')) for n in sys.modules)",
        ],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
    assert Frame and GameTracker and TableObservation and __version__ == "0.5.0"


def test_explicit_data_directory_beats_environment_and_cwd(tmp_path, monkeypatch):
    monkeypatch.setenv("DDZ_DATA_DIR", str(tmp_path / "environment"))
    monkeypatch.chdir(tmp_path)
    assert resolve_data_dir() == tmp_path / "environment"
    assert Store(tmp_path / "explicit").root == tmp_path / "explicit"


def test_installed_paths_use_user_directory_and_embedded_project(tmp_path, monkeypatch):
    import doudizhu_lab.paths as paths

    monkeypatch.delenv("DDZ_DATA_DIR", raising=False)
    monkeypatch.setattr(paths, "workspace_root", lambda: None)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "user"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    actual = resolve_data_dir()
    assert actual in (tmp_path / "user/DoudizhuLab", tmp_path / "xdg/doudizhu-lab")
    assert not actual.is_relative_to(PACKAGE_ROOT)


def test_adb_discovery_uses_environment_then_path(monkeypatch):
    from doudizhu_lab.devices.adb import discover_adb

    monkeypatch.setenv("ADB_PATH", "configured-adb")
    monkeypatch.setattr("doudizhu_lab.devices.adb.shutil.which", lambda _: "path-adb")
    assert discover_adb() == "configured-adb"
    monkeypatch.delenv("ADB_PATH")
    assert discover_adb() == "path-adb"


def test_scope_is_explicit_and_does_not_include_event_answers(tmp_path):
    path = tmp_path / "scope.json"
    value = {"source": "独立合成范围", "games": [{"start_us": 1, "end_us": 100, "incomplete": False}]}
    path.write_text(json.dumps(value), encoding="utf8")
    assert load_scope(path) == value


@pytest.mark.parametrize(
    "games",
    [
        [],
        [{"start_us": 3, "end_us": 2, "incomplete": False}],
        [{"start_us": 1, "end_us": 3, "incomplete": False}, {"start_us": 2, "end_us": 4, "incomplete": True}],
    ],
)
def test_scope_rejects_empty_inverted_or_overlapping_ranges(tmp_path, games):
    path = tmp_path / "scope.json"
    path.write_text(json.dumps({"source": "合成", "games": games}), encoding="utf8")
    with pytest.raises(ValueError):
        load_scope(path)


def test_only_runtime_ocr_assets_remain():
    from doudizhu_lab.vision.assets import ASSETS
    from doudizhu_lab.vision.ocr import build_backend

    assert set(ASSETS) == {"ocr_en", "ocr_zh", "ocr_det"}
    with pytest.raises(ValueError, match="未知"):
        build_backend("retired-backend", Path("not-created"))


def test_history_readonly_open_preserves_bytes_and_refuses_updates(tmp_path):
    import hashlib
    import sqlite3

    from doudizhu_lab.storage.session import SessionStore

    session = SessionStore(tmp_path)
    session.set_meta("history", {"original": True})
    session.close()
    path = tmp_path / "session.sqlite3"
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    source = SessionStore(tmp_path, readonly=True)
    try:
        assert source.metadata()["history"] == {"original": True}
        with pytest.raises(sqlite3.OperationalError):
            source.set_meta("history", {"original": False})
    finally:
        source.close()
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


def test_offline_injected_stream_does_not_create_phone_clients_or_need_server(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from doudizhu_lab.runtime.recording import RecordingPipeline

    def forbidden(*a, **kw):
        raise AssertionError("离线源不应准备手机服务器或创建ADB客户端")

    monkeypatch.setattr("doudizhu_lab.runtime.recording.setup_video", forbidden)
    monkeypatch.setattr("doudizhu_lab.runtime.recording.AdbClient", forbidden)
    stream = SimpleNamespace(close=lambda: None)
    pipeline = RecordingPipeline(Store(tmp_path), "replay", analyze=False, stream=stream)
    try:
        assert pipeline.client is None and pipeline.control_client is None and pipeline.stream is stream
    finally:
        pipeline.stop()
        pipeline.session.close()
    with pytest.raises(ValueError, match="离线"):
        RecordingPipeline(Store(tmp_path), "real-device", analyze=False, stream=stream)
