"""用独立环境内已安装wheel的Python运行：验证安装路径、快照、中文目录及窗口收尾。

示例：.venv-wheel/Scripts/python.exe -I tools/smoke_installed.py --output data/migration/安装包冒烟.json
本工具禁止ADB传输，不连接手机、不加载模型、不开始录制；临时资料退出时删除。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PySide6.QtCore import QCoreApplication, QEvent
from PySide6.QtWidgets import QApplication

from doudizhu_lab import __version__
from doudizhu_lab.devices.adb import AdbClient
from doudizhu_lab.domain import Profile
from doudizhu_lab.paths import PACKAGE_ROOT, project_files, workspace_root
from doudizhu_lab.storage.config import Store, read_rgb, write_rgb
from doudizhu_lab.storage.session import SessionStore
from doudizhu_lab.ui.window import MainWindow


def main() -> None:
    """失败即返回非零；成功保存独立报告，所有Qt对象均在主线程创建和关闭。"""
    parser = argparse.ArgumentParser(description="wheel安装后离线冒烟，不连接手机")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if workspace_root() is not None:
        raise RuntimeError("当前导入来自源码；请先安装wheel，并用独立环境的python -I执行")
    sent = []

    def forbidden(*a, **kw):
        sent.append(True)
        raise AssertionError("安装包冒烟不得调用ADB")

    AdbClient._run = forbidden
    resources = project_files()
    report = {
        "source": "installed_wheel",
        "version": __version__,
        "resources": sorted(resources),
        "passed": False,
    }
    application = QApplication([])
    application.setQuitOnLastWindowClosed(False)
    with tempfile.TemporaryDirectory(prefix="doudizhu-lab-wheel-") as temp:
        root = Path(temp) / "中文目录"
        store = Store(root)
        assert not store.profile.bot_controls["verified"]
        image = np.full((40, 80, 3), 220, np.uint8)
        write_rgb(root / "合成.png", image)
        assert np.array_equal(read_rgb(root / "合成.png"), image)
        session = SessionStore.create(root / "会话", Profile(), [], "offline")
        try:
            session.verify_snapshot()
            package = session.folder / "engine/src/doudizhu_lab"
            assert (package / "agents/worker.py").is_file()
            assert (package / "_vendor/bidding/BidModel.original.txt").is_file()
            assert session.metadata()["application_version"] == __version__
            report["snapshot_files"] = len(session.metadata()["engine_manifest"])
        finally:
            session.close()
        window = MainWindow(store, auto_refresh=False)
        window.show()
        application.processEvents()
        assert window.bot_mode.currentData() == "suggest" and window.bot is None and not window.running
        window.close()
        application.processEvents()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        assert not window.isVisible()
        result = subprocess.run(
            [sys.executable, "-I", "-X", "utf8", "-m", "doudizhu_lab", "--version"],
            cwd=temp,
            capture_output=True,
            text=True,
            encoding="utf8",
            timeout=15,
        )
        assert result.returncode == 0 and __version__ in result.stdout, result.stderr
        report.update(
            window_closed=True,
            default_suggest=True,
            device_commands=len(sent),
            chinese_paths=True,
            installed_under_site_packages="site-packages" in PACKAGE_ROOT.parts,
            qt_platform=application.platformName(),
            passed=not sent,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
