"""独立 Qt 进程验证 Windows 缩放下的工作台布局；截图仅写入测试临时目录。"""

import os
import subprocess
import sys

import pytest

PROBE = r"""
import sys
from pathlib import Path
from PySide6.QtCore import QPoint, QRect
from PySide6.QtGui import QFont, QFontDatabase
from PySide6.QtWidgets import QApplication, QScrollArea
from doudizhu_lab.domain.models import Device
from doudizhu_lab.runtime.bot import BotRuntime, ModelStatus
from doudizhu_lab.ui.window import MainWindow
from doudizhu_lab.ui.settings import SettingsDialog
from doudizhu_lab.ui.replay import SessionDialog
from doudizhu_lab.storage.config import Store
out = Path(sys.argv[1])
app = QApplication([])
for name in ("msyh.ttc", "msyhbd.ttc"):
    QFontDatabase.addApplicationFont("C:/Windows/Fonts/" + name)
app.setFont(QFont("Microsoft YaHei UI", 10))
w = MainWindow(Store(out), auto_refresh=False, auto_load_model=False, auto_preview=False)
w.timer.stop()
w.available_devices = [Device("test", "device", "已连接的安卓手机设备")]
w.select_device("test")
w.bot = BotRuntime(out, w.input_gate, threaded=False)
w.show()
for width, height in ((1440, 900), (1100, 720)):
    w.resize(width, height)
    for status in ("idle", "preparing", "running", "paused", "failed", "stopping"):
        w.task_status = status
        w.bot._model_status = ModelStatus("failed", "模型校验失败" * 50) if status == "failed" else ModelStatus("ready")
        w.message = "暂停原因：等待清晰画面后恢复。" if status == "paused" else ""
        w.render_dashboard()
        app.processEvents()
        app.processEvents()
        viewport = w.findChild(QScrollArea).viewport()
        for button in (w.action_button, w.stop_button):
            rect = QRect(button.mapTo(viewport, QPoint()), button.size())
            assert viewport.rect().contains(rect), (width, status, button.text(), rect, viewport.rect())
        for button in (w.settings_button, w.replay_button):
            assert w.rect().contains(QRect(button.mapTo(w, QPoint()), button.size()))
        assert w.canvas.width() >= 440 and w.canvas.height() >= 260
        w.grab().save(str(out / f"运行-{width}-{status}.png"))
d = SettingsDialog(w)
d.show()
app.processEvents()
assert d.rect().contains(QRect(d.buttons.mapTo(d, QPoint()), d.buttons.size()))
d.grab().save(str(out / "设置.png"))
d.close()
r = SessionDialog(w.store)
r.resize(1100, 720)
r.show()
app.processEvents()
canvas_right = r.canvas.mapTo(r, QPoint(r.canvas.width(), 0)).x()
tree_left = r.events.mapTo(r, QPoint()).x()
assert canvas_right < tree_left, (canvas_right, tree_left)
assert not r.play.isEnabled() and not r.edit_button.isEnabled()
r.grab().save(str(out / "回放.png"))
r.close()
w.close()
"""


@pytest.mark.gui
@pytest.mark.parametrize("scale", ["1", "1.25", "1.5"])
def test_workspace_and_replay_layout_at_windows_scales(tmp_path, scale):
    environment = dict(os.environ, QT_QPA_PLATFORM="offscreen", QT_SCALE_FACTOR=scale)
    result = subprocess.run(
        [sys.executable, "-c", PROBE, str(tmp_path)],
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=25,
    )
    assert result.returncode == 0, result.stderr
