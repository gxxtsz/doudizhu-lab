"""测试使用隔离目录和模拟设备，不向真实手机发送命令。"""

import gc
import os
from pathlib import Path
from time import monotonic

import numpy as np
import pytest

from doudizhu_lab.domain.models import Frame

# 自动化事件测试用无窗口平台并显式加载字体；原生窗口另外执行连续运行验收。
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def pytest_addoption(parser):
    """历史数据与真实模型必须显式启用，公共CI不会隐式下载或读取私人资料。"""
    parser.addoption("--local-data-dir", type=Path, help="既有本机样本目录，仅作只读回归输入")
    parser.addoption("--run-model-tests", action="store_true", help="启用已安装固定模型的本地测试")


def pytest_configure(config):
    """在收集测试前发布显式资料路径，使收集期样本存在性检查也使用同一来源。"""
    path = config.getoption("--local-data-dir")
    if path is not None:
        if not path.is_dir():
            raise pytest.UsageError("本机回归目录不存在")
        os.environ["DDZ_TEST_DATA_DIR"] = str(path.resolve())
    else:
        os.environ.pop("DDZ_TEST_DATA_DIR", None)


def pytest_collection_modifyitems(config, items):
    """清楚标明未启用的私人样本和真实模型用例，不将跳过算作通过。"""
    for item in items:
        if "local_data" in item.keywords and config.getoption("--local-data-dir") is None:
            item.add_marker(pytest.mark.skip(reason="需显式 --local-data-dir 提供本机历史资料"))
        if "model" in item.keywords and not config.getoption("--run-model-tests"):
            item.add_marker(pytest.mark.skip(reason="需显式 --run-model-tests 启用固定本机模型"))


@pytest.hookimpl(wrapper=True, tryfirst=True)
def pytest_runtest_teardown(item):
    """在 pytest-qt 关闭窗口之后，显式完成 GUI 线程的延迟删除。

    pytest-qt 的 processEvents 不等于应用主事件循环；deleteLater 的事件需要
    单独送达。及时释放已关闭窗口，避免其 Python 引用环拖到后续用例或后台线程
    的自动垃圾回收。保留自动 GC，仅在此明确的 GUI 生命周期边界再回收一次。
    """
    result = yield
    from PySide6.QtCore import QCoreApplication, QEvent

    if QCoreApplication.instance() is not None:
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        gc.collect()
    return result


@pytest.fixture(scope="session")
def qapp_args():
    """用 Qt 启动参数明确测试平台，避免平台选择只依赖进程环境。

    pytest-qt 创建唯一 QApplication 时传入平台和样式；外部仍可指定 windows 做原生测试。
    平台参数只固定测试环境，窗口生命周期是否安全仍由实际交互测试验证。
    """
    return ["斗地主界面测试", "-platform", os.environ["QT_QPA_PLATFORM"], "-style", "Fusion"]


@pytest.fixture(scope="session", autouse=True)
def chinese_test_fonts(qapp):
    """无窗口平台不会自动载入 Windows 字体；显式加载以保持中文布局可测。"""
    from PySide6.QtGui import QFont, QFontDatabase

    from doudizhu_lab.ui.lifetime import ensure_gui_lifetime

    ensure_gui_lifetime()

    # 整个测试会话共用 QApplication，某个用例关闭最后一个窗口后仍需继续创建窗口。
    # 应用的正常退出行为由原生独立进程检查，不能在用例间请求退出共享事件循环。
    qapp.setQuitOnLastWindowClosed(False)
    for name in ("msyh.ttc", "msyhbd.ttc", "segoeui.ttf"):
        path = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / name
        if path.is_file():
            QFontDatabase.addApplicationFont(str(path))
    qapp.setFont(QFont("Microsoft YaHei UI", 10))
    assert qapp.platformName() == os.environ["QT_QPA_PLATFORM"]


@pytest.fixture
def frame():
    return Frame("test-device", np.zeros((1080, 2400, 3), np.uint8), monotonic(), "2026-10-31T00:00:00Z", 12)
