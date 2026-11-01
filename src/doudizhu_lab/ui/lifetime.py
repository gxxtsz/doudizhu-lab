"""在 GUI 线程回收 Python 引用环，避免后台识别的内存分配触发 QWidget 析构。"""

import gc
import threading

from PySide6.QtCore import QCoreApplication, QEvent, QObject, QTimer


class GuiLifetime(QObject):
    """Qt 对象可能通过 Python 槽形成引用环；自动 GC 不保证在 GUI 线程运行。

    记录已证实后台分析线程会触发自动 GC。因此应用存活期间由 GUI 定时器进行
    分代回收，退出时恢复原 GC 状态；正常引用计数释放及 Qt 父子所有权照常工作。
    """

    def __init__(self, application):
        super().__init__(application)
        self.previous = gc.isenabled()
        gc.disable()
        self.cycles = 0
        self.timer = QTimer(self)
        self.timer.setInterval(500)
        self.timer.timeout.connect(self.collect)
        self.timer.start()
        application.aboutToQuit.connect(self.restore)

    def collect(self):
        """按固定 GUI 生命周期边界送达延迟销毁，再回收引用环，最长30秒全代回收。"""
        if threading.current_thread() is not threading.main_thread():
            raise RuntimeError("Qt 引用环只能在 GUI 线程回收")
        self.cycles += 1
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        gc.collect(2 if self.cycles % 60 == 0 else 0)

    def restore(self):
        self.timer.stop()
        if self.previous:
            gc.enable()


def ensure_gui_lifetime():
    """每个 QApplication 只安装一次，强所有权位于 application，随应用退出释放。"""
    app = QCoreApplication.instance()
    if app is None or threading.current_thread() is not threading.main_thread():
        raise RuntimeError("请在 GUI 线程创建 QApplication 后安装生命周期管理")
    if not hasattr(app, "_ddz_gui_lifetime"):
        app._ddz_gui_lifetime = GuiLifetime(app)
    return app._ddz_gui_lifetime
