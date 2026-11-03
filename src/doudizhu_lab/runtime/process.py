"""可回收执行进程与非阻塞父端代理；Qt 不进入执行进程，阻塞计算不进入界面线程。"""

from __future__ import annotations

import multiprocessing as mp
import os
import queue
import signal
import subprocess
import threading
from time import monotonic


class OwnedProcessTree:
    """仅管理本次创建的进程树；Windows 使用专属 Job，不按名称杀进程或重启 ADB。"""

    def __init__(self):
        self.handle = None
        if os.name == "nt":
            import ctypes
            from ctypes import wintypes

            self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            self.kernel.CreateJobObjectW.restype = wintypes.HANDLE
            self.kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
            self.kernel.OpenProcess.restype = wintypes.HANDLE
            self.kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
            self.kernel.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
            self.kernel.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
            self.kernel.CloseHandle.argtypes = [wintypes.HANDLE]
            self.handle = self.kernel.CreateJobObjectW(None, None)
            if not self.handle:
                raise OSError(ctypes.get_last_error(), "无法创建任务专属进程组")

    def assign(self, process):
        """在允许工作进程创建模型或录像子进程之前绑定归属；失败不能继续启动。"""
        self.pid = process.pid
        if self.handle:
            import ctypes

            handle = self.kernel.OpenProcess(0x0100 | 0x0001, False, process.pid)
            try:
                if not handle or not self.kernel.AssignProcessToJobObject(self.handle, handle):
                    raise OSError(ctypes.get_last_error(), "无法绑定任务执行进程")
            finally:
                if handle:
                    self.kernel.CloseHandle(handle)

    def terminate(self):
        """只终止专属组；公共 ADB 服务由父端在组外准备，不属于该组。"""
        if self.handle:
            self.kernel.TerminateJobObject(self.handle, 1)
        elif getattr(self, "pid", None):
            try:
                os.killpg(self.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    def close(self):
        """释放父端组句柄；未配置随句柄关闭杀进程，允许独立录像按健康超时尾录。"""
        if self.handle:
            self.kernel.CloseHandle(self.handle)
            self.handle = None


def _bootstrap(config, connection, heartbeat, cancelled, ready, entrypoint):
    """spawn 的纯 Python 入口；父端确认进程归属后才导入模型并创建子进程。"""
    if os.name != "nt":
        os.setsid()
    if not ready.wait(15):
        return
    if entrypoint is None:
        from doudizhu_lab.runtime.session_process import run_session_process

        entrypoint = run_session_process
    entrypoint(config, connection, heartbeat, cancelled)


class SessionProcess:
    """线程安全的父端代理；IPC 收发和图像解码都在后台，poll 只取本地消息。

    config 必须是冻结的纯 Python 数据；entrypoint 仅供隔离进程故障测试注入。
    输入租约由 GUI heartbeat 刷新，cancel 立即减权；kill 只用于收尾期限耗尽。
    """

    def __init__(self, config, *, entrypoint=None):
        config = dict(config)
        self.config, self.entrypoint = config, entrypoint
        context = mp.get_context("spawn")
        self.faulted = context.Event()
        config["_faulted"] = self.faulted
        self.connection, self.child_connection = context.Pipe()
        self.heartbeat_at = context.Value("d", monotonic(), lock=False)
        self.cancelled, self.ready = context.Event(), context.Event()
        self.process = context.Process(
            target=_bootstrap,
            args=(config, self.child_connection, self.heartbeat_at, self.cancelled, self.ready, entrypoint),
            name="斗地主任务执行",
            daemon=False,
        )
        self.tree = None
        self.events, self.outgoing = queue.SimpleQueue(), queue.SimpleQueue()
        self.latest = {}
        self.error = ""
        self._finished = threading.Event()
        self._launched = False
        self._launcher = threading.Thread(target=self._launch, name="启动隔离执行进程", daemon=True)
        self._launcher.start()

    def _launch(self):
        try:
            self.tree = OwnedProcessTree()
            self.process.start()
            self.tree.assign(self.process)
            self._launched = True
            self.ready.set()
            self.child_connection.close()
            threading.Thread(target=self._read, name="执行消息接收", daemon=True).start()
            threading.Thread(target=self._write, name="执行命令发送", daemon=True).start()
        except Exception as exc:
            self.error = str(exc)
            self.cancelled.set()
            if self.process.pid and self.process.is_alive():
                self.process.kill()
            self._finished.set()

    def _read(self):
        try:
            while True:
                value = self.connection.recv()
                if value.get("kind") == "snapshot":
                    if value.get("jpeg") is not None:
                        import cv2
                        import numpy as np

                        from doudizhu_lab.domain.models import Frame

                        rgb = cv2.cvtColor(
                            cv2.imdecode(np.frombuffer(value.pop("jpeg"), np.uint8), cv2.IMREAD_COLOR),
                            cv2.COLOR_BGR2RGB,
                        )
                        value["frame"] = Frame(rgb=rgb, **value.pop("frame_meta"))
                    self.latest = value
                else:
                    self.events.put(value)
        except (EOFError, OSError, ValueError) as exc:
            if not self.cancelled.is_set():
                self.error = str(exc) or "执行进程消息通道已关闭"
        finally:
            self._finished.set()

    def _write(self):
        try:
            while not self._finished.is_set():
                try:
                    command = self.outgoing.get(timeout=0.1)
                except queue.Empty:
                    continue
                if command["kind"] in ("start", "probe") and self.config.get("adb"):
                    # 在专属 Job 外准备公共 server，随后子进程的 adb 客户端只使用已有服务。
                    try:
                        subprocess.run(
                            [self.config["adb"], "start-server"],
                            capture_output=True,
                            timeout=8,
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                        )
                    except (OSError, subprocess.TimeoutExpired):
                        pass  # 子端按原设备状态产生结构化等待原因。
                if not self.cancelled.is_set() or command["kind"] in ("ack", "close"):
                    self.connection.send(command)
        except (OSError, EOFError, ValueError):
            self._finished.set()

    def send(self, kind: str, **value) -> None:
        """无阻塞登记命令；命令自带本实例任务和代号，迟到命令不能发往替代进程。"""
        self.outgoing.put({"kind": kind, **value})

    def heartbeat(self) -> None:
        """仅 GUI 主线程刷新输入租约；后台看护重试不能替代界面健康证据。"""
        self.heartbeat_at.value = monotonic()

    def poll(self) -> tuple[list[dict], dict]:
        """返回已接收的关键事件和最新只读快照，不等待跨进程锁或执行计算。"""
        values = []
        while True:
            try:
                values.append(self.events.get_nowait())
            except queue.Empty:
                return values, self.latest

    @property
    def alive(self) -> bool:
        return self._launcher.is_alive() or self._launched and self.process.is_alive()

    def cancel(self, *, fault: bool = False) -> None:
        """立即使租约失效；已发设备命令仍保留为在途，后台开始受期限约束的收尾。"""
        if fault:
            self.faulted.set()
        self.cancelled.set()
        self.send("close")

    def kill(self) -> None:
        """在父看护线程回收本实例及子进程；不等待旧线程、不触碰其他任务或公共服务。"""
        self.cancelled.set()
        if self.tree:
            self.tree.terminate()
        if self._launched and self.process.is_alive():
            self.process.kill()

    def dispose(self) -> None:
        """进程退出后关闭本地资源；读线程即使滞后，也只持有已经失效的本实例通道。"""
        self._finished.set()
        self.connection.close()
        if self._launched:
            self.process.join(timeout=0)
            if not self.process.is_alive():
                self.process.close()
        if self.tree:
            self.tree.terminate()
            self.tree.close()
