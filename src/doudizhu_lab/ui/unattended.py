"""无人值守工作台适配；保留既有只录像入口，连续任务交由隔离执行进程运行。"""

from __future__ import annotations

import queue
from copy import deepcopy
from dataclasses import replace
from time import monotonic

from doudizhu_lab.runtime.supervisor import TaskSupervisor
from doudizhu_lab.ui.window import MainWindow


class UnattendedWindow(MainWindow):
    """GUI 不持有生产识别器、录像流水线或机器人锁；关闭与停止均以取消意图优先。"""

    def __init__(self, *args, supervisor_factory=TaskSupervisor, **kwargs):
        """创建只读工作台；工厂仅供隔离 GUI 测试，不改变默认生产进程隔离行为。"""
        self.supervisor = None
        self._supervisor_factory = supervisor_factory
        self._supervisor_start_pending = ""
        self._supervisor_close_requested = False
        self._last_task_state = "idle"
        super().__init__(*args, **kwargs)

    @property
    def settings_locked(self):
        return bool(
            self._supervisor_start_pending
            or self.supervisor
            and self.supervisor.active
            or super().settings_locked
        )

    def load_bot(self):
        """异步预热隔离进程内的本地模型；失败重试不授予任务或设备输入。"""
        if self._exit_requested:
            return
        if self.supervisor is None:
            self.supervisor = self._supervisor_factory(
                self.store.root, self.store.profile, self.store.adb_path
            )
            self.bot = self.supervisor
        else:
            self.supervisor.commands.put(("load", None))
        self.render_dashboard()

    def start_continuous(self, *, require_model: bool = True):
        """显式开始无人值守任务；先让旧只读截图结束，重复点击不创建第二任务。"""
        if (
            self._exit_requested
            or self._supervisor_start_pending
            or self.supervisor
            and self.supervisor.active
        ):
            return
        if self.supervisor is None and not require_model:
            self.load_bot()
        if not self.supervisor or require_model and self.supervisor.model_status().state != "ready":
            self.message = "请等待本地模型就绪后开始任务"
            return
        if not self.selected_device or self.selected_device.state != "device":
            self.message = "请先选择已授权的在线手机"
            return
        self.pause_preview()
        self._startup_action = ""
        self._supervisor_start_pending = self.selected_device.serial
        self.task_status, self.message = "preparing", "等待旧画面读取完成后自动开始"
        self._saved_folder = None
        self._dispatch_pending()
        self.render_dashboard()

    def _maybe_startup(self):
        """显式 --continuous 已授权无人值守；初始模型失败也由任务自动重建，不要求点重试。"""
        if self._startup_action == "continuous":
            if (
                not self.busy
                and not self._exit_requested
                and self.selected_device
                and self.selected_device.state == "device"
            ):
                self.start_continuous(require_model=False)
            return
        super()._maybe_startup()

    def _dispatch_pending(self):
        if self._supervisor_start_pending:
            if self.busy or self._exit_requested:
                return
            serial, self._supervisor_start_pending = self._supervisor_start_pending, ""
            self.supervisor.start_task(serial)
            self.continuous = self.supervisor
            self._task_started = monotonic()
            return
        if not self.supervisor or not self.supervisor.active:
            super()._dispatch_pending()

    def _done(self, generation, kind, value, error):
        """启动意图后的迟到设备刷新只更新观察，不切换任务设备或触发只读配置锁异常。"""
        if kind == "devices" and self.settings_locked:
            if error:
                self.log(error)
            else:
                self.available_devices = list(value)
            return
        super()._done(generation, kind, value, error)

    def stop_and_save(self, reason="用户停止连续任务", *, outcome="stopped"):
        """GUI 立即取消恢复与租约；后台独立收尾，故障资料不因手动停止而删除。"""
        if isinstance(reason, bool):
            reason = "用户停止连续任务"
        self._supervisor_start_pending = self._startup_action = ""
        if self.supervisor and self.supervisor.active:
            self.supervisor.request_stop(reason)
            self._stop_requested = True
            self.task_status, self.message = "stopping", reason
            self.log(reason)
        else:
            self.task_status = "stopped"
        self.render_dashboard()

    def _tick(self):
        """只轮询本地快照并刷新 GUI 租约；任何执行进程阻塞都不会阻塞本回调。"""
        if self._closing:
            return
        now = monotonic()
        self._max_heartbeat_gap = max(self._max_heartbeat_gap, now - self._heartbeat_time)
        self._heartbeat_time = now
        self._heartbeats += 1
        if self.supervisor:
            self.supervisor.heartbeat()
            while True:
                try:
                    message = self.supervisor.logs.get_nowait()
                except queue.Empty:
                    break
                if message != getattr(self, "_last_supervisor_log", ""):
                    self.log(message)
                    self._last_supervisor_log = message
            if self.supervisor.close_window_requested:
                self.close()
                return
            snapshot = self.supervisor.snapshot
            frame = snapshot.get("frame")
            if frame and frame.frame_id != self._video_identity:
                try:
                    self.canvas.set_frame(frame)
                    self.frame, self._video_identity = frame, frame.frame_id
                except (BufferError, ValueError, RuntimeError) as exc:
                    self.supervisor.commands.put(("fault", f"设备画面呈现失败：{exc}"))
                    if self.supervisor.worker:
                        self.supervisor.worker.cancel(fault=True)
            self._table_state = snapshot.get("state", {})
            if self.continuous:
                progress = self.supervisor.progress()
                self._last_progress = progress
                self.task_status, self.message = progress.status, progress.reason
                if progress.status in ("stopped", "completed"):
                    self._stop_requested = False
                    self._saved_folder = self.supervisor.saved_folder
                    if (
                        self._last_task_state not in ("stopped", "completed")
                        and self.auto_preview
                        and not self._exit_requested
                    ):
                        self.start_preview()
                self._last_task_state = progress.status
        self._maybe_startup()
        self._dispatch_pending()
        if self.running and not self.settings_locked and not self.busy and now >= self._next_capture:
            self._next_capture = now + 1
            self._submit("capture", self.client.capture)
        self.render_dashboard()

    def dashboard_state(self):
        state = super().dashboard_state()
        if self.continuous and self.supervisor:
            progress = self.supervisor.progress()
            state = replace(
                state,
                status=progress.status,
                notice=progress.reason or state.notice,
                completed=progress.completed,
                elapsed=progress.elapsed,
                skipped=progress.skipped,
                retries=progress.retries,
                retry_at=progress.retry_at,
            )
        return state

    def render_dashboard(self):
        super().render_dashboard()
        if self.supervisor:
            state = self.dashboard_state()
            self.retry_button.setEnabled(not self.settings_locked)
            if self.continuous:
                self.policy_label.setWordWrap(True)
                info = f"异常跳过 {state.skipped} 局 · 自动恢复 {state.retries} 次"
                if state.retry_at is not None:
                    info += f" · {max(0, int(state.retry_at - monotonic()))} 秒后检查"
                self.policy_label.setText(self.policy_label.text() + "\n" + info)
                self.stop_button.setEnabled(self.supervisor.active)
                if self.supervisor.active and self.frame:
                    self.preview_label.setText(f"任务录制　·　{self.frame.width} × {self.frame.height}")

    def apply_settings(self, profile, adb_path, serial):
        super().apply_settings(profile, adb_path, serial)
        if self.supervisor:
            self.supervisor.commands.put(("configure", (deepcopy(profile), adb_path)))

    def closeEvent(self, event):
        """异步关闭全部本实例资源；不在 GUI 等待挂起的识别、模型或录像线程。"""
        if self._closing:
            event.accept()
            return
        self._exit_requested = True
        self.pause_preview()
        self._startup_action = self._supervisor_start_pending = ""
        if self.supervisor and not self._supervisor_close_requested:
            self._supervisor_close_requested = True
            self.supervisor.request_close()
        if self.busy or self.supervisor and not self.supervisor.closed.is_set():
            event.ignore()
            self.close_timer.start(100)
            return
        self._closing = True
        self.timer.stop()
        self.task_timer.stop()
        self.close_timer.stop()
        self.pool.close()
        event.accept()


def create_main_window(*args, **kwargs):
    """命令行生产入口；显式只录像及正式验收保持原始完整录制路径，不授予连续任务。"""
    window_type = (
        MainWindow
        if kwargs.get("startup_action") == "record" or kwargs.get("acceptance")
        else UnattendedWindow
    )
    return window_type(*args, **kwargs)
