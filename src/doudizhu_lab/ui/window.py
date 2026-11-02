"""连续游戏工作台；GUI 只协调状态，设备传输、模型和录像各自在后台运行。"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from time import monotonic

from PySide6.QtCore import Qt, QTimer, Slot
from PySide6.QtWidgets import QMainWindow

from doudizhu_lab.devices.adb import AdbClient
from doudizhu_lab.domain.models import RANK_LABELS, Device, Frame
from doudizhu_lab.domain.settings import Profile, validate_bot_profile
from doudizhu_lab.domain.table import PHASE_NAMES, SEAT_NAMES
from doudizhu_lab.runtime.bot import BotRuntime, InputGate, ModelStatus
from doudizhu_lab.runtime.continuous import ContinuousProgress, ContinuousRuntime
from doudizhu_lab.runtime.recording import RecordingPipeline
from doudizhu_lab.runtime.worker import SerialWorker, Task
from doudizhu_lab.storage.config import Store
from doudizhu_lab.ui.components import build_main_window
from doudizhu_lab.ui.lifetime import ensure_gui_lifetime
from doudizhu_lab.ui.replay import SessionDialog
from doudizhu_lab.ui.state import DashboardState, duration_text
from doudizhu_lab.vision.flow import SCENE_NAMES

DEVICE_STATES = {"device": "已授权", "unauthorized": "未授权", "offline": "离线", "no": "权限不足"}
TASK_NAMES = {
    "idle": "等待开始",
    "preparing": "正在准备",
    "running": "运行中",
    "paused": "已暂停",
    "stopping": "正在保存",
    "completed": "已完成",
    "stopped": "已停止",
    "failed": "需要处理",
}


class MainWindow(QMainWindow):
    """默认只读的运行窗口；自动预览与模型加载不授予输入，只有显式任务才允许执行。"""

    def __init__(
        self,
        store: Store | None = None,
        *,
        auto_refresh: bool = True,
        auto_load_model: bool = True,
        auto_preview: bool = True,
        startup_action: str = "",
        acceptance: bool = False,
    ) -> None:
        """在 GUI 线程创建窗口，定时器异步启动后台工作。

        startup_action 可为 preview/record/continuous，优先于自动预览，兼容命令行显式启动。
        测试可分别禁用刷新、模型及预览；acceptance 仅传给录像，不创建隐藏验收控件。
        无效启动动作抛出 ValueError；构造本身不发送手机输入，也不下载模型。
        """
        super().__init__()
        ensure_gui_lifetime()
        if startup_action not in ("", "preview", "record", "continuous"):
            raise ValueError("启动动作必须为预览、录像或连续游戏")
        self.store = store or Store()
        self.client = AdbClient(self.store.adb_path)
        self.pool = SerialWorker()
        self.input_gate = InputGate()
        self.bot: BotRuntime | None = None
        self.continuous: ContinuousRuntime | None = None
        self.available_devices: list[Device] = []
        self.selected_device: Device | None = None
        self.auto_preview, self.acceptance = auto_preview, acceptance
        self._startup_action = startup_action or ("preview" if auto_preview else "")
        self.busy = self.running = self._closing = self._exit_requested = False
        self.generation = 0
        self._kind = ""
        self._task = self._future = None
        self._requested_action = ""
        self._refresh_requested = False
        self._stop_requested = self._record_stopping = False
        self._finish_status = "stopped"
        self.task_status = "idle"
        self.message = ""
        self._resume_error = ""
        self._observed_model = None
        self._last_pause_reason = ""
        self.frame: Frame | None = None
        self.recording = None
        self._saved_folder: Path | None = None
        self._table_state: dict = {}
        self._video_identity = ""
        self._video_epoch = None
        self._last_progress: ContinuousProgress | None = None
        self._task_started = 0.0
        self._next_capture = 0.0
        self._capture_count = 0
        self._capture_durations: list[float] = []
        self._started = self._heartbeat_time = monotonic()
        self._heartbeats = 0
        self._max_heartbeat_gap = 0.0
        build_main_window(self)
        self.task_timer = QTimer(self)
        self.task_timer.setSingleShot(True)
        self.task_timer.timeout.connect(self._poll_task)
        self.close_timer = QTimer(self)
        self.close_timer.setSingleShot(True)
        self.close_timer.timeout.connect(self.close)
        self.timer = QTimer(self)
        self.timer.setInterval(100)
        self.timer.timeout.connect(self._tick)
        self.timer.start()
        if auto_load_model:
            QTimer.singleShot(0, self.load_bot)
        if auto_refresh:
            QTimer.singleShot(0, self.refresh_devices)
        self.render_dashboard()

    @property
    def settings_locked(self) -> bool:
        """GUI 线程读取配置锁；准备、录制、暂停和收尾期间均禁止改会话配置。"""
        return bool(
            self._requested_action
            or self._stop_requested
            or self.recording
            or self._kind in ("record_start", "record_stop")
        )

    def log(self, message: str) -> None:
        """在 GUI 线程追加纯文本日志及本地日志文件；磁盘写入失败不影响停止操作。"""
        text = f"{datetime.now():%H:%M:%S}  {message}"
        cursor = self.log_view.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        cursor.insertText(text + "\n")
        self.log_view.setTextCursor(cursor)
        self.latest_log.setText(text)
        self.latest_log.setToolTip(text)
        try:
            with (self.store.root / "运行日志.txt").open("a", encoding="utf-8") as output:
                output.write(text + "\n")
        except OSError:
            pass

    def toggle_log(self, expanded: bool) -> None:
        """仅在 GUI 线程展开或折叠日志；关键原因始终显示在任务面板。"""
        self.log_view.setVisible(expanded)
        self.log_toggle.setArrowType(Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow)

    def _submit(self, kind: str, action: Callable[[], object]) -> bool:
        """向单任务执行器提交纯 Python 操作；忙碌或关闭时返回 False，不建立排队任务。"""
        if self.busy or self._closing:
            return False
        self.busy, self._kind = True, kind
        self._task = Task(self.generation, kind, action)
        self._future = self.pool.start(self._task)
        self.task_timer.start(25)
        return True

    @Slot()
    def _poll_task(self) -> None:
        """在 GUI 线程回收 Future，再处理结果和待启动请求，避免串行执行器重入。"""
        if self._closing or self._future is None:
            return
        if not self._future.done():
            self.task_timer.start(25)
            return
        value, error = self._future.result()
        generation, kind = self._task.generation, self._task.kind
        self._task = self._future = None
        self.pool.release_completed()
        self.busy, self._kind = False, ""
        self._done(generation, kind, value, error)
        self._dispatch_pending()
        self.render_dashboard()

    def refresh_devices(self) -> None:
        """异步读取设备；截图在途时仅保留一次刷新请求，任务期间不改变设备。"""
        if self.settings_locked or self._exit_requested:
            return
        self.pause_preview()
        self._refresh_requested = True
        self._dispatch_pending()

    def select_device(self, serial: str) -> None:
        """在 GUI 线程选择已发现设备并清空旧帧；任务锁定时拒绝，选择本身不授权输入。"""
        if self.settings_locked:
            raise ValueError("请先停止并保存当前任务，再切换设备")
        device = next((d for d in self.available_devices if d.serial == serial), None)
        self.pause_preview()
        self.selected_device = device
        self.client = AdbClient(self.store.adb_path, device.serial if device else "")
        self.frame, self._table_state = None, {}
        self.canvas.clear()
        if (
            self.auto_preview
            and self._startup_action in ("", "preview")
            and device
            and device.state == "device"
        ):
            self.start_preview()
        self.render_dashboard()

    def pause_preview(self) -> None:
        """仅冻结只读预览并使迟到截图失效；不碰正在保存的录像资源，也不授予输入。"""
        self.running = False
        self.generation += 1
        self.canvas.control_enabled = False

    def start_preview(self) -> None:
        """设备已授权时启用串行只读截图；没有设备、正在录像或退出时保持停用。"""
        if self.recording or self._exit_requested:
            return
        if self.selected_device and self.selected_device.state == "device":
            self.running = True
            self._next_capture = 0

    def toggle_preview(self) -> None:
        """供命令行和模拟测试调用的只读预览开关；日常界面不创建预览控制按钮。"""
        self.pause_preview() if self.running else self.start_preview()

    def load_bot(self) -> None:
        """GUI 线程申请后台加载本地 WP；可重试失败，不重复创建线程，也不联网或授予输入。"""
        if self._exit_requested or self._closing:
            return
        if self.bot is None:
            self.bot = BotRuntime(self.store.root, self.input_gate)
        self.bot.start()
        self.render_dashboard()

    def primary_action(self) -> None:
        """主按钮根据任务结构化状态启动或恢复；不能通过按钮文字推断执行行为。"""
        if self.continuous and self.continuous.status == "paused" and not self._stop_requested:
            self.resume_continuous()
        else:
            self.start_continuous()

    def start_continuous(self) -> None:
        """申请一次连续任务，串行准备录像后启动既有流程；等待截图时也能立即取消。

        模型未就绪或设备未授权时只显示中文原因；本方法不直接发送手机点击。
        准备中的重复调用被忽略，模型与录像的重复创建均由控制器阻止。
        """
        self._request_recording("continuous")

    def start_recording(self) -> None:
        """兼容命令行只录像入口；记录手机画面，不授予机器人执行权限。"""
        self._request_recording("record")

    def _request_recording(self, action: str) -> None:
        """GUI 线程登记启动意图；先冻结旧预览，后台释放后才准备新会话。"""
        if self._exit_requested or self._requested_action or self._stop_requested or self._record_stopping:
            return
        if self.continuous and self.continuous.status in ("running", "paused"):
            return
        if self._kind not in ("", "capture"):
            self.message = "当前任务正在启动或收尾，请稍后再试"
            self.render_dashboard()
            return
        model = self.bot.model_status() if self.bot else ModelStatus("unloaded")
        if action == "continuous" and model.state != "ready":
            self.message = "请等待 WP 模型加载就绪；加载失败时可重试"
            self.render_dashboard()
            return
        if not self.selected_device or self.selected_device.state != "device":
            self.message = "请在设置中选择已授权的在线手机"
            self.render_dashboard()
            return
        self._startup_action = ""
        self.pause_preview()
        self._requested_action = action
        self._stop_requested = False
        self.task_status, self.message = (
            "preparing",
            "等待当前画面读取完成" if self.busy else "正在准备视频通道",
        )
        self._saved_folder = self._last_progress = None
        self._table_state = {}
        self._resume_error = ""
        self._task_started = monotonic()
        self._dispatch_pending()
        self.render_dashboard()

    def create_pipeline(self, serial: str) -> RecordingPipeline:
        """由工作线程创建录像后端；serial 是已选择设备，不启动视频或发送输入。

        模拟子类只替换此工厂。构造失败由 Task 捕获，GUI 展示中文原因。
        """
        return RecordingPipeline(self.store, serial, acceptance=self.acceptance)

    def _dispatch_pending(self) -> None:
        """GUI 线程在后台空闲时消费一次意图；停止和关闭优先于刷新及录像启动。"""
        if self.busy or self._closing:
            return
        if self._stop_requested:
            if self.recording and not self._record_stopping:
                self._begin_save()
            elif not self.recording:
                self._finish_save()
            return
        if self._refresh_requested:
            self._refresh_requested = False
            self._submit("devices", self.client.list_devices)
            return
        if not self._requested_action:
            return
        if self.recording:
            self._launch_continuous()
            return
        serial = self.selected_device.serial
        self.message = "正在准备视频通道"

        def start():
            pipeline = self.create_pipeline(serial)
            try:
                pipeline.start()
            except Exception:
                try:
                    pipeline.stop()
                finally:
                    pipeline.session.close()
                raise
            return pipeline

        self._submit("record_start", start)

    def _launch_continuous(self) -> None:
        """消费有效录像启动意图；迟到的回调在停止后不可能重新授予任务权限。"""
        if not self._requested_action or self._stop_requested or self._exit_requested or not self.recording:
            return
        action, self._requested_action = self._requested_action, ""
        if action == "record":
            self.task_status, self.message = "running", "仅记录画面，未启用自动操作"
            return
        try:
            if self.continuous:
                self.continuous.close()
            self.input_gate.allow()
            self.continuous = ContinuousRuntime(self.store.root, self.input_gate, self.bot, self.recording)
            self.continuous.start()
            self.task_status, self.message = "running", ""
            self.log("连续任务已开始，正在核对画面与场次")
        except (OSError, ValueError, RuntimeError) as exc:
            self.stop_and_save(f"连续任务启动失败：{exc}", outcome="failed")

    def set_bot_mode(self, mode: str) -> None:
        """程序化单局控制接口，供隔离模拟和自检调用，不在日常界面创建模式控件。

        执行必须具有实时视频快照；缺少模型或会话时抛出 ValueError，不发送输入。
        """
        if not self.bot or not hasattr(self.recording, "bot_snapshot"):
            raise ValueError("单局执行需要已加载模型和实时视频会话")
        self.bot.attach(self.recording)
        if mode in ("step", "auto"):
            self.input_gate.allow()
        self.bot.set_mode(mode)

    def stop_bot(self, reason: str = "用户停止连续任务") -> None:
        """立即撤销后续输入并取消启动意图，不等待 USB 或模型；供控制器与模拟自检调用。"""
        self.input_gate.suspend()
        self._requested_action = self._startup_action = ""
        if self.continuous:
            self.continuous.request_stop(reason)
        if self.bot:
            self.bot.set_mode("stop")

    def stop_and_save(self, reason: str = "用户停止连续任务", *, outcome: str = "stopped") -> None:
        """立即减权并异步保存；准备途中停止仍接收、关闭迟到的录像，不丢失资源。

        outcome 由控制器设置为 stopped/completed/failed；按钮信号的布尔值使用默认原因。
        可重复调用，不创建第二个收尾任务；已发出的在途单次命令仍按原账本记录。
        """
        if isinstance(reason, bool):
            reason = "用户停止连续任务"
        if self._closing:
            return
        active = bool(
            self.settings_locked or self.continuous and self.continuous.status in ("running", "paused")
        )
        if self.continuous and not self._stop_requested:
            self._last_progress = self.continuous.progress()
        self.stop_bot(reason)
        self.pause_preview()
        self._refresh_requested = False
        if not active and not self._stop_requested:
            self.render_dashboard()
            return
        if not self._stop_requested:
            self._finish_status = outcome
            self.message = reason
            self.log(reason + "；正在保存录像")
        self._stop_requested = True
        self.task_status = "stopping"
        self._dispatch_pending()
        self.render_dashboard()

    def _begin_save(self) -> None:
        """由 GUI 线程提交一次后台收尾；先关闭连续账本，再停止视频并关闭会话。"""
        pipeline, continuous = self.recording, self.continuous
        self._record_stopping = True

        def save():
            try:
                if continuous:
                    continuous.close()
            finally:
                try:
                    pipeline.stop()
                finally:
                    pipeline.session.close()
            return pipeline.session.folder

        self._submit("record_stop", save)

    def _finish_save(self) -> None:
        """GUI 线程结束保存状态；保留摘要，模型驻留，并按偏好恢复只读预览。"""
        self._stop_requested = self._record_stopping = False
        self._requested_action = ""
        self.task_status = self._finish_status
        if self._saved_folder:
            self.log("录像已保存，可打开本次回放")
        if self.auto_preview and not self._exit_requested:
            self.start_preview()

    def resume_continuous(self) -> None:
        """显式恢复既有任务，可信状态仍由运行时校验；被拒绝时直接展示中文原因。"""
        try:
            if not self.continuous or self._stop_requested:
                raise ValueError("没有可恢复的连续任务")
            self.input_gate.allow()
            self.continuous.resume()
            self.message = self._resume_error = ""
        except ValueError as exc:
            self.input_gate.suspend()
            self.message = self._resume_error = f"恢复被拒绝：{exc}"
            self.log(self.message)
        self.render_dashboard()

    def _done(self, generation: int, kind: str, value, error: str) -> None:
        """在 GUI 线程应用后台结果；录像结果无论代号是否过期都必须回收，截图仅接受当前代号。"""
        if kind == "capture" and (generation != self.generation or self._exit_requested):
            return
        if error:
            self.log(error)
            self.message = error
            if kind in ("record_start", "record_stop"):
                self.recording = None
                self._requested_action = ""
                self._stop_requested = self._record_stopping = False
                self.task_status = "failed"
                self._saved_folder = None
            elif kind == "capture":
                self.pause_preview()
                self.message = f"读取画面失败：{error}；请检查连接并在设置中刷新设备"
            return
        if kind == "devices":
            self.available_devices = list(value)
            previous = self.selected_device.serial if self.selected_device else ""
            authorized = [d for d in value if d.state == "device"]
            selected = next((d for d in value if d.serial == previous), None)
            selected = selected or (authorized[0] if len(authorized) == 1 else None)
            self.select_device(selected.serial if selected else "")
            self.log(
                f"发现 {len(value)} 台设备"
                + ("，请在设置中选择手机" if len(authorized) > 1 and not selected else "")
            )
        elif kind == "capture":
            self.frame = value
            self.canvas.set_frame(value)
            self._capture_count += 1
            self._capture_durations.append(value.capture_ms)
        elif kind == "record_start":
            self.recording = value
            self._video_epoch, self._video_identity = None, ""
            self.running = False
            self.log("视频通道已准备，正在接收画面")
            if not self._stop_requested and not self._exit_requested:
                QTimer.singleShot(0, self._launch_continuous)
        elif kind == "record_stop":
            self._saved_folder = Path(value)
            self.recording = None
            self.continuous = None
            if self.bot:
                self.bot.attach(None)
            self._finish_save()

    def _maybe_startup(self) -> None:
        """GUI 定时器消费一次启动参数；显式录像／连续优先于默认预览，不跨停止或重连恢复。"""
        if not self._startup_action or self.busy or self._exit_requested or not self.selected_device:
            return
        if self.selected_device.state != "device":
            return
        action = self._startup_action
        if action == "continuous" and (not self.bot or self.bot.model_status().state != "ready"):
            return
        self._startup_action = ""
        if action == "preview":
            self.start_preview()
        elif action == "record":
            self.start_recording()
        else:
            self.start_continuous()

    def _tick(self) -> None:
        """GUI 心跳只读轮询后台状态；显示失败先减权并暂停，不能越过 Qt 回调边界继续输入。"""
        if self._closing:
            return
        now = monotonic()
        self._max_heartbeat_gap = max(self._max_heartbeat_gap, now - self._heartbeat_time)
        self._heartbeat_time = now
        self._heartbeats += 1
        if self.continuous:
            thread = getattr(self.continuous, "thread", None)
            if not thread or not thread.is_alive():
                self.continuous.poll_stop_request()
            if self.continuous.close_window_requested:
                self.close()
                return
            if not self._stop_requested:
                progress = self.continuous.progress()
                self._last_progress = progress
                if progress.status == "paused" and progress.reason != self._last_pause_reason:
                    self._last_pause_reason = progress.reason
                    self.log("任务已暂停：" + progress.reason)
                if progress.status == "completed":
                    self.stop_and_save(progress.reason or "连续任务已完成", outcome="completed")
                elif self.continuous.stop_recording_requested:
                    self.stop_and_save("命令行请求停止连续任务")
        if self.bot:
            model = self.bot.model_status()
            if model != self._observed_model:
                self._observed_model = model
                if model.state == "failed":
                    self.log("WP 模型加载失败：" + model.error)
                elif model.state == "ready":
                    self.log("WP 模型已就绪")
            self.bot.attach(self.recording if hasattr(self.recording, "bot_snapshot") else None)
        if self.recording and not self._record_stopping:
            try:
                self._poll_video()
            except (BufferError, ValueError, RuntimeError) as exc:
                message = f"视频画面刷新失败，输入已停止：{exc}"
                self.input_gate.revoke()
                if self.continuous:
                    self.continuous.pause(message)
                elif self.bot:
                    self.bot.set_mode("stop")
                self.message = message
                if message != getattr(self, "_last_video_error", ""):
                    self.log(message)
                self._last_video_error = message
                self.render_dashboard()
                return
        self.input_gate.heartbeat()
        self._maybe_startup()
        self._dispatch_pending()
        if self.running and not self.recording and not self.busy and now >= self._next_capture:
            self._next_capture = now + 1
            self._submit("capture", self.client.capture)
        self.render_dashboard()

    def _poll_video(self) -> None:
        """仅呈现最新视频帧与独立的可信局面；画布保持只读，不叠加异帧识别框。"""
        snapshot = self.recording.poll()
        epoch = snapshot.get("codec_epoch", 0)
        if epoch != self._video_epoch:
            self._video_epoch, self._video_identity = epoch, ""
        for message in snapshot.get("messages", []):
            self.log(message)
        frame = snapshot.get("frame")
        if frame and frame.frame_id != self._video_identity and snapshot.get("allowed"):
            self.frame, self._video_identity = frame, frame.frame_id
            self.canvas.set_frame(frame)
        state = snapshot.get("state")
        if state:
            self._table_state = asdict(state) if not isinstance(state, dict) else dict(state)
        if snapshot.get("stopped") and not self._stop_requested:
            self.stop_and_save("视频已中断，已保留片段与缺口，请检查连接", outcome="failed")

    def dashboard_state(self) -> DashboardState:
        """在 GUI 线程复制展示数据，不解析状态文案；未知角色、行动与牌数不作推测。"""
        model = self.bot.model_status() if self.bot else ModelStatus("unloaded")
        device = self.selected_device
        authorized = [d for d in self.available_devices if d.state == "device"]
        unauthorized = any(d.state == "unauthorized" for d in self.available_devices)
        device_text = (
            f"{device.model or device.serial} · {DEVICE_STATES.get(device.state, '不可用')}"
            if device
            else "待选择设备"
            if len(authorized) > 1
            else "设备待授权"
            if unauthorized
            else "设备离线"
            if self.available_devices
            else "未连接设备"
        )
        status, notice = self.task_status, self.message
        progress = self._last_progress
        if self.continuous and not self._stop_requested and status not in ("stopped", "completed", "failed"):
            progress = self.continuous.progress()
            status = progress.status if progress.status != "idle" else status
            notice = (self._resume_error or progress.reason) if progress.status == "paused" else self.message
        settings = self.store.profile.continuous_settings
        scene = SCENE_NAMES.get(progress.scene, "等待确认画面") if progress else "等待开始"
        total = progress.total if progress else settings["max_games"]
        elapsed = (
            progress.elapsed
            if progress
            else (max(0, monotonic() - self._task_started) if self.recording else 0)
        )
        if self.recording and not self.continuous:
            scene = "仅记录画面"
        if status == "preparing":
            scene = "准备视频与任务"
        if status == "stopping":
            scene = "正在保存录像"
        state = self._table_state
        role, actor, remaining, action = "—", "—", ("—", "—", "—"), "等待新的牌局"
        playing = state.get("phase") == "playing" and not state.get("ended")
        if playing and (not progress or progress.scene == "playing"):
            landlord, order = state.get("landlord"), state.get("seat_order", [])
            if landlord == "self":
                role = "地主"
            elif landlord and set(order) == {"self", "left", "right"}:
                role = "地主下家" if order[(order.index(landlord) + 1) % 3] == "self" else "地主上家"
            actor = SEAT_NAMES.get(state.get("current_actor"), "待确认")
            remaining = tuple(
                str(state.get("remaining", {}).get(s))
                if state.get("remaining", {}).get(s) is not None
                else "—"
                for s in ("self", "left", "right")
            )
            suggestion = self.bot.suggestion if self.bot else None
            if (
                suggestion
                and suggestion.game_id == state.get("game_id")
                and suggestion.turn == state.get("turn")
            ):
                cards = " ".join(RANK_LABELS.get(c, c) for c in suggestion.cards) or "不出"
                action = f"{'规则策略' if suggestion.strategy == 'rules' else 'WP 建议'}：{cards}"
            else:
                action = "等待确认当前行动" if actor == "自己" else "等待对手行动"
        elif state.get("phase") == "settlement" or state.get("ended"):
            action = "本局已结束，正在处理结算"
        elif state.get("phase") in ("dealing", "bidding", "doubling"):
            action = PHASE_NAMES[state["phase"]]
        if model.state == "failed":
            notice = "WP 模型不可用：" + model.error
        if not notice:
            if model.state != "ready":
                notice = "正在后台加载本地 WP 模型…" if model.state == "loading" else "等待 WP 模型加载"
            elif not device:
                notice = (
                    "发现多台已授权设备，请在设置中选择一台"
                    if len(authorized) > 1
                    else "请在手机上授权 USB 调试，然后在设置中刷新设备"
                    if unauthorized
                    else "设备已离线，请检查 USB 连接并在设置中刷新设备"
                    if self.available_devices
                    else "请连接手机；多台设备时在设置中选择"
                )
            elif device.state != "device":
                notice = "请在手机上授权 USB 调试，然后在设置中刷新设备"
            elif status == "idle":
                notice = "准备就绪，点击开始即可自动匹配并连续游戏"
            elif status == "running":
                notice = "自动匹配、叫抢与出牌；录像同步保存在本机"
        return DashboardState(
            status=status,
            model_state=model.state,
            model_error=model.error,
            device_text=device_text,
            device_ready=bool(device and device.state == "device"),
            scene=scene,
            completed=progress.completed if progress else 0,
            total=total,
            elapsed=elapsed,
            time_limit=progress.time_limit if progress else settings["time_limit"],
            notice=notice,
            role=role,
            actor=actor,
            remaining=remaining,
            action=action,
            saved=bool(self._saved_folder and not self.settings_locked),
        )

    def render_dashboard(self) -> None:
        """仅在 GUI 线程把只读快照映射到控件；渲染不创建任务，也不授予设备输入。"""
        state = self.dashboard_state()
        self.connection_label.setText(state.device_text)
        self.connection_label.setToolTip(state.device_text)
        self._tone(self.connection_label, "success" if state.device_ready else "")
        self.model_label.setText(
            {
                "unloaded": "WP · 未加载",
                "loading": "WP · 加载中",
                "ready": "WP · 已就绪",
                "failed": "WP · 加载失败",
            }[state.model_state]
        )
        self._tone(
            self.model_label,
            {"ready": "success", "failed": "error", "loading": "warning"}.get(state.model_state, ""),
        )
        self.model_label.setToolTip(state.model_error)
        self.task_badge.setText(TASK_NAMES.get(state.status, "等待开始"))
        if state.status == "idle" and state.device_ready and state.model_state == "ready":
            self.task_badge.setText("准备就绪")
        tone = {"running": "success", "completed": "success", "paused": "warning", "failed": "error"}.get(
            state.status, ""
        )
        self._tone(self.task_badge, tone)
        self.scene_label.setText(state.scene)
        self.notice_label.setText(
            state.notice if len(state.notice) <= 100 else state.notice[:100] + "…详见运行日志"
        )
        self.notice_label.setToolTip(state.notice)
        self._tone(self.notice_label, "error" if state.model_state == "failed" else tone)
        self.retry_button.setVisible(state.model_state == "failed")
        self.retry_button.setEnabled(
            not self.settings_locked and not (self.bot and self.bot.thread and self.bot.thread.is_alive())
        )
        self.action_button.setText("恢复任务" if state.status == "paused" else "开始连续游戏")
        self.action_button.setEnabled(
            not self._exit_requested
            and state.model_state == "ready"
            and state.device_ready
            and state.status in ("idle", "stopped", "completed", "failed", "paused")
            and not self._requested_action
            and not self._record_stopping
        )
        self.stop_button.setEnabled(self.settings_locked or state.status in ("running", "paused"))
        self.saved_replay_button.setVisible(state.saved)
        self.replay_button.setEnabled(not self.settings_locked and not self._exit_requested)
        self.games_label.setText(f"{state.completed} / {state.total} 局")
        self.progress_bar.setRange(0, state.total)
        self.progress_bar.setValue(state.completed)
        self.elapsed_label.setText(f"{duration_text(state.elapsed)} / {duration_text(state.time_limit)}")
        profile = self.store.profile
        policy = {"strict": "严格", "infer": "唯一推断", "balanced": "平衡"}[profile.continuous_policy]
        bidding = "模型叫抢" if profile.bid_settings["mode"] == "model" else "不叫不抢"
        self.policy_label.setText(f"{profile.flow_room}场次 · {policy}模式 · {bidding}")
        self.role_label.setText("身份　" + state.role)
        self.actor_label.setText("行动　" + state.actor)
        self.remaining_label.setText("剩牌　自己 {} · 左侧 {} · 右侧 {}".format(*state.remaining))
        self.bot_action_label.setText(state.action)
        self.preview_label.setText(
            f"{'任务录制' if self.recording else '只读预览'}　·　{self.frame.width} × {self.frame.height}"
            if self.frame
            else "等待设备画面"
        )

    @staticmethod
    def _tone(widget, tone: str) -> None:
        """更新 Qt 动态颜色属性；仅在属性变化时重抛光，避免每个心跳重复计算样式。"""
        if widget.property("tone") != tone:
            widget.setProperty("tone", tone)
            widget.style().unpolish(widget)
            widget.style().polish(widget)

    def apply_settings(self, profile: Profile, adb_path: str, serial: str) -> None:
        """GUI 线程校验并原子保存设置；失败还原内存，任务锁定时拒绝，不改变既有标定字段。"""
        if self.settings_locked:
            raise ValueError("任务期间设置只读，请先停止并保存")
        validate_bot_profile(profile)
        old_profile, old_adb = self.store.profile, self.store.adb_path
        self.store.profile, self.store.adb_path = profile, adb_path
        try:
            self.store.save()
        except (OSError, ValueError):
            self.store.profile, self.store.adb_path = old_profile, old_adb
            raise
        self.select_device(serial)
        if old_adb != adb_path:
            self.refresh_devices()
        self.log("设置已保存，将用于下一次任务")

    def open_settings(self) -> None:
        """打开必要设置对话框；任务运行时可查看，但设备和任务字段不可修改。"""
        from doudizhu_lab.ui.settings import SettingsDialog

        dialog = SettingsDialog(self)
        dialog.exec()

    def open_current_session(self) -> None:
        """录像完成后定位本次会话；没有保存结果时不打开空路径。"""
        if self._saved_folder:
            self.open_sessions(self._saved_folder)

    def open_sessions(self, folder: Path | None = None) -> None:
        """在无录像任务时打开离线回放；关闭后恢复只读预览，回放不持有手机控制器。"""
        if self.settings_locked:
            return
        if isinstance(folder, bool):
            folder = None
        was_running = self.running
        self.pause_preview()
        dialog = SessionDialog(self.store, self, initial_folder=folder)
        dialog.exec()
        if was_running:
            self.start_preview()

    def closeEvent(self, event) -> None:
        """关闭先减权，再等待后台录像收尾；加载中的模型也只清理本实例，不触碰其他进程。"""
        if self._closing:
            event.accept()
            return
        self._exit_requested = True
        self.stop_and_save("正在退出并保存录像")
        if self.busy or self.recording or self._record_stopping:
            event.ignore()
            self.close_timer.start(100)
            return
        self._closing = True
        if self.continuous:
            self.continuous.close()
        if self.bot:
            self.bot.close()
        self.timer.stop()
        self.task_timer.stop()
        self.close_timer.stop()
        self.pool.close()
        elapsed = max(0, monotonic() - self._started)
        metrics = {
            "frames": self._capture_count,
            "elapsed_seconds": elapsed,
            "average_fps": self._capture_count / elapsed if elapsed else 0,
            "average_capture_ms": sum(self._capture_durations) / max(1, len(self._capture_durations)),
            "heartbeats": self._heartbeats,
            "maximum_heartbeat_gap_seconds": self._max_heartbeat_gap,
            "pending_tasks": self.pool.activeThreadCount(),
            "worker_limit": self.pool.maxThreadCount(),
        }
        try:
            (self.store.root / "界面运行统计.json").write_text(
                json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except OSError:
            pass
        event.accept()
