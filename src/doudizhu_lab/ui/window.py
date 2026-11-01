"""中文桌面调试窗口；设备操作和识别在单任务工作线程中完成。"""

from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from threading import Event
from time import monotonic

from PySide6.QtCore import Qt, QTimer, Slot
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QLabel,
    QMainWindow,
    QTreeWidgetItem,
)

from doudizhu_lab.devices.adb import AdbClient, AdbError
from doudizhu_lab.domain.models import RANK_LABELS, RANKS, REGION_NAMES, Frame, NormalizedRect
from doudizhu_lab.domain.table import PHASE_NAMES, SEAT_NAMES
from doudizhu_lab.runtime.bot import BotRuntime, InputGate
from doudizhu_lab.runtime.recording import RecordingPipeline
from doudizhu_lab.runtime.worker import SerialWorker, Task
from doudizhu_lab.storage.config import Store, read_rgb
from doudizhu_lab.ui.components import build_main_window
from doudizhu_lab.ui.lifetime import ensure_gui_lifetime
from doudizhu_lab.ui.replay import SessionDialog
from doudizhu_lab.vision.cards import ink_mask, normalize_glyph
from doudizhu_lab.vision.factory import create_recognizer

DEVICE_STATES = {"device": "已授权", "unauthorized": "未授权", "offline": "离线", "no": "权限不足"}


class MainWindow(QMainWindow):
    """桌面交互入口。running、generation、busy 共同阻止暂停或切换后的旧任务产生点击。"""

    def __init__(self, store: Store | None = None, *, auto_refresh: bool = True):
        super().__init__()
        ensure_gui_lifetime()
        self.store = store or Store()
        self.recognizer = create_recognizer(
            self.store.template_dir, self.store.profile, model_root=self.store.root
        )
        self.client = AdbClient(self.store.adb_path)
        self.pool = SerialWorker()
        self.input_gate = InputGate()
        self.bot = None
        self.continuous = None
        self._continuous_requested = False
        self.busy = False
        self._closing = False
        self.running = False
        self.generation = 0
        self._cancel = Event()
        self._pending_tap = None
        self._pending_recognition = False
        self._kind = ""
        self.frame: Frame | None = None
        self.result = None
        self.recording = None
        self._video_epoch = None
        self._video_identity = ""
        self._observation_identity = ""
        self._record_stopping = False
        self._last_size = None
        self._direction_ready = 0
        self._next_capture = 0.0
        self._last_error = ""
        self._capture_count = 0
        self._capture_durations = []
        self._started = monotonic()
        self._heartbeats = 0
        self._max_heartbeat_gap = 0.0
        self._heartbeat_time = monotonic()
        self._build_ui()
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
        if auto_refresh:
            QTimer.singleShot(0, self.refresh_devices)

    def _build_ui(self) -> None:
        """在GUI线程装配组件；后台任务统一由窗口方法管理。"""
        build_main_window(self)

    def log(self, message: str) -> None:
        """同时更新窗口和本地日志；原始画面仅在用户导出时保存。"""
        text = f"{datetime.now():%H:%M:%S}  {message}"
        self.log_view.append(text)
        try:
            with (self.store.root / "运行日志.txt").open("a", encoding="utf-8") as output:
                output.write(text + "\n")
        except OSError:
            pass

    def _submit(self, kind: str, action) -> bool:
        if self.busy or self._closing:
            return False
        self.busy = True
        self._kind = kind
        self._task = Task(self.generation, kind, action)
        self._future = self.pool.start(self._task)
        self.task_timer.start(25)
        return True

    @Slot()
    def _poll_task(self) -> None:
        """由 GUI 事件循环收取后台结果；工作线程完全不参与 Qt 对象的创建、销毁和回调。"""
        if self._closing:
            return
        if not self._future.done():
            self.task_timer.start(25)
            return
        value, error = self._future.result()
        generation, kind = self._task.generation, self._task.kind
        self._task = self._future = None
        self.pool.release_completed()
        self._done(generation, kind, value, error)

    def refresh_devices(self) -> None:
        """异步刷新设备；忙碌时不排队，用户可在当前任务结束后再按一次。"""
        if self.recording:
            return
        if not self._submit("devices", self.client.list_devices):
            self.log("正在读取画面，请稍后刷新设备")

    def _device_changed(self) -> None:
        if self.recording:
            return
        self.pause()
        device = self.devices.currentData()
        self.client = AdbClient(self.store.adb_path, device.serial if device else "")
        self.frame, self.result, self._last_size = None, None, None
        self._direction_ready = 0
        self.canvas.clear()
        self.result_tree.clear()
        self.connection_label.setText(
            f"{device.model or device.serial} · {DEVICE_STATES.get(device.state, '不可用')}"
            if device
            else "尚未连接"
        )

    def pause(self) -> None:
        """递增会话编号，使在途截图/识别结果失效；保留已显示帧供人工标定。"""
        self.stop_bot()
        self.running = False
        self._cancel.set()
        self._cancel = Event()
        self._pending_tap = None
        self.generation += 1
        self.canvas.control_enabled = False
        self.tap_button.setEnabled(False)
        self.start_button.setText("开始预览")

    def toggle_preview(self) -> None:
        if self.recording:
            self.log("正在整局记录，请先停止记录再切换截图预览")
            return
        if self.running:
            self.pause()
            self.log("预览已暂停，手机点击已停用")
            return
        device = self.devices.currentData()
        if device is None or device.state != "device":
            self.log("请选择已授权的在线设备；连接或授权后点击“刷新设备”")
            return
        if self.mode.currentData() in ("region", "template"):
            self.mode.setCurrentIndex(0)
        self.generation += 1
        self.running = True
        self._direction_ready = 0
        self._next_capture = 0
        self.start_button.setText("暂停 / 冻结")
        self.log("开始预览；截图串行执行，不积压旧帧")

    def _tick(self) -> None:
        now = monotonic()
        if self.continuous:
            self.continuous.poll_stop_request()
            if self.continuous.close_window_requested:
                self.close()
                return
            self.continuous_label.setText(self.continuous.description)
            self.continuous_resume.setEnabled(self.continuous.status == "paused")
            self.continuous_start.setEnabled(self.continuous.status not in ("running", "paused"))
            if self.continuous.status == "completed" and self.recording and not self._record_stopping:
                self.toggle_recording()
            elif self.continuous.stop_recording_requested and self.recording and not self._record_stopping:
                self.toggle_recording()
        if self.bot:
            self.bot.attach(self.recording if hasattr(self.recording, "bot_snapshot") else None)
            detail = ""
            if self.recording and hasattr(self.recording, "bot_snapshot"):
                current = self.recording.bot_snapshot()
                if current:
                    selected = (
                        " ".join(RANK_LABELS.get(c.rank, c.rank) for c in current.hand if c.selected) or "无"
                    )
                    if any(c.selected is None for c in current.hand):
                        selected += "（有未知）"
                    role = self.bot.suggestion.role if self.bot.suggestion else ""
                    role_name = {
                        "landlord": "地主",
                        "landlord_up": "地主上家",
                        "landlord_down": "地主下家",
                    }.get(role, "待确认")
                    detail = f"\n身份：{role_name} · 选中：{selected} · 执行：{self.bot.executor.stage}"
                    if (
                        self.bot.mode == "stop" or current.state["ended"]
                    ) and self.bot_mode.currentIndex() != 0:
                        self.bot_mode.blockSignals(True)
                        self.bot_mode.setCurrentIndex(0)
                        self.bot_mode.blockSignals(False)
            self.bot_label.setText(self.bot.status + detail)
        self._max_heartbeat_gap = max(self._max_heartbeat_gap, now - self._heartbeat_time)
        self._heartbeat_time = now
        self._heartbeats += 1
        if self.recording:
            try:
                self._poll_video()
            except (BufferError, ValueError, RuntimeError) as exc:
                # 显示异常不能越过 Qt 定时器边界，更不能继续授予输入心跳。
                # 采集线程仍保存录像；下一帧可重试显示，机器人须由用户明确恢复。
                self.stop_bot()
                self._cancel.set()
                self._pending_tap = None
                self._direction_ready = 0
                self._video_identity = ""
                self.canvas.control_enabled = False
                self.tap_button.setEnabled(False)
                message = f"视频画面刷新失败，输入已停止：{exc}"
                self.status_label.setText(message)
                if message != getattr(self, "_last_video_error", ""):
                    self.log(message)
                self._last_video_error = message
                return
        self.input_gate.heartbeat()
        enabled = bool(
            self.running
            and self._kind != "tap"
            and self._pending_tap is None
            and self.frame
            and self.frame.serial == self.client.serial
            and self.frame.age < 3
            and self._direction_ready >= 2
            and self.mode.currentData() == "control"
            and (self.recording is None or (self.recording.allowed and not self._record_stopping))
        )
        self.canvas.control_enabled = enabled
        self.tap_button.setEnabled(enabled)
        self.export_button.setEnabled(self.frame is not None)
        if not self.busy and self._pending_tap is not None:
            x, y, frame = self._pending_tap
            self._pending_tap = None
            self._send_tap(x, y, frame)
        elif not self.busy and self._pending_recognition:
            self._pending_recognition = False
            self.recognize_current()
        elif self.running and not self.recording and not self.busy and now >= self._next_capture:
            self._next_capture = now + 1.0
            client, profile = self.client, deepcopy(self.store.profile)

            def capture():
                frame = client.capture()
                return frame, self.recognizer.recognize(frame, profile, stabilize=True)

            self._submit("capture", capture)

    @Slot(int, str, object, str)
    def _done(self, generation: int, kind: str, value, error: str) -> None:
        self.busy = False
        self._kind = ""
        if self._closing or generation != self.generation:
            return
        if error:
            if kind == "record_stop":
                self.recording = None
                self._record_stopping = False
                self.pause()
                self._lock_recording(False)
                self.game_status.setText("收尾失败，已保存的录像和疑点可在会话中核对")
            if kind == "record_start":
                self._continuous_requested = False
                self._lock_recording(False)
                self.game_status.setText("整局记录不可用：" + error)
            if error != self._last_error:
                self.log(error)
                self._last_error = error
            self.connection_label.setText("操作失败 · 请检查设备")
            self.canvas.control_enabled = False
            if kind in ("capture", "tap"):
                self.pause()
            return
        self._last_error = ""
        if kind == "record_start":
            self.recording = value
            self._video_epoch = None
            self.running = True
            self._direction_ready = 0
            self.record_button.setText("停止记录")
            self.record_button.setEnabled(True)
            self.log(f"开始连续记录：{value.session.folder}；校准与模板已保存快照")
            if self._continuous_requested:
                QTimer.singleShot(0, self._launch_continuous)
        elif kind == "record_stop":
            self.recording = None
            self._record_stopping = False
            self.pause()
            self._lock_recording(False)
            self.game_status.setText("记录已保存，可打开会话进行回放纠错")
            self.log(f"录像收尾完成：{value}")
            if self.continuous and self.continuous.store:
                from doudizhu_lab.storage.runs import evaluate_run

                report = evaluate_run(self.continuous.store.folder)
                self.log(
                    f"连续任务报告已更新：完成{report['completed_games']}局；真机检查{'通过' if report['live_passed'] else '未通过'}"
                )
        elif kind == "devices":
            previous = self.client.serial
            self.devices.blockSignals(True)
            self.devices.clear()
            for device in value:
                self.devices.addItem(
                    f"{device.model or device.serial} · {DEVICE_STATES.get(device.state, '不可用')}", device
                )
            selection = next(
                (i for i, d in enumerate(value) if d.serial == previous), 0 if len(value) == 1 else -1
            )
            self.devices.setCurrentIndex(selection)
            self.devices.blockSignals(False)
            self._device_changed()
            self.log(f"发现 {len(value)} 台设备" + ("，请选择目标手机" if len(value) > 1 else ""))
        elif kind in ("capture", "tap", "recognize"):
            frame, result = value
            size = (frame.width, frame.height)
            self._direction_ready = self._direction_ready + 1 if size == self._last_size else 1
            if self._last_size and size != self._last_size:
                self.log("画面方向或尺寸发生变化，等待下一帧确认后恢复点击")
            self._last_size = size
            self.frame, self.result = frame, result
            self.canvas.set_frame(frame, result)
            self.x_input.setMaximum(frame.width - 1)
            self.y_input.setMaximum(frame.height - 1)
            self.render_results()
            if kind != "recognize":
                self._capture_count += 1
                self._capture_durations.append(frame.capture_ms)
            transport = {"raw": "原始帧", "png": "PNG", "file": "本地文件"}.get(
                frame.transport, frame.transport
            )
            self.status_label.setText(
                f"{frame.width}×{frame.height}  ·  截图 {frame.capture_ms:.0f} ms（{transport}）  ·  识别 {result.elapsed_ms:.0f} ms  ·  已读取 {self._capture_count} 帧"
            )
            self.connection_label.setText("本地截图" if frame.serial == "offline" else "USB 已连接")
            if kind == "tap":
                self.log("点击命令执行成功；已显示操作后画面，请核对游戏实际结果")

    def render_results(self) -> None:
        self.result_tree.clear()
        if self.result is None:
            return
        for region in self.result.regions:
            parent = QTreeWidgetItem([REGION_NAMES[region.region], region.status.value])
            self.result_tree.addTopLevelItem(parent)
            for card in region.cards:
                label = RANK_LABELS.get(card.rank, "无法确认")
                child = QTreeWidgetItem([label, f"{card.score:.3f}"])
                child.setToolTip(0, f"原图位置：{card.box.x}, {card.box.y}\n{card.reason}")
                parent.addChild(child)
            if region.message:
                parent.setToolTip(1, region.message)
            parent.setExpanded(True)

    def _mode_changed(self) -> None:
        mode = self.mode.currentData()
        if mode != "view":
            self.stop_bot()
        if self.recording and mode in ("region", "template"):
            self.mode.setCurrentIndex(0)
            self.log("录制期间配置只读；停止记录后可标定或采集模板")
            return
        self.canvas.mode = mode
        if mode != "control":
            self._cancel.set()
            self._cancel = Event()
            self._pending_tap = None
            self.canvas.control_enabled = False
            self.tap_button.setEnabled(False)
        if mode in ("region", "template"):
            self.pause()
        messages = {
            "view": "查看模式只显示画面，不向手机发送操作。",
            "control": "在新鲜画面上单击，或输入原图坐标。每次操作后会重新获取画面。",
            "region": "已冻结画面。选择目标区域，在画面上拖拽矩形以保存识别范围。",
            "template": "已冻结画面。普通牌只框住牌点（10 框住两位）；大小王框住完整竖排 JOKER。",
        }
        self.mode_help.setText(messages[mode])
        self.canvas.update()

    def request_tap(self, x: int, y: int) -> None:
        """只接受用户在控制模式的单次点击；预检新截图尺寸，并在执行后再截图。"""
        if not (
            self.canvas.control_enabled
            and self.frame
            and self.running
            and self.mode.currentData() == "control"
        ):
            self.log("当前不能点击：请启用点击控制、开始预览并等待新鲜画面")
            return
        frame = self.frame
        self.canvas.control_enabled = False
        self.tap_button.setEnabled(False)
        if self.busy:
            self._pending_tap = (x, y, frame)
            self.log("等待当前截图完成后执行这一次点击；重复点击已停用")
        else:
            self._send_tap(x, y, frame)

    def _send_tap(self, x: int, y: int, frame: Frame) -> None:
        """发送一个人工点击；暂停和切换设备会取消预检中尚未发送的输入命令。"""
        if not self.running or self.mode.currentData() != "control":
            return
        client, profile, cancel = self.client, deepcopy(self.store.profile), self._cancel
        self.stop_bot()
        token = self.input_gate.grant("manual")

        def tap():
            before = client.capture()
            if (before.width, before.height) != (frame.width, frame.height):
                raise AdbError("手机方向已变化，本次点击未发送，请重新开始预览")
            if cancel.is_set():
                raise AdbError("操作已取消，未发送点击")
            self.input_gate.tap("manual", token, client, x, y, frame, cancelled=cancel.is_set)
            after = client.capture()
            return after, self.recognizer.recognize(after, profile, stabilize=True)

        self.log(f"准备点击原图坐标 ({x}, {y})")
        self._submit("tap", tap)

    def selection_done(self, area: NormalizedRect) -> None:
        if self.frame is None or self.running or self.recording:
            return
        region = self.region.currentData()
        if self.mode.currentData() == "region":
            self.store.profile.regions[region] = area
            scale = self.frame.height / self.store.profile.base_height
            self.store.profile.glyph_heights = {
                key: height * scale for key, height in self.store.profile.glyph_heights.items()
            }
            self.store.profile.base_width, self.store.profile.base_height = (
                self.frame.width,
                self.frame.height,
            )
            try:
                self.store.save()
            except OSError as exc:
                self.log(f"区域配置保存失败：{exc}")
                return
            self.canvas.profile = self.store.profile
            self.canvas.update()
            self.log(f"已保存区域：{REGION_NAMES[region]}")
            self.recognize_current()
        elif self.mode.currentData() == "template":
            crop = area.pixels(self.frame.width, self.frame.height).crop(self.frame.rgb)
            if normalize_glyph(ink_mask(crop)) is None:
                self.log("裁剪内没有有效牌点，请重新框选")
                return
            dialog = QDialog(self)
            dialog.setWindowTitle("确认牌点模板")
            form = QFormLayout(dialog)
            preview = QLabel()
            preview_rgb = crop.copy()
            qimage = QImage(
                preview_rgb.data, crop.shape[1], crop.shape[0], crop.shape[1] * 3, QImage.Format.Format_RGB888
            ).copy()
            preview.setPixmap(
                QPixmap.fromImage(qimage).scaled(
                    150, 180, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
                )
            )
            form.addRow("裁剪预览", preview)
            label = QComboBox()
            for rank in RANKS:
                label.addItem(RANK_LABELS[rank], rank)
            form.addRow("牌点标签", label)
            group = region if region in ("hand", "bottom") else "table"
            form.addRow(QLabel(f"分组：{REGION_NAMES[region]}\n请确认框内只包含目标牌点字形。"))
            buttons = QDialogButtonBox(
                QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
            )
            buttons.button(QDialogButtonBox.StandardButton.Save).setText("保存模板")
            buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("取消")
            buttons.accepted.connect(dialog.accept)
            buttons.rejected.connect(dialog.reject)
            form.addRow(buttons)
            if dialog.exec() == QDialog.DialogCode.Accepted:
                try:
                    self.store.add_template(crop, label.currentData(), group, self.frame.captured_at)
                    self.log(f"已添加 {REGION_NAMES[region]} / {label.currentText()} 模板")
                    self.reload_templates()
                except (OSError, ValueError) as exc:
                    self.log(str(exc))

    def update_coverage(self) -> None:
        coverage = self.recognizer.coverage()
        self.coverage_label.setText(
            (
                "手牌：免训练OCR（试用）"
                if self.store.profile.hand_classifier == "pretrained-ocr"
                else f"手牌模板 {len(coverage['hand'])}/15"
            )
            + f" · 底牌模板 {len(coverage['bottom'])}/15 · 出牌模板 {len(coverage['table'])}/15"
        )
        self.coverage_label.setToolTip(
            "\n".join(
                f"{name}缺少：" + " ".join(RANK_LABELS[r] for r in RANKS if r not in coverage[key])
                for key, name in (("hand", "手牌"), ("bottom", "底牌"), ("table", "出牌"))
                if key != "hand" or self.store.profile.hand_classifier == "template"
            )
            + (
                "\n手牌使用预训练OCR；仍依赖候选分割和区域标定，尚未完成实时整局验收。"
                if self.store.profile.hand_classifier == "pretrained-ocr"
                else ""
            )
        )

    def reload_templates(self) -> None:
        if self.recording:
            return
        if self.busy:
            self.log("请等当前任务结束后再重新载入模板")
            return
        self.recognizer.reload()
        self.update_coverage()
        for error in self.recognizer.load_errors:
            self.log(error)
        self.recognize_current()

    def recognize_current(self) -> None:
        if self.busy:
            self._pending_recognition = True
            return
        if self.frame and not self.busy:
            frame, profile = self.frame, deepcopy(self.store.profile)
            self._submit("recognize", lambda: (frame, self.recognizer.recognize(frame, profile)))

    def import_image(self) -> None:
        self.pause()
        path, _ = QFileDialog.getOpenFileName(
            self, "导入本地截图", str(self.store.root), "图片 (*.png *.jpg *.jpeg)"
        )
        if path:
            try:
                self.frame = Frame(
                    "offline",
                    read_rgb(Path(path)),
                    monotonic(),
                    datetime.now(timezone.utc).isoformat(),
                    0,
                    "file",
                )
                self.result = None
                self.canvas.set_frame(self.frame)
                self.recognize_current()
                self.log(f"已导入 {Path(path).name}；本地截图不允许控制手机")
            except (OSError, ValueError) as exc:
                self.log(str(exc))

    def export_frame(self) -> None:
        if self.frame:
            try:
                folder = self.store.export(self.frame, self.result)
                self.log(f"已导出截图及识别结果：{folder}")
            except (OSError, ValueError) as exc:
                self.log(f"导出失败：{exc}")

    def choose_adb(self) -> None:
        if self.recording:
            return
        self.pause()
        path, _ = QFileDialog.getOpenFileName(
            self, "选择 adb.exe", str(Path(self.store.adb_path).parent), "ADB (adb.exe)"
        )
        if path:
            self.store.adb_path = path
            try:
                self.store.save()
            except OSError as exc:
                self.log(f"ADB 设置保存失败：{exc}")
                return
            self._device_changed()
            self.log("ADB 路径已保存，请刷新设备")

    def closeEvent(self, event) -> None:
        # 关闭只能清理一次；Qt 事件队列或测试框架可能再次调用 close()。
        # 等待设备线程结束之后，迟到的 done 信号也不得访问已关闭的子控件。
        self.stop_bot()
        if self._closing:
            event.accept()
            return
        if self.recording and not self._record_stopping:
            self.toggle_recording()
            event.ignore()
            self.close_timer.start(250)
            return
        if self._record_stopping or self._kind == "record_start":
            event.ignore()
            self.close_timer.start(250)
            return
        self._closing = True
        if self.continuous:
            self.continuous.close()
        if self.bot:
            self.bot.close()
        self.pause()
        self.timer.stop()
        self.task_timer.stop()
        self.close_timer.stop()
        self.pool.waitForDone(30000)
        self.pool.close()
        self._task = self._future = None
        # Windows 的 monotonic 可能多个调用落在同一时钟刻度。快速开关窗口
        # 时 elapsed 为零；不能从 Qt 虚方法抛出除零异常（旧版 Shiboken 会二次崩溃）。
        elapsed = max(0.0, monotonic() - self._started)
        metrics = {
            "frames": self._capture_count,
            "elapsed_seconds": elapsed,
            "average_fps": self._capture_count / elapsed if elapsed else 0.0,
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

    def _lock_recording(self, active: bool) -> None:
        """同一会话的配置不可变；停止时恢复入口，不影响只读显示与人工点击模式。"""
        for widget in (
            self.acceptance,
            self.devices,
            self.refresh_button,
            self.start_button,
            self.import_button,
            self.reload_button,
            self.settings_button,
            self.replay_button,
            self.continuous_policy,
            self.bidding_mode,
        ):
            widget.setEnabled(not active)
        for index in (2, 3):
            self.mode.model().item(index).setEnabled(not active)
        self.record_button.setText("停止记录" if active else "开始记录")
        self.record_button.setEnabled(True)
        self.timeline.setVisible(active)

    def toggle_recording(self) -> None:
        """一次启动录屏和整局跟踪；启动/停止均在后台收尾，不在 GUI 线程等待 ADB。"""
        if self.busy or self._record_stopping:
            self.log("当前任务正在收尾，请稍后操作")
            return
        if self.recording:
            self.stop_bot()
            pipeline = self.recording
            self._record_stopping = True
            self.record_button.setEnabled(False)
            self.canvas.control_enabled = False
            self.game_status.setText("正在停止记录并保存录像尾部…")

            def stop():
                pipeline.stop()
                folder = pipeline.session.folder
                pipeline.session.close()
                return folder

            self._submit("record_stop", stop)
            return
        device = self.devices.currentData()
        if not device or device.state != "device":
            self.log("请选择已授权的在线手机")
            return
        self.pause()
        self.mode.setCurrentIndex(0)
        self._lock_recording(True)
        self.record_button.setEnabled(False)
        self.game_status.setText("正在启动视频通道…")
        self.timeline.clear()
        acceptance = self.acceptance.isChecked()

        def start():
            pipeline = RecordingPipeline(self.store, device.serial, acceptance=acceptance)
            pipeline.start()
            return pipeline

        self._submit("record_start", start)

    def load_bot(self):
        """只启动本地模型进程；没有实时视频会话时保持没有设备输入能力。"""
        if self.bot is None:
            self.bot = BotRuntime(self.store.root, self.input_gate)
        self.bot.start()

    def calibrate_bot(self):
        """在已导入或冻结的画面上标定；录制期间模板和区域均只读。"""
        if self.recording or self.frame is None:
            self.log("请先停止记录，再导入或冻结一张有手牌和按钮的画面")
            return
        self.pause()
        from doudizhu_lab.ui.calibration import BotCalibrationDialog

        dialog = BotCalibrationDialog(self.store, self.frame, self)
        dialog.exec()

    def change_bot_mode(self, *_):
        """执行模式要求完整实时会话；回放和截图预览只能使用离线建议评测。"""
        mode = self.bot_mode.currentData()
        if self.continuous and self.continuous.status in ("running", "paused"):
            self.continuous.stop("用户切换单局模式")
        if self.bot is None:
            self.log("请先加载 WP 模型")
            self.bot_mode.blockSignals(True)
            self.bot_mode.setCurrentIndex(0)
            self.bot_mode.blockSignals(False)
            return
        if mode != "suggest" and not hasattr(self.recording, "bot_snapshot"):
            self.log("执行机器人需要实时视频会话；当前没有发送任何输入")
            self.bot_mode.setCurrentIndex(0)
            return
        if mode != "suggest":
            self.mode.setCurrentIndex(0)
            self._cancel.set()
            self._cancel = Event()
            self._pending_tap = None
        self.bot.attach(self.recording if hasattr(self.recording, "bot_snapshot") else None)
        self.bot.set_mode(mode)

    def execute_bot_step(self):
        """仅建议时显式执行当前一手；暂停后需用户点击，不能因新帧到来自动恢复。"""
        if self.bot_mode.currentData() == "suggest":
            self.bot_mode.setCurrentIndex(1)
        else:
            self.change_bot_mode()

    def stop_bot(self, *_):
        """立即撤销输入授权，随后由后台线程收尾；不会等待模型推理完成。"""
        self.input_gate.revoke()
        self._continuous_requested = False
        if self.continuous:
            self.continuous.stop()
        if self.bot:
            self.bot.set_mode("stop")
        self.bot_mode.blockSignals(True)
        self.bot_mode.setCurrentIndex(0)
        self.bot_mode.blockSignals(False)

    def start_continuous(self):
        """一次点击准备录像、模型及流程；匹配只能由后台在稳定大厅观察上触发。"""
        from doudizhu_lab.vision.assets import model_path

        if self.busy or self._record_stopping:
            self.log("请等待当前任务收尾")
            return
        if self.continuous and self.continuous.status in ("running", "paused"):
            self.log("已有连续任务，暂停后请使用恢复或立即停止")
            return
        try:
            if self.input_gate.transport != "simulation":
                for key in ("ocr_det", "ocr_en", "ocr_zh"):
                    model_path(self.store.root, key)
            if self.continuous:
                self.continuous.close()
                self.continuous = None
            self.mode.setCurrentIndex(0)
            self.load_bot()
            if self.recording is None:
                self.toggle_recording()
                self._continuous_requested = self._kind == "record_start"
            else:
                self._launch_continuous()
        except (OSError, ValueError) as exc:
            self.log(f"连续任务未启动：{exc}")

    def change_continuous_policy(self):
        """仅保存严格／唯一推断／平衡的策略偏好；录制期间只读，不保存自动运行授权。"""
        if self.recording or self.busy:
            return
        self.store.profile.continuous_policy = self.continuous_policy.currentData()
        self.store.save()

    def change_bidding_mode(self):
        """仅在未录制时保存叫抢偏好；这个选项本身不启动录像、匹配或授予手机输入。"""
        if self.recording:
            return
        self.store.profile.bid_settings["mode"] = self.bidding_mode.currentData()
        self.store.save()

    def _launch_continuous(self):
        """视频启动成功后再创建任务；失败或被用户取消的启动不会留下自动输入授权。"""
        from doudizhu_lab.runtime.continuous import ContinuousRuntime

        self._continuous_requested = False
        if not self.recording or self._record_stopping:
            return
        self.continuous = ContinuousRuntime(self.store.root, self.input_gate, self.bot, self.recording)
        self.continuous.start()
        self.log(f"连续任务已准备：{self.continuous.store.folder}；等待模型及已确认的经典洗牌大厅")

    def resume_continuous(self):
        """明确恢复按钮；无法恢复的历史缺口保留，不自动重置可信状态。"""
        try:
            if not self.continuous:
                raise ValueError("没有可恢复的连续任务")
            self.continuous.resume()
        except ValueError as exc:
            self.log(f"恢复被拒绝：{exc}")

    def _poll_video(self) -> None:
        """只显示最新帧，识别结果用源帧标识关联；不能把上一帧的框叠到新画面。"""
        snapshot = self.recording.poll()
        self.candidate_tree.clear()
        for candidate in snapshot.get("pending_candidates", []):
            title = f"{SEAT_NAMES[candidate['seat']]} · {'出牌' if candidate['kind'] == 'play' else '不出'}"
            item = QTreeWidgetItem([title, candidate.get("reason", "等待证据")])
            item.setToolTip(1, candidate.get("reason", ""))
            self.candidate_tree.addTopLevelItem(item)
        self.candidate_tree.setVisible(bool(snapshot.get("pending_candidates")))
        epoch = snapshot.get("codec_epoch", 0)
        if epoch != self._video_epoch:
            # 180° 旋转或编码器重启可能保持宽高不变，仍须废弃旧点击关联。
            # 取消正在做截图预检的点击，等新会话至少两帧后再开放控制。
            self._video_epoch = epoch
            self._cancel.set()
            self._cancel = Event()
            self._pending_tap = None
            self._direction_ready = 0
            self._last_size = None
            self._video_identity = ""
            self.canvas.control_enabled = False
            self.tap_button.setEnabled(False)
        from doudizhu_lab.ui.replay import KINDS

        for event in snapshot["events"]:
            content = " ".join(event.cards) or event.data.get("message", "")
            self.timeline.addTopLevelItem(
                QTreeWidgetItem(
                    [
                        f"{KINDS.get(event.kind, event.kind)} · {event.pts_us / 1e6:.3f}",
                        f"{SEAT_NAMES.get(event.seat, '')} {content}",
                        "已确认" if event.confirmed else "待核对",
                    ]
                )
            )
            if self.timeline.topLevelItemCount() > 200:
                self.timeline.takeTopLevelItem(0)
        if snapshot["events"]:
            self.timeline.scrollToBottom()
        for message in snapshot["messages"]:
            self.log(message)
        frame = snapshot["frame"]
        if frame and frame.frame_id != self._video_identity and snapshot["allowed"]:
            self._video_identity = frame.frame_id
            size = frame.width, frame.height
            self._direction_ready = self._direction_ready + 1 if size == self._last_size else 1
            self._last_size = size
            self.frame = frame
            result = snapshot["recognition"]
            self.result = result if result and result.frame_time == frame.captured_at else None
            self.canvas.set_frame(frame, self.result)
            self.x_input.setMaximum(frame.width - 1)
            self.y_input.setMaximum(frame.height - 1)
        observation, state = snapshot["observation"], snapshot["state"]
        if observation and observation.frame_id != self._observation_identity:
            self._observation_identity = observation.frame_id
            # 右侧明确展示最近完成的识别；导出仅使用与显示帧一致的 self.result。
            current = self.result
            self.result = snapshot["recognition"]
            self.render_results()
            self.result = current
            state_text = f"{state.game_id or '尚未确认开局'} · {PHASE_NAMES.get(state.phase, '无法确认')} · 地主：{SEAT_NAMES.get(state.landlord, '无法确认')} · 已确认 {state.turn} 次行动"
            state_text += " · 剩牌 " + "/".join(
                str(state.remaining[s]) if state.remaining[s] is not None else "?"
                for s in ("self", "left", "right")
            )
            if state.pending:
                state_text += " · 有疑点，等待赛后核对"
            candidates = snapshot.get("pending_candidates", [])
            if candidates:
                state_text += f" · {len(candidates)} 个候选正在等待证据"
            self.game_status.setText(state_text)
        metrics = snapshot["metrics"]
        elapsed = max(0.01, monotonic() - self.recording._started)
        self.status_label.setText(
            f"视频解码 {metrics['decoded_frames'] / elapsed:.2f} FPS · 识别 {metrics['analyzed_frames'] / elapsed:.2f} FPS · 编码数据 {metrics['encoded_bytes'] / 1024**2:.1f} MiB · 写盘队列 {self.recording.write_queue.qsize()}"
        )
        if snapshot["stopped"] and not self._record_stopping and not self.busy:
            self.log("视频已中断，正在保留片段与缺口；整局记录不可用")
            self.toggle_recording()

    def open_sessions(self) -> None:
        if self.recording:
            return
        self.pause()
        dialog = SessionDialog(self.store, self)
        dialog.exec()
