"""日常运行设置；只编辑连接与任务偏好，沿用已有模板和按钮标定。"""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

from PySide6.QtCore import QTime, QTimer
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QTimeEdit,
    QVBoxLayout,
)

from doudizhu_lab.domain.settings import PLAY_MODEL_NAMES
from doudizhu_lab.ui.components import label
from doudizhu_lab.ui.theme import STYLE

if TYPE_CHECKING:
    from doudizhu_lab.ui.window import MainWindow


class SettingsDialog(QDialog):
    """GUI 线程中的偏好编辑器；应用前不改配置，运行期间只读，取消不发布任何改动。"""

    def __init__(self, window: "MainWindow") -> None:
        """从窗口复制配置创建表单；定时同步设备列表与锁状态，不执行模型或手机输入。"""
        super().__init__(window)
        self.window = window
        self._devices = None
        self.setWindowTitle("设置")
        self.setStyleSheet(STYLE)
        self.resize(540, 650)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(12)
        layout.addWidget(label("运行设置", "title"))
        self.lock_note = label("更改将在下一次任务中使用。", "muted", wrap=True)
        layout.addWidget(self.lock_note)
        self.connection = QGroupBox("设备连接")
        form = QFormLayout(self.connection)
        form.setSpacing(8)
        self.devices = QComboBox()
        self.devices.setMinimumWidth(260)
        row = QHBoxLayout()
        row.addWidget(self.devices, 1)
        self.refresh_button = QPushButton("刷新")
        self.refresh_button.clicked.connect(window.refresh_devices)
        row.addWidget(self.refresh_button)
        form.addRow("目标设备", row)
        self.adb_path = QLineEdit(window.store.adb_path)
        path_row = QHBoxLayout()
        path_row.addWidget(self.adb_path, 1)
        browse = QPushButton("选择")
        browse.clicked.connect(self.choose_adb)
        path_row.addWidget(browse)
        form.addRow("ADB 路径", path_row)
        layout.addWidget(self.connection)
        self.task = QGroupBox("连续游戏")
        form = QFormLayout(self.task)
        # 新增模型选择后仍控制最小高度，保证150%缩放下保存按钮位于可见区域。
        form.setSpacing(6)
        profile = window.store.profile
        self.play_model = QComboBox()
        for backend, name in PLAY_MODEL_NAMES.items():
            self.play_model.addItem("DouZero " + name, backend)
        self.play_model.setCurrentIndex(self.play_model.findData(profile.play_model))
        form.addRow("出牌模型", self.play_model)
        form.addRow(label("沿用当前页面，不限制玩法名称或场次等级。", "muted", wrap=True))
        self.policy = QComboBox()
        for text, value in (("严格模式", "strict"), ("唯一推断", "infer"), ("平衡模式", "balanced")):
            self.policy.addItem(text, value)
        self.policy.setCurrentIndex(self.policy.findData(profile.continuous_policy))
        form.addRow("连续策略", self.policy)
        self.bidding = QComboBox()
        self.bidding.addItem("固定不叫不抢", "disabled")
        self.bidding.addItem("模型自动叫抢", "model")
        self.bidding.setCurrentIndex(self.bidding.findData(profile.bid_settings["mode"]))
        form.addRow("叫抢策略", self.bidding)
        self.games = QSpinBox()
        self.games.setRange(1, 30)
        self.games.setSuffix(" 局")
        self.games.setValue(profile.continuous_settings["max_games"])
        form.addRow("最多局数", self.games)
        self.time_limit = QTimeEdit()
        self.time_limit.setDisplayFormat("HH:mm:ss")
        self.time_limit.setTimeRange(QTime(0, 0, 1), QTime(4, 0, 0))
        self.time_limit.setTime(QTime(0, 0).addSecs(profile.continuous_settings["time_limit"]))
        form.addRow("最长时间", self.time_limit)
        form.addRow(label("到达时限后打完当前局停止；叫抢始终不加倍。", "muted", wrap=True))
        layout.addWidget(self.task)
        self.error = label("", "notice", wrap=True)
        self.error.hide()
        layout.addWidget(self.error)
        layout.addStretch()
        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        self.buttons.button(QDialogButtonBox.StandardButton.Save).setText("保存设置")
        self.buttons.button(QDialogButtonBox.StandardButton.Save).setObjectName("primary")
        self.buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("关闭")
        self.buttons.accepted.connect(self.save)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.sync)
        self.timer.start(150)
        self.sync()

    def sync(self) -> None:
        """在 GUI 线程同步设备和会话锁；保留尚未保存的用户选择，不覆盖表单偏好。"""
        locked = self.window.settings_locked
        self.connection.setEnabled(not locked)
        self.task.setEnabled(not locked)
        self.buttons.button(QDialogButtonBox.StandardButton.Save).setEnabled(not locked)
        self.lock_note.setText("任务进行中，连接和任务设置只读。" if locked else "更改将在下一次任务中使用。")
        devices = tuple(self.window.available_devices)
        if devices != self._devices:
            selected = self.devices.currentData()
            selected = selected or (self.window.selected_device.serial if self.window.selected_device else "")
            self._devices = devices
            self.devices.clear()
            self.devices.addItem("请选择设备", "")
            for device in devices:
                state = {"device": "已授权", "unauthorized": "未授权", "offline": "离线"}.get(
                    device.state, "不可用"
                )
                self.devices.addItem(f"{device.model or device.serial} · {state}", device.serial)
            self.devices.setCurrentIndex(max(0, self.devices.findData(selected)))

    def choose_adb(self) -> None:
        """仅选择本地 adb.exe 路径，不立即保存或连接；文件对话框取消时保持原值。"""
        path, _ = QFileDialog.getOpenFileName(self, "选择 adb.exe", self.adb_path.text(), "ADB (adb.exe)")
        if path:
            self.adb_path.setText(path)

    def save(self) -> None:
        """校验并发布表单副本；保存失败直接显示中文原因，已有配置与标定保持原值。"""
        profile = deepcopy(self.window.store.profile)
        profile.play_model = self.play_model.currentData()
        profile.continuous_policy = self.policy.currentData()
        profile.bid_settings["mode"] = self.bidding.currentData()
        profile.continuous_settings = {
            "max_games": self.games.value(),
            "time_limit": QTime(0, 0).secsTo(self.time_limit.time()),
        }
        try:
            self.window.apply_settings(profile, self.adb_path.text().strip(), self.devices.currentData())
        except (OSError, ValueError) as exc:
            self.error.setText(str(exc))
            self.error.show()
            return
        self.accept()
