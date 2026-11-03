"""日常运行设置；只编辑连接与任务偏好，沿用已有模板和按钮标定。"""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

from PySide6.QtCore import Qt, QTimer
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
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from doudizhu_lab.domain.settings import (
    BID_THRESHOLD_KEYS,
    PLAY_MODEL_NAMES,
    parse_bid_threshold,
    parse_max_games,
)
from doudizhu_lab.ui.components import label
from doudizhu_lab.ui.theme import STYLE

if TYPE_CHECKING:
    from doudizhu_lab.ui.window import MainWindow


class SettingsDialog(QDialog):
    """GUI 线程中的偏好编辑器；应用前不改配置，运行期间只读，取消不发布任何改动。"""

    def __init__(self, window: "MainWindow") -> None:
        """在 GUI 线程根据 window 当前配置创建待保存表单，不立即发布偏好。

        相同阈值回显单个数值；旧分阶段阈值不同则留空并显示原值，留空保存仍保留原值。
        表单可滚动，提示与保存按钮固定可见；定时同步锁状态，不执行模型或手机输入。
        """
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
        self.form_scroll = QScrollArea()
        self.form_scroll.setWidgetResizable(True)
        self.form_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        form_panel = QWidget()
        forms = QVBoxLayout(form_panel)
        forms.setContentsMargins(0, 0, 0, 0)
        forms.setSpacing(12)
        self.form_scroll.setWidget(form_panel)
        layout.addWidget(self.form_scroll, 1)
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
        forms.addWidget(self.connection)
        self.task = QGroupBox("连续游戏")
        form = QFormLayout(self.task)
        # 增加阈值后由滚动区域容纳表单，保存和关闭按钮始终位于外层布局。
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
        thresholds = [profile.bid_settings[key] for key in BID_THRESHOLD_KEYS]
        self._legacy_bid_thresholds = len(set(thresholds)) > 1
        self.bid_threshold = QLineEdit("" if self._legacy_bid_thresholds else str(thresholds[0]))
        self.bid_threshold.setPlaceholderText(
            "留空保留原设置，输入后统一" if self._legacy_bid_thresholds else "请输入 -10～10 的数值，例如 0.5"
        )
        self.bid_threshold.setToolTip("首叫、首次抢地主和回抢共用此值，保存后用于下一次任务")
        form.addRow("叫抢阈值", self.bid_threshold)
        form.addRow(label("评分高于阈值才叫抢，阈值越低越积极，数值不是胜率。", "muted", wrap=True))
        if self._legacy_bid_thresholds:
            form.addRow(
                label(
                    f"当前分阶段阈值：首叫 {thresholds[0]}、首次抢 {thresholds[1]}、回抢 {thresholds[2]}。"
                    "留空保留原设置，输入数值后三项统一。",
                    "muted",
                    wrap=True,
                )
            )
        # 使用文本和 Python 整数，避免 QSpinBox/QIntValidator 的固定宽度整数上限。
        self.games = QLineEdit(str(profile.continuous_settings["max_games"]))
        self.games.setPlaceholderText("请输入大于0的整数")
        self.games.setToolTip("达到设定局数后停止，局数不设上限")
        form.addRow("最多局数", self.games)
        form.addRow(label("按设定局数完成后停止，不限制总时长；叫抢始终不加倍。", "muted", wrap=True))
        forms.addWidget(self.task)
        forms.addStretch()
        self.error = label("", "notice", wrap=True)
        self.error.hide()
        layout.addWidget(self.error)
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
        self.bidding.currentIndexChanged.connect(self.sync)
        self.sync()

    def sync(self) -> None:
        """在 GUI 线程同步设备、会话锁及阈值可编辑状态，返回 None。

        仅空闲且选择模型自动叫抢时允许编辑阈值；切换策略和定时刷新不覆盖待保存文本。
        本方法只更新表单状态，不修改配置或发送设备输入。
        """
        locked = self.window.settings_locked
        self.connection.setEnabled(not locked)
        self.task.setEnabled(not locked)
        self.bid_threshold.setEnabled(not locked and self.bidding.currentData() == "model")
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
        """在 GUI 线程校验目标局数与统一叫抢阈值，再通过窗口原子发布配置副本。

        阈值有值时同时更新三个兼容字段；仅原配置阈值不同时允许留空以保留原值。
        返回 None；解析、校验或保存的 ValueError/OSError 在表单显示中文原因，失败时
        不关闭对话框、不发布配置。成功后关闭表单，既有标定不变，也不授予执行权限。
        """
        profile = deepcopy(self.window.store.profile)
        profile.play_model = self.play_model.currentData()
        profile.continuous_policy = self.policy.currentData()
        profile.bid_settings["mode"] = self.bidding.currentData()
        try:
            profile.continuous_settings = {"max_games": parse_max_games(self.games.text())}
            threshold_text = self.bid_threshold.text().strip()
            if threshold_text or not self._legacy_bid_thresholds:
                threshold = parse_bid_threshold(threshold_text)
                for key in BID_THRESHOLD_KEYS:
                    profile.bid_settings[key] = threshold
            self.window.apply_settings(profile, self.adb_path.text().strip(), self.devices.currentData())
        except (OSError, ValueError) as exc:
            self.error.setText(str(exc))
            self.error.show()
            return
        self.accept()
