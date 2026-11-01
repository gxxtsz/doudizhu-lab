"""主窗口组件装配；界面布局与业务协调分开维护。"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTextEdit,
    QTreeWidget,
    QVBoxLayout,
    QWidget,
)

from doudizhu_lab.domain.models import REGION_NAMES
from doudizhu_lab.ui.canvas import FrameCanvas
from doudizhu_lab.ui.theme import STYLE

if TYPE_CHECKING:
    from doudizhu_lab.ui.window import MainWindow


def build_main_window(self: "MainWindow") -> None:
    """仅在GUI线程创建窗口组件并连接信号；不启动设备、录像或模型任务。"""
    self.setWindowTitle("斗地主实验室 · 安卓真机调试工具")
    self.resize(1380, 880)
    self.setMinimumSize(1040, 700)
    root = QWidget()
    root.setObjectName("root")
    self.setCentralWidget(root)
    layout = QVBoxLayout(root)
    layout.setContentsMargins(20, 16, 20, 14)
    header = QHBoxLayout()
    titles = QVBoxLayout()
    title = QLabel("斗地主实验室")
    title.setObjectName("title")
    subtitle = QLabel("安卓真机 · 连续记录 · 牌面识别 · 回放纠错")
    subtitle.setObjectName("subtitle")
    titles.addWidget(title)
    titles.addWidget(subtitle)
    header.addLayout(titles)
    header.addStretch()
    self.connection_label = QLabel("尚未连接")
    header.addWidget(self.connection_label)
    layout.addLayout(header)
    toolbar = QHBoxLayout()
    self.devices = QComboBox()
    self.devices.setMinimumWidth(260)
    self.devices.currentIndexChanged.connect(self._device_changed)
    toolbar.addWidget(self.devices)
    self.refresh_button = QPushButton("刷新设备")
    self.refresh_button.clicked.connect(self.refresh_devices)
    toolbar.addWidget(self.refresh_button)
    self.start_button = QPushButton("开始预览")
    self.start_button.setObjectName("primary")
    self.start_button.clicked.connect(self.toggle_preview)
    toolbar.addWidget(self.start_button)
    self.mode = QComboBox()
    for label, value in (
        ("查看", "view"),
        ("点击控制", "control"),
        ("区域标定 · 冻结", "region"),
        ("模板采集 · 冻结", "template"),
    ):
        self.mode.addItem(label, value)
    self.mode.currentIndexChanged.connect(self._mode_changed)
    toolbar.addWidget(self.mode)
    toolbar.addStretch()
    self.import_button = QPushButton("导入截图")
    self.import_button.clicked.connect(self.import_image)
    toolbar.addWidget(self.import_button)
    self.export_button = QPushButton("导出当前帧")
    self.export_button.clicked.connect(self.export_frame)
    toolbar.addWidget(self.export_button)
    self.settings_button = QPushButton("ADB 路径")
    self.settings_button.clicked.connect(self.choose_adb)
    toolbar.addWidget(self.settings_button)
    layout.addLayout(toolbar)
    recording_bar = QHBoxLayout()
    self.record_button = QPushButton("开始记录")
    self.record_button.setObjectName("primary")
    self.record_button.clicked.connect(self.toggle_recording)
    recording_bar.addWidget(self.record_button)
    self.replay_button = QPushButton("会话 / 回放 / 纠错")
    self.replay_button.clicked.connect(self.open_sessions)
    recording_bar.addWidget(self.replay_button)
    self.acceptance = QCheckBox("正式验收（已冻结版本）")
    self.acceptance.setToolTip("先运行 ddz-check freeze-acceptance；代码、模板或配置有改动时拒绝开始验收。")
    recording_bar.addWidget(self.acceptance)
    self.game_status = QLabel("整局记录：未开始 · 视频与识别结果保存在本机")
    recording_bar.addWidget(self.game_status, 1)
    layout.addLayout(recording_bar)
    bot_bar = QHBoxLayout()
    self.bot_load = QPushButton("加载 WP 模型")
    self.bot_load.clicked.connect(self.load_bot)
    bot_bar.addWidget(self.bot_load)
    self.bot_calibrate = QPushButton("操作标定")
    self.bot_calibrate.clicked.connect(self.calibrate_bot)
    bot_bar.addWidget(self.bot_calibrate)
    self.bot_mode = QComboBox()
    for label, value in (("仅建议", "suggest"), ("单步执行", "step"), ("自动执行", "auto")):
        self.bot_mode.addItem(label, value)
    self.bot_mode.currentIndexChanged.connect(self.change_bot_mode)
    bot_bar.addWidget(self.bot_mode)
    self.bot_execute = QPushButton("执行当前一步 / 恢复")
    self.bot_execute.clicked.connect(self.execute_bot_step)
    bot_bar.addWidget(self.bot_execute)
    self.bot_stop = QPushButton("立即停止 / 人工接管")
    self.bot_stop.clicked.connect(self.stop_bot)
    bot_bar.addWidget(self.bot_stop)
    self.bot_label = QLabel("机器人：未加载 · 仅建议")
    self.bot_label.setWordWrap(True)
    bot_bar.addWidget(self.bot_label, 1)
    layout.addLayout(bot_bar)
    continuous_bar = QHBoxLayout()
    self.continuous_start = QPushButton("开始连续游戏")
    self.continuous_start.setObjectName("primary")
    self.continuous_start.clicked.connect(self.start_continuous)
    continuous_bar.addWidget(self.continuous_start)
    self.continuous_policy = QComboBox()
    for label, value in (("严格模式", "strict"), ("唯一推断", "infer"), ("平衡模式", "balanced")):
        self.continuous_policy.addItem(label, value)
    self.continuous_policy.setCurrentIndex(
        self.continuous_policy.findData(self.store.profile.continuous_policy)
    )
    self.continuous_policy.currentIndexChanged.connect(self.change_continuous_policy)
    continuous_bar.addWidget(self.continuous_policy)
    self.bidding_mode = QComboBox()
    self.bidding_mode.addItem("固定不叫不抢", "disabled")
    self.bidding_mode.addItem("模型自动叫抢", "model")
    self.bidding_mode.setCurrentIndex(self.bidding_mode.findData(self.store.profile.bid_settings["mode"]))
    self.bidding_mode.currentIndexChanged.connect(self.change_bidding_mode)
    continuous_bar.addWidget(self.bidding_mode)
    self.continuous_resume = QPushButton("恢复连续任务")
    self.continuous_resume.clicked.connect(self.resume_continuous)
    self.continuous_resume.setEnabled(False)
    continuous_bar.addWidget(self.continuous_resume)
    self.continuous_label = QLabel("连续任务：未开始 · 叫抢按所选策略，不加倍 · 最多5局或30分钟")
    self.continuous_label.setWordWrap(True)
    continuous_bar.addWidget(self.continuous_label, 1)
    layout.addLayout(continuous_bar)
    split = QSplitter(Qt.Orientation.Horizontal)
    self.canvas = FrameCanvas()
    self.canvas.profile = self.store.profile
    self.canvas.tapped.connect(self.request_tap)
    self.canvas.selected.connect(self.selection_done)
    split.addWidget(self.canvas)
    side = QWidget()
    side_layout = QVBoxLayout(side)
    side_layout.setContentsMargins(10, 0, 0, 0)
    self.mode_help = QLabel("查看模式只显示画面。选择点击控制后，可直接点击画面。")
    self.mode_help.setWordWrap(True)
    side_layout.addWidget(self.mode_help)
    calibration = QGroupBox("标定与模板")
    form = QFormLayout(calibration)
    self.region = QComboBox()
    for key, name in REGION_NAMES.items():
        self.region.addItem(name, key)
    form.addRow("目标区域", self.region)
    self.coverage_label = QLabel()
    self.coverage_label.setWordWrap(True)
    form.addRow(self.coverage_label)
    self.reload_button = QPushButton("重新载入模板并识别")
    self.reload_button.clicked.connect(self.reload_templates)
    form.addRow(self.reload_button)
    side_layout.addWidget(calibration)
    result_box = QGroupBox("当前帧识别")
    result_layout = QVBoxLayout(result_box)
    self.result_tree = QTreeWidget()
    self.result_tree.setHeaderLabels(["区域 / 牌点", "状态 / 匹配分数"])
    self.result_tree.setColumnWidth(0, 150)
    result_layout.addWidget(self.result_tree)
    note = QLabel(
        "匹配分数和模型评分都不是实际胜率。\n仅建议模式不发送输入；执行模式由独立状态机核对后操作。"
    )
    note.setWordWrap(True)
    note.setStyleSheet("color: #718293; font-size: 11px;")
    result_layout.addWidget(note)
    side_layout.addWidget(result_box, 1)
    self.candidate_tree = QTreeWidget()
    self.candidate_tree.setHeaderLabels(["待确认动作", "等待的证据"])
    self.candidate_tree.setMaximumHeight(105)
    self.candidate_tree.setVisible(False)
    side_layout.addWidget(self.candidate_tree)
    control_box = QGroupBox("原图坐标点击")
    row = QHBoxLayout(control_box)
    self.x_input, self.y_input = QSpinBox(), QSpinBox()
    for field in (self.x_input, self.y_input):
        field.setRange(0, 9999)
    self.x_input.setPrefix("X ")
    self.y_input.setPrefix("Y ")
    row.addWidget(self.x_input)
    row.addWidget(self.y_input)
    self.tap_button = QPushButton("点击一次")
    self.tap_button.clicked.connect(lambda: self.request_tap(self.x_input.value(), self.y_input.value()))
    row.addWidget(self.tap_button)
    side_layout.addWidget(control_box)
    side.setMinimumWidth(350)
    split.addWidget(side)
    split.setStretchFactor(0, 1)
    split.setStretchFactor(1, 0)
    split.setSizes([980, 350])
    layout.addWidget(split, 1)
    self.status_label = QLabel("等待设备；初始布局针对 2400×1080 横屏，可手动校准。")
    layout.addWidget(self.status_label)
    self.timeline = QTreeWidget()
    self.timeline.setHeaderLabels(["最近事件 / 源时间", "玩家 / 内容", "确认状态"])
    self.timeline.setMaximumHeight(110)
    self.timeline.setVisible(False)
    layout.addWidget(self.timeline)
    self.log_view = QTextEdit()
    self.log_view.setReadOnly(True)
    self.log_view.setMaximumHeight(95)
    self.log_view.document().setMaximumBlockCount(250)
    layout.addWidget(self.log_view)
    self.setStyleSheet(STYLE)
    self.update_coverage()
