"""连续游戏工作台组件；只创建界面，任务和输入权限由控制器负责。"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTextEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from doudizhu_lab.ui.canvas import FrameCanvas
from doudizhu_lab.ui.theme import STYLE

if TYPE_CHECKING:
    from doudizhu_lab.ui.window import MainWindow


def label(text: str, name: str = "", *, wrap: bool = False) -> QLabel:
    """在 GUI 线程创建标签；name 控制样式，wrap 控制换行，不访问设备。"""
    widget = QLabel(text)
    widget.setObjectName(name)
    widget.setWordWrap(wrap)
    widget.setMinimumWidth(0)
    widget.setTextFormat(Qt.TextFormat.PlainText)
    widget.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
    return widget


def card(title: str) -> tuple[QFrame, QVBoxLayout]:
    """在 GUI 线程返回带标题的白色面板和内容布局；不启动后台任务。"""
    box = QFrame()
    box.setObjectName("card")
    layout = QVBoxLayout(box)
    layout.setContentsMargins(18, 18, 18, 18)
    layout.setSpacing(10)
    layout.addWidget(label(title, "sectionTitle"))
    return box, layout


def build_main_window(window: "MainWindow") -> None:
    """在 GUI 线程装配只读画面和任务面板；不加载模型或授予设备输入。"""
    window.setWindowTitle("斗地主实验室 · 连续游戏工作台")
    window.resize(1440, 900)
    window.setMinimumSize(1100, 720)
    window.setStyleSheet(STYLE)
    root = QWidget()
    root.setObjectName("root")
    window.setCentralWidget(root)
    layout = QVBoxLayout(root)
    layout.setContentsMargins(26, 22, 26, 18)
    layout.setSpacing(20)
    header = QHBoxLayout()
    heading = QVBoxLayout()
    heading.setSpacing(4)
    heading.addWidget(label("斗地主实验室", "title"))
    heading.addWidget(label("连续游戏与录像回放", "subtitle"))
    header.addLayout(heading)
    header.addStretch()
    window.connection_label = label("未连接设备", "badge")
    window.connection_label.setMaximumWidth(230)
    window.model_label = label("模型 · 准备中", "badge")
    header.addWidget(window.connection_label)
    header.addSpacing(4)
    header.addWidget(window.model_label)
    header.addSpacing(14)
    window.settings_button = QPushButton("设置")
    window.settings_button.clicked.connect(window.open_settings)
    header.addWidget(window.settings_button)
    window.replay_button = QPushButton("录像回放")
    window.replay_button.clicked.connect(window.open_sessions)
    header.addWidget(window.replay_button)
    layout.addLayout(header)

    body = QHBoxLayout()
    body.setSpacing(20)
    preview, preview_layout = card("设备画面")
    preview_layout.setContentsMargins(16, 16, 16, 14)
    preview_layout.setSpacing(12)
    window.canvas = FrameCanvas()
    window.canvas.empty_text = "等待手机连接\n连接唯一授权设备后，将自动显示画面"
    window.canvas.setMinimumSize(440, 260)
    preview_layout.addWidget(window.canvas, 1)
    preview_footer = QHBoxLayout()
    window.preview_label = label("只读预览", "muted")
    preview_footer.addWidget(window.preview_label)
    preview_footer.addStretch()
    preview_footer.addWidget(label("画面仅供查看", "muted"))
    preview_layout.addLayout(preview_footer)
    body.addWidget(preview, 1)
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    scroll.setFixedWidth(320)
    scroll.setObjectName("taskPanel")
    side = QWidget()
    side_layout = QVBoxLayout(side)
    side_layout.setContentsMargins(0, 0, 0, 0)
    side_layout.setSpacing(16)
    task, task_layout = card("连续任务")
    window.task_badge = label("等待开始", "badge")
    task_layout.addWidget(window.task_badge, 0, Qt.AlignmentFlag.AlignLeft)
    window.scene_label = label("等待开始", "sectionTitle", wrap=True)
    task_layout.addWidget(window.scene_label)
    window.notice_label = label("正在准备出牌模型…", "notice", wrap=True)
    task_layout.addWidget(window.notice_label)
    window.retry_button = QPushButton("重试加载模型")
    window.retry_button.clicked.connect(window.load_bot)
    window.retry_button.hide()
    task_layout.addWidget(window.retry_button)
    window.action_button = QPushButton("开始连续游戏")
    window.action_button.setObjectName("primary")
    window.action_button.setMinimumHeight(30)
    window.action_button.clicked.connect(window.primary_action)
    task_layout.addWidget(window.action_button)
    window.stop_button = QPushButton("停止任务")
    window.stop_button.setObjectName("danger")
    window.stop_button.clicked.connect(window.stop_and_save)
    task_layout.addWidget(window.stop_button)
    window.saved_replay_button = QPushButton("查看本次回放")
    window.saved_replay_button.clicked.connect(window.open_current_session)
    window.saved_replay_button.hide()
    task_layout.addWidget(window.saved_replay_button)
    side_layout.addWidget(task)
    progress, progress_layout = card("任务进度")
    progress_layout.addWidget(label("已完成 / 目标", "eyebrow"))
    window.games_label = label("0 / 5 局", "metric")
    progress_layout.addWidget(window.games_label)
    window.progress_bar = QProgressBar()
    window.progress_bar.setTextVisible(False)
    progress_layout.addWidget(window.progress_bar)
    window.elapsed_label = label("00:00:00 / 00:30:00", "sectionTitle")
    progress_layout.addWidget(window.elapsed_label)
    window.policy_label = label("", "muted", wrap=True)
    progress_layout.addWidget(window.policy_label)
    side_layout.addWidget(progress)
    table, table_layout = card("当前局面")
    window.role_label = label("身份　—", wrap=True)
    window.actor_label = label("行动　—", wrap=True)
    table_layout.addWidget(window.role_label)
    table_layout.addWidget(window.actor_label)
    window.remaining_label = label("剩牌　自己 — · 左侧 — · 右侧 —", wrap=True)
    table_layout.addWidget(window.remaining_label)
    divider = QFrame()
    divider.setObjectName("divider")
    table_layout.addWidget(divider)
    window.bot_action_label = label("等待新的牌局", "muted", wrap=True)
    table_layout.addWidget(window.bot_action_label)
    side_layout.addWidget(table)
    side_layout.addStretch()
    scroll.setWidget(side)
    body.addWidget(scroll)
    layout.addLayout(body, 1)
    activity = QHBoxLayout()
    window.log_toggle = QToolButton()
    window.log_toggle.setObjectName("disclosure")
    window.log_toggle.setText("运行日志")
    window.log_toggle.setCheckable(True)
    window.log_toggle.setArrowType(Qt.ArrowType.RightArrow)
    window.log_toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
    window.log_toggle.toggled.connect(window.toggle_log)
    activity.addWidget(window.log_toggle)
    window.latest_log = label("等待设备与模型就绪", "muted")
    window.latest_log.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
    activity.addWidget(window.latest_log, 1)
    layout.addLayout(activity)
    window.log_view = QTextEdit()
    window.log_view.setReadOnly(True)
    window.log_view.setFixedHeight(140)
    window.log_view.document().setMaximumBlockCount(250)
    window.log_view.hide()
    layout.addWidget(window.log_view)
