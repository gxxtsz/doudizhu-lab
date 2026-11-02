"""精简的离线录像回放与基础纠错；窗口不持有设备或输入授权。"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QMessageBox,
    QPushButton,
    QSlider,
    QSplitter,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from doudizhu_lab.domain.models import RANKS
from doudizhu_lab.domain.table import PHASE_NAMES, SEAT_NAMES
from doudizhu_lab.domain.tracker import replay_events
from doudizhu_lab.storage.config import Store
from doudizhu_lab.storage.replay import VideoReplay, profile_from_dict
from doudizhu_lab.storage.session import SessionStore
from doudizhu_lab.ui.canvas import FrameCanvas
from doudizhu_lab.ui.components import card
from doudizhu_lab.ui.components import label as make_label
from doudizhu_lab.ui.state import duration_text
from doudizhu_lab.ui.theme import STYLE

KINDS = {
    "game_start": "开局",
    "setup": "初始信息",
    "phase": "阶段",
    "actor": "行动提示",
    "play": "出牌",
    "pass": "不出",
    "gap": "疑点 / 缺口",
    "game_end": "结算",
}


def parse_cards(text: str) -> list[str]:
    """将输入转换为牌点列表，保留重复张数；未知字词抛出中文 ValueError。"""
    cards = (
        text.upper().replace("小王", "SJ").replace("大王", "BJ").replace("，", " ").replace(",", " ").split()
    )
    if any(c not in RANKS for c in cards):
        raise ValueError("牌点请用空格分隔：3～10 J Q K A 2 小王 大王")
    return cards


class EventEditor(QDialog):
    """编辑事件的业务字段；保存只追加修订，不覆盖原事件。"""

    def __init__(self, event: dict, parent=None) -> None:
        """GUI 线程编辑事件副本；提交前不修改会话，非法牌点由 value 抛出 ValueError。"""
        super().__init__(parent)
        self.setStyleSheet(STYLE)
        self.setWindowTitle("纠正事件 / 初始信息")
        self.original = event
        form = self.form = QFormLayout(self)
        form.setContentsMargins(22, 20, 22, 20)
        form.setSpacing(12)
        self.setMinimumWidth(460)
        self.kind = QComboBox()
        for key, label in KINDS.items():
            self.kind.addItem(label, key)
        self.kind.setCurrentIndex(max(0, self.kind.findData(event.get("kind", "play"))))
        self.seat, self.landlord = QComboBox(), QComboBox()
        for field in (self.seat, self.landlord):
            field.addItem("无法确认", None)
            for key, label in SEAT_NAMES.items():
                field.addItem(label, key)
        self.seat.setCurrentIndex(max(0, self.seat.findData(event.get("seat"))))
        data = event.get("data", {})
        self.landlord.setCurrentIndex(max(0, self.landlord.findData(data.get("landlord"))))
        self.time = QDoubleSpinBox()
        self.time.setDecimals(6)
        self.time.setRange(0, 1e10)
        self.time.setValue(event.get("pts_us", 0) / 1_000_000)
        self.game = QLineEdit(event.get("game_id", ""))
        self.cards = QLineEdit(" ".join(event.get("cards", [])))
        self.hand = QLineEdit(" ".join(data.get("initial_hand") or []))
        self.bottom = QLineEdit(" ".join(data.get("bottom") or []))
        self.order = QComboBox()
        self.order.addItem("尚未核实", [])
        self.order.addItem("自己 → 左侧 → 右侧", ["self", "left", "right"])
        self.order.addItem("自己 → 右侧 → 左侧", ["self", "right", "left"])
        self.order.setCurrentIndex(max(0, self.order.findData(data.get("seat_order", []))))
        self.phase = QComboBox()
        for key, label in PHASE_NAMES.items():
            self.phase.addItem(label, key)
        self.phase.setCurrentIndex(max(0, self.phase.findData(data.get("phase", "unknown"))))
        self.remaining = QLineEdit(
            " ".join(
                str(data.get("remaining", {}).get(s)) if data.get("remaining", {}).get(s) is not None else "?"
                for s in SEAT_NAMES
            )
        )
        self.confirmed = QCheckBox("已对照录像确认此事件")
        self.confirmed.setChecked(event.get("confirmed", True))
        self.incomplete = QCheckBox("本局中途开始或存在采集缺口")
        self.incomplete.setChecked(data.get("capture_incomplete", True))
        for title, field in (
            ("事件", self.kind),
            ("源时间（秒）", self.time),
            ("牌局编号", self.game),
            ("玩家", self.seat),
            ("牌点", self.cards),
            ("地主", self.landlord),
            ("初始手牌", self.hand),
            ("底牌", self.bottom),
            ("剩牌：自己 左侧 右侧", self.remaining),
            ("核实后的轮转", self.order),
            ("阶段", self.phase),
        ):
            form.addRow(title, field)
        form.addRow(self.confirmed)
        form.addRow(self.incomplete)
        form.addRow(
            QLabel(
                "牌点用空格分隔；初始信息仅用于开局 / 初始信息事件。\n修订后将重算后续状态，矛盾仍会显示为疑点。"
            )
        )
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Save).setText("保存修订")
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("取消")
        buttons.accepted.connect(self.validate)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)
        self.kind.currentIndexChanged.connect(self.update_fields)
        self.update_fields()

    def update_fields(self) -> None:
        """按事件类型展示相关字段；不清空隐藏字段，切换类型或取消不会丢失原始数据。"""
        kind = self.kind.currentData()
        setup = kind in ("game_start", "setup")
        for field in (self.landlord, self.hand, self.bottom, self.order, self.remaining, self.incomplete):
            self.form.setRowVisible(field, setup)
        self.form.setRowVisible(self.phase, setup or kind == "phase")

    def value(self) -> dict:
        """返回待追加的事件副本；保留重复牌点和原有数据，非法输入抛出中文 ValueError。"""
        value = dict(self.original) | {
            "kind": self.kind.currentData(),
            "pts_us": round(self.time.value() * 1e6),
            "game_id": self.game.text().strip(),
            "seat": self.seat.currentData(),
            "cards": parse_cards(self.cards.text()),
            "confirmed": self.confirmed.isChecked(),
        }
        data = dict(value.get("data", {}))
        if value["kind"] in ("game_start", "setup"):
            counts = self.remaining.text().split()
            if len(counts) != 3:
                raise ValueError("剩牌数请依次填写自己、左侧、右侧，未知用 ?")
            data.update(
                landlord=self.landlord.currentData(),
                initial_hand=parse_cards(self.hand.text()) or None,
                bottom=parse_cards(self.bottom.text()) or None,
                seat_order=self.order.currentData(),
                remaining={s: None if n == "?" else int(n) for s, n in zip(SEAT_NAMES, counts)},
                capture_incomplete=self.incomplete.isChecked(),
                phase=self.phase.currentData(),
            )
        elif value["kind"] == "phase":
            data["phase"] = self.phase.currentData()
        value["data"] = data
        if not value["game_id"]:
            raise ValueError("请填写事件所属牌局编号")
        return value

    def validate(self) -> None:
        """GUI 线程校验后接受对话框；错误原地提示，尚不向会话写入修订。"""
        try:
            self.value()
            self.accept()
        except (ValueError, TypeError) as exc:
            QMessageBox.warning(self, "请检查修订", str(exc))


class SessionDialog(QDialog):
    """离线回放及追加纠错；定位使用单个后台线程，滑动只保留最新请求，不创建 ADB。"""

    def __init__(self, store: Store, parent=None, *, initial_folder: Path | None = None) -> None:
        """GUI 线程创建回放窗口；initial_folder 可定位刚完成的会话，缺失时选择最近会话。"""
        super().__init__(parent)
        self.store = store
        self.session = self.replay = self.frame = None
        self.future = None
        self.pending_seek = None
        self.job = ""
        self.initial_folder = initial_folder
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="离线回放")
        self.cancel = Event()
        self.setWindowTitle("录像回放 · 斗地主实验室")
        self.setStyleSheet(STYLE)
        self.resize(1380, 860)
        self.setMinimumSize(1100, 720)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 22, 24, 20)
        layout.setSpacing(16)
        heading = QHBoxLayout()
        heading.addWidget(make_label("录像回放", "title"))
        heading.addStretch()
        heading.addWidget(make_label("本地录像 · 基础纠错", "muted"))
        layout.addLayout(heading)
        top = QHBoxLayout()
        self.sessions = QComboBox()
        self.sessions.setPlaceholderText("暂无已保存的会话")
        self.sessions.setMinimumWidth(260)
        self.sessions.currentIndexChanged.connect(self.open_selected)
        top.addWidget(self.sessions, 1)
        refresh = QPushButton("刷新")
        refresh.clicked.connect(self.refresh)
        top.addWidget(refresh)
        self.games = QComboBox()
        self.games.setMinimumWidth(150)
        self.games.addItem("全部牌局", None)
        self.games.currentIndexChanged.connect(self.render_events)
        top.addWidget(self.games)
        layout.addLayout(top)
        self.split = QSplitter(Qt.Orientation.Horizontal)
        self.split.setHandleWidth(18)
        preview, preview_layout = card("录像画面")
        self.canvas = FrameCanvas()
        self.canvas.empty_text = "暂无可播放的录像"
        preview_layout.addWidget(self.canvas, 1)
        self.split.addWidget(preview)
        panel = QWidget()
        panel.setObjectName("card")
        side = QVBoxLayout(panel)
        side.setContentsMargins(16, 16, 16, 16)
        side.setSpacing(12)
        side.addWidget(make_label("事件与疑点", "sectionTitle"))
        self.events = QTreeWidget()
        self.events.setHeaderLabels(["时间 / 事件", "玩家 / 内容"])
        self.events.setColumnWidth(0, 150)
        self.events.itemDoubleClicked.connect(self.jump)
        self.events.itemSelectionChanged.connect(self.update_controls)
        side.addWidget(self.events, 1)
        actions = QHBoxLayout()
        self.edit_button = QPushButton("修正事件")
        self.edit_button.clicked.connect(self.edit)
        actions.addWidget(self.edit_button, 1)
        more = self.more_button = QPushButton("更多纠错")
        menu = QMenu(more)
        for title, action in (
            ("插入事件", self.insert),
            ("撤销选中事件", self.delete),
            ("撤销最近修订", self.undo),
        ):
            menu.addAction(title, action)
        more.setMenu(menu)
        actions.addWidget(more)
        side.addLayout(actions)
        self.state_label = make_label("选择会话后显示牌局与疑点", "muted", wrap=True)
        side.addWidget(self.state_label)
        panel.setMinimumWidth(320)
        self.split.addWidget(panel)
        self.split.setStretchFactor(0, 1)
        self.split.setStretchFactor(1, 0)
        self.split.setSizes([920, 350])
        layout.addWidget(self.split, 1)
        transport = QHBoxLayout()
        self.play = QPushButton("播放")
        self.play.setObjectName("primary")
        self.play.clicked.connect(self.toggle_play)
        transport.addWidget(self.play)
        self.frame_buttons = []
        for title, delta in (("上一帧", -1), ("下一帧", 1)):
            button = QPushButton(title)
            button.clicked.connect(lambda checked=False, d=delta: self.step(d))
            transport.addWidget(button)
            self.frame_buttons.append(button)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.sliderReleased.connect(
            lambda: self.seek(self.replay.start_us + self.slider.value() * 1000) if self.replay else None
        )
        transport.addWidget(self.slider, 1)
        self.time_label = QLabel("00:00:00")
        transport.addWidget(self.time_label)
        layout.addLayout(transport)
        self.status = make_label("双击事件或疑点定位录像；修订保留原始记录。", "muted", wrap=True)
        layout.addWidget(self.status)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.poll)
        self.timer.start(50)
        self.play_timer = QTimer(self)
        self.play_timer.setInterval(200)
        self.play_timer.timeout.connect(self.advance)
        self.refresh()

    def update_controls(self) -> None:
        """GUI 线程按录像和选中事件启用操作；空会话不出现可点击但无效的播放或修正。"""
        self.play.setEnabled(self.replay is not None)
        self.slider.setEnabled(self.replay is not None)
        for button in self.frame_buttons:
            button.setEnabled(self.replay is not None)
        selected = self.selected()
        self.edit_button.setEnabled(bool(self.session and selected and "event_id" in selected))
        self.more_button.setEnabled(self.session is not None)

    def refresh(self) -> None:
        """GUI 线程扫描本地会话，优先保留指定路径或当前选择；后台定位期间不切换会话。"""
        if self.future:
            return
        current = self.initial_folder or self.sessions.currentData()
        self.initial_folder = None
        self.sessions.blockSignals(True)
        self.sessions.clear()
        for database in sorted((self.store.root / "sessions").rglob("session.sqlite3"), reverse=True):
            folder = database.parent
            self.sessions.addItem(str(folder.relative_to(self.store.root / "sessions")), folder)
        selected = self.sessions.findData(current)
        self.sessions.setCurrentIndex(max(0, selected))
        self.sessions.blockSignals(False)
        self.open_selected()

    def open_selected(self) -> None:
        """GUI 线程打开所选数据库和视频索引，登记后台定位；读取失败直接显示中文原因。"""
        if self.future:
            return
        self.play_timer.stop()
        self.play.setText("播放")
        self.canvas.clear()
        self.frame = None
        if self.session:
            self.session.close()
        folder = self.sessions.currentData()
        self.session = self.replay = None
        self.events.clear()
        self.pending_seek = None
        if folder is None:
            self.update_controls()
            return
        try:
            self.session = SessionStore(folder)
            self.replay = VideoReplay(folder)
            self.canvas.profile = profile_from_dict(self.session.metadata()["profile"])
            self.slider.setRange(0, max(0, (self.replay.end_us - self.replay.start_us) // 1000))
            self.render_games()
            self.seek(self.replay.start_us)
        except Exception as exc:
            self.status.setText(f"会话读取失败：{exc}")
        self.update_controls()

    def render_games(self):
        current = self.games.currentData()
        self.games.blockSignals(True)
        self.games.clear()
        self.games.addItem("全部牌局", None)
        states = self.session.game_states()
        for ident, state in states.items():
            self.games.addItem(f"{ident} · {'完整' if state['complete'] else '待核对'}", ident)
        self.games.setCurrentIndex(max(0, self.games.findData(current)))
        self.games.blockSignals(False)
        self.render_events()

    def render_events(self):
        self.events.clear()
        if not self.session:
            return
        selected = self.games.currentData()
        events = self.session.effective_events()
        for event in events:
            if selected and event["game_id"] != selected:
                continue
            text = " ".join(event.get("cards", [])) or event.get("data", {}).get("message", "")
            label = f"{SEAT_NAMES.get(event.get('seat'), '')} {text}"
            if not event.get("confirmed", True):
                label += " [待确认]"
            item = QTreeWidgetItem(
                [f"{self.relative(event['pts_us']):.2f}s · {KINDS.get(event['kind'], event['kind'])}", label]
            )
            item.setData(0, Qt.ItemDataRole.UserRole, event)
            self.events.addTopLevelItem(item)
        for issue in self.session.rows("issues"):
            if selected and issue.get("game_id") not in (selected, ""):
                continue
            item = QTreeWidgetItem(
                [f"{self.relative(issue['pts_us']):.2f}s · 疑点", issue.get("message", issue["code"])]
            )
            item.setData(0, Qt.ItemDataRole.UserRole, issue)
            self.events.addTopLevelItem(item)
        states = self.session.game_states(events)
        problems = [p for key, s in states.items() if not selected or key == selected for p in s["problems"]]
        self.state_label.setText(
            f"{len(states)} 局 · {len(self.session.rows('issues'))} 条原始疑点 · {len(self.session.rows('revisions'))} 次修订\n"
            + "；".join(problems[-4:])
        )

    def relative(self, pts):
        return (pts - self.replay.start_us) / 1e6 if self.replay else 0

    def seek(self, pts, after=False):
        if self.replay:
            self.pending_seek = (pts, after)

    def poll(self) -> None:
        """GUI 线程收取解码结果并只提交最新定位请求；解码异常停止播放，不访问手机。"""
        if self.future and self.future.done():
            future, self.future = self.future, None
            try:
                result = future.result()
                self.frame = result
                self.canvas.set_frame(result)
                self.slider.setValue(round(self.relative(result.media_pts_us) * 1000))
                self.time_label.setText(duration_text(self.relative(result.media_pts_us)))
            except Exception as exc:
                self.status.setText(str(exc))
                self.play_timer.stop()
                self.play.setText("播放")
            self.sessions.setEnabled(True)
        if not self.future and self.pending_seek:
            pts, after = self.pending_seek
            self.pending_seek = None
            self.job = "seek"
            self.sessions.setEnabled(False)
            self.future = self.pool.submit(self.replay.at, pts, after=after)

    def step(self, delta):
        self.play_timer.stop()
        self.play.setText("播放")
        if self.frame:
            self.seek(self.frame.media_pts_us + delta, delta > 0)

    def toggle_play(self):
        if self.play_timer.isActive():
            self.play_timer.stop()
            self.play.setText("播放")
        elif self.replay:
            self.play_timer.start()
            self.play.setText("暂停")

    def advance(self):
        if self.frame and not self.future:
            self.seek(min(self.replay.end_us, self.frame.media_pts_us + 200_000), True)
            if self.frame.media_pts_us >= self.replay.end_us - 1000:
                self.toggle_play()

    def selected(self):
        item = self.events.currentItem()
        return item.data(0, Qt.ItemDataRole.UserRole) if item else None

    def jump(self, item, column=0):
        event = item.data(0, Qt.ItemDataRole.UserRole)
        self.seek(event["pts_us"])

    def edit(self):
        event = self.selected()
        if not event or "event_id" not in event:
            return
        dialog = EventEditor(event, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.session.revise(
                "boundary" if event["kind"] in ("game_start", "game_end") else "replace",
                event["event_id"],
                dialog.value(),
            )
            self.render_games()
            self.export()

    def insert(self):
        if not self.frame or not self.session:
            return
        event = {
            "game_id": self.games.currentData()
            or next(iter(replay_events(self.session.effective_events())), "g001-manual"),
            "pts_us": self.frame.media_pts_us,
            "kind": "play",
            "frame_id": self.frame.frame_id,
            "data": {},
        }
        dialog = EventEditor(event, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.session.revise("insert", "", dialog.value())
            self.render_games()
            self.export()

    def delete(self):
        event = self.selected()
        if event and "event_id" in event:
            self.session.revise("delete", event["event_id"], {})
            self.render_games()
            self.export()

    def undo(self):
        if self.session and self.session.rows("revisions"):
            rows = self.session.rows("revisions")
            disabled = set()
            for row in reversed(rows):
                if row["seq"] not in disabled and row["operation"] == "undo":
                    disabled.add(int(row["target"]))
            target = next(
                (r for r in reversed(rows) if r["operation"] != "undo" and r["seq"] not in disabled), None
            )
            if target:
                self.session.revise("undo", str(target["seq"]), {})
                self.render_games()
                self.export()

    def export(self) -> None:
        """纠错后导出派生结果；保留原始数据库记录，写盘异常通过状态文字说明。"""
        if self.session:
            try:
                self.session.export()
                self.status.setText("修订已保存，原始记录保留。")
            except Exception as exc:
                self.status.setText(f"导出失败：{exc}")

    def closeEvent(self, event) -> None:
        """关闭本窗口的播放定时器、后台定位和会话连接，不影响手机或其他窗口。"""
        self.cancel.set()
        self.timer.stop()
        self.play_timer.stop()
        self.pool.shutdown(wait=True, cancel_futures=True)
        if self.session:
            self.session.close()
        event.accept()
