"""会话浏览、录像逐帧核对及追加纠错窗口。整个窗口不持有 ADB 客户端。"""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
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
    QMessageBox,
    QPushButton,
    QSlider,
    QSplitter,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from doudizhu_lab.domain.models import RANKS, REGION_NAMES
from doudizhu_lab.domain.table import PHASE_NAMES, SEAT_NAMES
from doudizhu_lab.domain.tracker import replay_events
from doudizhu_lab.storage.config import Store, write_rgb
from doudizhu_lab.storage.replay import VideoReplay, profile_from_dict, reanalyze
from doudizhu_lab.storage.session import SessionStore
from doudizhu_lab.ui.canvas import FrameCanvas
from doudizhu_lab.vision.table import MarkerLibrary

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
MARKERS = {
    "landlord": "地主标志",
    "turn": "行动提示",
    "count": "剩牌数字",
    "pass": "不出文字",
    "phase": "阶段按钮",
    "result": "结算",
    "bottom_back": "底牌背面",
    "winner": "最后出牌提示",
}


def parse_cards(text: str) -> list[str]:
    """将用户输入转换为牌点列表，保留重复张数；未知字词明确报错。"""
    cards = (
        text.upper().replace("小王", "SJ").replace("大王", "BJ").replace("，", " ").replace(",", " ").split()
    )
    if any(c not in RANKS for c in cards):
        raise ValueError("牌点请用空格分隔：3～10 J Q K A 2 小王 大王")
    return cards


class EventEditor(QDialog):
    """编辑事件的业务字段；保存只追加修订，不覆盖原事件。"""

    def __init__(self, event: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("纠正事件 / 初始信息")
        self.original = event
        form = QFormLayout(self)
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

    def value(self) -> dict:
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

    def validate(self):
        try:
            self.value()
            self.accept()
        except (ValueError, TypeError) as exc:
            QMessageBox.warning(self, "请检查修订", str(exc))


class SessionDialog(QDialog):
    """回放与纠错完全离线；耗时定位和重算使用一个后台线程，滑动请求只保留最新位置。"""

    def __init__(self, store: Store, parent=None):
        super().__init__(parent)
        self.store = store
        self.session = self.replay = self.frame = None
        self.future = None
        self.pending_seek = None
        self.job = ""
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="离线回放")
        self.cancel = Event()
        self.setWindowTitle("会话 · 回放 · 纠错（离线，不发送手机输入）")
        self.resize(1380, 930)
        layout = QVBoxLayout(self)
        top = QHBoxLayout()
        self.sessions = QComboBox()
        self.sessions.setMinimumWidth(350)
        self.sessions.currentIndexChanged.connect(self.open_selected)
        top.addWidget(self.sessions, 1)
        refresh = QPushButton("刷新会话")
        refresh.clicked.connect(self.refresh)
        top.addWidget(refresh)
        self.games = QComboBox()
        self.games.currentIndexChanged.connect(self.render_events)
        top.addWidget(self.games)
        for title, action in (
            ("独立事件标注", self.independent_labels),
            ("导出整局 JSON", self.export),
            ("按原配置模板重算", self.analyze),
            ("使用新模板重算", lambda: self.analyze(True)),
        ):
            button = QPushButton(title)
            button.clicked.connect(action)
            top.addWidget(button)
        layout.addLayout(top)
        split = QSplitter(Qt.Orientation.Horizontal)
        self.canvas = FrameCanvas()
        self.canvas.selected.connect(self.selection)
        split.addWidget(self.canvas)
        panel = QWidget()
        side = QVBoxLayout(panel)
        self.events = QTreeWidget()
        self.events.setHeaderLabels(["时间 / 事件", "玩家 / 牌点 / 疑点"])
        self.events.setColumnWidth(0, 185)
        self.events.itemDoubleClicked.connect(self.jump)
        side.addWidget(self.events, 1)
        actions = QHBoxLayout()
        for title, action in (
            ("修改", self.edit),
            ("插入", self.insert),
            ("撤销事件", self.delete),
            ("撤销修订", self.undo),
        ):
            button = QPushButton(title)
            button.clicked.connect(action)
            actions.addWidget(button)
        side.addLayout(actions)
        self.state_label = QLabel("选择会话后显示牌局与疑点")
        self.state_label.setWordWrap(True)
        side.addWidget(self.state_label)
        split.addWidget(panel)
        split.setSizes([880, 450])
        layout.addWidget(split, 1)
        transport = QHBoxLayout()
        for title, delta in (("上一帧", -1), ("下一帧", 1)):
            button = QPushButton(title)
            button.clicked.connect(lambda checked=False, d=delta: self.step(d))
            transport.addWidget(button)
        self.play = QPushButton("播放")
        self.play.clicked.connect(self.toggle_play)
        transport.addWidget(self.play)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.sliderReleased.connect(
            lambda: self.seek(self.replay.start_us + self.slider.value() * 1000) if self.replay else None
        )
        transport.addWidget(self.slider, 1)
        self.time_label = QLabel("00:00")
        transport.addWidget(self.time_label)
        layout.addLayout(transport)
        sampling = QHBoxLayout()
        self.mode = QComboBox()
        for label, key in (
            ("查看录像", "view"),
            ("裁剪牌点模板", "card"),
            ("裁剪标记模板", "marker"),
            ("校准区域", "region"),
        ):
            self.mode.addItem(label, key)
        self.mode.currentIndexChanged.connect(self.mode_changed)
        sampling.addWidget(self.mode)
        self.target = QComboBox()
        sampling.addWidget(self.target)
        self.label = QLineEdit()
        self.label.setPlaceholderText("牌点或标记标签")
        sampling.addWidget(self.label)
        self.glyph = QDoubleSpinBox()
        self.glyph.setRange(5, 250)
        self.glyph.setValue(50)
        self.glyph.setPrefix("牌点高度 ")
        sampling.addWidget(self.glyph)
        sample = QPushButton("标注当前帧")
        sample.clicked.connect(self.annotate)
        sampling.addWidget(sample)
        layout.addLayout(sampling)
        self.status = QLabel("双击事件或疑点跳转录像；区域校准与新模板只影响后续录制 / 新分析版本。")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.poll)
        self.timer.start(50)
        self.play_timer = QTimer(self)
        self.play_timer.setInterval(200)
        self.play_timer.timeout.connect(self.advance)
        self.mode_changed()
        self.refresh()

    def refresh(self):
        if self.future:
            return
        current = self.sessions.currentData()
        self.sessions.blockSignals(True)
        self.sessions.clear()
        for database in sorted((self.store.root / "sessions").rglob("session.sqlite3"), reverse=True):
            folder = database.parent
            self.sessions.addItem(str(folder.relative_to(self.store.root / "sessions")), folder)
        selected = self.sessions.findData(current)
        self.sessions.setCurrentIndex(max(0, selected))
        self.sessions.blockSignals(False)
        self.open_selected()

    def open_selected(self):
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
        if folder is None:
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

    def poll(self):
        if self.future and self.future.done():
            future, self.future = self.future, None
            try:
                result = future.result()
                if self.job == "seek":
                    self.frame = result
                    self.canvas.set_frame(result)
                    self.slider.setValue(round(self.relative(result.media_pts_us) * 1000))
                    self.time_label.setText(f"{self.relative(result.media_pts_us):.3f} 秒")
                else:
                    self.status.setText(f"新分析版本已保存：{result}")
                    self.refresh()
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

    def independent_labels(self):
        """打开独立标注窗口；不传入当前预测列表，防止人工真值被自动结果预填。"""
        if self.session is None or self.future is not None:
            return
        from doudizhu_lab.ui.truth import TruthDialog

        try:
            dialog = TruthDialog(self.session.folder, self)
            dialog.exec()
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "独立标注不可用", str(exc))

    def export(self):
        if self.session:
            try:
                self.status.setText(f"已导出：{self.session.export()}")
            except Exception as exc:
                self.status.setText(f"导出失败：{exc}")

    def analyze(self, updated=False):
        if not self.session or self.future:
            return
        if self.session.metadata().get("status") == "recording":
            self.status.setText("录制尚未停止；请在停止后重新分析完整会话。")
            return
        self.play_timer.stop()
        self.pending_seek = None
        self.job = "analyze"
        self.sessions.setEnabled(False)
        self.status.setText("正在后台重新分析，原观察与修订保留；关闭窗口可取消。")
        self.future = self.pool.submit(
            reanalyze,
            self.session.folder,
            template_root=self.store.root if updated else None,
            cancel=self.cancel,
        )

    def mode_changed(self):
        self.play_timer.stop()
        self.target.clear()
        mode = self.mode.currentData()
        self.canvas.mode = "view" if mode == "view" else "template"
        self.canvas.control_enabled = False
        if mode == "card":
            for key, label in (("hand", "手牌"), ("bottom", "底牌"), ("table", "三家出牌")):
                self.target.addItem(label, key)
        elif mode == "marker":
            for key, label in MARKERS.items():
                self.target.addItem(label, key)
        elif mode == "region":
            for key, label in REGION_NAMES.items():
                self.target.addItem(label, key)
            for key in self.store.profile.indicator_regions:
                family, _, seat = key.partition("_")
                label = MARKERS.get(key, MARKERS.get(family, key))
                self.target.addItem(f"{SEAT_NAMES.get(seat, '公共')} · {label}", key)

    def selection(self, area):
        if not self.frame or not self.session:
            return
        mode, target = self.mode.currentData(), self.target.currentData()
        crop = area.pixels(self.frame.width, self.frame.height).crop(self.frame.rgb)
        game = next(
            (
                e["game_id"]
                for e in reversed(self.session.effective_events())
                if e["pts_us"] <= self.frame.media_pts_us
            ),
            "",
        )
        source = {
            "session": self.session.metadata().get("session_id"),
            "game": game,
            "frame": self.frame.frame_id,
            "pts_us": self.frame.media_pts_us,
            "image_sha256": hashlib.sha256(self.frame.rgb.tobytes()).hexdigest(),
            "area": asdict(area),
        }
        try:
            if mode == "card":
                ranks = parse_cards(self.label.text())
                if len(ranks) != 1:
                    raise ValueError("模板必须且只能填写一个牌点标签")
                path = self.store.add_template(crop, ranks[0], target, source)
                self.status.setText(f"牌点模板已保存：{path.name}")
            elif mode == "marker":
                label = self.label.text().strip()
                if target in ("turn", "landlord", "winner"):
                    label = "present"
                elif target == "count" and label.isdigit():
                    label = "n:" + label
                elif target in ("phase", "result", "bottom_back"):
                    label = {name: key for key, name in PHASE_NAMES.items()}.get(label, label)
                if not label:
                    raise ValueError(
                        "请填写标记标签：完整剩牌数字 0～20；阶段填写发牌、叫抢地主、加倍、出牌、结算；不出文字照图填写"
                    )
                path = MarkerLibrary(self.store.root / "markers").add(crop, target, label, source)
                self.status.setText(f"界面标记模板已保存：{path.name}")
            elif mode == "region":
                if target in self.store.profile.regions:
                    self.store.profile.regions[target] = area
                    self.store.profile.glyph_heights[target] = (
                        self.glyph.value() * self.store.profile.base_height / self.frame.height
                    )
                else:
                    self.store.profile.indicator_regions[target] = area
                self.store.save()
                self.status.setText("区域已保存到当前配置；原会话快照保持可重现。")
        except Exception as exc:
            self.status.setText(f"保存失败：{exc}")

    def annotate(self):
        if not self.frame:
            return
        dialog = QDialog(self)
        dialog.setWindowTitle("独立人工标注当前帧")
        form = QFormLayout(dialog)
        fields = {}
        for key, label in REGION_NAMES.items():
            fields[key] = QLineEdit("?")
            form.addRow(label, fields[key])
        form.addRow(
            QLabel(
                "空白表示无牌；? 表示未标注（不参与该区域评分）。\n请对照原图填写，不能复制自动预测当作真值。"
            )
        )
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Save).setText("保存标注")
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("取消")
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        try:
            labels = {
                key: None if field.text().strip() == "?" else parse_cards(field.text())
                for key, field in fields.items()
            }
            identity = hashlib.sha256(self.frame.rgb.tobytes()).hexdigest()
            folder = self.store.root / "annotations"
            folder.mkdir(exist_ok=True)
            write_rgb(folder / f"{identity[:16]}.png", self.frame.rgb)
            payload = {
                "version": 2,
                "frame_id": self.frame.frame_id,
                "pts_us": self.frame.media_pts_us,
                "source_session": self.session.metadata().get(
                    "source_session", self.session.metadata().get("session_id")
                ),
                "source_game": next(
                    (
                        e["game_id"]
                        for e in reversed(self.session.effective_events())
                        if e["pts_us"] <= self.frame.media_pts_us
                    ),
                    None,
                ),
                "image_sha256": identity,
                "regions": labels,
                "source": "manual",
            }
            (folder / f"{identity[:16]}.json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            self.status.setText("独立标注已保存到 data/annotations；评测前请按牌局隔离模板来源。")
        except Exception as exc:
            self.status.setText(str(exc))

    def closeEvent(self, event):
        self.cancel.set()
        self.timer.stop()
        self.play_timer.stop()
        self.pool.shutdown(wait=True, cancel_futures=True)
        if self.session:
            self.session.close()
        event.accept()
