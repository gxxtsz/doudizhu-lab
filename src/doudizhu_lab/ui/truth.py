"""独立标注页：只显示录像与人工标签，不载入预测，不具备手机输入能力。"""

from concurrent.futures import ThreadPoolExecutor

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
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

from doudizhu_lab.domain.table import SEAT_NAMES
from doudizhu_lab.storage.replay import VideoReplay
from doudizhu_lab.storage.truth import TruthStore, validate_truth
from doudizhu_lab.ui.canvas import FrameCanvas
from doudizhu_lab.ui.replay import parse_cards


class TruthDialog(QDialog):
    """逐帧独立标注窗口；定位线程只有一个，滑动时只保留最新请求。

    牌局、开局手牌和逐动作检查均由查看录像的人填写，不预填机器结果。
    保存进入独立 JSON 及历史版本，整个页面不持有 ADB、录制器或事件修订接口。
    """

    def __init__(self, folder, parent=None):
        super().__init__(parent)
        self.labels, self.video = TruthStore(folder), VideoReplay(folder)
        self.frame = self.future = self.pending = None
        self.editing = None
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="独立标注录像")
        self.setWindowTitle("独立录像标注 · 自动预测隐藏 · 不操作手机")
        self.resize(1450, 980)
        layout = QVBoxLayout(self)
        layout.addWidget(
            QLabel("请先选定连续验收区间，再逐帧标注。牌点用空格分隔；初始手牌为收底牌后的 20 / 17 张。")
        )
        top = QHBoxLayout()
        self.games = QComboBox()
        self.games.currentIndexChanged.connect(self.load_game)
        top.addWidget(self.games, 1)
        for text, slot in (
            ("登记全录像为连续核对区间", self.select_all),
            ("在此开始新局", self.start_game),
            ("在此结束本局", self.end_game),
            ("校验并发布标注", self.publish),
        ):
            button = QPushButton(text)
            button.clicked.connect(slot)
            top.addWidget(button)
        layout.addLayout(top)
        split = QSplitter(Qt.Orientation.Horizontal)
        self.canvas = FrameCanvas()
        self.canvas.mode = "view"
        self.canvas.control_enabled = False
        split.addWidget(self.canvas)
        panel = QWidget()
        form = QFormLayout(panel)
        self.role, self.seat, self.kind, self.order = QComboBox(), QComboBox(), QComboBox(), QComboBox()
        for combo in (self.role, self.seat):
            combo.addItem("请选择座位", None)
            for key, label in SEAT_NAMES.items():
                combo.addItem(label, key)
        self.kind.addItem("出牌", "play")
        self.kind.addItem("不出 / 要不起 / 过", "pass")
        self.order.addItem("请选择已核实轮转", [])
        self.order.addItem("自己 → 右侧 → 左侧", ["self", "right", "left"])
        self.order.addItem("自己 → 左侧 → 右侧", ["self", "left", "right"])
        self.hand, self.bottom = QLineEdit(), QLineEdit()
        self.cards, self.after_hand, self.counts = QLineEdit(), QLineEdit(), QLineEdit()
        self.incomplete = QCheckBox("首尾不完整局，单列")
        self.incomplete.setToolTip("首尾不完整的牌局单独保存，不参与五局完整牌局的验收成绩。")
        self.reviewed = QCheckBox("本动作已对照录像核对牌点、手牌与三家剩牌")
        for title, widget in (
            ("本局地主", self.role),
            ("初始手牌", self.hand),
            ("公开底牌", self.bottom),
            ("轮转顺序", self.order),
            ("不完整局", self.incomplete),
        ):
            form.addRow(title, widget)
        save_game = QPushButton("保存开局信息")
        save_game.clicked.connect(self.save_game)
        form.addRow(save_game)
        complete = QPushButton("本局已逐动作复核完毕")
        complete.clicked.connect(self.complete_game)
        form.addRow(complete)
        for title, widget in (
            ("动作玩家", self.seat),
            ("动作类型", self.kind),
            ("动作牌点", self.cards),
            ("动作后自己的手牌", self.after_hand),
            ("剩牌：自己 左侧 右侧", self.counts),
        ):
            form.addRow(title, widget)
        form.addRow(self.reviewed)
        add = QPushButton("保存当前帧的动作标注")
        add.clicked.connect(self.save_event)
        form.addRow(add)
        reset = QPushButton("清空动作编辑，准备新增")
        reset.clicked.connect(self.reset_event)
        form.addRow(reset)
        delete = QPushButton("撤回选中的动作标注")
        delete.clicked.connect(self.delete_event)
        form.addRow(delete)
        panel.setMinimumWidth(370)
        split.addWidget(panel)
        split.setSizes([1040, 390])
        layout.addWidget(split, 3)
        transport = QHBoxLayout()
        for title, delta in (("前一帧", -1), ("后一帧", 1), ("后移 1 秒", 1_000_000)):
            button = QPushButton(title)
            button.clicked.connect(lambda checked=False, d=delta: self.step(d))
            transport.addWidget(button)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, max(1, (self.video.end_us - self.video.start_us) // 1000))
        self.slider.valueChanged.connect(lambda value: self.seek(self.video.start_us + value * 1000))
        transport.addWidget(self.slider, 1)
        self.time = QLabel("尚未定位")
        transport.addWidget(self.time)
        layout.addLayout(transport)
        self.events = QTreeWidget()
        self.events.setHeaderLabels(["相对秒", "玩家", "动作", "牌点", "逐项核对"])
        self.events.itemDoubleClicked.connect(self.load_event)
        layout.addWidget(self.events, 1)
        self.status = QLabel("标注文件独立于自动事件及修订。双击人工动作可返回原帧并编辑。")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.poll)
        self.timer.start(35)
        self.refresh()
        self.seek(self.video.start_us)

    def guarded(self, operation):
        """把输入校验和媒体错误显示为中文提示；失败不伪装成保存成功。"""
        try:
            return operation()
        except (OSError, ValueError, TypeError, KeyError, StopIteration) as exc:
            QMessageBox.warning(self, "请检查独立标注", str(exc))
            return None

    def current_game(self):
        ident = self.games.currentData()
        return next((g for g in self.labels.document["games"] if g["game_id"] == ident), None)

    def refresh(self, ident=None):
        ident = ident or self.games.currentData()
        self.games.blockSignals(True)
        self.games.clear()
        for game in self.labels.document["games"]:
            suffix = "已复核" if game.get("reviewed") else "待复核"
            self.games.addItem(f"{game['game_id']} · {suffix}", game["game_id"])
        self.games.setCurrentIndex(max(0, self.games.findData(ident)))
        self.games.blockSignals(False)
        self.load_game()

    def load_game(self):
        game = self.current_game()
        self.events.clear()
        if game is None:
            return
        self.role.setCurrentIndex(max(0, self.role.findData(game.get("landlord"))))
        self.order.setCurrentIndex(max(0, self.order.findData(game.get("seat_order", []))))
        self.hand.setText(" ".join(game.get("initial_hand") or []))
        self.bottom.setText(" ".join(game.get("bottom") or []))
        self.incomplete.setChecked(game.get("incomplete", False))
        for event in self.labels.document["events"]:
            if event["game_id"] != game["game_id"]:
                continue
            item = QTreeWidgetItem(
                [
                    f"{(event['pts_us'] - self.video.start_us) / 1e6:.3f}",
                    SEAT_NAMES[event["seat"]],
                    "出牌" if event["kind"] == "play" else "不出",
                    " ".join(event.get("cards", [])),
                    "是" if event.get("reviewed") else "否",
                ]
            )
            item.setData(0, Qt.ItemDataRole.UserRole, event)
            self.events.addTopLevelItem(item)

    def select_all(self):
        if self.guarded(lambda: self.labels.select_range(self.video.start_us, self.video.end_us)) is None:
            self.status.setText("连续核对区间状态：" + str(self.labels.document.get("selection")))

    def start_game(self):
        if self.frame is None:
            return

        def run():
            if not self.labels.document.get("selection"):
                raise ValueError("请先登记连续核对区间")
            ident = self.labels.put_game(
                {"start_us": self.frame.media_pts_us, "start_frame": self.frame.frame_id, "end_us": None}
            )
            self.refresh(ident)
            self.reset_event()

        self.guarded(run)

    def end_game(self):
        game = self.current_game()
        if game and self.frame:
            self.guarded(
                lambda: self.labels.put_game(
                    game
                    | {"end_us": self.frame.media_pts_us, "end_frame": self.frame.frame_id, "reviewed": False}
                )
            )
            self.refresh()

    def save_game(self):
        def run():
            game = self.current_game()
            if game is None:
                raise ValueError("请先标记牌局开始")
            self.labels.put_game(
                game
                | {
                    "landlord": self.role.currentData(),
                    "initial_hand": parse_cards(self.hand.text()),
                    "bottom": parse_cards(self.bottom.text()),
                    "seat_order": self.order.currentData(),
                    "incomplete": self.incomplete.isChecked(),
                    "reviewed": False,
                }
            )
            self.refresh()

        self.guarded(run)

    def complete_game(self):
        def run():
            game = self.current_game()
            if game is None:
                raise ValueError("请先建立牌局")
            subset = {
                "games": [game | {"reviewed": True}],
                "events": [e for e in self.labels.document["events"] if e["game_id"] == game["game_id"]],
            }
            result = validate_truth(subset, require_review=True)
            if not result["valid"]:
                raise ValueError("；".join(result["problems"][:8]))
            self.labels.put_game(game | {"reviewed": True})
            self.refresh()
            self.status.setText("本局独立标注及逐动作状态检查通过。")

        self.guarded(run)

    def save_event(self):
        def run():
            game = self.current_game()
            if self.frame is None or game is None or self.seat.currentData() is None:
                raise ValueError("请选择牌局、玩家并定位录像帧")
            counts = [int(v) for v in self.counts.text().split()]
            if len(counts) != 3 or any(not 0 <= v <= 20 for v in counts):
                raise ValueError("剩牌依次填写自己、左侧、右侧三个 0～20 的数字")
            value = {
                "game_id": game["game_id"],
                "pts_us": self.frame.media_pts_us,
                "frame_id": self.frame.frame_id,
                "kind": self.kind.currentData(),
                "seat": self.seat.currentData(),
                "cards": parse_cards(self.cards.text()),
                "reviewed": self.reviewed.isChecked(),
                "checkpoint": {
                    "hand": parse_cards(self.after_hand.text()),
                    "remaining": dict(zip(SEAT_NAMES, counts)),
                },
            }
            if value["kind"] == "pass" and value["cards"]:
                raise ValueError("不出动作的牌点应为空")
            if self.editing:
                value["event_id"] = self.editing
            self.labels.put_event(value)
            self.refresh()
            self.reset_event()
            self.status.setText("人工动作已保存；原自动事件和修订记录没有变化。")

        self.guarded(run)

    def load_event(self, item, column=0):
        event = item.data(0, Qt.ItemDataRole.UserRole)
        self.editing = event["event_id"]
        self.seat.setCurrentIndex(self.seat.findData(event["seat"]))
        self.kind.setCurrentIndex(self.kind.findData(event["kind"]))
        self.cards.setText(" ".join(event.get("cards", [])))
        check = event.get("checkpoint", {})
        self.after_hand.setText(" ".join(check.get("hand") or []))
        self.counts.setText(" ".join(str(check.get("remaining", {}).get(s, "?")) for s in SEAT_NAMES))
        self.reviewed.setChecked(event.get("reviewed", False))
        self.seek(event["pts_us"], after=True)

    def reset_event(self):
        self.editing = None
        self.cards.clear()
        self.reviewed.setChecked(False)

    def delete_event(self):
        item = self.events.currentItem()
        if item:
            self.guarded(lambda: self.labels.delete_event(item.data(0, Qt.ItemDataRole.UserRole)["event_id"]))
            self.reset_event()
            self.refresh()

    def publish(self):
        result = self.guarded(lambda: self.labels.save(publish=True))
        if result:
            self.status.setText(f"独立标注已发布：{result}")

    def seek(self, pts, *, after=False):
        """提交最新定位请求；媒体访问在后台执行，快速拖动不会无限排队。"""
        self.pending = (max(self.video.start_us, min(self.video.end_us, pts)), after)
        if self.future is None:
            pts, after = self.pending
            self.pending = None
            self.future = self.pool.submit(self.video.at, pts, after=after)

    def step(self, delta):
        if self.frame:
            self.seek(self.frame.media_pts_us + delta, after=delta > 0)

    def poll(self):
        if self.future is not None and self.future.done():
            future, self.future = self.future, None
            try:
                self.frame = future.result()
                self.canvas.set_frame(self.frame)
                self.canvas.control_enabled = False
                self.time.setText(f"{(self.frame.media_pts_us - self.video.start_us) / 1e6:.3f} 秒")
                self.slider.blockSignals(True)
                self.slider.setValue((self.frame.media_pts_us - self.video.start_us) // 1000)
                self.slider.blockSignals(False)
            except Exception as exc:
                self.status.setText(f"录像定位失败：{exc}")
            if self.pending:
                pts, after = self.pending
                self.seek(pts, after=after)

    def closeEvent(self, event):
        self.timer.stop()
        self.pending = None
        self.pool.shutdown(wait=False, cancel_futures=True)
        event.accept()
