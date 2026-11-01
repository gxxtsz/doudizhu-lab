"""冻结图像上的操作区域校准；本对话框不持有设备输入接口。"""

from copy import deepcopy
from dataclasses import asdict

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

from doudizhu_lab.ui.canvas import FrameCanvas
from doudizhu_lab.vision.controls import add_button_template


class BotCalibrationDialog(QDialog):
    """框选按钮样本及搜索区，填写普通牌顶和选中上移量，保存归一化第三版配置。"""

    def __init__(self, store, frame, parent=None):
        super().__init__(parent)
        self.store, self.frame = store, frame
        self.config = deepcopy(store.profile.bot_controls)
        self.setWindowTitle("操作标定 · 冻结画面，不发送点击")
        self.resize(1100, 720)
        layout = QVBoxLayout(self)
        self.canvas = FrameCanvas(self)
        self.canvas.mode = "template"
        self.canvas.set_frame(frame)
        self.canvas.selected.connect(self.selected)
        layout.addWidget(self.canvas, 1)
        form = QFormLayout()
        self.target = QComboBox()
        self.target.addItem("出牌按钮搜索区域", ("region", "play", "", False))
        self.target.addItem("不出 / 要不起按钮搜索区域", ("region", "pass", "", False))
        for kind, label in (("play", "出牌"), ("pass", "不出"), ("pass", "要不起")):
            for enabled in (True, False):
                self.target.addItem(
                    f"{'启用' if enabled else '禁用'}按钮模板：{label}", ("template", kind, label, enabled)
                )
        form.addRow("拖拽用途", self.target)
        self.top, self.offset = QSpinBox(), QSpinBox()
        for widget in (self.top, self.offset):
            widget.setRange(1, frame.height - 1)
        config = self.config
        self.top.setValue(round((config.get("normal_top") or 0.698) * frame.height))
        self.offset.setValue(round((config.get("selected_offset") or 30 / 1080) * frame.height))
        form.addRow("未选中牌顶（原图像素）", self.top)
        form.addRow("选中上移量（原图像素）", self.offset)
        layout.addLayout(form)
        self.status = QLabel("模板请框住完整文字及背景；搜索区覆盖该按钮可能出现的全部位置。")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        actions = QHBoxLayout()
        save = QPushButton("保存布局标定")
        save.clicked.connect(self.save)
        close = QPushButton("关闭")
        close.clicked.connect(self.reject)
        actions.addWidget(save)
        actions.addWidget(close)
        layout.addLayout(actions)

    def selected(self, area):
        """接受控件的归一化框选，只写配置或模板，记录当前源帧身份供追溯。"""
        task, kind, label, enabled = self.target.currentData()
        try:
            if task == "region":
                self.config["buttons"][kind] = asdict(area)
                self.status.setText("搜索区域已暂存；点击保存布局标定后生效。")
            else:
                path = add_button_template(
                    self.store.root / "controls",
                    self.frame,
                    area.pixels(self.frame.width, self.frame.height),
                    kind=kind,
                    label=label,
                    enabled=enabled,
                    source={"method": "人工冻结画面框选", "captured_at": self.frame.captured_at},
                )
                self.status.setText(f"已保存按钮模板：{path.name}；将在下一次记录中使用。")
        except (ValueError, OSError) as exc:
            self.status.setText(str(exc))

    def save(self):
        """归一化保存绝对基准；全部牌被选中时仍使用此基准，不能从当前最小高度猜测。"""
        config = self.config
        if set(config["buttons"]) != {"play", "pass"}:
            self.status.setText("请先框选两种按钮的搜索区域。")
            return
        if self.offset.value() >= self.top.value():
            self.status.setText("选中上移量必须小于普通牌顶高度。")
            return
        config.update(
            normal_top=self.top.value() / self.frame.height,
            selected_offset=self.offset.value() / self.frame.height,
            top_tolerance=min(6 / self.frame.height, self.offset.value() / self.frame.height / 3),
            verified=True,
        )
        try:
            self.store.profile.bot_controls = deepcopy(config)
            self.store.save()
            self.status.setText("布局已保存；真机实际选牌仍需单独验证。")
        except (ValueError, OSError) as exc:
            self.status.setText(str(exc))
