"""工作台、设置和回放共用的浅色主题；尺寸均为 Qt 逻辑像素。"""

STYLE = """
QMainWindow, QDialog, QWidget#root { background: #f4f6f8; color: #243342; }
QWidget { font-family: "Microsoft YaHei UI"; font-size: 13px; color: #243342; }
QLabel { background: transparent; }
QLabel#title { font-size: 22px; font-weight: 700; color: #182b3b; }
QLabel#eyebrow { color: #778591; font-size: 11px; }
QLabel#sectionTitle { font-size: 15px; font-weight: 600; color: #233a48; }
QLabel#muted, QLabel#subtitle { color: #778591; }
QLabel#metric { font-size: 27px; font-weight: 600; color: #183c40; }
QLabel#badge { padding: 6px 10px; background: #e9eef2; border-radius: 6px; color: #536573; }
QLabel#badge[tone="success"] { background: #e3f2ed; color: #137765; }
QLabel#badge[tone="warning"] { background: #fff2db; color: #966321; }
QLabel#badge[tone="error"] { background: #fbe8e7; color: #b24b45; }
QFrame#card, QWidget#card { background: white; border: 1px solid #e1e7ec; border-radius: 10px; }
QFrame#divider { background: #e9edf1; border: none; max-height: 1px; }
QLabel#notice { padding: 10px; border-radius: 6px; background: #f0f5f6; color: #526b75; }
QLabel#notice[tone="error"] { background: #fbeeed; color: #a3443d; }
QLabel#notice[tone="warning"] { background: #fff5e4; color: #8c6127; }
QPushButton, QToolButton { background: white; border: 1px solid #dce3e9; border-radius: 6px; padding: 8px 14px; min-height: 20px; }
QPushButton:hover, QToolButton:hover { background: #eff6f4; border-color: #93bdb3; }
QPushButton:disabled, QToolButton:disabled { color: #9ba8b2; background: #f2f4f6; border-color: #e6eaee; }
QPushButton#primary { background: #147d70; color: white; border-color: #147d70; font-weight: 600; }
QPushButton#primary:hover { background: #116a60; }
QPushButton#primary:disabled { background: #c6ddd7; border-color: #c6ddd7; color: #f8fbfa; }
QPushButton#danger { color: #b74d45; border-color: #e8cdc9; background: #fffafa; }
QPushButton#danger:hover { background: #fbece9; }
QPushButton#danger:disabled { color: #b6a5a3; border-color: #e9e4e3; background: #f7f5f5; }
QToolButton#disclosure { border: none; background: transparent; padding: 3px 0; color: #5f7180; }
QComboBox, QSpinBox, QTimeEdit, QDoubleSpinBox, QLineEdit { padding: 7px 9px; border: 1px solid #dce3e9; border-radius: 6px; background: white; min-height: 20px; selection-background-color: #147d70; }
QComboBox:disabled, QSpinBox:disabled, QTimeEdit:disabled, QLineEdit:disabled { color: #85929c; background: #eef1f4; }
QTreeWidget, QTextEdit { background: white; border: 1px solid #e1e7ec; border-radius: 6px; outline: none; }
QTreeWidget::item { padding: 7px 3px; }
QTreeWidget::item:selected { background: #e1f0eb; color: #195d52; }
QHeaderView::section { background: #f3f6f8; padding: 9px; border: none; color: #72818d; font-weight: 500; }
QProgressBar { border: none; border-radius: 3px; background: #edf2f3; max-height: 6px; min-height: 6px; }
QProgressBar::chunk { background: #319b87; border-radius: 3px; }
QScrollArea { border: none; background: transparent; }
QScrollArea > QWidget > QWidget { background: transparent; }
QScrollBar:vertical { background: transparent; width: 6px; margin: 0; }
QScrollBar::handle:vertical { background: #cbd6dc; border-radius: 3px; min-height: 30px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: transparent; }
QSplitter::handle { background: transparent; width: 14px; }
QSlider::groove:horizontal { height: 4px; background: #dfe7e9; border-radius: 2px; }
QSlider::handle:horizontal { background: #147d70; width: 12px; margin: -5px 0; border-radius: 6px; }
QMenu { background: white; border: 1px solid #dce3e9; padding: 5px; }
QMenu::item { padding: 8px 24px; }
QMenu::item:selected { background: #e8f3ef; }
QGroupBox { border: 1px solid #e1e7ec; border-radius: 8px; margin-top: 14px; padding: 18px 12px 12px; background: white; font-weight: 600; }
QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 5px; }
"""
