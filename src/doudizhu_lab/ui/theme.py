"""桌面窗口公共样式。"""

STYLE = """
QMainWindow, QWidget#root { background: #f0f3f7; color: #263747; }
QLabel#title { font-size: 23px; font-weight: 700; color: #102b40; }
QLabel#subtitle { color: #667a8c; }
QGroupBox { border: 1px solid #d7e0e9; border-radius: 8px; margin-top: 12px; padding-top: 14px; background: #ffffff; font-weight: 600; }
QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 5px; }
QPushButton { background: #ffffff; border: 1px solid #cdd8e3; border-radius: 5px; padding: 7px 12px; color: #243b50; }
QPushButton:hover { background: #e6f4f1; border-color: #2b9f8b; }
QPushButton:disabled { color: #9aa9b6; background: #eef1f4; }
QPushButton#primary { background: #137f72; color: white; border-color: #137f72; font-weight: 600; }
QComboBox, QSpinBox, QLineEdit { padding: 5px; border: 1px solid #cdd8e3; border-radius: 4px; background: white; min-height: 20px; }
QTreeWidget, QTextEdit { background: #ffffff; border: 1px solid #d7e0e9; border-radius: 5px; }
QHeaderView::section { background: #e8eef4; padding: 6px; border: none; color: #526b80; }
"""
