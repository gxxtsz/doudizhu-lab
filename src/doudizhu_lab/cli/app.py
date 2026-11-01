"""桌面命令行入口；所有实际设备操作仍由窗口明确授权。"""

import argparse
import sys
from pathlib import Path

from PySide6.QtCore import QTimer
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication, QMessageBox

from doudizhu_lab import __version__
from doudizhu_lab.cli.console import configure_console
from doudizhu_lab.storage.config import Store
from doudizhu_lab.ui.window import MainWindow


def main() -> None:
    """启动桌面程序；--data-dir 用于隔离测试数据，--preview 用于连续预览验收。"""
    configure_console()
    parser = argparse.ArgumentParser(description="斗地主实验室")
    parser.add_argument("--data-dir", type=Path, help="本地运行数据目录")
    classifier = parser.add_mutually_exclusive_group()
    classifier.add_argument(
        "--hand-ocr", action="store_true", help="试用免训练手牌OCR；需要 vision 依赖与模型"
    )
    classifier.add_argument("--hand-template", action="store_true", help="本次启动使用传统手牌模板")
    parser.add_argument("--preview", action="store_true", help="仅一台授权设备时自动开始只读预览")
    parser.add_argument("--record", action="store_true", help="仅一台授权设备时自动开始整局记录")
    parser.add_argument(
        "--flow-room", choices=("初级", "中级", "当前"), help="当前表示沿用已授权的初级/中级场次"
    )
    parser.add_argument(
        "--continuous", action="store_true", help="显式启动连续游戏，默认5局或30分钟；需要vision依赖"
    )
    parser.add_argument(
        "--continuous-policy",
        choices=("strict", "infer", "balanced"),
        help="连续策略；平衡模式允许唯一推断和明确回合的规则接管",
    )
    parser.add_argument(
        "--bidding", choices=("disabled", "model"), help="叫抢策略：固定拒绝或现成模型自动叫抢；始终不加倍"
    )
    parser.add_argument(
        "--max-games", type=int, choices=range(1, 31), help="本次连续任务上限1～30局，不保存自动授权"
    )
    parser.add_argument(
        "--time-limit",
        type=int,
        choices=range(1, 14401),
        metavar="秒",
        help="本次任务时长上限，最长4小时；到时打完当前局停止",
    )
    parser.add_argument(
        "--acceptance", action="store_true", help="以已冻结版本记录独立验收批次，需配合 --record"
    )
    parser.add_argument("--quit-after", type=int, help="指定秒数后退出并保存运行统计，用于稳定性验证")
    parser.add_argument("--version", action="version", version=f"斗地主实验室 {__version__}")
    args = parser.parse_args()
    app = QApplication(sys.argv[:1])
    app.setFont(QFont("Microsoft YaHei UI", 10))
    try:
        store = Store(args.data_dir)
        if args.hand_ocr or args.hand_template:
            store.profile.hand_classifier = "pretrained-ocr" if args.hand_ocr else "template"
        if args.flow_room:
            store.profile.flow_room = args.flow_room
        if args.continuous_policy:
            store.profile.continuous_policy = args.continuous_policy
        if args.bidding:
            store.profile.bid_settings["mode"] = args.bidding
        if args.max_games:
            store.profile.continuous_settings["max_games"] = args.max_games
        if args.time_limit:
            store.profile.continuous_settings["time_limit"] = args.time_limit
        window = MainWindow(store)
    except (ValueError, OSError) as exc:
        QMessageBox.critical(None, "启动失败", str(exc))
        raise SystemExit(1) from exc
    window.show()
    window.acceptance.setChecked(args.acceptance)
    if args.preview or args.record or args.continuous:

        def start_when_ready():
            if not window.busy and window.devices.count() == 1 and not window.running:
                if args.continuous:
                    window.start_continuous()
                else:
                    window.toggle_recording() if args.record else window.toggle_preview()
            elif window.busy:
                QTimer.singleShot(250, start_when_ready)

        QTimer.singleShot(500, start_when_ready)
    if args.quit_after:
        QTimer.singleShot(args.quit_after * 1000, window.close)
    app.exec()


if __name__ == "__main__":
    main()
