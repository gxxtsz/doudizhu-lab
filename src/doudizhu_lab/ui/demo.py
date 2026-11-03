"""独立的原生窗口模拟入口；没有真实设备后端，也不查询 USB 或 ADB。"""

import argparse
import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from queue import Queue
from threading import Event, RLock, Thread
from time import monotonic

import cv2
import numpy as np
from PySide6.QtCore import QObject, QTimer
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication

from doudizhu_lab import __version__
from doudizhu_lab.cli.console import configure_console
from doudizhu_lab.domain.models import Device
from doudizhu_lab.domain.settings import PLAY_MODEL_NAMES
from doudizhu_lab.domain.table import TableObservation
from doudizhu_lab.runtime.bot import InputGate
from doudizhu_lab.simulation.bot import SimulatedDevice
from doudizhu_lab.storage.config import Store
from doudizhu_lab.storage.evidence import EvidenceWriter
from doudizhu_lab.storage.session import SessionStore
from doudizhu_lab.ui.window import MainWindow


class DemoPipeline:
    """实时节奏的本地模拟视频源；使用同一 GameTracker、模型进程和执行器。

    图像只是示意牌桌，布局来自模拟器，不计入真实牌面识别成绩。动作仍须通过
    坐标命中和真实规则确认；审计 source 固定为 simulation_bot，不能冒充真机。
    """

    bot_source = "simulation_bot"
    serial = "simulation"

    def __init__(self, store):
        self.profile = deepcopy(store.profile)
        self.device = SimulatedDevice(73100, "landlord")
        self.session = SessionStore.create(store.root / "simulation-ui", self.profile, [], self.serial)
        self.session.set_meta("source", "simulation_ui")
        self.session.set_meta("note", "本地模拟示意图；没有采集手机录像")
        self.evidence = EvidenceWriter(self.session.folder / "evidence")
        self.client = self
        self.stop_event, self.lock = Event(), RLock()
        self.allowed = True
        self._started = monotonic()
        self.write_queue = Queue()
        self.frame, self.latest = None, None
        self.events, self.recorded = [], set()
        self.frames = 0
        self.thread = None

    def start(self):
        """启动纯 Python 模拟源和已有证据写入器，不构造视频服务器或 ADB 命令。"""
        self.evidence.start()
        self.thread = Thread(target=self._run, name="本地模拟牌桌", daemon=True)
        self.thread.start()

    def _run(self):
        last_action = monotonic()
        while not self.stop_event.wait(0.12):
            with self.lock:
                if (
                    not self.device.env.game_over
                    and self.device.env.acting_player_position != self.device.roles["self"]
                    and monotonic() - last_action > 2
                ):
                    self.device.act(self.device.rng.choice(self.device.env.game_infoset.legal_actions))
                    last_action = monotonic()
                self.device.now = monotonic() - 0.12
                snap = self.device.render()
                # 标识牌点只用牌面符号；姓名、阶段和建议由主窗口的中文标签显示。
                rgb = snap.frame.rgb.copy()
                rgb[:] = (21, 83, 73)
                for card in snap.hand:
                    x, y = card.hit.x, card.hit.y - (10 if card.selected else 0)
                    cv2.rectangle(rgb, (x - 2, y - 8), (x + 20, 98), (240, 242, 239), -1)
                    cv2.putText(rgb, card.rank, (x, y + 7), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (25, 40, 50), 1)
                self.frame = replace(
                    snap.frame, rgb=np.frombuffer(rgb.tobytes(), dtype=np.uint8).reshape(rgb.shape)
                )
                self.latest = replace(snap, frame=self.frame)
                self.frames += 1
                self.session.add_observation(snap.observation)
                for event in snap.history:
                    if event["event_id"] not in self.recorded:
                        self.recorded.add(event["event_id"])
                        self.session.add_event(event)
                        from doudizhu_lab.domain.tracker import GameEvent

                        self.events.append(GameEvent(**event))

    def bot_snapshot(self):
        """返回不可变快照引用，与 GUI 的最新图像使用同一帧标识。"""
        with self.lock:
            return self.latest

    def tap(self, x, y, frame, *, max_age=1):
        """只允许模拟来源；坐标命中当前按钮或牌张，绝无转发真实输入的路径。"""
        with self.lock:
            if (
                frame.transport != "simulation"
                or frame.serial != self.serial
                or frame.age > max_age
                or self.stop_event.is_set()
            ):
                raise ValueError("模拟点击来源或时效无效")
            kind = "select"
            for button in self.device.last.buttons:
                b = button.hit
                if b.x <= x < b.x + b.width and b.y <= y < b.y + b.height:
                    kind = button.kind
            state = self.device.last.state
            self.device.send(kind, x, y, self.device.last, f"{state['game_id']}-{state['turn']}")

    def poll(self):
        """复用主窗口轮询协议，不把演示图计为编码视频字节。"""
        with self.lock:
            events, self.events = self.events, []
            return {
                "frame": self.frame,
                "observation": TableObservation(**self.latest.observation) if self.latest else None,
                "recognition": None,
                "state": deepcopy(self.device.tracker.state),
                "events": events,
                "messages": [],
                "allowed": self.allowed,
                "stopped": self.stop_event.is_set(),
                "pending_candidates": [],
                "metrics": {
                    "decoded_frames": self.frames,
                    "analyzed_frames": self.frames,
                    "encoded_bytes": 0,
                },
            }

    def stop(self):
        """先停止模拟反馈和证据线程，再保存结果；由窗口负责关闭会话连接。"""
        self.stop_event.set()
        if self.thread:
            self.thread.join(3)
        self.evidence.stop(rebuild=False)
        self.session.set_meta("status", "stopped")
        self.session.set_meta(
            "simulation_metrics",
            {
                "taps": self.device.taps,
                "invalid": self.device.invalid,
                "duplicate": self.device.duplicate,
                "frames": self.frames,
            },
        )


class DemoWindow(MainWindow):
    """复用新工作台的单局模拟器；只替换设备与录像工厂，不查询或连接手机。"""

    def __init__(self, store=None):
        """先禁用所有自动设备入口，换成模拟闸门后再异步加载模型。"""
        super().__init__(store, auto_refresh=False, auto_load_model=False, auto_preview=False)
        self.input_gate = InputGate(transport="simulation")
        self.client = None
        self.setWindowTitle("王者荣耀自动斗地主机器人 · Doudizhu Lab · 离线模拟")
        self.available_devices = [Device("simulation", "device", "本地模拟")]
        self.selected_device = self.available_devices[0]
        self.log("本地模拟器已就绪；模型就绪后可以开始模拟牌局。")
        QTimer.singleShot(0, self.load_bot)

    def create_pipeline(self, serial: str):
        """工作线程返回单局模拟源；serial 不会传给 ADB，构造本身不执行动作。"""
        return DemoPipeline(self.store)

    def primary_action(self) -> None:
        """单局演示的显式开始按钮；先记录模拟源，再明确授权单局自动执行。"""
        if not self.bot or not self.bot.ready.is_set():
            self.message = "请等待所选模型就绪后开始模拟"
            self.render_dashboard()
            return
        if not self.recording:
            self.start_recording()
            self._demo_auto_requested = True

    def _launch_continuous(self) -> None:
        """模拟录像就绪后消费一次开始意图，自检通过 start_recording 保持仅建议。"""
        super()._launch_continuous()
        if getattr(self, "_demo_auto_requested", False) and self.recording and not self._stop_requested:
            self._demo_auto_requested = False
            self.set_bot_mode("auto")

    def stop_bot(self, reason: str = "用户停止连续任务") -> None:
        """同步取消模拟自动开始意图，再调用生产控制器撤销输入。"""
        self._demo_auto_requested = False
        super().stop_bot(reason)

    def _tick(self) -> None:
        """复用生产心跳；单局模拟自然结束时自动保存，无需连续页面识别。"""
        super()._tick()
        if self.recording and not self.continuous and not self._record_stopping:
            snapshot = self.recording.bot_snapshot()
            if snapshot and snapshot.state["ended"] and not hasattr(self, "native_check"):
                self.stop_and_save("模拟牌局已完成", outcome="completed")

    def render_dashboard(self) -> None:
        """复用布局，只更换模拟任务名称；不创建旧模式或调试按钮。"""
        super().render_dashboard()
        self.action_button.setText("开始模拟牌局")
        self.settings_button.setEnabled(False)


class NativeCheck(QObject):
    """原生 GUI 自检：所选真实模型配合模拟输入，逐阶段检查取消及末手确认。"""

    def __init__(self, window, application):
        super().__init__(window)
        self.window, self.application = window, application
        self.stage, self.started, self.history = "启动", monotonic(), []
        self.pipeline = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.timer.start(100)
        self.report = {
            "source": "simulation_native_ui",
            "platform": application.platformName(),
            "passed": False,
        }

    def advance(self, stage):
        self.history.append({"stage": stage, "seconds": monotonic() - self.started})
        self.stage = stage

    def tick(self):
        """所有控件访问均在 GUI 定时器内；失败仍先取消输入并收尾，不直接终止进程。"""
        w = self.window
        try:
            if monotonic() - self.started > 100 and self.stage != "收尾":
                raise ValueError("原生模拟窗口自检超时")
            if self.stage == "启动":
                w.start_recording()
                self.advance("加载")
            elif self.stage == "加载" and w.recording and w.recording.bot_snapshot():
                self.pipeline = w.recording
                w.load_bot()
                self.advance("仅建议")
            elif self.stage == "仅建议" and w.bot and w.bot.suggestion:
                if self.pipeline.device.taps:
                    raise ValueError("仅建议模式产生了输入")
                w.set_bot_mode("step")
                self.advance("单步")
            elif self.stage == "单步" and w.bot.executor.stage == "已确认" and w.bot.mode == "suggest":
                self.first_turn = self.pipeline.device.tracker.state.turn
                w.set_bot_mode("auto")
                self.advance("自动")
            elif (
                self.stage == "自动"
                and self.pipeline.device.tracker.state.turn > self.first_turn + 2
                and w.bot.executor.stage == "已确认"
            ):
                w.stop_bot()
                self.stopped_at, self.stopped_taps = monotonic(), self.pipeline.device.taps
                self.advance("停止核验")
            elif self.stage == "停止核验" and monotonic() - self.stopped_at > 0.6:
                if self.pipeline.device.taps != self.stopped_taps:
                    raise ValueError("取消后仍产生新的模拟点击")
                w.set_bot_mode("auto")
                self.advance("等待结算")
            elif (
                self.stage == "等待结算"
                and self.pipeline.device.tracker.state.ended
                and w.bot.mode == "suggest"
            ):
                if w.bot.executor.stage != "已确认":
                    raise ValueError("末手提交未由结算画面确认")
                self.report["passed"] = True
                self.begin_close()
            elif self.stage == "收尾" and w.recording is None and not w.busy:
                self.timer.stop()
                self.report.update(stages=self.history, duration_seconds=monotonic() - self.started)
                if self.pipeline:
                    self.report.update(
                        session=str(self.pipeline.session.folder),
                        taps=self.pipeline.device.taps,
                        invalid=self.pipeline.device.invalid,
                        duplicate=self.pipeline.device.duplicate,
                    )
                w.close()
                self.report["worker_stopped"] = not (w.bot and w.bot.thread.is_alive()) and not (
                    self.pipeline and self.pipeline.thread.is_alive()
                )
                self.report["passed"] &= self.report["worker_stopped"]
                output = w.store.root / "bot-evaluation/native-ui.json"
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(json.dumps(self.report, ensure_ascii=False, indent=2), encoding="utf8")
                self.application.exit(0 if self.report["passed"] else 2)
        except Exception as exc:
            self.report.update(passed=False, error=str(exc))
            self.begin_close()

    def begin_close(self):
        self.window.stop_bot()
        if self.window.recording and not self.window._record_stopping:
            self.window.stop_and_save()
        self.advance("收尾")


class ContinuousDemoWindow(DemoWindow):
    """连续流程原生演示，页面为示意观察；所有设备入口均固定为本地模拟源。"""

    def primary_action(self) -> None:
        """模拟连续任务使用生产启动及恢复流程，不调用任何真实手机后端。"""
        MainWindow.primary_action(self)

    def render_dashboard(self) -> None:
        """连续演示复用生产按钮状态，仅标记模拟来源。"""
        MainWindow.render_dashboard(self)
        self.settings_button.setEnabled(False)
        if self.action_button.text() == "开始连续游戏":
            self.action_button.setText("开始模拟任务")

    def create_pipeline(self, serial: str):
        """返回连续页面模拟源；工作线程调用，不创建真实手机客户端。"""
        from doudizhu_lab.simulation.continuous import ContinuousSimPipeline

        return ContinuousSimPipeline(self.store, threaded=True)


class ContinuousNativeCheck(QObject):
    """用生产窗口验证连续启动、异常暂停、明确恢复、立即停止和后台释放。"""

    def __init__(self, window, application):
        super().__init__(window)
        self.window, self.application = window, application
        self.stage, self.started, self.history = "启动", monotonic(), []
        self.pipeline = self.runtime = None
        self.report = {
            "source": "simulation_continuous_native_ui",
            "platform": application.platformName(),
            "passed": False,
        }
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.timer.start(100)

    def inputs(self):
        return len(self.pipeline.flow_commands) + sum(d.taps for d in self.pipeline.devices)

    def advance(self, value):
        self.stage = value
        self.history.append({"stage": value, "seconds": monotonic() - self.started})

    def tick(self):
        w = self.window
        try:
            if monotonic() - self.started > 100 and self.stage != "收尾":
                raise ValueError("连续原生窗口自检超时")
            if self.stage == "启动" and w.bot and w.bot.ready.is_set():
                w.primary_action()
                self.advance("等待出牌")
            elif self.stage == "等待出牌" and w.continuous and w.bot.executor.stage == "已确认":
                self.pipeline, self.runtime = w.recording, w.continuous
                w.continuous.pause("原生窗口自检暂停")
                self.before, self.paused_at = self.inputs(), monotonic()
                self.advance("暂停核验")
            elif self.stage == "暂停核验" and monotonic() - self.paused_at > 0.7:
                if self.inputs() != self.before:
                    raise ValueError("暂停后仍出现新输入")
                w.resume_continuous()
                self.advance("恢复")
            elif self.stage == "恢复" and w.continuous.active and self.inputs() > self.before:
                w.stop_bot()
                self.before, self.paused_at = self.inputs(), monotonic()
                self.advance("停止核验")
            elif self.stage == "停止核验" and monotonic() - self.paused_at > 0.7:
                if self.inputs() != self.before:
                    raise ValueError("立即停止后仍出现新输入")
                self.report["passed"] = True
                self.begin_close()
            elif self.stage == "收尾" and w.recording is None and not w.busy:
                self.timer.stop()
                w.close()
                self.report.update(
                    stages=self.history,
                    duration_seconds=monotonic() - self.started,
                    worker_stopped=not (w.bot and w.bot.thread.is_alive())
                    and not (self.pipeline and self.pipeline.thread.is_alive()),
                    continuous_stopped=not (self.runtime and self.runtime.thread.is_alive()),
                    bidding_mode=w.store.profile.bid_settings["mode"],
                    bidding_selector_locked=w.settings_locked if w.recording else True,
                )
                self.report["passed"] &= self.report["worker_stopped"] and self.report["continuous_stopped"]
                target = w.store.root / "continuous-evaluation/原生窗口检查.json"
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(json.dumps(self.report, ensure_ascii=False, indent=2), encoding="utf8")
                self.application.exit(0 if self.report["passed"] else 2)
        except Exception as exc:
            self.report.update(passed=False, error=str(exc))
            self.begin_close()

    def begin_close(self):
        self.window.stop_bot()
        if self.window.recording and not self.window._record_stopping:
            self.window.stop_and_save()
        self.advance("收尾")


def main() -> None:
    """在 GUI 主线程解析命令行并启动无需手机的原生模拟窗口。

    本函数不接收 Python 参数，模型、配置目录与演示选项从进程命令行读取；模型
    及配置使用指定本机目录，演示日志独立保存。参数错误或窗口事件循环结束通过
    SystemExit 返回相应退出码；演示仅操作隔离模拟设备，不查询或连接真实手机。
    """
    configure_console()
    parser = argparse.ArgumentParser(
        description="王者荣耀自动斗地主机器人 · Doudizhu Lab：本地离线模拟窗口，不连接手机"
    )
    parser.add_argument(
        "--version", action="version", version=f"王者荣耀自动斗地主机器人 · Doudizhu Lab {__version__}"
    )
    parser.add_argument("--data-dir", type=Path, help="演示数据与模型所在目录，不连接手机")
    parser.add_argument(
        "--play-model", choices=tuple(PLAY_MODEL_NAMES), help="本次模拟的出牌模型，不保存偏好"
    )
    parser.add_argument("--check", action="store_true", help="自动检查原生窗口模式、停止与关闭并导出报告")
    parser.add_argument("--continuous", action="store_true", help="使用从大厅开始的多局模拟后端")
    parser.add_argument(
        "--bidding", choices=("disabled", "model"), help="本次演示使用的叫抢策略，不保存自动授权"
    )
    args = parser.parse_args()
    app = QApplication([])
    app.setFont(QFont("Microsoft YaHei UI", 10))
    store = Store(args.data_dir)
    if args.play_model:
        store.profile.play_model = args.play_model
    if args.bidding:
        store.profile.bid_settings["mode"] = args.bidding
    window = (ContinuousDemoWindow if args.continuous else DemoWindow)(store)
    window.show()
    if args.check:
        app.setQuitOnLastWindowClosed(False)
        window.native_check = (ContinuousNativeCheck if args.continuous else NativeCheck)(window, app)
    raise SystemExit(app.exec())


if __name__ == "__main__":
    main()
