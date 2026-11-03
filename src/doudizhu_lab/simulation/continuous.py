"""完整流程模拟源：真实规则及所选模型推理，共用生产协调器，不构造ADB客户端。"""

import json
from collections import Counter
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from queue import Queue
from threading import Event, RLock, Thread
from time import monotonic
from uuid import uuid4

import numpy as np

from doudizhu_lab.agents.backends import create_agent
from doudizhu_lab.agents.douzero import ROLES
from doudizhu_lab.domain.bot_types import encode
from doudizhu_lab.domain.models import Frame, PixelRect
from doudizhu_lab.domain.table import TableObservation
from doudizhu_lab.runtime.bot import BotRuntime, InputGate
from doudizhu_lab.runtime.continuous import ContinuousRuntime
from doudizhu_lab.simulation.bot import SimulatedDevice
from doudizhu_lab.storage.runs import evaluate_run
from doudizhu_lab.storage.session import SessionStore
from doudizhu_lab.vision.flow import FlowButton, FlowObservation


class EvidenceSink:
    """模拟源不制造假录像或假识别截图；只保留语义审计链接。"""

    def submit(self, *_):
        return True


class ContinuousSimPipeline:
    """从大厅开始的本地设备；点击只按当前坐标命中，不接受机器人目标牌作为结果。

    图像是示意底色，FlowObservation和手牌布局由模拟设备提供，因此此模拟验证流程、
    规则和输入协调，不证明真实像素识别性能。threaded用于原生窗口，虚拟时钟用于批量测试。
    """

    bot_source = "simulation_bot"
    serial = "simulation"

    def __init__(self, store, *, clock=monotonic, threaded=False, seed_start=91000, session_root=None):
        """创建隔离模拟；session_root 仅供进程集成测试指定临时会话根目录，不连接手机。"""
        self.store, self.clock, self.threaded = store, clock, threaded
        self._started = clock()
        self.profile = deepcopy(store.profile)
        if not self.profile.seat_order:
            self.profile.seat_order = ["self", "right", "left"]
        self.session = SessionStore.create(
            session_root or store.root / "continuous-simulation", self.profile, [], "simulation"
        )
        self.session.set_meta("source", "simulation_continuous")
        self.session.set_meta("note", "流程示意观察；未采集手机视频，也未检验真实像素识别")
        self.evidence, self.client, self.allowed = EvidenceSink(), self, True
        self.stop_event, self.lock = Event(), RLock()
        self.write_queue, self.thread = Queue(), None
        self.seed_start, self.device, self.game_number = seed_start, None, 0
        self.video_id = "simulation-" + uuid4().hex[:8]
        self.phase, self.phase_at = "lobby", self.clock()
        self.last_action_at = self.clock()
        self.pts, self.frames, self.latest, self.flow, self.frame = 0, 0, None, None, None
        self.flow_commands, self.devices = [], []
        self.recorded, self.events = set(), []
        self.fault = ""
        self.bid_enabled = self.profile.bid_settings["mode"] == "model"
        self.bid_prompt, self.bid_wait_until = "call", 0
        self.bid_dealt, self.bid_hand, self.bid_deal_id = None, (), ""
        self.ending_deadline, self.after_ending = 0, "settlement"
        from doudizhu_lab.runtime.operational import OperationalObserver

        self.operational = OperationalObserver(self.profile)

    def enable_flow(self):
        """模拟源已经提供流程观察，无需OCR依赖；不会下载模型或连接手机。"""

    def start(self):
        self.tick()
        if self.threaded:
            self.thread = Thread(target=self._run, name="连续流程模拟源", daemon=True)
            self.thread.start()

    def _run(self):
        while not self.stop_event.wait(0.12):
            self.tick()

    def _phase(self, value):
        self.phase, self.phase_at = value, self.clock()

    def tick(self):
        """推进新画面和对手动作，再经生产GameTracker生成公开观察；不读取机器人意图。"""
        with self.lock:
            if self.stop_event.is_set():
                return
            self.frames += 1
            self.pts += 120_000
            if self.phase == "matching" and self.clock() - self.phase_at >= 0.36:
                self.game_number += 1
                role = ROLES[(self.seed_start - 91000 + self.game_number - 1) % 3]
                self.device = SimulatedDevice(
                    self.seed_start + self.game_number - 1,
                    role,
                    stream_id=f"{self.video_id}-g{self.game_number}",
                    pts_offset=self.pts,
                )
                self.devices.append(self.device)
                if self.bid_enabled:
                    self.bid_dealt = {
                        seat: list(
                            (
                                Counter(self.device.initial[r])
                                - (Counter(self.device.bottom) if r == "landlord" else Counter())
                            ).elements()
                        )
                        for seat, r in self.device.roles.items()
                    } | {"bottom": self.device.bottom.copy()}
                    self.bid_hand = tuple(self.device.setup_frames[0].observation["hand"])
                    self.bid_deal_id = f"{self.video_id}-deal-{self.game_number}"
                    self.bid_prompt = "call" if self.game_number % 2 else "rob"
                    self.bid_wait_until = 0
                if self.operational and not self.bid_enabled:
                    # 双轨观察器必须看到相同的发牌/叫抢前缀，不能从出牌阶段中途初始化。
                    for setup in self.device.setup_frames:
                        setup = self._bind_frame(setup)
                        self.session.add_observation(setup.observation)
                        self.operational.transform(setup)
                    operating_events, notices = self.operational.drain()
                    for event in operating_events:
                        self.session.add_operational_event(event)
                    for notice in notices:
                        self.session.add_recovery(notice)
                self.pts = self.device.pts
                self._phase("dealing")
            elif self.phase == "dealing" and self.clock() - self.phase_at >= 0.36:
                self._phase("bidding")
            elif self.phase == "ending" and self.clock() >= self.ending_deadline:
                self._phase(self.after_ending)
            if (
                self.phase == "playing"
                and not self.device.env.game_over
                and self.device.env.acting_player_position != self.device.roles["self"]
                # 至少保留四帧供推断及确认；按整数微秒比较，避免时钟量级改变浮点舍入。
                and round((self.clock() - self.last_action_at) * 1_000_000) >= 480_000
            ):
                legal = self.device.env.game_infoset.legal_actions
                if self.fault == "hidden_play" and any(legal):
                    self.device.hidden_play_until_us = self.pts + 4_000_000
                    self.device.act(next(a for a in legal if a))
                    self.fault = ""
                else:
                    self.device.act(self.device.rng.choice(legal))
                self.last_action_at = self.clock()
            if self.device and self.phase in ("playing", "settlement", "showcase", "ending"):
                if self.device.env.game_over:
                    if self.phase == "playing":
                        self.after_ending = (
                            "showcase" if self.bid_enabled and self.game_number % 2 else "settlement"
                        )
                        self.ending_deadline = self.clock() + 4
                        self._phase("ending" if self.bid_enabled else "settlement")
                self.device.now, self.device.pts = self.clock() - 0.12, self.pts - 120_000
                snap = self.device.render()
                self.latest = self._bind_frame(snap)
                self.frame = self.latest.frame
                from doudizhu_lab.domain.inference import frame_context

                raw_observation = self.latest.observation
                raw_observation["frame_context"] = frame_context(self.latest.frame)
                raw_observation["frame_context"]["round_id"] = snap.state["game_id"]
                self.latest = replace(self.latest, observation_json=encode(raw_observation))
                self.session.add_observation(raw_observation)
                if self.operational:
                    # 模拟已知UI契约：领出只有出牌按钮；接口仍只读取公开局面。
                    if (
                        self.latest.observation["actor"] == "self"
                        and [] not in self.device.env.game_infoset.legal_actions
                    ):
                        self.latest = replace(
                            self.latest, buttons=tuple(b for b in self.latest.buttons if b.kind == "play")
                        )
                    self.latest = self.operational.transform(self.latest)
                    operational_events, notices = self.operational.drain()
                    for event in operational_events:
                        self.session.add_operational_event(event)
                    for notice in notices:
                        self.session.add_recovery(notice)
                for event in snap.history:
                    if event["event_id"] not in self.recorded:
                        self.recorded.add(event["event_id"])
                        self.session.add_event(event)
                        from doudizhu_lab.domain.tracker import GameEvent

                        self.events.append(GameEvent(**event))
            else:
                rgb = np.full((100, 600, 3), (21, 83, 73), np.uint8)
                self.frame = Frame(
                    self.serial,
                    rgb,
                    self.clock(),
                    str(self.pts),
                    0,
                    "simulation",
                    f"{self.video_id}-page-{self.frames}",
                    self.video_id,
                    self.pts,
                )
            specs = {
                "lobby": ("match", "开始匹配", PixelRect(450, 65, 100, 25)),
                "bidding": ("decline_bid", "不抢", PixelRect(250, 45, 100, 25)),
                "doubling": ("decline_double", "不加倍", PixelRect(250, 45, 100, 25)),
                "settlement": ("continue", "再来一局", PixelRect(450, 65, 100, 25)),
                "showcase": ("close_showcase", "继续", PixelRect(350, 65, 100, 25)),
            }
            spec = specs.get(self.phase)
            buttons = (FlowButton(*spec, True),) if spec else ()
            if self.phase == "settlement":
                buttons += (FlowButton("return_lobby", "返回大厅", PixelRect(150, 65, 100, 25), True),)
            if self.bid_enabled and self.phase == "bidding":
                if self.clock() < self.bid_wait_until:
                    buttons = ()
                else:
                    kind = "call_landlord" if self.bid_prompt == "call" else "rob_landlord"
                    label = "叫地主" if self.bid_prompt == "call" else "抢地主"
                    buttons = (
                        FlowButton(
                            "decline_bid",
                            "不叫" if self.bid_prompt == "call" else "不抢",
                            PixelRect(180, 45, 100, 25),
                            True,
                        ),
                        FlowButton(kind, label, PixelRect(350, 45, 100, 25), True),
                    )
                self.session.add_observation(
                    TableObservation(self.frame.frame_id, self.pts, phase="bidding", hand=list(self.bid_hand))
                )
            self.flow = FlowObservation(
                self.frame,
                self.phase,
                buttons,
                bid_hand=self.bid_hand if self.bid_enabled and self.phase == "bidding" else None,
                bid_deal_id=self.bid_deal_id if self.bid_enabled else "",
            )

    def _finish_bidding(self, accepted):
        """根据实际肯定/拒绝点击分配地主；原始17张牌与底牌保持不变。"""
        landlord = "self" if accepted else ("right" if self.game_number % 2 else "left")
        self.device = SimulatedDevice(
            self.seed_start + self.game_number - 1,
            "landlord",
            stream_id=f"{self.video_id}-g{self.game_number}",
            pts_offset=self.pts,
            dealt=self.bid_dealt,
            landlord_seat=landlord,
        )
        self.devices[-1] = self.device
        self.pts = self.device.pts
        for setup in self.device.setup_frames:
            setup = self._bind_frame(setup)
            self.session.add_observation(setup.observation)
            if self.operational:
                self.operational.transform(setup)
        if self.operational:
            events, notices = self.operational.drain()
            for event in events:
                self.session.add_operational_event(event)
            for notice in notices:
                self.session.add_recovery(notice)
        self._phase("doubling")

    def bot_snapshot(self):
        with self.lock:
            return self.latest

    def _bind_frame(self, snapshot):
        """模拟一次真实视频流中的多局画面：编码流不随牌局重建，原始分局标记单独绑定。"""
        from doudizhu_lab.domain.inference import frame_context

        frame = replace(snapshot.frame, stream_id=self.video_id)
        observation = snapshot.observation
        observation["frame_context"] = frame_context(frame) | {"round_id": snapshot.state["game_id"]}
        return replace(snapshot, frame=frame, observation_json=encode(observation))

    def flow_snapshot(self):
        with self.lock:
            return self.flow

    def tap(self, x, y, frame, *, max_age=1):
        """统一模拟输入出口；禁止任何真实来源，只有命中当前可用对象才改变页面或牌局。"""
        with self.lock:
            if (
                frame.transport != "simulation"
                or frame.serial != self.serial
                or not 0 <= self.clock() - frame.started <= max_age
                or self.stop_event.is_set()
            ):
                raise ValueError("模拟输入来源、时效或设备状态无效")
            if self.fault == "disconnect":
                raise OSError("模拟USB中断")
            button = next(
                (
                    b
                    for b in self.flow.buttons
                    if b.hit.x <= x < b.hit.x + b.hit.width and b.hit.y <= y < b.hit.y + b.hit.height
                ),
                None,
            )
            if button:
                self.flow_commands.append((self.phase, button.kind))
                if self.fault == "no_feedback":
                    return
                if button.kind == "match":
                    self._phase("matching")
                elif self.bid_enabled and button.kind in ("decline_bid", "call_landlord", "rob_landlord"):
                    if button.kind == "call_landlord":
                        # 对手随后抢地主，提供一次独立回抢机会；空按钮期不等于新的发牌。
                        self.bid_prompt = "rob"
                        self.bid_wait_until = self.clock() + 0.72
                    else:
                        self._finish_bidding(button.kind == "rob_landlord")
                elif button.kind == "decline_bid":
                    self._phase("dealing" if self.fault == "redeal" else "doubling")
                    self.fault = ""
                elif button.kind == "decline_double":
                    self._phase("playing")
                elif button.kind in ("continue", "return_lobby"):
                    self._phase("lobby")
                elif button.kind == "close_showcase":
                    self._phase("settlement")
                return
            if self.phase != "playing":
                return
            state = self.latest.state
            before = self.device.env.acting_player_position
            self.device.send("select", x, y, self.device.last, f"{state['game_id']}-{state['turn']}")
            if self.device.env.acting_player_position != before or self.device.env.game_over:
                self.last_action_at = self.clock()

    def poll(self):
        with self.lock:
            events, self.events = self.events, []
            return {
                "frame": self.frame,
                "observation": TableObservation(**self.latest.observation) if self.latest else None,
                "recognition": None,
                "state": self.device.tracker.state if self.device else None,
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
        self.stop_event.set()
        if self.thread:
            self.thread.join(3)
        self.session.set_meta("status", "stopped")


def simulate_continuous(store, *, runs=6, output=None):
    """用真实模型执行6个五局连续任务；模型输入保持生产公开信息接口，所有来源标为模拟。"""
    agent = create_agent(store.root, store.profile.play_model)
    agent.start()
    reports = []
    try:
        for number in range(runs):
            now = [monotonic()]

            def clock():
                return now[0]

            pipeline = ContinuousSimPipeline(store, clock=clock, seed_start=91000 + number * 5)
            pipeline.start()
            gate = InputGate(clock=clock, transport="simulation")
            bot = BotRuntime(store.root, gate, clock=clock, agent=agent, threaded=False)
            runtime = ContinuousRuntime(store.root, gate, bot, pipeline, clock=clock, threaded=False)
            runtime.start()
            try:
                for _ in range(18000):
                    now[0] += 0.12
                    pipeline.tick()
                    gate.heartbeat()
                    bot.pulse()
                    runtime.advance(pipeline.flow_snapshot(), pipeline.bot_snapshot())
                    if runtime.status != "running":
                        break
                if runtime.active:
                    runtime.pause("模拟步数超限")
                report = evaluate_run(runtime.store.folder)
                report["run_folder"] = str(runtime.store.folder)
                report["roles"] = [d.roles["self"] for d in pipeline.devices]
                report["invalid_submissions"] = sum(d.invalid for d in pipeline.devices)
                report["device_duplicate_submissions"] = sum(d.duplicate for d in pipeline.devices)
                report["bidding_commands"] = dict(
                    Counter(
                        k
                        for _, k in pipeline.flow_commands
                        if k in ("call_landlord", "rob_landlord", "decline_bid", "decline_double")
                    )
                )
                reports.append(report)
                print(
                    f"模拟任务{number + 1}/{runs}：{len(runtime.completed)}局，{runtime.status}，{runtime.reason}",
                    flush=True,
                )
            finally:
                runtime.close()
                bot.close()
                pipeline.stop()
                pipeline.session.close()
    finally:
        agent.close()
    roles = Counter(r for row in reports for r in row["roles"])
    model_bidding = store.profile.bid_settings["mode"] == "model"
    result = {
        "source": "simulation_continuous",
        "runs": reports,
        "roles": dict(roles),
        "passed": runs >= 6
        and all(
            r["passed"] and not r["invalid_submissions"] and not r["device_duplicate_submissions"]
            for r in reports
        )
        and all(roles[r] >= (1 if model_bidding else 10) for r in ROLES),
        "bidding_mode": store.profile.bid_settings["mode"],
        "bidding_commands": dict(sum((Counter(r["bidding_commands"]) for r in reports), Counter())),
        "identity_note": "模型叫抢时身份由实际叫抢决定，不强制每种身份10局"
        if model_bidding
        else "固定三种身份各至少10局",
        "play_model": store.profile.play_model,
        "note": "生产连续协调器、单局协调器和执行器，共用所选真实模型权重；未验证真实像素和手机输入。",
    }
    target = Path(output) if output else store.root / "continuous-evaluation/模拟连续任务.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf8")
    return result
