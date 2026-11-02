"""界面与纯机器人逻辑之间的后台协调器；取消和输入授权不依赖模型返回。"""

import hashlib
import threading
from dataclasses import asdict, dataclass
from time import monotonic

from doudizhu_lab.agents.douzero import Agent
from doudizhu_lab.agents.readiness import assess
from doudizhu_lab.domain.bot_types import encode
from doudizhu_lab.domain.models import RANK_LABELS
from doudizhu_lab.runtime.executor import ActionExecutor
from doudizhu_lab.storage.session import SessionStore


class InputGate:
    """人工和机器人共用的单次输入闸门；撤销立即生效，不等待在途 ADB 返回。

    generation 是授权代号，取消会使已排队任务失效。真正输入前再次核对；
    已跨过闸门的一个命令属于在途，不能承诺撤回。
    """

    def __init__(self, *, clock=monotonic, transport="video"):
        if transport not in ("video", "simulation"):
            raise ValueError("输入通道只支持实时视频或隔离模拟器")
        self.transport = transport
        self.clock = clock
        self.lock = threading.RLock()
        self.serial = threading.Lock()
        self.generation = 0
        self.owner = None
        self.blocked = False
        self.heartbeat_at = 0.0

    def grant(self, owner):
        """授予单一输入所有者，旧所有者的授权同时作废。"""
        with self.lock:
            self.generation += 1
            self.owner = None if self.blocked else owner
            return self.generation

    def revoke(self):
        """清除所有未开始输入的授权；不会阻塞 GUI 等待 ADB。"""
        return self.grant(None)

    def suspend(self) -> None:
        """立即锁住后续授权并撤销旧代号；不等待在途 USB，防止停止期间后台重新授予输入。"""
        with self.lock:
            self.blocked = True
            self.generation += 1
            self.owner = None

    def allow(self) -> None:
        """仅供用户显式开始或恢复时解除停止锁；解除本身不授予所有者，也不发送输入。"""
        with self.lock:
            self.blocked = False
            self.generation += 1
            self.owner = None

    def heartbeat(self):
        with self.lock:
            self.heartbeat_at = self.clock()

    def tap(self, owner, token, client, x, y, frame, *, guard=lambda: True, cancelled=lambda: False):
        """串行发送一个点按；回放来源在底层及这里双重拒绝，所有检验位于发送前。"""
        with self.serial:
            with self.lock:
                if token != self.generation or owner != self.owner or cancelled():
                    raise ValueError("输入授权已撤销")
                if owner in ("bot", "flow"):
                    if frame.transport != self.transport or self.clock() - self.heartbeat_at > 1.0:
                        raise ValueError("机器人输入需要实时视频和有效界面心跳")
                    if not guard():
                        raise ValueError("设备、方向、前台或局面已经变化")
            client.tap(x, y, frame, max_age=1.0 if owner in ("bot", "flow") else 3.0)


@dataclass(frozen=True)
class ModelStatus:
    """跨线程读取的模型加载结果；state 为 unloaded/loading/ready/failed，error 为中文原因。"""

    state: str
    error: str = ""


class BotRuntime:
    """单线程协调模型与执行器，Qt 只更新心跳、发模式命令及读取状态文字。

    推理期间也能立即撤销输入授权。线程内不持有或调用 QObject；状态变化通过
    普通 Python 数据返回 GUI。没有视频会话时允许加载模型，但没有设备输入能力。
    """

    def __init__(self, root, gate: InputGate, *, clock=monotonic, agent=None, threaded=True):
        self.root, self.gate = root, gate
        self.agent = agent or Agent(root / "models/douzero-wp")
        self.clock, self.threaded, self.owns_agent = clock, threaded, agent is None
        self.pipeline = None
        self.mode = "suggest"
        self.status = "模型未加载"
        self.suggestion = None
        self.stop_event = threading.Event()
        self.changed = threading.Event()
        self.lock = threading.RLock()
        self.generation = 0
        self.token = None
        self.executor = ActionExecutor(self._send, self._audit, clock=clock)
        self.thread = None
        self._game = ""
        self._decision_version = ""
        self._journal_session = None
        self._action_pipeline = None
        self._audit_session = None
        self.ready = threading.Event()
        self._model_status = ModelStatus("unloaded")
        self.wait_since = None
        self.pause_reason = ""

    def start(self):
        """后台加载模型；重复调用不会生成第二个推理进程或任务队列。"""
        if self.thread and self.thread.is_alive():
            return
        if self.stop_event.is_set():
            return
        with self.lock:
            self._model_status = ModelStatus("loading")
        self.ready.clear()
        if not self.threaded:
            try:
                self.agent.start()
            except Exception as exc:
                self._model_status = ModelStatus("failed", str(exc))
                raise
            self.ready.set()
            self._model_status = ModelStatus("ready")
            return
        self.thread = threading.Thread(target=self._run, name="机器人协调", daemon=True)
        self.thread.start()

    def model_status(self) -> ModelStatus:
        """返回线程安全的只读加载快照；GUI 可轮询，不等待推理，也不授予设备输入。"""
        with self.lock:
            return self._model_status

    def attach(self, pipeline):
        """切换会话只保留模型，立即撤销旧输入；自动模式不能跨会话恢复。"""
        with self.lock:
            if pipeline is self.pipeline:
                return
            self.set_mode("suggest")
            self.pipeline = pipeline
            self._game = self._decision_version = ""
            self._journal_session = None

    def set_mode(self, mode):
        """模式为 suggest/step/auto/stop；每次切换都使旧决策执行授权失效。"""
        if mode not in ("suggest", "step", "auto", "stop"):
            raise ValueError("机器人模式无效")
        with self.lock:
            self.gate.revoke()
            self.generation += 1
            self.mode = mode
            self.token = self.gate.grant("bot") if mode in ("step", "auto") else None
            self.changed.set()

    def arm_game(self, game_id):
        """连续任务仅为明确的新牌局授权；旧动作先撤销，不改变普通单局模式的结算停止行为。"""
        with self.lock:
            snapshot = self.pipeline.bot_snapshot() if self.pipeline else None
            if not snapshot or snapshot.state["game_id"] != game_id or snapshot.state["ended"]:
                raise ValueError("不能授权已结束或不属于当前快照的牌局")
            if snapshot.state["pending"] or snapshot.state["capture_incomplete"]:
                raise ValueError("不能授权存在历史缺口的牌局")
            self._game = game_id
            self.pause_reason, self.wait_since = "", None
            self.set_mode("auto")

    def _audit(self, record):
        pipeline = self._action_pipeline
        if pipeline is None:
            return
        # 机器人账本连接由协调线程独占。录制线程关闭自己的 SQLite 连接后，
        # 仍能保存“停止/在途未知”等最终审计，不能把停止日志写到下一会话。
        if self._audit_session is None or self._audit_session.folder != pipeline.session.folder:
            if self._audit_session:
                self._audit_session.close()
            self._audit_session = SessionStore(pipeline.session.folder)
        record["source"] = getattr(pipeline, "bot_source", "live_bot")
        decision = self.executor.decision
        record["strategy"] = decision.strategy if decision else "wp"
        record["model_id"] = decision.model_id if decision else self.agent.model_id
        record["configuration_version"] = 4
        record["config_id"] = hashlib.sha256(encode(pipeline.profile).encode()).hexdigest()
        self._audit_session.add_bot_action(record)
        snapshot = self.executor.last_snapshot
        if snapshot and record["stage"] in ("开始", "选牌已核对", "已确认", "暂停", "命令结果未知"):
            pipeline.evidence.submit(snapshot.frame, [record["action_id"]])

    def _send(self, kind, x, y, snapshot, action_id):
        pipeline, token = self.pipeline, self.token
        if pipeline is None or self.mode not in ("auto", "step") or token is None:
            raise ValueError("当前没有机器人输入授权")
        operating = getattr(pipeline, "operational", None)
        if operating and kind in ("play", "pass"):
            operating.note_submission(self.executor.decision, snapshot.frame.media_pts_us)

        def valid():
            newest = pipeline.bot_snapshot()
            raw = pipeline.frame
            rule_ready = (
                not newest or newest.state.get("strategy") != "rules" or assess(newest).status == "ready"
            )
            return bool(
                pipeline is self.pipeline
                and pipeline.allowed
                and not pipeline.stop_event.is_set()
                and newest
                and newest.version == snapshot.version
                and not newest.pending
                and rule_ready
                and raw
                and raw.serial == snapshot.frame.serial
                and raw.stream_id == snapshot.frame.stream_id
                and (raw.width, raw.height) == (snapshot.frame.width, snapshot.frame.height)
            )

        self.gate.tap(
            "bot",
            token,
            getattr(pipeline, "control_client", pipeline.client),
            x,
            y,
            snapshot.frame,
            guard=valid,
            cancelled=self.stop_event.is_set,
        )

    def _run(self):
        try:
            self.status = "正在加载三种身份的 WP 模型…"
            self.agent.start()
            self.ready.set()
            with self.lock:
                self._model_status = ModelStatus("ready")
            self.status = "模型就绪 · 等待完整开局和自己的回合"
            while not self.stop_event.wait(0.03):
                self.pulse()
        except Exception as exc:
            self.gate.revoke()
            self.mode = "stop"
            self.status = f"机器人暂停：{exc}"
            self.pause_reason = self.status
            with self.lock:
                self._model_status = ModelStatus("failed", str(exc))
        finally:
            self.ready.clear()
            self.gate.revoke()
            if self.owns_agent:
                self.agent.close()
            if self._audit_session:
                self._audit_session.close()
                self._audit_session = None
            if self.stop_event.is_set():
                with self.lock:
                    self._model_status = ModelStatus("unloaded")

    def pulse(self):
        """处理一轮最新观察；生产线程与确定性模拟共用，不在模拟中复制决策执行逻辑。"""
        check_running = getattr(self.agent, "check_running", None)
        if check_running:
            check_running()
        if self.changed.is_set():
            self.changed.clear()
            if self.executor.active:
                self.executor.cancel("用户切换模式或人工接管")
            self._decision_version = ""
        pipeline = self.pipeline
        if pipeline is None:
            return
        snapshot = pipeline.bot_snapshot()
        if self._journal_session != pipeline.session.folder:
            pipeline.session.set_meta(
                "bot",
                {
                    "model_id": self.agent.model_id,
                    "profile": asdict(pipeline.profile),
                    "source": getattr(pipeline, "bot_source", "live_bot"),
                    "initial_mode": "suggest",
                },
            )
            self._journal_session = pipeline.session.folder
        if snapshot is None:
            return
        state = snapshot.state
        if self._game and state["game_id"] != self._game:
            if self.executor.active:
                self.executor.cancel("已切换牌局，旧动作不再发送输入")
            self.set_mode("suggest")
            self.status = "已进入新牌局，请明确选择执行模式"
            self._game = state["game_id"]
            return
        if state["ended"]:
            self._finish_game(snapshot)
            return
        if state["game_id"]:
            self._game = state["game_id"]
        if self.mode == "stop":
            self.status = "已停止 · 可人工接管"
            return
        if self.mode in ("auto", "step") and (
            self.clock() - self.gate.heartbeat_at > 1.0
            or not pipeline.allowed
            or pipeline.stop_event.is_set()
        ):
            self.executor.cancel("界面心跳、视频或游戏前台已失效")
            self.status = self.executor.reason
            self.set_mode("stop")
            return
        if self.executor.active:
            self.executor.advance(snapshot)
            self.status = f"{self.executor.stage} · {self.executor.reason}"
            if self.executor.stage == "暂停":
                self.set_mode("stop")
            elif self.executor.stage == "已确认" and self.mode == "step":
                self.set_mode("suggest")
            if self.executor.stage == "已确认":
                operating = getattr(pipeline, "operational", None)
                if operating:
                    operating.note_confirmation(self.executor.decision)
            return
        readiness = assess(snapshot)
        if readiness.status != "ready":
            self.status = readiness.reason
            if readiness.bounded:
                self.wait_since = self.wait_since or self.clock()
            else:
                self.wait_since = None
            if readiness.status == "halt" or (
                self.wait_since is not None
                and self.clock() - self.wait_since
                > (6 if pipeline.profile.continuous_policy != "strict" else 3)
            ):
                if self.mode in ("auto", "step"):
                    self.pause_reason = self.status
                    self.set_mode("stop")
            return
        self.wait_since = None
        if snapshot.version != self._decision_version:
            generation = self.generation
            if state.get("strategy") == "rules":
                from doudizhu_lab.agents.rules import decide as rule_decide

                self.suggestion = rule_decide(snapshot)
            else:
                self.suggestion = self.agent.decide(snapshot)
            self._decision_version = snapshot.version
            if generation != self.generation:
                return
        self.status = f"{'规则接管' if self.suggestion.strategy == 'rules' else 'WP建议'}：{' '.join(RANK_LABELS.get(c, c) for c in self.suggestion.cards) or '不出'} · {self.suggestion.elapsed_ms:.1f} 毫秒"
        if self.mode in ("auto", "step"):
            newest = pipeline.bot_snapshot()
            if newest and newest.version == self.suggestion.version:
                self._action_pipeline = pipeline
                self.executor.settings.update(
                    {k: v for k, v in pipeline.profile.bot_settings.items() if k in self.executor.settings}
                )
                self.executor.start(self.suggestion, newest)

    def _finish_game(self, snapshot):
        """结算帧可同时包含末手成功证据，先完成反馈确认，再停止整局自动任务。"""
        if self.executor.active:
            if self.executor.stage == "等待反馈":
                self.executor.advance(snapshot)
            if self.executor.active:
                self.executor.cancel("已结算，尚未确认的动作留待人工核对")
        if self.executor.stage == "已确认":
            operating = getattr(self.pipeline, "operational", None)
            if operating:
                operating.note_confirmation(self.executor.decision)
        if self.mode not in ("suggest", "stop"):
            self.set_mode("suggest")
        self.status = "本局已结束，自动执行已停止"
        self._game = snapshot.state["game_id"]

    def close(self):
        """先撤销输入，再停止后台线程和模型进程；GUI 关闭不得遗留自动操作。"""
        self.gate.revoke()
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=4)
            if self.thread.is_alive():
                self.agent.interrupt()
                self.thread.join(timeout=3)
        if not self.threaded:
            if self.owns_agent:
                self.agent.close()
            if self._audit_session:
                self._audit_session.close()
                self._audit_session = None
        self.ready.clear()
        with self.lock:
            self._model_status = ModelStatus("unloaded")
