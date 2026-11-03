"""连续流程协调：页面事务与单局出牌共享输入闸门，任何未知结果都不盲目重试。"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from threading import Event, RLock, Thread
from time import monotonic
from uuid import uuid4

from doudizhu_lab.agents.bidding import BidAgent, BiddingController, decision_dict, verify_bid_model
from doudizhu_lab.devices.adb import InputRejected
from doudizhu_lab.runtime.faults import FaultScope, RuntimeFault
from doudizhu_lab.storage.runs import RunStore, evaluate_run
from doudizhu_lab.vision.flow import SCENE_NAMES


@dataclass(frozen=True)
class ContinuousProgress:
    """任务界面的只读进度；elapsed 为已运行秒数，不代表任务截止时间。

    completed 只统计已确认完成的牌局，total 为无业务上限的 Python 正整数。
    startup_phase 为 waiting_lobby 时等待大厅或结算续局入口，该取值保留旧接口兼容。
    大厅按钮稳定或结算续局提交得到页面反馈后才进入 ready 并授权新局流程；默认
    ready 兼容旧调用。等待仍属于 running，GUI 保留停止按钮。
    skipped 为异常跳过局数，retries 为工作会话恢复次数，retry_at 为下次自动检查的
    单调时钟时间或 None；这些展示字段不会恢复历史执行授权。
    """

    status: str
    scene: str
    reason: str
    completed: int
    total: int
    elapsed: float
    startup_phase: str = "ready"
    skipped: int = 0
    retries: int = 0
    retry_at: float | None = None


class ContinuousRuntime:
    """一个执行会话的连续流程；支持注入时钟及无后台线程的确定性模拟。

    pipeline 必须已录制，并提供同帧 flow_snapshot/bot_snapshot；bot 是生产单局协调器。
    start 不授予跨重启权限。暂停只撤销后续输入；已发命令保留在途记录，不能撤回。
    lobby_confirmed 仅记录本任务观察过有效大厅按钮，startup_entry 记录首次确认的新局
    入口。结算续局须收到页面推进反馈后才授权下一局；不读取旧 mode_verified 字段，
    不按玩法名称或场次等级授权；完整开局、稳定帧和提交确认仍由生产流程校验。
    unattended 会话由父任务看护器管理；替代进程创建新会话，累计进度由父账本保存。
    """

    def __init__(
        self,
        root,
        gate,
        bot,
        pipeline,
        *,
        clock=monotonic,
        threaded=True,
        bid_agent=None,
        unattended=False,
        commit=None,
        blocked_flow=None,
    ):
        """创建单次执行会话；unattended 由任务看护器显式启用。

        commit 在流程线程同步确认父任务账本落盘，失败时不得继续匹配；blocked_flow
        为前一个执行进程未确认的页面命令，只能观察页面推进后归档，不能重发。
        旧调用默认保留显式暂停行为，模拟与离线校验不隐式授予无人值守权限。
        """
        self.root, self.gate, self.bot, self.pipeline = root, gate, bot, pipeline
        self.clock, self.threaded = clock, threaded
        self.lock, self.stop_event = RLock(), Event()
        self.thread = None
        self.status, self.reason, self.scene = "idle", "", "unknown"
        self.store = None
        self.settings = dict(pipeline.profile.continuous_settings)
        self.completed, self.registered = [], set()
        self.current_game, self.armed_game = "", ""
        self.last_flow, self.pending_flow = None, None
        self.last_id, self.last_new_at = "", 0.0
        self.scene_epoch, self.sent_opportunities = 0, set()
        self._last_known_scene = ""
        self._seen_deal = ""
        self.candidate = None
        self.started = self.scene_since = self.clock()
        self.identity = None
        # 仅记录是否观察到有效大厅入口，不验证玩法名称或场次等级，也不跨任务恢复。
        self.lobby_confirmed = False
        self.startup_entry = ""
        self.startup_phase = "waiting_lobby"
        self._finish_wait = None
        self._ready_since = None
        self.close_window_requested = False
        self.stop_recording_requested = False
        self._stop_request_handled = False
        self.bid_agent, self.bidding = bid_agent, None
        self.bid_decision, self.bid_description = None, ""
        self._stop_intent = None
        self._progress_cache = ContinuousProgress("idle", "unknown", "", 0, self.settings["max_games"], 0)
        self.unattended, self.commit = unattended, commit
        self.fault = None
        self.quarantined = False
        self.skipped = set()
        self.blocked_flow = blocked_flow
        self._boundary_probe = None

    @property
    def active(self):
        return self.status == "running" and not self.stop_event.is_set()

    @property
    def entry_confirmed(self) -> bool:
        """本任务是否已确认新局入口；只允许有效大厅或收到推进反馈的结算续局。

        识别出结算页、点击返回大厅或关闭战绩展示都不能单独授权出牌。入口授权仅在
        本任务内存中有效，旧观察字段和历史账本不能恢复权限；新局仍须验证完整开局。
        供流程线程持锁读取，不发送输入；GUI 通过 progress 获取只读进度快照。
        """
        return self.lobby_confirmed or self.startup_entry == "settlement"

    def progress(self) -> ContinuousProgress:
        """返回只读局数与已用秒数，不发布任务时限；目标局数保持 Python 整数精度。

        设备操作持锁时返回上次快照，不让 GUI 等待 USB 或账本写入，不改变执行授权。
        """
        if not self.lock.acquire(blocking=False):
            return self._progress_cache
        try:
            self._progress_cache = ContinuousProgress(
                self.status,
                self.scene,
                self.reason,
                len(self.completed),
                self.settings["max_games"],
                max(0, self.clock() - self.started),
                self.startup_phase,
                len(self.skipped),
            )
            return self._progress_cache
        finally:
            self.lock.release()

    @property
    def description(self):
        """供GUI直接显示的中文进度；不在后台线程访问任何Qt对象。"""
        label = {
            "idle": "未开始",
            "running": "运行中",
            "paused": "已暂停",
            "completed": "已完成",
            "stopped": "已停止",
        }[self.status]
        return (
            f"连续任务：{label} · {SCENE_NAMES[self.scene]} · {len(self.completed)}/{self.settings['max_games']}局 · {max(0, self.clock() - self.started):.0f}秒"
            + (f" · {self.reason}" if self.reason else "")
            + (f" · {self.bid_description}" if self.scene == "bidding" and self.bid_description else "")
        )

    def start(self):
        """在已有录像上启动并预热；实际页面输入必须等待模型、前台与稳定观察全部就绪。"""
        with self.lock:
            if self.status != "idle":
                raise ValueError("任务已创建；暂停后请明确恢复，重新运行请创建新任务")
            self.store = RunStore.create(self.root, self.pipeline)
            if hasattr(self.pipeline, "enable_automatic_retention"):
                self.pipeline.enable_automatic_retention()
            self.status, self.started = "running", self.clock()
            self.reason = "正在确认大厅或结算页入口"
            self.store.set("startup_phase", self.startup_phase)
            self.store.add("startup_wait_started", at=self.started)
            self.scene_since = self.last_new_at = self.started
            self.bot.attach(self.pipeline)
            self.bot.set_mode("suggest")
            self.bot.start()
            settings = self.pipeline.profile.bid_settings
            agent = self.bid_agent
            if settings["mode"] == "model" and agent is None:
                agent = BidAgent(self.root / "models/bidding")
                try:
                    manifest = verify_bid_model(agent.folder)
                    self.pipeline.session.snapshot_bidding_model(agent.folder)
                    self.store.set("bidding_model", manifest)
                except (ValueError, OSError) as exc:
                    self.store.add("bidding_unavailable", reason=str(exc))
            self.bidding = BiddingController(settings, agent, threaded=self.threaded)
            if self.threaded:
                self.thread = Thread(target=self._run, name="连续斗地主流程", daemon=True)
                self.thread.start()
            else:
                self.pipeline.enable_flow()

    def _run(self):
        try:
            if hasattr(self.pipeline, "stage_started"):
                self.pipeline.stage_started("流程识别初始化")
            self.pipeline.enable_flow()
            if hasattr(self.pipeline, "stage_finished"):
                self.pipeline.stage_finished("流程识别初始化")
            while not self.stop_event.wait(0.04):
                self.poll_stop_request()
                if self.stop_event.is_set():
                    break
                if self.active:
                    if hasattr(self.pipeline, "stage_started"):
                        self.pipeline.stage_started("连续流程处理")
                    try:
                        self.advance(self.pipeline.flow_snapshot(), self.pipeline.bot_snapshot())
                    finally:
                        if hasattr(self.pipeline, "stage_finished"):
                            self.pipeline.stage_finished("连续流程处理")
        except Exception as exc:
            self.pause(f"连续任务异常：{exc}")
        finally:
            self.gate.revoke()
            if self.store:
                evaluate_run(self.store.folder)

    def poll_stop_request(self):
        """应用自己的只减权停止接口；已完成任务也允许GUI轮询后正常关闭，不发送手机输入。"""
        if self.store is None or self._stop_request_handled:
            return
        request = self.store.folder / "停止请求.json"
        if request.is_file():
            import json

            value = json.loads(request.read_text(encoding="utf8"))
            self._stop_request_handled = True
            self.close_window_requested = bool(value.get("close_window"))
            self.stop_recording_requested = True
            self.stop("命令行请求停止连续任务")

    def pause(self, reason, *, manual=False, scope=FaultScope.SESSION, code="runtime_failure"):
        """立即撤销流程及出牌授权；系统暂停固定故障，显式用户暂停清理当前窗口。

        manual为原因类型，不通过中文文案推断。任何此前已固定故障优先保留，恢复
        仍须原有可信状态检查；本接口不隐式授权新一局。
        unattended 下非人工故障按 scope 分流：ROUND 隔离本局并继续观察，SESSION
        请求父进程重建，EXTERNAL 请求只读探测；code 为稳定故障代码，不解析 reason。
        """
        with self.lock:
            if self.status != "running":
                return
            if self.unattended and not manual:
                self.gate.revoke()
                self.bot.set_mode("stop")
                if scope == FaultScope.ROUND:
                    self._skip_round(code, reason)
                    return
                self.fault = RuntimeFault(code, reason, scope)
                self.status = "waiting_external" if scope == FaultScope.EXTERNAL else "recovering"
                self.reason = reason
                self.store.set("status", self.status)
                self.store.add("recovery_needed", code=code, scope=scope, reason=reason)
                return
            self.gate.revoke()
            self.bot.set_mode("stop")
            if manual and hasattr(self.pipeline, "abort_recording_window"):
                self.pipeline.abort_recording_window()
            elif not manual and hasattr(self.pipeline, "report_failure"):
                self.pipeline.report_failure("continuous_pause", reason)
            self.status, self.reason = "paused", reason
            self.store.set("status", self.status)
            self.store.add("paused", reason=reason, game_id=self.current_game, at=self.clock())
            if self.last_flow:
                self.pipeline.evidence.submit(self.last_flow.frame, ["continuous-pause"])

    def _commit(self, event_kind, **value):
        """先写入执行账本，再等待父任务持久确认；本接口不直接发送设备输入。"""
        self.store.add(event_kind, **value)
        if self.commit:
            self.commit(event_kind, value)

    def _skip_round(self, code, reason):
        """隔离一次异常牌局并固定录像；重复观察只登记一次，保持流程观察线程运行。"""
        if not self.quarantined:
            self._boundary_probe = None
        self.quarantined = True
        self.reason = reason + "；等待明确结算后自动续局"
        game = self.current_game or self._seen_deal
        if not game or game in self.skipped:
            return
        self.skipped.add(game)
        pts = self.last_flow.frame.media_pts_us if self.last_flow else 0
        if hasattr(self.pipeline, "pin_game"):
            self.pipeline.pin_game(game, code, reason, pts)
        self._commit("game_skipped", game_id=game, code=code, reason=reason, pts_us=pts)

    def _quarantine_boundary(self, flow):
        """仅以新鲜双帧终局入口结束隔离；动画、未知画面及旧局跟踪缺口不是边界。"""
        reliable = (
            flow.scene in ("lobby", "room")
            and any(b.kind == "match" and b.enabled for b in flow.buttons)
            or flow.scene in ("settlement", "showcase")
            and any(
                b.enabled
                and (
                    b.kind in ("continue", "close_showcase")
                    or b.kind == "return_lobby"
                    and b.label == "返回大厅"
                )
                for b in flow.buttons
            )
        )
        if not reliable:
            self._boundary_probe = None
            return False
        if not self._boundary_probe or self._boundary_probe[0] != flow.scene:
            self._boundary_probe = flow.scene, flow.frame.media_pts_us
            return False
        if flow.frame.media_pts_us - self._boundary_probe[1] < 100_000:
            return False
        self.bot.executor.cancel("异常局已明确结束，保留原提交结果")
        if hasattr(self.pipeline, "finish_skipped_game"):
            self.pipeline.finish_skipped_game(self.current_game or self._seen_deal, flow.frame.media_pts_us)
        self._commit("round_boundary", game_id=self.current_game or self._seen_deal)
        self.current_game = self.armed_game = ""
        self.quarantined = False
        self.reason = "上一局异常已跳过，正在继续"
        return True

    def request_stop(self, reason: str = "用户停止连续任务", *, manual: bool = True) -> None:
        """GUI 的非阻塞停止意图；立即减权并停止循环，正式账本由后台 close/stop 收尾。

        不取得流程锁，因此即便已有一个 ADB 命令在途也立即返回。窗口同时锁住输入闸门，
        防止正在结束的一轮后台工作重新授权；已发命令仍保留原本的在途记录。
        """
        self.gate.revoke()
        self.bot.set_mode("stop")
        self._stop_intent = (reason, manual)
        self.stop_event.set()

    def stop(self, reason="用户停止连续任务", *, manual=True):
        """停止后不再恢复本任务；先撤销输入，再让线程和录像分别收尾。"""
        with self.lock:
            if self._stop_intent:
                (reason, manual), self._stop_intent = self._stop_intent, None
            self.gate.revoke()
            self.bot.set_mode("stop")
            if self.status not in ("completed", "stopped", "idle"):
                self.status, self.reason = "stopped", reason
                self.store.set("status", self.status)
                self.store.add("manual_takeover" if manual else "stopped", reason=reason, at=self.clock())
            self.stop_event.set()

    def resume(self):
        """用户显式恢复前核查实时源、页面及历史；未知提交和有缺口的本局拒绝恢复。"""
        with self.lock:
            if self.status != "paused":
                raise ValueError("当前任务不是暂停状态")
            if getattr(self.pipeline, "failed", False):
                raise ValueError("已发生程序异常，正在保存故障资料；请收尾后开始新任务")
            flow, snapshot = self.pipeline.flow_snapshot(), self.pipeline.bot_snapshot()
            if (
                not flow
                or flow.blocked_reason
                or flow.scene == "unknown"
                or not self.pipeline.allowed
                or self.clock() - flow.frame.started > 1
            ):
                raise ValueError("恢复前需要清晰、在线且可控制的新画面")
            if snapshot and snapshot.state["game_id"] == self.current_game and not snapshot.state["ended"]:
                state = snapshot.state
                if state["pending"] or state["capture_incomplete"]:
                    raise ValueError("本局存在真实历史缺口，不能恢复机器人出牌")
                if (state["game_id"], state["turn"]) in self.bot.executor.sent_turns:
                    raise ValueError("本轮已有未确认提交，禁止恢复后重复提交")
            if self.pending_flow:
                raise ValueError("存在未确认页面命令，请先人工处理并开始新任务")
            self.status, self.reason = "running", ""
            self.last_id, self.last_new_at = "", self.clock()
            self.armed_game = ""
            self.store.set("status", "running")
            self.store.add("resumed", at=self.clock())

    def close(self, *, evaluate: bool = True) -> None:
        """退出时等待在途ADB返回后再关账本；不让迟到记录写入下一任务。

        evaluate=True保留旧的隔离调用方式；生产收尾使用False，待机器人专用账本、
        采集与会话数据库全部关闭后由收尾方评测，关闭评测连接后才执行媒体清理。
        """
        self.stop(manual=False)
        if self.thread:
            self.thread.join(10)
            if self.thread.is_alive():
                raise RuntimeError("连续任务后台尚未退出，账本保持打开")
        if self.bidding:
            self.bidding.close()
            self.bidding = None
        if self.store:
            if evaluate:
                evaluate_run(self.store.folder)
            self.store.close()
            self.store = None

    def _complete(self, reason):
        self.gate.revoke()
        self.bot.set_mode("suggest")
        self.status, self.reason = "completed", reason
        self.store.set("status", self.status)
        self.store.add("finished", reason=reason, at=self.clock())
        self.stop_event.set()

    def _limited(self) -> bool:
        """只按已确认完成的局数判断任务目标，不限制总用时或等待大厅的时长。"""
        return len(self.completed) >= self.settings["max_games"]

    def _confirm_startup_entry(self, flow, entry: str) -> None:
        """记录首次可信新局入口并结束启动等待；调用方必须已经完成入口证据校验。

        entry 为 lobby 时已取得两帧稳定匹配按钮，为 settlement 时结算续局命令已经
        得到新页面确认。记录只解释本任务从何处启动，不补造旧局历史，也不增加局数。
        本方法在流程线程锁内调用，不发送设备输入。
        """
        if self.startup_phase != "waiting_lobby":
            return
        self.startup_phase, self.startup_entry, self.reason = "ready", entry, ""
        if entry == "lobby":
            self.lobby_confirmed = True
        self.store.set("startup_phase", self.startup_phase)
        self.store.set("startup_entry", entry)
        self._commit(
            "startup_wait_finished",
            entry=entry,
            frame_id=flow.frame.frame_id,
            pts_us=flow.frame.media_pts_us,
            at=self.clock(),
        )

    def _wait_for_entry(self, flow) -> bool:
        """处理首次新局入口确认前的等待；True 表示本帧已消费，不进入单局授权。

        大厅匹配按钮稳定后可开始新局；明确结算页优先续局，也可使用完整文字的返回
        大厅按钮，战绩展示仅允许关闭。点击均经过原有双帧、时效和发送前检查，续局
        还须等待页面反馈才能结束启动等待。当前牌局仅观察，不写 game_complete，
        不把缺少旧局开局历史计作程序故障。本方法在流程线程锁内调用。
        """
        if self.startup_phase != "waiting_lobby":
            return False
        if flow.scene in ("lobby", "room"):
            button = next((b for b in flow.buttons if b.kind == "match" and b.enabled), None)
            self.reason = "正在确认大厅入口" if button else "等待可确认的大厅匹配入口"
            if button is None:
                self.candidate = None
                return True
            signature = (f"{self.scene_epoch}:match:{button.label}", button.hit.x // 15, button.hit.y // 15)
            if not self.candidate or self.candidate[0] != signature:
                self.candidate = (signature, flow.frame.media_pts_us)
                return True
            if flow.frame.media_pts_us - self.candidate[1] < 100_000:
                return True
            self._confirm_startup_entry(flow, "lobby")
            return False
        if flow.scene in ("unknown", "matching"):
            # 无法识别的页面仍使用原有五秒规则，不能用等待大厅掩盖识别故障。
            self.reason = "等待确认大厅或结算续局入口"
            return False
        if flow.scene == "settlement":
            has_continue = any(b.kind == "continue" and b.enabled for b in flow.buttons)
            has_return = any(
                b.kind == "return_lobby" and b.label == "返回大厅" and b.enabled for b in flow.buttons
            )
            if has_continue:
                self.reason = "正在确认继续游戏按钮"
            elif has_return:
                self.reason = "正在确认返回大厅按钮"
            else:
                self.reason = "本局已结束，等待可用的继续游戏或返回大厅按钮；也可手动返回"
            if not (has_continue or has_return):
                self.candidate = None
                return True
            return False
        if flow.scene == "showcase":
            has_close = any(b.kind == "close_showcase" and b.enabled for b in flow.buttons)
            self.reason = "正在确认战绩展示的继续按钮" if has_close else "等待可确认的战绩展示关闭按钮"
            if not has_close:
                self.candidate = None
            return not has_close
        self.reason = "等待当前牌局结束后开始下一局"
        self.candidate = None
        self.gate.revoke()
        return True

    def advance(self, flow, snapshot):
        """消费一个最新同帧观察；相同帧不增加稳定次数，超时由单调时钟而非帧数判断。"""
        with self.lock:
            if not self.active:
                return
            now = self.clock()
            if not self.bot.ready.is_set():
                if self.bot.thread and not self.bot.thread.is_alive():
                    self.pause(self.bot.status)
                return
            self._ready_since = self._ready_since or now
            # 阻断弹窗即使OCR耗时超过点击时效，也只能用来撤销输入，不能被“画面过期”
            # 丢弃后继续沿用旧牌局。仅接受当前设备、视频和方向的证据，防止跨会话串用。
            if flow is not None and flow.blocked_reason:
                blocked_identity = (
                    flow.frame.serial,
                    flow.frame.stream_id,
                    flow.frame.width,
                    flow.frame.height,
                )
                if self.identity is None or blocked_identity == self.identity:
                    self.store.observation(flow)
                    self.pause(flow.blocked_reason, scope=FaultScope.EXTERNAL, code="blocked_page")
                    return
            if (
                now - self.gate.heartbeat_at > 1
                or self.pipeline.stop_event.is_set()
                or getattr(self.pipeline, "failed", False)
            ):
                self.pause("界面心跳或视频已中断")
                return
            if not self.pipeline.allowed:
                if self.identity is not None or now - self._ready_since > 20:
                    self.pause("游戏前台或设备状态失效", scope=FaultScope.EXTERNAL, code="foreground")
                return
            recording = getattr(self.pipeline, "record_ready", None)
            if recording is not None and not recording.is_set():
                if now - self._ready_since > 20:
                    self.pause("录像首个关键帧尚未落盘，不能开始匹配")
                return
            # 首次还需冷启动流程OCR，不能把它与运行中的视频中断混为一谈。
            # 最多预热20秒，期间绝不点击；一旦接受第一帧，恢复正常三秒中断保护。
            if self.identity is None and (flow is None or now - flow.frame.started > 1):
                if now - self._ready_since > 20:
                    self.pause("流程识别初始化后仍没有清晰首帧")
                return
            if flow is None or flow.frame.frame_id == self.last_id:
                raw = self.pipeline.frame
                budget = 8 if raw is not None and now - raw.started <= 1 else 3
                if now - max(self.last_new_at, self._ready_since) > budget:
                    self.pause(f"连续流程超过{budget}秒没有新观察")
                return
            if now - flow.frame.started > 1:
                raw = self.pipeline.frame
                budget = 8 if raw is not None and now - raw.started <= 1 else 3
                if now - self.last_new_at > budget:
                    self.pause("流程画面持续过期")
                return
            identity = (flow.frame.serial, flow.frame.stream_id, flow.frame.width, flow.frame.height)
            if self.identity and identity != self.identity:
                self.pause("设备、视频会话或方向发生变化")
                return
            self.identity = identity
            self.last_id, self.last_new_at, self.last_flow = flow.frame.frame_id, now, flow
            self.store.observation(flow)
            if snapshot and flow.scene in ("dealing", "bidding", "doubling"):
                deal = snapshot.state["game_id"]
                if deal and deal != self._seen_deal:
                    self._seen_deal = deal
                    self.scene_epoch += 1
                    self.candidate = None
                    self._commit(
                        "deal_attempt",
                        game_id=deal,
                        frame_id=flow.frame.frame_id,
                        pts_us=flow.frame.media_pts_us,
                    )
            if flow.blocked_reason:
                self.pause(flow.blocked_reason, scope=FaultScope.EXTERNAL, code="blocked_page")
                return
            if (
                snapshot
                and not self.quarantined
                and self.armed_game == snapshot.state["game_id"]
                and (snapshot.state["pending"] or snapshot.state["capture_incomplete"])
            ):
                self.pause("本局出现真实历史缺口或矛盾", scope=FaultScope.ROUND, code="history_gap")
                return
            if flow.scene != self.scene:
                self.scene, self.scene_since = flow.scene, now
                if flow.scene != "unknown" and flow.scene != self._last_known_scene:
                    self.scene_epoch += 1
                    self._last_known_scene = flow.scene
                self.candidate = None
                self.store.add(
                    "scene", scene=flow.scene, frame_id=flow.frame.frame_id, pts_us=flow.frame.media_pts_us
                )
                self.pipeline.evidence.submit(flow.frame, [f"flow-{flow.scene}"])
            if self.blocked_flow:
                # 跨执行进程的旧命令永不补发；明确离开原页面后只归档未知结果，不补造确认。
                if flow.scene in ("unknown", self.blocked_flow["origin_scene"]):
                    self.reason = "旧页面命令结果未知，正在等待页面自然推进"
                    return
                self._commit("flow_retired", action_id=self.blocked_flow["action_id"], scene=flow.scene)
                self.blocked_flow = None
            if self.quarantined:
                if not self._quarantine_boundary(flow):
                    return
            if self.pending_flow and now - self.pending_flow["sent_at"] > 8:
                self.pause(
                    "页面操作在证据等待窗口内仍无法唯一确认，禁止重复点击",
                    scope=FaultScope.EXTERNAL,
                    code="unknown_flow",
                )
                return
            if self.pending_flow:
                pending = self.pending_flow
                if (
                    pending["kind"] == "continue"
                    and snapshot
                    and getattr(snapshot, "frame", None) is not None
                    and snapshot.frame.media_pts_us > pending["pts_us"]
                    and snapshot.state["game_id"] != pending.get("origin_game_id")
                    and snapshot.observation.get("phase") in ("dealing", "bidding", "doubling", "playing")
                    and len(snapshot.observation.get("hand") or []) in (17, 20)
                    and not snapshot.state["capture_incomplete"]
                    and not snapshot.state["pending"]
                    and (
                        snapshot.frame.serial,
                        snapshot.frame.stream_id,
                        snapshot.frame.width,
                        snapshot.frame.height,
                    )
                    == self.identity
                ):
                    probe = pending.get("new_round_probe")
                    hand_signature = tuple(sorted(snapshot.observation["hand"]))
                    raw_context = snapshot.observation.get("frame_context", {})
                    if (
                        not probe
                        or probe[0] != snapshot.state["game_id"]
                        or probe[3] != hand_signature
                        or probe[4] != raw_context
                    ):
                        pending["new_round_probe"] = (
                            snapshot.state["game_id"],
                            snapshot.frame.media_pts_us,
                            snapshot.frame.frame_id,
                            hand_signature,
                            raw_context,
                        )
                    elif (
                        snapshot.frame.frame_id != probe[2]
                        and snapshot.frame.media_pts_us - probe[1] >= 100_000
                    ):
                        self._commit(
                            "flow_confirmed",
                            action_id=pending["action_id"],
                            frame_id=snapshot.frame.frame_id,
                            pts_us=snapshot.frame.media_pts_us,
                            evidence="new_round",
                            proof={
                                "version": 2,
                                "frames": [probe[2], snapshot.frame.frame_id],
                                "game_id": probe[0],
                                "origin_game_id": pending.get("origin_game_id"),
                                "raw_round_id": snapshot.observation.get("frame_context", {}).get("round_id"),
                            },
                        )
                        self._confirm_startup_entry(flow, "settlement")
                        self.pending_flow, self.candidate = None, None
                elif pending.get("new_round_probe"):
                    pending.pop("new_round_probe", None)
            if self.pending_flow:
                pending = self.pending_flow
                # 所有模式在旧按钮已消失时继续只读取证，最多8秒；仅消失并不等于成功。
                # 明确的新页面或本人反馈才确认，等待期间不重复点击，也不补造出牌历史。
                evidence_window = 8
                present = any(
                    b.kind == pending["kind"]
                    and b.label == pending["label"]
                    and abs(b.hit.x - pending["button_box"][0]) < 15
                    and abs(b.hit.y - pending["button_box"][1]) < 15
                    for b in flow.buttons
                )
                expected = {
                    "match": {"matching", "dealing", "bidding", "doubling", "playing", "room_confirmation"},
                    "continue": {
                        "lobby",
                        "room",
                        "matching",
                        "dealing",
                        "bidding",
                        "doubling",
                        "playing",
                        "room_confirmation",
                    },
                    "return_lobby": {"lobby", "room"},
                    "decline_bid": {"bidding", "dealing", "doubling", "playing"},
                    "call_landlord": {"bidding", "dealing", "doubling", "playing"},
                    "rob_landlord": {"bidding", "dealing", "doubling", "playing"},
                    "decline_double": {"dealing", "doubling", "playing"},
                    "confirm_room": {"lobby", "room", "matching", "dealing", "bidding"},
                    "close_showcase": {"settlement", "lobby", "room"},
                }[pending["kind"]]
                feedback = (
                    pending["kind"] in flow.feedbacks
                    and flow.frame.started - pending["sent_at"] <= evidence_window
                )
                if not present and (flow.scene in expected or feedback):
                    advanced = (
                        flow.scene != pending["origin_scene"]
                        and flow.frame.started - pending["sent_at"] <= evidence_window
                    )
                    # 加倍按钮收起后，短动画可能只留下牌桌外壳，被保守归为dealing。
                    # 这不是阶段推进证据，只允许通过两个新帧的按钮消失确认，不猜OCR近似字。
                    if pending["kind"] == "decline_double" and flow.scene == "dealing":
                        advanced = False
                    if advanced or feedback:
                        pending["absent_us"] = flow.frame.media_pts_us - 100_000
                    elif pending.get("absent_us") is None:
                        pending["absent_us"] = flow.frame.media_pts_us
                    if flow.frame.media_pts_us - pending["absent_us"] >= 100_000:
                        self._commit(
                            "flow_confirmed",
                            action_id=pending["action_id"],
                            frame_id=flow.frame.frame_id,
                            pts_us=flow.frame.media_pts_us,
                            evidence="own_feedback"
                            if feedback
                            else ("scene_change" if advanced else "button_disappearance"),
                        )
                        if pending.get("bidding"):
                            self.bidding.committed(self.bid_decision, pending["bid_deal"])
                        if pending["kind"] == "continue" and pending["origin_scene"] == "settlement":
                            # 仅在新页面确认续局后授权；点击返回或关闭战绩都不代表已进入新局。
                            self._confirm_startup_entry(flow, "settlement")
                        self.pending_flow, self.candidate = None, None
                else:
                    pending["absent_us"] = None
                if self.pending_flow:
                    wait_for_evidence = not present and (
                        flow.scene == "unknown" or pending.get("absent_us") is not None
                    )
                    if wait_for_evidence and not pending.get("evidence_wait_logged"):
                        pending["evidence_wait_logged"] = True
                        self.store.add(
                            "flow_evidence_wait",
                            action_id=pending["action_id"],
                            frame_id=flow.frame.frame_id,
                            pts_us=flow.frame.media_pts_us,
                            reason="旧按钮消失，等待新阶段或明确反馈",
                            deadline_seconds=8,
                        )
                    return
            if self._wait_for_entry(flow):
                return
            if flow.scene == "unknown":
                if now - self.scene_since > 5:
                    self.pause("未知页面持续五秒", scope=FaultScope.EXTERNAL, code="unknown_page")
                return
            if flow.scene == "matching":
                if now - self.scene_since > 120:
                    self.pause("匹配等待超过120秒", scope=FaultScope.EXTERNAL, code="matching_timeout")
                return
            if flow.scene == "playing":
                if not self.entry_confirmed:
                    self.pause("连续任务须从已确认的大厅或结算续局入口开始，不能中途接管未知牌局")
                    return
                self._playing(snapshot)
                return
            if flow.scene == "ending" and now - self.scene_since > 10:
                self.pause(
                    "结算已确认，但十秒内没有出现可确认的结算按钮",
                    scope=FaultScope.ROUND,
                    code="ending_timeout",
                )
                return
            if flow.scene in ("settlement", "showcase", "ending"):
                if not self._settlement(snapshot):
                    return
            if flow.scene in ("lobby", "room"):
                if self._limited():
                    self._complete("已完成目标局数，不再匹配")
                    return
                if not any(button.kind == "match" and button.enabled for button in flow.buttons):
                    self.candidate = None
                    return
                self.lobby_confirmed = True
            if flow.scene in ("settlement", "showcase", "ending") and self._limited():
                self._complete("当前局结束，已完成目标局数")
                return
            preferences = {
                "lobby": ("match",),
                "room": ("match",),
                "bidding": ("decline_bid",),
                "doubling": ("decline_double",),
                "settlement": ("continue", "return_lobby"),
                "room_confirmation": ("confirm_room",),
                "showcase": ("close_showcase",),
            }
            bid_decision = None
            bid_deal = (
                getattr(flow, "bid_deal_id", "") or self._seen_deal or f"before-play-{len(self.completed)}"
            )
            if flow.scene == "bidding":
                bid_decision = self.bidding.choose(flow, bid_deal)
                if bid_decision is None:
                    self.bid_description = "等待叫抢手牌及模型评估"
                    self.candidate = None
                    return
                self.bid_decision = bid_decision
                self.bid_description = (
                    f"{'叫地主' if bid_decision.kind == 'call_landlord' else '抢地主' if bid_decision.kind == 'rob_landlord' else '不叫/不抢'}"
                    + (
                        f" · 评分{bid_decision.score:.3f} / 阈值{bid_decision.threshold:.3f}"
                        if bid_decision.score is not None
                        else f" · {bid_decision.reason}"
                    )
                )
                preferences["bidding"] = (bid_decision.kind,)
            button = next(
                (
                    b
                    for kind in preferences.get(flow.scene, ())
                    for b in flow.buttons
                    if b.kind == kind
                    and b.enabled
                    and (
                        self.startup_phase != "waiting_lobby"
                        or b.kind in ("continue", "close_showcase")
                        or b.label == "返回大厅"
                    )
                ),
                None,
            )
            if button is None:
                self.candidate = None
                return
            if not self.entry_confirmed and button.kind not in (
                "continue",
                "return_lobby",
                "confirm_room",
                "close_showcase",
            ):
                self.pause("任务必须先从已确认的大厅或结算续局入口开始")
                return
            opportunity = (
                bid_decision.opportunity
                if bid_decision
                else f"{self.scene_epoch}:{button.kind}:{button.label}"
            )
            if opportunity in self.sent_opportunities:
                return
            signature = (opportunity, button.hit.x // 15, button.hit.y // 15)
            if not self.candidate or self.candidate[0] != signature:
                self.candidate = (signature, flow.frame.media_pts_us)
                return
            if flow.frame.media_pts_us - self.candidate[1] < 100_000:
                return
            if bid_decision:
                self.store.add(
                    "bidding_decision",
                    **decision_dict(bid_decision),
                    frame_id=flow.frame.frame_id,
                    pts_us=flow.frame.media_pts_us,
                    public_markers=list(flow.bid_markers),
                )
            self._tap(button, flow, opportunity, bidding=bid_decision, bid_deal=bid_deal)

    def _tap(self, button, flow, opportunity, *, bidding=None, bid_deal=""):
        """持久化页面意图后重新取当前按钮，单次发送并单独记录实际帧与源身份。

        调用方持有协调锁；等待USB时释放锁让停止立即撤销授权。明确未发送才撤销占位，
        返回未知时保留在途状态，只读取证至八秒，不发第二次页面命令。
        """
        from doudizhu_lab.domain.inference import frame_context

        if self.bot.executor.active:
            self.pause("页面输入与在途出牌冲突")
            return
        self.bot.set_mode("suggest")
        token = self.gate.grant("flow")
        action_id = uuid4().hex
        origin_snapshot = self.pipeline.bot_snapshot()
        record = dict(
            action_id=action_id,
            opportunity=opportunity,
            kind=button.kind,
            label=button.label,
            frame_id=flow.frame.frame_id,
            pts_us=flow.frame.media_pts_us,
            sent_at=self.clock(),
            origin_scene=flow.scene,
            frame_context=frame_context(flow.frame),
            origin_round_id=(
                origin_snapshot.observation.get("frame_context", {}).get("round_id")
                if origin_snapshot
                else None
            ),
            origin_game_id=(origin_snapshot.state["game_id"] if origin_snapshot else self.current_game),
            button_box=[button.hit.x, button.hit.y, button.hit.width, button.hit.height],
            x=button.hit.x + button.hit.width // 2,
            y=button.hit.y + button.hit.height // 2,
        )
        if bidding:
            record.update(bidding=decision_dict(bidding), bid_deal=bid_deal)
        self.pending_flow = record.copy()
        self.sent_opportunities.add(opportunity)
        # 意图先持久化到父任务；ACK期间画面可能变旧，真正发送前重新定位同一机会。
        self._commit("flow_intent", **record)
        self.pipeline.evidence.submit(flow.frame, [action_id])

        def guard():
            newest = self.pipeline.flow_snapshot()
            raw = self.pipeline.frame
            return bool(
                self.active
                and self.pipeline.allowed
                and not self.pipeline.stop_event.is_set()
                and not getattr(self.pipeline, "failed", False)
                and newest
                and newest.scene == flow.scene
                and (
                    not bidding
                    or (
                        newest.bid_deal_id == flow.bid_deal_id
                        and (
                            bidding.kind == "decline_bid"
                            or Counter(newest.bid_hand or ()) == Counter(bidding.hand)
                        )
                    )
                )
                and raw
                and (raw.serial, raw.stream_id, raw.width, raw.height) == self.identity
                and any(
                    b.kind == button.kind
                    and b.label == button.label
                    and b.enabled
                    and abs(b.hit.x - button.hit.x) < 15
                    and abs(b.hit.y - button.hit.y) < 15
                    for b in newest.buttons
                )
            )

        # advance持有状态锁；设备命令可能等待USB，发送前释放，确保GUI停止不被它阻塞。
        self.lock.release()
        sent_recorded = False
        try:
            newest = self.pipeline.flow_snapshot()
            if newest and newest.scene == flow.scene and newest.frame.frame_id != flow.frame.frame_id:
                replacement = next(
                    (
                        b
                        for b in newest.buttons
                        if b.kind == button.kind
                        and b.label == button.label
                        and b.enabled
                        and abs(b.hit.x - button.hit.x) < 15
                        and abs(b.hit.y - button.hit.y) < 15
                    ),
                    None,
                )
                if replacement and (
                    not bidding
                    or (
                        newest.bid_deal_id == flow.bid_deal_id
                        and Counter(newest.bid_hand or ()) == Counter(bidding.hand)
                    )
                ):
                    flow, button = newest, replacement
                    self.store.observation(flow)
            record.update(
                frame_id=flow.frame.frame_id,
                pts_us=flow.frame.media_pts_us,
                sent_at=self.clock(),
                button_box=[button.hit.x, button.hit.y, button.hit.width, button.hit.height],
                x=button.hit.x + button.hit.width // 2,
                y=button.hit.y + button.hit.height // 2,
            )
            self.pending_flow = record.copy()
            box = button.hit
            self.gate.tap(
                "flow",
                token,
                getattr(self.pipeline, "control_client", self.pipeline.client),
                box.x + box.width // 2,
                box.y + box.height // 2,
                flow.frame,
                guard=guard,
                cancelled=lambda: not self.active,
            )
            sent_recorded = True
            self._commit("flow_sent", **record)
            self.store.add("flow_returned", action_id=action_id, at=self.clock())
        except InputRejected as exc:
            self._commit("flow_not_sent", action_id=action_id, message=str(exc), code=exc.code)
            self.pending_flow = None
            self.sent_opportunities.discard(opportunity)
            self.candidate = None
            if exc.code == "stale_frame":
                self.reason = "画面过期，等待新帧重新核对页面按钮"
            elif self.active:
                self.pause(f"页面输入未发送：{exc}")
        except Exception as exc:
            if not sent_recorded:
                sent_recorded = True
                self._commit("flow_sent", **record)
            self.store.add("flow_unknown", action_id=action_id, message=str(exc))
            if (
                self.pipeline.stop_event.is_set()
                or getattr(self.pipeline, "failed", False)
                or not self.pipeline.allowed
            ):
                self.pause("设备、前台或采集已失效，页面命令保持未确认")
            else:
                self.reason = "页面命令结果未知，只读等待推进证据，禁止重复点击"
        finally:
            self.lock.acquire()

    def _playing(self, snapshot):
        if snapshot is None:
            return
        state = snapshot.state
        if state["ended"] or state["phase"] != "playing":
            return
        if self.unattended and self.armed_game != state["game_id"]:
            # 新局即使未达到出牌授权条件，也必须有独立异常编号，不能误固定上一局录像。
            self.current_game = state["game_id"]
        if state["pending"] or state["capture_incomplete"]:
            self.pause("出牌前发现历史缺口，不能用猜测状态启动", scope=FaultScope.ROUND, code="history_gap")
            return
        if self.armed_game == state["game_id"]:
            if self.bot.mode == "stop":
                self.pause(
                    self.bot.pause_reason or self.bot.executor.reason or self.bot.status,
                    scope=FaultScope.ROUND,
                    code="bot_round_halted",
                )
            return
        if not state["landlord"] or not state["bottom"] or not state["initial_hand"]:
            # 新旧局切换的跟踪候选可能早于稳定的身份/底牌快照；先等待取证，不猜底牌。
            startup_wait = 8 if self.pipeline.profile.continuous_policy != "strict" else 3
            if self.clock() - self.scene_since > startup_wait:
                self.pause(
                    "开局取证窗口结束后身份、底牌或手牌仍不完整",
                    scope=FaultScope.ROUND,
                    code="incomplete_setup",
                )
            return
        self.current_game = self.armed_game = state["game_id"]
        self._finish_wait = None
        self._commit(
            "game_armed",
            game_id=self.current_game,
            frame_id=snapshot.frame.frame_id,
            pts_us=snapshot.frame.media_pts_us,
        )
        self.bot.arm_game(self.current_game)

    def _settlement(self, snapshot):
        if not self.current_game:
            return True  # 可从已结束页面续局或返回大厅，不能把启动前的旧局计为完成。
        if self.current_game in self.registered:
            return True
        self._finish_wait = self._finish_wait or self.clock()
        if (
            snapshot is None
            or not snapshot.state["ended"]
            or snapshot.state["game_id"] != self.current_game
            or self.bot.executor.active
        ):
            if self.clock() - self._finish_wait > 3:
                self.pause("结算后末手或牌局边界无法确认", scope=FaultScope.ROUND, code="incomplete_end")
            return False
        state = snapshot.state
        actions = self.pipeline.session.rows("bot_actions")
        starts = {
            a["action_id"]: a.get("decision", {})
            for a in actions
            if a.get("game_id") == self.current_game and a["stage"] == "开始"
        }
        confirmed = {
            starts[a["action_id"]].get("turn")
            for a in actions
            if a.get("game_id") == self.current_game and a["stage"] == "已确认" and a["action_id"] in starts
        }
        moves = [e for e in snapshot.history if e["kind"] in ("play", "pass")]
        expected = {i for i, e in enumerate(moves) if e["seat"] == "self"}
        verified = bool(
            state["complete"]
            and not state["pending"]
            and not state["capture_incomplete"]
            and confirmed == expected
            and expected
        )
        rule_game = state.get("strategy") == "rules"
        confirmed_ids = {
            a["action_id"]
            for a in actions
            if a.get("game_id") == self.current_game and a["stage"] == "已确认"
        }
        if rule_game and self.pipeline.profile.continuous_policy == "balanced":
            # 局部规则不宣称补齐原前史；完成依据改为明确结算、本人手牌监控及全部提交确认。
            model_turns = {
                d["turn"]
                for ident, d in starts.items()
                if ident in confirmed_ids and d.get("strategy", "wp") in ("wp", "resnet")
            }
            verified = bool(
                not state.get("rule_halt_reason")
                and not state["capture_incomplete"]
                and starts
                and set(starts) == confirmed_ids
                and model_turns == expected
            )
        if not verified:
            if self.clock() - self._finish_wait > 3:
                self.pause(
                    "本局存在未确认出牌、托管或不完整历史，不能计为机器人完成",
                    scope=FaultScope.ROUND,
                    code="unverified_game",
                )
            return False
        start = next(
            (e["pts_us"] for e in snapshot.history if e["kind"] == "game_start"), snapshot.frame.media_pts_us
        )
        self._commit(
            "game_complete",
            game_id=self.current_game,
            start_us=start,
            end_us=snapshot.frame.media_pts_us,
            verified=True,
            own_actions=len(confirmed_ids) if rule_game else len(expected),
            confirmed_actions=len(confirmed_ids) if rule_game else len(confirmed),
            degraded=rule_game,
            strict_complete=bool(state["complete"] and not rule_game),
            raw_visual_complete=not any(e.get("source") == "operational_inferred" for e in snapshot.history),
            inferred_events=sum(e.get("source") == "operational_inferred" for e in snapshot.history),
            strategy_counts=dict(Counter(d.get("strategy", "wp") for d in starts.values())),
            frame_id=snapshot.frame.frame_id,
        )
        self.registered.add(self.current_game)
        self.completed.append(self.current_game)
        if hasattr(self.pipeline, "complete_game"):
            self.pipeline.complete_game(self.current_game, snapshot.frame.media_pts_us)
        self.bot.set_mode("suggest")
        return True
