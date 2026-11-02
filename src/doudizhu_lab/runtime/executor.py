"""逐帧出牌事务；输入命令成功与游戏接受动作严格分离。"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict
from time import monotonic
from uuid import uuid4

from doudizhu_lab.agents.douzero import TO_NUMBER, context
from doudizhu_lab.agents.readiness import ObservationWait
from doudizhu_lab.devices.adb import InputRejected
from doudizhu_lab.domain.bot_types import Decision, ObservationSnapshot


def execution_context(snapshot, strategy="wp"):
    """按明确策略核查输入；规则接管绝不调用需要完整前史的DouZero编码接口。"""
    if strategy == "rules":
        from doudizhu_lab.agents.rules import context as rule_context

        return rule_context(snapshot)
    if strategy not in ("wp", "resnet"):
        raise ValueError("未知出牌策略")
    return context(snapshot)


class ActionExecutor:
    """一次只执行一个动作；send 为受独占与心跳保护的单次输入接口。

    advance 由后台协调线程在新快照到达时调用，时钟可注入以进行确定性故障测试。
    取消只撤销未来命令，已发出的命令保留在审计中；提交结果不明永不自动重试。
    """

    ACTIVE = {"选牌", "核对", "等待反馈"}

    def __init__(self, send, audit=lambda record: None, *, clock=monotonic, settings=None):
        self.send, self.audit, self.clock = send, audit, clock
        self.settings = {"max_frame_age": 1.0, "selection_timeout": 12.0, "confirmation_timeout": 3.0} | (
            settings or {}
        )
        self.stage, self.reason = "等待", ""
        self.decision = None
        self.action_id = ""
        self.sent_turns = set()
        self.last_frame = ""
        self.last_snapshot = None

    @property
    def active(self):
        return self.stage in self.ACTIVE

    def _log(self, stage, snapshot=None, **extra):
        snapshot = snapshot or self.last_snapshot
        record = {"action_id": self.action_id, "stage": stage, "monotonic": self.clock(), **extra}
        if snapshot:
            record.update(
                game_id=snapshot.state["game_id"],
                frame_id=snapshot.frame.frame_id,
                pts_us=snapshot.frame.media_pts_us,
                version=snapshot.version,
                revision=snapshot.revision,
            )
        self.audit(record)

    def start(self, decision: Decision, snapshot: ObservationSnapshot):
        """绑定当前回合及合法动作；拒绝同一回合再次提交或执行失效建议。"""
        if self.active:
            raise ValueError("已有执行中的动作")
        if decision.version != snapshot.version:
            raise ValueError("建议对应的局面已经变化")
        state = snapshot.state
        if (decision.game_id, decision.turn) != (state["game_id"], state["turn"]):
            raise ValueError("建议所属牌局或轮次错误")
        if (decision.game_id, decision.turn) in self.sent_turns:
            raise ValueError("本轮已经提交，禁止重复操作")
        data = execution_context(snapshot, decision.strategy)
        if sorted(TO_NUMBER[c] for c in decision.cards) not in data["legal_actions"]:
            raise ValueError("建议不是合法动作")
        self.decision, self.last_snapshot = decision, snapshot
        self.action_id = uuid4().hex
        self.stage, self.reason = "选牌", ""
        self.started = self.clock()
        self.stable_since = None
        self.last_frame = ""
        self.tap_pending = None
        self.last_selection = None
        self.wait_since = None
        self.layout_wait_since = None
        self.identity = (
            snapshot.frame.serial,
            snapshot.frame.stream_id,
            snapshot.frame.width,
            snapshot.frame.height,
        )
        self.before_hand = Counter(state["hand"])
        self.before_counts = dict(state["remaining"])
        self.rule_feedback = None
        self.confirmation_evidence = None
        self._log("开始", snapshot, decision=asdict(decision), input_context=data)

    def cancel(self, reason: str):
        """停止后续输入；保留已提交或在途命令的信息，人工恢复不隐式重发。"""
        self.reason = reason
        if self.decision is not None and self.stage != "暂停":
            self._log("暂停", reason=reason, previous_stage=self.stage)
        self.stage = "暂停"

    def _confirmed(self, snapshot):
        if self.decision.strategy == "rules":
            from doudizhu_lab.agents.rules import feedback

            value = feedback(
                snapshot, self.decision, self.before_hand, self.before_counts, self.submitted_pts
            )
            if value is None:
                self.rule_feedback = None
                return False
            if self.rule_feedback is None or self.rule_feedback[0] != value:
                self.rule_feedback = value, snapshot.frame.media_pts_us, [snapshot.frame.frame_id]
                return False
            self.rule_feedback[2].append(snapshot.frame.frame_id)
            if snapshot.frame.media_pts_us - self.rule_feedback[1] < 100_000:
                return False
            self.confirmation_evidence = value | {
                "frames": self.rule_feedback[2][-2:],
                "first_pts_us": self.rule_feedback[1],
                "last_pts_us": snapshot.frame.media_pts_us,
            }
            return True
        state, obs = snapshot.state, snapshot.observation
        # 必须找到本轮、提交后由观察器确认的自己动作；后台操作日志不能构成证据。
        moves = [e for e in snapshot.history if e["kind"] in ("play", "pass")]
        if len(moves) <= self.decision.turn:
            return False
        event = moves[self.decision.turn]
        self.confirmation_evidence = {
            "basis": event.get("source", "automatic"),
            "event_id": event.get("event_id"),
        }
        if (
            event["seat"] != "self"
            or event["kind"] != self.decision.kind
            or Counter(event["cards"]) != Counter(self.decision.cards)
            or not event.get("confirmed", True)
        ):
            self.cancel("观察到的本轮动作与提交意图不同，请人工核对")
            return False
        if (
            state["pending"]
            or state["game_id"] != self.decision.game_id
            or event["pts_us"] <= self.submitted_pts
        ):
            return False
        if self.decision.cards:
            expected = self.before_hand - Counter(self.decision.cards)
            # 最后一手可以由明确结算和零剩牌证据确认，不要求消失的手牌仍能识别。
            hand_ok = obs.get("hand") is not None and Counter(obs["hand"]) == expected
            final = not expected and state["ended"] and state["remaining"]["self"] == 0
            return (
                (hand_ok or final)
                and Counter(state["hand"] or []) == expected
                and (state["turn"] > self.decision.turn or state["ended"])
            )
        # 不出字样可能在行动者稳定前已经消失。观察器已保存本轮提交后的两帧明确提示，
        # 就不再要求它持续留在当前画面；不得用推断事件或单纯按钮消失替代实际反馈。
        proof = event.get("data", {}).get("evidence_frames", [])
        observed = event.get("source", "automatic") in ("automatic", "operational_observed")
        inference = event.get("data", {}).get("inference", {})
        own_inferred = bool(
            event.get("source") == "operational_inferred"
            and inference.get("own_pass_after_submission")
            and inference.get("submission_pts_us") == self.submitted_pts
            and len(set(inference.get("frames", []))) == 2
            and inference.get("to_us", 0) - inference.get("from_us", 0) >= 100_000
            and inference.get("from_us", 0) > self.submitted_pts
        )
        hand_ok = obs.get("hand") is not None and Counter(obs["hand"]) == self.before_hand
        return bool(
            state["turn"] > self.decision.turn
            and hand_ok
            and Counter(state["hand"] or []) == self.before_hand
            and (observed and ("self" in obs.get("passes", []) or len(set(proof)) >= 2) or own_inferred)
        )

    def advance(self, snapshot: ObservationSnapshot | None):
        """在每个新源帧上推进一次；相同帧不能凑够稳定次数，也不能反复点击。"""
        if not self.active:
            return
        now = self.clock()
        if (
            snapshot is not None
            and (snapshot.frame.serial, snapshot.frame.stream_id, snapshot.frame.width, snapshot.frame.height)
            != self.identity
        ):
            self.cancel("设备、方向或视频会话已变化")
            return
        if self.stage == "等待反馈":
            if snapshot is not None and not snapshot.allowed:
                self.cancel("提交后视频中断或画面过期，结果尚未确认")
                return
            if snapshot and snapshot.frame.frame_id != self.last_frame:
                self.last_frame = snapshot.frame.frame_id
                self.last_snapshot = snapshot
                if self._confirmed(snapshot):
                    self.stage = "已确认"
                    self._log(
                        "已确认",
                        snapshot,
                        confirmation_ms=(now - self.submitted_at) * 1000,
                        feedback=self.confirmation_evidence,
                    )
                    return
            # 等待反馈不会发出输入，允许分析短暂落后；原始时间戳不刷新。
            # 炸弹/王炸会遮住桌面牌点，最多多等2秒取得已发生动作的实际证据。
            cards = self.decision.cards
            animation = (len(cards) == 4 and len(set(cards)) == 1) or set(cards) == {"SJ", "BJ"}
            feedback_budget = self.settings["confirmation_timeout"] + (2 if animation else 0)
            if now - self.submitted_at > feedback_budget:
                self.cancel("提交后无法确认结果，禁止重复提交，请人工接管")
            return
        if now - self.started > self.settings["selection_timeout"]:
            self.cancel("选牌或按钮确认超过时间预算，请人工接管")
            return
        if (
            snapshot is None
            or not snapshot.allowed
            or now - snapshot.frame.started > self.settings["max_frame_age"]
        ):
            self.cancel("没有新鲜且允许控制的画面")
            return
        if snapshot.frame.frame_id == self.last_frame:
            return
        self.last_frame, self.last_snapshot = snapshot.frame.frame_id, snapshot
        if snapshot.version != self.decision.version:
            self.cancel("执行期间局面或行动轮次变化")
            return
        try:
            execution_context(snapshot, self.decision.strategy)
        except ObservationWait as exc:
            self.wait_since = now if self.wait_since is None else self.wait_since
            self.stable_since = None
            self.reason = str(exc)
            if now - self.wait_since > 3:
                self.cancel("等待当前画面证据超过三秒，请人工接管")
            return
        except ValueError as exc:
            self.cancel(str(exc))
            return
        self.wait_since = None
        if snapshot.control_reason or not snapshot.hand or any(c.selected is None for c in snapshot.hand):
            self.layout_wait_since = now if self.layout_wait_since is None else self.layout_wait_since
            self.stable_since = None
            self.reason = snapshot.control_reason or "无法确认选中牌"
            if now - self.layout_wait_since > 3:
                self.cancel("选牌布局连续三秒无法确认，请人工接管")
            return
        self.layout_wait_since = None
        if Counter(c.rank for c in snapshot.hand) != self.before_hand:
            self.cancel("可点击牌面与确认手牌不一致")
            return
        selected = Counter(c.rank for c in snapshot.hand if c.selected)
        responding_to_tap = self.tap_pending is not None
        if self.tap_pending:
            previous, sent_at = self.tap_pending
            if selected == previous:
                if now - sent_at > 1.0:
                    self.cancel("点击后选中状态未变化，请人工检查")
                return
            if sum((selected - previous).values()) + sum((previous - selected).values()) != 1:
                self.cancel("单次点按后多张选择意外变化，请人工接管")
                return
            self.tap_pending = None
        if self.last_selection is not None and selected != self.last_selection and not responding_to_tap:
            self.cancel("没有在途点击时选择发生变化，可能已有人工介入")
            return
        self.last_selection = selected
        target = Counter(self.decision.cards)
        extra, missing = selected - target, target - selected
        card = next((c for c in snapshot.hand if c.selected and extra[c.rank]), None)
        if card is None:
            card = next((c for c in snapshot.hand if not c.selected and missing[c.rank]), None)
        if card:
            self.stable_since = None
            self.stage = "选牌"
            self._tap("select", card.hit, snapshot)
            if self.active:
                self.tap_pending = selected, now
            return
        button = next((b for b in snapshot.buttons if b.kind == self.decision.kind and b.enabled), None)
        if button is None:
            self.stable_since = None
            self.stage = "核对"
            self.reason = "按钮缺失、遮挡或禁用，等待清晰画面"
            return
        self.stage = "核对"
        self.reason = ""
        if self.stable_since is None:
            self.stable_since = snapshot.frame.media_pts_us
            return
        if snapshot.frame.media_pts_us - self.stable_since < 100_000:
            return
        self._log("选牌已核对", snapshot, selected=list(selected.elements()))
        # 先登记提交占位。即使命令超时，也不能假定手机未收到并重新提交。
        self.sent_turns.add((self.decision.game_id, self.decision.turn))
        self.submitted_pts = snapshot.frame.media_pts_us
        self.submitted_at = now
        self.stage = "等待反馈"
        self._tap(self.decision.kind, button.hit, snapshot)

    def _tap(self, kind, rect, snapshot):
        x, y = rect.x + rect.width // 2, rect.y + rect.height // 2
        self._log("准备输入", snapshot, kind=kind, x=x, y=y)
        try:
            self.send(kind, x, y, snapshot, self.action_id)
        except InputRejected as exc:
            self._log("输入未发送", snapshot, kind=kind, message=str(exc))
            if kind in ("play", "pass"):
                self.sent_turns.discard((self.decision.game_id, self.decision.turn))
            self.cancel(f"输入在发送前被拒绝：{exc}")
            return
        except Exception as exc:
            self._log("命令结果未知", snapshot, kind=kind, message=str(exc))
            self.cancel(f"输入失败或执行结果未知：{exc}")
            return
        self._log("命令已返回", snapshot, kind=kind, x=x, y=y)
