"""与原始事件分开保存的确定性运行状态；各模式共用证明，平衡模式另支持规则接管。"""

from collections import Counter
from copy import deepcopy
from dataclasses import asdict, replace
from hashlib import sha256
from threading import RLock

from doudizhu_lab.agents.rules import RuleRound, role_for
from doudizhu_lab.domain.bot_types import SnapshotSequence, encode
from doudizhu_lab.domain.inference import deductions, frame_context, make_proof, matches, verify_proof
from doudizhu_lab.domain.round_evidence import resolve_round_binding
from doudizhu_lab.domain.table import TableObservation
from doudizhu_lab.domain.tracker import GameTracker, apply_event


class OperationalObserver:
    """第二套公开状态，不覆盖严格观察或严格事件。

    strict、infer、balanced共用可独立复核的第二版确定性证明。balanced另外允许在
    没有在途提交、没有采集中断且当前回合可重新确认时采用局部规则；新局重新建立完整模型状态。
    """

    def __init__(self, profile):
        self.policy = profile.continuous_policy
        self.tracker = GameTracker(profile.seat_order, config=profile.tracker_settings)
        self.history, self.new_events, self.notices = [], [], []
        self.sequence = SnapshotSequence()
        self.rule = None
        self.open_submission = None
        self.submission_pts = None
        self.lead_signature = profile.bot_controls.get("lead_signature")
        self.inference_probe, self.inferred = None, {}
        self.last_frame = ""
        self.anchors, self.deal_anchors = [], []
        self.raw_anchors, self.raw_deal_anchors = [], []
        self.pending_proofs = {}
        self.round_context = None
        self.round_binding = self.opening = None
        self.binding_error = ""
        self.boundary_observations = {}
        self.lock = RLock()

    def _bind_round(self, snapshot):
        """仅在首个动作以前关联原始开局；确认迟到时继续等证，不沿用上一局编号。"""
        if self.opening is None or self.binding_error:
            return
        context = snapshot.observation.get("frame_context", {})
        if self.round_binding is not None:
            if context != self.round_context and not self.tracker.state.ended:
                self.binding_error = "已绑定牌局的原始编号、设备或视频流发生变化"
            return
        if self.tracker.state.turn or self.tracker._active_started or self.open_submission is not None:
            self.binding_error = "出牌开始前未能唯一绑定原始发牌边界"
            return
        binding = resolve_round_binding(self.opening, snapshot.history, self.boundary_observations)
        if binding is None:
            return
        self.round_binding, self.round_context = binding, dict(binding.frame_context)
        # 绑定前的观察可能仍带旧局号，不能用于新局的双帧证明。
        self.anchors, self.deal_anchors = [], []
        self.raw_anchors, self.raw_deal_anchors = [], []
        self.pending_proofs.clear()

    def reject_submission(self, decision):
        """只在底层明确保证未发送时撤销占位；命令超时或结果未知不能调用。

        同时清理规则视图的在途意图，避免一次被拒绝的点击阻止后续新帧重新核对。
        """
        with self.lock:
            if self.open_submission == decision:
                self.open_submission = self.submission_pts = None
                if self.rule:
                    self.rule.intent = self.rule.expected_hand = None

    @staticmethod
    def _anchor_pair(buffer, raw):
        """保留短期原始帧并选取相隔至少100毫秒的最近双帧。

        高帧率下相邻两帧可能仅相隔几十毫秒，不能一直只检查相邻帧；当前帧始终是
        后锚点，绝不将旧双帧的结果套到更新的画面。设备、视频流、分局或时间倒退时清空。
        """
        if buffer and (
            raw["pts_us"] <= buffer[-1]["pts_us"]
            or raw.get("frame_context") != buffer[-1].get("frame_context")
        ):
            buffer.clear()
        buffer.append(raw)
        buffer[:] = [o for o in buffer[-512:] if raw["pts_us"] - o["pts_us"] <= 8_000_000]
        previous = next((o for o in reversed(buffer[:-1]) if raw["pts_us"] - o["pts_us"] >= 100_000), None)
        return [previous, raw] if previous is not None else [raw]

    def _augment(self, obs):
        """只补充已求出唯一答案的运行视图；原始锚点先复制保存，绝不回写截图观察。"""
        raw = asdict(obs)
        self.anchors = self._anchor_pair(self.raw_anchors, raw)
        if obs.phase in ("dealing", "bidding") and len(obs.hand or []) == 17:
            self.deal_anchors = self._anchor_pair(self.raw_deal_anchors, raw)
        if self.opening is not None and (self.round_binding is None or self.binding_error):
            return
        if self.round_context is not None and obs.frame_context != self.round_context:
            return  # 仍让直接观察器识别新局，但不跨原始分局边界拼接证明。
        for claim in deductions(self.tracker.state, self.anchors, deal_values=self.deal_anchors):
            kind, seat = claim["kind"], claim.get("seat")
            if kind == "setup":
                for key, value in claim["data"].items():
                    if key != "initial_hand":
                        setattr(obs, key, value)
            elif kind == "play":
                weak = [c for c in self.tracker._candidates if c["seat"] == seat]
                if (
                    weak
                    and not any(c["ready"] for c in weak)
                    and any(c["kind"] == "play" and c["after_count"] != obs.remaining[seat] for c in weak)
                ):
                    # 动画遮住手牌可能先产生错误的少牌候选。完整双帧已经唯一证明真实动作时，
                    # 撤回仍未获牌面支持的弱候选，从可信前缀重建差分；不能覆盖任何已就绪证据。
                    self.tracker._candidates = [c for c in self.tracker._candidates if c not in weak]
                    self.tracker._counts[seat] = self.tracker.state.remaining[seat]
                    if seat == "self":
                        self.tracker._hand = deepcopy(self.tracker.state.hand)
                    presence = self.tracker._presence.get(seat)
                    if presence and presence.get("candidate") in weak:
                        presence.update(candidate=None, consumed=True, consumed_turn=self.tracker.state.turn)
                if obs.phase == "settlement" and obs.remaining.get(seat) == 0:
                    # 结算已经覆盖最后出牌区时，先把已证明的末手交给合法动作检查。
                    # 胜者标记仍保留，跟踪器确认末手后立即发出结束事件；原始阶段不变。
                    obs.phase = "playing"
                if all(
                    Counter(o.get("plays", {}).get(seat) or []) == Counter(claim["cards"])
                    for o in self.anchors
                ):
                    continue
                obs.plays[seat] = list(claim["cards"])
            elif kind == "pass":
                if all(seat in o.get("passes", []) for o in self.anchors):
                    continue
                if seat not in obs.passes:
                    obs.passes.append(seat)
                obs.pass_labels[seat] = "pass"
            self.pending_proofs[kind, seat] = (deepcopy(self.anchors), deepcopy(self.deal_anchors))

    def note_submission(self, decision, pts_us):
        """登记单次提交的在途意图；未知结果时禁止用策略切换绕开确认。"""
        with self.lock:
            self.open_submission = decision
            self.submission_pts = pts_us
            if self.rule and decision.strategy == "rules":
                self.rule.submitted(decision, pts_us)

    def note_confirmation(self, decision):
        """执行器取得真实反馈后清除在途；命令返回、建议和选牌完成都不能调用本接口。"""
        with self.lock:
            if (
                self.open_submission
                and self.open_submission.game_id == decision.game_id
                and self.open_submission.turn == decision.turn
            ):
                self.open_submission = None
                self.submission_pts = None
            if self.rule and decision.strategy == "rules":
                self.rule.confirmed(decision)

    def _infer_passes(self, obs):
        """兼容旧开发用例的第一版不出视图；生产transform仅调用统一的_augment证明路径。"""
        state = self.tracker.state
        if state.pending or state.capture_incomplete or not state.last_play or not state.current_actor:
            self.inference_probe = None
            return
        actor = obs.actor
        if actor not in state.seat_order or actor == state.current_actor:
            self.inference_probe = None
            return
        if obs.hand is None or Counter(obs.hand) != Counter(state.hand or []):
            self.inference_probe = None
            return
        if any(
            obs.remaining.get(s) is None or obs.remaining[s] != state.remaining[s] for s in state.seat_order
        ):
            self.inference_probe = None
            return
        index = state.seat_order.index(state.current_actor)
        distance = (state.seat_order.index(actor) - index) % 3
        seats = [state.seat_order[(index + i) % 3] for i in range(distance)]
        if not seats or len(seats) > 2 - state.passes:
            return
        for seat in seats:
            visible = obs.plays.get(seat)
            own_pass = bool(
                seat == "self"
                and self.open_submission is not None
                and self.open_submission.kind == "pass"
                and self.open_submission.game_id == state.game_id
                and self.open_submission.turn == state.turn
                and self.submission_pts is not None
                and obs.pts_us > self.submission_pts
            )
            previous = next(
                (
                    e["cards"]
                    for e in reversed(self.history)
                    if e.get("seat") == seat and e["kind"] in ("play", "pass")
                ),
                [],
            )
            if (visible is None and not own_pass) or visible and Counter(visible) != Counter(previous):
                # 新牌动画或牌区未知时，旧计数可能尚未刷新，不能把它解释成不出。
                self.inference_probe = None
                return
        signature = (
            state.game_id,
            state.turn,
            actor,
            tuple(sorted(obs.hand)),
            tuple(obs.remaining[s] for s in state.seat_order),
        )
        if not self.inference_probe or self.inference_probe[0] != signature:
            self.inference_probe = signature, obs.pts_us, [obs.frame_id]
            return
        frames = self.inference_probe[2]
        if obs.frame_id not in frames:
            frames.append(obs.frame_id)
        if obs.pts_us - self.inference_probe[1] < 100_000 or len(frames) < 2:
            return
        for seat in seats:
            if seat not in obs.passes:
                obs.passes.append(seat)
                obs.pass_labels[seat] = "pass"
                self.inferred[seat] = {
                    "basis": "明确行动者前进，三家剩牌数与本人手牌不变，合法轮转唯一",
                    "frames": frames[-2:],
                    "from_us": self.inference_probe[1],
                    "to_us": obs.pts_us,
                    "remaining": dict(obs.remaining),
                    "actor": actor,
                    "own_pass_after_submission": bool(
                        seat == "self"
                        and self.open_submission is not None
                        and self.open_submission.kind == "pass"
                    ),
                    "submission_pts_us": self.submission_pts if seat == "self" else None,
                }
                obs.marker_evidence["inferred_pass_" + seat] = self.inferred[seat] | {"accepted": True}

    def transform(self, snapshot):
        """从原始同帧观察生成执行快照；只替换运行视图，原始输入对象和事件不变。"""
        with self.lock:
            return self._transform(snapshot)

    def _transform(self, snapshot):
        if snapshot.frame.frame_id == self.last_frame:
            return self.latest
        self.last_frame = snapshot.frame.frame_id
        obs = TableObservation(**snapshot.observation)
        obs.frame_context = obs.frame_context or frame_context(snapshot.frame)
        self.boundary_observations[obs.frame_id] = asdict(obs)
        # 只保留有界开局查找窗口；已经确认的绑定独立保存，不依赖此缓存继续存在。
        if len(self.boundary_observations) > 1024:
            self.boundary_observations.pop(next(iter(self.boundary_observations)))
        self._bind_round(snapshot)
        if snapshot.state["capture_incomplete"]:
            self.anchors, self.deal_anchors = [], []
            self.raw_anchors, self.raw_deal_anchors = [], []
            self.pending_proofs.clear()
            self.latest = snapshot
            return snapshot
        self._augment(obs)
        deadline = self.submission_pts + 8_000_000 if self.submission_pts is not None else None
        if (
            obs.phase == "settlement"
            and self.open_submission is not None
            and deadline is not None
            and obs.pts_us <= deadline
            and self.tracker.state.phase == "playing"
            and not any(v == 0 for v in self.tracker.state.remaining.values())
        ):
            # 已提交但结果未知时，结算页不能先把尚待核验的末手冲成缺口；只读等待
            # 最多八秒。模型意图不补牌，缺少双帧零牌及末手证据仍不能确认或计为完成。
            obs.phase = "playing"
        before = deepcopy(self.tracker.state)
        update = self.tracker.observe(obs, action_deadline_us=deadline)
        for event in update.events:
            if event.kind == "game_start":
                self.history = []
                self.inferred.clear()
                self.inference_probe = None
                self.open_submission = None
                self.submission_pts = None
                raw = asdict(TableObservation(**snapshot.observation))
                raw["frame_context"] = dict(obs.frame_context)
                self.opening = asdict(event)
                self.round_binding = self.round_context = None
                self.binding_error = ""
                self._bind_round(snapshot)
                self.anchors = [raw]
                self.raw_anchors = [raw]
                self.deal_anchors = (
                    [raw] if obs.phase in ("dealing", "bidding") and len(obs.hand or []) == 17 else []
                )
                self.raw_deal_anchors = deepcopy(self.deal_anchors)
                self.pending_proofs.clear()
            event.source = "operational_observed"
            pending = self.pending_proofs.get((event.kind, event.seat))
            if pending and event.confirmed:
                values, deal = pending
                for claim in deductions(before, values, deal_values=deal):
                    if matches(asdict(event), claim):
                        event.data["inference"] = make_proof(before, claim, values, deal_values=deal)
                        raw_map = {o["frame_id"]: o for o in [*values, *deal]}
                        if not verify_proof(
                            asdict(event), before, raw_map, expected_context=self.round_context
                        ):
                            raise ValueError("运行时确定性证明无法独立复核")
                        event.source = "operational_inferred"
                        if self.round_binding is not None:
                            event.data["round_binding"] = asdict(self.round_binding)
                        self.pending_proofs.pop((event.kind, event.seat), None)
                        break
            if event.confirmed:
                self.history.append(asdict(event))
            self.new_events.append(asdict(event))
            before = apply_event(before, event)
        state = asdict(update.state)
        if self.opening is not None or not state["game_id"]:
            state["round_binding_ready"] = self.round_binding is not None and not self.binding_error
            state["round_binding_error"] = self.binding_error
            if self.binding_error:
                state["pending"] = True
                state["problems"] = [*state["problems"], self.binding_error]
        if self.rule and self.rule.state["game_id"] != state["game_id"]:
            self.rule = None
            self.open_submission = None
            self.submission_pts = None
        # 在完整前缀内自动学习正向领出按钮位置，后续接管不能仅凭空牌区猜领出。
        if (
            not state["pending"]
            and state["current_actor"] == "self"
            and not state["last_play"]
            and obs.actor == "self"
        ):
            play = [b for b in snapshot.buttons if b.kind == "play"]
            if len(play) == 1 and not any(b.kind == "pass" for b in snapshot.buttons):
                button = play[0]
                self.lead_signature = {
                    "center": [
                        (button.hit.x + button.hit.width / 2) / snapshot.frame.width,
                        (button.hit.y + button.hit.height / 2) / snapshot.frame.height,
                    ],
                    "frame_id": snapshot.frame.frame_id,
                    "pts_us": snapshot.frame.media_pts_us,
                    "basis": "完整可信前缀中的本人领出布局",
                }
        if (
            state["pending"]
            and self.policy == "balanced"
            and self.rule is None
            and self.open_submission is None
        ):
            try:
                role_for(state)
                if not state["initial_hand"] or not state["bottom"] or state["hand"] is None:
                    raise ValueError("开局不完整")
            except ValueError:
                pass
            else:
                self.rule = RuleRound(state, lead_signature=self.lead_signature)
                self.notices.append(
                    {
                        "kind": "rule_takeover",
                        "game_id": state["game_id"],
                        "frame_id": snapshot.frame.frame_id,
                        "pts_us": snapshot.frame.media_pts_us,
                        "reason": list(state["problems"]),
                    }
                )
        text, history = encode(state), encode(self.history)
        result = replace(
            snapshot,
            state_json=text,
            history_json=history,
            observation_json=encode(obs),
            version=sha256((text + history).encode()).hexdigest(),
            pending=bool(update.pending_candidates),
        )
        if self.rule:
            result = self.rule.project(result)
        self.latest = self.sequence.publish(result)
        return self.latest

    def drain(self):
        """取走本轮运行事件与降级通知，供独立账本追加；不会清除原始识别历史。"""
        with self.lock:
            events, notices = self.new_events, self.notices
            self.new_events, self.notices = [], []
            return events, notices
