"""平衡运行状态：与严格原始事件分开保存的唯一推断和规则接管。"""

from collections import Counter
from dataclasses import asdict, replace
from hashlib import sha256
from threading import RLock

from doudizhu_lab.agents.rules import RuleRound, role_for
from doudizhu_lab.domain.bot_types import SnapshotSequence, encode
from doudizhu_lab.domain.table import TableObservation
from doudizhu_lab.domain.tracker import GameTracker


class OperationalObserver:
    """第二套公开状态，不覆盖严格观察或严格事件。

    infer只接受计数守恒与明确行动者形成的唯一不出推断；balanced另外允许在没有在途
    提交、没有采集中断且当前回合可重新确认时采用局部规则。每局重新尝试完整WP状态。
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
        self.lock = RLock()

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
        if self.policy == "strict" or snapshot.frame.frame_id == self.last_frame:
            return snapshot if self.policy == "strict" else self.latest
        self.last_frame = snapshot.frame.frame_id
        obs = TableObservation(**snapshot.observation)
        if snapshot.state["capture_incomplete"]:
            self.latest = snapshot
            return snapshot
        self._infer_passes(obs)
        update = self.tracker.observe(obs)
        for event in update.events:
            if event.kind == "game_start":
                self.history = []
                self.inferred.clear()
                self.inference_probe = None
                self.open_submission = None
                self.submission_pts = None
            if event.kind == "pass" and event.seat in self.inferred:
                event.source = "operational_inferred"
                event.data["inference"] = self.inferred.pop(event.seat)
            else:
                event.source = "operational_observed"
            if event.confirmed:
                self.history.append(asdict(event))
            self.new_events.append(asdict(event))
        state = asdict(update.state)
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
