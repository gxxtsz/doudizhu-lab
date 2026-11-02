"""明确区分旧录像建议检查和模拟设备闭环；任何入口都没有真实设备输入能力。"""

from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path
from random import Random
from time import monotonic

import numpy as np

from doudizhu_lab._vendor.douzero.env.env import DummyAgent
from doudizhu_lab._vendor.douzero.env.game import GameEnv
from doudizhu_lab.agents.backends import create_agent
from doudizhu_lab.agents.douzero import DECK, ROLES, TO_RANK, Agent, context
from doudizhu_lab.domain.bot_types import ActionButton, HandCard, ObservationSnapshot, SnapshotSequence
from doudizhu_lab.domain.models import Frame, PixelRect
from doudizhu_lab.domain.table import TableObservation
from doudizhu_lab.domain.tracker import GameTracker
from doudizhu_lab.runtime.executor import ActionExecutor
from doudizhu_lab.storage.session import SessionStore


class SimulatedDevice:
    """拥有完整模拟发牌的设备后端，向观察器只暴露自己手牌、公共牌和行动提示。

    点击通过当前坐标命中选牌或按钮，提交实际选中牌给独立上游游戏环境。
    GameTracker 消费随后稳定的观察产生事件；绝不把执行器的建议直接写入可信状态。
    """

    def __init__(
        self, seed: int, role: str, *, stream_id="sim", pts_offset=0, dealt=None, landlord_seat=None
    ):
        """可选dealt为模拟器私有的三家17张牌和底牌；只用于生成真实环境，不传给决策器。

        叫抢模拟先固定整副发牌，再由实际点击确定地主，避免先决定身份、事后伪造叫抢。
        旧调用不传此参数时保持原有固定身份与固定种子的回归行为。
        """
        self.rng = Random(seed)
        self.stream_id = stream_id
        if dealt is not None:
            if (
                landlord_seat not in ("self", "right", "left")
                or set(dealt) != {"self", "right", "left", "bottom"}
                or any(len(dealt[s]) != (3 if s == "bottom" else 17) for s in dealt)
                or Counter(c for cards in dealt.values() for c in cards) != DECK
            ):
                raise ValueError("模拟叫抢后的发牌不合法")
            role = ROLES[(-["self", "right", "left"].index(landlord_seat)) % 3]
        self.roles = {
            "self": role,
            "right": ROLES[(ROLES.index(role) + 1) % 3],
            "left": ROLES[(ROLES.index(role) + 2) % 3],
        }
        self.seats = {v: k for k, v in self.roles.items()}
        self.env = GameEnv({r: DummyAgent(r) for r in ROLES})
        deck = list(DECK.elements())
        self.rng.shuffle(deck)
        self.initial = {
            "landlord": sorted(deck[:20]),
            "landlord_down": sorted(deck[20:37]),
            "landlord_up": sorted(deck[37:]),
        }
        self.bottom = sorted(deck[17:20])
        if dealt is not None:
            self.bottom = sorted(dealt["bottom"])
            self.initial = {
                self.roles[s]: sorted(dealt[s] + (self.bottom if s == landlord_seat else []))
                for s in self.roles
            }
        # 环境会原地移除已出的牌；保留私有初始发牌副本用于检查叫抢后的17→20守恒。
        self.env.card_play_init(
            {role: cards.copy() for role, cards in self.initial.items()}
            | {"three_landlord_cards": self.bottom.copy()}
        )
        self.tracker = GameTracker(["self", "right", "left"])
        self.history, self.audits = [], []
        self.plays = {s: [] for s in self.roles}
        self.passes = set()
        self.selected = set()
        self.now = monotonic()
        self.pts = 1_000_000 + pts_offset
        self.last = None
        self.sequence = SnapshotSequence()
        self.submissions = set()
        self.invalid = self.duplicate = self.taps = 0
        self.fault = ""
        self.hide_passes = False
        self.hidden_play_until_us = 0
        self.button_dx, self.buttons_enabled = 0, True
        self.setup_frames = []
        for phase in ("bidding", "doubling", "playing"):
            for _ in range(3):
                self.setup_frames.append(self.render(phase=phase))

    def render(self, *, phase="playing"):
        """推进一帧源时钟并运行生产 GameTracker；选牌只改变布局，不改变历史事实。"""
        self.now += 0.12
        self.pts += 120_000
        hand = self.env.info_sets[self.roles["self"]].player_hand_cards
        shown = hand.copy()
        if phase == "bidding" and self.roles["self"] == "landlord":
            shown = list((Counter(hand) - Counter(self.bottom)).elements())
        actor = self.seats[self.env.acting_player_position] if not self.env.game_over else None
        evidence = {}
        if self.env.game_over:
            winner = next(s for s, r in self.roles.items() if not self.env.info_sets[r].player_hand_cards)
            evidence[f"winner_{winner}"] = {"accepted": True}
        obs = TableObservation(
            f"{self.stream_id}-{self.pts}",
            self.pts,
            phase=phase,
            landlord=self.seats["landlord"] if phase != "bidding" else None,
            bottom=[TO_RANK[n] for n in self.bottom] if phase != "bidding" else None,
            hand=[TO_RANK[n] for n in shown],
            remaining={s: len(self.env.info_sets[r].player_hand_cards) for s, r in self.roles.items()},
            actor=actor,
            plays={s: [TO_RANK[n] for n in c] for s, c in self.plays.items()},
            passes=list(self.passes),
            pass_labels={s: "不出" for s in self.passes},
            marker_evidence=evidence,
        )
        if phase == "playing" and self.hide_passes:
            obs.passes, obs.pass_labels = [], {}
        if phase == "playing" and obs.pts_us < self.hidden_play_until_us:
            for seat in ("left", "right"):
                if obs.plays[seat]:
                    obs.plays[seat] = None
        update = self.tracker.observe(obs)
        for event in update.events:
            if event.kind == "game_start":
                self.history.clear()
            if event.confirmed:
                self.history.append(asdict(event))
        frame = Frame(
            "simulation",
            np.zeros((100, 600, 3), np.uint8),
            self.now,
            str(self.pts),
            0,
            "simulation",
            obs.frame_id,
            self.stream_id,
            self.pts,
        )
        snapshot = ObservationSnapshot.create(
            frame,
            None,
            obs,
            update.state,
            self.history,
            pending=bool(update.pending_candidates),
            allowed=True,
        )
        layout = tuple(
            HandCard(TO_RANK[n], i, PixelRect(i * 25 + 5, 65, 15, 20), i in self.selected)
            for i, n in enumerate(hand)
        )
        buttons = (
            ActionButton("play", PixelRect(420 + self.button_dx, 10, 70, 30), self.buttons_enabled, "出牌"),
            ActionButton("pass", PixelRect(510 + self.button_dx, 10, 70, 30), self.buttons_enabled, "不出"),
        )
        self.last = self.sequence.publish(replace(snapshot, hand=layout, buttons=buttons, control_reason=""))
        return self.last

    def act(self, cards):
        """模拟游戏接受动作后的可见变化；是否接受由上游规则环境验证。"""
        role = self.env.acting_player_position
        seat = self.seats[role]
        self.env.players[role].set_action(cards)
        self.env.step()
        self.plays[seat] = cards.copy()
        self.passes.discard(seat)
        if not cards:
            self.passes.add(seat)
        if not self.env.game_over:
            next_seat = self.seats[self.env.acting_player_position]
            self.plays[next_seat] = []
            self.passes.discard(next_seat)
        if seat == "self":
            self.selected.clear()

    def send(self, kind, x, y, snapshot, action_id):
        """命中当前布局，不能读取执行器的目标牌；支持无反馈与错误选牌故障注入。"""
        self.taps += 1
        if self.fault == "disconnect":
            raise OSError("模拟设备已断线")
        # 设备只以实际坐标命中为准。kind 是执行器的操作意图，不能让它绕过
        # 按钮位置、可用状态或命中校验直接改变模拟牌局。
        target_button = next(
            (
                b
                for b in self.last.buttons
                if b.enabled
                and b.hit.x <= x < b.hit.x + b.hit.width
                and b.hit.y <= y < b.hit.y + b.hit.height
            ),
            None,
        )
        target_card = next(
            (
                c
                for c in self.last.hand
                if c.hit.x <= x < c.hit.x + c.hit.width and c.hit.y <= y < c.hit.y + c.hit.height
            ),
            None,
        )
        if not target_button and not target_card:
            return
        kind = target_button.kind if target_button else "select"
        if kind == "select":
            if self.fault == "missed_click":
                return
            target = next(
                (
                    c.index
                    for c in self.last.hand
                    if c.hit.x <= x < c.hit.x + c.hit.width and c.hit.y <= y < c.hit.y + c.hit.height
                ),
                None,
            )
            if target is None:
                raise ValueError("模拟选牌未命中")
            if self.fault == "wrong_card":
                target = (target + 1) % len(self.last.hand)
                self.fault = ""
            self.selected.symmetric_difference_update({target})
            return
        if self.fault == "lost_submission":
            return
        if action_id in self.submissions:
            self.duplicate += 1
            raise ValueError("模拟游戏发现重复提交")
        self.submissions.add(action_id)
        hand = self.env.info_sets[self.roles["self"]].player_hand_cards
        cards = sorted(hand[i] for i in self.selected) if kind == "play" else []
        if (
            self.env.acting_player_position != self.roles["self"]
            or cards not in self.env.game_infoset.legal_actions
        ):
            self.invalid += 1
            raise ValueError("模拟游戏拒绝非法动作")
        self.act(cards)


def simulate(agent, games):
    """固定种子按三种身份轮换，生产模型、执行器和状态机通过模拟坐标反馈形成闭环。"""
    reports = []
    for number in range(games):
        role = ROLES[number % 3]
        device = SimulatedDevice(73100 + number, role)
        executor = ActionExecutor(device.send, device.audits.append, clock=lambda: device.now)
        decisions, failure = [], ""
        for _ in range(400):
            if device.env.game_over:
                break
            if device.tracker.state.pending:
                failure = "模拟观察产生疑点：" + "；".join(device.tracker.state.problems)
                break
            if device.env.acting_player_position == role:
                try:
                    decision = agent.decide(device.last)
                    decisions.append(asdict(decision))
                    executor.start(decision, device.last)
                    for _ in range(140):
                        executor.advance(device.render())
                        if not executor.active:
                            break
                    if executor.stage != "已确认":
                        failure = executor.reason or executor.stage
                        break
                except ValueError as exc:
                    failure = str(exc)
                    break
            else:
                action = device.rng.choice(device.env.game_infoset.legal_actions)
                device.act(action)
                for _ in range(3):
                    device.render()
        for _ in range(3):
            device.render()
        valid = bool(
            device.env.game_over
            and device.tracker.state.complete
            and not failure
            and not device.invalid
            and not device.duplicate
        )
        reports.append(
            {
                "seed": 73100 + number,
                "role": role,
                "passed": valid,
                "failure": failure,
                "actions": device.tracker.state.turn,
                "decisions": decisions,
                "invalid_submissions": device.invalid,
                "duplicate_submissions": device.duplicate,
                "taps": device.taps,
                "audit": device.audits,
            }
        )
    return {
        "source": "simulation",
        "games": reports,
        "passed": games >= 30
        and all(r["passed"] for r in reports)
        and all(sum(r["role"] == role for r in reports) >= 10 for role in ROLES),
        "note": "模拟观察验证状态机与坐标反馈，不替代真实像素识别和手机输入验证",
    }


def replay_advice(agent: Agent, folder: Path, scope: Path) -> dict:
    """只读消费原观察；scope为预先指定边界，绝不进入模型输入或修订原数据。

    返回合法建议与缺口拒绝报告，不评估不同出牌的后续胜负。读取或格式错误上抛；
    只能在离线线程调用，不接受设备输入对象。已保存原始观察和快照不会被改写。
    """
    from doudizhu_lab.evaluation.scope import load_scope

    scope_value = load_scope(scope)
    session = SessionStore(folder, readonly=True)
    try:
        meta = session.metadata()
        observations = session.rows("observations")
    finally:
        session.close()
    tracker = GameTracker(meta["profile"]["seat_order"], config=meta["profile"].get("tracker_settings"))
    history, games, seen = [], {}, set()
    sequence = SnapshotSequence()
    for raw in observations:
        obs = TableObservation(**raw)
        update = tracker.observe(obs)
        for event in update.events:
            if event.kind == "game_start":
                history = []
            if event.confirmed:
                history.append(asdict(event))
        state = update.state
        if not state.game_id:
            continue
        report = games.setdefault(
            state.game_id,
            {
                "game_id": state.game_id,
                "decisions": [],
                "blocked": {},
                "pending_seen": False,
                "complete": False,
                "first_us": obs.pts_us,
                "last_us": obs.pts_us,
                "decisions_after_gap": 0,
            },
        )
        report["pending_seen"] |= state.pending
        report["complete"] = state.complete
        report["last_us"] = obs.pts_us
        if state.current_actor != "self" or state.phase != "playing":
            continue
        # 这里只检查保存的语义观察，明确使用无图像的 replay 来源，不能传到真实输入接口。
        frame = Frame(
            "replay",
            np.zeros((1, 1, 3), np.uint8),
            monotonic(),
            obs.frame_id,
            0,
            "replay",
            obs.frame_id,
            "replay",
            obs.pts_us,
        )
        snap = ObservationSnapshot.create(
            frame, None, obs, state, history, pending=bool(update.pending_candidates)
        )
        snap = sequence.publish(snap)
        try:
            context(snap)
        except ValueError as exc:
            key = str(exc)
            report["blocked"][key] = report["blocked"].get(key, 0) + 1
            continue
        key = state.game_id, state.turn
        if key in seen:
            continue
        seen.add(key)
        decision = agent.decide(snap)
        report["decisions_after_gap"] += int(report["pending_seen"])
        report["decisions"].append(
            asdict(decision) | {"source_frame": obs.frame_id, "source_pts_us": obs.pts_us}
        )
    ordered = list(games.values())
    # 只按先前已经人工核对的八个连续时间区间选局，包含有真实缺口的第七局。
    # 这份边界清单不提供事件、底牌或手牌，也绝不进入 context/decide。
    scope = scope_value
    selected, used = [], set()
    for interval in scope["games"]:
        candidates = [
            (max(0, min(g["last_us"], interval["end_us"]) - max(g["first_us"], interval["start_us"])), g)
            for g in ordered
            if g["game_id"] not in used
        ]
        overlap, match = max(candidates, key=lambda x: x[0], default=(0, None))
        matched = match if overlap >= 0.8 * (interval["end_us"] - interval["start_us"]) else None
        if matched:
            used.add(matched["game_id"])
        good = bool(
            matched
            and not matched["decisions_after_gap"]
            and (
                (matched["pending_seen"] and not matched["complete"])
                if interval["incomplete"]
                else (matched["complete"] and matched["decisions"] and not matched["pending_seen"])
            )
        )
        selected.append(
            {
                "reference": interval,
                "matched_game_id": matched["game_id"] if matched else None,
                "passed": good,
            }
        )
    return {
        "source": "saved_observations_replay",
        "session": str(folder),
        "games": ordered,
        "selected_games": selected,
        "selection_source": scope["source"],
        "passed": bool(selected)
        and len(selected) == len(scope["games"])
        and all(g["passed"] for g in selected),
        "note": "建议合法性与拒绝缺口检查；不计算模型建议与人工出牌的一致率，不推断其他决策的后续胜负",
    }


def evaluate_bot(
    store, *, mode: str, session: Path | None = None, games: int = 30, scope: Path | None = None
) -> dict:
    """命令行统一入口；只创建模型进程和模拟输入对象，不导入或调用设备输入方法。"""
    agent = create_agent(store.root, store.profile.play_model)
    try:
        agent.start()
        if mode == "simulation":
            if not 1 <= games <= 10000:
                raise ValueError("模拟局数必须为1～10000")
            report = simulate(agent, games)
        else:
            if session is None or scope is None:
                raise ValueError("录像建议评测需要 --session 和 --scope")
            report = replay_advice(agent, session, scope)
        report.update(model_id=agent.model_id, play_model=agent.backend, live_verified=False)
        return report
    finally:
        agent.close()
