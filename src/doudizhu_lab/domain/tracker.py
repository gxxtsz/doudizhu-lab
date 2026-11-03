"""保守牌局状态机与纯函数重放。仅使用公开画面；不会调用 Qt、ADB 或模型。"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from hashlib import sha256

from doudizhu_lab.domain.models import RANKS, TrackerSettings
from doudizhu_lab.domain.table import SEAT_NAMES, SEATS, TableObservation


def canonical(cards):
    """牌点多重集合的稳定排序；重复张数保留，不受选牌上移或展示次序影响。"""
    return sorted(cards, key=RANKS.index) if cards is not None else None


@dataclass
class GameEvent:
    """单个事实或候选。confirmed=False 表示尚不可进入可信状态，仍保留录像证据。"""

    event_id: str
    game_id: str
    pts_us: int
    kind: str
    frame_id: str
    seat: str | None = None
    cards: list[str] = field(default_factory=list)
    data: dict = field(default_factory=dict)
    confirmed: bool = True
    source: str = "automatic"


@dataclass
class GameState:
    """最近可信事件前缀的状态；pending 后观察仍继续保存，但不偷偷更新为可信值。"""

    game_id: str = ""
    phase: str = "unknown"
    landlord: str | None = None
    initial_hand: list[str] | None = None
    hand: list[str] | None = None
    bottom: list[str] | None = None
    remaining: dict[str, int | None] = field(default_factory=lambda: dict.fromkeys(SEATS))
    current_actor: str | None = None
    last_play: list[str] = field(default_factory=list)
    last_player: str | None = None
    passes: int = 0
    turn: int = 0
    seat_order: list[str] = field(default_factory=list)
    played: dict[str, list[str]] = field(default_factory=lambda: {s: [] for s in SEATS})
    pending: bool = False
    capture_incomplete: bool = False
    ended: bool = False
    complete: bool = False
    problems: list[str] = field(default_factory=list)


def _problem(state, text):
    state.pending = True
    state.complete = False
    if text not in state.problems:
        state.problems.append(text)


def _next(state, seat):
    if len(state.seat_order) != 3 or seat not in state.seat_order:
        return None
    return state.seat_order[(state.seat_order.index(seat) + 1) % 3]


def apply_event(state: GameState, event: GameEvent | dict) -> GameState:
    """确定性地应用一条事件，返回新状态；校验失败冻结可信前缀并保留原因。

    人工改写不绕过牌点、牌数、轮转、手牌归属及牌张上限检查。
    game_start 是唯一重置边界；跨局事件不能混入已有状态。
    """
    e = asdict(event) if isinstance(event, GameEvent) else event
    result = deepcopy(state)
    kind, data = e.get("kind"), e.get("data", {})
    if kind == "game_start":
        result = GameState(
            game_id=e["game_id"], capture_incomplete=bool(data.get("capture_incomplete", True))
        )
    elif e.get("game_id") != state.game_id:
        _problem(result, "事件所属牌局不一致")
        return result
    if not e.get("confirmed", True):
        _problem(result, "存在未确认事件")
        return result
    if result.ended and kind not in ("game_start", "gap", "game_end"):
        _problem(result, "结算后的事件不能继续写入上一局，请调整分局边界")
        return result
    if kind == "gap":
        _problem(result, data.get("message", "存在观察缺口"))
        if data.get("capture_lost"):
            result.capture_incomplete = True
        return result
    if kind == "game_end":
        result.ended = True
        result.phase = "settlement"
        result.complete = bool(
            not result.pending
            and not result.capture_incomplete
            and result.initial_hand is not None
            and len(result.initial_hand) == (20 if result.landlord == "self" else 17)
            and result.landlord
            and result.bottom
            and len(result.seat_order) == 3
            and any(v == 0 for v in result.remaining.values())
            and result.hand is not None
            and result.remaining["self"] == len(result.hand)
            and all(
                result.remaining[seat] is not None
                and result.remaining[seat] + len(result.played[seat])
                == (20 if seat == result.landlord else 17)
                for seat in SEATS
            )
        )
        return result
    if result.pending:
        return result
    if kind in ("game_start", "setup"):
        # 先在副本中校验全部开局字段；失败时保留进入本事件前的可信内容。
        previous = deepcopy(result)
        hand = data.get("initial_hand")
        if hand is not None and (
            any(c not in RANKS for c in hand)
            or len(hand) > 20
            or any(n > (1 if c in ("SJ", "BJ") else 4) for c, n in Counter(hand).items())
        ):
            _problem(result, "初始手牌含无效牌点或重复张数超限")
            return result
        if hand is not None and result.turn:
            _problem(result, "出牌后不能重新初始化手牌，请修订开局信息")
            return result
        for key in ("phase", "landlord", "bottom", "initial_hand", "seat_order", "current_actor"):
            if key in data and data[key] is not None:
                setattr(result, key, deepcopy(data[key]))
        if "initial_hand" in data and data["initial_hand"] is not None:
            result.hand = canonical(data["initial_hand"])
        if "remaining" in data:
            if any(key not in SEATS for key in data["remaining"]):
                _problem(previous, "剩余牌数的玩家标识无效")
                return previous
            result.remaining.update({k: v for k, v in data["remaining"].items() if v is not None})
        if result.landlord not in (*SEATS, None):
            _problem(result, "地主座位无效")
        if result.bottom is not None and (
            len(result.bottom) != 3 or any(c not in RANKS for c in result.bottom)
        ):
            _problem(result, "底牌必须是三张有效牌")
        if result.initial_hand is not None and any(c not in RANKS for c in result.initial_hand):
            _problem(result, "初始手牌含无效牌点")
        if result.seat_order and sorted(result.seat_order) != sorted(SEATS):
            _problem(result, "座位轮转顺序无效")
        if any(
            not isinstance(v, int) or isinstance(v, bool) or not 0 <= v <= 20
            for v in result.remaining.values()
            if v is not None
        ):
            _problem(result, "剩余牌数超出范围")
        if result.pending:
            for problem in result.problems:
                _problem(previous, problem)
            return previous
        return result
    if kind == "phase":
        result.phase = data["phase"]
        return result
    if kind == "actor":
        if e.get("seat") in SEATS:
            result.current_actor = e["seat"]
        return result
    if kind not in ("play", "pass"):
        _problem(result, f"未知事件类型：{kind}")
        return result
    seat, cards = e.get("seat"), e.get("cards", [])
    if seat not in SEATS or any(c not in RANKS for c in cards):
        _problem(result, "玩家或牌点无效")
        return result
    if result.current_actor is not None and result.current_actor != seat:
        _problem(result, "行动玩家与确认轮次不一致")
        return result
    if kind == "pass":
        if cards or not result.last_play:
            _problem(result, "领出轮或没有上家有效出牌时不能确认不出")
            return result
        result.passes += 1
        result.current_actor = _next(result, seat)
        if result.passes == 2:
            result.current_actor = result.last_player
            result.last_play = []
            result.passes = 0
        result.turn += 1
        return result
    if not cards or result.remaining[seat] is None or len(cards) > result.remaining[seat]:
        _problem(result, "出牌张数与剩余牌数矛盾或缺少计数依据")
        return result
    for rank, count in Counter(cards).items():
        if count > (1 if rank in ("SJ", "BJ") else 4):
            _problem(result, "同一牌点超出一副牌的数量")
            return result
    if seat == "self":
        if result.hand is None or Counter(cards) - Counter(result.hand):
            _problem(result, "自己的出牌不包含在已确认手牌中")
            return result
    visible = Counter(result.initial_hand or [])
    for opponent in ("left", "right"):
        visible.update(result.played[opponent])
        if seat == opponent:
            visible.update(cards)
    if any(count > (1 if rank in ("SJ", "BJ") else 4) for rank, count in visible.items()):
        _problem(result, "已知手牌与对手历史出牌超过一副牌数量")
        return result
    if seat == "self":
        result.hand = canonical(list((Counter(result.hand) - Counter(cards)).elements()))
    result.remaining[seat] -= len(cards)
    result.played[seat].extend(cards)
    result.last_play = canonical(cards)
    result.last_player = seat
    result.passes = 0
    result.turn += 1
    result.current_actor = _next(result, seat)
    return result


@dataclass
class TrackerUpdate:
    """单帧更新。pending_candidates 是尚在等待证据的预览，不能当作已确认事件。"""

    state: GameState
    events: list[GameEvent] = field(default_factory=list)
    issues: list[dict] = field(default_factory=list)
    pending_candidates: list[dict] = field(default_factory=list)


class GameTracker:
    """在最多三秒的证据窗口内确认动作；状态只消费已经通过规则校验的事件。

    原始观察、候选和可信状态分别保存。短暂动画只延迟确认；窗口超时和真实缺口才冻结
    当前可信前缀。所有判断使用媒体时间，离线速度、GUI 定时器和电脑时钟均不影响结果。
    """

    def __init__(self, seat_order: list[str] | None = None, *, config: dict | None = None):
        self.config = TrackerSettings(**(config or {}))
        self.state = GameState(seat_order=list(seat_order or []))
        self.order = list(seat_order or [])
        self._stable = {}
        self._last_pts = -1
        self._last_frame = ""
        self._last_event_pts = -1
        self._game_number = self._event_number = 0
        self._phase = "unknown"
        self._actor = None
        self._counts = dict.fromkeys(SEATS)
        self._hand = None
        self._transitions = Counter()
        self._issues = set()
        self._candidates = []
        self._presence = {}
        self._role_initialized = False
        self._active_started = False
        self._deal_hand = None
        self._redeal_probe = None
        self._action_deadline_us = None

    def _value(self, key, value, pts):
        """普通牌面/计数要求连续一致；未知立即打断，不与不出文字的容错混用。"""
        if value is None:
            self._stable.pop(key, None)
            return None
        signature = repr(value)
        previous = self._stable.get(key)
        if previous is None or previous[0] != signature:
            self._stable[key] = (signature, pts, 1, value, pts, [self._last_frame])
            return None
        frames = (previous[5] + [self._last_frame])[-2:]
        self._stable[key] = (signature, previous[1], previous[2] + 1, value, pts, frames)
        return value if previous[2] >= 1 and pts - previous[1] >= self.config.stable_us else None

    def _first(self, key, default):
        return self._stable.get(key, (None, default))[1]

    def _opening_evidence(self, obs, hand, counts, cards) -> dict | None:
        """以完整开局后的首轮稳定证据确认出牌开始，不依赖本人按钮是否已经出现。

        仅使用领域数据和纯牌型规则：地主必须领出，牌面与剩牌差分均已通过至少
        100ms的双帧稳定，自己领出还须核对手牌差分。返回源证据，绝不修改原观察、
        凭空补牌或解除历史缺口；无法确认时返回None继续等待现有阶段证据。
        """
        if (
            self._active_started
            or not self._role_initialized
            or self.state.pending
            or self.state.capture_incomplete
            or self.state.ended
            or not self.state.initial_hand
        ):
            return None
        role = self.state.landlord
        if role not in SEATS or len(self.order) != 3:
            return None
        if len(self.state.initial_hand) != (20 if role == "self" else 17):
            return None
        before, after, value = self._counts[role], counts[role], cards[role]
        if before is None or after is None or not value or before - after != len(value):
            return None
        if role == "self" and (
            hand is None
            or self._hand is None
            or Counter(self._hand) - Counter(hand) != Counter(value)
            or Counter(hand) - Counter(self._hand)
        ):
            return None
        # 固定的第三方纯牌型算法不加载模型、不产生设备输入；2与大小王使用非连续编码，
        # 防止把包含2或王的序列当成顺子。普通出牌仍沿用现有守恒和轮转校验。
        from doudizhu_lab._vendor.douzero.env.move_generator import MovesGener

        numbers = {rank: index + 3 for index, rank in enumerate(RANKS[:12])} | {"2": 17, "SJ": 20, "BJ": 30}
        encoded = sorted(numbers[c] for c in value)
        if not any(sorted(move) == encoded for move in MovesGener(encoded).gen_moves()):
            return None
        probe = apply_event(
            self.state, {"game_id": self.state.game_id, "kind": "setup", "data": {"current_actor": role}}
        )
        probe = apply_event(
            probe, {"game_id": self.state.game_id, "kind": "play", "seat": role, "cards": value}
        )
        if probe.pending:
            return None
        return {
            "seat": role,
            "cards": list(value),
            "before_count": before,
            "after_count": after,
            "first_evidence_us": self._first("count_" + role, obs.pts_us),
            "evidence_frames": list(
                dict.fromkeys(self._stable["play_" + role][5] + self._stable["count_" + role][5])
            ),
        }

    def _event(
        self,
        update,
        obs,
        kind,
        *,
        seat=None,
        cards=None,
        data=None,
        confirmed=True,
        first_us=None,
        frame_id=None,
    ):
        """记录语义时间及确认时间；保证追加顺序与确定性重放使用的时间顺序一致。"""
        self._event_number += 1
        first_us = obs.pts_us if first_us is None else first_us
        timestamp = max(first_us, self._last_event_pts)
        payload = dict(data or {})
        payload.setdefault("first_evidence_us", first_us)
        payload.setdefault("confirmed_at_us", obs.pts_us if confirmed else None)
        payload.setdefault("evidence_frames", [frame_id or obs.frame_id])
        event = GameEvent(
            f"{self.state.game_id}-e{self._event_number:05d}",
            self.state.game_id,
            timestamp,
            kind,
            frame_id or obs.frame_id,
            seat,
            cards or [],
            payload,
            confirmed,
        )
        before = self.state
        self.state = apply_event(before, event)
        if kind in ("play", "pass") and not before.pending and self.state.pending:
            event.confirmed = False
            event.data["confirmed_at_us"] = None
            event.data["rejection"] = self.state.problems[-1]
        self._last_event_pts = timestamp
        update.events.append(event)
        return event

    def _issue(self, update, obs, code, message):
        identity = f"{self.state.game_id}:{code}:{message}"
        if identity in self._issues:
            return
        self._issues.add(identity)
        update.issues.append(
            {
                "issue_id": sha256(identity.encode()).hexdigest()[:20],
                "game_id": self.state.game_id,
                "pts_us": obs.pts_us,
                "frame_id": obs.frame_id,
                "code": code,
                "message": message,
            }
        )
        self._event(update, obs, "gap", data={"message": message})

    def _redeal_boundary(self, obs, phase, hand):
        """用重新增长的发牌序列确认流局重发，不靠空区域或阶段文字闪烁推断。

        叫抢期间背面底牌会持续显示，阶段可能从 bidding 回落到 dealing；因此不能
        只比较阶段。必须已记住旧的完整 17 张牌、尚未确定地主，再看到至少三次
        逐步增加且包含前次牌点的发牌画面，最终得到另一副稳定的 17 张手牌。
        返回第一张发牌证据的时间及帧号；未证实时不改原牌局边界。
        """
        if (
            self.state.ended
            or self._active_started
            or self.state.landlord
            or self.state.bottom
            or not self._deal_hand
            or len(self._deal_hand) != 17
            or phase not in ("dealing", "bidding")
        ):
            self._redeal_probe = None
            return None
        probe = self._redeal_probe
        if probe and obs.pts_us - probe["first_us"] > self.config.candidate_wait_us:
            self._redeal_probe = probe = None
        raw = canonical(obs.hand)
        if not raw:
            return None
        if len(raw) == 17:
            if (
                probe
                and probe["growth"] >= 3
                and hand is not None
                and Counter(hand) != Counter(self._deal_hand)
                and not (Counter(probe["hand"]) - Counter(hand))
            ):
                self._redeal_probe = None
                return probe["first_us"], probe["frame_id"]
            return None
        if obs.phase != "dealing" or not 0 < len(raw) < 17:
            return None
        if probe and not (Counter(probe["hand"]) - Counter(raw)):
            if len(raw) > len(probe["hand"]):
                probe["growth"] += 1
                probe["hand"] = raw
        else:
            self._redeal_probe = {"first_us": obs.pts_us, "frame_id": obs.frame_id, "hand": raw, "growth": 1}
        return None

    def _start(self, update, obs, phase, hand, bottom, landlord, counts, actor, *, redeal=None):
        if self.state.game_id and not self.state.ended:
            self._drain(update, obs, flush=True)
            self._event(
                update,
                obs,
                "game_end",
                first_us=redeal[0] if redeal else None,
                data={
                    "reason": "流局重发" if redeal else "重新发牌或未观察到结算",
                    "abandoned_deal": bool(redeal),
                },
            )
        self._game_number += 1
        self.state = GameState(
            game_id=f"g{self._game_number:03d}-{sha256(obs.frame_id.encode()).hexdigest()[:8]}"
        )
        self._counts, self._hand = dict(counts), hand
        self._actor = None
        self._candidates.clear()
        self._presence.clear()
        self._role_initialized = self._active_started = False
        self._redeal_probe = None
        self._deal_hand = hand if phase in ("dealing", "bidding") and hand and len(hand) == 17 else None
        self._event(
            update,
            obs,
            "game_start",
            first_us=redeal[0] if redeal else self._first("phase", obs.pts_us),
            frame_id=redeal[1] if redeal else None,
            data={
                "phase": phase,
                "initial_hand": hand,
                "bottom": bottom,
                "landlord": landlord,
                "remaining": counts,
                "seat_order": self.order,
                "current_actor": actor if phase == "playing" else None,
                "capture_incomplete": phase not in ("dealing", "bidding"),
            },
        )
        self._phase = phase

    def _setup(self, update, obs, phase, hand, bottom, landlord, counts):
        """将叫抢、收底牌与第一轮动作分开；初始化数据永远不覆盖已经发生的出牌。"""
        setup = {}
        role = landlord or self.state.landlord
        if landlord and self.state.landlord and landlord != self.state.landlord:
            self._issue(update, obs, "landlord", "同一牌局的地主标记发生矛盾")
        if landlord and not self.state.landlord:
            setup["landlord"] = landlord
        if bottom and self.state.bottom is None:
            setup["bottom"] = bottom
        before_play = not self._active_started and not self.state.turn
        if before_play:
            if phase in ("dealing", "bidding") and hand and len(hand) == 17:
                self._deal_hand = hand
            # 只在真正的准备阶段更新手牌；进入出牌的第一帧可能已经少了一张。
            if phase in ("dealing", "bidding", "doubling") and hand is not None:
                if self.state.initial_hand is None or len(hand) >= len(self.state.initial_hand):
                    setup["initial_hand"] = hand
                    self._hand = hand
                    if not self._role_initialized:
                        setup["remaining"] = {s: n for s, n in counts.items() if n is not None}
                        self._counts.update(setup["remaining"])
            known_bottom = bottom or self.state.bottom
            if role and not self._role_initialized and not self.state.capture_incomplete:
                # 现有标准54张牌规则给出开局张数，绝不推测对手牌点。
                initial = {s: 20 if s == role else 17 for s in SEATS}
                setup["remaining"] = initial
                self._counts = dict(initial)
                if role == "self" and self._deal_hand is not None and known_bottom:
                    complete_hand = canonical(self._deal_hand + known_bottom)
                    setup["initial_hand"] = complete_hand
                    self._hand = complete_hand
                    self._role_initialized = True
                elif role != "self":
                    self._role_initialized = True
                elif hand is not None and len(hand) == 20:
                    setup["initial_hand"] = hand
                    self._hand = hand
                    self._role_initialized = True
            if phase == "playing":
                self._active_started = True
                if not self.state.capture_incomplete and role:
                    setup["current_actor"] = role
                elif self.state.current_actor is None:
                    setup["current_actor"] = self._actor or obs.actor or role
        elif not self.state.turn and not self._role_initialized and not self.state.capture_incomplete:
            # 准备阶段已看到17张牌，但身份/底牌的稳定结果可能晚到出牌阶段。
            # 只有当前仍完整显示开局张数、未确认过任何行动且预发手牌与当前手牌守恒，
            # 才允许补全；不能把已经打出的第一手覆盖为另一副“初始手牌”。
            known_bottom = bottom or self.state.bottom
            if role and self._deal_hand is not None and known_bottom:
                expected = canonical(self._deal_hand + known_bottom) if role == "self" else self._deal_hand
                initial = {s: 20 if s == role else 17 for s in SEATS}
                if (
                    hand is not None
                    and Counter(hand) == Counter(expected)
                    and all(count is None or count == initial[seat] for seat, count in counts.items())
                ):
                    setup.update(initial_hand=expected, remaining=initial, current_actor=role)
                    self._hand, self._counts, self._role_initialized = expected, dict(initial), True
        if setup and any(
            any(self.state.remaining.get(s) != n for s, n in value.items())
            if key == "remaining"
            else getattr(self.state, key, None) != value
            for key, value in setup.items()
        ):
            self._event(update, obs, "setup", data=setup)

    def _candidate(self, obs, kind, seat, **values):
        item = {
            "kind": kind,
            "seat": seat,
            "first_us": obs.pts_us,
            "frame_id": obs.frame_id,
            "evidence_frames": [obs.frame_id],
            "cards": [],
            "ready": False,
            "reason": "等待一致证据",
        }
        item.update(values)
        self._candidates.append(item)
        return item

    def _passes(self, obs):
        """仅明确出牌阶段文字可产生候选；短暂漏检最多桥接 300ms，不跨新动作。"""
        for seat in SEATS:
            presence = self._presence.get(seat)
            label = obs.pass_labels.get(seat, "pass") if seat in obs.passes else None
            if not label:
                # 明确轮到该座位的新一轮，且另两家已完成行动，足以结束旧提示的
                # 300ms 漏检桥接。否则快速再次不出会被旧 consumed 标记吞掉。
                new_turn = bool(
                    presence
                    and presence["consumed"]
                    and obs.actor == self._actor == seat
                    and self.state.turn - presence.get("consumed_turn", self.state.turn) >= 2
                )
                if presence and (new_turn or obs.pts_us - presence["last_us"] > self.config.pass_gap_us):
                    presence["absent"] = True
                continue
            if label not in ("pass", "过", "不要", "不出", "要不起"):
                continue
            if presence and presence["consumed"]:
                # 已记载的文字短暂闪烁不会变成第二次动作。下一轮必须经过明确消失，
                # 且另外两个座位已行动；新轮的提示再重新收集两次稳定证据。
                if (
                    presence.get("absent")
                    and self.state.turn - presence.get("consumed_turn", self.state.turn) >= 2
                ):
                    presence = None
                else:
                    presence["last_us"] = obs.pts_us
                    continue
            if presence and obs.pts_us - presence["last_us"] > self.config.pass_gap_us:
                # 同一尚未确认的动作可重新开始稳定检查，但不留下重复候选或续长三秒上限。
                presence["first_us"], presence["seen"] = obs.pts_us, 0
            if presence is None:
                presence = {
                    "first_us": obs.pts_us,
                    "last_us": obs.pts_us,
                    "seen": 0,
                    "consumed": False,
                    "candidate": None,
                    "label": label,
                }
                self._presence[seat] = presence
            presence["last_us"] = obs.pts_us
            presence["seen"] += 1
            if presence["consumed"]:
                continue
            # 同一持续字样变化仍属于当前出现区间；绝不因文字变形重复记不出。
            candidate = presence["candidate"]
            if candidate is None:
                candidate = self._candidate(obs, "pass", seat, label=label)
                presence["candidate"] = candidate
            if obs.frame_id not in candidate["evidence_frames"]:
                candidate["evidence_frames"] = (candidate["evidence_frames"] + [obs.frame_id])[-8:]
            candidate["ready"] = (
                presence["seen"] >= 2 and obs.pts_us - presence["first_us"] >= self.config.stable_us
            )
            candidate["reason"] = "等待行动轮次" if candidate["ready"] else "等待第二次一致的不出提示"

    def _plays(self, update, obs, hand, counts, cards):
        # 遮挡可让完整手牌暂时只剩少数可见字形。三秒窗口内若完整原手牌重新稳定
        # 出现，而候选从未同时得到桌面牌组支持，则撤回这条弱候选。原观察仍永久保存；
        # 已确认动作、对手牌数矛盾或不同手牌不能走此恢复路径。
        for item in list(self._candidates):
            if (
                item["kind"] == "play"
                and item["seat"] == "self"
                and not item["ready"]
                and hand is not None
                and item.get("before_hand") is not None
                and canonical(hand) == item["before_hand"]
                and counts["self"] == item["before_count"]
                and obs.pts_us - item["first_us"] < self.config.candidate_wait_us
            ):
                self._candidates.remove(item)
                self._counts["self"], self._hand = counts["self"], hand
                self._stable.pop("increase_self", None)
                # 暂时少牌时曾压住旧“不出”提示；撤回遮挡候选时也须恢复它的
                # 轮次基准，否则会把下一次真实不出误当作本轮已消费过的提示。
                restored = item.get("presence_before")
                if restored is None:
                    self._presence.pop("self", None)
                else:
                    restored = deepcopy(restored)
                    old_candidate = restored.get("candidate")
                    if old_candidate:
                        active = next(
                            (
                                c
                                for c in self._candidates
                                if c["kind"] == "pass"
                                and c["seat"] == "self"
                                and c["first_us"] == old_candidate["first_us"]
                            ),
                            None,
                        )
                        if active is not None:
                            active.update(old_candidate)
                            restored["candidate"] = active
                    self._presence["self"] = restored
                update.issues.append(
                    {
                        "game_id": self.state.game_id,
                        "pts_us": obs.pts_us,
                        "frame_id": obs.frame_id,
                        "code": "occlusion_recovered",
                        "message": "完整原手牌重新出现，撤回未得到出牌牌面支持的遮挡候选",
                    }
                )
        for seat in SEATS:
            before, after = self._counts[seat], counts[seat]
            if after is not None and before is not None and after < before:
                removed = (
                    canonical(list((Counter(self._hand or []) - Counter(hand or [])).elements()))
                    if seat == "self" and hand is not None and self._hand is not None
                    else None
                )
                self._candidate(
                    obs,
                    "play",
                    seat,
                    before_count=before,
                    after_count=after,
                    removed=removed,
                    before_hand=canonical(self._hand) if seat == "self" else None,
                    presence_before=deepcopy(self._presence.get(seat)),
                    first_us=self._first("count_" + seat, obs.pts_us),
                )
                self._counts[seat] = after
                if seat == "self":
                    self._hand = hand
                # 明确的牌数减少与旧不出标记冲突；后者不能延用到本次出牌。
                old = self._presence.get(seat)
                if old and old.get("candidate") in self._candidates and not old["consumed"]:
                    old["candidate"]["reason"] = "不出提示与该玩家出牌证据冲突"
                    old["candidate"]["ready"] = False
                if old:
                    old.update(consumed=True, consumed_turn=self.state.turn, absent=False)
            elif before is None and after is not None:
                self._counts[seat] = after
            elif after is not None and before is not None and after > before:
                key = "increase_" + seat
                if (
                    self._value(key, after, obs.pts_us) is not None
                    and obs.pts_us - self._first(key, obs.pts_us) >= self.config.candidate_wait_us
                ):
                    self._candidate(
                        obs,
                        "play",
                        seat,
                        before_count=before,
                        after_count=after,
                        first_us=self._first(key, obs.pts_us),
                        reason="出牌阶段剩牌数增加",
                    )
                    self._counts[seat] = after
            else:
                self._stable.pop("increase_" + seat, None)
        for item in self._candidates:
            if item["kind"] != "play":
                continue
            seat, value = item["seat"], cards[item["seat"]]
            count = counts[seat]
            # 已经出现下次计数时不拿新牌面反填上次动作；早期证据由候选自己保留。
            if count != item["after_count"] or not value:
                continue
            delta = item["before_count"] - item["after_count"]
            if len(value) != delta or (seat == "self" and item["removed"] != canonical(value)):
                item["reason"] = "牌面、剩牌差分或自己的手牌差分暂不一致"
                continue
            item["cards"] = value
            item["ready"] = True
            item["reason"] = "等待行动轮次"
            if obs.frame_id not in item["evidence_frames"]:
                item["evidence_frames"] = (item["evidence_frames"] + [obs.frame_id])[-8:]

    def _publish(self, update, obs, item, confirmed):
        data = {key: item[key] for key in ("before_count", "after_count", "label") if key in item}
        data["evidence_frames"] = item["evidence_frames"]
        if not confirmed:
            data["rejection"] = item["reason"]
        self._event(
            update,
            obs,
            item["kind"],
            seat=item["seat"],
            cards=item["cards"],
            data=data,
            confirmed=confirmed,
            first_us=item["first_us"],
            frame_id=item["frame_id"],
        )
        self._candidates.remove(item)
        presence = self._presence.get(item["seat"])
        if presence and presence.get("candidate") is item:
            presence["consumed"] = True
            presence["consumed_turn"] = self.state.turn

    def _drain(self, update, obs, *, flush=False):
        """轮次前面的证据优先；就绪但越轮的动作先等候，超时才冻结并保留候选。"""
        while self._candidates:
            ready = [c for c in self._candidates if c["ready"]]
            if self.state.pending:
                for item in sorted(ready, key=lambda c: c["first_us"]):
                    self._publish(update, obs, item, False)
            else:
                seat = self.state.current_actor
                choices = [c for c in ready if seat is None or c["seat"] == seat]
                if choices:
                    item = min(choices, key=lambda c: c["first_us"])
                    probe = {
                        "game_id": self.state.game_id,
                        "kind": item["kind"],
                        "seat": item["seat"],
                        "cards": item["cards"],
                    }
                    checked = apply_event(self.state, probe)
                    if not checked.pending:
                        self._publish(update, obs, item, True)
                        continue
                    item["reason"] = checked.problems[-1]
            expired = [
                c
                for c in self._candidates
                if flush
                or obs.pts_us
                >= max(
                    c["first_us"] + self.config.candidate_wait_us,
                    min(c["first_us"] + 8_000_000, self._action_deadline_us or 0),
                )
            ]
            if not expired:
                break
            first = min(expired, key=lambda c: c["first_us"])
            if not self.state.pending:
                message = f"{SEAT_NAMES[first['seat']]}动作证据在窗口内无法确认：{first['reason']}"
                self._issue(
                    update, obs, "missing_play" if first["kind"] == "play" else "missing_pass", message
                )
            self._publish(update, obs, first, False)

    def observe(self, obs: TableObservation, *, action_deadline_us: int | None = None) -> TrackerUpdate:
        """消费一帧并返回已确认事件、疑点和当前候选；重复或倒序源帧不会推进状态。

        action_deadline_us仅供运行观察器在真实单次提交后延长只读取证期限，使用源媒体
        微秒时间。普通候选沿用冻结配置的三秒窗口，延长也不能超过候选首证据后八秒；
        不刷新原始时间、不证明已出牌，不会恢复已确认的缺口或推迟新局边界的清算。
        """
        update = TrackerUpdate(self.state)
        if obs.frame_id == self._last_frame or obs.pts_us <= self._last_pts:
            return update
        self._action_deadline_us = action_deadline_us
        self._last_frame, self._last_pts = obs.frame_id, obs.pts_us
        phase = self._value("phase", obs.phase if obs.phase != "unknown" else None, obs.pts_us)
        hand = self._value("hand", canonical(obs.hand), obs.pts_us)
        bottom = self._value("bottom", canonical(obs.bottom), obs.pts_us)
        landlord = self._value("landlord", obs.landlord, obs.pts_us)
        actor = self._value("actor", obs.actor, obs.pts_us)
        counts = {s: self._value("count_" + s, obs.remaining.get(s), obs.pts_us) for s in SEATS}
        cards = {s: self._value("play_" + s, canonical(obs.plays.get(s)), obs.pts_us) for s in SEATS}
        # 已进入出牌后，残留的准备动画不能重开同一副手牌。新17张手牌仍可明确
        # 标识漏观察结算后的下一局；正常结算后的新局沿用既有边界规则。
        if self._active_started and not self.state.ended and phase in ("dealing", "bidding", "doubling"):
            if phase == "doubling" or hand is None or len(hand) != 17 or hand == self._hand:
                phase = None
        preparation = phase in ("dealing", "bidding")
        redeal = self._redeal_boundary(obs, phase or self._phase, hand)
        new_game = bool(
            phase in ("dealing", "bidding", "doubling", "playing")
            and (
                (not self.state.game_id and (preparation or bool(hand)))
                or self.state.ended
                and (
                    preparation
                    or bool(hand and len(hand) in (17, 20) and bottom and bottom != self.state.bottom)
                )
                or preparation
                and (self._active_started or self._phase == "playing")
                or redeal is not None
            )
        )
        if new_game:
            self._start(update, obs, phase, hand, bottom, landlord, counts, actor, redeal=redeal)
        if not self.state.game_id or self.state.ended:
            update.state = deepcopy(self.state)
            return update
        if phase == "settlement":
            self._drain(update, obs, flush=True)
            self._event(update, obs, "game_end", first_us=self._first("phase", obs.pts_us))
            self._phase = phase
            update.state = deepcopy(self.state)
            return update
        if actor and actor != self._actor:
            if self._actor and self._phase == "playing":
                self._transitions[(self._actor, actor)] += 1
            self._actor = actor
            if not self.order:
                for second, third in (("left", "right"), ("right", "left")):
                    chain = [("self", second), (second, third), (third, "self")]
                    if all(self._transitions[p] for p in chain) and all(
                        not self._transitions[(b, a)] for a, b in chain
                    ):
                        self.order = ["self", second, third]
                        self._event(update, obs, "setup", data={"seat_order": self.order})
        self._setup(update, obs, phase, hand, bottom, landlord, counts)
        opening = (
            self._opening_evidence(obs, hand, counts, cards)
            if phase in (None, "dealing", "bidding", "doubling")
            else None
        )
        if opening is not None:
            phase = "playing"
            self._active_started = True
            self._event(
                update,
                obs,
                "setup",
                data={"current_actor": opening["seat"]},
                first_us=opening["first_evidence_us"],
            )
        if phase and phase != self._phase and not (self._active_started and phase == "doubling"):
            self._phase = phase
            self._event(
                update,
                obs,
                "phase",
                data={"phase": phase} | ({"basis": "stable_opening_play", **opening} if opening else {}),
                first_us=opening["first_evidence_us"] if opening else self._first("phase", obs.pts_us),
            )
        if self._phase == "playing":
            if self.state.current_actor is None and not self.state.turn and actor:
                self._event(update, obs, "actor", seat=actor)
            self._plays(update, obs, hand, counts, cards)
            self._passes(obs)
            self._drain(update, obs)
            # 同帧刚确认上一家的出牌后，轮次才前进到这里。再次登记已稳定的
            # 新行动者和旧提示消失，避免下一帧的快速不出仍继承上一轮 consumed。
            for seat, presence in self._presence.items():
                if (
                    presence["consumed"]
                    and seat not in obs.passes
                    and obs.actor == self._actor == seat
                    and self.state.turn - presence.get("consumed_turn", self.state.turn) >= 2
                ):
                    presence["absent"] = True
            # 结算镜头中的“最后出牌”是明确结束证据；必须先确认末手牌组和零剩牌，
            # 不能因为没有手牌或某个空区域就结束。随后残留的出牌画面不再重开同局。
            winners = [
                s
                for s in SEATS
                if self.state.remaining[s] == 0 and obs.marker_evidence.get("winner_" + s, {}).get("accepted")
            ]
            if len(winners) == 1 and not self.state.pending and not self._candidates:
                self._event(
                    update,
                    obs,
                    "game_end",
                    first_us=self._first("count_" + winners[0], obs.pts_us),
                    data={"reason": "明确末手标记与已确认零剩牌"},
                )
                self._phase = "settlement"
        for warning in obs.warnings:
            self._issue(update, obs, "observation", warning)
        update.state = deepcopy(self.state)
        update.pending_candidates = deepcopy(self._candidates)
        return update

    def mark_gap(self, obs: TableObservation, message: str, *, capture_lost: bool = False) -> TrackerUpdate:
        """显式采集缺口立即冻结可信前缀；不得利用中断前后两帧拼成稳定观察。"""
        update = TrackerUpdate(self.state)
        if self.state.game_id and not self.state.ended:
            self._event(update, obs, "gap", data={"message": message, "capture_lost": capture_lost})
            self._drain(update, obs, flush=True)
        self._stable.clear()
        self._presence.clear()
        self._redeal_probe = None
        update.state = deepcopy(self.state)
        return update


def replay_events(events: list[dict]) -> dict:
    """重放导出或修订后的事件，返回每局状态；同一输入始终得到相同结果。"""
    states = {}
    for event in sorted(events, key=lambda e: e["pts_us"]):
        game_id = event.get("game_id", "")
        state = states.get(game_id, GameState())
        try:
            states[game_id] = apply_event(state, event)
        except (KeyError, ValueError, TypeError) as exc:
            state = deepcopy(state)
            _problem(state, f"事件格式无效：{exc}")
            states[game_id] = state
    return {key: asdict(value) for key, value in states.items()}
