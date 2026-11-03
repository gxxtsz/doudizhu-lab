"""公开信息的确定性推导和证明复核；不读取模型建议、输入意图或对手私有手牌。"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, is_dataclass
from hashlib import sha256
from typing import TYPE_CHECKING

from doudizhu_lab.domain.models import RANKS, Frame
from doudizhu_lab.domain.table import SEATS

if TYPE_CHECKING:
    from doudizhu_lab.domain.tracker import GameState

EVIDENCE_POLICY_VERSION = 2
EVIDENCE_WAIT_SECONDS = 8.0


def state_hash(state: GameState | dict) -> str:
    """对推导前的完整可信状态取稳定指纹；复核方必须自行重放得到同一状态。"""
    value = asdict(state) if is_dataclass(state) else state
    return sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def frame_context(frame: Frame) -> dict:
    """绑定原始观察与设备、视频编码会话、尺寸；不刷新源时间或授予输入权限。"""
    return {
        "serial": frame.serial,
        "stream_id": frame.stream_id,
        "width": frame.width,
        "height": frame.height,
    }


def _pair(values):
    if len(values) != 2:
        return False
    a, b = values
    context = a.get("frame_context")
    return bool(
        a["frame_id"] != b["frame_id"]
        and 100_000 <= b["pts_us"] - a["pts_us"] <= 8_000_000
        and context
        and context == b.get("frame_context")
        and all(key in context for key in ("serial", "stream_id", "width", "height"))
        and context["serial"]
        and context["stream_id"]
        and context["width"] > 0
        and context["height"] > 0
    )


def _cards(value):
    return sorted(value, key=RANKS.index)


def _legal(state, seat, cards):
    from doudizhu_lab.agents.douzero import TO_NUMBER, legal_actions

    numbers = sorted(TO_NUMBER[c] for c in cards)
    target = [TO_NUMBER[c] for c in state.last_play]
    hand = [TO_NUMBER[c] for c in (state.hand if seat == "self" else cards)]
    return bool(cards) and numbers in legal_actions(hand, target)


def _unique_opponent(state, count):
    """在两家合并公开剩余牌集合中枚举单个动作；多解或超过4096个候选立即拒绝。

    合并集合只是可用牌张上界，不构造任何一家的隐藏手牌。即使真实动作存在，
    搜索预算耗尽也返回未知，绝不把已扫描到的唯一候选冒充全局唯一。
    """
    from doudizhu_lab.agents.douzero import DECK, TO_RANK

    available = Counter({TO_RANK[n]: count for n, count in DECK.items()})
    used = Counter(state.hand or [])
    for cards in state.played.values():
        used.update(cards)
    if used - available:
        return None
    available.subtract(used)
    if sum(available.values()) != sum(state.remaining[s] for s in SEATS if s != "self"):
        return None
    ranks = [r for r in RANKS if available[r]]
    found, scanned = [], 0

    def generate(index, left, cards):
        if not left:
            yield cards
        elif index < len(ranks):
            rank = ranks[index]
            for n in range(min(left, available[rank]) + 1):
                yield from generate(index + 1, left - n, cards + [rank] * n)

    for candidate in generate(0, count, []):
        scanned += 1
        if scanned > 4096:
            return None
        if _legal(state, state.current_actor, candidate):
            found.append(candidate)
            if len(found) > 1:
                return None
    return found[0] if len(found) == 1 else None


def _continuation(state, cards, values):
    """验证漏识别动作后至当前行动者的唯一公开链，最多跨过另外两家。

    后续有牌数减少时必须直接看清该家的完整双帧牌组；只有首个漏识别动作
    可以由其他约束求解。这样快速被大王压过的小王也能确认，不猜两次未知出牌。
    """
    from doudizhu_lab.domain.tracker import apply_event

    a, b = values
    seat = state.current_actor
    probe = apply_event(state, {"game_id": state.game_id, "kind": "play", "seat": seat, "cards": cards})
    if probe.pending:
        return None
    passes = []
    if probe.remaining[seat] == 0:
        return passes if all(probe.remaining[s] == b["remaining"][s] for s in SEATS) else None
    actor = b.get("actor")
    for _ in range(2):
        if probe.current_actor == actor:
            break
        following = probe.current_actor
        drop = probe.remaining[following] - b["remaining"][following]
        if drop > 0:
            visible = [o.get("plays", {}).get(following) for o in values]
            if not all(v and len(v) == drop for v in visible) or Counter(visible[0]) != Counter(visible[1]):
                return None
            move = visible[0]
            if not _legal(probe, following, move):
                return None
            kind = "play"
        elif drop == 0:
            move, kind = [], "pass"
            passes.append({"kind": "pass", "seat": following, "cards": []})
        else:
            return None
        probe = apply_event(probe, {"game_id": state.game_id, "kind": kind, "seat": following, "cards": move})
        if probe.pending:
            return None
    if probe.current_actor != actor or any(probe.remaining[s] != b["remaining"][s] for s in SEATS):
        return None
    return passes


def deductions(state: GameState, values: Sequence[dict], *, deal_values: Sequence[dict] = ()) -> list[dict]:
    """仅由可信前缀和两个原始锚点求出可唯一确定的事实，返回独立的结论列表。

    返回项使用事件的kind/seat/cards/data形状。本人出牌还附after_hand，便于在
    后续动画遮住牌面时确认已得到证据的反馈。所有结论均需再次经过事件合法性校验。
    """
    if state.pending or state.capture_incomplete or state.ended or not state.game_id or not _pair(values):
        return []
    a, b = values
    if any(o.get("landlord") not in (None, state.landlord) for o in values) and state.landlord:
        return []
    if a.get("hand") != b.get("hand") or a.get("remaining") != b.get("remaining"):
        return []
    hand, counts = b.get("hand"), b.get("remaining", {})
    claims = []
    if not state.turn:
        setup = {}
        owners = [s for s in SEATS if counts.get(s) == 20]
        role = state.landlord
        if not role and len(owners) == 1 and all(counts.get(s) == 17 for s in SEATS if s != owners[0]):
            owner = owners[0]
            if (
                hand is not None
                and len(hand) == (20 if owner == "self" else 17)
                and all(o.get("landlord") in (None, owner) for o in values)
            ):
                setup["landlord"] = role = owner
        original = (
            state.initial_hand if state.initial_hand is not None and len(state.initial_hand) == 17 else None
        )
        if original is None and _pair(deal_values):
            x, y = deal_values
            if (
                y["pts_us"] < a["pts_us"]
                and x.get("frame_context") == a.get("frame_context")
                and x.get("hand") == y.get("hand")
                and len(y.get("hand") or []) == 17
                and all(o.get("phase") in ("dealing", "bidding") for o in deal_values)
            ):
                original = y["hand"]
        if (
            role == "self"
            and not state.bottom
            and original is not None
            and hand is not None
            and len(hand) == 20
            and not (Counter(original) - Counter(hand))
            and counts.get("self") == 20
            and all(counts.get(s) == 17 for s in ("left", "right"))
            and all(not o.get("plays", {}).get(s) for o in values for s in SEATS)
        ):
            bottom = _cards(list((Counter(hand) - Counter(original)).elements()))
            if len(bottom) == 3 and all(
                not o.get("bottom") or len(o["bottom"]) != 3 or Counter(o["bottom"]) == Counter(bottom)
                for o in values
            ):
                setup["bottom"] = bottom
                if _cards(state.initial_hand or []) != _cards(hand):
                    setup["initial_hand"] = _cards(hand)
        if setup:
            claims.append({"kind": "setup", "data": setup})
    if state.phase != "playing" or state.current_actor not in state.seat_order or state.hand is None:
        return claims
    if any(o.get("phase") not in ("playing", "unknown", "settlement") for o in values):
        return claims
    if any(counts.get(s) is None or state.remaining[s] is None for s in SEATS):
        return claims
    actor = b.get("actor")
    if actor != a.get("actor"):
        return claims
    delta = {s: state.remaining[s] - counts[s] for s in SEATS}
    if any(v < 0 for v in delta.values()):
        return claims
    changed = [s for s in SEATS if delta[s]]
    seat = state.current_actor
    if delta[seat] > 0:
        final = counts[seat] == 0 and all(
            o.get("marker_evidence", {}).get("winner_" + seat, {}).get("accepted") for o in values
        )
        if not final and actor not in state.seat_order:
            return claims
        if seat == "self":
            if hand is None or Counter(hand) - Counter(state.hand):
                return claims
            cards = _cards(list((Counter(state.hand) - Counter(hand)).elements()))
        else:
            if hand is None or Counter(hand) != Counter(state.hand):
                return claims
            visible = [o.get("plays", {}).get(seat) for o in values]
            if all(v and len(v) == delta[seat] for v in visible) and Counter(visible[0]) == Counter(
                visible[1]
            ):
                return claims  # 已有完整双帧牌面，交给直接观察器，不启动组合搜索。
            cards = _unique_opponent(state, delta[seat])
        if cards is None or len(cards) != delta[seat] or not _legal(state, seat, cards):
            return claims
        following = _continuation(state, cards, values)
        if following is None:
            return claims
        # 有效完整牌组与守恒结论相反才算冲突；动画中的残缺牌组不是另一份完整事实。
        for o in values:
            visible = o.get("plays", {}).get(seat)
            if visible and len(visible) == len(cards) and Counter(visible) != Counter(cards):
                return claims
        play = {"kind": "play", "seat": seat, "cards": cards}
        if seat == "self":
            play["after_hand"] = _cards(hand)
        claims.append(play)
        if not final:
            claims.extend(following)
    elif (
        not changed
        and hand is not None
        and Counter(hand) == Counter(state.hand)
        and state.last_play
        and actor in state.seat_order
    ):
        distance = (state.seat_order.index(actor) - state.seat_order.index(seat)) % 3
        if not 1 <= distance <= 2 - state.passes:
            return claims
        seats = [state.seat_order[(state.seat_order.index(seat) + i) % 3] for i in range(distance)]
        if any(o.get("plays", {}).get(s) for o in values for s in seats):
            return claims
        claims.extend({"kind": "pass", "seat": s, "cards": []} for s in seats)
    return claims


def matches(event: dict, conclusion: dict) -> bool:
    """比较证明实际覆盖的事件字段；setup中同帧直接识别的其他字段允许同时存在。"""
    if event["kind"] != conclusion["kind"]:
        return False
    if event["kind"] == "setup":
        return all(event.get("data", {}).get(k) == v for k, v in conclusion["data"].items())
    return event.get("seat") == conclusion.get("seat") and Counter(event.get("cards", [])) == Counter(
        conclusion.get("cards", [])
    )


def make_proof(
    state: GameState, conclusion: dict, values: Sequence[dict], *, deal_values: Sequence[dict] = ()
) -> dict:
    """记录原始锚点引用和前置状态指纹；此对象本身不是通过复核的凭据。"""
    return {
        "version": 2,
        "kind": conclusion["kind"],
        "before_hash": state_hash(state),
        "frames": [o["frame_id"] for o in values],
        "from_us": values[0]["pts_us"],
        "to_us": values[-1]["pts_us"],
        "baseline_frames": [o["frame_id"] for o in deal_values],
        "conclusion": conclusion,
        "basis": "公开手牌与计数守恒、合法牌型及唯一轮转；双帧独立取证",
    }


def verify_proof(
    event: dict, before: GameState, observations: dict[str, dict], *, expected_context: dict | None = None
) -> bool:
    """从原始观察重新求解并验证证明；不使用运行时标记、模型建议或发送日志作答案。

    expected_context来自本局已确认开局的原始帧；给定时还核对设备、编码流、尺寸与
    原始分局标记一致。before由已验证前缀重放产生，observations按原始frame_id索引。
    """
    proof = event.get("data", {}).get("inference", {})
    try:
        values = [observations[f] for f in proof["frames"]]
        baseline = [observations[f] for f in proof.get("baseline_frames", [])]
        if expected_context is not None and any(o.get("frame_context") != expected_context for o in values):
            return False
        if (
            proof.get("version") != 2
            or event.get("game_id") != before.game_id
            or proof["before_hash"] != state_hash(before)
            or proof["from_us"] != values[0]["pts_us"]
            or proof["to_us"] != values[-1]["pts_us"]
            or proof["kind"] != event["kind"]
            or event["pts_us"] > values[-1]["pts_us"]
            or event.get("data", {}).get("confirmed_at_us", values[-1]["pts_us"]) < values[-1]["pts_us"]
        ):
            return False
        return any(
            c == proof["conclusion"] and matches(event, c)
            for c in deductions(before, values, deal_values=baseline)
        )
    except (KeyError, ValueError, TypeError, IndexError):
        return False
