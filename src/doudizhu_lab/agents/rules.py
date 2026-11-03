"""平衡模式的局部回合与保守规则策略；不构造缺失历史，也不调用设备。"""

from collections import Counter
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
from time import perf_counter

from doudizhu_lab.agents.douzero import ROLES, TO_NUMBER, TO_RANK, legal_actions
from doudizhu_lab.agents.readiness import ObservationWait
from doudizhu_lab.domain.bot_types import Decision, encode


def role_for(state):
    """仅按已核实地主与座位顺序映射身份，不读取或分配对手隐藏手牌。"""
    order = state.get("seat_order", [])
    if sorted(order) != ["left", "right", "self"] or state.get("landlord") not in order:
        raise ValueError("规则接管仍需要明确地主及座位轮转")
    return ROLES[(order.index("self") - order.index(state["landlord"])) % 3]


def visible_target(snapshot, lead_signature=None):
    """返回有视觉依据的当前目标及座位；无法排除旧牌或遮挡时抛出等待。

    领出需要已由可信回合标定的单独出牌按钮位置；不能仅凭空牌区推断领出。
    跟牌需要不出按钮及完整的对手牌区；两家都显示牌时必须能唯一比较大小。
    """
    obs = snapshot.observation
    if obs.get("actor") != "self" or not snapshot.buttons:
        raise ObservationWait("等待明确的本人行动按钮")
    passing = [b for b in snapshot.buttons if b.kind == "pass"]
    plays = [b for b in snapshot.buttons if b.kind == "play"]
    if lead_signature and plays and not passing:
        expected = lead_signature["center"]
        for button in plays:
            center = (
                (button.hit.x + button.hit.width / 2) / snapshot.frame.width,
                (button.hit.y + button.hit.height / 2) / snapshot.frame.height,
            )
            if max(abs(a - b) for a, b in zip(center, expected)) <= 0.012:
                return [], None, "已标定的领出按钮布局"
    if not passing:
        raise ObservationWait("尚未确认领出布局或跟牌按钮")
    choices = []
    for seat in ("left", "right"):
        if seat in obs.get("passes", []):
            continue
        cards = obs.get("plays", {}).get(seat)
        if cards is None:
            raise ObservationWait("对手当前牌面被遮挡，不能确定目标")
        if cards:
            numbers = sorted(TO_NUMBER[c] for c in cards)
            if numbers not in legal_actions(numbers, []):
                raise ObservationWait("当前目标不是完整合法牌型")
            choices.append((cards, seat))
    if not choices:
        raise ObservationWait("跟牌目标没有明确显示")
    if len(choices) == 1:
        return *choices[0], "唯一可见有效出牌及跟牌按钮"
    a, b = choices
    an, bn = sorted(TO_NUMBER[c] for c in a[0]), sorted(TO_NUMBER[c] for c in b[0])
    if an in legal_actions(an, bn):
        return *a, "两家可见牌组可唯一比较大小"
    if bn in legal_actions(bn, an):
        return *b, "两家可见牌组可唯一比较大小"
    raise ObservationWait("两家显示牌组不能唯一确定当前目标")


def context(snapshot):
    """规则执行器的最小合法动作输入；明确标记历史不完整，不能交给DouZero冒充完整局面。"""
    state, obs = snapshot.state, snapshot.observation
    if state.get("strategy") != "rules" or state.get("capture_incomplete"):
        raise ValueError("当前不是允许规则接管的实时局面")
    if state.get("rule_halt_reason"):
        raise ValueError(state["rule_halt_reason"])
    if state["ended"] or state["current_actor"] != "self":
        raise ObservationWait("规则策略等待自己的明确回合")
    evidence = obs.get("marker_evidence", {}).get("rule_target", {})
    if not evidence.get("accepted") or evidence.get("frame_id") != snapshot.frame.frame_id:
        raise ObservationWait(evidence.get("reason", "等待当前回合目标补证"))
    if obs.get("hand") is None or Counter(obs["hand"]) != Counter(state["hand"]):
        raise ObservationWait("规则策略仍需要完整且一致的本人手牌")
    if obs.get("actor") != "self":
        raise ObservationWait("当前帧尚未确认本人行动")
    target = state["rule_target"]
    if Counter(evidence.get("cards", [])) != Counter(target):
        raise ObservationWait("目标牌证据已经变化")
    hand = sorted(TO_NUMBER[c] for c in state["hand"])
    previous = sorted(TO_NUMBER[c] for c in target)
    return {
        "strategy": "rules",
        "history_complete": False,
        "player_position": role_for(state),
        "player_hand_cards": hand,
        "target": previous,
        "target_owner": state.get("rule_target_owner"),
        "remaining": dict(state["remaining"]),
        "seat_order": list(state["seat_order"]),
        "legal_actions": legal_actions(hand, previous),
    }


def decide(snapshot):
    """从固定规则生成一个合法动作：能结束则结束，尽量少拆组合及炸弹，不冒充WP模型。"""
    started = perf_counter()
    data = context(snapshot)
    hand, actions = data["player_hand_cards"], data["legal_actions"]
    finishes = [a for a in actions if len(a) == len(hand)]
    if finishes:
        chosen = min(finishes)
    elif (
        data["target"]
        and data["target_owner"] not in (None, snapshot.state["landlord"])
        and data["player_position"] != "landlord"
    ):
        chosen = []
    else:
        nonpass = [a for a in actions if a]
        ordinary = [a for a in nonpass if not (a == [20, 30] or len(a) == 4 and len(set(a)) == 1)]
        if data["target"] and not ordinary and [] in actions:
            chosen = []
        else:

            def cost(action):
                remaining = Counter(hand) - Counter(action)
                broken = sum(n == 4 and 0 < remaining[c] < 4 for c, n in Counter(hand).items())
                return len(remaining) + broken * 4 - len(action) * 0.12 + sum(action) * 0.003

            chosen = min(ordinary or nonpass, key=cost)
    state = snapshot.state
    return Decision(
        tuple(TO_RANK[n] for n in chosen),
        snapshot.version,
        state["game_id"],
        state["turn"],
        data["player_position"],
        "保守规则-v1",
        (perf_counter() - started) * 1000,
        0,
        snapshot.revision,
        "rules",
    )


def feedback(snapshot, decision, before_hand, before_counts, submitted_pts, *, seat_order=()):
    """检查规则动作后的单帧反馈，返回可供双帧核对的依据；本函数不确认命令意图。

    出牌需要实际手牌差分、本人桌面牌及行动推进，末手允许明确零剩牌结束标记。
    不出使用本人完整提示；提示漏检时只允许行动者前进且三家计数和本人手牌都未变。
    """
    obs = snapshot.observation
    if snapshot.frame.media_pts_us <= submitted_pts:
        return None
    hand = Counter(obs["hand"]) if obs.get("hand") is not None else None
    actor = obs.get("actor")
    if decision.cards:
        expected = before_hand - Counter(decision.cards)
        final = (
            not expected
            and obs.get("remaining", {}).get("self") == 0
            and obs.get("marker_evidence", {}).get("winner_self", {}).get("accepted")
        )
        ordinary = (
            hand == expected
            and Counter(obs.get("plays", {}).get("self") or []) == Counter(decision.cards)
            and actor in ("left", "right")
        )
        # 局部规则仍可验证自己的精确手牌差分；需要三家计数与唯一下一行动者，
        # 不能用规则建议或按钮消失代替反馈，也不能据此宣称补齐全局历史。
        visible = obs.get("plays", {}).get("self")
        difference = bool(
            len(seat_order) == 3
            and "self" in seat_order
            and hand == expected
            and actor == seat_order[(seat_order.index("self") + 1) % 3]
            and obs.get("remaining", {}).get("self") == sum(expected.values())
            and all(obs.get("remaining", {}).get(s) == before_counts.get(s) for s in ("left", "right"))
            and not (
                visible
                and len(visible) == len(decision.cards)
                and Counter(visible) != Counter(decision.cards)
            )
        )
        if not (final or ordinary or difference):
            return None
        return {
            "basis": "explicit_final"
            if final
            else ("hand_play_and_actor" if ordinary else "hand_difference_and_unique_actor"),
            "hand": sorted(expected.elements()),
            "cards": list(decision.cards),
        }
    if hand != before_hand:
        return None
    if "self" in obs.get("passes", []):
        return {"basis": "explicit_pass", "hand": sorted(before_hand.elements()), "cards": []}
    if actor in ("left", "right") and all(
        before_counts.get(s) is not None and obs.get("remaining", {}).get(s) == before_counts[s]
        for s in ("self", "left", "right")
    ):
        return {
            "basis": "inferred_pass_actor_and_unchanged_counts",
            "hand": sorted(before_hand.elements()),
            "cards": [],
        }
    return None


class RuleRound:
    """历史冻结后的局部回合，保留原始缺口；只在明确当前目标上执行规则，不补造前史。"""

    def __init__(self, state, *, lead_signature=None):
        self.state = deepcopy(state)
        self.state.update(
            strategy="rules",
            history_incomplete=True,
            complete=False,
            pending=False,
            rule_halt_reason="",
            rule_target=[],
            rule_target_owner=None,
        )
        self.state["turn"] = 1_000_000 + state["turn"]
        self.state["current_actor"] = None
        self.lead_signature = lead_signature
        self.intent = None
        self.credited = False
        self.expected_hand = None
        self.last_id, self.stable = "", None
        self.count_probes = {}

    def submitted(self, decision, pts_us):
        """只登记在途意图，不因此减少手牌或生成已发生事件；未知反馈禁止开始下一动作。"""
        self.intent = {"decision": decision, "pts_us": pts_us, "before": Counter(self.state["hand"])}
        self.expected_hand = self.intent["before"] - Counter(decision.cards)

    def confirmed(self, decision):
        """仅在执行器取得双帧反馈后清除在途意图，不把ADB命令返回当成确认。"""
        if self.intent and self.intent["decision"].turn == decision.turn:
            self.intent = None
            self.credited = True

    def project(self, snapshot):
        """发布与原图同帧的局部状态；临时漏检只影响就绪证据，不让选牌上移改变行动编号。"""
        obs = snapshot.observation
        if snapshot.frame.frame_id == self.last_id:
            return self._snapshot(snapshot, obs)
        self.last_id = snapshot.frame.frame_id
        ended = obs.get("phase") == "settlement" or any(
            obs.get("remaining", {}).get(s) == 0
            and obs.get("marker_evidence", {}).get("winner_" + s, {}).get("accepted")
            for s in ("self", "left", "right")
        )
        actor = obs.get("actor")
        hand = obs.get("hand")
        target, owner, basis = None, None, "等待本人回合或目标"
        if actor == "self":
            try:
                target, owner, basis = visible_target(snapshot, self.lead_signature)
            except ObservationWait as exc:
                basis = str(exc)
        signature = (
            actor,
            tuple(sorted(hand)) if hand is not None else None,
            tuple(sorted(target)) if target is not None else None,
            owner,
            ended,
        )
        if self.stable is None or self.stable[0] != signature:
            self.stable = signature, snapshot.frame.media_pts_us
        stable = snapshot.frame.media_pts_us - self.stable[1] >= 100_000
        if stable:
            old_actor = self.state["current_actor"]
            if hand is not None and Counter(hand) != Counter(self.state["hand"] or []):
                expected = self.expected_hand
                if expected is None or Counter(hand) != expected:
                    self.state["rule_halt_reason"] = "历史缺口后出现无法追溯的本人手牌变化"
                else:
                    self.state["hand"] = list(hand)
                    self.state["remaining"]["self"] = len(hand)
            if (
                old_actor == "self"
                and actor in ("left", "right")
                and self.intent is None
                and not self.credited
            ):
                self.state["rule_halt_reason"] = "本人回合未经机器人提交便结束，可能已有托管或人工输入"
            if actor == "self" and old_actor != "self":
                self.state["turn"] += 1
                self.credited = False
                self.expected_hand = None
            if actor is not None:
                self.state["current_actor"] = actor
                if old_actor == "self" and actor != "self":
                    self.credited = False
            for seat, count in obs.get("remaining", {}).items():
                previous = self.count_probes.get(seat)
                if count is None:
                    self.count_probes.pop(seat, None)
                elif previous is None or previous[0] != count:
                    self.count_probes[seat] = count, snapshot.frame.media_pts_us
                elif snapshot.frame.media_pts_us - previous[1] >= 100_000:
                    self.state["remaining"][seat] = count
            if target is not None:
                self.state["rule_target"], self.state["rule_target_owner"] = list(target), owner
            if ended:
                self.state["ended"], self.state["phase"] = True, "settlement"
        accepted = stable and target is not None and not self.state["rule_halt_reason"]
        obs.setdefault("marker_evidence", {})["rule_target"] = {
            "accepted": accepted,
            "frame_id": snapshot.frame.frame_id,
            "pts_us": snapshot.frame.media_pts_us,
            "cards": target,
            "reason": basis,
        }
        return self._snapshot(snapshot, obs)

    def _snapshot(self, snapshot, obs):
        state_json = encode(self.state)
        semantics = dict(self.state)
        # 规则动作只由自己的手牌和当前目标决定；对手数字短暂刷新不能使选牌失效。
        semantics.pop("remaining", None)
        version = sha256((encode(semantics) + snapshot.history_json).encode()).hexdigest()
        return replace(
            snapshot, state_json=state_json, observation_json=encode(obs), version=version, pending=False
        )
