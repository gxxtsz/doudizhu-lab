"""区分可恢复的单帧问题与真实历史矛盾；等待期间禁止设备输入。"""

from dataclasses import dataclass


class ObservationWait(ValueError):
    """当前像素证据不足，允许在同一行动轮等待；不表示可以忽略可信历史。"""


@dataclass(frozen=True)
class DecisionReadiness:
    """status 为 ready/wait/halt；bounded 表示等待最多三秒，普通对手回合无需此限制。"""

    status: str
    reason: str = ""
    bounded: bool = False


def assess(snapshot) -> DecisionReadiness:
    """只读取公开快照，以结构化结果通知协调器；不依靠中文异常文本判断故障类型。"""
    from doudizhu_lab.agents.douzero import context

    state = snapshot.state
    if state.get("strategy") == "rules":
        from doudizhu_lab.agents.rules import context as rule_context

        if state.get("rule_halt_reason") or state.get("capture_incomplete"):
            return DecisionReadiness("halt", state.get("rule_halt_reason") or "采集中断，不能规则接管")
        if state["ended"] or state["current_actor"] != "self":
            return DecisionReadiness("wait", "规则接管：等待本人的明确回合")
        try:
            rule_context(snapshot)
        except ObservationWait as exc:
            return DecisionReadiness("wait", str(exc), True)
        except ValueError as exc:
            return DecisionReadiness("halt", str(exc))
        return DecisionReadiness("ready", "规则接管")
    if state["pending"] or state["capture_incomplete"]:
        return DecisionReadiness("halt", "存在真实历史缺口或不完整开局，请人工接管")
    obs = snapshot.observation
    actor = obs.get("actor")
    if (
        state["phase"] == "playing"
        and not state["ended"]
        and actor is not None
        and actor != state["current_actor"]
    ):
        return DecisionReadiness("wait", "行动提示与可信轮次不一致，等待动作证据", True)
    if state["ended"] or state["phase"] != "playing" or state["current_actor"] != "self":
        return DecisionReadiness("wait", "等待自己的出牌回合")
    try:
        context(snapshot)
    except ObservationWait as exc:
        return DecisionReadiness("wait", str(exc), True)
    except ValueError as exc:
        return DecisionReadiness("halt", str(exc))
    return DecisionReadiness("ready")
