"""原始分局与运行分局的唯一关联；运行和离线评测使用同一套只读边界检查。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class RoundEvidenceBinding:
    """一副真实发牌的证据归属，不要求两套跟踪器生成相同的牌局编号。

    first_evidence_us 是双方开局事件共同的源媒体边界；confirmed_frame_id 和
    confirmed_at_us 指向原始跟踪器实际确认新局的观察，而不是仍带旧编号的首帧。
    frame_context 保留原始设备、视频流、尺寸及 raw_game_id；不能用于恢复输入授权。
    调用者持久化时使用 dataclasses.asdict，不得修改其中的原始上下文字段。
    """

    game_id: str
    raw_game_id: str
    first_evidence_us: int
    confirmed_frame_id: str
    confirmed_at_us: int
    frame_context: dict


def resolve_round_binding(
    start: dict, raw_events: Sequence[dict], observations: Mapping[str, dict]
) -> RoundEvidenceBinding | None:
    """按唯一原始发牌边界解析分局，缺证、重复、跨设备或跨视频流时返回 None。

    start 是运行开局，raw_events 必须来自原始 events，observations 必须来自不可变
    原始观察。两边都必须确认从发牌／叫抢开始，且 first_evidence_us 完全相同。
    原始确认可能晚于运行开局；确认帧尚未到达时返回 None，调用方只能继续只读等待。
    兼容第二版旧事件没有 confirmed_frame_id 的情况：从确认源时间唯一寻找原始帧，
    绝不把首帧的旧 round_id 当成新局，也不通过删除 round_id 校验来绕过跨局保护。
    """
    try:
        if start["kind"] != "game_start" or not start.get("confirmed", True):
            return None
        data = start["data"]
        first = data["first_evidence_us"]
        if data.get("capture_incomplete", True) or data.get("phase") not in ("dealing", "bidding"):
            return None
        matches = [
            e
            for e in raw_events
            if e.get("kind") == "game_start"
            and e.get("confirmed", True)
            and e.get("data", {}).get("first_evidence_us") == first
        ]
        if len(matches) != 1:
            return None
        original = matches[0]
        origin = original["data"]
        confirmed = origin["confirmed_at_us"]
        if (
            origin.get("capture_incomplete", True)
            or origin.get("phase") not in ("dealing", "bidding")
            or not isinstance(first, int)
            or isinstance(first, bool)
            or not isinstance(confirmed, int)
            or confirmed < first
        ):
            return None
        fid = origin.get("confirmed_frame_id")
        if fid:
            candidates = [observations[fid]]
        else:
            candidates = [o for o in observations.values() if o["pts_us"] == confirmed]
        if len(candidates) != 1 or candidates[0]["pts_us"] != confirmed:
            return None
        anchor = candidates[0]
        context = anchor["frame_context"]
        source = observations[start["frame_id"]]["frame_context"]
        if (
            not context.get("serial")
            or not context.get("stream_id")
            or context.get("round_id") != original["game_id"]
            or anchor.get("phase") not in ("dealing", "bidding", "unknown")
            or any(context.get(k) != source.get(k) for k in ("serial", "stream_id", "width", "height"))
            or context.get("width", 0) <= 0
            or context.get("height", 0) <= 0
        ):
            return None
        return RoundEvidenceBinding(
            start["game_id"], original["game_id"], first, anchor["frame_id"], confirmed, dict(context)
        )
    except (KeyError, TypeError, ValueError):
        return None
