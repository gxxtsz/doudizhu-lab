"""与视觉实现无关的公开牌桌观察契约；None与空列表语义严格区分。"""

from dataclasses import dataclass, field

SEATS = ("self", "left", "right")
SEAT_NAMES = {"self": "自己", "left": "左侧", "right": "右侧"}
PHASE_NAMES = {
    "unknown": "无法确认",
    "dealing": "发牌",
    "bidding": "叫抢地主",
    "doubling": "加倍",
    "playing": "出牌",
    "settlement": "结算",
}


@dataclass
class TableObservation:
    """一帧的可见证据。None 代表无法确认；空牌列表仅用于明确的空区域。

    pts_us 为会话媒体时间；frame_id 唯一绑定原始帧。scores 是模板分数而非概率。
    自己手牌与各家出牌均保留重复张数，对手未公开的手牌不在数据模型内。
    """

    frame_id: str
    pts_us: int
    captured_at: str = ""
    phase: str = "unknown"
    landlord: str | None = None
    actor: str | None = None
    hand: list[str] | None = None
    bottom: list[str] | None = None
    plays: dict[str, list[str] | None] = field(default_factory=lambda: dict.fromkeys(SEATS))
    remaining: dict[str, int | None] = field(default_factory=lambda: dict.fromkeys(SEATS))
    passes: list[str] = field(default_factory=list)
    scores: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    regions: list[dict] = field(default_factory=list)
    elapsed_ms: float = 0.0
    pass_labels: dict[str, str] = field(default_factory=dict)
    # 每项保存原图框、原始文字、匹配分数及拒识原因；旧观察缺少该字段时使用空字典。
    marker_evidence: dict[str, dict] = field(default_factory=dict)
