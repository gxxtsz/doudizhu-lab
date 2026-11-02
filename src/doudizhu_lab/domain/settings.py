"""纯配置契约及验证；不访问磁盘、ADB、Qt或模型进程。"""

from dataclasses import asdict, dataclass, field

import numpy as np

from doudizhu_lab.domain.models import NormalizedRect, TrackerSettings

# 连续流程只授权这些洗牌场次；配置、命令行和视觉扫描共用，避免“当前”漏掉高级场。
FLOW_ROOMS: tuple[str, ...] = ("初级", "中级", "高级")
PLAY_MODEL_NAMES = {"resnet": "ResNet", "wp": "WP"}
CONFIGURATION_VERSION = 5


def default_bid_settings() -> dict[str, str | float]:
    """旧配置默认仍不叫抢；显式选模型后使用分别可调的首叫、抢地主及回抢阈值。

    首叫0.2、首次抢0.5沿用上游起点；回抢提高到0.6是本工具稳健策略，并非经过本游戏
    胜率校准的最优值。三个值均为原始评分阈值，不是概率，暂不改变不加倍策略。
    """
    return {"mode": "disabled", "call_threshold": 0.2, "rob_threshold": 0.5, "reclaim_threshold": 0.6}


def default_regions() -> dict[str, NormalizedRect]:
    """2400×1080 王者默认牌桌的初始区域；左右出牌区域允许在窗口中重新校准。"""
    return {
        "hand": NormalizedRect(0.105, 0.66, 0.79, 0.32),
        "bottom": NormalizedRect(0.273, 0.003, 0.065, 0.090),
        "left": NormalizedRect(0.105, 0.185, 0.28, 0.235),
        "right": NormalizedRect(0.49, 0.215, 0.26, 0.22),
        "self_play": NormalizedRect(0.31, 0.47, 0.38, 0.215),
    }


@dataclass
class Profile:
    """牌桌标定配置。threshold/margin 控制拒识，不能解释成识别概率。"""

    name: str = "王者经典 · 2400×1080"
    base_width: int = 2400
    base_height: int = 1080
    regions: dict[str, NormalizedRect] = field(default_factory=default_regions)
    glyph_heights: dict[str, float] = field(
        default_factory=lambda: {"hand": 74, "bottom": 40, "left": 50, "right": 50, "self_play": 50}
    )
    threshold: float = 0.84
    margin: float = 0.055
    # 默认为历史模板算法；显式选择时手牌改用固定预训练OCR，其余区域保持模板。
    hand_classifier: str = "template"
    # 只保存出牌模型偏好；旧配置缺少此字段时升级为 ResNet，不恢复任何执行授权。
    play_model: str = "resnet"
    # 仅保存策略和上限；运行授权、当前模式与未完成任务不得跨重启恢复。
    continuous_settings: dict = field(default_factory=lambda: {"max_games": 5, "time_limit": 1800})
    # 只进入用户明确选择的场次；不因游戏弹窗自动升级。
    flow_room: str = "初级"
    # 策略偏好可以保存，实际连续执行授权仍不保存；旧配置保持严格模式。
    continuous_policy: str = "strict"
    # 只保存叫抢策略，不保存执行授权；旧配置继续固定不叫不抢。
    bid_settings: dict = field(default_factory=default_bid_settings)
    # 视觉标记与牌面分开存放，避免把“不抢”等阶段文字当作牌点。
    indicator_regions: dict[str, NormalizedRect] = field(
        default_factory=lambda: {
            "landlord_self": NormalizedRect(0.09, 0.25, 0.06, 0.11),
            # 昵称长短会推动标志和倒计时横向移动，区域覆盖同一座位的名称栏。
            "landlord_left": NormalizedRect(0.30, 0.11, 0.125, 0.10),
            "landlord_right": NormalizedRect(0.66, 0.15, 0.17, 0.10),
            "turn_self": NormalizedRect(0.28, 0.50, 0.41, 0.18),
            "turn_left": NormalizedRect(0.30, 0.10, 0.125, 0.11),
            "turn_right": NormalizedRect(0.64, 0.14, 0.17, 0.12),
            "count_left": NormalizedRect(0.43, 0.129, 0.14, 0.064),
            "count_right": NormalizedRect(0.79, 0.158, 0.14, 0.065),
            "pass_self": NormalizedRect(0.31, 0.47, 0.38, 0.215),
            "pass_left": NormalizedRect(0.20, 0.19, 0.22, 0.24),
            "pass_right": NormalizedRect(0.59, 0.215, 0.20, 0.22),
            "phase": NormalizedRect(0.29, 0.49, 0.42, 0.20),
            "result": NormalizedRect(0.45, 0.85, 0.50, 0.145),
            "bottom_back": NormalizedRect(0.46, 0.003, 0.08, 0.10),
            "winner_self": NormalizedRect(0.30, 0.48, 0.40, 0.25),
            "winner_left": NormalizedRect(0.20, 0.19, 0.23, 0.26),
            "winner_right": NormalizedRect(0.59, 0.215, 0.20, 0.25),
        }
    )
    seat_order: list[str] = field(default_factory=list)
    tracker_settings: dict = field(default_factory=lambda: asdict(TrackerSettings()))
    # 第三版操作标定：归一化牌顶高度及按钮区域；未标定时禁止实际机器人点击。
    bot_controls: dict = field(
        default_factory=lambda: {
            "normal_top": None,
            "selected_offset": None,
            "top_tolerance": 0.008,
            "buttons": {},
            "verified": False,
        }
    )
    bot_settings: dict = field(
        default_factory=lambda: {
            "max_frame_age": 1.0,
            "selection_timeout": 12.0,
            "confirmation_timeout": 3.0,
            "heartbeat_timeout": 1.0,
        }
    )


def validate_bot_profile(profile: Profile) -> None:
    """只读检验配置，合法时返回 None，字段无效时抛出 ValueError。

    profile 可保持未完成的操作标定，但不能含负数、非有限值或放宽后的超时。
    flow_room 必须为已支持的洗牌场次或“当前”；“当前”仅允许沿用 FLOW_ROOMS。
    本函数不修改配置、不访问设备，可在加载配置或 GUI 保存前调用，不授予输入权限。
    """
    if profile.hand_classifier not in ("template", "pretrained-ocr"):
        raise ValueError("手牌分类器配置无效")
    if not isinstance(profile.play_model, str) or profile.play_model not in PLAY_MODEL_NAMES:
        raise ValueError("出牌模型必须为 resnet 或 wp")
    settings = profile.continuous_settings
    bidding = profile.bid_settings
    if set(bidding) != set(default_bid_settings()) or bidding["mode"] not in ("disabled", "model"):
        raise ValueError("叫抢策略配置无效")
    if any(
        type(bidding[k]) not in (int, float) or not np.isfinite(bidding[k]) or not -10 <= bidding[k] <= 10
        for k in ("call_threshold", "rob_threshold", "reclaim_threshold")
    ):
        raise ValueError("叫抢阈值必须是有限原始评分")
    if not bidding["call_threshold"] <= bidding["rob_threshold"] <= bidding["reclaim_threshold"]:
        raise ValueError("稳健叫抢策略要求首叫阈值不高于抢地主及回抢阈值")
    if profile.continuous_policy not in ("strict", "infer", "balanced"):
        raise ValueError("连续策略必须为严格、唯一推断或平衡模式")
    if profile.flow_room not in (*FLOW_ROOMS, "当前"):
        raise ValueError(f"场次必须为{'、'.join(FLOW_ROOMS)}，或沿用这些洗牌场次的当前场次")
    if set(settings) != {"max_games", "time_limit"} or any(
        type(settings[k]) is not int or not 0 < settings[k] <= limit
        for k, limit in (("max_games", 30), ("time_limit", 14400))
    ):
        raise ValueError("连续任务上限必须为1～30局、1～14400秒；默认仍为5局或30分钟")
    config = profile.bot_controls
    for name in ("normal_top", "selected_offset", "top_tolerance", "glyph_top_padding"):
        value = config.get(name)
        if value is not None and (
            not isinstance(value, (int, float)) or not np.isfinite(value) or not 0 < value < 1
        ):
            raise ValueError(f"机器人布局参数无效：{name}")
    if config.get("verified"):
        if not config.get("normal_top") or not config.get("selected_offset"):
            raise ValueError("已标定布局缺少牌顶或上移量")
        if (
            config["selected_offset"] >= config["normal_top"]
            or 2 * config.get("top_tolerance", 0.008) >= config["selected_offset"]
        ):
            raise ValueError("选中与未选中的高度范围重叠")
    for kind, rect in config.get("buttons", {}).items():
        if kind not in ("play", "pass"):
            raise ValueError("不支持的机器人按钮类型")
        NormalizedRect(**rect)
    limits = {
        "max_frame_age": 1.0,
        "selection_timeout": 12.0,
        "confirmation_timeout": 3.0,
        "heartbeat_timeout": 1.0,
    }
    for name, value in profile.bot_settings.items():
        if (
            name not in limits
            or not isinstance(value, (int, float))
            or not np.isfinite(value)
            or not 0 < value <= limits[name]
        ):
            raise ValueError(f"机器人超时参数超出上限：{name}")
