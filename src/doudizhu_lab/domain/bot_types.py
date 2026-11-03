"""机器人模块的不可变边界；观察事实、模型意图和执行结果分别保存。"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, is_dataclass, replace

import numpy as np

from doudizhu_lab.domain.models import Frame, PixelRect


def encode(value) -> str:
    """以稳定 JSON 序列化，防止调用方修改嵌套对象后改变已发布快照。"""
    return json.dumps(asdict(value) if is_dataclass(value) else value, ensure_ascii=False, sort_keys=True)


@dataclass(frozen=True)
class HandCard:
    """当前布局中的一张牌；index 不跨帧复用，selected=None 表示不可执行。"""

    rank: str
    index: int
    hit: PixelRect
    selected: bool | None


@dataclass(frozen=True)
class ActionButton:
    """当前帧内的操作按钮；enabled 必须有视觉证据，不能只根据位置推断。"""

    kind: str
    hit: PixelRect
    enabled: bool
    label: str


@dataclass(frozen=True)
class ObservationSnapshot:
    """同一源帧对应的事实快照；JSON 属性每次返回副本，RGB 底层为不可写 bytes。

    version 是语义哈希，不随重复帧、牌面排列或选中上移改变。history 仅保存本局已确认
    事件；pending 只表示待确认观察，不会被转换为出牌事实。offline 来源永不获得输入能力。
    """

    frame: Frame
    state_json: str
    observation_json: str
    history_json: str
    recognition_json: str
    version: str
    pending: bool = False
    allowed: bool = False
    hand: tuple[HandCard, ...] = ()
    buttons: tuple[ActionButton, ...] = ()
    control_reason: str = "尚未完成操作区域识别"
    revision: int = 0

    @property
    def state(self) -> dict:
        return json.loads(self.state_json)

    @property
    def observation(self) -> dict:
        return json.loads(self.observation_json)

    @property
    def history(self) -> list[dict]:
        return json.loads(self.history_json)

    @classmethod
    def create(cls, frame, recognition, observation, state, history, *, pending=False, allowed=False):
        """在分析结果产生的线程内一次构造；拒绝图像与语义观察的帧标识不一致。"""
        obs = asdict(observation) if is_dataclass(observation) else observation
        from doudizhu_lab.domain.inference import frame_context

        obs = dict(obs)
        obs["frame_context"] = dict(obs.get("frame_context") or frame_context(frame))
        raw_state = asdict(state) if is_dataclass(state) else state
        obs["frame_context"].setdefault("round_id", raw_state.get("game_id", ""))
        if obs["frame_id"] != frame.frame_id or obs["pts_us"] != frame.media_pts_us:
            raise ValueError("机器人快照的观察与图像不是同一帧")
        if recognition is not None and recognition.frame_time != frame.captured_at:
            raise ValueError("机器人快照的牌面识别与图像不是同一帧")
        state_text, history_text = encode(state), encode(history)
        rgb = np.frombuffer(frame.rgb.tobytes(), dtype=np.uint8).reshape(frame.rgb.shape)
        immutable = replace(frame, rgb=rgb)
        return cls(
            immutable,
            state_text,
            encode(obs),
            history_text,
            encode(recognition) if recognition else "null",
            hashlib.sha256((state_text + history_text).encode()).hexdigest(),
            pending,
            allowed,
        )


@dataclass(frozen=True)
class Decision:
    """模型建议而非事实；绑定局面版本和行动轮次，旧建议不得执行。

    strategy 明确区分 resnet、wp 和 rules；默认 wp 保留旧账本及调用方兼容。
    value 保存对应后端的原始评分，既非校准胜率，也不能在不同后端之间直接比较。
    """

    cards: tuple[str, ...]
    version: str
    game_id: str
    turn: int
    role: str
    model_id: str
    elapsed_ms: float
    value: float
    revision: int = 0
    strategy: str = "wp"

    @property
    def kind(self) -> str:
        return "play" if self.cards else "pass"


class SnapshotSequence:
    """单个分析源的递增局面版本；只有语义哈希变化才递增，不由画面帧号驱动。

    调用者在唯一分析线程中发布，随后可将不可变引用交给其他线程。version 仍保存
    内容哈希供跨进程比对，revision 则提供可读的递增版本号，两者随决策写入账本。
    """

    def __init__(self):
        self.version, self.revision = "", 0

    def publish(self, snapshot: ObservationSnapshot) -> ObservationSnapshot:
        """重复画面、选牌上移及布局变化不递增；新行动、身份、手牌和疑点变化会递增。"""
        if snapshot.version != self.version:
            self.version = snapshot.version
            self.revision += 1
        return replace(snapshot, revision=self.revision)
