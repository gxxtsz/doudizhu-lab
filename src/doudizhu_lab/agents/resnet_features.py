"""固定评测版 ResNet 的公开信息编码；本模块不导入 PyTorch、Qt 或设备实现。"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from doudizhu_lab._vendor.douzero.env.env import _cards2array, _get_one_hot_array

# 上游评测使用固定兼容特征，并未把真实叫抢、加倍结果作为输入。
# 严格保留其浮点数值（尤其是 0.5），不得将这些常量写成观察事实。
COMPATIBILITY_FEATURES = np.array([1, 0.5, 1, 1, 1, 1, 1, 1, -4, 1, 1, 1, 1, 1, 1], dtype=np.float32)
COMPATIBILITY_FEATURES.setflags(write=False)
ROLE_ORDER = ("landlord", "landlord_up", "landlord_down")


def get_obs(infoset: SimpleNamespace, actions: list[list[int]] | None = None) -> dict:
    """把已验证公开局面编码为固定上游网络所需的两个 float32 张量。

    infoset 只需自己的手牌、两家合并剩余牌、剩余底牌、公开已出牌、剩牌计数及行动历史。
    actions 可提供连续候选分片，默认编码全部 legal_actions；不改变输入或候选顺序。
    z_batch 为 N×40×54：候选动作、计数、本人手牌、合并未知牌、剩余底牌、三家已出牌、
    32次倒序历史。历史不足时在前面填 -1，真实“不出”填零，两者不可混同。
    x_batch 为 N×15 的兼容常量；不读取 all_handcards 或真实叫抢、倍数信息。
    """
    actions = infoset.legal_actions if actions is None else actions
    if not actions or infoset.player_position not in ROLE_ORDER:
        raise ValueError("ResNet 需要有效身份和非空合法候选动作")
    history = np.full((32, 54), -1, dtype=np.float32)
    recent = infoset.card_play_action_seq[-32:][::-1]
    for index, cards in enumerate(recent, start=32 - len(recent)):
        history[index] = _cards2array(cards)
    counts = np.concatenate(
        [
            _get_one_hot_array(infoset.num_cards_left_dict[role], 20 if role == "landlord" else 17)
            for role in ROLE_ORDER
        ]
    )
    state = np.vstack(
        [
            counts,
            _cards2array(infoset.player_hand_cards),
            _cards2array(infoset.other_hand_cards),
            _cards2array(infoset.three_landlord_cards),
            *[_cards2array(infoset.played_cards[role]) for role in ROLE_ORDER],
            history,
        ]
    ).astype(np.float32)
    z_batch = np.empty((len(actions), 40, 54), dtype=np.float32)
    z_batch[:, 1:] = state
    for index, cards in enumerate(actions):
        z_batch[index, 0] = _cards2array(cards)
    return {
        "position": infoset.player_position,
        "legal_actions": actions,
        "z_batch": z_batch,
        "x_batch": np.repeat(COMPATIBILITY_FEATURES[None, :], len(actions), axis=0),
    }
