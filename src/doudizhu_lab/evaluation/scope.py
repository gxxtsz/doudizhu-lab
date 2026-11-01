"""显式录像评测范围；只决定选局，不提供模型输入或人工事件答案。"""

from __future__ import annotations

import json
from pathlib import Path


def load_scope(path: Path) -> dict:
    """只读解析预先选定的连续牌局边界，拒绝空、逆序、重叠或不完整的范围格式。

    start_us/end_us使用录像源时间，不是墙上时间；incomplete显式声明首尾不完整局。
    调用者负责范围的独立人工来源，本函数不将机器预测当作真值。文件／JSON错误及
    范围错误直接上抛，由CLI显示中文诊断；不创建数据库、Qt或设备对象。
    """
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("source"), str) or not value["source"]:
        raise ValueError("范围清单必须说明source来源")
    games = value.get("games")
    if not isinstance(games, list) or not games:
        raise ValueError("范围清单games不能为空")
    previous = -1
    for game in games:
        if (
            not isinstance(game, dict)
            or type(game.get("start_us")) is not int
            or type(game.get("end_us")) is not int
            or type(game.get("incomplete")) is not bool
            or game["start_us"] < 0
            or game["start_us"] >= game["end_us"]
            or game["start_us"] < previous
        ):
            raise ValueError("牌局范围必须是有序、不重叠的源时间区间，且明确incomplete")
        previous = game["end_us"]
    return value
