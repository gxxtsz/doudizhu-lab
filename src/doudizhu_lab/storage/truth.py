"""独立录像标注。只读取媒体信息，不读取自动观察、自动事件或人工修订。"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from doudizhu_lab.domain.models import RANKS
from doudizhu_lab.domain.table import SEATS
from doudizhu_lab.domain.tracker import GameState, apply_event, canonical
from doudizhu_lab.storage.replay import VideoReplay
from doudizhu_lab.storage.session import SessionStore


def media_manifest(folder: Path) -> list[dict]:
    """逐块计算原录像片段哈希，绑定标注与实际媒体；不加载整段录像到内存。

    分析版本可引用源录像，但来源会话编号仍明确保存。缺失、未收尾的文件直接报错，
    不将有索引而没有可读取录像的会话用于正式验收。
    """
    video = VideoReplay(folder)
    result = []
    for item in video.segments:
        if not item.get("closed"):
            raise ValueError("录像尚未收尾，不能发布独立验收标注")
        path = video.video / item["file"]
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while block := stream.read(4 * 1024 * 1024):
                digest.update(block)
        result.append(
            {
                "file": item["file"],
                "bytes": path.stat().st_size,
                "sha256": digest.hexdigest(),
                "start_us": item["global_start_us"],
                "end_us": item["global_end_us"],
            }
        )
    return result


def truth_events(document: dict) -> list[dict]:
    """把人工标注转换为用于校验的纯状态事件；这是标注校验，不是生成真值。

    初始手牌指收底牌完成时的手牌（地主 20 张，农民 17 张）。公开底牌不重复相加。
    此接口不会写入会话，也不会改动任何预测。
    """
    events = []
    for game in document.get("games", []):
        role = game.get("landlord")
        ident = game["game_id"]
        events.append(
            {
                "event_id": ident + "-start",
                "game_id": ident,
                "pts_us": game["start_us"],
                "frame_id": game.get("start_frame", ""),
                "kind": "game_start",
                "source": "independent",
                "data": {
                    "phase": "playing",
                    "landlord": role,
                    "initial_hand": game.get("initial_hand"),
                    "bottom": game.get("bottom"),
                    "seat_order": game.get("seat_order", []),
                    "current_actor": role,
                    "remaining": {s: 20 if s == role else 17 for s in SEATS},
                    "capture_incomplete": bool(game.get("incomplete", False)),
                },
            }
        )
        events.extend(deepcopy(e) for e in document.get("events", []) if e.get("game_id") == ident)
        if game.get("end_us") is not None:
            events.append(
                {
                    "event_id": ident + "-end",
                    "game_id": ident,
                    "pts_us": game["end_us"],
                    "frame_id": game.get("end_frame", ""),
                    "kind": "game_end",
                    "source": "independent",
                }
            )
    return sorted(events, key=lambda e: e["pts_us"])


def validate_truth(document: dict, *, require_review: bool = False) -> dict:
    """核对手工牌局边界、逐动作顺序、牌数和手牌；返回问题与逐动作状态。

    草稿允许信息未补齐，问题列表不会因此被隐藏。正式发布还要求每局及每个动作
    都有独立录像核对记录、时间来源和手牌/剩牌检查，不能只将机器预测改名为标注。
    """
    problems, states, checkpoints = [], {}, {}
    games = sorted(document.get("games", []), key=lambda g: g["start_us"])
    ids = [g["game_id"] for g in games]
    if len(ids) != len(set(ids)):
        problems.append("牌局编号重复")
    for previous, game in zip(games, games[1:]):
        if previous.get("end_us") is None or previous["end_us"] >= game["start_us"]:
            problems.append("牌局时间范围重叠或前一局没有结束时间")
    actions = document.get("events", [])
    if len({e.get("event_id") for e in actions}) != len(actions):
        problems.append("动作编号重复")
    for game in games:
        ident = game["game_id"]
        if game.get("incomplete"):
            continue
        role = game.get("landlord")
        hand = game.get("initial_hand") or []
        if role not in SEATS or len(hand) != (20 if role == "self" else 17):
            problems.append(f"{ident}：地主或初始手牌未核对完整")
        if game.get("end_us") is None or game["end_us"] <= game["start_us"]:
            problems.append(f"{ident}：缺少有效结算边界")
        if require_review and not game.get("reviewed"):
            problems.append(f"{ident}：尚未完成整局逐帧复核")
        if require_review and (not game.get("start_frame") or not game.get("end_frame")):
            problems.append(f"{ident}：缺少边界来源帧")
    game_map = {g["game_id"]: g for g in games}
    for event in actions:
        game = game_map.get(event.get("game_id"))
        if not game or event.get("kind") not in ("play", "pass") or event.get("seat") not in SEATS:
            problems.append("动作所属牌局、类型或座位无效")
            continue
        if not game["start_us"] <= event["pts_us"] <= (game.get("end_us") or event["pts_us"]):
            problems.append(f"{event.get('event_id')}：动作超出牌局边界")
        if any(c not in RANKS for c in event.get("cards", [])):
            problems.append("动作含无效牌点")
        if (
            require_review
            and not game.get("incomplete")
            and (not event.get("frame_id") or not event.get("reviewed") or not event.get("checkpoint"))
        ):
            problems.append(f"{event.get('event_id')}：缺少逐动作录像与状态核对记录")
    try:
        for event in truth_events(document):
            ident = event["game_id"]
            if game_map[ident].get("incomplete"):
                continue
            state = apply_event(states.get(ident, GameState()), event)
            states[ident] = state
            if state.pending:
                problems.extend(f"{event['event_id']}：{p}" for p in state.problems)
            if event["kind"] in ("play", "pass"):
                checkpoints[event["event_id"]] = asdict(state)
                check = event.get("checkpoint", {})
                if check.get("hand") is not None and canonical(check["hand"]) != state.hand:
                    problems.append(f"{event['event_id']}：录像手牌与推演不一致")
                if check.get("remaining") is not None and check["remaining"] != state.remaining:
                    problems.append(f"{event['event_id']}：录像剩牌数与推演不一致")
                if require_review and (check.get("hand") is None or check.get("remaining") is None):
                    problems.append(f"{event['event_id']}：未逐项填写手牌和三家剩牌")
        for ident, state in states.items():
            if not game_map[ident].get("incomplete") and not state.complete:
                problems.append(f"{ident}：整局牌数、结束状态或动作顺序未闭合")
    except (KeyError, ValueError, TypeError) as exc:
        problems.append(f"标注字段无效：{exc}")
    selection = document.get("selection")
    if selection:
        for game in games:
            if (
                not selection["start_us"]
                <= game["start_us"]
                < (game.get("end_us") or game["start_us"])
                <= selection["end_us"]
            ):
                problems.append(f"{game['game_id']}：牌局不在预选连续区间内")
    return {
        "valid": not problems,
        "problems": list(dict.fromkeys(problems)),
        "states": {k: asdict(v) for k, v in states.items()},
        "checkpoints": checkpoints,
    }


class TruthStore:
    """独立标注文件及版本历史；构造时只读取会话元数据和录像索引。

    不提供从预测或修订导入真值的接口。所有添加/修改先落草稿，每次保存保留旧版哈希；
    发布只在严格校验后设置 independent_review，后续编辑会自动撤回该发布状态。
    """

    def __init__(self, folder: Path):
        self.folder = folder.resolve()
        self.path = self.folder / "annotations" / "independent.json"
        if self.path.exists():
            self.document = json.loads(self.path.read_text(encoding="utf-8"))
        else:
            session = SessionStore(self.folder)
            meta = session.metadata()
            session.close()
            video = VideoReplay(self.folder)
            self.document = {
                "version": 1,
                "independent_review": False,
                "method": "隐藏预测，逐帧独立手工标注",
                "source": {
                    "session_id": meta["session_id"],
                    "start_us": video.start_us,
                    "end_us": video.end_us,
                },
                "selection": None,
                "games": [],
                "events": [],
                "tolerance_us": 1_000_000,
                "boundary_tolerance_us": 2_000_000,
            }

    def select_range(self, start_us: int, end_us: int) -> None:
        """在标注前登记连续验收区间；已有动作后不允许根据识别成绩重选。

        边界首尾的非完整局应单独建为 incomplete，不计入五局完整验收。
        """
        if self.document["events"]:
            raise ValueError("已经开始标注，不能重新挑选验收区间")
        source = self.document["source"]
        if not source["start_us"] <= start_us < end_us <= source["end_us"]:
            raise ValueError("所选区间必须在录像时间范围内")
        self.document["selection"] = {
            "start_us": start_us,
            "end_us": end_us,
            "locked_at": datetime.now(timezone.utc).isoformat(),
            "method": "预先登记的连续录像区间，首尾不完整局单列",
        }
        self.save()

    def put_game(self, game: dict) -> str:
        """追加或修改人工牌局边界和开局字段；编号不依赖自动分局编号。"""
        value = deepcopy(game)
        value.setdefault("game_id", "人工局-" + uuid4().hex[:8])
        value.setdefault("reviewed", False)
        old = next((g for g in self.document["games"] if g["game_id"] == value["game_id"]), None)
        if old is not None:
            self.document["games"].remove(old)
        self.document["games"].append(value)
        self.document["games"].sort(key=lambda g: g["start_us"])
        self.save()
        return value["game_id"]

    def put_event(self, event: dict) -> str:
        """保存一条独立动作标注；事件须包含来源帧，修改后该局回到待复核状态。"""
        value = deepcopy(event)
        value.setdefault("event_id", "人工动作-" + uuid4().hex[:12])
        if not value.get("frame_id") or value.get("kind") not in ("play", "pass"):
            raise ValueError("动作须从录像帧标注，类型只能为出牌或不出")
        game = next((g for g in self.document["games"] if g["game_id"] == value["game_id"]), None)
        if game is None:
            raise ValueError("请先建立独立牌局边界")
        game["reviewed"] = False
        self.document["events"] = [
            e for e in self.document["events"] if e["event_id"] != value["event_id"]
        ] + [value]
        self.document["events"].sort(key=lambda e: e["pts_us"])
        self.save()
        return value["event_id"]

    def delete_event(self, event_id: str) -> None:
        """删除草稿动作并保留文件历史；不触碰任何会话事件或人工修订。"""
        event = next(e for e in self.document["events"] if e["event_id"] == event_id)
        self.document["events"].remove(event)
        for game in self.document["games"]:
            if game["game_id"] == event["game_id"]:
                game["reviewed"] = False
        self.save()

    def save(self, *, publish: bool = False) -> Path:
        """原子保存草稿或正式标注；每次覆盖前保存旧文件，便于撤回错误标注版本。"""
        self.document["independent_review"] = False
        if publish:
            check = validate_truth(self.document, require_review=True)
            if not check["valid"] or not self.document.get("selection"):
                raise ValueError(
                    "尚不能发布独立标注：" + "；".join(check["problems"][:5] or ["未预选连续区间"])
                )
            self.document["source"]["video_manifest"] = media_manifest(self.folder)
            self.document["independent_review"] = True
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            content = self.path.read_bytes()
            digest = hashlib.sha256(content).hexdigest()
            history = self.path.parent / "history"
            history.mkdir(exist_ok=True)
            (history / f"{digest}.json").write_bytes(content)
        self.document["updated_at"] = datetime.now(timezone.utc).isoformat()
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.document, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(self.path)
        return self.path
