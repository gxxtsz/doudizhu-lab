"""DouZero 的公开信息适配及独立推理进程管理；此模块不导入 torch 或 Qt。"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import threading
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Empty, Queue
from time import perf_counter
from types import SimpleNamespace

from doudizhu_lab._vendor.douzero.env.game import GameEnv
from doudizhu_lab.domain.bot_types import Decision, ObservationSnapshot
from doudizhu_lab.domain.models import RANKS
from doudizhu_lab.domain.settings import PLAY_MODEL_NAMES

SOURCE_COMMIT = "c01062592f362cfb77d6af14a34c2cbcfa614fd2"
ROLES = ("landlord", "landlord_down", "landlord_up")
WEIGHT_HASHES = {
    "landlord": "132e26479fcb69f457ddbf1ad32a7bdc3aa73cde80a28cc91f29a62c6075955a",
    "landlord_down": "964720faf583f4905662c94aaf73b1a97352b76ae3686fcc949beabbacdd95d1",
    "landlord_up": "b849ea518e66f088d635d3a29956da880f7b5ca89fabc5f70fc356c106ca6a15",
}
TO_NUMBER = dict(zip(RANKS, [*range(3, 15), 17, 20, 30], strict=True))
TO_RANK = {v: k for k, v in TO_NUMBER.items()}
DECK = Counter({n: 1 if n in (20, 30) else 4 for n in TO_RANK})


def legal_actions(hand: list[int], previous: list[int]) -> list[list[int]]:
    """复用固定上游的纯规则函数，只提供自己的手牌及最近有效牌组。

    不初始化 GameEnv 的其他玩家手牌，不通过编造完整发牌获得合法动作。
    """
    proxy = SimpleNamespace(
        info_sets={"self": SimpleNamespace(player_hand_cards=hand)},
        acting_player_position="self",
        card_play_action_seq=[previous] if previous else [],
    )
    actions = GameEnv.get_legal_card_play_actions(proxy)
    return [list(v) for v in dict.fromkeys(tuple(sorted(a)) for a in actions)]


def context(snapshot: ObservationSnapshot) -> dict:
    """只从已确认公开事实生成模型输入；缺口、不完整开局及矛盾均拒绝推理。

    模型的 other_hand_cards 是两家未知牌合并集合。不能读取模拟器 all_handcards，
    也不能将缺失历史填成不出；缺少一次行动即与 state.turn 不匹配并拒绝。
    """
    state, obs = snapshot.state, snapshot.observation
    from doudizhu_lab.agents.readiness import ObservationWait

    if state["pending"] or state["capture_incomplete"]:
        raise ValueError("存在历史缺口或待确认动作，请人工接管")
    if snapshot.pending:
        raise ObservationWait("动作候选尚未确认，等待连续画面")
    if state["ended"] or state["phase"] != "playing" or state["current_actor"] != "self":
        raise ValueError("当前不是自己的可信出牌回合")
    if not state["landlord"] or not state["bottom"] or not state["initial_hand"]:
        raise ObservationWait("开局身份、底牌或初始手牌不完整")
    if sorted(state["seat_order"]) != ["left", "right", "self"]:
        raise ValueError("尚未核实座位轮转")
    if obs.get("hand") is None or Counter(obs["hand"]) != Counter(state["hand"] or []):
        raise ObservationWait("当前画面手牌与可信状态不一致")
    own_button = obs.get("marker_evidence", {}).get("phase", {})
    own_turn = obs.get("actor") == "self" or (
        obs.get("actor") is None
        and own_button.get("accepted")
        and own_button.get("semantic") == "self_action_buttons"
    )
    if obs.get("phase") != "playing" or not own_turn:
        raise ObservationWait("当前画面未确认自己的行动提示")
    start = state["seat_order"].index(state["landlord"])
    mapping = {state["seat_order"][(start + i) % 3]: role for i, role in enumerate(ROLES)}
    moves = [e for e in snapshot.history if e["kind"] in ("play", "pass")]
    if len(moves) != state["turn"] or any(not e.get("confirmed", True) for e in moves):
        raise ValueError("已确认动作历史不连续")
    if any(e["game_id"] != state["game_id"] for e in snapshot.history):
        raise ValueError("动作历史混入其他牌局")
    if state["seat_order"][(start + state["turn"]) % 3] != "self":
        raise ValueError("行动轮次与当前玩家矛盾")
    hand = sorted(TO_NUMBER[c] for c in state["hand"])
    played = {mapping[s]: sorted(TO_NUMBER[c] for c in cards) for s, cards in state["played"].items()}
    visible = Counter(hand)
    for cards in played.values():
        visible.update(cards)
    if visible - DECK:
        raise ValueError("公开牌张超过一副牌")
    for seat, count in state["remaining"].items():
        if (
            not isinstance(count, int)
            or count <= 0
            or count + len(state["played"][seat]) != (20 if seat == state["landlord"] else 17)
        ):
            raise ValueError("剩牌数与历史牌张不一致")
    if len(hand) != state["remaining"]["self"] or len(state["initial_hand"]) != (
        20 if state["landlord"] == "self" else 17
    ):
        raise ValueError("自己的手牌数量不一致")
    last = {role: [] for role in ROLES}
    sequence = []
    bombs = 0
    history_played = {seat: Counter() for seat in state["seat_order"]}
    effective, passes = [], 0
    for i, event in enumerate(moves):
        expected = state["seat_order"][(start + i) % 3]
        if event["seat"] != expected:
            raise ValueError("历史行动顺序不连续")
        cards = sorted(TO_NUMBER[c] for c in event["cards"])
        if (event["kind"] == "pass") != (not cards):
            raise ValueError("历史动作类型与牌点不一致")
        if cards not in legal_actions(cards, effective):
            raise ValueError("历史牌型、压牌或领出权矛盾")
        history_played[event["seat"]].update(event["cards"])
        if cards:
            effective, passes = cards, 0
        else:
            passes += 1
            if passes == 2:
                effective, passes = [], 0
        sequence.append(cards)
        last[mapping[event["seat"]]] = cards
        bombs += int(cards == [20, 30] or len(cards) == 4 and len(set(cards)) == 1)
    if any(history_played[s] != Counter(state["played"][s]) for s in history_played):
        raise ValueError("公开出牌统计与历史不一致")
    if Counter(state["hand"]) + history_played["self"] != Counter(state["initial_hand"]):
        raise ValueError("初始手牌与当前手牌、已出牌不守恒")
    if effective != sorted(TO_NUMBER[c] for c in state["last_play"]):
        raise ValueError("最近有效出牌与历史不一致")
    # 上游特征的 last_move 为最近动作或上一手；两次不出后应为空。
    previous = sequence[-1] if sequence else []
    if sequence and not previous:
        previous = sequence[-2] if len(sequence) >= 2 else []
    actions = legal_actions(hand, sorted(TO_NUMBER[c] for c in state["last_play"]))
    return {
        "player_position": mapping["self"],
        "player_hand_cards": hand,
        "other_hand_cards": sorted((DECK - visible).elements()),
        "num_cards_left_dict": {mapping[s]: n for s, n in state["remaining"].items()},
        "played_cards": played,
        "last_move": previous,
        "last_move_dict": last,
        "card_play_action_seq": sequence,
        "bomb_num": bombs,
        "legal_actions": actions,
        # 对齐上游“底牌剩余情况”：地主打出的同点牌按多重集合扣除，不读取隐藏手牌。
        "three_landlord_cards": sorted(
            (Counter(TO_NUMBER[c] for c in state["bottom"]) - Counter(played["landlord"])).elements()
        ),
    }


def verify_models(folder: Path, backend: str = "wp") -> dict:
    """每次启动核对完整权重与固定来源清单；不接受随机初始化或悄悄缺少参数。"""
    if backend == "resnet":
        from doudizhu_lab.agents.backends import verify_resnet_models

        return verify_resnet_models(folder)
    if backend != "wp":
        raise ValueError("未知出牌模型")
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    if manifest["source_commit"] != SOURCE_COMMIT or set(manifest["weights"]) != set(ROLES):
        raise ValueError("模型来源清单不匹配")
    for role, entry in manifest["weights"].items():
        if entry["sha256"] != WEIGHT_HASHES[role]:
            raise ValueError("模型清单与固定版本哈希不同")
        if hashlib.sha256((folder / f"{role}.ckpt").read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError(f"模型文件校验失败：{role}")
    return manifest


def setup_agent(root: Path) -> dict:
    """下载固定提交中的 WP 副本并运行独立进程检查；原子替换完整文件后才发布清单。"""
    folder = root / "models/douzero-wp"
    folder.mkdir(parents=True, exist_ok=True)

    def fetch(role):
        url = f"https://raw.githubusercontent.com/tianqiraf/DouZero_For_HappyDouDiZhu/{SOURCE_COMMIT}/baselines/douzero_WP/{role}.ckpt"
        target = folder / f"{role}.ckpt"
        with urllib.request.urlopen(url, timeout=60) as response:
            data = response.read()
        if len(data) < 1_000_000 or not data.startswith(b"PK"):
            raise ValueError("模型下载内容不是完整权重")
        if hashlib.sha256(data).hexdigest() != WEIGHT_HASHES[role]:
            raise ValueError("下载的模型与固定版本哈希不同")
        temporary = target.with_suffix(".tmp")
        temporary.write_bytes(data)
        temporary.replace(target)
        return role, {"url": url, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}

    with ThreadPoolExecutor(max_workers=3) as pool:
        weights = dict(pool.map(fetch, ROLES))
    license_url = (
        f"https://raw.githubusercontent.com/tianqiraf/DouZero_For_HappyDouDiZhu/{SOURCE_COMMIT}/LICENSE"
    )
    with urllib.request.urlopen(license_url, timeout=30) as response:
        (folder / "LICENSE-社区仓库.txt").write_bytes(response.read())
    manifest = {
        "source_commit": SOURCE_COMMIT,
        "origin": "社区发布的 WP 副本，未声称与原官方下载字节相同",
        "weights": weights,
    }
    (folder / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    agent = Agent(folder)
    try:
        agent.start()
        manifest["verified_roles"] = list(ROLES)
        verification = agent.ready_info | {
            "model_id": agent.model_id,
            "source_commit": SOURCE_COMMIT,
            "strict_state_dict": True,
            "weights_only": True,
        }
        (folder / "verification.json").write_text(
            json.dumps(verification, ensure_ascii=False, indent=2), encoding="utf8"
        )
    finally:
        agent.close()
    return manifest


class Agent:
    """持有一个隐藏窗口的 CPU 推理子进程；启动预热与单次推理分别限时。

    子进程只有结构化公开信息输入，没有 ADB 或 Qt 对象。超时直接结束该子进程，
    丢弃迟到结果；再次启动必须由上层明确恢复。不同调用由锁串行，禁止推理积压。
    """

    def __init__(self, folder: Path, *, backend: str = "wp"):
        """绑定明确模型类型；旧 Agent(folder) 继续使用 WP，构造时不加载或下载权重。"""
        if backend not in PLAY_MODEL_NAMES:
            raise ValueError("未知出牌模型")
        self.backend = backend
        self.display_name = PLAY_MODEL_NAMES[backend]
        self.folder = folder.resolve()
        self.process = None
        self.responses = Queue()
        self.lock = threading.RLock()
        self.model_id = ""
        self.ready_info = {}

    def _read(self, process, responses):
        try:
            for line in process.stdout:
                responses.put(json.loads(line))
        except (ValueError, OSError):
            pass
        finally:
            responses.put({"error": "模型进程已经退出"})

    def check_running(self):
        """核查已预热的推理子进程仍存活；不自动重启，也不等待下一次自己的回合。

        协调线程每轮调用本接口，使模型在叫抢或对手回合退出时也立即撤销后续页面输入。
        它仅执行本地进程状态查询，避免与进行中的推理争用锁或产生新的模型请求。
        """
        process = self.process
        if process is None or process.poll() is not None:
            raise RuntimeError("模型子进程已经退出，请停止任务后重新检查模型")

    def start(self):
        """核验权重、加载三种身份并预热；进程就绪后每次决策才使用 2 秒限时。"""
        with self.lock:
            if self.process is not None and self.process.poll() is None:
                return
            try:
                manifest = (
                    verify_models(self.folder)
                    if self.backend == "wp"
                    else verify_models(self.folder, self.backend)
                )
            except (OSError, ValueError, KeyError, TypeError) as exc:
                error_type = OSError if isinstance(exc, OSError) else ValueError
                raise error_type(
                    f"{self.display_name} 模型未准备或校验失败：{exc}；"
                    f"请执行 ddz-check setup-agent --model {self.backend}，或在设置中选择已准备的模型"
                ) from exc
            self.model_id = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
            self.responses = Queue()
            error = (self.folder / "推理错误.log").open("a", encoding="utf-8")
            try:
                self.process = subprocess.Popen(
                    [
                        sys.executable,
                        "-u",
                        "-X",
                        "utf8",
                        "-m",
                        "doudizhu_lab.agents.worker",
                        str(self.folder),
                        self.backend,
                    ],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=error,
                    encoding="utf-8",
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                )
            finally:
                error.close()
            threading.Thread(target=self._read, args=(self.process, self.responses), daemon=True).start()
            result = self._receive(30)
            if not result.get("ready"):
                self.close()
                raise ValueError(result.get("error", "模型未就绪"))
            self.ready_info = result

    def _receive(self, timeout):
        try:
            result = self.responses.get(timeout=timeout)
        except Empty:
            self.close()
            raise ValueError("模型推理超时，已停止进程") from None
        if not isinstance(result, dict):
            self.close()
            raise ValueError("模型返回的协议数据必须为 JSON 对象")
        if "error" in result:
            self.close()
            raise ValueError(result["error"])
        return result

    def decide(self, snapshot: ObservationSnapshot) -> Decision:
        """返回与输入版本绑定的单次建议；仅允许已预热进程，不在自己的回合临时加载模型。"""
        value = context(snapshot)
        with self.lock:
            if self.process is None or self.process.poll() is not None:
                raise ValueError("模型未就绪，请先加载模型")
            start = perf_counter()
            self.process.stdin.write(json.dumps(value) + "\n")
            self.process.stdin.flush()
            output = self._receive(2)
            cards = output["cards"]
            if (
                not isinstance(cards, list)
                or any(type(card) is not int for card in cards)
                or cards not in value["legal_actions"]
            ):
                raise ValueError("模型返回了非法动作")
            score = output["value"]
            if type(score) not in (int, float) or not float("-inf") < score < float("inf"):
                raise ValueError("模型返回了无效评分")
            state = snapshot.state
            return Decision(
                tuple(TO_RANK[n] for n in cards),
                snapshot.version,
                state["game_id"],
                state["turn"],
                value["player_position"],
                self.model_id,
                (perf_counter() - start) * 1000,
                score,
                snapshot.revision,
                strategy=self.backend,
            )

    def close(self):
        """只清理本实例创建的子进程，不触碰其他 Python 或 ADB 进程。"""
        with self.lock:
            process, self.process = self.process, None
            if process is not None:
                if process.poll() is None:
                    process.terminate()
                process.wait(timeout=5)
                process.stdin.close()
                process.stdout.close()

    def interrupt(self):
        """关闭期间打断尚在预热或推理的子进程，不等待推理锁；收尾由 close 完成。"""
        process = self.process
        if process is not None and process.poll() is None:
            process.terminate()
