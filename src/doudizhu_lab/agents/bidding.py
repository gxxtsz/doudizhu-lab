"""免训练叫抢决策：自己的17张牌交给固定模型，公开叫抢阶段只用于选择阈值。

评分模型只理解手牌，不声称它理解王者的倍数、叫抢顺序或对手策略。阶段差异由可追溯
策略处理；识别或模型失败选择已确认可用的拒绝按钮，输入结果不明仍由流程执行器暂停。
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from queue import Queue
from time import perf_counter

from doudizhu_lab.agents.douzero import Agent
from doudizhu_lab.domain.models import RANKS
from doudizhu_lab.domain.settings import default_bid_settings as default_bid_settings
from doudizhu_lab.paths import PACKAGE_ROOT

BID_COMMIT = "6d20491da216c989d23cf2883498758fb78e9f8d"
BID_REPOSITORY = "https://github.com/Vincentzyx/DouZero_For_HLDDZ_FullAuto"
BID_SHA256 = "320e9d96e3aa6f962e27f0b8f5f1d97c15af719bc5cb30ed5423b9c025bb4801"
BID_MODEL_ID = "FullAuto-Net2-" + BID_SHA256[:16]
BID_KINDS = frozenset({"call_landlord", "rob_landlord", "decline_bid"})


def encode_hand(hand):
    """验证自己的完整17张牌并复现上游 RealToOnehot；保留重复牌，大小王各至多一张。"""
    if not isinstance(hand, (tuple, list)) or len(hand) != 17 or any(c not in RANKS for c in hand):
        raise ValueError("叫抢需要完整且合法的17张手牌")
    counts = Counter(hand)
    if any(n > (1 if c in ("SJ", "BJ") else 4) for c, n in counts.items()):
        raise ValueError("叫抢手牌超过一副牌的重复上限")
    return [float(counts[rank] > row) for row in range(4) for rank in RANKS]


def verify_bid_model(folder):
    """逐次启动核对固定来源及文件哈希；存在文件但不兼容同样拒绝，不下载或随机初始化。"""
    folder = Path(folder)
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf8"))
    if manifest.get("source_commit") != BID_COMMIT or manifest.get("sha256") != BID_SHA256:
        raise ValueError("叫抢模型来源清单不匹配")
    if hashlib.sha256((folder / "bid_weights.pkl").read_bytes()).hexdigest() != BID_SHA256:
        raise ValueError("叫抢模型文件哈希校验失败")
    return manifest


def setup_bidding(root):
    """准备固定叫抢权重、来源和许可证，再在CPU子进程严格检查参数及17张牌推理。

    仅在明确执行安装命令时下载；运行中不会下载权重。上游部分代码有未加载网络的
    predict路径，本工具只移植与已发布 bid_weights.pkl 匹配的 Net2/predict_score。
    """
    folder = Path(root) / "models/bidding"
    folder.mkdir(parents=True, exist_ok=True)
    base = f"https://raw.githubusercontent.com/Vincentzyx/DouZero_For_HLDDZ_FullAuto/{BID_COMMIT}/"
    target = folder / "bid_weights.pkl"
    if not target.is_file() or hashlib.sha256(target.read_bytes()).hexdigest() != BID_SHA256:
        with urllib.request.urlopen(base + "weights/bid_weights.pkl", timeout=60) as response:
            data = response.read()
        if hashlib.sha256(data).hexdigest() != BID_SHA256:
            raise ValueError("下载的叫抢权重哈希不匹配")
        temporary = target.with_suffix(".tmp")
        temporary.write_bytes(data)
        temporary.replace(target)
    vendor = PACKAGE_ROOT / "_vendor/bidding"
    shutil.copy2(vendor / "LICENSE", folder / "LICENSE")
    manifest = {
        "model_id": BID_MODEL_ID,
        "source_commit": BID_COMMIT,
        "repository": BID_REPOSITORY,
        "url": base + "weights/bid_weights.pkl",
        "sha256": BID_SHA256,
        "bytes": target.stat().st_size,
        "architecture": "BidModel.Net2",
        "input": "自己的17张牌，4×15计数矩阵按行展开",
        "output": "未经王者校准的原始叫抢评分，不是实际胜率",
        "license": "Apache-2.0",
    }
    (folder / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf8")
    agent = BidAgent(folder)
    try:
        agent.start()
        samples = [
            list("3334445566789JQKA"),
            ["3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A", "2", "2", "2", "SJ", "BJ"],
        ]
        checks = [{"hand": h, "score": agent.score(h)[0]} for h in samples]
        verification = {
            "passed": True,
            "strict_state_dict": True,
            "weights_only": True,
            "device": "cpu",
            "samples": checks,
            **agent.ready_info,
        }
        (folder / "verification.json").write_text(
            json.dumps(verification, ensure_ascii=False, indent=2), encoding="utf8"
        )
    finally:
        agent.close()
    return manifest | {"verification": verification}


class BidAgent(Agent):
    """复用已有子进程收发与关闭机制；只向独立叫抢进程发送17张手牌，单次限时1秒。"""

    def start(self):
        with self.lock:
            if self.process is not None and self.process.poll() is None:
                return
            verify_bid_model(self.folder)
            self.model_id = BID_MODEL_ID
            self.responses = Queue()
            error = (self.folder / "推理错误.log").open("a", encoding="utf8")
            try:
                self.process = subprocess.Popen(
                    [
                        sys.executable,
                        "-u",
                        "-X",
                        "utf8",
                        "-m",
                        "doudizhu_lab.agents.bid_worker",
                        str(self.folder),
                    ],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=error,
                    encoding="utf8",
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                )
            finally:
                error.close()
            threading.Thread(target=self._read, args=(self.process, self.responses), daemon=True).start()
            result = self._receive(30)
            if not result.get("ready"):
                self.close()
                raise ValueError("叫抢模型未就绪")
            self.ready_info = result

    def score(self, hand):
        """返回原始评分与耗时；不接受底牌、对手分配或当前游戏对象。"""
        encode_hand(hand)
        with self.lock:
            self.check_running()
            started = perf_counter()
            self.process.stdin.write(json.dumps({"hand": list(hand)}) + "\n")
            self.process.stdin.flush()
            output = self._receive(1)
            value = output["score"]
            if not isinstance(value, (int, float)) or not float("-inf") < value < float("inf"):
                raise ValueError("叫抢评分无效")
            return value, (perf_counter() - started) * 1000


@dataclass(frozen=True)
class BidDecision:
    """绑定发牌尝试与叫抢机会的意图；评分、阈值、回退原因和来源均进入独立流程账本。"""

    opportunity: str
    kind: str
    stage: str
    hand: tuple[str, ...]
    score: float | None
    threshold: float
    model_id: str
    elapsed_ms: float = 0
    reason: str = ""
    evidence_frames: tuple[str, ...] = ()


class BiddingController:
    """有界异步叫抢评估；同一机会只锁定一次意图，迟到评分不能改变已选择的动作。"""

    def __init__(self, settings, agent=None, *, threaded=True):
        self.settings, self.agent, self.threaded = dict(settings), agent, threaded
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="叫抢模型") if threaded else None
        self.preparation = self.pool.submit(agent.start) if self.pool and agent else None
        self.error = ""
        if agent and not threaded:
            try:
                agent.start()
            except Exception as exc:
                self.error = str(exc)
        self.candidates, self.decisions, self.requests = {}, {}, {}
        self.positive_deals = set()

    def choose(self, flow, deal):
        """只读取当前同帧手牌及可见按钮；两帧手牌一致后推理，等待超过3秒则拒绝叫抢。"""
        buttons = {b.kind: b for b in flow.buttons if b.enabled and b.kind in BID_KINDS}
        negative = buttons.get("decline_bid")
        if negative is None:
            return None
        stage = "call" if negative.label in ("不叫", "不叫地主") else "rob"
        if stage == "rob" and deal in self.positive_deals:
            stage = "reclaim"
        key = f"{deal}:bid:{'call' if stage == 'call' else 'rob'}"
        if key in self.decisions:
            decision = self.decisions[key]
            if decision.kind != "decline_bid" and Counter(flow.bid_hand or ()) != Counter(decision.hand):
                # 指令尚未发送时手牌变化使肯定意图失效；回退拒绝，禁止使用旧牌评分点击。
                decision = replace(
                    decision,
                    kind="decline_bid",
                    score=None,
                    reason="点击前手牌发生变化，回退拒绝",
                    evidence_frames=(),
                )
                self.decisions[key] = decision
            return decision
        threshold = self.settings[f"{stage}_threshold"]
        hand = tuple(flow.bid_hand or ())
        item = self.candidates.setdefault(key, {"first": flow.frame.media_pts_us, "hand": None, "frames": []})
        reason, score, elapsed = "", None, 0
        if self.settings["mode"] == "disabled":
            reason = "已选择固定不叫不抢"
        elif self.preparation and not self.preparation.done():
            reason = "叫抢模型仍在预热"
        else:
            if self.preparation:
                try:
                    self.preparation.result()
                except Exception as exc:
                    self.error = str(exc)
                self.preparation = None
            if self.error or self.agent is None:
                reason = f"叫抢模型不可用：{self.error or '未加载'}"
            else:
                try:
                    encode_hand(hand)
                except ValueError:
                    reason = "当前手牌未完整确认"
                else:
                    if item["hand"] != hand:
                        item.update(hand=hand, frames=[(flow.frame.frame_id, flow.frame.media_pts_us)])
                    elif item["frames"][-1][0] != flow.frame.frame_id:
                        item["frames"].append((flow.frame.frame_id, flow.frame.media_pts_us))
                    frames = item["frames"]
                    if frames[-1][1] - frames[0][1] < 100_000:
                        reason = "等待两帧一致的17张手牌"
                    else:
                        request = self.requests.get(key)
                        if request and request[0] != hand:
                            reason = "评估期间手牌发生变化"
                        elif request is None:
                            if self.pool:
                                self.requests[key] = (hand, self.pool.submit(self.agent.score, hand))
                                reason = "正在评估叫抢手牌"
                            else:
                                try:
                                    score, elapsed = self.agent.score(hand)
                                except Exception as exc:
                                    reason = f"叫抢模型失败：{exc}"
                        elif not request[1].done():
                            reason = "正在评估叫抢手牌"
                        else:
                            try:
                                score, elapsed = request[1].result()
                            except Exception as exc:
                                reason = f"叫抢模型失败：{exc}"
        if (
            reason
            and self.settings["mode"] != "disabled"
            and not (self.error or "失败" in reason or "不可用" in reason)
        ):
            if flow.frame.media_pts_us - item["first"] < 3_000_000:
                return None
            reason += "，已超过3秒等待，回退拒绝"
        positive = "call_landlord" if stage == "call" else "rob_landlord"
        kind = positive if score is not None and score > threshold and positive in buttons else "decline_bid"
        if score is not None and score > threshold and positive not in buttons:
            reason = "模型倾向叫抢，但当前肯定按钮未确认，回退拒绝"
        result = BidDecision(
            key,
            kind,
            stage,
            hand,
            score,
            threshold,
            getattr(self.agent, "model_id", ""),
            elapsed,
            reason or "原始评分与当前阶段阈值比较",
            tuple(f[0] for f in item["frames"][:1] + item["frames"][-1:]),
        )
        self.decisions[key] = result
        return result

    def committed(self, decision, deal):
        """仅在页面命令取得反馈后记录本人已叫抢，用于之后回抢阈值；不修改出牌历史。"""
        if decision.kind in ("call_landlord", "rob_landlord"):
            self.positive_deals.add(deal)

    def close(self):
        """先打断本实例的模型进程，再等待唯一工作线程退出；不会触碰出牌WP进程。"""
        if self.agent:
            self.agent.interrupt()
        if self.pool:
            self.pool.shutdown(wait=True, cancel_futures=True)
        if self.agent:
            self.agent.close()


def decision_dict(decision):
    """将不可变意图导出为JSON字典，不将它写成已经发生的叫抢事实。"""
    return asdict(decision)
