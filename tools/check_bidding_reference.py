"""复现500组手牌的上游特征和评分一致性；只加载固定权重，不连接手机。"""

import ast
import hashlib
import json
from pathlib import Path
from random import Random

import torch
from torch import nn
from torch.nn import functional as F

from doudizhu_lab.agents.bid_network import BidNetwork
from doudizhu_lab.agents.bidding import encode_hand, verify_bid_model
from doudizhu_lab.agents.douzero import DECK, TO_RANK
from doudizhu_lab.storage.config import Store


def main():
    """只执行哈希已核实源文件的Net2和编码函数；不执行其全局初始化或设备代码。"""
    root = Path(__file__).resolve().parents[1]
    source = (root / "src/doudizhu_lab/_vendor/bidding/BidModel.original.txt").read_bytes()
    if (
        hashlib.sha256(source).hexdigest()
        != "906ce0bd498991263196a828663316f4e5e550db587639c25c274ee4418b8ef7"
    ):
        raise ValueError("上游参考代码哈希不匹配，拒绝执行")
    tree = ast.parse(source.decode("utf8"))
    selected = ast.Module(
        body=[
            n
            for n in tree.body
            if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in ("Net2", "RealToOnehot")
        ],
        type_ignores=[],
    )
    namespace = {"torch": torch, "nn": nn, "F": F}
    exec(compile(selected, "已校验的上游Net2", "exec"), namespace)
    torch.set_num_threads(2)
    folder = Store().root / "models/bidding"
    verify_bid_model(folder)
    weights = torch.load(folder / "bid_weights.pkl", weights_only=True, map_location="cpu")
    reference, actual = namespace["Net2"]().eval(), BidNetwork().eval()
    reference.load_state_dict(weights, strict=True)
    actual.load_state_dict(weights, strict=True)
    random = Random(1101)
    rows, encoding_error = [], 0
    symbols = {"10": "T", "SJ": "X", "BJ": "D"}
    for _ in range(500):
        cards = list(DECK.elements())
        random.shuffle(cards)
        hand = [TO_RANK[c] for c in cards[:17]]
        encoded = torch.tensor(encode_hand(hand))
        expected = namespace["RealToOnehot"]("".join(symbols.get(c, c) for c in hand)).flatten()
        encoding_error = max(encoding_error, float((expected - encoded).abs().max()))
        rows.append(encoded)
    features = torch.stack(rows)
    with torch.inference_mode():
        expected, observed = reference(features).flatten(), actual(features).flatten()
    difference = float((expected - observed).abs().max())
    report = {
        "source": "offline_upstream_parity",
        "hands": len(rows),
        "seed": 1101,
        "max_encoding_difference": encoding_error,
        "max_score_difference": difference,
        "passed": encoding_error == 0 and difference < 1e-6,
        "call_rate_at_0.2": float((observed > 0.2).float().mean()),
        "rob_rate_at_0.5": float((observed > 0.5).float().mean()),
        "reclaim_rate_at_0.6": float((observed > 0.6).float().mean()),
        "note": "随机手牌上的评分分布及上游计算一致性，不是实际胜率或叫抢收益评测",
    }
    target = Store().root / "bidding-evaluation/上游一致性核验.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
