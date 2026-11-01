"""叫抢模型独立进程；标准输出仅用于结构化协议，不持有 ADB、Qt 或对手手牌。"""

import json
import sys
from pathlib import Path

import torch

from doudizhu_lab.agents.bid_network import BidNetwork
from doudizhu_lab.agents.bidding import encode_hand, verify_bid_model


def main():
    """严格加载已校验权重并预热；错误退出，禁止用未加载的网络产生评分。"""
    try:
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        folder = Path(sys.argv[1])
        verify_bid_model(folder)
        network = BidNetwork()
        weights = torch.load(folder / "bid_weights.pkl", map_location="cpu", weights_only=True)
        network.load_state_dict(weights, strict=True)
        network.eval()
        with torch.inference_mode():
            network(torch.zeros((1, 60)))
        print(
            json.dumps({"ready": True, "parameter_shapes": {k: list(v.shape) for k, v in weights.items()}}),
            flush=True,
        )
        for line in sys.stdin:
            data = json.loads(line)
            # 只接受自己的17张手牌；其他牌、底牌、牌局对象不能跨过进程边界。
            if set(data) != {"hand"}:
                raise ValueError("叫抢模型输入包含未支持字段")
            features = torch.tensor([encode_hand(data["hand"])], dtype=torch.float32)
            with torch.inference_mode():
                score = network(features).item()
            if not torch.isfinite(torch.tensor(score)):
                raise ValueError("叫抢评分不是有限数值")
            print(json.dumps({"score": score}), flush=True)
    except Exception as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False), flush=True)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
