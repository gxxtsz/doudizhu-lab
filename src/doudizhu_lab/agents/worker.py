"""仅在独立进程内加载 PyTorch；标准输出专用于逐行 JSON 协议。"""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from doudizhu_lab._vendor.douzero.dmc.models import model_dict
from doudizhu_lab._vendor.douzero.env.env import get_obs
from doudizhu_lab.agents.douzero import ROLES, verify_models


def score_actions(
    model: torch.nn.Module, infoset: SimpleNamespace, backend: str = "wp", *, batch_size: int = 256
) -> np.ndarray:
    """按连续候选分片计算原始评分，返回顺序与 legal_actions 完全一致。

    分片只限制峰值内存，不改变局面或历史；网络必须先进入 eval 模式。严格检查每片
    N×1 输出和有限性，调用方使用首次出现的最大值，保留上游候选平分时的选择顺序。
    """
    if batch_size <= 0 or not infoset.legal_actions:
        raise ValueError("模型需要非空候选及正数分片大小")
    values = []
    with torch.inference_mode():
        for offset in range(0, len(infoset.legal_actions), batch_size):
            actions = infoset.legal_actions[offset : offset + batch_size]
            if backend == "resnet":
                from doudizhu_lab.agents.resnet_features import get_obs as resnet_obs

                obs = resnet_obs(infoset, actions)
            elif backend == "wp":
                obs = get_obs(SimpleNamespace(**(vars(infoset) | {"legal_actions": actions})))
            else:
                raise ValueError("未知出牌模型")
            output = model(
                torch.from_numpy(obs["z_batch"]), torch.from_numpy(obs["x_batch"]), return_value=True
            )["values"].numpy()
            if output.shape != (len(actions), 1) or not np.isfinite(output).all():
                raise ValueError("模型评分维度错误或含非有限数值")
            values.append(output[:, 0])
    return np.concatenate(values)


def main() -> None:
    """严格加载所有参数并预热三种身份；异常输出结构化错误，不回退随机网络。"""
    try:
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        folder = Path(sys.argv[1])
        backend = sys.argv[2] if len(sys.argv) > 2 else "wp"
        verify_models(folder, backend)
        if backend == "resnet":
            from doudizhu_lab._vendor.douzero_resnet.models import ResnetModel

            constructors = dict.fromkeys(ROLES, ResnetModel)
        else:
            constructors = model_dict
        models = {}
        for role in ROLES:
            model = constructors[role]()
            weights = torch.load(folder / f"{role}.ckpt", map_location="cpu", weights_only=True)
            model.load_state_dict(weights, strict=True)
            model.eval()
            with torch.inference_mode():
                model(
                    torch.zeros((1, 40, 54) if backend == "resnet" else (1, 5, 162)),
                    torch.zeros((1, 15 if backend == "resnet" else 373 if role == "landlord" else 484)),
                    return_value=True,
                )
            models[role] = model
        print(
            json.dumps(
                {
                    "ready": True,
                    "torch_version": torch.__version__,
                    "device": "cpu",
                    "backend": backend,
                    "parameter_shapes": {
                        role: {name: list(value.shape) for name, value in model.state_dict().items()}
                        for role, model in models.items()
                    },
                }
            ),
            flush=True,
        )
        for line in sys.stdin:
            data = json.loads(line)
            values = score_actions(models[data["player_position"]], SimpleNamespace(**data), backend)
            best = int(values.argmax())
            print(
                json.dumps({"cards": data["legal_actions"][best], "value": float(values[best])}), flush=True
            )
    except Exception as exc:
        print(json.dumps({"error": f"模型加载或推理失败：{exc}"}, ensure_ascii=False), flush=True)
        raise


if __name__ == "__main__":
    main()
