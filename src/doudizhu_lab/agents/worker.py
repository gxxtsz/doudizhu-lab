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


def main():
    """严格加载所有参数并预热三种身份；异常输出结构化错误，不回退随机网络。"""
    try:
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        folder = Path(sys.argv[1])
        verify_models(folder)
        models = {}
        for role in ROLES:
            model = model_dict[role]()
            weights = torch.load(folder / f"{role}.ckpt", map_location="cpu", weights_only=True)
            model.load_state_dict(weights, strict=True)
            model.eval()
            with torch.inference_mode():
                model(
                    torch.zeros((1, 5, 162)),
                    torch.zeros((1, 373 if role == "landlord" else 484)),
                    return_value=True,
                )
            models[role] = model
        print(
            json.dumps(
                {
                    "ready": True,
                    "torch_version": torch.__version__,
                    "device": "cpu",
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
            obs = get_obs(SimpleNamespace(**data))
            with torch.inference_mode():
                values = (
                    models[data["player_position"]](
                        torch.from_numpy(obs["z_batch"]), torch.from_numpy(obs["x_batch"]), return_value=True
                    )["values"]
                    .numpy()
                    .reshape(-1)
                )
            if not np.isfinite(values).all():
                raise ValueError("模型输出含非有限数值")
            best = int(values.argmax())
            print(
                json.dumps({"cards": data["legal_actions"][best], "value": float(values[best])}), flush=True
            )
    except Exception as exc:
        print(json.dumps({"error": f"模型加载或推理失败：{exc}"}, ensure_ascii=False), flush=True)
        raise


if __name__ == "__main__":
    main()
