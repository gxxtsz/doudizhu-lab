"""出牌模型选择、固定 ResNet 资产准备及验证；导入本模块不会加载 PyTorch。"""

from __future__ import annotations

import hashlib
import json
import shutil
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

from doudizhu_lab.domain.settings import PLAY_MODEL_NAMES
from doudizhu_lab.paths import PACKAGE_ROOT

if TYPE_CHECKING:
    from doudizhu_lab.agents.douzero import Agent

RESNET_COMMIT = "eaf0436676c199507fe5420e464daa951a83d704"
RESNET_REPOSITORY = "https://github.com/godmoves/perfectdou"
RESNET_WEIGHTS = {
    "landlord": (14199345, "df11501facc740b10bb5f8ea32e7476b4aa32af9eff4371ffb6c27aef85009ef"),
    "landlord_down": (14199345, "ef6b9afc45c4286f59edc447c161162104257b03ad33eadce91496c5f0810a01"),
    "landlord_up": (14200817, "a4b502fa1a7a04b2cce34c404e7f6f5c5c6b54e7865c1129b136e1e5aa28e72c"),
}
RESNET_SCHEMA = "public-resnet-40x54-compat15-v1"


def model_folder(root: Path, backend: str) -> Path:
    """返回所选模型的数据目录；未知后端直接拒绝，不猜测文件名或自动降级。"""
    if backend not in PLAY_MODEL_NAMES:
        raise ValueError("出牌模型必须为 resnet 或 wp")
    return Path(root) / f"models/douzero-{backend}"


def create_agent(root: Path, backend: str = "resnet") -> Agent:
    """按明确偏好创建尚未启动的 CPU 代理；不读设备、不联网、不授予输入权限。"""
    from doudizhu_lab.agents.douzero import Agent

    return Agent(model_folder(root, backend), backend=backend)


def verify_resnet_models(folder: Path) -> dict:
    """核验固定来源、编码版本及三角色权重；清单自身不能覆盖程序固定的校验值。"""
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf8"))
    if (
        manifest.get("backend") != "resnet"
        or manifest.get("source_commit") != RESNET_COMMIT
        or manifest.get("feature_schema") != RESNET_SCHEMA
        or set(manifest.get("weights", {})) != set(RESNET_WEIGHTS)
    ):
        raise ValueError("ResNet 模型来源或特征版本不匹配")
    for role, (size, digest) in RESNET_WEIGHTS.items():
        entry = manifest["weights"][role]
        target = folder / f"{role}.ckpt"
        if entry.get("sha256") != digest or entry.get("bytes") != size:
            raise ValueError(f"ResNet 固定权重清单不匹配：{role}")
        if target.stat().st_size != size or hashlib.sha256(target.read_bytes()).hexdigest() != digest:
            raise ValueError(f"ResNet 权重文件校验失败：{role}")
    return manifest


def setup_play_model(root: Path, backend: str = "resnet") -> dict:
    """显式准备指定模型；ResNet 在隔离目录下载、严格预热后才发布正式文件。

    默认 ResNet，wp 分支保留旧安装入口及原权重。固定地址、大小与哈希同时校验，
    已有正确文件只复制复用；任何下载或预热失败均不发布新清单，不接触真实设备。
    """
    from doudizhu_lab.agents.douzero import Agent, setup_agent

    folder = model_folder(root, backend)
    if backend == "wp":
        return setup_agent(Path(root))
    folder.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="resnet-prepare-", dir=folder.parent) as temporary:
        staging = Path(temporary)

        def fetch(role):
            """只下载或复用一份固定权重，校验后写入本次隔离准备目录。"""
            size, digest = RESNET_WEIGHTS[role]
            url = (
                f"https://raw.githubusercontent.com/godmoves/perfectdou/{RESNET_COMMIT}/"
                f"perfectdou/model/douzero/resnet/resnet_{role}.ckpt"
            )
            cached = folder / f"{role}.ckpt"
            data = cached.read_bytes() if cached.is_file() else b""
            if len(data) != size or hashlib.sha256(data).hexdigest() != digest:
                with urllib.request.urlopen(url, timeout=60) as response:
                    data = response.read(size + 1)
            if len(data) != size or hashlib.sha256(data).hexdigest() != digest:
                raise ValueError(f"ResNet 下载内容与固定权重不同：{role}")
            (staging / f"{role}.ckpt").write_bytes(data)
            return role, {"url": url, "bytes": size, "sha256": digest}

        with ThreadPoolExecutor(max_workers=3) as pool:
            weights = dict(pool.map(fetch, RESNET_WEIGHTS))
        manifest = {
            "backend": "resnet",
            "source_commit": RESNET_COMMIT,
            "repository": RESNET_REPOSITORY,
            "feature_schema": RESNET_SCHEMA,
            "origin": "公开评测仓库中的社区 ResNet 副本，兼容特征固定，不代表真实叫抢观察",
            "weights": weights,
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf8"
        )
        shutil.copy2(PACKAGE_ROOT / "_vendor/douzero_resnet/LICENSE", staging / "LICENSE-社区仓库.txt")
        agent = Agent(staging, backend="resnet")
        try:
            agent.start()
            verification = agent.ready_info | {
                "model_id": agent.model_id,
                "source_commit": RESNET_COMMIT,
                "strict_state_dict": True,
                "weights_only": True,
            }
        finally:
            agent.close()
        (staging / "verification.json").write_text(
            json.dumps(verification, ensure_ascii=False, indent=2), encoding="utf8"
        )
        folder.mkdir(exist_ok=True)
        # 清单最后发布；各角色文件始终来自同一固定版本，不发布未预热的安装结果。
        for name in [
            *(f"{r}.ckpt" for r in RESNET_WEIGHTS),
            "LICENSE-社区仓库.txt",
            "verification.json",
            "manifest.json",
        ]:
            (staging / name).replace(folder / name)
    return manifest | {"verified_roles": list(RESNET_WEIGHTS), "verification": verification}
