"""统一运行资料、程序资源和复现快照的路径，避免安装位置与设备路径耦合。"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
ENGINE_FILES = {
    "recognition.py": "vision/cards.py",
    "table.py": "vision/table.py",
    "tracker.py": "domain/tracker.py",
    "storage.py": "storage/config.py",
    "models.py": "domain/models.py",
    "last_card.py": "vision/last_card.py",
    "flow.py": "vision/flow.py",
    "bot_controls.py": "vision/controls.py",
    "pretrained.py": "vision/ocr.py",
    "pretrained_models.py": "vision/assets.py",
    "settings.py": "domain/settings.py",
    "observation.py": "domain/table.py",
}


def workspace_root() -> Path | None:
    """返回实际源码工作区；只接受本发行包的src布局，不把site-packages认作工程。

    只读访问，可由任意线程调用。不通过当前工作目录猜测资源位置，切换目录不影响结果。
    """
    root = PACKAGE_ROOT.parent.parent
    manifest = root / "pyproject.toml"
    if PACKAGE_ROOT.parent.name == "src" and manifest.is_file():
        if 'name = "doudizhu-lab"' in manifest.read_text(encoding="utf-8"):
            return root
    return None


def resolve_data_dir(explicit: Path | None = None) -> Path:
    """按参数、DDZ_DATA_DIR、源码工作区、用户目录依次定位可写数据目录。

    本函数不创建目录；创建与权限错误由Store处理。安装版默认位于Windows本地应用
    数据目录，其他平台使用XDG数据目录作为路径退路，不代表已完成跨平台真机验证。
    """
    if explicit is not None:
        return Path(explicit).expanduser().resolve()
    if configured := os.environ.get("DDZ_DATA_DIR"):
        return Path(configured).expanduser().resolve()
    if root := workspace_root():
        return root / "data"
    if os.name == "nt":
        local = Path(os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData/Local")))
        return local / "DoudizhuLab"
    return Path(os.environ.get("XDG_DATA_HOME", str(Path.home() / ".local/share"))) / "doudizhu-lab"


def project_files() -> dict[str, Path]:
    """提供快照所需工程元数据；wheel内使用构建时嵌入的同版副本。

    源码与安装版都必须具备完整文件，缺失时明确失败，不创建不完整的可复现快照。
    返回路径只供读取，不修改安装文件；可在录制后台调用。
    """
    root = workspace_root() or PACKAGE_ROOT / "_resources/project"
    names = ("pyproject.toml", "uv.lock", "README.md", "LICENSE", "NOTICE")
    paths = {name: root / name for name in names}
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"程序复现资源缺失：{name}")
    return paths


def engine_hashes() -> dict[str, str]:
    """计算当前引擎文件哈希，保留历史键名以比较既有会话，不执行快照代码。"""
    return {
        name: hashlib.sha256((PACKAGE_ROOT / rel).read_bytes()).hexdigest()
        for name, rel in ENGINE_FILES.items()
    }
