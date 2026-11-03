"""只读核查构建内容；防止本机数据、权重或缓存进入wheel与源码包。"""

from __future__ import annotations

import tarfile
import zipfile
from pathlib import Path


def check_archive(path: Path) -> None:
    """核查单个发行包的文件名与关键复现资源；异常以ValueError使CI失败。

    只读取压缩包，不解压、不执行其中代码。规则不以忽略文件代替检查构建结果。
    """
    if path.suffix == ".whl":
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
        required = ("pyproject.toml", "uv.lock", "README.md", "LICENSE", "NOTICE")
        for name in required:
            if f"doudizhu_lab/_resources/project/{name}" not in names:
                raise ValueError(f"wheel缺少复现资源：{name}")
    else:
        with tarfile.open(path, "r:gz") as archive:
            names = [m.name for m in archive.getmembers() if m.isfile()]
        names = [n.split("/", 1)[-1] for n in names]
    package_prefix = "doudizhu_lab/" if path.suffix == ".whl" else "src/doudizhu_lab/"
    for required in (
        "_vendor/douzero_resnet/LICENSE",
        "_vendor/douzero_resnet/provenance.json",
        "_vendor/douzero_resnet/models.py",
        "agents/resnet_features.py",
        "agents/backends.py",
        "domain/inference.py",
        "runtime/supervisor.py",
        "runtime/session_process.py",
        "runtime/process.py",
        "storage/task_media.py",
        "ui/unattended.py",
    ):
        if package_prefix + required not in names:
            raise ValueError(f"发行包缺少运行源码或来源声明：{required}")
    forbidden = {"data", ".venv", "__pycache__", ".idea", ".git", ".pytest_cache", ".ruff_cache"}
    suffixes = {".mkv", ".mp4", ".onnx", ".ckpt", ".pkl", ".pt", ".sqlite3", ".pyc", ".log"}
    for name in names:
        if forbidden.intersection(Path(name).parts) or Path(name).suffix in suffixes:
            raise ValueError(f"发行包包含本机资料或缓存：{name}")
        if "bot_replay_scope.json" in name or "yolo" in name.lower():
            raise ValueError(f"发行包包含已退役或私人范围文件：{name}")
    print(f"通过：{path.name}，{len(names)}个文件，无本机资料或权重")


def main() -> None:
    """检查当前工程dist中的wheel和源码包，两种包均须存在；不触碰用户数据。"""
    folder = Path(__file__).resolve().parents[1] / "dist"
    wheels, sources = list(folder.glob("*.whl")), list(folder.glob("*.tar.gz"))
    if not wheels or not sources:
        raise SystemExit("请先执行 uv build 生成wheel和源码包")
    for path in wheels + sources:
        check_archive(path)


if __name__ == "__main__":
    main()
