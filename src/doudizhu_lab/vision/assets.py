"""免训练模型的固定来源和本地校验；推理只读取图片，不拥有手机输入能力。"""

from __future__ import annotations

import hashlib
import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from doudizhu_lab.paths import PACKAGE_ROOT

ASSETS = {
    "ocr_det": {
        "file": "ppocr-v5/ch_PP-OCRv5_det_mobile.onnx",
        "sha256": "4d97c44a20d30a81aad087d6a396b08f786c4635742afc391f6621f5c6ae78ae",
    },
    "ocr_en": {
        "file": "ppocr-v5/en_PP-OCRv5_rec_mobile.onnx",
        "sha256": "c3461add59bb4323ecba96a492ab75e06dda42467c9e3d0c18db5d1d21924be8",
    },
    "ocr_zh": {
        "file": "ppocr-v5/ch_PP-OCRv5_rec_mobile.onnx",
        "sha256": "5825fc7ebf84ae7a412be049820b4d86d77620f204a041697b0494669b1742c5",
    },
}
for key, value in ASSETS.items():
    if key.startswith("ocr_"):
        task = "det" if key == "ocr_det" else "rec"
        value.update(
            url=f"https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.10.0/onnx/PP-OCRv5/{task}/{Path(value['file']).name}",
            source="PaddleOCR PP-OCRv5；RapidOCR 3.10.0 发布的 ONNX 转换副本",
            license="Apache-2.0",
        )


def model_path(root: Path, key: str) -> Path:
    """检查文件内容后返回本地路径；推理时绝不自动联网下载或使用损坏权重。"""
    value = ASSETS[key]
    path = root / "models/pretrained-vision" / value["file"]
    if not path.is_file():
        raise ValueError("缺少免训练模型，请先运行 uv run --extra vision ddz-check setup-vision")
    if hashlib.sha256(path.read_bytes()).hexdigest() != value["sha256"]:
        raise ValueError(f"免训练模型校验失败：{path.name}")
    return path


def setup_vision(root: Path) -> dict:
    """下载固定版本 ONNX 并核对项目固定哈希；只安装推理资产，不训练、不加载 pickle。"""
    try:
        import requests
    except ImportError as exc:
        raise ValueError("请先运行 uv sync --frozen --extra vision") from exc
    folder = root / "models/pretrained-vision"
    folder.mkdir(parents=True, exist_ok=True)

    def fetch(pair):
        key, asset = pair
        path = folder / asset["file"]
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != asset["sha256"]:
            error = None
            for _ in range(3):
                try:
                    with requests.get(asset["url"], timeout=(20, 90), stream=True) as response:
                        response.raise_for_status()
                        temporary = path.with_suffix(".download")
                        with temporary.open("wb") as output:
                            for block in response.iter_content(1024 * 1024):
                                output.write(block)
                    if hashlib.sha256(temporary.read_bytes()).hexdigest() != asset["sha256"]:
                        raise ValueError(f"下载内容校验失败：{key}")
                    temporary.replace(path)
                    error = None
                    break
                except (requests.RequestException, OSError, ValueError) as exc:
                    error = exc
            if error:
                raise ValueError(f"模型下载失败：{key}：{error}") from error
        return key, dict(asset, bytes=path.stat().st_size)

    with ThreadPoolExecutor(max_workers=4) as pool:
        entries = dict(pool.map(fetch, ASSETS.items()))
    notices = PACKAGE_ROOT / "_vendor/pretrained_vision"
    shutil.copytree(notices, folder / "licenses", dirs_exist_ok=True)
    manifest = {"trained_in_this_project": False, "assets": entries}
    (folder / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf8")
    return manifest
