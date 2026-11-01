"""按已冻结配置创建识别器；基础模式不导入任何可选神经网络运行库。"""

from pathlib import Path

from doudizhu_lab.storage.config import Profile
from doudizhu_lab.vision.cards import Recognizer


def create_recognizer(
    template_dir: Path,
    profile: Profile,
    *,
    model_root: Path,
    include_bundled: bool = True,
):
    """统一预览、录制和离线重算的分类器选择，默认兼容历史模板模式。

    model_root 是包含 models/pretrained-vision 的数据或会话目录；模型只从此目录
    加载并校验。显式选择OCR后缺少模型或依赖会报错，绝不静默切换方法。
    """
    if profile.hand_classifier == "template":
        return Recognizer(template_dir, include_bundled=include_bundled)
    if profile.hand_classifier == "pretrained-ocr":
        from doudizhu_lab.vision.ocr import MixedCards

        return MixedCards(model_root, template_dir, include_bundled=include_bundled)
    raise ValueError("不支持的手牌分类器配置")
