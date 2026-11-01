"""固定 PaddleOCR 与可选手牌混合识别；所有分类器只读像素，不发送输入。"""

from __future__ import annotations

import re
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING

import cv2
import numpy as np

from doudizhu_lab.domain.models import (
    Recognition,
)
from doudizhu_lab.vision.assets import model_path
from doudizhu_lab.vision.cards import Recognizer, glyph_color, ink_mask
from doudizhu_lab.vision.table import TableRecognizer

if TYPE_CHECKING:
    from rapidocr import RapidOCR


def parse_ranks(text: str) -> list[str] | None:
    """严格解析神经网络识别出的牌点串；保留重复，拒绝把O/I/B等相似字母猜成数字。"""
    text = re.sub(r"\s+", "", text).upper()
    values = re.findall(r"10|[3-9JQKA2]", text)
    return values if values and "".join(values) == text else None


def make_ocr(root: Path) -> RapidOCR:
    """固定 PP-OCRv5 中英模型与本地路径，禁用方向分类，不触发默认v6模型下载。"""
    try:
        from rapidocr import EngineType, LangDet, LangRec, ModelType, OCRVersion, RapidOCR
    except ImportError as exc:
        raise ValueError("请先运行 uv sync --frozen --extra vision 安装OCR推理依赖") from exc

    return RapidOCR(
        params={
            "Global.use_cls": False,
            "Global.text_score": 0.5,
            "Global.log_level": "error",
            "Global.model_root_dir": str(root / "models/pretrained-vision/ppocr-v5"),
            "Det.engine_type": EngineType.ONNXRUNTIME,
            "Rec.engine_type": EngineType.ONNXRUNTIME,
            "Det.ocr_version": OCRVersion.PPOCRV5,
            "Rec.ocr_version": OCRVersion.PPOCRV5,
            "Det.model_type": ModelType.MOBILE,
            "Rec.model_type": ModelType.MOBILE,
            "Det.lang_type": LangDet.CH,
            "Rec.lang_type": LangRec.CH,
            "Det.model_path": str(model_path(root, "ocr_det")),
            "Rec.model_path": str(model_path(root, "ocr_zh")),
            "EngineConfig.onnxruntime.intra_op_num_threads": 2,
            "EngineConfig.onnxruntime.inter_op_num_threads": 1,
            "Det.limit_side_len": 736,
            "Det.limit_type": "max",
        }
    )


class OcrGlyphs(Recognizer):
    """对照实验：保留传统候选分割，只用预训练英文识别替换模板分类；不是端到端神经检测。"""

    def __init__(self, root):
        # 不加载牌点模板，只复用几何分割；目录无需存在，也不会创建。
        super().__init__(root / "models/pretrained-vision/no-card-templates", include_bundled=False)
        self.engine = make_ocr(root)
        self.engine.cfg.Rec.model_path = str(model_path(root, "ocr_en"))
        from rapidocr import LangRec

        self.engine.cfg.Rec.lang_type = LangRec.EN

    def classifier_ready(self, group):
        """分类器依赖已校验的预训练模型，不依赖本机是否采集过牌点模板。"""
        return True

    def classify(self, crop, group, profile):
        """仅输入裁剪像素；分数不足或多牌字符不作为单张牌。大小王竖排暂时保持拒识。"""
        image = cv2.copyMakeBorder(crop, 8, 8, 8, 8, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        result = self.engine(cv2.cvtColor(image, cv2.COLOR_RGB2BGR), use_det=False, use_cls=False)
        if not result.txts:
            return None, 0, "预训练OCR未读出字符"
        text, score = result.txts[0], float(result.scores[0])
        ranks = parse_ranks(text)
        if score >= 0.5 and ranks and len(ranks) == 1:
            return ranks[0], score, "预训练OCR字形分类"
        return None, score, f"预训练OCR无法确认：{text}"


class OcrHybrid(OcrGlyphs):
    """普通牌用英文识别，竖排JOKER用完整OCR检测识别，再核对颜色；无需训练。"""

    def __init__(self, root):
        super().__init__(root)
        self.joker_engine = make_ocr(root)

    def classify(self, crop, group, profile):
        """对至少三段墨迹的竖排候选确认JOKER；不将单独J或花色猜成大小王。"""
        rows = (ink_mask(crop) > 0).any(axis=1)
        starts = int(np.count_nonzero(rows & ~np.r_[False, rows[:-1]]))
        if crop.shape[0] > crop.shape[1] * 1.8 and starts >= 3:
            padded = cv2.copyMakeBorder(crop, 8, 8, 8, 8, cv2.BORDER_CONSTANT, value=(255, 255, 255))
            result = self.joker_engine(cv2.cvtColor(padded, cv2.COLOR_RGB2BGR), use_cls=False)
            text = "".join(result.txts or ()).replace(" ", "").upper()
            score = min(result.scores) if result.scores else 0.0
            color = glyph_color(crop)
            if text == "JOKER" and score >= 0.75 and color in ("red", "black"):
                return "BJ" if color == "red" else "SJ", float(score), "预训练OCR确认JOKER字形及颜色"
            return None, float(score), f"竖排字形无法确认：{text}"
        return super().classify(crop, group, profile)


class MixedCards:
    """手牌字形使用免训练OCR，其余区域使用现有模板；不以模板偷偷修正OCR手牌。

    定位仍沿用传统候选分割，因而并非端到端神经网络布局检测。两个子识别器只处理
    各自区域，保留原图坐标、拒识及稳定性检查；分数含义由各张牌的 reason 标注。
    """

    def __init__(self, root, template_dir, *, include_bundled=True):
        self.hand = OcrHybrid(root)
        self.other = Recognizer(template_dir, include_bundled=include_bundled)

    @property
    def templates(self):
        """供模板来源隔离评测使用；手牌OCR不会访问此模板列表。"""
        return self.other.templates

    @property
    def load_errors(self):
        """暴露模板加载错误，保持桌面重新载入流程兼容；不把错误吞成空列表。"""
        return self.other.load_errors

    def reload(self):
        """只重新载入小牌模板，固定模型须重新启动后校验，不在录制期间更换。"""
        self.other.reload()

    def coverage(self):
        """返回真实模板覆盖；手牌模型能力由界面单独说明，不伪装成15个采集模板。"""
        return self.other.coverage()

    def recognize(self, frame, profile, *, stabilize=False, table_confirmed=False):
        """只按明确配置合并不同区域，保留未知项；不在两种输出中挑选更像答案的一方。"""
        start = perf_counter()
        profile = TableRecognizer._bottom_layout(frame, profile)
        hand = self.hand.recognize(
            frame, profile, regions={"hand"}, stabilize=stabilize, table_confirmed=table_confirmed
        )
        others = self.other.recognize(
            frame,
            profile,
            regions=set(profile.regions) - {"hand"},
            stabilize=stabilize,
            table_confirmed=table_confirmed,
        )
        by_region = {r.region: r for r in hand.regions + others.regions}
        return Recognition(
            [by_region[k] for k in profile.regions], frame.captured_at, (perf_counter() - start) * 1000
        )


def build_backend(name: str, root: Path) -> Recognizer | MixedCards:
    """按显式名称创建离线识别器；默认实时识别器不受本模块安装和评测影响。"""
    if name == "hand-ocr":
        return MixedCards(root, root / "templates")
    if name == "template":
        return Recognizer(root / "templates")
    raise ValueError("未知免训练识别器")
