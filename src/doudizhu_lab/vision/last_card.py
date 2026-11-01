"""复用预训练OCR补充完整短提示与末手字形；所有结果仍需独立跨帧确认。"""

import cv2
import numpy as np

from doudizhu_lab.domain.models import CardObservation, PixelRect, RegionStatus
from doudizhu_lab.vision.cards import glyph_color, ink_mask
from doudizhu_lab.vision.ocr import make_ocr


class LastCardOCR:
    """识别末手上方可见的竖排JOKER/JOKE；不从手牌守恒或未知对手牌反推牌点。

    JOKE的四个字母已唯一指向王类，普通J及JO/JOK均拒识。还须有竖排比例、白牌背景、
    严格颜色和明确末手标记；不扩大到普通阶段，也不把得分解释成实际正确概率。
    同一工作线程的已预热模型还供PassTextOCR读取完整不出提示，不新增模型或训练。
    """

    def __init__(self, root):
        self.engine = make_ocr(root)
        # 在大厅阶段完成检测和识别模型的冷启动；这只是固定文字推理，不进行训练。
        sample = np.full((96, 280, 3), 255, np.uint8)
        cv2.putText(sample, "JOKER", (8, 68), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 0, 0), 3)
        self.engine(sample, use_det=True, use_cls=False)

    def enrich(self, frame, observation, recognition, profile):
        """补同帧完整短提示与可见王类，不写事件；跨帧、剩牌差分和轮转仍由Tracker确认。"""
        from doudizhu_lab.vision.action_text import PassTextOCR

        PassTextOCR(engine=self.engine).enrich(frame, observation, profile)
        for seat, key in (("self", "self_play"), ("left", "left"), ("right", "right")):
            evidence = observation.marker_evidence.get("winner_" + seat, {})
            if (
                not evidence.get("accepted")
                or observation.remaining.get(seat) != 0
                or not evidence.get("box")
            ):
                continue
            region = next(r for r in recognition.regions if r.region == key)
            if region.status == RegionStatus.OK:
                continue
            x, y, w, _ = evidence["box"]
            area = profile.regions[key].pixels(frame.width, frame.height)
            pad = round(20 * frame.height / profile.base_height)
            x0, x1 = max(area.x, x - pad), min(area.x + area.width, x + w + pad)
            y0, y1 = area.y, min(area.y + area.height, y - 2)
            if x1 <= x0 or y1 - y0 < 50 * frame.height / profile.base_height:
                continue
            crop = frame.rgb[y0:y1, x0:x1]
            result = self.engine(cv2.cvtColor(crop, cv2.COLOR_RGB2BGR), use_det=True, use_cls=False)
            found = []
            if result.boxes is None:
                continue
            for polygon, text, score in zip(result.boxes, result.txts, result.scores):
                if text.replace(" ", "").upper() not in ("JOKER", "JOKE") or score < 0.95:
                    continue
                left, top = polygon.min(axis=0)
                right, bottom = polygon.max(axis=0)
                if bottom - top < (right - left) * 2.2:
                    continue
                box = PixelRect(
                    x0 + int(left), y0 + int(top), max(1, int(right - left)), max(1, int(bottom - top))
                )
                glyph = box.crop(frame.rgb)
                white = np.min(glyph, axis=2) > 160
                if np.mean(white) < 0.35 or np.mean(ink_mask(glyph) > 0) < 0.05:
                    continue
                color = glyph_color(glyph)
                found.append(
                    CardObservation(
                        "BJ" if color == "red" else "SJ",
                        box,
                        float(score),
                        "末手标记上方竖排" + text + "及红黑颜色；原文至少四字母",
                    )
                )
            if len(found) == 1:
                region.status, region.cards = RegionStatus.OK, found
                region.message = "末手OCR专项识别，尚须跨帧和轮次确认"
                observation.plays[seat] = [found[0].rank]
                observation.marker_evidence["last_card_" + seat] = {
                    "accepted": True,
                    "frame_id": frame.frame_id,
                    "pts_us": frame.media_pts_us,
                    "rank": found[0].rank,
                    "score": found[0].score,
                    "reason": found[0].reason,
                }
        from dataclasses import asdict

        observation.regions = [asdict(r) for r in recognition.regions]
        return observation, recognition
