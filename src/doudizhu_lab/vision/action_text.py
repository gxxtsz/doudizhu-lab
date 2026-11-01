"""本地预训练OCR补充短暂不出提示；只读取完整金色文字，不从动作意图推测结果。"""

import cv2
import numpy as np

from doudizhu_lab.vision.table import MarkerLibrary


class PassTextOCR:
    """复用工作线程已预热的PP-OCRv5，仅在模板没有确认该座位时识别。

    engine由调用方注入并独占，不新建模型或下载资产。只接受完整“不出／不要／要不起”
    且得分至少0.98的文字；不抢、不加倍、单个“不”、数字和模糊输出一律拒绝。
    识别结果仍须跟踪器两帧相隔至少100ms及合法行动轮次确认，不能直接生成事件。
    """

    def __init__(self, *, engine):
        self.engine = engine

    def enrich(self, frame, observation, profile):
        """为同一源帧补全短提示证据；保留原模板结果，不接收模型动作或未知手牌。

        金色轮廓先限定字号、宽高比和占比，过滤昵称、按钮大底色及倒计时圆圈；
        自己出牌按钮明确可见时不扫描本人提示。多个不同候选出现时保持未知。
        """
        if observation.frame_id != frame.frame_id or observation.pts_us != frame.media_pts_us:
            return
        if observation.phase in ("bidding", "doubling", "settlement"):
            return
        if not observation.landlord or not observation.bottom:
            return
        scale = frame.height / profile.base_height
        for seat in ("self", "left", "right"):
            if seat in observation.passes:
                continue
            controls = observation.marker_evidence.get("phase", {})
            if (
                seat == "self"
                and controls.get("accepted")
                and controls.get("semantic") == "self_action_buttons"
            ):
                continue
            key = "pass_" + seat
            rect = profile.indicator_regions[key].pixels(frame.width, frame.height)
            rgb = rect.crop(frame.rgb)
            mask = MarkerLibrary._mask(rgb, "pass")
            candidates = []
            for window, x, y in MarkerLibrary._pass_windows(mask, scale):
                h, w = window.shape
                if not (45 * scale <= h <= 150 * scale and 1.2 <= w / h <= 4.5):
                    continue
                if not 0.10 <= np.mean(window > 0) <= 0.60:
                    continue
                crop = rgb[y : y + h, x : x + w]
                result = self.engine(cv2.cvtColor(crop, cv2.COLOR_RGB2BGR), use_det=False, use_cls=False)
                if result.txts is None or len(result.txts) != 1:
                    continue
                text = result.txts[0].replace(" ", "")
                score = float(result.scores[0])
                if text in ("不出", "不要", "要不起") and score >= 0.98:
                    candidates.append((text, score, [rect.x + x, rect.y + y, w, h]))
            if len(candidates) != 1:
                continue
            label, score, box = candidates[0]
            observation.passes.append(seat)
            observation.pass_labels[seat] = label
            observation.scores[key] = score
            observation.marker_evidence[key] = {
                "accepted": True,
                "label": label,
                "score": score,
                "box": box,
                "frame_id": frame.frame_id,
                "pts_us": frame.media_pts_us,
                "source": "ppocr-v5-complete-gold-text",
                "reason": "完整金色不出提示OCR，仍需跨帧和轮转确认",
            }
