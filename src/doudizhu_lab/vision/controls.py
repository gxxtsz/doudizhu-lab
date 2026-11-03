"""真实牌面布局与操作按钮识别；缺少视觉依据时保留未知，不推测可点击性。"""

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np

from doudizhu_lab.domain.bot_types import ActionButton, HandCard, ObservationSnapshot
from doudizhu_lab.domain.models import NormalizedRect, PixelRect
from doudizhu_lab.storage.config import read_rgb, write_rgb


def add_button_template(
    folder: Path, frame, box: PixelRect, *, kind: str, label: str, enabled: bool, source: dict
):
    """保存人工框定的按钮文字与背景样本；出处包含源帧、原图哈希、矩形和可用状态。

    kind 只支持 play/pass，label 限定出牌、不出、要不起。必须包含足够背景，
    匹配时同时检查 RGB 颜色，不能用灰度字形把禁用按钮当作启用按钮。
    """
    if (kind, label) not in (("play", "出牌"), ("pass", "不出"), ("pass", "要不起")):
        raise ValueError("操作按钮标签无效")
    if (
        box.width < 30
        or box.height < 20
        or box.x < 0
        or box.y < 0
        or box.x + box.width > frame.width
        or box.y + box.height > frame.height
    ):
        raise ValueError("请框选完整按钮文字及周围背景")
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"button-{uuid4().hex[:12]}.png"
    write_rgb(path, box.crop(frame.rgb))
    data = {
        "kind": kind,
        "label": label,
        "enabled": enabled,
        "height": frame.height,
        "source": source
        | {
            "frame_id": frame.frame_id,
            "pts_us": frame.media_pts_us,
            "image_sha256": hashlib.sha256(frame.rgb.tobytes()).hexdigest(),
            "box": [box.x, box.y, box.width, box.height],
        },
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    path.with_suffix(".json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


class ControlRecognizer:
    """单帧操作对象检测；跨帧稳定由执行器确认，源模板随会话冻结。"""

    def __init__(self, folder: Path):
        self.templates = []
        for path in sorted(folder.glob("*.json")):
            value = json.loads(path.read_text(encoding="utf-8"))
            image = path.with_suffix(".png")
            if hashlib.sha256(image.read_bytes()).hexdigest() != value["sha256"]:
                raise ValueError("操作按钮模板哈希不一致")
            self.templates.append((value, read_rgb(image)))

    def enrich(self, snapshot: ObservationSnapshot, profile) -> ObservationSnapshot:
        """为同一帧补充选中状态及点击位置；不会改变语义版本或发送设备输入。"""
        config = profile.bot_controls
        if (
            not config.get("verified")
            or config.get("normal_top") is None
            or config.get("selected_offset") is None
        ):
            return replace(snapshot, control_reason="操作区域尚未标定")
        frame, obs = snapshot.frame, snapshot.observation
        recognition = json.loads(snapshot.recognition_json)
        if not recognition:
            return replace(snapshot, control_reason="本帧缺少牌点定位")
        region = next((r for r in recognition["regions"] if r["region"] == "hand"), None)
        if not region or region["status"] != "已识别":
            return replace(snapshot, control_reason="本帧手牌无法完整定位")
        cards = sorted(region["cards"], key=lambda c: c["box"]["x"])
        normal = config["normal_top"] * frame.height
        selected = normal - config["selected_offset"] * frame.height
        tolerance = config.get("top_tolerance", 0.005) * frame.height
        scale = frame.height / profile.base_height
        hand = []
        for index, card in enumerate(cards):
            box = card["box"]
            x = round(box["x"] + box["width"] * 0.45)
            # 牌点离牌顶约18px。先用字形顶边排除动画中间高度，再核对该高度
            # 下的浅色牌边。不能从搜索窗里第一条白色带取边：人物衣服及上移
            # 邻牌的底层同样可能是白色，曾导致未选中的2/K被错误判为已选中。
            estimated = box["y"] - config.get("glyph_top_padding", 18 / 1080) * frame.height
            edge = min((normal, selected), key=lambda value: abs(value - estimated))
            if abs(edge - estimated) > tolerance:
                edge = None
            if edge is not None:
                top, bottom = round(edge + 5 * scale), round(edge + 12 * scale)
                strip = frame.rgb[top:bottom, max(0, x - 3) : x + 4].astype(np.int16)
                white = (strip.min(axis=2) > 175) & ((strip.max(axis=2) - strip.min(axis=2)) < 45)
                if not strip.size or float(white.mean()) < 0.8:
                    # 双王高亮金框可能恰好覆盖窄采样列。字形高度已经排除动画中间态，
                    # 再在该牌内部多个位置检查浅色顶边；不移动到邻牌或人物白色衣服。
                    limit = (
                        cards[index + 1]["box"]["x"] - 3
                        if index + 1 < len(cards)
                        else box["x"] + box["width"]
                    )
                    samples = []
                    for probe in (box["x"] + box["width"] * r for r in (0.2, 0.65, 0.85)):
                        px = round(probe)
                        if px - 2 < box["x"] or px + 3 > limit:
                            continue
                        accepted = False
                        for offset in (0, round(12 * scale)):
                            patch = frame.rgb[top + offset : bottom + offset, px - 2 : px + 3].astype(
                                np.int16
                            )
                            valid = (patch.min(axis=2) > 175) & ((patch.max(axis=2) - patch.min(axis=2)) < 45)
                            accepted |= bool(patch.size and float(valid.mean()) >= 0.8)
                        samples.append(accepted)
                    if sum(samples) < 2:
                        edge = None
            is_selected = None
            if edge is not None:
                if abs(edge - normal) <= tolerance:
                    is_selected = False
                elif abs(edge - selected) <= tolerance:
                    is_selected = True
            y = round((edge if edge is not None else box["y"]) + 92 * scale)
            if index + 1 < len(cards) and x >= cards[index + 1]["box"]["x"]:
                is_selected = None
            hit = PixelRect(max(0, x - 3), max(0, min(frame.height - 7, y - 3)), 7, 7)
            hand.append(HandCard(card["rank"], index, hit, is_selected))
        buttons = self.detect_buttons(frame, obs, profile)
        reason = "选中高度或牌边界无法确认" if any(c.selected is None for c in hand) else ""
        return replace(snapshot, hand=tuple(hand), buttons=tuple(buttons), control_reason=reason)

    def detect_buttons(self, frame, obs, profile):
        """独立检测完整按钮及背景；叫抢、加倍、发牌和结算画面不产生操作按钮。"""
        config = profile.bot_controls
        scale = frame.height / profile.base_height
        if obs.get("phase") not in ("playing", "unknown") or obs.get("actor") not in (None, "self"):
            return []
        buttons = []
        for value, template in self.templates:
            area = config.get("buttons", {}).get(value["kind"])
            if not area:
                continue
            rect = NormalizedRect(**area).pixels(frame.width, frame.height)
            crop = rect.crop(frame.rgb)
            ratio = frame.height / value["height"]
            sample = cv2.resize(template, None, fx=ratio, fy=ratio) if abs(ratio - 1) > 0.001 else template
            h, w = sample.shape[:2]
            if h > crop.shape[0] or w > crop.shape[1]:
                continue
            _, score, _, (x, y) = cv2.minMaxLoc(cv2.matchTemplate(crop, sample, cv2.TM_CCOEFF_NORMED))
            difference = np.abs(crop[y : y + h, x : x + w].astype(np.float32) - sample).mean()
            if score >= 0.90 and difference <= 22:
                buttons.append(
                    ActionButton(
                        value["kind"],
                        PixelRect(rect.x + x, rect.y + y, w, h),
                        value["enabled"],
                        value["label"],
                    )
                )
        # 同种按钮出现两个互不重叠候选或启用/禁用冲突时，不允许选一个猜测执行。
        for kind in ("play", "pass"):
            found = [b for b in buttons if b.kind == kind]
            if found and (
                len({b.enabled for b in found}) > 1
                or max(b.hit.x for b in found) - min(b.hit.x for b in found) > 30 * scale
            ):
                buttons = [b for b in buttons if b.kind != kind]
        return buttons

    def augment_observation(self, frame, observation, profile):
        """在跟踪和落盘前加入本帧按钮证据；要不起按钮本身可确认自己的行动回合。"""
        from dataclasses import asdict

        buttons = self.detect_buttons(frame, asdict(observation), profile)
        if buttons:
            observation.marker_evidence["control_buttons"] = {
                "accepted": True,
                "buttons": [asdict(b) for b in buttons],
                "frame_id": frame.frame_id,
            }
            observation.phase = "playing"
            observation.actor = "self"
        return buttons
