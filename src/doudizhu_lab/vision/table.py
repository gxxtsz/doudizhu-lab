"""牌桌语义观察：牌面、固定标记和数字模板；不根据记牌器推测隐藏手牌。"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from doudizhu_lab.domain.models import Frame, NormalizedRect, Recognition, RegionStatus
from doudizhu_lab.domain.table import PHASE_NAMES as PHASE_NAMES
from doudizhu_lab.domain.table import SEAT_NAMES as SEAT_NAMES
from doudizhu_lab.domain.table import SEATS as SEATS
from doudizhu_lab.domain.table import TableObservation as TableObservation
from doudizhu_lab.storage.config import Profile, read_rgb, write_rgb
from doudizhu_lab.vision.cards import Recognizer


class MarkerLibrary:
    """固定图标/文字模板库，区域名限定用途，牌点与界面文字的模板不会混用。

    元数据为 region、label、source；数字标签为 0～9，phase/result 使用 PHASE_NAMES。
    所有样本来自人工确认的画面，弱匹配或相近候选返回 None。
    """

    def __init__(self, folder: Path):
        self.folder = folder
        self.templates = []
        self.errors = []
        self._scaled = {}
        self.reload()

    @staticmethod
    def _mask(rgb, family):
        """按标记用途消除动画背景差异，仍保留完整字形，不从牌面或昵称猜数字。

        计数牌使用浅色数字和边框；红色告警卡面、蓝色普通卡面的背景都被移除。
        不出字样使用金黄色字形，排除人物和蓝色牌桌背景。其他标记保留灰度模板。
        """
        if family == "pass":
            return ((rgb[:, :, 0] > 170) & (rgb[:, :, 1] > 130) & (rgb[:, :, 2] < 120)).astype(np.uint8) * 255
        if family == "count":
            mask = ((rgb[:, :, 0] > 170) & (rgb[:, :, 1] > 170)).astype(np.uint8) * 255
            # 发光的牌框偶尔与数字同为白色。去掉细长边框，防止“10”的边框
            # 在整体相关分数中压过字形差异而匹配成“18”；数字“1”仍有足够宽度。
            _, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
            for i, (_, _, width, height, area) in enumerate(stats[1:], 1):
                if (
                    height > width * 5
                    or width > height * 5
                    or area < 5
                    or height > max(35, rgb.shape[0] * 0.58)
                ):
                    mask[labels == i] = 0
            return mask
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    def reload(self):
        self.templates.clear()
        self.errors.clear()
        self._scaled.clear()
        for path in sorted(self.folder.glob("*.json")):
            try:
                item = json.loads(path.read_text(encoding="utf-8"))
                if item.get("disabled", False):
                    continue
                rgb = read_rgb(path.with_suffix(".png"))
                gray = self._mask(rgb, item["region"].split("_", 1)[0])
                if min(gray.shape) < 3 or float(gray.std()) < 3:
                    raise ValueError("模板没有足够的图形信息")
                self.templates.append((item, gray))
            except (OSError, ValueError, KeyError) as exc:
                self.errors.append(f"标记模板 {path.name} 无效：{exc}")

    def add(self, rgb, region: str, label: str, source: dict) -> Path:
        """保存手工确认的裁剪及来源信息；禁止空标签和无图形裁剪。"""
        if not region or not label or min(rgb.shape[:2]) < 3:
            raise ValueError("标记区域、标签和裁剪不能为空")
        digest = hashlib.sha256(rgb.tobytes()).hexdigest()
        self.folder.mkdir(parents=True, exist_ok=True)
        path = self.folder / f"marker-{digest[:16]}.png"
        write_rgb(path, rgb)
        path.with_suffix(".json").write_text(
            json.dumps(
                {
                    "region": region,
                    "label": label,
                    "source": source,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        self.reload()
        return path

    def matches(self, rgb, region: str, scale: float = 1.0):
        """返回区域内候选 (标签, 分数, x, y, w, h)，模板按基准画面比例缩放。"""
        if not rgb.size:
            return []
        found = []
        family = region.split("_", 1)[0]
        gray = self._mask(rgb, family)
        windows = self._pass_windows(gray, scale) if family == "pass" else [(gray, 0, 0)]
        if not windows:
            return []
        complete_numbers = family == "count" and self._has_complete_numbers()
        for index, (item, template) in enumerate(self.templates):
            if item["region"] not in (region, family):
                continue
            if complete_numbers and not item["label"].startswith("n:"):
                continue
            if family == "pass":
                # 角色镜头透视和弹出动画会分别改变文字的横向、纵向尺寸。
                # 只按完整金色轮廓拟合有限比例，仍用完整词模板及同一个 0.88 阈值。
                # 单个“要”或“要不”不能用文字前缀命中来代替完整“不出”证据。
                for window, left, top in windows:
                    fitted = self._fit_pass(template, window, scale)
                    if fitted is not None:
                        sample, x, y = fitted
                        h, w = sample.shape
                        score = float(
                            cv2.matchTemplate(window[y : y + h, x : x + w], sample, cv2.TM_CCOEFF_NORMED)[
                                0, 0
                            ]
                        )
                        if score >= 0.88:
                            found.append((item["label"], score, x + left, y + top, w, h))
            # 不出提示在弹出阶段会缩放；只枚举有限尺寸，保持原有 0.88 接受阈值。
            factors = (0.8, 0.9, 1.0, 1.1) if family == "pass" else (1.0,)
            for factor in factors:
                ratio = round(scale * factor, 3)
                cache_key = index, ratio
                sample = self._scaled.get(cache_key)
                if sample is None:
                    sample = (
                        template if abs(ratio - 1) < 0.005 else cv2.resize(template, None, fx=ratio, fy=ratio)
                    )
                    self._scaled[cache_key] = sample
                h, w = sample.shape
                for window, left, top in windows:
                    if h > window.shape[0] or w > window.shape[1]:
                        continue
                    score = cv2.matchTemplate(window, sample, cv2.TM_CCOEFF_NORMED)
                    for _ in range(4 if family == "count" else 1):
                        _, maximum, _, loc = cv2.minMaxLoc(score)
                        if maximum < 0.88:
                            break
                        x, y = loc
                        score[max(0, y - h // 2) : y + h // 2 + 1, max(0, x - w // 2) : x + w // 2 + 1] = -1
                        if complete_numbers:
                            patch = rgb[y + top : y + top + h, x + left : x + left + w].astype(np.int16)
                            r, g, b = np.moveaxis(patch, -1, 0)
                            blue = (b > 140) & (b > r * 1.25) & (r > 70) & (g > 90)
                            red = (r > 140) & (r > g * 1.4) & (r > b * 1.3)
                            # 数字必须位于浅蓝/红色计数牌中。二值字形匹配之后仍核对
                            # 完整牌框背景，不能把昵称“……16”当成剩牌 16。
                            if float(np.mean(blue | red)) < 0.5:
                                continue
                        if complete_numbers and self._holes(window[y : y + h, x : x + w]) != self._holes(
                            sample
                        ):
                            continue
                        found.append((item["label"], float(maximum), x + left, y + top, w, h))
        return sorted(found, key=lambda m: m[1], reverse=True)

    @staticmethod
    def _fit_pass(template, window, scale):
        """将完整提示字形拟合到同一轮廓框，限制两轴缩放和宽高变形。

        该步骤只补偿字号及轻微透视，不补齐缺字，不把整词拆成独立弱证据。
        """
        source = cv2.findNonZero(template)
        target = cv2.findNonZero(window)
        if source is None or target is None:
            return None
        sx, sy, sw, sh = cv2.boundingRect(source)
        x, y, w, h = cv2.boundingRect(target)
        fx, fy = w / sw / scale, h / sh / scale
        if not (0.65 <= fx <= 1.4 and 0.65 <= fy <= 1.4 and 0.75 <= fx / fy <= 1.35):
            return None
        sample = cv2.resize(template[sy : sy + sh, sx : sx + sw], (w, h), interpolation=cv2.INTER_LINEAR)
        return sample, x, y

    @staticmethod
    def _holes(mask):
        """数字的封闭内孔用于排除 0/8 等形近误匹配，不改变相关分数阈值。"""
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if hierarchy is None:
            return 0
        return sum(h[3] >= 0 and cv2.contourArea(c) >= 3 for c, h in zip(contours, hierarchy[0]))

    @staticmethod
    def _pass_windows(gray, scale):
        """先定位金色大字的连通轮廓，再在小窗口做严格模板匹配。

        轮廓只负责缩小计算范围，不直接判断文字。相邻动画字符会合并；昵称小字不进入窗口。
        与在整个牌桌区域枚举尺寸相比，可避免动画模板增多后拖慢实时识别。
        """
        contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = [cv2.boundingRect(c) for c in contours]
        boxes = [b for b in boxes if b[2] >= 20 * scale and b[3] >= 40 * scale]
        groups = []
        for x, y, w, h in sorted(boxes):
            merged = [x, y, x + w, y + h]
            for old in list(groups):
                if (
                    min(old[3], merged[3]) > max(old[1], merged[1])
                    and max(old[0], merged[0]) - min(old[2], merged[2]) < 35 * scale
                ):
                    merged = [
                        min(old[0], merged[0]),
                        min(old[1], merged[1]),
                        max(old[2], merged[2]),
                        max(old[3], merged[3]),
                    ]
                    groups.remove(old)
            groups.append(merged)
        pad = max(6, round(8 * scale))
        result = []
        for x0, y0, x1, y1 in groups:
            x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
            x1, y1 = min(gray.shape[1], x1 + pad), min(gray.shape[0], y1 + pad)
            result.append((gray[y0:y1, x0:x1], x0, y0))
        return result

    def _has_complete_numbers(self):
        return any(
            item["region"].split("_", 1)[0] == "count" and item["label"].startswith("n:")
            for item, _ in self.templates
        )

    def detect(self, rgb, region: str, scale=1.0, *, numeric=False):
        """返回标签、匹配分数和可追溯证据；box 是本次输入裁剪内的像素框。

        未命中与相近候选均返回 label=None，不能当作明确的空区域。
        数字优先使用完整计数卡；旧的单字数字模板仍由 number() 兼容读取。
        """
        if numeric and not self._has_complete_numbers():
            value, score = self.number(rgb, region, scale)
            return value, score, {"label": value, "score": score, "box": None, "legacy_digits": True}
        candidates = self.matches(rgb, region, scale)
        if numeric:
            candidates = [c for c in candidates if c[0].startswith("n:")]
        if not candidates:
            return None, 0.0, {"label": None, "score": 0.0, "box": None, "reason": "未命中标记"}
        first = candidates[0]
        other = next((c for c in candidates if c[0] != first[0]), None)
        evidence = {"label": first[0], "score": first[1], "box": list(first[2:]), "reason": ""}
        # 只有同一标签的所有候选模板都声明相同用途，才提供该语义证据。
        # 老模板或普通阶段文字没有声明时不推断成操作按钮，保持旧配置兼容。
        semantics = {
            item.get("semantic")
            for item, _ in self.templates
            if item["region"] in (region, region.split("_", 1)[0]) and item["label"] == first[0]
        }
        if len(semantics) == 1 and None not in semantics:
            evidence["semantic"] = next(iter(semantics))
        if other and first[1] - other[1] < 0.04:
            evidence["reason"] = "不同标签的匹配分数过近"
            return None, first[1], evidence
        label = int(first[0][2:]) if numeric else first[0]
        if numeric and not 0 <= label <= 20:
            evidence["reason"] = "剩牌数字超出范围"
            return None, first[1], evidence
        return label, first[1], evidence

    def identify(self, rgb, region: str, scale=1.0):
        """兼容首版的二元返回接口；需要原图证据的调用方使用 detect。"""
        label, score, _ = self.detect(rgb, region, scale)
        return label, score

    def number(self, rgb, region: str, scale=1.0):
        # 完整计数牌包含背景边框，能排除相邻玩家昵称中的数字；它是优先方案。
        # 标签 n:17 表示完整数字 17，旧版单个数字模板保留兼容。
        if self._has_complete_numbers():
            label, score, _ = self.detect(rgb, region, scale, numeric=True)
            return label, score
        accepted = []
        for candidate in self.matches(rgb, region, scale):
            label, score, x, y, w, h = candidate
            if not label.isdigit() or len(label) != 1:
                continue
            overlap = [other for other in accepted if abs(x - other[2]) < min(w, other[4]) * 0.65]
            if overlap:
                if overlap[0][0] != label and overlap[0][1] - score < 0.04:
                    return None, score
                continue
            accepted.append(candidate)
        accepted.sort(key=lambda c: c[2])
        if not 1 <= len(accepted) <= 2:
            return None, 0.0
        value = int("".join(c[0] for c in accepted))
        return (value, min(c[1] for c in accepted)) if 0 <= value <= 20 else (None, 0.0)


class TableRecognizer:
    """组合单帧牌点与独立界面标记；跨帧稳定和事件确认由 GameTracker 处理。"""

    def __init__(self, cards: Recognizer, markers: MarkerLibrary):
        self.cards, self.markers = cards, markers

    def observe(self, frame: Frame, profile: Profile) -> tuple[TableObservation, Recognition]:
        """单帧识别公开牌桌；兼容记牌器开关导致的底牌位置变化，保留人工自定义区域。"""
        start = perf_counter()
        profile = self._bottom_layout(frame, profile)
        result = self.cards.recognize(frame, profile)
        obs = TableObservation(
            frame.frame_id or frame.captured_at, frame.media_pts_us or 0, frame.captured_at
        )
        for region in result.regions:
            values = (
                [c.rank for c in region.cards]
                if region.status == RegionStatus.OK
                else ([] if region.status == RegionStatus.EMPTY else None)
            )
            if region.region == "hand":
                obs.hand = values
                obs.remaining["self"] = len(values) if values is not None else None
            elif region.region == "bottom":
                obs.bottom = values if values is not None and len(values) == 3 else None
            else:
                obs.plays["self" if region.region == "self_play" else region.region] = values
        landlords, actors, phases, winners = [], [], [], []
        scale = frame.height / profile.base_height
        # 先识别控制按钮等界面证据，再处理动作文字；不依赖配置 JSON 的键顺序。
        areas = sorted(profile.indicator_regions.items(), key=lambda pair: pair[0].startswith("pass_"))
        for key, area in areas:
            rect = area.pixels(frame.width, frame.height)
            crop = rect.crop(frame.rgb)
            controls = obs.marker_evidence.get("phase", {})
            if (
                key == "pass_self"
                and controls.get("accepted")
                and controls.get("semantic") == "self_action_buttons"
            ):
                # 本机布局中的“出牌”按钮与自己出牌后的金色文字互斥。
                # 跳过按钮大色块上的昂贵滑窗搜索；这里只表示尚未出现自己的动作
                # 提示，不产生不出事件，不影响另两座位的提示，也不使用轮次猜动作。
                label, score = None, 0.0
                evidence = {
                    "label": None,
                    "score": 0.0,
                    "box": None,
                    "reason": "自己操作按钮可见，等待动作提示",
                }
            else:
                label, score, evidence = self.markers.detect(
                    crop, key, scale, numeric=key.startswith("count_")
                )
            if evidence["box"]:
                evidence["box"][0] += rect.x
                evidence["box"][1] += rect.y
            evidence.update(accepted=label is not None, frame_id=obs.frame_id, pts_us=obs.pts_us)
            obs.marker_evidence[key] = evidence
            obs.scores[key] = score
            if label is None:
                continue
            family, _, seat = key.partition("_")
            if family == "count":
                obs.remaining[seat] = label
            elif family == "landlord":
                landlords.append(seat)
            elif family == "turn":
                actors.append(seat)
            elif family == "winner":
                winners.append(seat)
            elif family == "pass" and label in ("不出", "要不起", "不要", "过", "pass"):
                obs.passes.append(seat)
                obs.pass_labels[seat] = label
            elif key in ("phase", "result") and label in PHASE_NAMES:
                phases.append(label)
            elif key == "bottom_back" and label == "dealing":
                phases.append("dealing")
        obs.landlord = landlords[0] if len(landlords) == 1 else None
        obs.actor = actors[0] if len(actors) == 1 else None
        if "settlement" in phases:
            obs.phase = "settlement"
        elif "bidding" in phases:
            obs.phase = "bidding"
        elif "doubling" in phases:
            obs.phase = "doubling"
        elif "playing" in phases:
            obs.phase = "playing"
        elif "dealing" in phases:
            obs.phase = "dealing"
        elif (
            obs.hand
            and obs.bottom
            and obs.landlord
            and any(obs.plays.get(seat) for seat in SEATS)
            and any(
                count is not None and count < (20 if seat == obs.landlord else 17)
                for seat, count in obs.remaining.items()
                if seat != obs.landlord or count != 17
            )
        ):
            # 刚翻底牌、地主尚未收完三张时也有地主标志，不能提前进入出牌阶段。
            # 缺少阶段文字时，只有公开剩牌已低于开局数量才提供出牌阶段证据。
            obs.phase = "playing"
        if len(landlords) > 1 or len(actors) > 1:
            obs.warnings.append("多个座位标记冲突")
        if len(winners) == 1:
            winner = winners[0]
            # “最后出牌”是公开的零剩牌证据，不能由空手牌区域自行推断胜负。
            if all(r.status == RegionStatus.NOT_TABLE for r in result.regions):
                result = self.cards.recognize(frame, profile, table_confirmed=True)
                for region in result.regions:
                    if region.region in ("left", "right", "self_play"):
                        obs.plays["self" if region.region == "self_play" else region.region] = (
                            [c.rank for c in region.cards] if region.status == RegionStatus.OK else None
                        )
            obs.remaining[winner] = 0
            if winner == "self":
                obs.hand = []
            if obs.phase == "unknown":
                obs.phase = "playing"
        result.elapsed_ms = (perf_counter() - start) * 1000
        # 保存识别位置、分数和拒识状态，而不只保存已猜出的牌点，供赛后审计。
        obs.regions = [asdict(region) for region in result.regions]
        obs.elapsed_ms = result.elapsed_ms
        return obs, result

    @staticmethod
    def _bottom_layout(frame: Frame, profile: Profile) -> Profile:
        """默认布局只在明确没有长记牌器、中央存在浅色牌块时选择中央底牌。

        长记牌器的数字从不作为牌点真值。人工修改过的底牌区域始终优先；
        不改变持久配置，具体识别框随当前观察保存，可从同一像素确定性重现。
        """
        if profile.regions["bottom"] != NormalizedRect(0.273, 0.003, 0.065, 0.090):
            return profile

        def light(area):
            rgb = area.pixels(frame.width, frame.height).crop(frame.rgb).astype(np.int16)
            return float(np.mean((rgb.min(axis=2) > 150) & (rgb.max(axis=2) - rgb.min(axis=2) < 45)))

        bar = light(NormalizedRect(0.55, 0.01, 0.16, 0.075))
        center = NormalizedRect(0.462, 0.003, 0.072, 0.090)
        if bar < 0.30 and light(center) > 0.35:
            profile = deepcopy(profile)
            profile.regions["bottom"] = center
        return profile
