"""固定牌面的可解释识别：白色牌面定位、牌点分割、模板比较和保守拒识。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from doudizhu_lab.domain.models import (
    RANKS,
    CardObservation,
    Frame,
    PixelRect,
    Recognition,
    RegionResult,
    RegionStatus,
)
from doudizhu_lab.paths import resolve_data_dir
from doudizhu_lab.storage.config import Profile, read_rgb


def ink_mask(rgb: np.ndarray) -> np.ndarray:
    """提取牌点的黑色/红色墨迹；使用 int16 计算，避免 uint8 差值溢出。"""
    r, g, b = np.moveaxis(rgb.astype(np.int16), -1, 0)
    dark = np.maximum(np.maximum(r, g), b) < 145
    red = (r * 10 > g * 14) & (r * 10 > b * 13) & (g < 165) & (r > 100)
    return ((dark | red) * 255).astype(np.uint8)


def white_mask(rgb: np.ndarray) -> np.ndarray:
    """寻找灰白色扑克牌背景，排除大部分蓝色牌桌与红色角色装饰。"""
    # 三个独立通道逐元素比较与 axis=2 归约完全等价，避免对每个 RGB 小向量
    # 重复执行通用归约。原图百万像素时此处占据显著耗时，阈值和像素结果不变。
    r, g, b = np.moveaxis(rgb, -1, 0)
    maximum = np.maximum(np.maximum(r, g), b)
    minimum = np.minimum(np.minimum(r, g), b)
    return ((minimum > 155) & (maximum - minimum < 65)).astype(np.uint8) * 255


def normalize_glyph(mask: np.ndarray) -> np.ndarray | None:
    """将完整字形紧边裁剪后等比缩放到 48×64；保留宽高比例以区分 10、J 与 JOKER。"""
    points = cv2.findNonZero(mask)
    if points is None:
        return None
    x, y, w, h = cv2.boundingRect(points)
    if w < 2 or h < 3:
        return None
    scale = min(44 / w, 60 / h)
    dw, dh = max(1, round(w * scale)), max(1, round(h * scale))
    glyph = cv2.resize(mask[y : y + h, x : x + w], (dw, dh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((64, 48), np.float32)
    ox, oy = (48 - dw) // 2, (64 - dh) // 2
    canvas[oy : oy + dh, ox : ox + dw] = glyph / 255.0
    return canvas


@dataclass
class Template:
    rank: str
    group: str
    glyph: np.ndarray
    color: str
    source: str | dict


def glyph_color(rgb: np.ndarray) -> str:
    """根据墨迹中的红色占比区分红、黑；只用于大小王，不作为普通牌花色输出。"""
    ink = ink_mask(rgb) > 0
    if not np.any(ink):
        return "black"
    data = rgb.astype(np.int16)
    red = (data[:, :, 0] > data[:, :, 1] * 1.4) & (data[:, :, 0] > data[:, :, 2] * 1.3)
    return "red" if float(np.mean(red[ink])) > 0.4 else "black"


def similarity(a: np.ndarray, b: np.ndarray) -> float:
    """余弦相似度并允许一像素平移；返回模板匹配分数，不能解释成概率。"""
    norm = float(np.linalg.norm(a) * np.linalg.norm(b))
    if norm < 1e-6:
        return 0.0
    best = 0.0
    for dy, dx in ((0, 0), (0, 1), (0, -1), (1, 0), (-1, 0)):
        shifted = np.roll(b, (dy, dx), axis=(0, 1))
        best = max(best, float(np.sum(a * shifted)) / norm)
    return min(1.0, best)


class Recognizer:
    """读取字形模板，对单帧五个独立区域识别；不保存牌局历史，也不调用 ADB。"""

    def __init__(self, template_dir: Path, *, include_bundled: bool = True):
        self.template_dir = template_dir
        self.include_bundled = include_bundled
        self.templates: list[Template] = []
        self.load_errors: list[str] = []
        self._previous_live = None
        self._banks = {}
        self.reload()

    def reload(self) -> None:
        """加载模板及配套 JSON；坏模板记录警告并跳过，避免把未标注图片猜成牌点。"""
        self.templates.clear()
        self.load_errors.clear()
        folders = [self.template_dir]
        if self.include_bundled:
            # 本次交付采集的模板属于运行数据，存放在 Git 忽略目录，不夹带进源码版本库。
            defaults = resolve_data_dir() / "templates"
            if defaults.resolve() != self.template_dir.resolve():
                folders.insert(0, defaults)
        for folder in folders:
            for metadata in sorted(folder.glob("*.json")):
                try:
                    item = json.loads(metadata.read_text(encoding="utf-8"))
                    if item["rank"] not in RANKS or item["group"] not in ("hand", "bottom", "table"):
                        raise ValueError("模板标签或分组无效")
                    rgb = read_rgb(metadata.with_suffix(".png"))
                    glyph = normalize_glyph(ink_mask(rgb))
                    if glyph is None:
                        raise ValueError("没有有效牌点墨迹")
                    self.templates.append(
                        Template(item["rank"], item["group"], glyph, glyph_color(rgb), item.get("source", ""))
                    )
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    self.load_errors.append(f"模板 {metadata.name} 无法加载：{exc}")
        # 预先计算模板的五种微小平移；批量点积代替每张牌重复进行 Python 嵌套循环。
        # 评分仍是同一个余弦相似度，阈值及拒识语义不变。
        self._banks = {}
        for group in ("hand", "bottom", "table"):
            rows, labels = [], []
            for template in self.templates:
                if template.group != group:
                    continue
                norm = float(np.linalg.norm(template.glyph))
                for dy, dx in ((0, 0), (0, 1), (0, -1), (1, 0), (-1, 0)):
                    rows.append(np.roll(template.glyph, (dy, dx), axis=(0, 1)).ravel() / max(norm, 1e-8))
                    labels.append((template.rank, template.color))
            if rows:
                self._banks[group] = (np.asarray(rows, dtype=np.float32), labels)

    def coverage(self) -> dict[str, list[str]]:
        """列出各组已经具备的牌点标签，便于用户明确看到还缺哪些模板。"""
        return {
            group: sorted({t.rank for t in self.templates if t.group == group})
            for group in ("hand", "bottom", "table")
        }

    def candidates(self, rgb: np.ndarray, expected_height: float, *, multi_row=False) -> list[PixelRect]:
        """在白色牌面上找最上方的一排牌点，合并 10 的双字符及竖排 JOKER。

        使用白色邻域约束排除背景人物和提示文字。仅输出候选位置，最终标签由模板决定。
        """
        components = self._candidate_components(rgb, expected_height)
        return self._candidate_boxes(components, expected_height, multi_row=multi_row)

    def classifier_ready(self, group: str) -> bool:
        """查询分类器是否可用；模板按覆盖判断，预训练实现可按模型就绪状态覆盖。"""
        return any(t.group == group for t in self.templates)

    def candidate_rows(
        self, rgb: np.ndarray, expected_height: float
    ) -> tuple[list[PixelRect], list[PixelRect]]:
        """共用一次像素分割，分别返回主排候选与包含第二排的候选。

        白色邻域、墨迹连通分量和竖排 JOKER 只依赖当前裁剪，与是否检查第二排无关。
        此接口复用这些中间结果；两种排布仍各自执行原有的牌点筛选规则，不跨帧缓存。
        """
        components = self._candidate_components(rgb, expected_height)
        return (
            self._candidate_boxes(components, expected_height, multi_row=False),
            self._candidate_boxes(components, expected_height, multi_row=True),
        )

    def _candidate_components(self, rgb: np.ndarray, expected_height: float):
        """提取同一裁剪共用的连通分量；空图或没有候选时返回 None。"""
        if rgb.size == 0:
            return None
        white = white_mask(rgb)
        ink = ink_mask(rgb)
        # 扩张白色背景以包容字形内部的黑色像素，同时限制背景噪声。
        support = cv2.dilate(white, np.ones((15, 15), np.uint8))
        ink = cv2.bitwise_and(ink, support)
        count, _, stats, _ = cv2.connectedComponentsWithStats(ink)
        blobs = []
        for x, y, w, h, area in stats[1:count]:
            if x <= 1 or y <= 1 or x + w >= rgb.shape[1] - 1 or y + h >= rgb.shape[0] - 1:
                continue
            if (
                h < max(5, expected_height * 0.20)
                or h > expected_height * 1.45
                # 细长牌边阴影不能被当成数字 1 再与 A/其他牌点合并。
                # 实际 10 的“1”和竖排 JOKER 字母均宽于字高的 12%。
                or w < max(2, expected_height * 0.12)
                or w > expected_height * 1.05
                or area < 12
            ):
                continue
            pad = max(3, round(expected_height * 0.09))
            x0, y0 = max(0, x - pad), max(0, y - pad)
            around = white[y0 : min(rgb.shape[0], y + h + pad), x0 : min(rgb.shape[1], x + w + pad)]
            if np.mean(around > 0) < 0.36:
                continue
            blobs.append((int(x), int(y), int(w), int(h)))
        if not blobs:
            return None
        joker_groups = []
        for x, y, w, h in blobs:
            if h >= expected_height * 0.85:
                continue
            chain = sorted(
                [
                    b
                    for b in blobs
                    if b[1] >= y + h and abs((b[0] + b[2] / 2) - (x + w / 2)) < max(w * 0.5, 4)
                ],
                key=lambda b: b[1],
            )
            end, used = y + h, []
            for b in chain:
                if b[1] - end > max(h * 0.8, 8) or b[3] > h * 1.5:
                    break
                used.append(b)
                end = b[1] + b[3]
                if len(used) == 4:
                    break
            if len(used) == 4:
                left = min([x] + [b[0] for b in used])
                right = max([x + w] + [b[0] + b[2] for b in used])
                joker_groups.append((left, y, right - left, end - y))
        # 被选中的大王牌，其红色 R 可能与金色描边的暗影连在一起。
        # 独立红色分量恢复五个字母的边界，避免把四字母截断模板误当作完整 JOKER。
        channels = rgb.astype(np.int16)
        red = (
            (channels[:, :, 0] > channels[:, :, 1] * 1.4)
            & (channels[:, :, 0] > channels[:, :, 2] * 1.3)
            & (channels[:, :, 1] < 165)
        ).astype(np.uint8)
        # 黑色 R 在手牌最左端可能与半透明倍数提示的灰底粘连；更严格的纯黑
        # 分量能分开它们。仍要求五个连续字母，不以残缺 JOKE 推断整张王。
        red |= ((rgb[:, :, 0] < 90) & (rgb[:, :, 1] < 90) & (rgb[:, :, 2] < 90)).astype(np.uint8)
        _, _, red_stats, _ = cv2.connectedComponentsWithStats(red)
        red_parts = [
            tuple(map(int, b[:4]))
            for b in red_stats[1:]
            if expected_height * 0.2 <= b[3] < expected_height * 0.85 and b[4] >= 8
        ]
        for x, y, w, h in red_parts:
            aligned = sorted(
                [b for b in red_parts if b[1] >= y and abs(b[0] + b[2] / 2 - x - w / 2) < max(w * 0.45, 4)],
                key=lambda b: b[1],
            )[:5]
            # 小底牌的K/E抗锯齿后会粘成一块，实际五字母只产生四个连通块。
            # 仅允许一个约双字高的窄块；最终仍须完整JOKER模板及红黑颜色分类，
            # 不以普通J、四字母JOKE或单纯纵向长条直接判为王。
            heights = sorted(b[3] for b in aligned)
            joined = (
                len(aligned) == 4
                and heights[1] > 0
                and 1.65 * heights[1] <= heights[-1] <= 2.6 * heights[1]
                and sum(b[3] >= heights[1] * 1.65 for b in aligned) == 1
                and max(b[2] for b in aligned) <= expected_height * 0.45
            )
            if (len(aligned) == 5 or joined) and all(
                0 <= b[1] - a[1] - a[3] < max(h * 0.8, 8) for a, b in zip(aligned, aligned[1:])
            ):
                left = min(b[0] for b in aligned)
                right = max(b[0] + b[2] for b in aligned)
                height = aligned[-1][1] + aligned[-1][3] - y
                if height >= (right - left) * 3.5:
                    joker_groups.append((left, y, right - left, height))
        merged_jokers = []
        for j in sorted(joker_groups, key=lambda box: box[3], reverse=True):
            if not any(
                abs(k[0] + k[2] / 2 - j[0] - j[2] / 2) < max(k[2], j[2]) * 0.6
                and (
                    abs(k[1] - j[1]) < expected_height * 0.2
                    # 末手标记贴近 JOKER 的底部时，其笔画可能与 O/K/E/R
                    # 组成第二条短链。已定位的完整长链包住该链时只保留一次；
                    # 不能把同一张王下半部再输出为一张无法确认的牌。
                    or (k[1] <= j[1] and j[1] + j[3] <= k[1] + k[3])
                )
                for k in merged_jokers
            ):
                merged_jokers.append(j)
        joker_groups = merged_jokers
        # 普通牌点高度接近 expected_height；小字号 JOKER 的首字也允许参与。
        primary = [b for b in blobs if b[3] >= expected_height * 0.85]
        eligible = [
            b
            for b in blobs
            if b[3] >= expected_height * 0.62
            and not any(
                j[0] <= b[0] + b[2] / 2 <= j[0] + j[2] and j[1] - expected_height * 0.05 <= b[1] < j[1] + j[3]
                for j in joker_groups
            )
        ] + joker_groups
        if not eligible:
            return None
        first_y = min(b[1] for b in (primary + joker_groups or eligible))
        return white, blobs, eligible, first_y

    @staticmethod
    def _candidate_boxes(components, expected_height: float, *, multi_row: bool) -> list[PixelRect]:
        """由同帧分割结果构造一排或两排候选；只读共享分量，返回独立坐标列表。"""
        if components is None:
            return []
        white, blobs, eligible, first_y = components
        row = sorted(
            [
                b
                for b in eligible
                if first_y - expected_height * 0.15
                <= b[1]
                <= first_y + expected_height * (1.9 if multi_row else 0.9)
            ],
            key=lambda b: (
                (round((b[1] - first_y) / (expected_height * 0.8)), b[0]) if multi_row else (0, b[0])
            ),
        )
        # 每列只取最靠上的字形；抬起牌的花色可能与未抬起牌的牌点处于同一高度。
        row = [
            b
            for b in row
            if not any(
                a[1] + a[3] <= b[1]
                and (not multi_row or b[1] - a[1] < expected_height * 1.1)
                and abs(a[0] + a[2] / 2 - b[0] - b[2] / 2) < max(a[2], b[2]) * 0.55
                for a in row
            )
        ]
        groups: list[list[int]] = []
        for x, y, w, h in row:
            if groups:
                px, py, pw, ph = groups[-1]
                gap = x - (px + pw)
                # “1”较窄，且与“0”紧邻，不能把相邻两张相同牌合并。
                if (
                    0 <= gap <= expected_height * 0.16
                    and pw < expected_height * 0.43
                    and abs(y - py) < expected_height * 0.22
                ):
                    groups[-1] = [px, min(y, py), x + w - px, max(y + h, py + ph) - min(y, py)]
                    continue
            groups.append([x, y, w, h])
        boxes = []
        for x, y, w, h in groups:
            if w < expected_height * 0.18:
                continue
            if expected_height * 1.22 < h < expected_height * 1.6:
                continue
            # 扑克牌牌点上方有平整的浅色留白；衣袖、牌边阴影虽靠近白色区域，
            # 通常没有这条留白。只利用几何外观过滤背景，不根据预测牌点删疑难牌。
            # 竖排 JOKER 印刷位置更靠近牌顶；使用紧邻字形的留白，避免把小底牌
            # 上方的灰色圆角边框也纳入白色比例检查而整张漏掉。
            top_scale = 0.05 if h >= expected_height * 1.6 else 0.10
            top = max(0, y - max(2, round(expected_height * top_scale)))
            end_top = max(top + 1, y - max(1, round(expected_height * 0.02)))
            strip = white[top:end_top, x : x + w]
            if strip.size and float(np.mean(strip > 0)) < 0.78:
                continue
            below = [
                b
                for b in blobs
                if b[1] > y + h * 0.6
                and b[1] < y + expected_height * 4.2
                and abs((b[0] + b[2] / 2) - (x + w / 2)) < max(w * 0.6, 6)
            ]
            chain = sorted(below, key=lambda b: b[1])
            if h < expected_height * 0.85 and len(chain) >= 3:
                # 竖排 JOKER 要求紧密连续，避免把下面的独立花色或按钮拼接进来。
                end = y + h
                used = []
                for b in chain:
                    if b[1] - end > max(h * 0.7, 7):
                        break
                    used.append(b)
                    end = b[1] + b[3]
                if len(used) >= 3:
                    left = min([x] + [b[0] for b in used])
                    right = max([x + w] + [b[0] + b[2] for b in used])
                    x, w, h = left, right - left, end - y
            boxes.append(PixelRect(x, y, w, h))
        return boxes

    def classify(self, crop: np.ndarray, group: str, profile: Profile) -> tuple[str | None, float, str]:
        """比较同组模板，阈值不足或前两名太接近时拒识；大小王还须通过颜色检查。"""
        glyph = normalize_glyph(ink_mask(crop))
        if glyph is None:
            return None, 0, "没有有效字形"
        scores: dict[str, float] = {}
        if group in self._banks:
            bank, labels = self._banks[group]
            values = np.einsum("ij,j->i", bank, glyph.ravel(), optimize=False) / max(
                float(np.linalg.norm(glyph)), 1e-8
            )
            color = glyph_color(crop)
            for (rank, template_color), value in zip(labels, values):
                if rank in ("BJ", "SJ") and template_color != color:
                    continue
                scores[rank] = max(scores.get(rank, 0), min(1.0, float(value)))
        ranked = sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
        if not ranked:
            return None, 0, "本组尚无有效模板"
        rank, score = ranked[0]
        margin = score - (ranked[1][1] if len(ranked) > 1 else 0)
        if score < profile.threshold or margin < profile.margin:
            return None, score, "匹配不足或候选接近"
        return rank, score, ""

    def recognize(
        self,
        frame: Frame,
        profile: Profile,
        *,
        stabilize: bool = False,
        table_confirmed: bool = False,
        regions: set[str] | None = None,
    ) -> Recognition:
        """识别当前稳定布局；非横屏/非牌桌明确返回状态，不把缺失或失败伪装成空牌。

        regions 可限制分类区域，但牌桌判定仍使用完整校准配置。混合识别器据此避免
        对同一张牌同时运行两个分类器；不传入时保持旧版五区域识别行为。
        """
        start = perf_counter()
        results = []
        ratio = frame.width / frame.height
        expected_ratio = profile.base_width / profile.base_height
        bottom_crop = profile.regions["bottom"].pixels(frame.width, frame.height).crop(frame.rgb)
        hand_crop = profile.regions["hand"].pixels(frame.width, frame.height).crop(frame.rgb)
        candidate_cache = {}
        if abs(ratio - expected_ratio) < 0.15:
            for key, crop in (("hand", hand_crop), ("bottom", bottom_crop)):
                candidate_cache[key] = self.candidates(
                    crop, profile.glyph_heights[key] * frame.height / profile.base_height
                )
        table_like = abs(ratio - expected_ratio) < 0.15 and (
            table_confirmed
            or (len(candidate_cache.get("bottom", [])) == 3 and np.mean(white_mask(bottom_crop) > 0) > 0.24)
            or (len(candidate_cache.get("hand", [])) >= 2 and np.mean(white_mask(hand_crop) > 0) > 0.18)
        )
        for region, area in profile.regions.items():
            if regions is not None and region not in regions:
                continue
            if not table_like:
                results.append(
                    RegionResult(region, RegionStatus.NOT_TABLE, message="请进入横屏牌桌，或校准当前布局")
                )
                continue
            rect = area.pixels(frame.width, frame.height)
            crop = rect.crop(frame.rgb)
            expected_h = profile.glyph_heights[region] * frame.height / profile.base_height
            boxes, extra = (
                (candidate_cache[region], [])
                if region in candidate_cache
                else self.candidate_rows(crop, expected_h)
            )
            if region not in ("hand", "bottom") and boxes:
                # 长组合折成第二排；第二排与第一排花色处于同一高度，因此只加入
                # 严格匹配完整牌点的新增位置。主排的未知牌仍保留为无法确认。
                # 若第二排遮挡而遗漏，状态机的剩牌差分会阻止把残缺组合确认成动作。
                first_y = min(b.y for b in boxes)
                for candidate in extra:
                    if candidate.y < first_y + expected_h * 0.95:
                        continue
                    rank, score, _ = self.classify(candidate.crop(crop), "table", profile)
                    if rank is not None and score >= max(profile.threshold, 0.92):
                        if not any(abs(b.x - candidate.x) < 4 and abs(b.y - candidate.y) < 4 for b in boxes):
                            boxes.append(candidate)
            if region == "self_play":
                # 自己的动作按钮与倒计时同处此区域，但数字明显小于桌面牌点。
                # 按用户标定的字高过滤，不能把倒计时的 2 记录成打出了一张 2。
                boxes = [box for box in boxes if box.height >= expected_h * 0.82]
            group = region if region in ("hand", "bottom") else "table"
            if not boxes:
                white_ratio = float(np.mean(white_mask(crop) > 0))
                status = (
                    RegionStatus.UNCERTAIN if white_ratio > 0.08 or region == "bottom" else RegionStatus.EMPTY
                )
                message = (
                    "底牌尚未翻开或无法确认"
                    if region == "bottom"
                    else ("存在牌面或遮挡，请检查区域及动画" if status == RegionStatus.UNCERTAIN else "")
                )
                results.append(RegionResult(region, status, message=message))
                continue
            cards = []
            for box in boxes:
                rank, score, reason = self.classify(box.crop(crop), group, profile)
                cards.append(
                    CardObservation(
                        rank, PixelRect(box.x + rect.x, box.y + rect.y, box.width, box.height), score, reason
                    )
                )
            has_templates = self.classifier_ready(group)
            status = (
                RegionStatus.NO_TEMPLATES
                if not has_templates
                else (RegionStatus.OK if all(c.rank for c in cards) else RegionStatus.UNCERTAIN)
            )
            results.append(RegionResult(region, status, cards))
        recognition = Recognition(results, frame.captured_at, (perf_counter() - start) * 1000)
        if stabilize:
            # 连续预览至少两帧位置和牌点一致才确认；首次出现、移动或动画中的牌先拒识。
            # 保存的是未加稳定性标记的结果，避免“上一帧被拒识”导致永远无法恢复。
            previous = self._previous_live
            self._previous_live = (frame.serial, frame.width, frame.height, frame.started, recognition)
            valid = (
                previous is not None
                and previous[:3] == (frame.serial, frame.width, frame.height)
                and 0 < frame.started - previous[3] <= 3.5
            )
            guarded = []
            last_regions = {r.region: r for r in previous[4].regions} if valid else {}
            for region in results:
                last = last_regions.get(region.region)
                stable = (
                    last is not None
                    and last.status == region.status
                    and len(last.cards) == len(region.cards)
                    and all(
                        a.rank == b.rank
                        and max(
                            abs(a.box.x - b.box.x),
                            abs(a.box.y - b.box.y),
                            abs(a.box.width - b.box.width),
                            abs(a.box.height - b.box.height),
                        )
                        <= 3
                        for a, b in zip(last.cards, region.cards)
                    )
                )
                if region.status in (RegionStatus.OK, RegionStatus.EMPTY) and not stable:
                    guarded.append(
                        RegionResult(
                            region.region,
                            RegionStatus.UNCERTAIN,
                            [
                                CardObservation(None, c.box, c.score, "画面变化，等待下一帧稳定")
                                for c in region.cards
                            ],
                            "画面变化，等待下一帧稳定",
                        )
                    )
                else:
                    guarded.append(region)
            recognition = Recognition(guarded, frame.captured_at, (perf_counter() - start) * 1000)
        return recognition
