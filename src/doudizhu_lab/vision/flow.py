"""牌局外页面的同帧观察与保守按钮定位；OCR不读取手牌答案，不直接操作设备。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from time import monotonic

import cv2
import numpy as np

from doudizhu_lab.domain.models import Frame, PixelRect
from doudizhu_lab.domain.settings import FLOW_ROOMS
from doudizhu_lab.vision.ocr import make_ocr

SCENE_NAMES = {
    "lobby": "斗地主大厅",
    "room": "场次房间",
    "matching": "匹配中",
    "dealing": "发牌",
    "bidding": "叫抢地主",
    "doubling": "加倍",
    "playing": "出牌",
    "settlement": "结算",
    "showcase": "战绩展示",
    "ending": "结算动画等待按钮",
    "unknown": "页面待确认",
    "room_confirmation": "已选场次确认",
}
LABELS = {
    "开始匹配": "match",
    "不叫": "decline_bid",
    "不叫地主": "decline_bid",
    "不抢": "decline_bid",
    "不抢地主": "decline_bid",
    "叫地主": "call_landlord",
    "叫": "call_landlord",
    "抢地主": "rob_landlord",
    "抢": "rob_landlord",
    "不加倍": "decline_double",
    "继续": "continue",
    "继续游戏": "continue",
    "继续对局": "continue",
    "再来一局": "continue",
    "返回大厅": "return_lobby",
    "返回": "return_lobby",
    "退出": "return_lobby",
    "确定": "confirm_room",
}


@dataclass(frozen=True)
class FlowButton:
    """当前帧内有文字、位置和背景证据的按钮；enabled=False 的对象永不交给执行器。"""

    kind: str
    label: str
    hit: PixelRect
    enabled: bool
    score: float = 1.0


@dataclass(frozen=True)
class FlowObservation:
    """流程观察与原图绑定；table_frame 只允许来自相同源帧的牌局观察。

    scene 描述当前界面，不代表历史可信。即使画面已结算，也须单独核对单局执行器
    和 GameTracker，才能把这局记成机器人完成。texts 保留原始OCR供定位误判。
    """

    frame: Frame
    scene: str
    buttons: tuple[FlowButton, ...] = ()
    texts: tuple[str, ...] = ()
    blocked_reason: str = ""
    mode_verified: bool = False
    feedbacks: tuple[str, ...] = ()
    # 叫抢输入只能使用本帧完整手牌；不能拼接之后翻开的底牌或最新其他帧的识别。
    bid_hand: tuple[str, ...] | None = None
    # 公开只读叫抢字样用于审计；评分网络本身只接受手牌，未知座位不能强行归属。
    bid_markers: tuple[dict, ...] = ()
    bid_deal_id: str = ""


def compact(text: str) -> str:
    """移除空白和常见标点，保留中文语义；不纠正相似字符，不做模糊按钮点击。"""
    return re.sub(r"[\s·：:，,。！!、_\-—]", "", text)


def _lobby_mode_verified(text: str, room: str) -> bool:
    """核对同帧 OCR 拼接文本中的完整洗牌场次标题，返回是否符合场次设置。

    “洗牌”与“无法进入初级场”等无关描述不能拼成授权；多个不同的标题视为冲突，
    不洗牌模式也不在支持范围。调用者仅合并本次扫描的文字，本函数不访问设备。
    """
    detected = {r for r in FLOW_ROOMS if f"洗牌{r}场" in text}
    return "不洗牌" not in text and len(detected) == 1 and (room == "当前" or room in detected)


def select_flow_source(snapshot, decoded: Frame | None, *, now=None):
    """选择页面识别的真实源帧，绝不将旧牌面观察重新绑定到较新的图像。

    正常牌桌使用同帧分析结果，叫抢可取得该帧17张手牌。结算后或分析落后超过600ms
    时改用较新的解码帧，并清空牌面输入；ended/deal仅作为最后可信状态的扫描提示。
    原started、媒体时间及帧标识原样保留，点击仍须经过1秒时效检查。换设备/视频/方向
    时不携带旧状态提示。now仅供测试注入，不用于生成或刷新任何Frame的时间戳。
    """
    if snapshot is None:
        return decoded, None, False, ""
    frame, state = snapshot.frame, snapshot.state
    if decoded is not None:

        def identity(f):
            return f.serial, f.stream_id, f.width, f.height

        if identity(frame) != identity(decoded):
            return decoded, None, False, ""
        age = (monotonic() if now is None else now) - frame.started
        if decoded.started > frame.started and (state["ended"] or age > 0.6):
            return decoded, None, bool(state["ended"]), state["game_id"]
    return frame, snapshot.observation, bool(state["ended"]), state["game_id"]


class FlowRecognizer:
    """固定PP-OCRv5读取页面文字，用区域和按钮色块交叉验证；每个流程线程独占实例。"""

    def __init__(self, root):
        self.engine = make_ocr(root)

    def _text(self, rgb, offset=(0, 0)):
        result = self.engine(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), use_det=True, use_cls=False)
        if result.boxes is None:
            return []
        found = []
        for polygon, text, score in zip(result.boxes, result.txts, result.scores):
            left, top = polygon.min(axis=0)
            right, bottom = polygon.max(axis=0)
            box = PixelRect(
                int(left) + offset[0],
                int(top) + offset[1],
                max(1, int(right - left)),
                max(1, int(bottom - top)),
            )
            found.append((compact(text), float(score), box))
        return found

    def _words(self, frame, phase, room):
        """返回带原图坐标的 OCR 候选，仅扫描按钮、场次和页面标题。

        room 使用与配置相同的场次范围；右下角已确认洗牌场次与按钮时立即返回，
        避免高级大厅继续扫描昵称、角色和排行榜而让输入证据过期。不发送设备输入。
        """

        def crop(x, y, w, h):
            left, top = round(x * frame.width), round(y * frame.height)
            return self._text(
                frame.rgb[top : round((y + h) * frame.height), left : round((x + w) * frame.width)],
                (left, top),
            )

        words = list(crop(0.64, 0.70, 0.35, 0.29))
        labels = "".join(t for t, _, _ in words)
        footer_labels = {t for t, score, box in words if score >= 0.95 and self._colored(frame, box)}
        if {"继续", "分享"} <= footer_labels or ("退出" in footer_labels and "继续游戏" in footer_labels):
            # 大幅连胜海报的中部大字识别可超过1秒；底部两个有色按钮已足以确认页面，
            # 不再扫描角色和海报。省下耗时，使新帧仍满足输入时效，而非放宽旧帧限制。
            return words
        if getattr(self, "_ending_scan", False):
            # 已确认本局结束、结算按钮尚未出现：只观察底部，不扫描胜利动画的大字。
            # 空白不是按钮消失或阶段推进证据；流程层收到没有输入对象的独立ending状态。
            return words
        footer_button = any(
            t in ("开始匹配", "继续游戏", "再来一局", "退出", "返回大厅")
            and score >= 0.85
            and self._colored(frame, box)
            for t, score, box in words
        )
        if footer_button and (_lobby_mode_verified(labels, room) or phase == "settlement"):
            return words
        words += crop(0.25, 0.4, 0.5, 0.37)
        if phase in ("bidding", "dealing", "unknown"):
            words += crop(0.18, 0.16, 0.67, 0.30)
        words += crop(0.06, 0, 0.19, 0.13)
        words += crop(0.86, 0, 0.13, 0.13)
        # 相邻裁剪的边界可能重复命中文字；同位置同文字只保留一次。
        return list({(t, b.x, b.y, b.width, b.height): (t, s, b) for t, s, b in words}.values())

    @staticmethod
    def _colored(frame, box):
        """文字周边须有蓝青或金橙按钮底色；灰色禁用、白色聊天泡和纯文字不算按钮。"""
        pad = max(4, round(box.height * 0.18))
        x, y = max(0, box.x - pad), max(0, box.y - pad)
        crop = frame.rgb[
            y : min(frame.height, box.y + box.height + pad), x : min(frame.width, box.x + box.width + pad)
        ]
        r, g, b = np.moveaxis(crop.astype(np.int16), -1, 0)
        cyan = (r > 80) & (g > 175) & (b > 185) & (b - r > 10)
        gold = (r > 165) & (g > 95) & (r - b > 55) & (g - b > 20)
        color = cyan | gold
        # 金色“不抢”反馈本身也是彩色文字，必须核对文字框外的连续按钮底色，
        # 防止把已经点击成功后的只读提示误判成按钮仍未消失。
        outside = np.ones(color.shape, dtype=bool)
        bx, by = box.x - x, box.y - y
        outside[by : by + box.height, bx : bx + box.width] = False
        return bool(outside.any() and np.mean(color[outside]) > 0.45)

    def observe(self, frame, table=None, *, room="初级", deal_id="", ended=False):
        """只读识别当前源帧，返回与 frame 绑定的 FlowObservation，不发送设备输入。

        table 仅在 frame_id 相同时提供阶段和手牌证据；deal_id、ended 只提示结算扫描。
        room 接受初、中、高级或“当前”，明确选择时只能匹配该场次；“当前”沿用
        FLOW_ROOMS 中的洗牌场次。切换弹窗必须明确写出目标且符合选择，否则返回
        带中文阻断原因的观察。按钮仍需独立通过文字分数、区域及底色校验。
        实例由流程识别线程独占；出牌阶段仅检查托管区域，OCR 初始化或推理异常上抛。
        """
        table = table or {}
        if not hasattr(self, "_known_end_deals"):
            self._known_end_deals, self._end_latched = set(), False
        if ended and deal_id not in self._known_end_deals:
            self._known_end_deals.add(deal_id)
            self._end_latched = True
        if deal_id != getattr(self, "_ending_deal", None):
            self._ending_deal = deal_id
            if not self._end_latched:
                self._last_footer_scene = ""
        # 正式结算按钮出现后可能进入大厅或匹配，恢复普通扫描。展示页关闭后的动画
        # 仍只等待正式结算，不把短暂空白伪装为页面命令已经成功。
        # 动画中的残留牌可能暂时触发跟踪器的下一局候选；该候选不能撤销已经确认的
        # 结算。只由真实大厅/匹配/叫抢页面解除等待；不以此生成或修订任何牌局事件。
        self._ending_scan = bool(
            self._end_latched and getattr(self, "_last_footer_scene", "") != "settlement"
        )
        phase = table.get("phase", "unknown") if table.get("frame_id") == frame.frame_id else "unknown"
        if table.get("frame_id") == frame.frame_id and any(
            table.get("remaining", {}).get(seat) == 0
            and table.get("marker_evidence", {}).get(f"winner_{seat}", {}).get("accepted")
            for seat in ("self", "left", "right")
        ):
            # 胜负大字动画可能遮住阶段模板；同帧“最后出牌”与零剩牌只确认页面结束，
            # 完整牌局仍由跟踪器和动作账本交叉核对，不能凭页面将末手补成成功。
            phase = "settlement"
        if phase == "playing" and not self._ending_scan:
            x, y = int(frame.width * 0.82), 0
            text = self._text(frame.rgb[: int(frame.height * 0.14), x:], (x, y))
            blocked = "检测到游戏内托管，不能计为机器人出牌" if any("取消托管" in t[0] for t in text) else ""
            return FlowObservation(frame, "playing", texts=tuple(t[0] for t in text), blocked_reason=blocked)
        # 背面底牌在叫抢期间持续存在；不能因此跳过不叫/不抢按钮的OCR。
        words = self._words(frame, phase, room)
        texts = tuple(t[0] for t in words)
        joined = "|".join(texts)
        # 健康系统会在牌局中覆盖牌桌并强制退出。它不是普通动画，也不能沿用旧牌桌
        # 状态推断为可以继续操作；保留文字证据，让流程在无任何点击的情况下停下。
        if any(w in joined for w in ("健康系统", "呵护双眼", "请您休息一下", "暂时无法登录", "无法登陆游戏")):
            return FlowObservation(
                frame,
                "unknown",
                texts=texts,
                blocked_reason="游戏健康系统要求休息，连续任务停止，不能继续本局",
            )
        blocked = next(
            (
                t
                for t in texts
                if any(
                    w in t
                    for w in (
                        "豆不足",
                        "欢乐豆不足",
                        "充值",
                        "实名认证",
                        "重新登录",
                        "网络连接异常",
                        "购买并",
                        "补充欢乐豆",
                    )
                )
            ),
            "",
        )
        if blocked:
            return FlowObservation(
                frame, "unknown", texts=texts, blocked_reason=f"需要人工处理页面：{blocked}"
            )
        confirmation_text = "".join(texts)
        mode_verified = _lobby_mode_verified(confirmation_text, room)
        # OCR 可能把提示分成多行；只拼接同帧文本，不从“无法进入初级场”等来源描述
        # 猜目标。每条“已为您切换”都必须跟随明确的“至/到某场”，且目标一致。
        room_confirmation = "已为您切换" in confirmation_text
        if room_confirmation:
            targets = re.findall(r"已为您切换(?:至|到)(?:洗牌)?([\u4e00-\u9fff]{1,8}?)场", confirmation_text)
            if (
                len(targets) != confirmation_text.count("已为您切换")
                or len(set(targets)) != 1
                or targets[0] not in FLOW_ROOMS
            ):
                return FlowObservation(
                    frame, "unknown", texts=texts, blocked_reason="无法确认切换后的受支持洗牌场次，请人工确认"
                )
            approved = FLOW_ROOMS if room == "当前" else (room,)
            if targets[0] not in approved:
                return FlowObservation(
                    frame,
                    "unknown",
                    texts=texts,
                    blocked_reason=f"游戏要求切换{targets[0]}场，当前选择为{room}场，不能自动确认",
                )
        matching = any(
            w in joined for w in ("匹配中", "正在匹配", "取消匹配", "寻找玩家", "对局开启中", "对局开启")
        )
        raw = []
        for label, score, box in words:
            kind = LABELS.get(label)
            if kind is None or score < 0.85:
                continue
            cx, cy = (box.x + box.width / 2) / frame.width, (box.y + box.height / 2) / frame.height
            in_region = (
                (kind == "match" and 0.6 < cx < 1 and 0.7 < cy < 1)
                or (
                    kind in ("decline_bid", "call_landlord", "rob_landlord", "decline_double")
                    and 0.12 < cx < 0.85
                    and 0.4 < cy < 0.75
                )
                or (kind in ("continue", "return_lobby") and 0.15 < cx < 0.98 and 0.65 < cy < 0.99)
                or (kind == "confirm_room" and room_confirmation and 0.3 < cx < 0.7 and 0.6 < cy < 0.8)
            )
            if in_region and self._colored(frame, box):
                raw.append(FlowButton(kind, label, box, True, score))
        kinds = {b.kind for b in raw}
        feedbacks = tuple(
            {
                LABELS[t]
                for t, score, box in words
                if t in LABELS
                and LABELS[t] in ("decline_bid", "call_landlord", "rob_landlord", "decline_double")
                and score >= 0.85
                and 0.4 < (box.x + box.width / 2) / frame.width < 0.6
                and 0.52 < (box.y + box.height / 2) / frame.height < 0.70
                and box.height >= frame.height * 0.06
                and not self._colored(frame, box)
            }
        )
        # 进桌动画中先出现“斗地主”和右上“托管”，随后才发牌。它是可确认的牌桌
        # 外壳，不能要求已经有手牌才承认匹配已进入牌桌；不据此初始化任何牌张。
        board_shell = any(
            "斗地主" in t and box.y < frame.height * 0.12 and box.x < frame.width * 0.3 for t, _, box in words
        ) and any(
            t == "托管" and box.y < frame.height * 0.12 and box.x > frame.width * 0.82 for t, _, box in words
        )
        showcase = any(b.kind == "continue" and b.label == "继续" for b in raw) and any(
            text == "分享"
            and score >= 0.95
            and box.x > frame.width * 0.65
            and box.y > frame.height * 0.8
            and self._colored(frame, box)
            for text, score, box in words
        )
        if showcase:
            # 连胜等战绩展示的“继续”只关闭展示，随后仍是结算页；与下一局按钮分开记账。
            scene = "showcase"
            raw = [
                FlowButton("close_showcase", b.label, b.hit, b.enabled, b.score)
                for b in raw
                if b.kind == "continue" and b.label == "继续"
            ]
        elif room_confirmation:
            scene = "room_confirmation"
        elif matching:
            scene = "matching"
        elif "match" in kinds and mode_verified:
            scene = "lobby"
        elif kinds & {"decline_bid", "call_landlord", "rob_landlord"} and phase in (
            "bidding",
            "unknown",
            "dealing",
        ):
            scene = "bidding"
        elif "decline_double" in kinds and phase in ("doubling", "unknown"):
            scene = "doubling"
        elif "decline_double" in feedbacks and phase in ("doubling", "unknown", "dealing"):
            scene = "doubling"
        elif set(feedbacks) & {"decline_bid", "call_landlord", "rob_landlord"} and phase in (
            "bidding",
            "unknown",
            "dealing",
        ):
            scene = "bidding"
        elif self._ending_scan and not (kinds & {"continue", "return_lobby"}):
            scene = "ending"
        elif (
            phase == "settlement"
            or {"return_lobby", "continue"} <= kinds
            and any(b.label == "继续游戏" for b in raw)
            or kinds & {"continue", "return_lobby"}
            and any(t in joined for t in ("胜利", "失败", "本局", "结算", "再来一局"))
        ):
            scene = "settlement"
        else:
            scene = (
                phase
                if phase in ("dealing", "bidding", "doubling")
                else ("dealing" if board_shell else "unknown")
            )
        # 页面已知但文字冲突时不开放按钮；尤其不能把叫抢的“不抢”转成“不出”。
        allowed = {
            "lobby": {"match"},
            "room": {"match"},
            "bidding": {"decline_bid", "call_landlord", "rob_landlord"},
            "doubling": {"decline_double"},
            "settlement": {"continue", "return_lobby"},
            "showcase": {"close_showcase"},
            "room_confirmation": {"confirm_room"},
        }
        buttons = tuple(b for b in raw if b.kind in allowed.get(scene, set()))
        if scene in ("settlement", "showcase") and buttons:
            self._last_footer_scene = scene
        if scene in ("lobby", "room", "matching", "bidding"):
            self._end_latched = False
            self._last_footer_scene = ""
        # 单字叫/抢只作为按钮候选；公开字样要求完整文字、较高分数及已知座位区域。
        markers = []
        if scene == "bidding":
            for text, score, box in words:
                if (
                    text not in ("叫地主", "抢地主", "不叫", "不抢", "不叫地主", "不抢地主")
                    or score < 0.98
                    or self._colored(frame, box)
                ):
                    continue
                cx, cy = (box.x + box.width / 2) / frame.width, (box.y + box.height / 2) / frame.height
                seat = (
                    "left"
                    if 0.18 < cx < 0.46 and 0.16 < cy < 0.46
                    else "right"
                    if 0.58 < cx < 0.85 and 0.16 < cy < 0.46
                    else "self"
                    if 0.3 < cx < 0.7 and 0.50 < cy < 0.72
                    else None
                )
                if seat:
                    markers.append(
                        {
                            "seat": seat,
                            "label": text,
                            "score": score,
                            "box": [box.x, box.y, box.width, box.height],
                        }
                    )
        hand = table.get("hand") if table.get("frame_id") == frame.frame_id and scene == "bidding" else None
        return FlowObservation(
            frame,
            scene,
            buttons,
            texts,
            mode_verified=mode_verified,
            feedbacks=feedbacks,
            bid_hand=tuple(hand) if hand is not None else None,
            bid_markers=tuple(markers),
            bid_deal_id=deal_id,
        )
