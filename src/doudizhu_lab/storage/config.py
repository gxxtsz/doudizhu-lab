"""本地校准、模板与导出文件管理；所有运行数据留在被 Git 忽略的 data 目录。"""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np

from doudizhu_lab.devices.adb import discover_adb
from doudizhu_lab.domain.models import RANKS, Frame, NormalizedRect, Recognition, TrackerSettings
from doudizhu_lab.domain.settings import CONFIGURATION_VERSION, default_bid_settings, default_regions
from doudizhu_lab.domain.settings import Profile as Profile
from doudizhu_lab.domain.settings import validate_bot_profile as validate_bot_profile
from doudizhu_lab.paths import resolve_data_dir


def read_rgb(path: Path) -> np.ndarray:
    """使用字节读取图片，兼容 Windows 中文路径；解码失败抛出明确异常。"""
    data = path.read_bytes()
    if not data:
        raise ValueError(f"图片文件为空：{path.name}")
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"图片无法解码：{path.name}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def write_rgb(path: Path, rgb: np.ndarray) -> None:
    """将 RGB 图像保存为 PNG，不依赖 OpenCV 对中文路径的支持。"""
    ok, data = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise ValueError("图片编码失败")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data.tobytes())


class Store:
    """管理用户配置，损坏配置不静默覆盖；调用者收到错误后可以选择重置。"""

    def __init__(self, root: Path | None = None):
        self.root = resolve_data_dir(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.template_dir = self.root / "templates"
        self.template_dir.mkdir(exist_ok=True)
        self.settings_path = self.root / "settings.json"
        self.profile = Profile()
        self.adb_path = discover_adb()
        self.settings_version = CONFIGURATION_VERSION
        self._bidding_upgrade = False
        if self.settings_path.exists():
            try:
                raw = json.loads(self.settings_path.read_text(encoding="utf-8"))
                self.adb_path = str(raw.get("adb_path", self.adb_path))
                p = raw.get("profile", {})
                self._bidding_upgrade = "bid_settings" not in p
                self.settings_version = int(raw.get("version", 1))
                regions = {k: NormalizedRect(**v) for k, v in p.get("regions", {}).items()}
                if self.settings_version == 1 and regions:
                    regions.setdefault("self_play", default_regions()["self_play"])
                heights = dict(p.get("glyph_heights", self.profile.glyph_heights))
                if self.settings_version == 1:
                    heights.setdefault("self_play", 50 * int(p.get("base_height", 1080)) / 1080)
                self.profile = Profile(
                    name=p.get("name", self.profile.name),
                    base_width=int(p.get("base_width", 2400)),
                    base_height=int(p.get("base_height", 1080)),
                    regions=regions or default_regions(),
                    glyph_heights=heights,
                    threshold=float(p.get("threshold", 0.84)),
                    margin=float(p.get("margin", 0.055)),
                    hand_classifier=p.get("hand_classifier", "template"),
                    play_model=p.get("play_model", "resnet"),
                    continuous_settings=p.get("continuous_settings", self.profile.continuous_settings),
                    flow_room=p.get("flow_room", "初级"),
                    continuous_policy=p.get("continuous_policy", "strict"),
                    bid_settings=p.get("bid_settings", default_bid_settings()),
                    indicator_regions={
                        k: NormalizedRect(**v) for k, v in p.get("indicator_regions", {}).items()
                    }
                    or self.profile.indicator_regions,
                    seat_order=p.get("seat_order", []),
                    tracker_settings=p.get("tracker_settings", asdict(TrackerSettings())),
                    bot_controls=p.get("bot_controls", self.profile.bot_controls),
                    bot_settings=p.get("bot_settings", self.profile.bot_settings),
                )
                TrackerSettings(**self.profile.tracker_settings)
                validate_bot_profile(self.profile)
                if (
                    set(self.profile.regions) != set(default_regions())
                    or set(self.profile.glyph_heights) != set(default_regions())
                    or not all(
                        isinstance(v, (int, float)) and np.isfinite(v) and v > 0
                        for v in self.profile.glyph_heights.values()
                    )
                    or not (0 < self.profile.threshold <= 1 and 0 <= self.profile.margin < 1)
                    or min(self.profile.base_width, self.profile.base_height) <= 0
                    or (
                        self.profile.seat_order
                        and sorted(self.profile.seat_order) != ["left", "right", "self"]
                    )
                ):
                    raise ValueError("配置字段超出支持范围")
            except (OSError, TypeError, ValueError, KeyError, AttributeError) as exc:
                raise ValueError(f"配置文件损坏，请备份后修复或移走：{self.settings_path}") from exc

    def save(self) -> None:
        """以同目录临时文件原子替换配置，避免程序退出时留下半个 JSON 文件。"""
        validate_bot_profile(self.profile)
        if self._bidding_upgrade and self.settings_path.is_file():
            backup = self.root / "settings.before-bidding.bak.json"
            if not backup.exists():
                shutil.copy2(self.settings_path, backup)
            self._bidding_upgrade = False
        target = self.settings_path.with_suffix(".tmp")
        if self.settings_path.exists() and self.settings_version < CONFIGURATION_VERSION:
            backup = self.settings_path.with_name(f"settings.v{self.settings_version}.bak.json")
            if not backup.exists():
                shutil.copy2(self.settings_path, backup)
        target.write_text(
            json.dumps(
                {
                    "version": CONFIGURATION_VERSION,
                    "adb_path": self.adb_path,
                    "profile": asdict(self.profile),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        target.replace(self.settings_path)
        self.settings_version = CONFIGURATION_VERSION

    def add_template(self, rgb: np.ndarray, rank: str, group: str, source: str | dict) -> Path:
        """保存人工确认的牌点裁剪和标签；group 分为 hand、bottom、table。

        普通牌请只框选牌点字符，10 必须框住两个字符，大小王需框住完整竖排 JOKER。
        source 兼容旧帧名称；第二版使用字典记录会话、牌局、媒体时间戳、原图哈希及裁剪框，
        以便隔离模板与验证牌局，并从录像追溯实际字形。
        """
        if rank not in RANKS or group not in ("hand", "bottom", "table"):
            raise ValueError("模板标签或分组无效")
        if rgb.ndim != 3 or min(rgb.shape[:2]) < 3:
            raise ValueError("模板裁剪太小")
        path = self.template_dir / f"{group}_{rank}_{uuid4().hex[:10]}.png"
        write_rgb(path, rgb)
        path.with_suffix(".json").write_text(
            json.dumps({"rank": rank, "group": group, "source": source}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return path

    def export(self, frame: Frame, recognition: Recognition | None) -> Path:
        """导出同一帧截图和识别 JSON；拒绝把其他帧的识别结果写到当前截图旁。"""
        if recognition is not None and recognition.frame_time != frame.captured_at:
            raise ValueError("识别结果与截图不属于同一帧")
        folder = self.root / "exports" / (datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid4().hex[:6])
        folder.mkdir(parents=True)
        write_rgb(folder / "画面.png", frame.rgb)
        payload = {
            "version": 1,
            "device": frame.serial,
            "captured_at": frame.captured_at,
            "width": frame.width,
            "height": frame.height,
            "capture_ms": frame.capture_ms,
            "transport": frame.transport,
            "profile": asdict(self.profile),
            "recognition": asdict(recognition) if recognition else None,
        }
        (folder / "识别结果.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return folder
