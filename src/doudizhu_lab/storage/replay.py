"""离线录像读取、独立重算与事件评测；本模块没有设备输入能力。"""

from __future__ import annotations

import json
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import local
from time import monotonic

import av

from doudizhu_lab.domain.models import Frame, NormalizedRect
from doudizhu_lab.domain.tracker import GameTracker
from doudizhu_lab.storage.config import Profile
from doudizhu_lab.storage.session import SessionStore
from doudizhu_lab.vision.factory import create_recognizer
from doudizhu_lab.vision.table import MarkerLibrary, TableRecognizer


def profile_from_dict(value: dict) -> Profile:
    """恢复会话内的校准快照；补充旧版本不存在的区域，不读取当前用户配置。"""
    fields = dict(value)
    for name in ("regions", "indicator_regions"):
        defaults = getattr(Profile(), name)
        fields[name] = defaults | {k: NormalizedRect(**v) for k, v in fields.get(name, {}).items()}
    fields["glyph_heights"] = Profile().glyph_heights | fields.get("glyph_heights", {})
    return Profile(**fields)


class VideoReplay:
    """按源媒体时间定位 MKV；缺口不复用上一张图像，不把回放帧包装为实时手机帧。"""

    def __init__(self, folder: Path):
        self.folder = folder.resolve()
        with_session = (self.folder / "session.sqlite3").exists()
        if with_session:
            session = SessionStore(self.folder, readonly=True)
            metadata = session.metadata()
            session.close()
            source = metadata.get("video_source")
            if source:
                self.folder = (self.folder / source).resolve()
        self.video = self.folder / "video"
        self.segments = json.loads((self.video / "index.json").read_text(encoding="utf-8"))
        for segment in self.segments:
            if Path(segment["file"]).name != segment["file"]:
                raise ValueError("录像索引含非法文件路径")
        self.start_us = min((s["global_start_us"] for s in self.segments), default=0)
        self.end_us = max((s["global_end_us"] for s in self.segments), default=0)

    def _frame(self, raw, segment) -> Frame:
        pts = segment["global_start_us"] + int(raw.pts * raw.time_base * 1_000_000)
        identity = f"{segment['file']}@{pts}"
        return Frame(
            "replay",
            raw.to_ndarray(format="rgb24"),
            monotonic(),
            identity,
            0,
            "replay",
            identity,
            self.folder.name,
            pts,
        )

    def at(self, pts_us: int, *, after: bool = False) -> Frame:
        """读取指定时间的帧；after 用于逐帧向后，普通定位返回不晚于目标的最近帧。

        MKV 内部时间相对分段首帧，接口始终使用 scrcpy 源时间。落在缺口抛出异常。
        """
        segment = next(
            (s for s in self.segments if s["global_start_us"] <= pts_us <= s["global_end_us"] + 1000), None
        )
        if segment is None:
            # 相邻关键帧分段之间的正常帧间隔不属于录像缺口。
            segment = next(
                (
                    a
                    for a, b in zip(self.segments, self.segments[1:])
                    if a["global_end_us"] < pts_us < b["global_start_us"]
                    and b["global_start_us"] - a["global_end_us"] <= 250_000
                ),
                None,
            )
            if segment is None:
                retention = self.folder / "留存记录.json"
                if retention.exists():
                    value = json.loads(retention.read_text(encoding="utf-8"))
                    if any(s["start_us"] <= pts_us <= s["end_us"] for s in value.get("removed_segments", [])):
                        raise ValueError("该时间录像已按正常完成或主动中止策略清理")
                raise ValueError("该时间没有录像，可能处于传输或前台切换缺口")
        with av.open(str(self.video / segment["file"])) as container:
            stream = container.streams.video[0]
            stream.thread_count = 2
            relative = max(0, pts_us - segment["global_start_us"])
            offset = int(relative / 1_000_000 / stream.time_base)
            container.seek(offset, stream=stream, backward=True)
            previous = None
            for raw in container.decode(stream):
                timestamp = segment["global_start_us"] + int(raw.pts * raw.time_base * 1_000_000)
                if timestamp >= pts_us if after else timestamp > pts_us:
                    return self._frame(raw if after or previous is None else previous, segment)
                previous = raw
            if previous is not None:
                next_index = self.segments.index(segment) + 1
                if after and next_index < len(self.segments):
                    next_segment = self.segments[next_index]
                    if next_segment["global_start_us"] - segment["global_end_us"] <= 250_000:
                        return self.at(next_segment["global_start_us"], after=True)
                return self._frame(previous, segment)
        raise ValueError("片段无法解码或已截断")

    def frames(
        self, *, fps: float = 10, closed_only: bool = True, start_us=None, end_us=None, minimum_spacing=False
    ):
        """顺序解码并按媒体时钟采样；fps=0 返回每个解码帧，不会改变完整录像。

        可选 start_us/end_us 是包含边界的源媒体时间，供逐段标注和问题复查使用。
        从起点之前的关键帧解码，但只返回区间内的帧；缺口不会用相邻片段填充。
        minimum_spacing沿用连续任务的最小采样间隔，使相同录像重算遵守同一证据时间门槛。
        """
        if start_us is not None and end_us is not None and start_us > end_us:
            raise ValueError("回放起始时间不能晚于结束时间")
        interval = int(1_000_000 / fps) if fps > 0 else 0
        next_pts = 0
        for segment in self.segments:
            if closed_only and not segment["closed"]:
                continue
            if start_us is not None and segment["global_end_us"] < start_us:
                continue
            if end_us is not None and segment["global_start_us"] > end_us:
                break
            with av.open(str(self.video / segment["file"])) as container:
                stream = container.streams.video[0]
                stream.thread_count = 2
                if start_us is not None and start_us > segment["global_start_us"]:
                    relative = (start_us - segment["global_start_us"]) / 1_000_000
                    container.seek(int(relative / stream.time_base), stream=stream, backward=True)
                for raw in container.decode(stream):
                    pts = segment["global_start_us"] + int(raw.pts * raw.time_base * 1_000_000)
                    if start_us is not None and pts < start_us:
                        continue
                    if end_us is not None and pts > end_us:
                        break
                    if pts >= next_pts:
                        yield self._frame(raw, segment)
                        next_pts = (
                            pts + interval
                            if not next_pts or not interval or minimum_spacing
                            else next_pts + ((pts - next_pts) // interval + 1) * interval
                        )

    def audit(self) -> dict:
        """逐帧解码审计录像完整性，不重写索引或补造丢失帧。

        相同文件哈希的审计可复用。未关闭片段、解码异常、解码帧数或末尾时间
        与索引不一致均为失败；只成功打开容器不能证明录像没有截断。
        """
        from doudizhu_lab.storage.truth import media_manifest

        results = []
        for segment in self.segments:
            count, last, error = 0, None, ""
            try:
                with av.open(str(self.video / segment["file"])) as container:
                    stream = container.streams.video[0]
                    stream.thread_count = 2
                    for raw in container.decode(stream):
                        count += 1
                        last = segment["global_start_us"] + int(raw.pts * raw.time_base * 1_000_000)
            except Exception as exc:
                error = f"录像解码失败：{exc}"
            if not segment.get("closed"):
                error = error or "片段未正常收尾"
            if count != segment.get("frames") or last is None or abs(last - segment["global_end_us"]) > 2_000:
                error = error or "解码帧数或末尾时间与索引不一致，可能发生截断"
            results.append(
                {
                    "file": segment["file"],
                    "decoded_frames": count,
                    "last_us": last,
                    "passed": not error,
                    "error": error,
                }
            )
        try:
            manifest = media_manifest(self.folder)
        except (OSError, ValueError) as exc:
            manifest = []
            results.append({"passed": False, "error": str(exc)})
        return {
            "version": 1,
            "passed": bool(results) and all(r["passed"] for r in results),
            "segments": results,
            "video_manifest": manifest,
        }


def reanalyze(
    folder: Path,
    *,
    template_root: Path | None = None,
    fps: float = 10,
    progress=None,
    cancel=None,
    workers: int = 2,
) -> Path:
    """创建独立分析版本，保留原观察、原事件及所有人工修订。

    默认使用录制时的配置和模板。显式 template_root 才采用补采后的模板，且复制到新版本。
    progress 接收已分析帧数，cancel 为 threading.Event；取消的版本明确标为未完成。
    workers 仅并行计算独立帧，结果仍按源时间顺序进入状态机；内存队列最多为线程数两倍。
    """
    if not 1 <= workers <= 4:
        raise ValueError("离线分析线程数必须在 1～4 之间")
    source = SessionStore(folder)
    meta = source.metadata()
    try:
        source.verify_snapshot()
    except Exception:
        source.close()
        raise
    if template_root is not None:
        from doudizhu_lab.storage.config import Store

        profile = Store(template_root).profile
    else:
        profile = profile_from_dict(meta["profile"])
    templates = template_root or folder / "templates"
    child = SessionStore.create(
        folder / "analyses",
        profile,
        [templates / "templates", templates / "markers", templates / "controls"],
        "replay",
        model_root=template_root or folder,
    )
    import os

    video = VideoReplay(folder)
    child.set_meta("video_source", os.path.relpath(video.folder, child.folder))
    child.set_meta("source_session", meta.get("session_id"))
    child.set_meta("status", "analyzing")
    child.set_meta("analysis_fps", fps)
    child.set_meta("analysis_workers", workers)
    minimum_spacing = bool(meta.get("analysis_sampling", {}).get("minimum_spacing"))
    child.set_meta("analysis_sampling", {"fps_limit": fps, "minimum_spacing": minimum_spacing})
    child.set_meta("source_status", meta.get("status"))
    child.set_meta("source_engine_available", bool(meta.get("engine_manifest")))
    if meta.get("flow_enabled"):
        child.snapshot_vision_models(template_root or folder)
        child.set_meta("flow_enabled", True)
    child.set_meta("same_engine", meta.get("engine_sha256") == child.metadata().get("engine_sha256"))
    thread_data = local()

    def analyze_frame(frame):
        if not hasattr(thread_data, "recognizer"):
            from doudizhu_lab.vision.controls import ControlRecognizer

            thread_data.recognizer = TableRecognizer(
                create_recognizer(
                    child.folder / "templates/templates",
                    profile,
                    model_root=child.folder,
                    include_bundled=False,
                ),
                MarkerLibrary(child.folder / "templates/markers"),
            )
            thread_data.controls = ControlRecognizer(child.folder / "templates/controls")
            if meta.get("flow_enabled"):
                from doudizhu_lab.vision.last_card import LastCardOCR

                thread_data.last_card = LastCardOCR(child.folder)
        observation, recognition = thread_data.recognizer.observe(frame, profile)
        thread_data.controls.augment_observation(frame, observation, profile)
        if meta.get("flow_enabled"):
            thread_data.last_card.enrich(frame, observation, recognition, profile)
        return frame, observation

    def ordered_frames():
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="离线独立帧分析") as executor:
            pending = deque()
            for frame in video.frames(fps=fps, minimum_spacing=minimum_spacing):
                if cancel is not None and cancel.is_set():
                    break
                pending.append(executor.submit(analyze_frame, frame))
                if len(pending) >= workers * 2:
                    yield pending.popleft().result()
            while pending:
                yield pending.popleft().result()

    tracker = GameTracker(profile.seat_order, config=profile.tracker_settings)
    count = 0
    try:
        previous_pts = None
        for frame, observation in ordered_frames():
            if cancel is not None and cancel.is_set():
                child.set_meta("status", "cancelled")
                break
            if previous_pts is not None and frame.media_pts_us - previous_pts > 1_000_000:
                update = tracker.mark_gap(observation, "录像时间轴存在缺口", capture_lost=True)
                for event in update.events:
                    child.add_event(event)
            previous_pts = frame.media_pts_us
            child.add_observation(observation)
            update = tracker.observe(observation)
            for event in update.events:
                child.add_event(event)
            for issue in update.issues:
                child.add_issue(issue)
            count += 1
            if progress and count % 50 == 0:
                progress(count)
        else:
            child.set_meta("status", "analyzed")
        for issue in source.gap_intervals():
            child.add_issue(issue)
        child.set_meta(
            "source_analysis_gaps",
            [g for g in source.gap_intervals(include_analysis=True) if g["scope"] == "analysis"],
        )
        child.set_meta("analyzed_frames", count)
        child.export()
        return child.folder
    except Exception as exc:
        child.set_meta("status", "failed")
        child.add_issue({"code": "reanalysis", "message": str(exc), "pts_us": previous_pts or 0})
        child.export()
        raise
    finally:
        child.close()
        source.close()


def evaluate_session(folder: Path, labels: Path | None = None) -> dict:
    """兼容已有调用入口，统一使用按牌局和事件顺序的新版评测实现。"""
    from doudizhu_lab.evaluation.session import evaluate_session as evaluate

    return evaluate(folder, labels)
