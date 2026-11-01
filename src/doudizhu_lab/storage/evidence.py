"""事件证据截图的有界后台写入；保存迟缓不能阻塞牌桌观察，也不能静默漏报。"""

from __future__ import annotations

import json
import queue
import shutil
import threading
from copy import deepcopy
from pathlib import Path

from doudizhu_lab.domain.models import Frame
from doudizhu_lab.storage.config import write_rgb


class EvidenceWriter:
    """最多持有八张原图，繁忙时只保留源时间引用；录像收尾后可补生成截图。

    submit 由识别线程调用，stop 只在识别线程结束后调用。每个索引条目记录真实保存
    状态和实际媒体时间，失败的截图不会标为成功；视频完整性由录像层独立判断。
    """

    def __init__(self, folder: Path, *, capacity=8, on_error=None):
        self.folder = folder
        self.folder.mkdir(parents=True, exist_ok=True)
        self.queue = queue.Queue(maxsize=capacity)
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.entries = {}
        self.on_error = on_error
        self.maximum_queue = 0
        self.thread = None

    def start(self):
        """启动唯一 PNG 写入线程；线程只读取 Frame，不修改识别或牌局状态。"""
        self.thread = threading.Thread(target=self._run, name="证据截图写盘", daemon=True)
        self.thread.start()

    def submit(self, frame: Frame, event_ids: list[str]) -> bool:
        """非阻塞提交；返回 False 表示已登记延后保存，不代表完整录像丢失。"""
        name = frame.frame_id or frame.captured_at
        if Path(name).name != name or any(c in name for c in '<>:"/\\|?*'):
            raise ValueError("证据帧标识不能用于文件名")
        with self.lock:
            if name in self.entries:
                self.entries[name]["event_ids"] = sorted(set(self.entries[name]["event_ids"] + event_ids))
                return self.entries[name]["status"] in ("queued", "saved", "rebuilt")
            self.entries[name] = {
                "frame_id": name,
                "pts_us": frame.media_pts_us,
                "event_ids": event_ids,
                "status": "queued",
                "actual_pts_us": None,
                "error": "",
            }
        try:
            if self.stop_event.is_set():
                raise queue.Full
            self.queue.put_nowait(frame)
            self.maximum_queue = max(self.maximum_queue, self.queue.qsize())
            return True
        except queue.Full:
            with self.lock:
                self.entries[name]["status"] = "deferred"
            return False

    def _save(self):
        with self.lock:
            value = deepcopy(self.entries)
        path = self.folder / "index.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(path)

    def _write(self, name, frame, status):
        try:
            if shutil.disk_usage(self.folder).free < 2 * 1024**3:
                raise OSError("剩余磁盘空间不足 2 GiB，证据截图延后保存")
            write_rgb(self.folder / f"{name}.png", frame.rgb)
            with self.lock:
                self.entries[name].update(status=status, actual_pts_us=frame.media_pts_us, error="")
        except Exception as exc:
            with self.lock:
                self.entries[name].update(status="failed", error=str(exc))
            if self.on_error:
                self.on_error(name, str(exc))

    def _run(self):
        while not self.stop_event.is_set() or not self.queue.empty():
            try:
                frame = self.queue.get(timeout=0.2)
            except queue.Empty:
                continue
            self._write(frame.frame_id or frame.captured_at, frame, "saved")
            try:
                self._save()
            except OSError as exc:
                if self.on_error:
                    self.on_error("", f"证据索引写入失败：{exc}")

    def stop(self, *, rebuild=True) -> bool:
        """等待已提交截图收尾，再从已关闭录像补采延后项；返回线程是否已完全结束。"""
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=5)
            if self.thread.is_alive():
                return False
        if rebuild and any(i["status"] in ("deferred", "failed") for i in self.entries.values()):
            from doudizhu_lab.storage.replay import VideoReplay

            try:
                video = VideoReplay(self.folder.parent)
                for name, item in list(self.entries.items()):
                    if item["status"] not in ("deferred", "failed") or item["pts_us"] is None:
                        continue
                    if shutil.disk_usage(self.folder).free < 2 * 1024**3:
                        break
                    try:
                        before = video.at(item["pts_us"])
                        after = video.at(item["pts_us"], after=True)
                        frame = min((before, after), key=lambda f: abs(f.media_pts_us - item["pts_us"]))
                        if abs(frame.media_pts_us - item["pts_us"]) > 2_000:
                            raise ValueError("录像中没有该源帧，保留缺口而不拿邻帧替代")
                        self._write(name, frame, "rebuilt")
                    except Exception as exc:
                        item.update(status="failed", error=str(exc))
            except (OSError, ValueError, KeyError) as exc:
                if self.on_error:
                    self.on_error("", f"证据补采未完成：{exc}")
        self._save()
        return True

    def statistics(self) -> dict:
        """返回保存数量和队列峰值，供会话报告区分实时保存、延后补采与失败。"""
        with self.lock:
            values = list(self.entries.values())
        return {
            "maximum_queue": self.maximum_queue,
            "pending": self.queue.qsize(),
            **{
                name: sum(i["status"] == name for i in values)
                for name in ("saved", "rebuilt", "deferred", "failed")
            },
        }
