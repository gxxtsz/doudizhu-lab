"""无人值守任务的异常媒体轮转；只删除本次登记且已经关闭的媒体文件。"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from doudizhu_lab.storage.json_io import atomic_json
from doudizhu_lab.storage.retention import checked_tree


class TaskMediaRotator:
    """看护器登记归属，单个后台归档线程轮转；遍历登记快照，删除决定先持久化再执行。"""

    def __init__(self, root, task_folder, policy, *, request_rotation=None):
        self.root, self.folder, self.policy = Path(root), Path(task_folder), policy
        self.sessions = {}
        self.removed = []
        self.request_rotation = request_rotation

    def register(self, folder: Path, *, closed: bool = False) -> None:
        """登记本任务会话及写入者退出事实；路径必须位于 sessions 直接子目录且无链接。"""
        folder = checked_tree(Path(folder), self.root / "sessions")
        self.sessions[folder] = closed or self.sessions.get(folder, False)

    def rotate(self) -> int:
        """轮转较旧异常窗口；活动片段、共享片段与历史文件不删，I/O 失败留待下次重试。

        调用者在后台运行；空间检查不是完整磁盘配额保证，仍须遵守写盘端 2 GiB 底线。
        返回删除的文件数量，删除记录在任务目录永久保留，不能当作真实采集缺口。
        """
        registered = self.sessions.copy()
        for record in self.removed:
            if record["status"] == "planned":
                folder = next((p for p in registered if p.name == record["session"]), None)
                if folder is None or not registered[folder]:
                    continue
                checked_tree(folder, self.root / "sessions")
                path = folder / record["file"]
                if path.parent not in (folder / "video", folder / "evidence"):
                    raise ValueError("待重试轮转文件路径越界")
                path.unlink(missing_ok=True)
                record["status"] = "removed"
                atomic_json(self.folder / "媒体轮转记录.json", {"removed": self.removed})
        groups, total = [], 0
        for folder, closed in registered.items():
            checked_tree(folder, self.root / "sessions")
            try:
                ledger = json.loads((folder / "留存记录.json").read_text(encoding="utf8"))
                segments = json.loads((folder / "video/index.json").read_text(encoding="utf8"))
            except FileNotFoundError:
                continue
            known = {s["file"] for s in segments}
            for entry in ledger.get("removed_segments", []):
                if (
                    entry.get("reason") == "task_media_rotation"
                    and entry.get("status") == "planned"
                    and entry["file"] not in known
                ):
                    segments.append(
                        {
                            "file": entry["file"],
                            "global_start_us": entry["start_us"],
                            "global_end_us": entry["end_us"],
                            "closed": True,
                        }
                    )
            windows = [w for w in ledger.get("windows", []) if w.get("anomalies")]
            if closed and (ledger.get("faults") or ledger.get("active", {}).get("anomalies")):
                final = dict(ledger["active"])
                if final.get("end_us") is None:
                    final["end_us"] = max(
                        [ledger.get("last_pts_us", 0), *(s["global_end_us"] for s in segments)]
                    )
                windows.append(final)
            active = ledger.get("active", {})
            try:
                evidence = json.loads((folder / "evidence/index.json").read_text(encoding="utf8"))
            except FileNotFoundError:
                evidence = {}
            for window in windows:
                end = window.get("end_us")
                if end is None:
                    continue
                owned = []
                for segment in segments:
                    if not closed and not segment.get("closed"):
                        continue
                    left, right = segment["global_start_us"], segment["global_end_us"]
                    if right < window["start_us"] or left > end:
                        continue
                    if not closed and right >= active.get("start_us", 0):
                        continue
                    # 同片段覆盖较新异常局时，归较新窗口管理，不能提前轮转掉其解码上下文。
                    if any(
                        w is not window and w["start_us"] > window["start_us"] and right >= w["start_us"]
                        for w in windows
                    ):
                        continue
                    name = segment["file"]
                    if Path(name).name != name:
                        raise ValueError("任务媒体片段路径越界")
                    path = folder / "video" / name
                    if path.is_file():
                        owned.append(path)
                for name, item in evidence.items():
                    pts = item.get("pts_us")
                    if pts is not None and window["start_us"] <= pts < end:
                        if Path(name).name != name:
                            raise ValueError("任务截图标识路径越界")
                        path = folder / "evidence" / f"{name}.png"
                        if path.is_file() and item.get("status") in ("saved", "rebuilt", "rotation_planned"):
                            owned.append(path)
                size = sum(p.stat().st_size for p in owned)
                if owned:
                    groups.append((folder, window, owned, size))
                    total += size
        removed = 0
        count = len(groups)
        for folder, window, files, size in groups:
            pressure = shutil.disk_usage(self.root).free < self.policy.cleanup_free_bytes
            if count <= self.policy.media_games and total <= self.policy.media_bytes and not pressure:
                break
            if not registered[folder]:
                # 活动会话的索引由录像进程独占；只能向所属写入者请求轮转，不能并发改索引。
                if self.request_rotation:
                    self.request_rotation(
                        folder,
                        [p.name for p in files if p.parent.name == "video"],
                        window["start_us"],
                        window["end_us"],
                    )
                total -= size
                count -= 1
                continue
            index_path = folder / "video/index.json"
            index = json.loads(index_path.read_text(encoding="utf8"))
            ledger_path = folder / "留存记录.json"
            ledger = json.loads(ledger_path.read_text(encoding="utf8"))
            names = {p.name for p in files}
            ledger.setdefault("removed_segments", []).extend(
                {
                    "file": s["file"],
                    "start_us": s["global_start_us"],
                    "end_us": s["global_end_us"],
                    "reason": "task_media_rotation",
                }
                for s in index
                if s["file"] in names
            )
            atomic_json(ledger_path, ledger)
            evidence_path = folder / "evidence/index.json"
            if evidence_path.is_file():
                entries = json.loads(evidence_path.read_text(encoding="utf8"))
                for path in files:
                    if path.parent.name == "evidence" and path.stem in entries:
                        entries[path.stem]["status"] = "policy_rotated"
                atomic_json(evidence_path, entries)
            # 原索引是列表，使用同目录临时文件发布，避免改变旧回放格式。
            from doudizhu_lab.storage.json_io import retry_file_io

            remaining = [s for s in index if s["file"] not in names]
            temporary = index_path.with_suffix(".rotation.tmp")
            temporary.write_text(json.dumps(remaining, ensure_ascii=False), encoding="utf8")
            retry_file_io(lambda: temporary.replace(index_path))
            for path in files:
                record = {
                    "session": folder.name,
                    "game_id": window.get("game_id", ""),
                    "file": str(path.relative_to(folder)),
                    "bytes": path.stat().st_size,
                    "reason": "task_media_rotation",
                    "status": "planned",
                }
                self.removed.append(record)
                atomic_json(self.folder / "媒体轮转记录.json", {"removed": self.removed})
                path.unlink(missing_ok=True)
                record["status"] = "removed"
                removed += 1
            total -= size
            count -= 1
        if self.removed:
            atomic_json(self.folder / "媒体轮转记录.json", {"removed": self.removed})
        return removed
