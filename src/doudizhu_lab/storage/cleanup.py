"""旧录像整会话清理；只接受明确列举、已结束且位于当前数据根下的资料。"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
from contextlib import closing
from pathlib import Path

from doudizhu_lab.storage.retention import atomic_json, checked_tree, remove_tree


def read_metadata(path: Path) -> dict:
    """只读读取两种历史数据库的元数据，不迁移模式、不创建缺失数据库。"""
    with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as db:
        return {row[0]: json.loads(row[1]) for row in db.execute("SELECT * FROM meta")}


def protected_files(root: Path) -> dict:
    """记录全局配置、模板和模型的指纹；大模型记录大小和修改时间，小文件额外计算哈希。"""
    paths = set(root.glob("settings*.json"))
    for name in ("templates", "markers", "controls", "models"):
        if (root / name).exists():
            paths.update(p for p in (root / name).rglob("*") if p.is_file())
    result = {}
    for path in sorted(paths):
        value = path.stat()
        result[path.relative_to(root).as_posix()] = {
            "size": value.st_size,
            "mtime_ns": value.st_mtime_ns,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest() if value.st_size < 4 * 1024**2 else None,
        }
    return result


def legacy_inventory(root: Path) -> dict:
    """生成当前会话及其直接关联任务的精确清单，包含字节数和受保护资料指纹。

    检查阶段允许列出运行中会话，但执行阶段会拒绝删除。不会把其他数据根中的
    旧引用、模拟任务或独立评测目录误纳入清理范围。
    """
    root = root.resolve()
    sessions, runs = [], []
    for database in sorted((root / "sessions").glob("*/session.sqlite3")):
        folder = checked_tree(database.parent, root / "sessions")
        meta = read_metadata(database)
        files = [p for p in folder.rglob("*") if p.is_file()]
        sessions.append(
            {
                "path": str(folder),
                "status": meta.get("status"),
                "bytes": sum(p.stat().st_size for p in files),
                "video_count": sum(
                    p.suffix.lower() in (".mkv", ".mp4", ".webm", ".avi", ".mov") for p in files
                ),
            }
        )
    identities = {os.path.normcase(s["path"]) for s in sessions}
    for database in sorted((root / "runs").glob("*/run.sqlite3")):
        meta = read_metadata(database)
        source = meta.get("session")
        if source and os.path.normcase(str(Path(source).resolve())) in identities:
            folder = checked_tree(database.parent, root / "runs")
            runs.append(
                {
                    "path": str(folder),
                    "status": meta.get("status"),
                    "bytes": sum(p.stat().st_size for p in folder.rglob("*") if p.is_file()),
                }
            )
    return {
        "version": 1,
        "root": str(root),
        "sessions": sessions,
        "runs": runs,
        "protected": protected_files(root),
        "deleted": [],
        "errors": [],
        "freed_bytes": 0,
    }


def apply_legacy_cleanup(root: Path, manifest_path: Path) -> dict:
    """按持久化清单删除旧会话和关联任务；可重复执行，逐项记录真实结果。

    所有路径、链接和运行状态在第一次删除前复核。文件占用时留下错误和已完成
    清单，不强杀进程，不退回通配符删除，也不清理清单生成后出现的新会话。
    """
    root = root.resolve()
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    if Path(value["root"]) != root:
        raise ValueError("清理清单不属于当前数据目录")
    targets = []
    for category, database in (("sessions", "session.sqlite3"), ("runs", "run.sqlite3")):
        for entry in value[category]:
            path = checked_tree(Path(entry["path"]), root / category)
            if not path.exists():
                continue
            meta = read_metadata(path / database)
            active = ("recording",) if category == "sessions" else ("running", "paused")
            if meta.get("status") in active:
                raise RuntimeError(f"资料仍在使用，须先正常停止：{path}")
            targets.append((entry, path, root / category))
    value["errors"] = []
    value.setdefault("free_space_before", shutil.disk_usage(root).free)
    for entry, path, allowed_root in targets:
        try:
            remove_tree(path, allowed_root)
            if str(path) not in value["deleted"]:
                value["deleted"].append(str(path))
                value["freed_bytes"] += entry["bytes"]
        except (OSError, ValueError) as exc:
            value["errors"].append({"path": str(path), "error": str(exc)})
        atomic_json(manifest_path, value)
    value["protected_unchanged"] = protected_files(root) == value["protected"]
    value["free_space_after"] = shutil.disk_usage(root).free
    value["released_space_bytes"] = max(0, value["free_space_after"] - value["free_space_before"])
    value["remaining"] = [
        entry["path"] for entry in value["sessions"] + value["runs"] if Path(entry["path"]).exists()
    ]
    atomic_json(manifest_path, value)
    return value
