"""录像处置账本与安全清理；先登记决定，再删除不再需要的已关闭媒体。"""

from __future__ import annotations

import json
import os
import shutil
import stat
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from time import monotonic, sleep

from doudizhu_lab.storage.json_io import atomic_json as atomic_json


class StopReason(StrEnum):
    """收尾原因；异常不可被之后的用户停止或正常完成覆盖。"""

    USER = "user_aborted"
    COMPLETED = "completed"
    FAULT = "fault"


@dataclass(frozen=True)
class RecordingResult:
    """后台收尾结果；无媒体时回放路径为空，摘要和故障资料仍可独立存在。

    primary_fault为最早发生的故障对象，供界面保留首因；无故障或旧调用时为空。
    cleanup_pending表示资料保存成功但清理仍被共享占用阻挡，不等同于对局或采集故障。
    所有路径来自本次会话收尾，不授权读取之外的设备操作。
    """

    summary_path: Path
    replay_folder: Path | None
    reason: StopReason
    diagnostics_folder: Path | None = None
    primary_fault: dict | None = None
    cleanup_pending: bool = False


def checked_session(path: Path, root: Path) -> Path:
    """验证活动会话根目录及祖先，不枚举仍在原子发布临时文件的目录树。

    此接口不允许整目录删除。具体媒体删除仍须核对目标文件及其祖先没有重解析点；
    完整目录删除必须使用checked_tree并保证全部写入者、读取者已经结束。
    """
    path, root = Path(os.path.abspath(path)), Path(os.path.abspath(root))
    if path.parent != root or path.resolve() != path or root.resolve() != root:
        raise ValueError(f"会话路径越界或包含目录链接：{path}")
    for entry in (root, path):
        if entry.exists():
            attributes = entry.lstat()
            if stat.S_ISLNK(attributes.st_mode) or getattr(attributes, "st_file_attributes", 0) & 0x400:
                raise ValueError(f"会话路径包含链接：{entry}")
    return path


def checked_media(path: Path, folder: Path) -> Path:
    """仅验证本会话video/evidence直接子文件及祖先；不跟随链接或扫描无关临时文件。"""
    folder = checked_session(folder, folder.parent)
    if path.parent not in (folder / "video", folder / "evidence") or path.name in ("", ".", ".."):
        raise ValueError("轮转媒体文件路径越界")
    for entry in (path.parent, path):
        try:
            attributes = entry.lstat()
        except FileNotFoundError:
            continue  # 正常轮转可能已处置同一个文件；缺失不等于可以改删其他路径。
        if stat.S_ISLNK(attributes.st_mode) or getattr(attributes, "st_file_attributes", 0) & 0x400:
            raise ValueError(f"媒体路径包含链接：{entry}")
    return path


def checked_tree(path: Path, root: Path) -> Path:
    """验证待删目标是指定根目录的直接子目录，并拒绝整棵树中的符号链接和目录联接。

    本接口只检查，不删除。解析后的目标必须与原绝对路径相同，禁止利用 Windows
    重解析点越过清理范围；调用方须在所有写入者退出后使用返回值。
    """
    path, root = Path(os.path.abspath(path)), Path(os.path.abspath(root))
    if path.parent != root or path.resolve() != path or root.resolve() != root:
        raise ValueError(f"清理路径越界或包含目录链接：{path}")
    for current, directories, files in os.walk(path, followlinks=False):
        for entry in [Path(current), *(Path(current) / n for n in directories + files)]:
            attributes = entry.lstat()
            if stat.S_ISLNK(attributes.st_mode) or getattr(attributes, "st_file_attributes", 0) & 0x400:
                raise ValueError(f"清理目录包含链接，已拒绝删除：{entry}")
    return path


def remove_tree(path: Path, root: Path) -> None:
    """删除已停止使用的受控目录；占用、权限及路径检查错误直接交给调用方报告。"""
    checked = checked_tree(path, root)
    if checked.exists():
        shutil.rmtree(checked)


def playable_video(folder: Path) -> bool:
    """快速检查回放索引是否引用非空媒体，不解码整局，也不宣称通过完整性验收。"""
    try:
        data = json.loads((folder / "video/index.json").read_text(encoding="utf-8"))
        return any(
            Path(s["file"]).name == s["file"]
            and s.get("frames", 0) > 0
            and (folder / "video" / s["file"]).is_file()
            and (folder / "video" / s["file"]).stat().st_size > 0
            for s in data
        )
    except (OSError, ValueError, TypeError, KeyError):
        return False


class RetentionLedger:
    """由录制进程独占的留存账本；时间全部使用源媒体微秒，不使用墙上时间裁剪。

    当前窗口从上一个正常结算延续至下一局，覆盖大厅、匹配及叫抢。故障固定当前
    窗口的全部已录内容；已关闭的正常窗口才允许清理，活动片段始终保留。
    """

    def __init__(self, folder: Path, *, automatic: bool = False):
        """为新会话创建账本；automatic允许清理正常窗口，默认完整保留显式录像。"""
        self.folder = Path(folder)
        self.path = self.folder / "留存记录.json"
        self.automatic = automatic
        self.windows = []
        self.active = {"game_id": "", "start_us": 0, "end_us": None}
        self.faults = []
        self.removed = []
        self.frontier = -1
        self.reason = None
        self.last_pts = 0
        self.persist()

    def persist(self) -> None:
        """原子保存完整处置决定；写入失败时禁止继续清理媒体。"""
        atomic_json(self.path, self.value())

    def value(self) -> dict:
        """返回可跨进程读取的处置快照；历史窗口与真实采集缺口分别保存。"""
        return {
            "version": 1,
            "automatic": self.automatic,
            "windows": self.windows,
            "active": self.active,
            "faults": self.faults,
            "primary_fault": min(self.faults, key=lambda f: f["occurred_at"]) if self.faults else None,
            "removed_segments": self.removed,
            "discard_before_us": self.frontier,
            "reason": self.reason,
            "last_pts_us": self.last_pts,
        }

    def begin_game(self, game_id: str) -> None:
        """为当前临时窗口绑定牌局编号；保留开局前的流程画面，不截掉触发原因。"""
        if game_id and self.active["game_id"] != game_id:
            self.active["game_id"] = game_id
            self.persist()

    def complete_game(self, game_id: str, pts_us: int) -> None:
        """确认正常跟踪至结算；重复或迟到的结算不推进清理边界，也不覆盖故障。"""
        if any(w["game_id"] == game_id and w["outcome"] in ("completed", "skipped") for w in self.windows):
            return
        if pts_us < self.active["start_us"]:
            return
        if self.faults:
            return
        outcome = "skipped" if self.active.get("anomalies") else "completed"
        self.windows.append(dict(self.active, game_id=game_id, end_us=pts_us, outcome=outcome))
        self.frontier = pts_us
        self.active = {"game_id": "", "start_us": pts_us, "end_us": None}
        self.persist()

    def pin_game(self, game_id: str, code: str, message: str) -> None:
        """固定当前异常局窗口而不触发进程尾录；同类故障只记录一次，不阻止后续正常局。"""
        self.active["game_id"] = game_id or self.active["game_id"]
        anomalies = self.active.setdefault("anomalies", [])
        if not any(a["code"] == code for a in anomalies):
            anomalies.append({"code": code, "message": message, "pts_us": self.last_pts})
            self.persist()

    def skip_game(self, game_id: str, pts_us: int) -> None:
        """仅在可信页面边界关闭已固定窗口；不宣称原历史完整，重复边界不产生新窗口。"""
        if self.active.get("anomalies"):
            self.complete_game(game_id, pts_us)

    def fault(self, code: str, message: str, *, health=None, occurred_at: float | None = None) -> bool:
        """固定当前窗口并登记同类首个故障；返回是否新增，不延长外层尾录截止时间。

        occurred_at为本机源进程单调时间，省略时取当前时刻；value按该时间展示首因，
        与控制命令交付顺序无关。调用方须持有录像锁，持久化失败原样上抛。
        """
        if any(f["code"] == code for f in self.faults):
            return False
        self.faults.append(
            {
                "code": code,
                "message": message,
                "occurred_at": monotonic() if occurred_at is None else occurred_at,
                "pts_us": self.last_pts,
                "game_id": self.active["game_id"],
                "health": health or {},
            }
        )
        self.reason = StopReason.FAULT
        self.persist()
        return True

    def abort_window(self) -> None:
        """用户主动暂停时结束临时窗口；已固定的故障不会被该操作清除。"""
        if self.faults:
            return
        self.windows.append(dict(self.active, end_us=self.last_pts, outcome="user_aborted"))
        self.frontier = self.last_pts
        self.active = {"game_id": "", "start_us": self.last_pts, "end_us": None, "started_midgame": True}
        self.persist()

    def finish(self, reason: StopReason) -> None:
        """持久化最终决定；主动中止可以清理，任何先前故障拥有最高留存优先级。"""
        self.reason = StopReason.FAULT if self.faults else StopReason(reason)
        if not self.faults:
            self.frontier = self.last_pts
        self.active["end_us"] = self.last_pts
        self.persist()

    def prune(self, writer) -> None:
        """删除清理边界之前的封闭片段，并同步索引；调用方必须独占录像写入锁。

        先将待删片段登记到账本，再更新索引、删除文件。任何步骤失败都会上报故障，
        不会将仍在写入的片段或当前故障窗口交给删除接口。
        """
        if not self.automatic:
            return
        candidates = [
            s
            for s in writer.segments
            if s.closed
            and s.global_end_us < self.frontier
            and (not self.faults or s.global_end_us < self.active["start_us"])
            and not any(
                w.get("anomalies")
                and s.global_end_us >= w["start_us"]
                and s.global_start_us <= (w.get("end_us") if w.get("end_us") is not None else self.last_pts)
                for w in [*self.windows, self.active]
            )
        ]
        if not candidates:
            return
        names = {s.file for s in candidates}
        for segment in candidates:
            if Path(segment.file).name != segment.file:
                raise ValueError("录像片段路径越界")
            self.removed.append(
                {
                    "file": segment.file,
                    "start_us": segment.global_start_us,
                    "end_us": segment.global_end_us,
                    "reason": "policy_cleanup",
                }
            )
        self.persist()
        writer.segments[:] = [s for s in writer.segments if s.file not in names]
        writer._index()
        for name in names:
            (writer.folder / name).unlink(missing_ok=True)


def finalize_retention(folder: Path, root: Path, *, runtime: dict | None = None) -> RecordingResult:
    """数据库、截图线程和录像进程全部关闭后归档；无异常的自动会话仅留下摘要。

    留存决定缺失或未正常收尾时按异常保留，避免重启后误删证据。完整录制和正式
    验收不会走自动删除分支；重复调用在摘要已发布且会话已清理时返回相同结果。
    """
    summary = root / "recording-summaries" / f"{folder.name}.json"
    if not folder.exists() and summary.is_file():
        value = json.loads(summary.read_text(encoding="utf-8"))
        return RecordingResult(summary, None, StopReason(value["reason"]))
    path = folder / "留存记录.json"
    value = (
        json.loads(path.read_text(encoding="utf-8"))
        if path.exists()
        else (json.loads(summary.read_text(encoding="utf-8")) if summary.exists() else {})
    )
    if runtime is not None:
        value["runtime"] = runtime
    reason = StopReason.FAULT if value.get("faults") else StopReason(value.get("reason") or StopReason.FAULT)
    pinned = any(w.get("anomalies") for w in [*value.get("windows", []), value.get("active", {})])
    discard = value.get("automatic") and reason != StopReason.FAULT and not value.get("faults") and not pinned
    summary.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(
        summary,
        value
        | {
            "reason": reason,
            "session_id": folder.name,
            "media_status": "cleanup_pending" if discard else "retained",
        },
    )
    if discard:
        deadline = monotonic() + 5
        try:
            while True:
                try:
                    remove_tree(folder, root / "sessions")
                    break
                except OSError as exc:
                    from doudizhu_lab.storage.json_io import temporary_access_error

                    if not temporary_access_error(exc) or monotonic() >= deadline:
                        raise
                    sleep(min(0.1, max(0, deadline - monotonic())))
        except (OSError, ValueError) as exc:
            from doudizhu_lab.storage.json_io import temporary_access_error

            pending = temporary_access_error(exc)
            atomic_json(
                summary,
                value
                | {
                    "reason": reason,
                    "session_id": folder.name,
                    "media_status": "cleanup_pending" if pending else "cleanup_failed",
                    "cleanup_error": str(exc),
                },
            )
            if pending:
                return RecordingResult(
                    summary, folder if playable_video(folder) else None, reason, cleanup_pending=True
                )
            raise
        atomic_json(summary, value | {"reason": reason, "session_id": folder.name, "media_status": "cleaned"})
    replay = folder if not discard and playable_video(folder) else None
    return RecordingResult(
        summary, replay, reason, folder if not discard else None, value.get("primary_fault")
    )
