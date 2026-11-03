"""本地会话账本：原始观察只追加，人工修订单独留痕，支持确定性回放。"""

from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
import threading
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from doudizhu_lab import __version__
from doudizhu_lab.paths import PACKAGE_ROOT, engine_hashes, project_files

# 原始采集损失不能通过删除推演事件抹去，离线重算也须继承这些缺口。
HARD_GAP_CODES = frozenset(
    {
        "transport",
        "recording",
        "foreground",
        "orientation",
        "finalize",
        "device",
        "startup",
        "disk_backlog",
        "shutdown",
        "reanalysis",
        "decode",
    }
)

ANALYSIS_GAP_CODES = frozenset({"analysis", "decode_backlog", "analysis_decode"})


def game_intervals(events: list[dict], end_us: int) -> dict[str, tuple[int, int]]:
    """从当前有效边界建立各局时间范围；修订分局后仍按同一源时间轴匹配缺口。

    只使用该局的开局与结算，未结算时截到下一局开局或会话末尾。事件中的旧 game_id
    不能改变采集缺口的物理时间归属，也不能将结算后的中断算到已完成的上一局。
    """
    ordered = sorted(events, key=lambda e: e["pts_us"])
    starts = [e for e in ordered if e["kind"] == "game_start"]
    result = {}
    for index, start in enumerate(starts):
        ends = [
            e["pts_us"]
            for e in ordered
            if e["kind"] == "game_end" and e["game_id"] == start["game_id"] and e["pts_us"] >= start["pts_us"]
        ]
        limit = starts[index + 1]["pts_us"] if index + 1 < len(starts) else end_us
        result[start["game_id"]] = (start["pts_us"], min(ends) if ends else max(start["pts_us"], limit))
    return result


def json_value(value):
    """将公共数据类转换为 JSON 值，不接收图像数组或设备操作对象。"""
    return asdict(value) if is_dataclass(value) else value


def dumps(value) -> str:
    return json.dumps(json_value(value), ensure_ascii=False, separators=(",", ":"))


def release_fingerprint(metadata: dict) -> str:
    """计算算法、锁定依赖、配置和全部模板的冻结指纹；文档文字不影响算法版本。

    使用会话内已复制文件的清单，路径统一为斜线，排除运行目录和创建时间等易变字段。
    核验文件内容还需调用 verify_snapshot，不能只比较声称的哈希。
    """

    def manifest(name):
        return sorted(
            (m["path"].replace("\\", "/"), m["sha256"])
            for m in metadata.get(name, [])
            if Path(m["path"]).name != "README.md"
        )

    data = {
        "engine": manifest("engine_manifest"),
        "templates": manifest("template_manifest"),
        "profile": metadata["profile"],
    }
    if metadata.get("vision_model_manifest"):
        data["vision_models"] = manifest("vision_model_manifest")
    if metadata.get("bidding_model_manifest"):
        data["bidding_model"] = manifest("bidding_model_manifest")
    return hashlib.sha256(
        json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def snapshot_template_dirs(store) -> list[Path]:
    """返回录制和冻结共用的有效模板目录，包含本机基础模板与隔离运行目录。

    自定义 data-dir 不能导致冻结空模板、录制却自动补入基础模板。重复目录由
    SessionStore.create 去重；同名而内容不同的模板仍会拒绝，避免来源不明确。
    """
    from doudizhu_lab.storage.config import Store

    base = Store().root
    return [
        base / "templates",
        store.root / "templates",
        base / "markers",
        store.root / "markers",
        base / "controls",
        store.root / "controls",
    ]


def freeze_release(store) -> Path:
    """保存下一批验收使用的完整只读基准；该动作只准备版本，不开始采集或操作手机。"""
    base = store.root
    release = SessionStore.create(
        base / "acceptance/releases", store.profile, snapshot_template_dirs(store), "offline", model_root=base
    )
    try:
        release.set_meta("status", "frozen_release")
        meta = release.metadata()
        release.verify_snapshot()
        payload = {
            "release_id": meta["session_id"],
            "fingerprint": release_fingerprint(meta),
            "created_at": meta["created_at"],
            "folder": str(release.folder.resolve()),
        }
        target = base / "acceptance/current-release.json"
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(target)
        return target
    finally:
        release.close()


def reserve_acceptance(store, session) -> dict:
    """在首个视频包到达前核对冻结版本并登记验收批次；任何源码/模板变更均需重新冻结。"""
    path = store.root / "acceptance/current-release.json"
    if not path.is_file():
        raise ValueError("请先运行 ddz-check freeze-acceptance 冻结程序、配置和模板")
    release = json.loads(path.read_text(encoding="utf-8"))
    frozen = SessionStore(Path(release["folder"]))
    try:
        frozen.verify_snapshot()
        session.verify_snapshot()
        if (
            release_fingerprint(frozen.metadata()) != release["fingerprint"]
            or release_fingerprint(session.metadata()) != release["fingerprint"]
        ):
            raise ValueError("当前程序、配置或模板已变化，请完成回归并重新冻结后再验收")
    finally:
        frozen.close()
    return release | {
        "batch_id": session.metadata()["session_id"],
        "reserved_at": datetime.now(timezone.utc).isoformat(),
        "selection": "连续完整牌局；首尾不完整局单列，禁止按成绩挑选",
    }


class SessionStore:
    """一个录制会话对应一个目录和 SQLite 数据库。

    小事务与线程锁保证写入观察、事件及修订时不交叉使用游标；WAL 保留已提交数据。
    打开旧会话不会修改原始识别结果；导出时同时保留原始与修订后的事件。
    """

    def __init__(self, folder: Path, *, readonly: bool = False):
        """打开会话；readonly用于历史审计，禁止建库、改模式或写入。

        只读入口要求数据库已存在，仍能读取已提交的WAL内容；不会以immutable模式
        忽略未合并事务。默认可写模式保留录制与纠错兼容，所有访问仍由实例锁串行化。
        """
        self.folder = folder = Path(folder)
        self.lock = threading.RLock()
        if readonly:
            path = folder / "session.sqlite3"
            if not path.is_file():
                raise FileNotFoundError("找不到既有会话数据库")
            self.db = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, check_same_thread=False)
            self.db.row_factory = sqlite3.Row
            return
        folder.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(folder / "session.sqlite3", check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY,value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS observations (seq INTEGER PRIMARY KEY,frame_id TEXT UNIQUE,pts_us INTEGER,payload TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS events (seq INTEGER PRIMARY KEY,event_id TEXT UNIQUE,game_id TEXT,pts_us INTEGER,kind TEXT,payload TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS issues (seq INTEGER PRIMARY KEY,issue_id TEXT UNIQUE,game_id TEXT,pts_us INTEGER,code TEXT,payload TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS revisions (seq INTEGER PRIMARY KEY,created_at TEXT,operation TEXT,target TEXT,payload TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS bot_actions (seq INTEGER PRIMARY KEY,action_id TEXT,game_id TEXT,stage TEXT,pts_us INTEGER,payload TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS operational_events (seq INTEGER PRIMARY KEY,event_id TEXT UNIQUE,game_id TEXT,pts_us INTEGER,kind TEXT,payload TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS recoveries (seq INTEGER PRIMARY KEY,payload TEXT NOT NULL);
            CREATE INDEX IF NOT EXISTS observation_time ON observations(pts_us);
            CREATE INDEX IF NOT EXISTS event_game ON events(game_id,pts_us);
        """)
        self.db.commit()

    @classmethod
    def create(
        cls, root: Path, profile, template_dirs: list[Path], device: str, *, model_root: Path | None = None
    ) -> "SessionStore":
        """创建会话并复制模板及可选OCR模型，记录原文件 SHA-256 与配置快照。

        model_root 指向当前模型所在数据目录；未传入时使用项目默认数据。只复制已校验
        的三份OCR资产，不下载、不包含未使用的YOLO权重，重算不依赖后来更换的模型。
        """
        vision_sources = []
        if profile.hand_classifier == "pretrained-ocr":
            from doudizhu_lab.storage.config import Store
            from doudizhu_lab.vision.assets import ASSETS, model_path

            for key in ("ocr_det", "ocr_en", "ocr_zh"):
                asset = ASSETS[key]
                vision_sources.append((model_path(model_root or Store().root, key), asset))
        ident = datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid4().hex[:6]
        result = cls(root / "sessions" / ident)
        result.set_meta("session_id", ident)
        result.set_meta("created_at", datetime.now(timezone.utc).isoformat())
        result.set_meta("device", device)
        result.set_meta("profile", json_value(profile))
        result.set_meta("version", 2)
        result.set_meta("evidence_policy_version", 2)
        result.set_meta("analysis_mode", "offline" if device in ("replay", "offline", "test") else "live")
        result.set_meta("tracker_settings", json_value(profile.tracker_settings))
        result.set_meta("engine_sha256", engine_hashes())
        # 源码工作区与安装后的wheel共用同一清单。仅复制程序与重现元数据，排除运行资料。
        snapshot = result.folder / "engine"
        destination_package = snapshot / "src" / "doudizhu_lab"
        destination_package.mkdir(parents=True, exist_ok=True)
        engine_manifest = []
        source_files = [
            p
            for p in PACKAGE_ROOT.rglob("*")
            if p.is_file()
            and "__pycache__" not in p.parts
            and "_resources" not in p.parts
            and (p.suffix in (".py", ".json", ".txt", ".md") or p.name in ("LICENSE", "NOTICE"))
        ]
        for path in source_files:
            destination = destination_package / path.relative_to(PACKAGE_ROOT)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
            engine_manifest.append(
                {
                    "path": destination.relative_to(snapshot).as_posix(),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
        for name, path in project_files().items():
            destination = snapshot / name
            shutil.copy2(path, destination)
            engine_manifest.append({"path": name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        result.set_meta("engine_manifest", engine_manifest)
        result.set_meta("application_version", __version__)
        result.set_meta("status", "recording")
        manifest = []
        seen = set()
        for folder in template_dirs:
            if folder.resolve() in seen:
                continue
            seen.add(folder.resolve())
            target = result.folder / "templates" / folder.name
            target.mkdir(parents=True, exist_ok=True)
            for path in folder.glob("*"):
                if path.suffix not in (".png", ".json"):
                    continue
                destination = target / path.name
                if destination.exists() and destination.read_bytes() != path.read_bytes():
                    raise ValueError(f"模板文件名冲突：{path.name}")
                shutil.copy2(path, destination)
                manifest.append(
                    {
                        "path": str(destination.relative_to(result.folder)),
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    }
                )
        result.set_meta("template_manifest", manifest)
        if vision_sources:
            models = []
            for source, asset in vision_sources:
                destination = result.folder / "models/pretrained-vision" / asset["file"]
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
                models.append(dict(asset, path=str(destination.relative_to(result.folder))))
            result.set_meta("vision_model_manifest", models)
        return result

    def set_meta(self, key: str, value) -> None:
        """事务写入会话元数据；配置、模板哈希、采集指标均用 JSON 值保存。"""
        with self.lock, self.db:
            self.db.execute("INSERT OR REPLACE INTO meta VALUES (?,?)", (key, dumps(value)))

    def snapshot_vision_models(self, root: Path) -> None:
        """在连续任务输入授权前复制并校验流程OCR资产；不会改变已保存的原始观察。

        普通模板录制不强制安装OCR。用户显式启动连续任务时再准备这三份固定模型，
        元数据和实际文件均进入快照校验，缺少依赖或模型会阻止匹配。
        """
        from doudizhu_lab.vision.assets import ASSETS, model_path

        sources = [(model_path(root, k), ASSETS[k]) for k in ("ocr_det", "ocr_en", "ocr_zh")]
        manifest = []
        for source, asset in sources:
            destination = self.folder / "models/pretrained-vision" / asset["file"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            if source.resolve() != destination.resolve():
                shutil.copy2(source, destination)
            manifest.append(dict(asset, path=str(destination.relative_to(self.folder))))
        self.set_meta("vision_model_manifest", manifest)
        self.verify_snapshot()

    def metadata(self) -> dict:
        """返回独立的元数据字典，调用者修改字典不会隐式写入数据库。"""
        with self.lock:
            return {row[0]: json.loads(row[1]) for row in self.db.execute("SELECT key,value FROM meta")}

    def snapshot_bidding_model(self, folder: Path) -> None:
        """在首次叫抢输入前复制已验证权重、来源和许可证；评分可用当时的文件独立复现。"""
        from doudizhu_lab.agents.bidding import verify_bid_model

        verify_bid_model(folder)
        destination = self.folder / "models/bidding"
        destination.mkdir(parents=True, exist_ok=True)
        manifest = []
        for name in ("bid_weights.pkl", "manifest.json", "LICENSE", "verification.json"):
            source = folder / name
            if not source.is_file() and name == "verification.json":
                continue
            target = destination / name
            shutil.copy2(source, target)
            manifest.append(
                {
                    "path": str(target.relative_to(self.folder)),
                    "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                }
            )
        self.set_meta("bidding_model_manifest", manifest)

    def reject_acceptance(self, reason: str) -> dict:
        """将已停止且未通过的正式批次登记为开发资料，保留原录像和实时成绩。

        后续重算和人工纠错不会取消此登记。正式评测始终拒绝通过，须重新冻结并录制
        新批次。重复调用返回最早的失败原因，避免无意覆盖验收审计记录。
        """
        meta = self.metadata()
        if not reason.strip():
            raise ValueError("请填写验收失败的具体原因")
        if not meta.get("acceptance") or meta.get("status") == "recording":
            raise ValueError("请先停止一个正式验收批次，再登记为开发回归资料")
        previous = meta.get("acceptance_outcome", {})
        if previous.get("disposition") == "development":
            return previous
        outcome = {
            "disposition": "development",
            "passed": False,
            "reviewed_at": datetime.now(timezone.utc).isoformat(),
            "reason": reason.strip(),
        }
        self.set_meta("acceptance_outcome", outcome)
        return outcome

    def add_observation(self, observation) -> None:
        """同一帧重复投递不会追加第二条观察；源时间戳永久保留。"""
        value = json_value(observation)
        with self.lock, self.db:
            self.db.execute(
                "INSERT OR IGNORE INTO observations(frame_id,pts_us,payload) VALUES (?,?,?)",
                (value["frame_id"], value["pts_us"], dumps(value)),
            )

    def add_event(self, event) -> None:
        """追加由稳定观察生成的事件；event_id 幂等，人工改动应调用 revise。"""
        value = json_value(event)
        with self.lock, self.db:
            self.db.execute(
                "INSERT OR IGNORE INTO events(event_id,game_id,pts_us,kind,payload) VALUES (?,?,?,?,?)",
                (value["event_id"], value["game_id"], value["pts_us"], value["kind"], dumps(value)),
            )

    def add_issue(self, issue) -> None:
        """保存疑点或采集缺口；缺少编号时生成独立 UUID，原始缺口不会被修订删除。"""
        value = json_value(issue)
        value.setdefault("issue_id", uuid4().hex)
        with self.lock, self.db:
            self.db.execute(
                "INSERT OR IGNORE INTO issues(issue_id,game_id,pts_us,code,payload) VALUES (?,?,?,?,?)",
                (
                    value["issue_id"],
                    value.get("game_id", ""),
                    value.get("pts_us", 0),
                    value["code"],
                    dumps(value),
                ),
            )

    def add_operational_event(self, event: dict) -> None:
        """追加平衡模式运行事件；推断来源独立留痕，绝不覆盖严格events或原始observations。"""
        with self.lock, self.db:
            self.db.execute(
                "INSERT OR IGNORE INTO operational_events(event_id,game_id,pts_us,kind,payload) VALUES (?,?,?,?,?)",
                (event["event_id"], event["game_id"], event["pts_us"], event["kind"], dumps(event)),
            )

    def add_recovery(self, value: dict) -> None:
        """记录唯一推断、规则接管及其源帧，不把模式切换当成真实出牌。"""
        with self.lock, self.db:
            self.db.execute("INSERT INTO recoveries(payload) VALUES (?)", (dumps(value),))

    def add_bot_action(self, value: dict) -> None:
        """追加机器人意图及执行阶段；绝不向 events 写入未经画面确认的动作。"""
        with self.lock, self.db:
            self.db.execute(
                "INSERT INTO bot_actions(action_id,game_id,stage,pts_us,payload) VALUES (?,?,?,?,?)",
                (
                    value["action_id"],
                    value.get("game_id", ""),
                    value["stage"],
                    value.get("pts_us", 0),
                    dumps(value),
                ),
            )

    def rows(self, table: str) -> list[dict]:
        """仅允许固定表名，返回追加顺序；不接受用户拼接 SQL。"""
        if table not in (
            "observations",
            "events",
            "issues",
            "revisions",
            "bot_actions",
            "operational_events",
            "recoveries",
        ):
            raise ValueError("未知会话数据类型")
        with self.lock:
            if table == "revisions":
                return [
                    dict(row) | {"payload": json.loads(row["payload"])}
                    for row in self.db.execute("SELECT * FROM revisions ORDER BY seq")
                ]
            return [
                json.loads(row[0]) for row in self.db.execute(f"SELECT payload FROM {table} ORDER BY seq")
            ]

    def verify_snapshot(self) -> None:
        """核验保存的算法和模板文件；路径不得逃逸会话目录，缺失或篡改必须明确报错。

        旧版本没有算法副本时只检查其模板清单；可通过 metadata 判断是否具备源码复现条件。
        本方法不执行副本中的任何代码。
        """
        metadata = self.metadata()
        for root, manifest in (
            (self.folder / "engine", metadata.get("engine_manifest", [])),
            (self.folder, metadata.get("template_manifest", [])),
            (self.folder, metadata.get("vision_model_manifest", [])),
            (self.folder, metadata.get("bidding_model_manifest", [])),
        ):
            for entry in manifest:
                path = (root / entry["path"]).resolve()
                if not path.is_relative_to(root.resolve()) or not path.is_file():
                    raise ValueError("会话快照文件缺失或路径无效")
                if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
                    raise ValueError("会话快照校验失败：" + path.name)

    def revise(self, operation: str, target: str, value: dict) -> int:
        """追加人工修订；undo 的 target 为修订序号，原观察与原事件永不覆盖。

        insert/replace/delete 修改事件；boundary 调整 game_start/game_end 的时间或归属。
        每次调用后由 replay_events 重算整段状态，非法牌数不会被人工修订直接洗成成功。
        """
        if operation not in ("insert", "replace", "delete", "boundary", "undo"):
            raise ValueError("未知修订操作")
        if operation in ("replace", "delete", "boundary") and target not in {
            e["event_id"] for e in self.effective_events()
        }:
            raise ValueError("找不到要修订的事件")
        if operation == "undo" and (
            not target.isdigit() or int(target) not in {r["seq"] for r in self.rows("revisions")}
        ):
            raise ValueError("找不到要撤销的修订")
        with self.lock, self.db:
            cur = self.db.execute(
                "INSERT INTO revisions(created_at,operation,target,payload) VALUES (?,?,?,?)",
                (datetime.now(timezone.utc).isoformat(), operation, target, dumps(value)),
            )
            return cur.lastrowid

    def effective_events(self) -> list[dict]:
        """按修订顺序重建有效事件；同时间戳使用原序号排序，结果跨重启一致。"""
        events = {e["event_id"]: dict(e) for e in self.rows("events")}
        revisions = self.rows("revisions")
        disabled = set()
        for revision in reversed(revisions):
            if revision["seq"] not in disabled and revision["operation"] == "undo":
                disabled.add(int(revision["target"]))
        for revision in revisions:
            if revision["seq"] in disabled:
                continue
            operation, target, value = revision["operation"], revision["target"], revision["payload"]
            if operation == "delete":
                events.pop(target, None)
            elif operation in ("replace", "boundary") and target in events:
                events[target] = events[target] | value | {"event_id": target, "source": "manual"}
            elif operation == "insert":
                ident = f"manual-{revision['seq']}"
                events[ident] = value | {"event_id": ident, "source": "manual"}
        ordered = sorted(events.values(), key=lambda e: e["pts_us"])
        # 人工移动开局边界后，普通事件按新的时间区间重新归局；不覆盖其原始记录。
        boundary_changed = any(r["operation"] == "boundary" and r["seq"] not in disabled for r in revisions)
        if boundary_changed:
            starts = [e for e in ordered if e["kind"] == "game_start"]
            for event in ordered:
                if event["kind"] in ("game_start", "game_end"):
                    continue
                owner = next((e for e in reversed(starts) if e["pts_us"] <= event["pts_us"]), None)
                if owner:
                    event["game_id"] = owner["game_id"]
        return ordered

    def game_states(self, events: list[dict] | None = None) -> dict:
        """推演状态并叠加不可抹去的采集缺口，供界面、导出、评测使用同一可信规则。"""
        from doudizhu_lab.domain.tracker import replay_events

        events = self.effective_events() if events is None else events
        states = replay_events(events)
        media_end = self.media_end_us()
        intervals = game_intervals(events, media_end)
        hard_gaps = self.gap_intervals()
        for ident, state in states.items():
            start, end = intervals.get(ident, (0, media_end))
            if any(g["start_us"] <= end and g["end_us"] >= start for g in hard_gaps):
                state["complete"] = False
                state["capture_incomplete"] = True
        return states

    def media_end_us(self) -> int:
        """返回当前可见的会话末尾媒体时间；同时考虑录像和观察，避免分析滞后截短缺口。"""
        with self.lock:
            row = self.db.execute("SELECT MAX(pts_us) FROM observations").fetchone()
            last = row[0] or 0
            event_last = self.db.execute("SELECT MAX(pts_us) FROM events").fetchone()[0] or 0
        video = (self.folder / self.metadata().get("video_source", ".")).resolve() / "video/index.json"
        try:
            segments = json.loads(video.read_text(encoding="utf-8"))
            return max(last, event_last, max((s["global_end_us"] for s in segments), default=0))
        except (OSError, ValueError, KeyError):
            return max(last, event_last)

    def gap_intervals(self, *, include_analysis=False) -> list[dict]:
        """解析追加的缺口起止记录；不修改原疑点，也不以人工修订消除采集事实。

        gap_closed 通过 gap_id 引用原缺口。未闭合缺口延伸到当前会话末尾；旧版没有
        时间范围的记录按其原牌局保守处理，未知归属则覆盖整个会话。分析缺口可单列，
        不能把实时解码掉帧误称为完整录像已丢失。
        """
        issues = self.rows("issues")
        closures = {
            i.get("gap_id"): i.get("end_us", i.get("pts_us", 0)) for i in issues if i["code"] == "gap_closed"
        }
        end = self.media_end_us()
        intervals = game_intervals(self.rows("events"), end)
        result = []
        for issue in issues:
            analysis = issue.get("scope") == "analysis" or issue["code"] in ANALYSIS_GAP_CODES
            if (analysis and not include_analysis) or (not analysis and issue["code"] not in HARD_GAP_CODES):
                continue
            if "start_us" in issue:
                start = issue["start_us"]
                finish = issue.get("end_us")
                finish = closures.get(issue["issue_id"], finish)
                open_ended = finish is None
                finish = max(start, end) if finish is None else finish
            else:
                start, finish = intervals.get(issue.get("game_id"), (0, end))
                open_ended = True
            result.append(
                issue
                | {
                    "start_us": start,
                    "end_us": max(start, finish),
                    "open_ended": open_ended,
                    "scope": "analysis" if analysis else "capture",
                }
            )
        # 未正常关闭的片段可能来自进程退出；不依赖操作者补填缺口，也不波及
        # 之前已经关闭的独立片段。正常录制收尾后 closed 会更新为 true。
        index = (self.folder / self.metadata().get("video_source", ".")).resolve() / "video/index.json"
        if index.exists():
            try:
                for segment in json.loads(index.read_text(encoding="utf-8")):
                    if not segment.get("closed"):
                        result.append(
                            {
                                "issue_id": "unclosed-" + segment["file"],
                                "code": "unclosed_segment",
                                "scope": "capture",
                                "message": "录像片段未正常收尾",
                                "start_us": segment["global_start_us"],
                                "end_us": max(segment["global_end_us"], end),
                                "open_ended": True,
                            }
                        )
            except (OSError, ValueError, KeyError) as exc:
                result.append(
                    {
                        "issue_id": "invalid-video-index",
                        "code": "video_index",
                        "scope": "capture",
                        "message": f"录像索引不可读取：{exc}",
                        "start_us": 0,
                        "end_us": end,
                        "open_ended": True,
                    }
                )
        return result

    def export(self, destination: Path | None = None) -> Path:
        """导出原始与人工修订两份事件及重放结论；数据与录像索引可离线交叉核对。"""
        effective = self.effective_events()
        output = destination or self.folder / "整局记录.json"
        payload = {
            "version": 2,
            "metadata": self.metadata(),
            "observations": self.rows("observations"),
            "automatic_events": self.rows("events"),
            "effective_events": effective,
            "issues": self.rows("issues"),
            "revisions": self.rows("revisions"),
            "bot_actions": self.rows("bot_actions"),
            "operational_events": self.rows("operational_events"),
            "recoveries": self.rows("recoveries"),
            "replay": self.game_states(effective),
        }
        # 传输、写盘和前台缺口是原始采集事实，删掉一个推演 gap 事件也不能将其抹去。
        hard_gaps = self.gap_intervals()
        payload["capture_gaps"] = hard_gaps
        payload["analysis_gaps"] = [
            g for g in self.gap_intervals(include_analysis=True) if g["scope"] == "analysis"
        ]
        video_folder = (self.folder / self.metadata().get("video_source", ".")).resolve() / "video"
        index_path = video_folder / "index.json"
        payload["video_index"] = str(index_path)
        payload["video_segments"] = (
            json.loads(index_path.read_text(encoding="utf-8")) if index_path.exists() else []
        )
        evidence_index = self.folder / "evidence/index.json"
        try:
            evidence_entries = (
                json.loads(evidence_index.read_text(encoding="utf-8")) if evidence_index.exists() else {}
            )
        except (OSError, ValueError):
            evidence_entries = {}
        payload["evidence_index"] = evidence_entries
        for kind in ("automatic_events", "effective_events"):
            for event in payload[kind]:
                segment = next(
                    (
                        s
                        for s in payload["video_segments"]
                        if s["global_start_us"] <= event["pts_us"] <= s["global_end_us"]
                    ),
                    None,
                )
                event["media"] = (
                    {"segment": segment["file"], "offset_us": event["pts_us"] - segment["global_start_us"]}
                    if segment
                    else None
                )
                evidence = self.folder / "evidence" / f"{event.get('frame_id', '')}.png"
                if evidence.is_file():
                    event["evidence"] = str(evidence.relative_to(self.folder))
                event["confirmation_evidence"] = [
                    item
                    for item in evidence_entries.values()
                    if event["event_id"] in item.get("event_ids", [])
                ]
        payload["complete_trusted"] = (
            bool(payload["replay"])
            and all(state["complete"] for state in payload["replay"].values())
            and not hard_gaps
        )
        output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return output

    def close(self):
        """结束本连接；调用方须先停止写入线程，已提交事务保留在本地账本。"""
        with self.lock:
            self.db.close()
