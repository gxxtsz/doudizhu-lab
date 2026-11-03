"""父进程任务看护器：有效进度、取消与恢复不依赖任何执行进程的锁。"""

from __future__ import annotations

import queue
import threading
from collections.abc import Callable
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from time import monotonic
from uuid import uuid4

from doudizhu_lab.domain.settings import PLAY_MODEL_NAMES, Profile
from doudizhu_lab.runtime.bot import ModelStatus
from doudizhu_lab.runtime.continuous import ContinuousProgress
from doudizhu_lab.runtime.faults import RecoveryPolicy
from doudizhu_lab.runtime.process import SessionProcess
from doudizhu_lab.storage.runs import RunStore, evaluate_run
from doudizhu_lab.storage.task_media import TaskMediaRotator


class TaskSupervisor:
    """后台看护线程独占账本和执行生命周期，GUI 仅读快照并提交非阻塞意图。

    root/profile/adb 是冻结配置；factory、clock、threaded 仅用于确定性故障测试。
    start_task 才建立任务授权，不从历史数据库自动恢复运行。请求停止立即撤销租约，
    磁盘、USB、IPC 和旧线程收尾都不会阻塞 GUI；本类不直接向手机发送操作。
    """

    def __init__(
        self,
        root: Path,
        profile: Profile,
        adb: str,
        *,
        factory: Callable = SessionProcess,
        clock=monotonic,
        policy: RecoveryPolicy | None = None,
        threaded: bool = True,
        simulation: bool = False,
    ) -> None:
        self.root, self.profile, self.adb = Path(root), deepcopy(profile), adb
        self.factory, self.clock, self.policy = factory, clock, policy or RecoveryPolicy()
        self.simulation, self.threaded = simulation, threaded
        self.commands, self.logs = queue.SimpleQueue(), queue.SimpleQueue()
        self.cancel_requested, self.close_requested, self.closed = (
            threading.Event(),
            threading.Event(),
            threading.Event(),
        )
        self.worker = self.store = self.media = None
        self.generation = 0
        self.serial = self.task_id = ""
        self.status, self.reason, self.scene = "idle", "", "unknown"
        self.started = self.clock()
        self.completed, self.skipped = set(), set()
        self.round_ids, self.current_round = {}, ""
        self.awaiting_boundary = False
        self.retry_count = self.failure_count = 0
        self.retry_at = None
        self.blocked_flow = None
        self._retiring = self._finishing = False
        self._retire_deadline = self._start_deadline = self._finish_deadline = 0
        self._next_status = "recovering"
        self._probing = self._starting = False
        self._idle_failed = False
        self._parent_fault = None
        self._media_folders = set()
        self._last_media_at = 0
        self._media_thread = None
        self.snapshot = {}
        self.saved_folder = None
        self.close_window_requested = False
        self._stop_request_seen = self._preserve_completed = False
        self._model = ModelStatus(
            "loading", backend=profile.play_model, display_name=PLAY_MODEL_NAMES[profile.play_model]
        )
        self._progress = ContinuousProgress(
            "idle", "unknown", "", 0, profile.continuous_settings["max_games"], 0
        )
        self.thread = None
        if threaded:
            self.thread = threading.Thread(target=self._run, name="连续任务看护", daemon=True)
            self.thread.start()

    @property
    def active(self):
        return self.status in ("preparing", "running", "recovering", "waiting_external", "stopping")

    def start_task(self, serial: str) -> None:
        """GUI 显式启动同一设备任务；重复启动不排队，设备来源由界面事先核验。"""
        if self.active or self.close_requested.is_set():
            return
        self.cancel_requested.clear()
        self.status, self.reason = "preparing", "正在准备无人值守任务"
        self.commands.put(("start", serial))

    def switch_model(self, backend: str) -> None:
        """仅空闲时切换模型；不改变运行任务冻结的模型，拒绝未知后端。"""
        if backend not in PLAY_MODEL_NAMES or self.active:
            raise ValueError("任务期间不能切换模型，或模型名称无效")
        self.profile.play_model = backend
        self._model = ModelStatus("loading", backend=backend, display_name=PLAY_MODEL_NAMES[backend])
        self.commands.put(("switch", backend))

    def heartbeat(self) -> None:
        """仅供 GUI 定时器调用；停止请求之后不续租，不以后台活跃掩盖界面失联。"""
        worker = self.worker
        if (
            worker
            and (not self.cancel_requested.is_set() or not self.active)
            and not self.close_requested.is_set()
        ):
            worker.heartbeat()

    def request_stop(self, reason: str = "用户停止连续任务") -> None:
        """立即锁住恢复并撤销当前进程租约；不等待设备、看护锁或收尾完成。"""
        if self.status == "completed":
            self._preserve_completed = True
        self.cancel_requested.set()
        worker = self.worker
        if worker:
            worker.cancel()
        self.commands.put(("stop", reason))

    def request_close(self) -> None:
        """关闭窗口时取消全部自动续跑；closed 置位前由 GUI 保持异步收尾显示。"""
        self.close_requested.set()
        self.request_stop("正在退出并保存任务")

    def progress(self) -> ContinuousProgress:
        """返回上次发布的不可变快照，绝不取得账本或 USB 锁。"""
        return self._progress

    def model_status(self) -> ModelStatus:
        """返回执行进程发布的模型状态；进程重建时旧就绪状态立即失效。"""
        return self._model

    @property
    def suggestion(self):
        from doudizhu_lab.domain.bot_types import Decision

        value = self.snapshot.get("suggestion")
        return Decision(**value) if value else None

    def _spawn(self):
        self._idle_failed = False
        self.generation += 1
        self.worker = self.factory(
            {
                "root": str(self.root),
                "profile": deepcopy(self.profile),
                "adb": self.adb,
                "generation": self.generation,
            }
        )
        self._model = ModelStatus(
            "loading", backend=self.profile.play_model, display_name=PLAY_MODEL_NAMES[self.profile.play_model]
        )
        self._start_deadline = self.clock() + self.policy.startup_seconds

    def _start(self, serial):
        if self.cancel_requested.is_set():
            self.status = "stopped"
            return
        if self.store:
            self.store.close()
            self.store = None
        self.serial, self.started = serial, self.clock()
        self.completed, self.skipped, self.round_ids = set(), set(), {}
        self.current_round = ""
        self.awaiting_boundary = False
        self.retry_count = self.failure_count = 0
        self.retry_at, self.blocked_flow = None, None
        self._finishing = self._probing = self._starting = False
        self._media_folders = set()
        self.saved_folder = None
        self._stop_request_seen = self._preserve_completed = self.close_window_requested = False
        self._parent_fault = None
        self.store = RunStore.create_task(self.root, self.profile, serial, simulation=self.simulation)
        self.store.set("recovery_policy", asdict(self.policy))
        self.task_id = self.store.folder.name
        self.media = TaskMediaRotator(
            self.root, self.store.folder, self.policy, request_rotation=self._rotate_active_media
        )
        self._start_session()

    def _rotate_active_media(self, folder, names, start_us, end_us):
        if self.worker and not self._retiring:
            self.worker.send(
                "rotate_media", folder=str(folder), names=names, start_us=start_us, end_us=end_us
            )

    def _start_session(self):
        if self.cancel_requested.is_set() or self.close_requested.is_set():
            return
        if self.worker is None:
            self._spawn()
        self.status, self.reason = "preparing", "正在自动建立执行会话"
        self._starting, self._probing = True, False
        self._start_deadline = self.clock() + self.policy.startup_seconds
        self.worker.send(
            "start",
            task_id=self.task_id,
            serial=self.serial,
            profile=deepcopy(self.profile),
            blocked_flow=self.blocked_flow,
            remaining=self.profile.continuous_settings["max_games"] - len(self.completed),
        )

    def _record(self, ident, event_kind, **value):
        if self.store is None:
            raise RuntimeError("父任务账本尚未就绪，不能确认任何执行事实")
        return self.store.commit_event(ident, event_kind, value)

    def _commit(self, event):
        kind, value = event["event_kind"], event["value"]
        game = value.get("game_id", "")
        key = (self.generation, game)
        if kind in ("deal_attempt", "game_armed", "game_complete", "game_skipped"):
            if key not in self.round_ids:
                self.round_ids[key] = (
                    self.current_round if self.awaiting_boundary and self.current_round else uuid4().hex
                )
            round_id = self.round_ids[key]
            self.current_round = round_id
            value = {**value, "round_id": round_id, "generation": self.generation}
            if kind in ("game_complete", "game_skipped"):
                if round_id not in self.completed and round_id not in self.skipped:
                    self._record("outcome:" + round_id, kind, **value)
                    (self.completed if kind == "game_complete" else self.skipped).add(round_id)
                    if kind == "game_complete":
                        self.failure_count = 0
                # 重复或相反的结果只确认交付，不改变已提交的单局决定。
            else:
                self._record(event["event_id"], kind, **value)
        else:
            self._record(event["event_id"], kind, **value)
        if kind in ("flow_intent", "flow_sent"):
            self.blocked_flow = value
        elif kind in ("flow_confirmed", "flow_retired", "flow_not_sent"):
            if self.blocked_flow and self.blocked_flow["action_id"] == value["action_id"]:
                self.blocked_flow = None
        if kind in ("round_boundary", "startup_wait_finished"):
            self.current_round = ""
            self.awaiting_boundary = False
        # 所有事实先持久化，ACK 才能解除子进程下一次输入的等待。
        self.worker.send("ack", event_id=event["event_id"])

    def _recover(self, code, reason, *, external=False):
        if self._retiring or self.cancel_requested.is_set():
            return
        if self.current_round and self.current_round not in self.completed | self.skipped:
            self._record(
                "outcome:" + self.current_round,
                "game_skipped",
                round_id=self.current_round,
                code=code,
                reason=reason,
                generation=self.generation,
            )
            self.skipped.add(self.current_round)
        self.awaiting_boundary = bool(self.current_round)
        self.failure_count += 1
        self.retry_count += 1
        self._record(
            uuid4().hex,
            "recovery",
            code=code,
            reason=reason,
            generation=self.generation,
            at=self.clock(),
            external=external,
        )
        self.reason = reason
        self.status = self._next_status = "waiting_external" if external else "recovering"
        self.store.set("status", self.status)
        self.logs.put(reason + "；将自动检查并续跑")
        self._begin_retire()

    def _begin_retire(self):
        if self._retiring:
            return
        self._retiring = True
        self._retire_deadline = self.clock() + self.policy.shutdown_seconds
        if self.worker:
            self.worker.cancel(fault=not self.cancel_requested.is_set() and self._next_status != "idle")
        else:
            self._retired()

    def _flush_parent_fault(self):
        """资源恢复后先核对父账本，再补写此前无法保存的故障；没有确认不能新开执行会话。"""
        pending = self._parent_fault
        if not pending or not self.store:
            return
        outcomes = self.store.task_outcomes()
        self.completed = {r for r, kind in outcomes.items() if kind == "game_complete"}
        self.skipped = {r for r, kind in outcomes.items() if kind == "game_skipped"}
        game = pending["round_id"]
        if game and game not in outcomes:
            self._record(
                "outcome:" + game,
                "game_skipped",
                round_id=game,
                code="parent_storage",
                reason=pending["message"],
                generation=pending["generation"],
            )
            self.skipped.add(game)
        self._record(
            pending["id"],
            "recovery",
            code="parent_storage",
            reason=pending["message"],
            generation=pending["generation"],
            external=True,
        )
        self._parent_fault = None

    def _retired(self):
        if self.worker:
            self.worker.dispose()
        self.worker = None
        self._retiring = self._starting = self._probing = self._finishing = False
        self.snapshot = {}
        for folder in self._media_folders:
            if self.media:
                self.media.register(folder, closed=True)
        if not self.cancel_requested.is_set() and not self.close_requested.is_set():
            self._flush_parent_fault()
        if (
            self._next_status == "idle"
            and not self.cancel_requested.is_set()
            and not self.close_requested.is_set()
        ):
            self.status, self.retry_at = "idle", None
        elif self.cancel_requested.is_set() or self.close_requested.is_set():
            self.status, self.retry_at = "completed" if self._preserve_completed else "stopped", None
            if self.store:
                self.store.set("status", self.status)
                evaluate_run(self.store.folder)
        elif len(self.completed) >= self.profile.continuous_settings["max_games"]:
            self._complete()
        else:
            self.status = self._next_status
            delay = (
                self.policy.external_seconds
                if self.status == "waiting_external"
                else self.policy.backoff[min(max(self.failure_count - 1, 0), len(self.policy.backoff) - 1)]
            )
            self.retry_at = self.clock() + delay

    def _complete(self):
        self.status, self.reason, self.retry_at = "completed", "已完成目标有效局数", None
        self._finishing = False
        self.store.set("status", "completed")
        evaluate_run(self.store.folder)

    def _event(self, event):
        if event.get("generation") != self.generation:
            return
        if event.get("task_id") != self.task_id:
            return
        kind = event["kind"]
        if kind == "commit" and not self.cancel_requested.is_set() and not self._retiring:
            self._commit(event)
        elif kind == "session_created":
            folder = Path(event["folder"])
            self.media.register(folder)
            self._media_folders.add(folder)
            self.store.set("sessions", [str(p) for p in sorted(self._media_folders)])
        elif kind == "session":
            self._record(
                "session:" + str(self.generation),
                "session",
                folder=event["folder"],
                run_folder=event["run_folder"],
                generation=self.generation,
            )
            self._starting = False
            if not self._retiring and not self.cancel_requested.is_set():
                self.status, self.reason = "running", "正在确认新局入口"
                self.store.set("status", self.status)
        elif kind == "session_closed":
            self.media.register(Path(event["folder"]), closed=True)
            if event.get("replay"):
                self.saved_folder = Path(event["replay"])
        elif kind == "fault":
            if event.get("blocked_flow"):
                self.blocked_flow = event["blocked_flow"]
            self._recover(event["code"], event["message"], external=event["scope"] == "external")
        elif kind == "probe" and not self.cancel_requested.is_set() and not self._retiring:
            self._probing = False
            if event["ready"]:
                self._start_session()
            else:
                self.status = "waiting_external"
                self.reason = event["reason"]
                self.retry_at = self.clock() + self.policy.external_seconds
        elif kind == "finished" and self._finishing and not self.cancel_requested.is_set():
            self._complete()

    def tick(self) -> None:
        """推进一次看护状态机；仅后台线程或注入时钟的测试调用，不持有执行进程锁。"""
        try:
            while True:
                kind, value = self.commands.get_nowait()
                if kind == "start":
                    self._start(value)
                elif kind == "stop":
                    self.reason = value
                    if self.active or self.worker:
                        self.status = "stopping"
                        self._begin_retire()
                    else:
                        self.status = "stopped"
                elif kind == "switch":
                    if self.worker:
                        self.worker.send("switch", backend=value)
                elif kind == "load" and self.worker:
                    self.worker.send("switch", backend=self.profile.play_model)
                elif kind == "load":
                    self._spawn()
                elif kind == "fault":
                    self._recover("gui_render", value)
                elif kind == "configure":
                    self.profile, self.adb = value
                    self._next_status = "idle"
                    self._begin_retire()
        except queue.Empty:
            pass
        now = self.clock()
        if self.worker:
            events, snapshot = self.worker.poll()
            for event in events:
                self._event(event)
            if snapshot and snapshot.get("generation") == self.generation:
                if snapshot.get("model"):
                    self._model = ModelStatus(**snapshot["model"])
                old_frame = self.snapshot.get("frame")
                self.snapshot = snapshot
                if old_frame and "frame" not in snapshot:
                    self.snapshot = {**snapshot, "frame": old_frame}
                progress = snapshot.get("progress")
                if (
                    progress
                    and snapshot.get("task_id") == self.task_id
                    and not self._retiring
                    and not self.cancel_requested.is_set()
                ):
                    self.scene = progress["scene"]
                    if self.status == "running":
                        self.reason = progress["reason"]
                    if (
                        progress["status"] == "completed"
                        and not self._finishing
                        and self.status in ("running", "preparing")
                        and len(self.completed) >= self.profile.continuous_settings["max_games"]
                    ):
                        self._finishing = True
                        self._finish_deadline = now + self.policy.shutdown_seconds
                        self.status, self.reason = "stopping", "目标已达成，正在保存最后会话"
                        self.worker.send("finish")
            if self._retiring:
                if not self.worker.alive:
                    self._retired()
                elif now >= self._retire_deadline:
                    self.worker.kill()
            elif self.active:
                if not self.worker.alive:
                    self._recover("process_exit", self.worker.error or "执行进程提前退出")
                elif self._finishing and now >= self._finish_deadline:
                    self._recover("shutdown_timeout", "最后会话收尾超时")
                elif (self._starting or self._probing) and now >= self._start_deadline:
                    self._recover("startup_timeout", "执行会话启动或探测超时")
                elif not self._starting and not self._probing and not self._finishing:
                    stamp = self.snapshot.get("updated_at")
                    if stamp and now - stamp > 3:
                        self._recover("worker_stalled", "执行进程超过三秒未发布状态")
            elif not self.worker.alive:
                error = self.worker.error or "模型预热进程已退出"
                self.worker.dispose()
                self.worker = None
                self._idle_failed = True
                self._model = ModelStatus(
                    "failed", error, self.profile.play_model, PLAY_MODEL_NAMES[self.profile.play_model]
                )
        if self.store and not self._stop_request_seen:
            request = self.store.folder / "停止请求.json"
            if request.is_file():
                import json

                value = json.loads(request.read_text(encoding="utf8"))
                self._stop_request_seen = True
                self.close_window_requested = bool(value.get("close_window"))
                self.request_stop("命令行请求停止连续任务")
        if not self._retiring and not self.close_requested.is_set():
            if (
                self.worker is None
                and not self._idle_failed
                and self.status in ("idle", "stopped", "completed")
            ):
                self._spawn()
            elif (
                not self.cancel_requested.is_set()
                and self.status in ("recovering", "waiting_external")
                and self.retry_at is not None
                and now >= self.retry_at
            ):
                self.retry_at = None
                self._flush_parent_fault()
                if self.store and len(self.completed) >= self.profile.continuous_settings["max_games"]:
                    self._complete()
                elif self.store is None:
                    self._start(self.serial)
                elif self.status == "waiting_external":
                    if self.worker is None:
                        self._spawn()
                    self._probing = True
                    self._start_deadline = now + self.policy.startup_seconds
                    self.worker.send(
                        "probe", task_id=self.task_id, serial=self.serial, blocked_flow=self.blocked_flow
                    )
                else:
                    self._start_session()
        if (
            self.media
            and now - self._last_media_at >= 5
            and not (self._media_thread and self._media_thread.is_alive())
        ):
            self._last_media_at = now
            media = self.media

            def archive():
                try:
                    media.rotate()
                    if self.media is media and self.saved_folder:
                        from doudizhu_lab.storage.retention import playable_video

                        if not playable_video(self.saved_folder):
                            self.saved_folder = None
                except (OSError, ValueError) as exc:
                    self.logs.put(f"媒体轮转暂未完成，稍后自动重试：{exc}")

            if self.threaded:
                self._media_thread = threading.Thread(target=archive, name="任务异常媒体轮转", daemon=True)
                self._media_thread.start()
            else:
                archive()
        self._progress = ContinuousProgress(
            self.status,
            self.scene,
            self.reason,
            len(self.completed),
            self.profile.continuous_settings["max_games"],
            max(0, now - self.started),
            "ready",
            len(self.skipped),
            self.retry_count,
            self.retry_at,
        )

    def _run(self):
        try:
            while True:
                try:
                    self.tick()
                except Exception as exc:
                    # 父账本不能确认时不回复 ACK，减权并等待资源恢复，不能假装已经提交。
                    self.reason = f"父任务保存或恢复暂不可用：{exc}"
                    if self.store and self._parent_fault is None:
                        self._parent_fault = {
                            "id": uuid4().hex,
                            "message": self.reason,
                            "round_id": self.current_round,
                            "generation": self.generation,
                        }
                        self.awaiting_boundary = bool(self.current_round)
                        self.retry_count += 1
                    if self.cancel_requested.is_set() or self.close_requested.is_set():
                        self.status = "completed" if self._preserve_completed else "stopped"
                        self.retry_at = None
                        if self.worker and not self._retiring:
                            self._begin_retire()
                    elif self.worker and not self._retiring:
                        self.status = self._next_status = "waiting_external"
                        self._begin_retire()
                    elif self.worker is None:
                        self.status = self._next_status = "waiting_external"
                        self.retry_at = self.clock() + self.policy.external_seconds
                    self._progress = ContinuousProgress(
                        self.status,
                        self.scene,
                        self.reason,
                        len(self.completed),
                        self.profile.continuous_settings["max_games"],
                        max(0, self.clock() - self.started),
                        "ready",
                        len(self.skipped),
                        self.retry_count,
                        self.retry_at,
                    )
                    self.logs.put(self.reason)
                if self.close_requested.is_set() and self.worker is None:
                    break
                self.close_requested.wait(
                    0.1
                ) if not self.close_requested.is_set() else threading.Event().wait(0.05)
        finally:
            try:
                if self.store:
                    self.store.close()
            finally:
                self.closed.set()
