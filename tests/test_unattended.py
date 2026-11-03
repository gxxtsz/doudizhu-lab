"""无人值守恢复回归；只使用隔离目录、模拟手机和专属测试子进程。"""

import json
import subprocess
import sys
from dataclasses import asdict, replace
from time import monotonic, sleep
from types import SimpleNamespace

import pytest

from doudizhu_lab.runtime.bot import InputGate
from doudizhu_lab.runtime.faults import RecoveryPolicy
from doudizhu_lab.runtime.process import SessionProcess
from doudizhu_lab.runtime.supervisor import TaskSupervisor
from doudizhu_lab.storage.config import Profile
from doudizhu_lab.storage.retention import RetentionLedger
from doudizhu_lab.storage.runs import RunStore, evaluate_run
from doudizhu_lab.storage.task_media import TaskMediaRotator
from tests.test_continuous import playing
from tests.test_continuous import run as run


def test_unverified_settlement_is_skipped_and_next_complete_game_counts(run, monkeypatch):
    run.runtime.unattended = True
    run.runtime.settings["max_games"] = 2
    original = run.pipeline.session.rows
    monkeypatch.setattr(
        run.pipeline.session, "rows", lambda table: [] if table == "bot_actions" else original(table)
    )
    for _ in range(12000):
        run.tick()
        if run.runtime.skipped:
            break
    assert len(run.runtime.skipped) == 1 and not run.runtime.completed
    assert run.runtime.active and run.runtime.quarantined
    monkeypatch.setattr(run.pipeline.session, "rows", original)
    for _ in range(12000):
        run.tick()
        if run.runtime.status == "completed":
            break
    assert len(run.runtime.completed) == 2
    assert len(run.pipeline.devices) == 3
    assert not sum(d.invalid + d.duplicate for d in run.pipeline.devices)


def test_midgame_gap_waits_for_natural_end_and_never_rearms_same_game(run):
    run.runtime.unattended = True
    playing(run)
    game = run.runtime.current_game
    run.pipeline.device.tracker.state.pending = True
    run.pipeline.operational.tracker.state.pending = True
    run.tick()
    assert run.runtime.quarantined and run.runtime.status == "running"
    before = run.pipeline.device.taps
    for _ in range(1500):
        device = run.pipeline.device
        if not device.env.game_over and device.env.acting_player_position == device.roles["self"]:
            # 模拟手机自己的超时托管；不经过应用输入出口，不计作机器人完成。
            device.act(device.env.game_infoset.legal_actions[0])
        run.tick()
        if not run.runtime.quarantined:
            break
    assert device.taps == before
    assert game in run.runtime.skipped and game not in run.runtime.completed
    assert not run.runtime.quarantined


def test_unknown_flow_remains_blocked_after_execution_replacement(run):
    run.runtime.unattended = True
    run.runtime.blocked_flow = {"origin_scene": "lobby", "action_id": "old-command"}
    run.tick(40)
    assert not run.pipeline.flow_commands and run.runtime.active
    flow = run.pipeline.flow_snapshot()
    run.runtime.advance(replace(flow, scene="unknown", frame=replace(flow.frame, frame_id="unknown")), None)
    assert run.runtime.blocked_flow
    run.runtime.advance(replace(flow, scene="matching", frame=replace(flow.frame, frame_id="advanced")), None)
    assert run.runtime.blocked_flow is None
    assert not run.pipeline.flow_commands


def test_revoked_process_lease_cannot_be_regranted(run):
    live = [True]
    gate = InputGate(clock=run.clock, transport="simulation", lease=lambda: live[0])
    gate.allow()
    token = gate.grant("flow")
    gate.heartbeat()
    live[0] = False
    assert gate.grant("flow") != token and gate.owner is None
    with pytest.raises(ValueError, match="撤销"):
        gate.tap("flow", token, run.pipeline, 1, 1, run.pipeline.frame)


class FakeSession:
    """只模拟执行事实与进程寿命；手机状态位于工厂外，不随执行进程重建。"""

    def __init__(self, config, phone):
        self.config, self.phone = config, phone
        self.alive, self.error = True, ""
        self.events, self.sent = [], []
        self.task_id = ""
        self.latest = {}
        self.cancelled = False
        self.killed = False

    def event(self, kind, **value):
        self.events.append(
            {"kind": kind, "task_id": self.task_id, "generation": self.config["generation"], **value}
        )

    def fact(self, kind, game, event_id=None, **value):
        self.event(
            "commit", event_kind=kind, event_id=event_id or kind + game, value={"game_id": game, **value}
        )

    def send(self, kind, **value):
        self.sent.append((kind, value))
        if kind in ("start", "probe"):
            self.task_id = value["task_id"]

    def poll(self):
        result, self.events = self.events, []
        return result, self.latest

    def heartbeat(self):
        pass

    def cancel(self, *, fault=False):
        self.cancelled = True
        if not self.phone.hang:
            self.alive = False

    def kill(self):
        self.killed, self.alive = True, False

    def dispose(self):
        pass


@pytest.fixture
def supervised(tmp_path):
    now = [1000.0]
    phone = SimpleNamespace(hang=False, physical_round=0, reset_count=0)
    workers = []

    def factory(config):
        worker = FakeSession(config, phone)
        workers.append(worker)
        return worker

    supervisor = TaskSupervisor(
        tmp_path, Profile(), "unused", factory=factory, clock=lambda: now[0], threaded=False, simulation=True
    )
    supervisor.start_task("same-phone")
    supervisor.tick()
    yield SimpleNamespace(supervisor=supervisor, now=now, phone=phone, workers=workers)
    supervisor.request_stop()
    phone.hang = False
    for worker in workers:
        worker.alive = False
    supervisor.tick()
    if supervisor.store:
        supervisor.store.close()


def test_300_valid_outcomes_survive_restarts_without_resetting_device(supervised):
    s, now = supervised.supervisor, supervised.now
    s.profile.continuous_settings["max_games"] = 300
    s.store.set("profile", asdict(s.profile))
    for attempt in range(1, 304):
        worker = s.worker
        supervised.phone.physical_round += 1
        game = str(supervised.phone.physical_round)
        worker.fact("game_armed", game)
        s.tick()
        if attempt in (1, 152, 301):
            worker.event("fault", scope="session", code="injected_failure", message="模拟执行故障")
            s.tick()
            s.tick()
            assert s.worker is None
            assert len(s.completed) == attempt - len(s.skipped)
            now[0] = s.retry_at
            s.tick()
            assert s.worker is not worker
            # 模拟手机自然结束旧局后提供新入口；新进程本身不重置手机状态。
            s.worker.fact("startup_wait_finished", "")
            s.tick()
        else:
            worker.fact("game_complete", game, verified=True)
            s.tick()
            # 同一事实重复交付不得计两次。
            worker.fact("game_complete", game, event_id="duplicate-" + game, verified=True)
            s.tick()
    assert len(s.completed) == 300 and len(s.skipped) == 3
    assert supervised.phone.reset_count == 0 and supervised.phone.physical_round == 303
    s._complete()
    report = evaluate_run(s.store.folder)
    assert report["target_reached"] and report["completed_games"] == 300
    assert not report["passed"]


def test_shutdown_hang_is_killed_and_stop_cancels_all_retries(supervised):
    s, now = supervised.supervisor, supervised.now
    worker = s.worker
    supervised.phone.hang = True
    worker.event("fault", scope="session", code="blocked_recognizer", message="识别线程阻塞")
    s.tick()
    assert s.status == "recovering" and worker.cancelled
    now[0] += 45
    s.tick()
    assert worker.killed
    s.request_stop()
    s.tick()
    assert s.status == "stopped" and s.retry_at is None
    now[0] += 100000
    s.tick()
    assert not any(kind == "start" for kind, value in s.worker.sent) if s.worker else True


def test_external_wait_uses_light_probes_and_resumes_same_device(supervised):
    s, now = supervised.supervisor, supervised.now
    s.worker.event("fault", scope="external", code="device", message="设备离线")
    s.tick()
    s.tick()
    assert s.status == "waiting_external" and s.retry_at == now[0] + 60
    now[0] += 59
    s.tick()
    assert s.worker is None
    now[0] += 1
    s.tick()
    worker = s.worker
    assert worker.sent[-1][0] == "probe" and worker.sent[-1][1]["serial"] == "same-phone"
    for _ in range(5):
        worker.event("probe", ready=False, reason="仍离线")
        s.tick()
        now[0] += 60
        s.tick()
    assert not any(kind == "start" for kind, value in worker.sent)
    worker.event("probe", ready=True, reason="已恢复")
    s.tick()
    assert worker.sent[-1][0] == "start" and worker.sent[-1][1]["serial"] == "same-phone"


def test_stale_generation_cannot_commit_or_reauthorize(supervised):
    s = supervised.supervisor
    s._event(
        {
            "kind": "commit",
            "generation": -1,
            "task_id": s.task_id,
            "event_id": "late",
            "event_kind": "game_complete",
            "value": {"game_id": "old"},
        }
    )
    assert not s.completed
    assert not any(k == "ack" for k, value in s.worker.sent)


def test_same_physical_round_keeps_id_across_repeated_worker_failures(supervised):
    s, now = supervised.supervisor, supervised.now
    s.worker.fact("game_armed", "old-tracker-id")
    s.tick()
    first_round = s.current_round
    for index in range(3):
        s.worker.event("fault", scope="session", code="analysis", message="重复识别故障")
        s.tick()
        s.tick()
        now[0] = s.retry_at
        s.tick()
        s.worker.fact("deal_attempt", "replacement-tracker-" + str(index))
        s.tick()
        assert s.current_round == first_round
        assert len(s.skipped) == 1
    s.worker.fact("startup_wait_finished", "")
    s.tick()
    s.worker.fact("game_armed", "actual-new-game")
    s.tick()
    assert s.current_round != first_round


def test_local_anomaly_does_not_stop_capture_or_prune_pinned_context(tmp_path):
    ledger = RetentionLedger(tmp_path)
    ledger.begin_game("bad")
    ledger.pin_game("bad", "history_gap", "历史缺失")
    ledger.last_pts = 10
    ledger.skip_game("bad", 10)
    ledger.begin_game("good")
    ledger.complete_game("good", 20)
    assert not ledger.faults
    assert [w["outcome"] for w in ledger.windows] == ["skipped", "completed"]


def test_task_media_rotation_keeps_history_and_active_media(tmp_path):
    task = tmp_path / "runs/task"
    task.mkdir(parents=True)
    rotator = TaskMediaRotator(tmp_path, task, replace(RecoveryPolicy(), media_games=1))
    for name in ("history", "old", "new", "active"):
        folder = tmp_path / "sessions" / name
        (folder / "video").mkdir(parents=True)
        ledger = RetentionLedger(folder)
        ledger.pin_game(name, "fault", "测试故障")
        ledger.last_pts = 10
        ledger.skip_game(name, 10)
        (folder / "video/x.mkv").write_bytes(b"test")
        (folder / "video/index.json").write_text(
            json.dumps([{"file": "x.mkv", "global_start_us": 0, "global_end_us": 10, "closed": True}]),
            encoding="utf8",
        )
        if name != "history":
            rotator.register(folder, closed=name != "active")
    rotator.rotate()
    assert not (tmp_path / "sessions/old/video/x.mkv").exists()
    assert (tmp_path / "sessions/history/video/x.mkv").exists()
    assert (tmp_path / "sessions/active/video/x.mkv").exists()
    assert (task / "媒体轮转记录.json").is_file()


def blocked_process(config, connection, heartbeat, cancelled):
    """独立测试进程及其孙进程都永久阻塞，模拟无法用 Python 线程取消的识别库。"""
    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(300)"],
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    connection.send({"kind": "child", "pid": child.pid})
    while True:
        sleep(1)


def test_real_blocked_process_tree_is_reclaimed_without_phone(tmp_path):
    worker = SessionProcess({"root": str(tmp_path), "generation": 1}, entrypoint=blocked_process)
    try:
        deadline = monotonic() + 15
        event = None
        while monotonic() < deadline:
            values, _ = worker.poll()
            if values:
                event = values[0]
                break
            sleep(0.05)
        assert event and event["kind"] == "child"
        worker.cancel()
        assert worker.alive
        worker.kill()
        deadline = monotonic() + 5
        while worker.alive and monotonic() < deadline:
            sleep(0.05)
        assert not worker.alive
        if sys.platform == "win32":
            import ctypes

            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel.OpenProcess.restype = ctypes.c_void_p
            handle = kernel.OpenProcess(0x1000, False, event["pid"])
            if handle:
                code = ctypes.c_ulong()
                kernel.GetExitCodeProcess(ctypes.c_void_p(handle), ctypes.byref(code))
                kernel.CloseHandle(ctypes.c_void_p(handle))
                assert code.value != 259
    finally:
        worker.kill()
        sleep(0.05)
        worker.dispose()


def test_task_outcomes_commit_idempotently_and_old_format_stays_readable(tmp_path):
    store = RunStore.create_task(tmp_path, Profile(), "test", simulation=True)
    try:
        assert store.commit_event("round:one", "game_complete", {"round_id": "one"})
        assert not store.commit_event("round:one", "game_complete", {"round_id": "one"})
        result = evaluate_run(store.folder)
        assert result["completed_games"] == 1 and not result["target_reached"]
    finally:
        store.close()


def simulation_process(config, connection, heartbeat, cancelled):
    """运行真实执行入口和真实机器人事务，仅替换手机及模型，不连接 ADB。"""
    import doudizhu_lab.devices.adb as adb_module
    import doudizhu_lab.runtime.bot as bot_module
    import doudizhu_lab.runtime.recording as recording_module
    from doudizhu_lab.runtime.session_process import run_session_process
    from doudizhu_lab.simulation.continuous import ContinuousSimPipeline
    from doudizhu_lab.storage.retention import RecordingResult, StopReason
    from tests.test_continuous import LegalTestAgent

    original_bot = bot_module.BotRuntime
    original_gate = bot_module.InputGate

    def make_bot(root, gate, **kwargs):
        return original_bot(root, gate, agent=LegalTestAgent())

    def make_gate(**kwargs):
        return original_gate(transport="simulation", **kwargs)

    def reject_adb(*args, **kwargs):
        raise AssertionError("隔离集成测试禁止创建真实 ADB 客户端")

    class Pipeline(ContinuousSimPipeline):
        def __init__(self, store, serial, **kwargs):
            super().__init__(store, threaded=True, session_root=store.root)

        def stop(self, reason=StopReason.USER):
            super().stop()

        def finalize_recording(self):
            return RecordingResult(self.session.folder / "摘要.json", None, StopReason.COMPLETED)

    bot_module.BotRuntime = make_bot
    bot_module.InputGate = make_gate
    adb_module.AdbClient = reject_adb
    recording_module.RecordingPipeline = Pipeline
    run_session_process(config, connection, heartbeat, cancelled)


def test_production_ipc_commits_and_finishes_without_real_device(tmp_path):
    from doudizhu_lab.storage.config import Store

    profile = Profile()
    profile.continuous_settings = {"max_games": 1}
    Store(tmp_path)
    supervisor = TaskSupervisor(
        tmp_path,
        profile,
        "",
        simulation=True,
        factory=lambda cfg: SessionProcess(cfg, entrypoint=simulation_process),
    )
    try:
        deadline = monotonic() + 90
        while supervisor.model_status().state != "ready" and monotonic() < deadline:
            supervisor.heartbeat()
            sleep(0.03)
        assert supervisor.model_status().state == "ready", supervisor.reason
        supervisor.start_task("simulation")
        while supervisor.status not in ("completed", "waiting_external") and monotonic() < deadline:
            supervisor.heartbeat()
            sleep(0.03)
        assert supervisor.status == "completed", supervisor.reason
        report = evaluate_run(supervisor.store.folder)
        assert report["target_reached"] and report["completed_games"] == 1
        assert report["sessions"] and not report["skipped"]
    finally:
        supervisor.request_close()
        supervisor.closed.wait(8)
        if supervisor.worker:
            supervisor.worker.kill()
        assert supervisor.closed.wait(5)


@pytest.mark.gui
def test_unattended_window_shows_wait_and_stops_without_resume(qtbot, tmp_path):
    from doudizhu_lab.domain.models import Device
    from doudizhu_lab.storage.config import Store
    from doudizhu_lab.ui.unattended import UnattendedWindow

    phone = SimpleNamespace(hang=False)

    def factory(root, profile, adb):
        def make_worker(cfg):
            worker = FakeSession(cfg, phone)
            worker.latest = {
                "generation": cfg["generation"],
                "updated_at": monotonic(),
                "model": {"state": "ready", "backend": "resnet", "display_name": "ResNet"},
            }
            return worker

        return TaskSupervisor(root, profile, "", factory=make_worker)

    window = UnattendedWindow(
        Store(tmp_path), auto_refresh=False, auto_preview=False, supervisor_factory=factory
    )
    qtbot.addWidget(window)
    window.available_devices = [Device("simulation", "device", "模拟手机")]
    window.select_device("simulation")
    window.show()
    qtbot.waitUntil(
        lambda: window.supervisor is not None and window.supervisor.model_status().state == "ready"
    )
    qtbot.waitUntil(lambda: window.action_button.isEnabled())
    window.action_button.click()
    qtbot.waitUntil(lambda: bool(window.supervisor.task_id))
    window._done(window.generation, "devices", [], "")
    assert window.selected_device.serial == "simulation"
    window.supervisor.worker.task_id = window.supervisor.task_id
    window.supervisor.worker.event("fault", scope="external", code="offline", message="设备离线")
    qtbot.waitUntil(lambda: window.task_badge.text() == "等待条件恢复")
    assert window.stop_button.isEnabled() and not window.action_button.isEnabled()
    assert "设备离线" in window.notice_label.text()
    window.stop_button.click()
    qtbot.waitUntil(lambda: window.supervisor.status == "stopped")
    window.close()
    qtbot.waitUntil(lambda: not window.isVisible(), timeout=5000)
    assert window.supervisor.closed.is_set()


def test_300_complete_games_use_real_round_tracking_with_three_bad_settlements(run, monkeypatch):
    """完整54张牌模拟而非完成事件替身；异常局之外必须实际确认全部本人动作。"""
    from doudizhu_lab.runtime.faults import FaultScope

    run.runtime.unattended = True
    run.runtime.settings["max_games"] = 300
    run.pipeline.profile.continuous_settings["max_games"] = 300
    run.runtime.store.set("profile", asdict(run.pipeline.profile))
    settle = run.runtime._settlement

    def injected(snapshot):
        game = run.runtime.current_game
        if game and game not in run.runtime.skipped and run.pipeline.game_number in (1, 152, 301):
            run.runtime.pause("模拟结算证据不足", scope=FaultScope.ROUND, code="settlement_test")
            return False
        return settle(snapshot)

    monkeypatch.setattr(run.runtime, "_settlement", injected)
    for _ in range(1_000_000):
        run.tick()
        if not run.runtime.active:
            break
    assert run.runtime.status == "completed", run.runtime.description
    assert len(run.runtime.completed) == 300 and len(run.runtime.skipped) == 3
    assert len(run.pipeline.devices) == 303
    assert not sum(d.invalid + d.duplicate for d in run.pipeline.devices)
    result = evaluate_run(run.runtime.store.folder)
    assert result["target_reached"] and result["completed_games"] == 300
    assert all(g["own_actions"] == g["confirmed_actions"] for g in result["games"])


def test_parent_does_not_ack_completion_before_durable_commit(supervised, monkeypatch):
    s = supervised.supervisor
    s.worker.fact("game_armed", "one")
    s.tick()
    ack_count = sum(kind == "ack" for kind, value in s.worker.sent)

    def fail(*args, **kwargs):
        raise OSError("模拟磁盘不可写")

    monkeypatch.setattr(s.store, "commit_event", fail)
    s.worker.fact("game_complete", "one")
    with pytest.raises(OSError):
        s.tick()
    assert not s.completed
    assert sum(kind == "ack" for kind, value in s.worker.sent) == ack_count
    monkeypatch.undo()


def test_retries_have_no_attempt_cap_and_backoff_resets_after_valid_round(supervised):
    s, now = supervised.supervisor, supervised.now
    for expected in (2, 5, 10, 30, 60, 60, 60):
        s.worker.event("fault", scope="session", code="analysis", message="重复异常")
        s.tick()
        s.tick()
        assert s.retry_at - now[0] == expected
        now[0] = s.retry_at
        s.tick()
    s.worker.fact("game_complete", "good")
    s.tick()
    s.worker.event("fault", scope="session", code="analysis", message="再次异常")
    s.tick()
    s.tick()
    assert s.retry_at - now[0] == 2


def test_completed_snapshot_does_not_start_second_finalize(supervised):
    s = supervised.supervisor
    s.profile.continuous_settings = {"max_games": 1}
    s.store.set("profile", asdict(s.profile))
    s.worker.fact("game_complete", "one")
    s.tick()
    s.worker.latest = {
        "generation": s.generation,
        "task_id": s.task_id,
        "updated_at": s.clock(),
        "progress": {"scene": "settlement", "status": "completed", "reason": "完成"},
    }
    s.tick()
    assert s._finishing
    s.worker.event("finished")
    s.tick()
    for _ in range(5):
        s.tick()
    assert s.status == "completed"
    assert sum(kind == "finish" for kind, value in s.worker.sent) == 1
    (s.store.folder / "停止请求.json").write_text('{"close_window":true}', encoding="utf8")
    s.tick()
    s.tick()
    assert s.close_window_requested and s.status == "completed"


def test_screenshot_rotation_keeps_queued_and_future_evidence(tmp_path):
    from doudizhu_lab.storage.evidence import EvidenceWriter

    writer = EvidenceWriter(tmp_path / "evidence")
    for name, pts, status in (("old", 1, "saved"), ("queued", 2, "queued"), ("future", 10, "saved")):
        writer.entries[name] = {"pts_us": pts, "status": status}
        (writer.folder / f"{name}.png").write_bytes(b"test")
    assert writer.rotate_window(0, 10) == 1
    assert not (writer.folder / "old.png").exists()
    assert (writer.folder / "queued.png").exists() and (writer.folder / "future.png").exists()
    assert writer.entries["old"]["status"] == "policy_rotated"


def test_locked_media_is_retried_after_index_decision(tmp_path, monkeypatch):
    from pathlib import Path

    task = tmp_path / "runs/task"
    task.mkdir(parents=True)
    folder = tmp_path / "sessions/fault"
    (folder / "video").mkdir(parents=True)
    ledger = RetentionLedger(folder)
    ledger.pin_game("bad", "gap", "历史缺口")
    ledger.skip_game("bad", 10)
    media = folder / "video/x.mkv"
    media.write_bytes(b"test")
    (folder / "video/index.json").write_text(
        json.dumps([{"file": "x.mkv", "global_start_us": 0, "global_end_us": 10, "closed": True}]),
        encoding="utf8",
    )
    rotator = TaskMediaRotator(tmp_path, task, replace(RecoveryPolicy(), media_games=0))
    rotator.register(folder, closed=True)
    original = Path.unlink
    locked = [True]

    def unlink(path, *args, **kwargs):
        if path == media and locked[0]:
            raise PermissionError("模拟媒体占用")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", unlink)
    with pytest.raises(PermissionError):
        rotator.rotate()
    assert media.exists()
    locked[0] = False
    rotator.rotate()
    assert not media.exists()
    assert rotator.removed[0]["status"] == "removed"


def test_cleanup_refuses_closed_session_owned_by_waiting_task(tmp_path):
    import sqlite3

    from doudizhu_lab.storage.cleanup import apply_legacy_cleanup, legacy_inventory

    folder = tmp_path / "sessions/fault"
    folder.mkdir(parents=True)
    with sqlite3.connect(folder / "session.sqlite3") as db:
        db.execute("CREATE TABLE meta(key TEXT PRIMARY KEY,payload TEXT)")
        db.execute("INSERT INTO meta VALUES ('status','\"stopped\"')")
    task = RunStore.create_task(tmp_path, Profile(), "test")
    try:
        task.set("sessions", [str(folder.resolve())])
        task.set("status", "waiting_external")
        inventory = legacy_inventory(tmp_path)
        manifest = tmp_path / "manifest.json"
        manifest.write_text(json.dumps(inventory), encoding="utf8")
        with pytest.raises(RuntimeError, match="仍在使用"):
            apply_legacy_cleanup(tmp_path, manifest)
        assert folder.exists() and task.folder.exists()
    finally:
        task.close()


def test_incomplete_new_game_does_not_reuse_previous_round_id(run):
    run.runtime.unattended = True
    playing(run)
    previous = run.runtime.current_game
    snapshot = run.pipeline.bot_snapshot()
    run.runtime._playing(
        SimpleNamespace(state={**snapshot.state, "game_id": "incomplete-next", "pending": True})
    )
    assert run.runtime.current_game == "incomplete-next"
    assert "incomplete-next" in run.runtime.skipped and previous not in run.runtime.skipped


@pytest.mark.gui
def test_explicit_cli_start_does_not_require_manual_model_retry(qtbot, tmp_path):
    from doudizhu_lab.domain.models import Device
    from doudizhu_lab.storage.config import Store
    from doudizhu_lab.ui.unattended import UnattendedWindow

    phone = SimpleNamespace(hang=False)

    def factory(root, profile, adb):
        return TaskSupervisor(root, profile, "", factory=lambda cfg: FakeSession(cfg, phone))

    window = UnattendedWindow(
        Store(tmp_path),
        auto_refresh=False,
        auto_preview=False,
        startup_action="continuous",
        supervisor_factory=factory,
    )
    qtbot.addWidget(window)
    window.available_devices = [Device("simulation", "device")]
    window.select_device("simulation")
    window.show()
    qtbot.waitUntil(lambda: window.supervisor is not None and bool(window.supervisor.task_id))
    assert window.supervisor.model_status().state != "ready"
    assert window.supervisor.active and not window._startup_action
    window.close()
    qtbot.waitUntil(lambda: not window.isVisible(), timeout=5000)


def test_uncertain_parent_commit_is_reconciled_before_recovery(supervised):
    s = supervised.supervisor
    s.worker.fact("game_armed", "one")
    s.tick()
    ident = s.current_round
    # 模拟 COMMIT 已落盘但调用方收到异常，内存尚未增加有效局数。
    s.store.commit_event("outcome:" + ident, "game_complete", {"round_id": ident, "game_id": "one"})
    s._parent_fault = {
        "id": "storage-failure",
        "message": "磁盘提交反馈未知",
        "round_id": ident,
        "generation": s.generation,
    }
    s._next_status = "waiting_external"
    s._begin_retire()
    s.tick()
    assert s.completed == {ident} and not s.skipped
    assert not s._parent_fault
    report = evaluate_run(s.store.folder)
    assert report["completed_games"] == 1 and len(report["recoveries"]) == 1


def test_stop_remains_terminal_when_status_cannot_be_saved(tmp_path, monkeypatch):
    phone = SimpleNamespace(hang=False)
    s = TaskSupervisor(tmp_path, Profile(), "", factory=lambda cfg: FakeSession(cfg, phone))
    try:
        s.start_task("simulation")
        deadline = monotonic() + 5
        while not s.store and monotonic() < deadline:
            sleep(0.01)
        assert s.store

        def fail(*args, **kwargs):
            raise OSError("磁盘仍不可写")

        monkeypatch.setattr(s.store, "set", fail)
        s.request_stop()
        deadline = monotonic() + 3
        while s.progress().status != "stopped" and monotonic() < deadline:
            sleep(0.01)
        assert s.progress().status == "stopped" and s.retry_at is None
    finally:
        s.request_close()
        assert s.closed.wait(5)
