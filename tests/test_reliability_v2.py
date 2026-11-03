"""新证据窗口、明确未发送重试和后台清理的故障注入；没有真机输入。"""

import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from doudizhu_lab.devices.adb import InputRejected
from doudizhu_lab.runtime.capture import CaptureWorker
from doudizhu_lab.runtime.executor import ActionExecutor
from doudizhu_lab.runtime.faults import RecoveryPolicy
from doudizhu_lab.simulation.bot import SimulatedDevice
from doudizhu_lab.storage.retention import RetentionLedger, StopReason, finalize_retention
from doudizhu_lab.storage.task_media import TaskMediaRotator
from tests.test_bot import decision
from tests.test_continuous import run as run
from tests.test_continuous import startup_frame


@pytest.mark.parametrize("fault", ["stale_before_send", "unknown_after_send", "late_feedback"])
def test_submit_is_unique_and_feedback_can_arrive_after_three_seconds(fault):
    device = SimulatedDevice(73100, "landlord")
    records, attempts, delayed = [], [], []
    rejected = False

    def send(kind, x, y, snap, ident):
        nonlocal rejected
        if kind == "play":
            if fault == "stale_before_send" and not rejected:
                rejected = True
                raise InputRejected("合成旧帧", code="stale_frame")
            attempts.append(kind)
            if fault == "late_feedback":
                delayed.append((device.now + 3.4, x, y, ident))
                return
        device.send(kind, x, y, snap, ident)
        if fault == "unknown_after_send" and kind == "play":
            raise TimeoutError("已执行但返回丢失")

    executor = ActionExecutor(send, records.append, clock=lambda: device.now)
    executor.start(decision(device), device.last)
    for _ in range(150):
        snap = device.render()
        if delayed and device.now >= delayed[0][0]:
            _, x, y, ident = delayed.pop()
            device.send("play", x, y, snap, ident)
            snap = device.render()
        executor.advance(snap)
        if not executor.active:
            break
    assert executor.stage == "已确认", executor.reason
    assert attempts == ["play"]
    if fault == "late_feedback":
        assert records[-1]["confirmation_ms"] > 3400


def test_cleanup_sharing_conflict_remains_pending_and_is_retried(tmp_path, monkeypatch):
    import doudizhu_lab.storage.retention as module

    folder = tmp_path / "sessions/owned"
    folder.mkdir(parents=True)
    ledger = RetentionLedger(folder, automatic=True)
    ledger.finish(StopReason.USER)
    original_remove = module.remove_tree
    now = [0.0]

    def clock():
        now[0] += 1
        return now[0]

    def locked(*args):
        raise PermissionError(13, "合成共享占用")

    monkeypatch.setattr(module, "monotonic", clock)
    monkeypatch.setattr(module, "sleep", lambda seconds: None)
    monkeypatch.setattr(module, "remove_tree", locked)
    result = finalize_retention(folder, tmp_path)
    assert result.cleanup_pending and folder.exists()
    assert json.loads(result.summary_path.read_text(encoding="utf8"))["media_status"] == "cleanup_pending"
    monkeypatch.setattr(module, "remove_tree", original_remove)
    task = tmp_path / "runs/task"
    task.mkdir(parents=True)
    rotator = TaskMediaRotator(tmp_path, task, RecoveryPolicy())
    rotator.register(folder, closed=True)
    rotator.rotate()
    assert not folder.exists()
    assert json.loads(result.summary_path.read_text(encoding="utf8"))["media_status"] == "cleaned"


def test_active_media_validation_does_not_walk_vanishing_temporary_files(tmp_path, monkeypatch):
    folder = tmp_path / "sessions/active"
    folder.mkdir(parents=True)
    temporary = folder / ".主程序健康.json.example.tmp"
    temporary.write_text("{}", encoding="utf8")
    original = Path.lstat

    def checked(path, *args, **kwargs):
        assert path != temporary, "活动会话校验不应遍历原子发布临时文件"
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "lstat", checked)
    rotator = TaskMediaRotator(tmp_path, tmp_path / "runs/task", RecoveryPolicy())
    rotator.register(folder)
    assert rotator.rotate() == 0


def test_finishing_health_is_quiet_but_has_a_bounded_deadline():
    worker = SimpleNamespace(phase="finishing", finishing_deadline=45, fault=Mock())
    CaptureWorker._monitor_health(worker, 4)
    worker.fault.assert_not_called()
    CaptureWorker._monitor_health(worker, 46)
    assert worker.fault.call_args.args[0] == "shutdown_timeout"


def test_detach_closes_dedicated_action_database(tmp_path):
    from doudizhu_lab.runtime.bot import BotRuntime, InputGate
    from doudizhu_lab.storage.session import SessionStore

    agent = SimpleNamespace(model_id="合成模型")
    bot = BotRuntime(tmp_path, InputGate(), agent=agent, threaded=False)
    pipe = SimpleNamespace()
    bot.attach(pipe)
    store = SessionStore(tmp_path / "sessions/one")
    bot._audit_session = store
    bot.attach(None)
    assert bot._audit_session is None
    with pytest.raises(sqlite3.ProgrammingError):
        store.db.execute("SELECT 1")


def test_page_stale_rejection_reacquires_evidence_without_rebuilding(run, monkeypatch):
    actual = run.pipeline.tap
    attempts = []

    def tap(*args, **kwargs):
        attempts.append(args[2].frame_id)
        if len(attempts) == 1:
            raise InputRejected("合成旧帧", code="stale_frame")
        return actual(*args, **kwargs)

    monkeypatch.setattr(run.pipeline, "tap", tap)
    run.tick(2)
    assert run.runtime.active and not run.runtime.pending_flow and not run.pipeline.flow_commands
    run.tick(4)
    assert run.runtime.active and len(run.pipeline.flow_commands) == 1
    assert attempts[0] != attempts[1]
    assert not run.runtime.store.db.execute("SELECT 1 FROM journal WHERE kind='recovery_needed'").fetchone()


def test_parent_ack_delay_refreshes_actual_source_before_sending(run):
    intents = []

    def commit(kind, value):
        if kind == "flow_intent":
            intents.append(value["frame_id"])
            run.now[0] += 1.2
            run.pipeline.tick()
            run.gate.heartbeat()

    run.runtime.commit = commit
    run.tick(2)
    assert len(run.pipeline.flow_commands) == 1
    sent = json.loads(
        run.runtime.store.db.execute("SELECT payload FROM journal WHERE kind='flow_sent'").fetchone()[0]
    )
    assert sent["frame_id"] != intents[0]


def test_stale_flow_results_wait_eight_seconds_while_video_is_fresh(run):
    from dataclasses import replace

    run.pipeline.fault = "no_feedback"
    run.tick(2)
    flow = run.pipeline.flow_snapshot()
    origin = run.now[0]
    for elapsed in (2, 4, 7.5):
        run.now[0] = origin + elapsed
        run.pipeline.frame = replace(run.pipeline.frame, started=run.now[0])
        run.gate.heartbeat()
        run.runtime.advance(replace(flow, frame=replace(flow.frame, frame_id=str(elapsed))), None)
        assert run.runtime.active
    run.now[0] = origin + 8.2
    run.pipeline.frame = replace(run.pipeline.frame, started=run.now[0])
    run.gate.heartbeat()
    run.runtime.advance(flow, None)
    assert not run.runtime.active and len(run.pipeline.flow_commands) == 1


def test_continue_can_be_confirmed_by_two_new_round_frames_without_page_ocr(run):
    from dataclasses import replace

    from doudizhu_lab.domain.bot_types import ObservationSnapshot
    from doudizhu_lab.domain.models import PixelRect
    from doudizhu_lab.domain.table import TableObservation
    from doudizhu_lab.domain.tracker import GameState
    from doudizhu_lab.vision.flow import FlowButton

    run.pipeline.fault = "no_feedback"
    button = FlowButton("continue", "再来一局", PixelRect(450, 65, 100, 25), True)
    startup_frame(run, "settlement", buttons=(button,))
    startup_frame(run, "settlement", buttons=(button,))
    old = run.pipeline.flow_snapshot().frame
    for i in range(2):
        run.now[0] += 0.15
        frame = replace(
            old,
            frame_id=f"new-round-{i}",
            media_pts_us=old.media_pts_us + 1_000_000 + i * 150_000,
            started=run.now[0],
        )
        observation = TableObservation(
            frame.frame_id,
            frame.media_pts_us,
            phase="bidding",
            hand=["3"] * 4 + ["4"] * 4 + ["5"] * 4 + ["6"] * 4 + ["7"],
        )
        from doudizhu_lab.domain.inference import frame_context

        observation.frame_context = frame_context(frame) | {"round_id": "new"}
        run.pipeline.session.add_observation(observation)
        snap = ObservationSnapshot.create(
            frame, None, observation, GameState(game_id="new", phase="bidding"), []
        )
        flow = replace(run.pipeline.flow_snapshot(), frame=frame, scene="unknown", buttons=())
        run.gate.heartbeat()
        run.runtime.advance(flow, snap)
    assert run.runtime.pending_flow is None and run.runtime.entry_confirmed
    assert len(run.pipeline.flow_commands) == 1 and not run.runtime.armed_game


def test_strict_completion_and_audit_accept_proved_passes_without_raw_rewrite(run):
    from doudizhu_lab.storage.runs import evaluate_run

    run.pipeline.profile.continuous_policy = run.pipeline.operational.policy = "strict"
    run.pipeline.profile.continuous_settings["max_games"] = run.runtime.settings["max_games"] = 5
    for _ in range(18000):
        if run.pipeline.device:
            run.pipeline.device.hide_passes = True
        run.tick()
        if not run.runtime.active:
            break
    assert run.runtime.status == "completed", run.runtime.reason
    report = evaluate_run(run.runtime.store.folder)
    assert report["passed"] and report["inferred_events"] > 0 and not report["inference_proof_errors"]
    assert (
        report["directly_observed_events"] > 0
        and report["valid_inferred_events"] == report["inferred_events"]
    )
    assert all(g["strict_complete"] for g in report["games"])
    assert not report["rules_actions"]
    run.runtime.store.set("evidence_policy_version", 1)
    legacy = evaluate_run(run.runtime.store.folder)
    assert not legacy["passed"]
    run.runtime.store.set("evidence_policy_version", 2)
    db = run.pipeline.session.db
    recorded = next(
        (seq, value)
        for seq, payload in db.execute("SELECT seq,payload FROM bot_actions")
        if (value := json.loads(payload))["stage"] == "命令已返回" and value.get("kind") in ("play", "pass")
    )
    db.execute("DELETE FROM bot_actions WHERE seq=?", (recorded[0],))
    db.commit()
    altered = evaluate_run(run.runtime.store.folder)
    assert not altered["passed"] and recorded[1]["action_id"] in altered["submission_proof_errors"]
    assert any(a["game_id"] == recorded[1]["game_id"] and not a["passed"] for a in altered["game_audits"])


def test_continue_room_switch_is_positive_progress_without_reclick(run):
    from doudizhu_lab.domain.models import PixelRect
    from doudizhu_lab.vision.flow import FlowButton

    run.pipeline.fault = "no_feedback"
    button = FlowButton("continue", "再来一局", PixelRect(450, 65, 100, 25), True)
    startup_frame(run, "settlement", buttons=(button,))
    startup_frame(run, "settlement", buttons=(button,))
    assert run.runtime.pending_flow
    startup_frame(run, "room_confirmation", buttons=(FlowButton("confirm_room", "确定", button.hit, True),))
    assert run.runtime.pending_flow is None and run.runtime.entry_confirmed
    assert len(run.pipeline.flow_commands) == 1


def test_parent_report_separates_direct_inferred_rules_and_invalid_proofs(tmp_path):
    from doudizhu_lab.storage.runs import evaluate_task

    child = tmp_path / "child"
    child.mkdir()
    (child / "连续任务报告.json").write_text(
        json.dumps(
            {
                "passed": False,
                "directly_observed_events": 3,
                "valid_inferred_events": 2,
                "rules_actions": 1,
                "submission_proof_errors": ["missing-send"],
            }
        ),
        encoding="utf8",
    )
    report = evaluate_task(
        tmp_path,
        {
            "source": "simulation_continuous",
            "evidence_policy_version": 2,
            "status": "completed",
            "profile": {"continuous_settings": {"max_games": 1}},
        },
        [
            ("session", {"run_folder": str(child), "folder": "isolated-session"}),
            ("game_complete", {"game_id": "g"}),
        ],
    )
    assert not report["passed"] and report["directly_observed_events"] == 3
    assert report["valid_inferred_events"] == 2 and report["rules_actions"] == 1
    assert report["proof_errors"] == [
        {"session": "isolated-session", "kind": "submission_proof_errors", "error": "missing-send"}
    ]
