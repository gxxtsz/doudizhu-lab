"""执行进程入口：模型、识别和录像故障均被父进程的恢复期限约束。"""

from __future__ import annotations

import shutil
import threading
from dataclasses import asdict
from pathlib import Path
from time import monotonic
from uuid import uuid4


def run_session_process(config, connection, parent_heartbeat, cancelled):
    """执行父端冻结的任务指令；只有活租约和完整新局证据同时存在才允许手机输入。

    connection 传递结构化消息，不创建 Qt；关键账本事件等待父端持久确认。
    cancelled 或父界面失联只撤销授权；收尾挂起由父进程回收整个专属进程组。
    """
    import cv2

    from doudizhu_lab.devices.adb import AdbClient
    from doudizhu_lab.runtime.bot import BotRuntime, InputGate
    from doudizhu_lab.runtime.capture import foreground
    from doudizhu_lab.runtime.continuous import ContinuousRuntime
    from doudizhu_lab.runtime.recording import RecordingPipeline
    from doudizhu_lab.storage.config import Store
    from doudizhu_lab.storage.retention import StopReason
    from doudizhu_lab.storage.runs import evaluate_run, refresh_run_retention
    from doudizhu_lab.vision.flow import FlowRecognizer

    root = Path(config["root"])
    store = Store(root)
    store.profile, store.adb_path = config["profile"], config["adb"]
    gate = InputGate(lease=lambda: not cancelled.is_set() and monotonic() - parent_heartbeat.value < 1)
    bot = BotRuntime(root, gate, play_model=store.profile.play_model)
    sending, ack_lock = threading.Lock(), threading.Lock()
    acks = {}
    runtime = pipeline = None
    task_id, generation = "", config["generation"]
    fault_sent, finishing = False, False
    last_frame, last_snapshot = "", 0.0

    def send(kind, **value):
        with sending:
            connection.send({"kind": kind, "task_id": task_id, "generation": generation, **value})

    def commit(kind, value):
        ident = uuid4().hex
        event = threading.Event()
        with ack_lock:
            acks[ident] = event
        send("commit", event_id=ident, event_kind=kind, value=value)
        try:
            while not event.wait(0.05):
                if cancelled.is_set() or monotonic() - parent_heartbeat.value > 3:
                    raise RuntimeError("任务租约已失效，账本未确认，禁止继续输入")
        finally:
            with ack_lock:
                acks.pop(ident, None)

    def finish(reason):
        nonlocal runtime, pipeline, finishing
        if finishing:
            return
        finishing = True
        gate.revoke()
        run_folder = runtime.store.folder if runtime and runtime.store else None
        folder = pipeline.session.folder if pipeline else None
        try:
            if pipeline:
                pipeline.begin_finishing()
            if runtime:
                runtime.close(evaluate=False)
            bot.attach(None)
            if pipeline:
                try:
                    pipeline.stop(reason)
                finally:
                    pipeline.session.close()
                if run_folder:
                    evaluate_run(run_folder)
                result = pipeline.finalize_recording()
                send("session_closed", folder=str(folder), replay=str(result.replay_folder or ""))
            if run_folder:
                refresh_run_retention(run_folder)
            send("finished")
        except Exception as exc:
            send("fault", code="shutdown", scope="session", message=f"执行会话收尾失败：{exc}")
        finally:
            bot.attach(None)
            runtime = pipeline = None
            finishing = False

    def probe(command):
        if not storage_ready():
            send("probe", ready=False, reason="等待磁盘空间或录像配额恢复")
            return
        if bot.model_status().state == "failed":
            bot.start()
        client = AdbClient(store.adb_path, command["serial"], timeout=3)
        devices = client.list_devices()
        if not any(d.serial == command["serial"] and d.state == "device" for d in devices):
            send("probe", ready=False, reason="等待原设备恢复在线授权状态")
            return
        window = client._run(["shell", "dumpsys", "window"]).decode("utf8", errors="replace")
        power = client._run(["shell", "dumpsys", "power"]).decode("utf8", errors="replace")
        if not foreground(window, power):
            send("probe", ready=False, reason="等待原游戏恢复前台及亮屏状态")
            return
        frame = client.capture()
        flow = FlowRecognizer(root).observe(frame, room="当前")
        blocked = command.get("blocked_flow")
        ready = bool(
            bot.ready.is_set()
            and not flow.blocked_reason
            and flow.scene != "unknown"
            and not (blocked and flow.scene == blocked["origin_scene"])
        )
        send(
            "probe",
            ready=ready,
            reason=flow.blocked_reason
            or (
                "旧页面命令仍未取得推进证据"
                if blocked and flow.scene == blocked["origin_scene"]
                else "等待页面或本地模型恢复"
            ),
            scene=flow.scene,
        )

    def storage_ready():
        return (
            shutil.disk_usage(root).free >= 2 * 1024**3
            and sum(p.stat().st_size for p in (root / "sessions").rglob("*.mkv")) < 20 * 1024**3
        )

    try:
        bot.start()
        while not cancelled.is_set():
            if monotonic() - parent_heartbeat.value > 3:
                fault_sent = True
                if pipeline:
                    pipeline.report_failure("parent_heartbeat", "父界面心跳已失效，按故障保留当前会话")
                break
            while connection.poll(0):
                command = connection.recv()
                kind = command["kind"]
                if kind == "ack":
                    with ack_lock:
                        event = acks.get(command["event_id"])
                        if event:
                            event.set()
                elif kind == "close":
                    cancelled.set()
                elif kind == "switch":
                    if bot.play_model == command["backend"] and bot.model_status().state == "failed":
                        bot.start()
                    else:
                        bot.switch_model(command["backend"])
                    store.profile.play_model = command["backend"]
                elif kind == "probe":
                    task_id = command["task_id"]
                    try:
                        probe(command)
                    except Exception as exc:
                        send("probe", ready=False, reason=f"等待设备、页面或本地资源恢复：{exc}")
                elif kind == "start":
                    if runtime:
                        continue
                    task_id = command["task_id"]
                    fault_sent = False
                    if not storage_ready():
                        fault_sent = True
                        send(
                            "fault", scope="external", code="disk_space", message="等待磁盘空间或录像配额恢复"
                        )
                        continue
                    store.profile = command["profile"]
                    store.profile.continuous_settings = {"max_games": command["remaining"]}
                    pipeline = RecordingPipeline(store, command["serial"], automatic=True)
                    # 创建后立即登记归属，启动中失败也不会丢失其异常资料或允许删除历史目录。
                    send("session_created", folder=str(pipeline.session.folder))
                    pipeline.start()
                    gate.allow()
                    runtime = ContinuousRuntime(
                        root,
                        gate,
                        bot,
                        pipeline,
                        unattended=True,
                        commit=commit,
                        blocked_flow=command.get("blocked_flow"),
                    )
                    runtime.start()
                    send("session", folder=str(pipeline.session.folder), run_folder=str(runtime.store.folder))
                elif kind == "finish":
                    finish(StopReason.COMPLETED)
                elif (
                    kind == "rotate_media" and pipeline and str(pipeline.session.folder) == command["folder"]
                ):
                    pipeline.stream.command("rotate_media", names=command["names"])
                    pipeline.evidence.rotate_window(command["start_us"], command["end_us"])
            if cancelled.is_set():
                break
            gate.heartbeat()
            now = monotonic()
            if now - last_snapshot >= 0.2:
                last_snapshot = now
                state = {}
                frame = None
                if pipeline and not finishing:
                    snapshot = pipeline.poll()
                    pipeline._gui_at = parent_heartbeat.value
                    frame = snapshot.get("frame")
                    raw_state = snapshot.get("state")
                    state = (
                        raw_state if isinstance(raw_state, dict) else asdict(raw_state) if raw_state else {}
                    )
                    if snapshot.get("stopped") and not fault_sent:
                        fault_sent = True
                        fault = snapshot.get("primary_fault") or {}
                        send(
                            "fault",
                            scope="external"
                            if fault.get("code") in ("disk_space", "video_quota", "device", "foreground")
                            else "session",
                            code=fault.get("code", "pipeline"),
                            message=fault.get("message", "采集或识别会话中断"),
                        )
                progress = asdict(runtime.progress()) if runtime else None
                if runtime and runtime.fault and not fault_sent:
                    fault_sent = True
                    send(
                        "fault",
                        **asdict(runtime.fault),
                        blocked_flow=runtime.pending_flow or runtime.blocked_flow,
                    )
                data = {
                    "model": asdict(bot.model_status()),
                    "progress": progress,
                    "state": state,
                    "suggestion": asdict(bot.suggestion) if bot.suggestion else None,
                    "updated_at": now,
                }
                if frame and frame.frame_id != last_frame:
                    last_frame = frame.frame_id
                    ok, jpeg = cv2.imencode(".jpg", cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR))
                    if ok:
                        # 不使用 asdict(frame)，避免复制数兆字节原图；只有预览压缩，原录像不变。
                        data["frame_meta"] = {k: v for k, v in vars(frame).items() if k != "rgb"}
                        data["jpeg"] = jpeg.tobytes()
                send("snapshot", **data)
            cancelled.wait(0.02)
    except (EOFError, BrokenPipeError):
        fault_sent = True
    except Exception as exc:
        fault_sent = True
        if pipeline:
            try:
                pipeline.report_failure("session_exception", f"执行进程异常：{exc}")
            except Exception:
                pass
        try:
            send("fault", code="session_exception", scope="session", message=f"执行进程异常：{exc}")
            while not cancelled.wait(0.05) and monotonic() - parent_heartbeat.value < 3:
                pass
        except (EOFError, OSError):
            pass
    finally:
        gate.suspend()
        try:
            if runtime or pipeline:
                parent_fault = config.get("_faulted")
                finish(
                    StopReason.FAULT
                    if fault_sent or parent_fault and parent_fault.is_set()
                    else StopReason.USER
                )
        finally:
            bot.close()
            connection.close()
