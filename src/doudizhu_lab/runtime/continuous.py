"""连续流程协调：页面事务与单局出牌共享输入闸门，任何未知结果都不盲目重试。"""

from __future__ import annotations

from collections import Counter
from threading import Event, RLock, Thread
from time import monotonic
from uuid import uuid4

from doudizhu_lab.agents.bidding import BidAgent, BiddingController, decision_dict, verify_bid_model
from doudizhu_lab.devices.adb import InputRejected
from doudizhu_lab.storage.runs import RunStore, evaluate_run
from doudizhu_lab.vision.flow import SCENE_NAMES


class ContinuousRuntime:
    """一次显式启动对应一个连续任务；支持注入时钟及无后台线程的确定性模拟。

    pipeline 必须已录制，并提供同帧 flow_snapshot/bot_snapshot；bot 是生产单局协调器。
    start 不授予跨重启权限。暂停只撤销后续输入；已发命令保留在途记录，不能撤回。
    """

    def __init__(self, root, gate, bot, pipeline, *, clock=monotonic, threaded=True, bid_agent=None):
        self.root, self.gate, self.bot, self.pipeline = root, gate, bot, pipeline
        self.clock, self.threaded = clock, threaded
        self.lock, self.stop_event = RLock(), Event()
        self.thread = None
        self.status, self.reason, self.scene = "idle", "", "unknown"
        self.store = None
        self.settings = dict(pipeline.profile.continuous_settings)
        self.completed, self.registered = [], set()
        self.current_game, self.armed_game = "", ""
        self.last_flow, self.pending_flow = None, None
        self.last_id, self.last_new_at = "", 0.0
        self.scene_epoch, self.sent_opportunities = 0, set()
        self._last_known_scene = ""
        self._seen_deal = ""
        self.candidate = None
        self.started = self.scene_since = self.clock()
        self.identity = None
        self.mode_verified = False
        self._finish_wait = None
        self._ready_since = None
        self.close_window_requested = False
        self.stop_recording_requested = False
        self._stop_request_handled = False
        self.bid_agent, self.bidding = bid_agent, None
        self.bid_decision, self.bid_description = None, ""

    @property
    def active(self):
        return self.status == "running"

    @property
    def description(self):
        """供GUI直接显示的中文进度；不在后台线程访问任何Qt对象。"""
        label = {
            "idle": "未开始",
            "running": "运行中",
            "paused": "已暂停",
            "completed": "已完成",
            "stopped": "已停止",
        }[self.status]
        return (
            f"连续任务：{label} · {SCENE_NAMES[self.scene]} · {len(self.completed)}/{self.settings['max_games']}局 · {max(0, self.clock() - self.started):.0f}秒"
            + (f" · {self.reason}" if self.reason else "")
            + (f" · {self.bid_description}" if self.scene == "bidding" and self.bid_description else "")
        )

    def start(self):
        """在已有录像上启动并预热；实际页面输入必须等待模型、前台与稳定观察全部就绪。"""
        with self.lock:
            if self.status != "idle":
                raise ValueError("任务已创建；暂停后请明确恢复，重新运行请创建新任务")
            self.store = RunStore.create(self.root, self.pipeline)
            self.status, self.started = "running", self.clock()
            self.scene_since = self.last_new_at = self.started
            self.bot.attach(self.pipeline)
            self.bot.set_mode("suggest")
            self.bot.start()
            settings = self.pipeline.profile.bid_settings
            agent = self.bid_agent
            if settings["mode"] == "model" and agent is None:
                agent = BidAgent(self.root / "models/bidding")
                try:
                    manifest = verify_bid_model(agent.folder)
                    self.pipeline.session.snapshot_bidding_model(agent.folder)
                    self.store.set("bidding_model", manifest)
                except (ValueError, OSError) as exc:
                    self.store.add("bidding_unavailable", reason=str(exc))
            self.bidding = BiddingController(settings, agent, threaded=self.threaded)
            if self.threaded:
                self.thread = Thread(target=self._run, name="连续斗地主流程", daemon=True)
                self.thread.start()
            else:
                self.pipeline.enable_flow()

    def _run(self):
        try:
            self.pipeline.enable_flow()
            while not self.stop_event.wait(0.04):
                self.poll_stop_request()
                if self.stop_event.is_set():
                    break
                if self.active:
                    self.advance(self.pipeline.flow_snapshot(), self.pipeline.bot_snapshot())
        except Exception as exc:
            self.pause(f"连续任务异常：{exc}")
        finally:
            self.gate.revoke()
            if self.store:
                evaluate_run(self.store.folder)

    def poll_stop_request(self):
        """应用自己的只减权停止接口；已完成任务也允许GUI轮询后正常关闭，不发送手机输入。"""
        if self.store is None or self._stop_request_handled:
            return
        request = self.store.folder / "停止请求.json"
        if request.is_file():
            import json

            value = json.loads(request.read_text(encoding="utf8"))
            self._stop_request_handled = True
            self.close_window_requested = bool(value.get("close_window"))
            self.stop_recording_requested = True
            self.stop("命令行请求停止连续任务")

    def pause(self, reason):
        """立即撤销流程及出牌授权；暂停原因与在途动作保留，不能自动进入下一局。"""
        with self.lock:
            if self.status != "running":
                return
            self.gate.revoke()
            self.bot.set_mode("stop")
            self.status, self.reason = "paused", reason
            self.store.set("status", self.status)
            self.store.add("paused", reason=reason, game_id=self.current_game, at=self.clock())
            if self.last_flow:
                self.pipeline.evidence.submit(self.last_flow.frame, ["continuous-pause"])

    def stop(self, reason="用户停止连续任务", *, manual=True):
        """停止后不再恢复本任务；先撤销输入，再让线程和录像分别收尾。"""
        with self.lock:
            self.gate.revoke()
            self.bot.set_mode("stop")
            if self.status not in ("completed", "stopped", "idle"):
                self.status, self.reason = "stopped", reason
                self.store.set("status", self.status)
                self.store.add("manual_takeover" if manual else "stopped", reason=reason, at=self.clock())
            self.stop_event.set()

    def resume(self):
        """用户显式恢复前核查实时源、页面及历史；未知提交和有缺口的本局拒绝恢复。"""
        with self.lock:
            if self.status != "paused":
                raise ValueError("当前任务不是暂停状态")
            flow, snapshot = self.pipeline.flow_snapshot(), self.pipeline.bot_snapshot()
            if (
                not flow
                or flow.blocked_reason
                or flow.scene == "unknown"
                or not self.pipeline.allowed
                or self.clock() - flow.frame.started > 1
            ):
                raise ValueError("恢复前需要清晰、在线且可控制的新画面")
            if snapshot and snapshot.state["game_id"] == self.current_game and not snapshot.state["ended"]:
                state = snapshot.state
                if state["pending"] or state["capture_incomplete"]:
                    raise ValueError("本局存在真实历史缺口，不能恢复机器人出牌")
                if (state["game_id"], state["turn"]) in self.bot.executor.sent_turns:
                    raise ValueError("本轮已有未确认提交，禁止恢复后重复提交")
            if self.pending_flow:
                raise ValueError("存在未确认页面命令，请先人工处理并开始新任务")
            self.status, self.reason = "running", ""
            self.last_id, self.last_new_at = "", self.clock()
            self.armed_game = ""
            self.store.set("status", "running")
            self.store.add("resumed", at=self.clock())

    def close(self):
        """退出时等待在途ADB返回后再关账本；不让迟到记录写入下一任务。"""
        self.stop(manual=False)
        if self.thread:
            self.thread.join(10)
            if self.thread.is_alive():
                raise RuntimeError("连续任务后台尚未退出，账本保持打开")
        if self.bidding:
            self.bidding.close()
            self.bidding = None
        if self.store:
            evaluate_run(self.store.folder)
            self.store.close()
            self.store = None

    def _complete(self, reason):
        self.gate.revoke()
        self.bot.set_mode("suggest")
        self.status, self.reason = "completed", reason
        self.store.set("status", self.status)
        self.store.add("finished", reason=reason, at=self.clock())
        self.stop_event.set()

    def _limited(self):
        return (
            len(self.completed) >= self.settings["max_games"]
            or self.clock() - self.started >= self.settings["time_limit"]
        )

    def advance(self, flow, snapshot):
        """消费一个最新同帧观察；相同帧不增加稳定次数，超时由单调时钟而非帧数判断。"""
        with self.lock:
            if not self.active:
                return
            now = self.clock()
            if not self.bot.ready.is_set():
                if self.bot.thread and not self.bot.thread.is_alive():
                    self.pause(self.bot.status)
                return
            self._ready_since = self._ready_since or now
            # 阻断弹窗即使OCR耗时超过点击时效，也只能用来撤销输入，不能被“画面过期”
            # 丢弃后继续沿用旧牌局。仅接受当前设备、视频和方向的证据，防止跨会话串用。
            if flow is not None and flow.blocked_reason:
                blocked_identity = (
                    flow.frame.serial,
                    flow.frame.stream_id,
                    flow.frame.width,
                    flow.frame.height,
                )
                if self.identity is None or blocked_identity == self.identity:
                    self.store.observation(flow)
                    self.pause(flow.blocked_reason)
                    return
            if now - self.gate.heartbeat_at > 1 or self.pipeline.stop_event.is_set():
                self.pause("界面心跳或视频已中断")
                return
            if not self.pipeline.allowed:
                if self.identity is not None or now - self._ready_since > 20:
                    self.pause("游戏前台或设备状态失效")
                return
            recording = getattr(self.pipeline, "record_ready", None)
            if recording is not None and not recording.is_set():
                if now - self._ready_since > 20:
                    self.pause("录像首个关键帧尚未落盘，不能开始匹配")
                return
            # 首次还需冷启动流程OCR，不能把它与运行中的视频中断混为一谈。
            # 最多预热20秒，期间绝不点击；一旦接受第一帧，恢复正常三秒中断保护。
            if self.identity is None and (flow is None or now - flow.frame.started > 1):
                if now - self._ready_since > 20:
                    self.pause("流程识别初始化后仍没有清晰首帧")
                return
            if flow is None or flow.frame.frame_id == self.last_id:
                if now - max(self.last_new_at, self._ready_since) > 3:
                    self.pause("连续流程超过三秒没有新观察")
                return
            if now - flow.frame.started > 1:
                if now - self.last_new_at > 3:
                    self.pause("流程画面持续过期")
                return
            identity = (flow.frame.serial, flow.frame.stream_id, flow.frame.width, flow.frame.height)
            if self.identity and identity != self.identity:
                self.pause("设备、视频会话或方向发生变化")
                return
            self.identity = identity
            self.last_id, self.last_new_at, self.last_flow = flow.frame.frame_id, now, flow
            self.store.observation(flow)
            if snapshot and flow.scene in ("dealing", "bidding", "doubling"):
                deal = snapshot.state["game_id"]
                if deal and deal != self._seen_deal:
                    self._seen_deal = deal
                    self.scene_epoch += 1
                    self.candidate = None
                    self.store.add(
                        "deal_attempt",
                        game_id=deal,
                        frame_id=flow.frame.frame_id,
                        pts_us=flow.frame.media_pts_us,
                    )
            if flow.blocked_reason:
                self.pause(flow.blocked_reason)
                return
            if (
                snapshot
                and self.armed_game == snapshot.state["game_id"]
                and (snapshot.state["pending"] or snapshot.state["capture_incomplete"])
            ):
                self.pause("本局出现真实历史缺口或矛盾")
                return
            if flow.scene != self.scene:
                self.scene, self.scene_since = flow.scene, now
                if flow.scene != "unknown" and flow.scene != self._last_known_scene:
                    self.scene_epoch += 1
                    self._last_known_scene = flow.scene
                self.candidate = None
                self.store.add(
                    "scene", scene=flow.scene, frame_id=flow.frame.frame_id, pts_us=flow.frame.media_pts_us
                )
                self.pipeline.evidence.submit(flow.frame, [f"flow-{flow.scene}"])
            if self.pending_flow:
                pending = self.pending_flow
                # 平衡模式在旧按钮已消失时优先继续取证，最多8秒；仅消失并不等于成功。
                # 明确的新页面或本人反馈才确认，等待期间不重复点击，也不补造出牌历史。
                balanced = self.pipeline.profile.continuous_policy != "strict"
                evidence_window = 8 if balanced else 3
                present = any(
                    b.kind == pending["kind"]
                    and b.label == pending["label"]
                    and abs(b.hit.x - pending["button_box"][0]) < 15
                    and abs(b.hit.y - pending["button_box"][1]) < 15
                    for b in flow.buttons
                )
                expected = {
                    "match": {"matching", "dealing", "bidding", "doubling", "playing", "room_confirmation"},
                    "continue": {"lobby", "room", "matching", "dealing", "bidding", "doubling", "playing"},
                    "return_lobby": {"lobby", "room"},
                    "decline_bid": {"bidding", "dealing", "doubling", "playing"},
                    "call_landlord": {"bidding", "dealing", "doubling", "playing"},
                    "rob_landlord": {"bidding", "dealing", "doubling", "playing"},
                    "decline_double": {"dealing", "doubling", "playing"},
                    "confirm_room": {"lobby", "room", "matching", "dealing", "bidding"},
                    "close_showcase": {"settlement", "lobby", "room"},
                }[pending["kind"]]
                feedback = (
                    pending["kind"] in flow.feedbacks
                    and flow.frame.started - pending["sent_at"] <= evidence_window
                )
                if not present and (flow.scene in expected or feedback):
                    advanced = (
                        flow.scene != pending["origin_scene"]
                        and flow.frame.started - pending["sent_at"] <= evidence_window
                    )
                    # 加倍按钮收起后，短动画可能只留下牌桌外壳，被保守归为dealing。
                    # 这不是阶段推进证据，只允许通过两个新帧的按钮消失确认，不猜OCR近似字。
                    if pending["kind"] == "decline_double" and flow.scene == "dealing":
                        advanced = False
                    if advanced or feedback:
                        pending["absent_us"] = flow.frame.media_pts_us - 100_000
                    elif pending.get("absent_us") is None:
                        pending["absent_us"] = flow.frame.media_pts_us
                    if flow.frame.media_pts_us - pending["absent_us"] >= 100_000:
                        self.store.add(
                            "flow_confirmed",
                            action_id=pending["action_id"],
                            frame_id=flow.frame.frame_id,
                            pts_us=flow.frame.media_pts_us,
                            evidence="own_feedback"
                            if feedback
                            else ("scene_change" if advanced else "button_disappearance"),
                        )
                        if pending.get("bidding"):
                            self.bidding.committed(self.bid_decision, pending["bid_deal"])
                        self.pending_flow, self.candidate = None, None
                else:
                    pending["absent_us"] = None
                if self.pending_flow:
                    wait_for_evidence = (
                        balanced
                        and not present
                        and (flow.scene == "unknown" or pending.get("absent_us") is not None)
                    )
                    if wait_for_evidence and not pending.get("evidence_wait_logged"):
                        pending["evidence_wait_logged"] = True
                        self.store.add(
                            "flow_evidence_wait",
                            action_id=pending["action_id"],
                            frame_id=flow.frame.frame_id,
                            pts_us=flow.frame.media_pts_us,
                            reason="旧按钮消失，等待新阶段或明确反馈",
                            deadline_seconds=8,
                        )
                    if now - pending["sent_at"] > (8 if wait_for_evidence else 3):
                        self.pause("页面操作在证据等待窗口内仍无法唯一确认，禁止重复点击")
                    return
            if flow.scene == "unknown":
                if now - self.scene_since > 5:
                    self.pause("未知页面持续五秒")
                return
            if flow.scene == "matching":
                if now - self.scene_since > 120:
                    self.pause("匹配等待超过120秒")
                return
            if flow.scene == "playing":
                if not self.mode_verified:
                    self.pause("连续任务须从已确认场次的大厅开始，不能中途接管未知牌局")
                    return
                self._playing(snapshot)
                return
            if flow.scene == "ending" and now - self.scene_since > 10:
                self.pause("结算已确认，但十秒内没有出现可确认的结算按钮")
                return
            if flow.scene in ("settlement", "showcase", "ending"):
                if not self._settlement(snapshot):
                    return
            if flow.scene in ("lobby", "room"):
                if self._limited():
                    self._complete("已达到局数或时间上限，不再匹配")
                    return
                if not flow.mode_verified:
                    self.pause("无法确认经典洗牌与当前场次")
                    return
                self.mode_verified = True
            if flow.scene in ("settlement", "showcase", "ending") and self._limited():
                self._complete("当前局结束，已达到局数或时间上限")
                return
            preferences = {
                "lobby": ("match",),
                "room": ("match",),
                "bidding": ("decline_bid",),
                "doubling": ("decline_double",),
                "settlement": ("continue", "return_lobby"),
                "room_confirmation": ("confirm_room",),
                "showcase": ("close_showcase",),
            }
            bid_decision = None
            bid_deal = (
                getattr(flow, "bid_deal_id", "") or self._seen_deal or f"before-play-{len(self.completed)}"
            )
            if flow.scene == "bidding":
                bid_decision = self.bidding.choose(flow, bid_deal)
                if bid_decision is None:
                    self.bid_description = "等待叫抢手牌及模型评估"
                    self.candidate = None
                    return
                self.bid_decision = bid_decision
                self.bid_description = (
                    f"{'叫地主' if bid_decision.kind == 'call_landlord' else '抢地主' if bid_decision.kind == 'rob_landlord' else '不叫/不抢'}"
                    + (
                        f" · 评分{bid_decision.score:.3f} / 阈值{bid_decision.threshold:.3f}"
                        if bid_decision.score is not None
                        else f" · {bid_decision.reason}"
                    )
                )
                preferences["bidding"] = (bid_decision.kind,)
            if flow.scene == "settlement" and not self.mode_verified:
                preferences["settlement"] = ("return_lobby",)
            button = next(
                (
                    b
                    for kind in preferences.get(flow.scene, ())
                    for b in flow.buttons
                    if b.kind == kind and b.enabled
                ),
                None,
            )
            if button is None:
                self.candidate = None
                return
            if not self.mode_verified and button.kind not in (
                "return_lobby",
                "confirm_room",
                "close_showcase",
            ):
                self.pause("任务必须先从已确认场次的大厅开始")
                return
            opportunity = (
                bid_decision.opportunity
                if bid_decision
                else f"{self.scene_epoch}:{button.kind}:{button.label}"
            )
            if opportunity in self.sent_opportunities:
                return
            signature = (opportunity, button.hit.x // 15, button.hit.y // 15)
            if not self.candidate or self.candidate[0] != signature:
                self.candidate = (signature, flow.frame.media_pts_us)
                return
            if flow.frame.media_pts_us - self.candidate[1] < 100_000:
                return
            if bid_decision:
                self.store.add(
                    "bidding_decision",
                    **decision_dict(bid_decision),
                    frame_id=flow.frame.frame_id,
                    pts_us=flow.frame.media_pts_us,
                    public_markers=list(flow.bid_markers),
                )
            self._tap(button, flow, opportunity, bidding=bid_decision, bid_deal=bid_deal)

    def _tap(self, button, flow, opportunity, *, bidding=None, bid_deal=""):
        if self.bot.executor.active:
            self.pause("页面输入与在途出牌冲突")
            return
        self.bot.set_mode("suggest")
        token = self.gate.grant("flow")
        action_id = uuid4().hex
        record = dict(
            action_id=action_id,
            opportunity=opportunity,
            kind=button.kind,
            label=button.label,
            frame_id=flow.frame.frame_id,
            pts_us=flow.frame.media_pts_us,
            sent_at=self.clock(),
            origin_scene=flow.scene,
            button_box=[button.hit.x, button.hit.y, button.hit.width, button.hit.height],
            x=button.hit.x + button.hit.width // 2,
            y=button.hit.y + button.hit.height // 2,
        )
        if bidding:
            record.update(bidding=decision_dict(bidding), bid_deal=bid_deal)
        self.pending_flow = record.copy()
        self.sent_opportunities.add(opportunity)
        self.store.add("flow_sent", **record)
        self.pipeline.evidence.submit(flow.frame, [action_id])

        def guard():
            newest = self.pipeline.flow_snapshot()
            raw = self.pipeline.frame
            return bool(
                self.active
                and self.pipeline.allowed
                and not self.pipeline.stop_event.is_set()
                and newest
                and newest.scene == flow.scene
                and (
                    not bidding
                    or bidding.kind == "decline_bid"
                    or (
                        Counter(newest.bid_hand or ()) == Counter(bidding.hand)
                        and newest.bid_deal_id == flow.bid_deal_id
                    )
                )
                and raw
                and (raw.serial, raw.stream_id, raw.width, raw.height) == self.identity
                and any(
                    b.kind == button.kind
                    and b.label == button.label
                    and b.enabled
                    and abs(b.hit.x - button.hit.x) < 15
                    and abs(b.hit.y - button.hit.y) < 15
                    for b in newest.buttons
                )
            )

        # advance持有状态锁；设备命令可能等待USB，发送前释放，确保GUI停止不被它阻塞。
        self.lock.release()
        try:
            box = button.hit
            self.gate.tap(
                "flow",
                token,
                getattr(self.pipeline, "control_client", self.pipeline.client),
                box.x + box.width // 2,
                box.y + box.height // 2,
                flow.frame,
                guard=guard,
                cancelled=lambda: not self.active,
            )
            self.store.add("flow_returned", action_id=action_id, at=self.clock())
        except InputRejected as exc:
            self.store.add("flow_not_sent", action_id=action_id, message=str(exc))
            self.pending_flow = None
            self.sent_opportunities.discard(opportunity)
            self.pause(f"页面输入未发送：{exc}")
        except Exception as exc:
            self.store.add("flow_unknown", action_id=action_id, message=str(exc))
            self.pause(f"页面命令结果未知：{exc}")
        finally:
            self.lock.acquire()

    def _playing(self, snapshot):
        if snapshot is None:
            return
        state = snapshot.state
        if state["ended"] or state["phase"] != "playing":
            return
        if state["pending"] or state["capture_incomplete"]:
            self.pause("出牌前发现历史缺口，不能用猜测状态启动")
            return
        if self.armed_game == state["game_id"]:
            if self.bot.mode == "stop":
                self.pause(self.bot.pause_reason or self.bot.executor.reason or self.bot.status)
            return
        if not state["landlord"] or not state["bottom"] or not state["initial_hand"]:
            # 新旧局切换的跟踪候选可能早于稳定的身份/底牌快照；先等待取证，不猜底牌。
            startup_wait = 8 if self.pipeline.profile.continuous_policy != "strict" else 3
            if self.clock() - self.scene_since > startup_wait:
                self.pause("开局取证窗口结束后身份、底牌或手牌仍不完整")
            return
        self.current_game = self.armed_game = state["game_id"]
        self._finish_wait = None
        self.store.add(
            "game_armed",
            game_id=self.current_game,
            frame_id=snapshot.frame.frame_id,
            pts_us=snapshot.frame.media_pts_us,
        )
        self.bot.arm_game(self.current_game)

    def _settlement(self, snapshot):
        if not self.current_game:
            return True  # 可从大厅外的已结束页面返回大厅，不能把这局计为完成。
        if self.current_game in self.registered:
            return True
        self._finish_wait = self._finish_wait or self.clock()
        if (
            snapshot is None
            or not snapshot.state["ended"]
            or snapshot.state["game_id"] != self.current_game
            or self.bot.executor.active
        ):
            if self.clock() - self._finish_wait > 3:
                self.pause("结算后末手或牌局边界无法确认")
            return False
        state = snapshot.state
        actions = self.pipeline.session.rows("bot_actions")
        starts = {
            a["action_id"]: a.get("decision", {})
            for a in actions
            if a.get("game_id") == self.current_game and a["stage"] == "开始"
        }
        confirmed = {
            starts[a["action_id"]].get("turn")
            for a in actions
            if a.get("game_id") == self.current_game and a["stage"] == "已确认" and a["action_id"] in starts
        }
        moves = [e for e in snapshot.history if e["kind"] in ("play", "pass")]
        expected = {i for i, e in enumerate(moves) if e["seat"] == "self"}
        verified = bool(
            state["complete"]
            and not state["pending"]
            and not state["capture_incomplete"]
            and confirmed == expected
            and expected
        )
        rule_game = state.get("strategy") == "rules"
        confirmed_ids = {
            a["action_id"]
            for a in actions
            if a.get("game_id") == self.current_game and a["stage"] == "已确认"
        }
        if rule_game and self.pipeline.profile.continuous_policy == "balanced":
            # 局部规则不宣称补齐原前史；完成依据改为明确结算、本人手牌监控及全部提交确认。
            wp_turns = {
                d["turn"]
                for ident, d in starts.items()
                if ident in confirmed_ids and d.get("strategy", "wp") == "wp"
            }
            verified = bool(
                not state.get("rule_halt_reason")
                and not state["capture_incomplete"]
                and starts
                and set(starts) == confirmed_ids
                and wp_turns == expected
            )
        if not verified:
            if self.clock() - self._finish_wait > 3:
                self.pause("本局存在未确认出牌、托管或不完整历史，不能计为机器人完成")
            return False
        start = next(
            (e["pts_us"] for e in snapshot.history if e["kind"] == "game_start"), snapshot.frame.media_pts_us
        )
        self.store.add(
            "game_complete",
            game_id=self.current_game,
            start_us=start,
            end_us=snapshot.frame.media_pts_us,
            verified=True,
            own_actions=len(confirmed_ids) if rule_game else len(expected),
            confirmed_actions=len(confirmed_ids) if rule_game else len(confirmed),
            degraded=rule_game,
            strict_complete=bool(
                state["complete"]
                and not rule_game
                and not any(e.get("source") == "operational_inferred" for e in snapshot.history)
            ),
            inferred_events=sum(e.get("source") == "operational_inferred" for e in snapshot.history),
            strategy_counts=dict(Counter(d.get("strategy", "wp") for d in starts.values())),
            frame_id=snapshot.frame.frame_id,
        )
        self.registered.add(self.current_game)
        self.completed.append(self.current_game)
        self.bot.set_mode("suggest")
        return True
