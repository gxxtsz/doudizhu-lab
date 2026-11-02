"""固定 ResNet 的参考编码、配置迁移、线程切换及真实模型回归；不连接手机。"""

from __future__ import annotations

import ast
import hashlib
import io
import json
from collections import Counter
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from doudizhu_lab.agents.backends import RESNET_WEIGHTS, create_agent, setup_play_model, verify_resnet_models
from doudizhu_lab.agents.douzero import DECK, ROLES, context
from doudizhu_lab.agents.resnet_features import COMPATIBILITY_FEATURES, get_obs
from doudizhu_lab.domain.settings import Profile, validate_bot_profile
from doudizhu_lab.paths import PACKAGE_ROOT
from doudizhu_lab.runtime.bot import BotRuntime, InputGate
from doudizhu_lab.runtime.executor import execution_context
from doudizhu_lab.simulation.bot import SimulatedDevice, simulate
from doudizhu_lab.storage.config import Store

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def upstream_encoder():
    """只执行固定原始源码中的纯编码函数；不导入上游环境、设备或共享库。"""
    tree = ast.parse((FIXTURES / "resnet_env.py.txt").read_text(encoding="utf8"))
    functions = {
        "_get_one_hot_array",
        "_cards2array",
        "_action_seq_list2array",
        "_process_action_seq",
        "_get_one_hot_bomb",
        "_get_obs_resnet",
    }
    nodes = [
        node
        for node in tree.body
        if (
            isinstance(node, ast.FunctionDef)
            and node.name in functions
            or isinstance(node, ast.Assign)
            and any(
                isinstance(t, ast.Name) and t.id in ("Card2Column", "NumOnes2Array") for t in node.targets
            )
        )
    ]
    namespace = {"np": np, "Counter": Counter}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "固定上游编码参考", "exec"), namespace)
    return namespace["_get_obs_resnet"]


def public_info(role="landlord", history_length=0):
    """提供可独立核对的公开编码输入；隐藏手牌属性一旦被生产编码访问就失败。"""

    class PublicOnly(SimpleNamespace):
        @property
        def all_handcards(self):
            raise AssertionError("生产编码不得访问各家的隐藏手牌")

    hand = [3, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 17, 17, 20, 30]
    if role == "landlord":
        hand += [4, 5, 6]
    return PublicOnly(
        player_position=role,
        player_hand_cards=hand,
        other_hand_cards=sorted((DECK - Counter(hand)).elements()),
        three_landlord_cards=[3, 4, 5],
        played_cards={r: [] for r in ROLES},
        num_cards_left_dict={r: 20 if r == "landlord" else 17 for r in ROLES},
        legal_actions=[[], [3], [3, 3]],
        card_play_action_seq=([[], [3], [4, 4]] * 16)[:history_length],
        last_move=[],
        bomb_num=0,
    )


@pytest.mark.parametrize("role", ROLES)
@pytest.mark.parametrize("length", [0, 1, 2, 31, 32, 39])
def test_features_match_fixed_upstream_without_hidden_hands(upstream_encoder, role, length):
    info = public_info(role, length)
    actual = get_obs(info)
    reference = SimpleNamespace(**vars(info))
    reference.bid_info = COMPATIBILITY_FEATURES[:12].reshape(4, 3)
    reference.multiply_info = COMPATIBILITY_FEATURES[12:]
    reference.all_handcards = dict.fromkeys(ROLES, [])
    expected = upstream_encoder(reference, role)
    np.testing.assert_array_equal(actual["z_batch"], expected["z_batch"])
    np.testing.assert_array_equal(actual["x_batch"], expected["x_batch"])
    assert actual["z_batch"].shape == (3, 40, 54)
    assert actual["x_batch"].shape == (3, 15)
    assert actual["x_batch"].dtype == actual["z_batch"].dtype == np.float32
    assert actual["x_batch"][0, 1] == 0.5


@pytest.mark.parametrize("bottom", [[], [3], [3, 4, 5]])
def test_remaining_bottom_matches_reference(upstream_encoder, bottom):
    info = public_info(history_length=2)
    info.three_landlord_cards = bottom
    actual = get_obs(info)
    reference = SimpleNamespace(
        **vars(info),
        bid_info=COMPATIBILITY_FEATURES[:12],
        multiply_info=[1, 1, 1],
        all_handcards=dict.fromkeys(ROLES, []),
    )
    np.testing.assert_array_equal(actual["z_batch"], upstream_encoder(reference, "landlord")["z_batch"])
    assert np.all(actual["z_batch"][:, 8:38] == -1)
    assert np.all(actual["z_batch"][:, 39] == 0)  # 真实不出不能与缺失历史混为一谈。


def test_context_tracks_remaining_bottom_and_both_models_use_strict_checks():
    device = SimulatedDevice(73100, "landlord")
    for _ in range(100):
        if device.env.game_over:
            break
        if device.env.acting_player_position == "landlord":
            info = context(device.last)
            assert Counter(info["three_landlord_cards"]) == Counter(device.env.three_landlord_cards)
            assert execution_context(device.last, "resnet") == execution_context(device.last, "wp")
        device.act(device.env.game_infoset.legal_actions[-1])
        for _ in range(3):
            device.render()
    changed = replace(device.last, state_json=json.dumps(device.last.state | {"capture_incomplete": True}))
    with pytest.raises(ValueError):
        execution_context(changed, "resnet")


def test_registry_defaults_to_resnet_and_old_agent_stays_wp(tmp_path):
    from doudizhu_lab.agents.douzero import Agent

    assert create_agent(tmp_path).backend == Profile().play_model == "resnet"
    assert create_agent(tmp_path, "wp").folder == (tmp_path / "models/douzero-wp").resolve()
    assert Agent(tmp_path).backend == "wp"
    for backend in ("resnet", "wp"):
        agent = create_agent(tmp_path, backend)
        with pytest.raises(OSError, match=f"setup-agent --model {backend}"):
            agent.start()
        assert agent.process is None
    with pytest.raises(ValueError):
        create_agent(tmp_path, "unknown")


@pytest.mark.parametrize(
    "result", [[1, 2], {"cards": [3.0], "value": 1.0}, {"cards": [], "value": float("inf")}]
)
def test_agent_rejects_invalid_protocol_cards_and_values(tmp_path, result):
    agent = create_agent(tmp_path)
    agent.process = Mock()
    agent.process.poll.return_value = None
    agent.responses.put(result)
    try:
        with pytest.raises(ValueError):
            agent.decide(SimulatedDevice(73100, "landlord").last)
    finally:
        agent.close()


def test_v4_config_upgrade_backup_and_model_validation(tmp_path):
    original = json.dumps({"version": 4, "profile": {"threshold": 0.91}}, ensure_ascii=False)
    target = tmp_path / "settings.json"
    target.write_text(original, encoding="utf8")
    store = Store(tmp_path)
    assert store.profile.play_model == "resnet" and target.read_text(encoding="utf8") == original
    store.profile.play_model = "wp"
    store.save()
    assert (tmp_path / "settings.v4.bak.json").read_text(encoding="utf8") == original
    assert Store(tmp_path).profile.play_model == "wp" and store.settings_version == 6
    profile = Profile(play_model="unknown")
    with pytest.raises(ValueError, match="出牌模型"):
        validate_bot_profile(profile)


def test_fixed_manifest_rejects_changed_origin_sizes_and_bytes(tmp_path):
    from doudizhu_lab.agents.backends import RESNET_COMMIT, RESNET_SCHEMA

    manifest = {
        "backend": "resnet",
        "source_commit": RESNET_COMMIT,
        "feature_schema": RESNET_SCHEMA,
        "weights": {
            role: {"bytes": size, "sha256": digest} for role, (size, digest) in RESNET_WEIGHTS.items()
        },
    }
    for role in ROLES:
        (tmp_path / f"{role}.ckpt").write_bytes(b"bad")
    target = tmp_path / "manifest.json"
    target.write_text(json.dumps(manifest), encoding="utf8")
    with pytest.raises(ValueError, match="权重文件校验失败"):
        verify_resnet_models(tmp_path)
    manifest["source_commit"] = "changed"
    target.write_text(json.dumps(manifest), encoding="utf8")
    with pytest.raises(ValueError, match="来源或特征"):
        verify_resnet_models(tmp_path)


def test_setup_only_publishes_after_warmup_and_reuses_verified_cache(tmp_path, monkeypatch):
    """固定微型资产替身仅验证安装事务；真实参数严格加载由模型用例验证。"""
    from doudizhu_lab.agents import backends

    content = {role: ("测试权重-" + role).encode() for role in ROLES}
    monkeypatch.setattr(
        backends,
        "RESNET_WEIGHTS",
        {role: (len(data), hashlib.sha256(data).hexdigest()) for role, data in content.items()},
    )
    folder = tmp_path / "models/douzero-resnet"
    folder.mkdir(parents=True)
    (folder / "manifest.json").write_text("原有清单", encoding="utf8")
    downloads = []

    def fetch(url, timeout):
        downloads.append(url)
        role = url.rsplit("/", 1)[1].removeprefix("resnet_").removesuffix(".ckpt")
        return io.BytesIO(content[role])

    monkeypatch.setattr(backends.urllib.request, "urlopen", fetch)
    fake = Mock(model_id="测试模型", ready_info={"ready": True})
    fake.start.side_effect = ValueError("预热失败")
    monkeypatch.setattr("doudizhu_lab.agents.douzero.Agent", lambda *args, **kwargs: fake)
    with pytest.raises(ValueError, match="预热失败"):
        setup_play_model(tmp_path)
    assert (folder / "manifest.json").read_text(encoding="utf8") == "原有清单"
    assert not list(folder.glob("*.ckpt")) and fake.close.called
    fake.start.side_effect = None
    setup_play_model(tmp_path)
    assert verify_resnet_models(folder)["backend"] == "resnet"
    before = len(downloads)
    setup_play_model(tmp_path)
    assert len(downloads) == before and (folder / "verification.json").is_file()


def test_cli_model_override_does_not_save_preferences(tmp_path, monkeypatch):
    import sys

    from doudizhu_lab.cli import app

    store = Store(tmp_path)
    store.save()
    before = store.settings_path.read_bytes()
    captured = []
    monkeypatch.setattr(sys, "argv", ["ddz-lab", "--data-dir", str(tmp_path), "--play-model", "wp"])
    monkeypatch.setattr(app, "QApplication", Mock(return_value=Mock()))
    monkeypatch.setattr(
        app, "MainWindow", lambda config, **kwargs: captured.append(config.profile.play_model) or Mock()
    )
    app.main()
    assert captured == ["wp"] and store.settings_path.read_bytes() == before


def test_resnet_provenance_and_references_have_fixed_hashes():
    folder = PACKAGE_ROOT / "_vendor/douzero_resnet"
    provenance = json.loads((folder / "provenance.json").read_text(encoding="utf8"))
    for entry in provenance["files"]:
        assert hashlib.sha256((folder / entry["file"]).read_bytes()).hexdigest() == entry["sha256"]
    for name, source in (
        ("models.py", "resnet_models.py.txt"),
        ("../../agents/resnet_features.py", "resnet_env.py.txt"),
    ):
        expected = next(e["upstream_sha256"] for e in provenance["files"] if e["file"] == name)
        assert hashlib.sha256((FIXTURES / source).read_bytes()).hexdigest() == expected


class LoadingAgent:
    """以事件精确控制加载先后，不依赖真实模型、网络或睡眠猜测。"""

    def __init__(self, backend, *, blocked=False):
        self.backend, self.model_id = backend, "测试-" + backend
        self.started, self.release = Event(), Event()
        self.closed = False
        if not blocked:
            self.release.set()

    def start(self):
        self.started.set()
        assert self.release.wait(4)

    def close(self):
        self.closed = True

    def interrupt(self):
        pass  # 模拟取消后仍迟到成功的旧加载任务。


@pytest.mark.gui
def test_switch_while_loading_discards_late_ready_and_closes_before_next_load(qtbot, tmp_path, monkeypatch):
    old, new = LoadingAgent("resnet", blocked=True), LoadingAgent("wp", blocked=True)
    monkeypatch.setattr(
        "doudizhu_lab.runtime.bot.create_agent", lambda root, backend: old if backend == "resnet" else new
    )
    runtime = BotRuntime(tmp_path, InputGate())
    runtime.start()
    try:
        qtbot.waitUntil(old.started.is_set)
        runtime.suggestion = object()
        runtime.switch_model("wp")
        assert runtime.suggestion is None and not runtime.ready.is_set() and runtime.gate.owner is None
        old.release.set()
        qtbot.waitUntil(new.started.is_set)
        assert old.closed and runtime.model_status().backend == "wp"
        assert runtime.model_status().state == "loading" and not runtime.ready.is_set()
        new.release.set()
        qtbot.waitUntil(runtime.ready.is_set)
        assert runtime.model_status().model_id == new.model_id
        runtime.set_mode("auto")
        with pytest.raises(ValueError, match="任务期间"):
            runtime.switch_model("resnet")
    finally:
        old.release.set()
        new.release.set()
        runtime.close()
    assert new.closed and not runtime.thread.is_alive()


@pytest.mark.gui
def test_switch_during_failed_worker_cleanup_still_starts_next_model(qtbot, tmp_path, monkeypatch):
    closing, finish = Event(), Event()

    class FailedAgent(LoadingAgent):
        def start(self):
            raise ValueError("加载失败")

        def close(self):
            closing.set()
            assert finish.wait(4)
            super().close()

    old, new = FailedAgent("resnet"), LoadingAgent("wp")
    monkeypatch.setattr(
        "doudizhu_lab.runtime.bot.create_agent", lambda root, backend: old if backend == "resnet" else new
    )
    runtime = BotRuntime(tmp_path, InputGate())
    runtime.start()
    try:
        qtbot.waitUntil(closing.is_set)
        assert runtime.model_status().state == "failed"
        runtime.switch_model("wp")
        finish.set()
        qtbot.waitUntil(runtime.ready.is_set)
        assert runtime.model_status().backend == "wp" and old.closed
    finally:
        finish.set()
        runtime.close()


@pytest.mark.gui
def test_settings_switch_persists_and_failed_save_leaves_current_model(qtbot, tmp_path, monkeypatch):
    from doudizhu_lab.ui.settings import SettingsDialog
    from doudizhu_lab.ui.window import MainWindow

    monkeypatch.setattr("doudizhu_lab.runtime.bot.create_agent", lambda root, backend: LoadingAgent(backend))
    window = MainWindow(Store(tmp_path), auto_refresh=False, auto_load_model=False, auto_preview=False)
    qtbot.addWidget(window)
    window.load_bot()
    qtbot.waitUntil(window.bot.ready.is_set)
    dialog = SettingsDialog(window)
    qtbot.addWidget(dialog)
    assert dialog.play_model.currentData() == "resnet"
    dialog.play_model.setCurrentIndex(dialog.play_model.findData("wp"))
    dialog.save()
    qtbot.waitUntil(lambda: window.bot.model_status().backend == "wp" and window.bot.ready.is_set())
    assert Store(tmp_path).profile.play_model == "wp"
    window.render_dashboard()
    assert window.model_label.text() == "WP · 已就绪"
    profile = deepcopy(window.store.profile)
    profile.play_model = "resnet"
    monkeypatch.setattr(window.store, "save", Mock(side_effect=OSError("保存失败")))
    with pytest.raises(OSError):
        window.apply_settings(profile, window.store.adb_path, "")
    assert window.store.profile.play_model == window.bot.play_model == "wp"
    window._requested_action = "continuous"
    with pytest.raises(ValueError, match="任务期间"):
        window.apply_settings(profile, window.store.adb_path, "")
    window._requested_action = ""
    window.close()


@pytest.mark.model
def test_real_resnet_all_roles_closed_loop_and_original_network_parity():
    import torch
    from torch import nn
    from torch.nn import functional as F

    from doudizhu_lab._vendor.douzero_resnet.models import ResnetModel
    from doudizhu_lab.agents.worker import score_actions

    root = Path(__file__).resolve().parents[1] / "data"
    if not (root / "models/douzero-resnet/manifest.json").exists():
        pytest.skip("本机尚未准备固定 ResNet 权重")
    torch.set_num_threads(2)
    tree = ast.parse((FIXTURES / "resnet_models.py.txt").read_text(encoding="utf8"))
    classes = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name in ("BasicBlock", "ResnetModel")
    ]
    namespace = {"torch": torch, "nn": nn, "F": F, "np": np}
    exec(compile(ast.Module(body=classes, type_ignores=[]), "固定上游网络参考", "exec"), namespace)
    agent = create_agent(root)
    try:
        agent.start()
        for role in ROLES:
            weights = torch.load(agent.folder / f"{role}.ckpt", map_location="cpu", weights_only=True)
            actual, reference = ResnetModel().eval(), namespace["ResnetModel"]().eval()
            actual.load_state_dict(weights, strict=True)
            reference.load_state_dict(weights, strict=True)
            info = public_info(role, 39)
            info.legal_actions *= 100
            obs = get_obs(info)
            with torch.inference_mode():
                expected = reference(
                    torch.from_numpy(obs["z_batch"]), torch.from_numpy(obs["x_batch"]), return_value=True
                )["values"].numpy()[:, 0]
            values = score_actions(actual, info, "resnet")
            np.testing.assert_allclose(values, expected, rtol=1e-5, atol=1e-6)
            assert values.argmax() == expected.argmax()
        result = simulate(agent, 3)
        assert all(
            row["passed"] and not row["invalid_submissions"] and not row["duplicate_submissions"]
            for row in result["games"]
        )
    finally:
        agent.close()
    assert agent.process is None


@pytest.mark.model
def test_worker_rejects_wrong_shape_nonfinite_and_missing_parameters(monkeypatch, capsys):
    import sys

    import torch

    from doudizhu_lab._vendor.douzero_resnet.models import ResnetModel
    from doudizhu_lab.agents import worker

    for output in (torch.zeros(3, 2), torch.full((3, 1), float("nan"))):
        with pytest.raises(ValueError, match="评分"):
            worker.score_actions(lambda *args, **kwargs: {"values": output}, public_info(), "resnet")
    weights = ResnetModel().state_dict()
    del weights["linear4.weight"]
    monkeypatch.setattr(worker, "verify_models", lambda *args: None)
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: weights)
    monkeypatch.setattr(torch, "set_num_interop_threads", lambda *args: None)
    monkeypatch.setattr(sys, "argv", ["worker", "unused", "resnet"])
    with pytest.raises(RuntimeError, match="Missing key"):
        worker.main()
    assert "error" in json.loads(capsys.readouterr().out)


@pytest.mark.model
@pytest.mark.gui
def test_real_model_switch_resnet_to_wp_and_back(qtbot):
    root = Path(__file__).resolve().parents[1] / "data"
    if not all((root / f"models/douzero-{name}/manifest.json").is_file() for name in ("wp", "resnet")):
        pytest.skip("本机需要同时准备两种固定模型")
    runtime = BotRuntime(root, InputGate())
    try:
        runtime.start()
        qtbot.waitUntil(runtime.ready.is_set, timeout=30000)
        original_id = runtime.model_status().model_id
        assert runtime.model_status().backend == "resnet"
        for name in ("wp", "resnet"):
            previous = runtime.agent
            runtime.switch_model(name)
            assert not runtime.ready.is_set()
            qtbot.waitUntil(runtime.ready.is_set, timeout=30000)
            assert runtime.model_status().backend == name and previous.process is None
        assert runtime.model_status().model_id == original_id and runtime.gate.owner is None
    finally:
        runtime.close()
    assert runtime.agent.process is None and not runtime.thread.is_alive()
