"""连续任务局数契约、旧配置迁移和命令行兼容；所有数据使用隔离目录。"""

import json
import sys
from dataclasses import asdict
from unittest.mock import Mock

import pytest

from doudizhu_lab.domain.settings import Profile, parse_max_games, validate_bot_profile
from doudizhu_lab.storage.config import Store


@pytest.mark.parametrize("games", [1, 5, 31, 2**31, 2**64, 10**100])
def test_positive_game_counts_roundtrip_without_fixed_integer_limit(tmp_path, games):
    store = Store(tmp_path)
    store.profile.continuous_settings = {"max_games": games}
    validate_bot_profile(store.profile)
    store.save()
    assert Store(tmp_path).profile.continuous_settings == {"max_games": games}
    saved = json.loads(store.settings_path.read_text(encoding="utf8"))
    assert saved["version"] == 7 and saved["profile"]["continuous_settings"] == {"max_games": games}


@pytest.mark.parametrize("value", [None, True, False, 0, -1, 1.5, "5"])
def test_game_count_contract_rejects_non_positive_integers(value):
    profile = Profile(continuous_settings={"max_games": value})
    with pytest.raises(ValueError, match="大于0的整数"):
        validate_bot_profile(profile)


@pytest.mark.parametrize("settings", [None, [], {}, {"max_games": 5, "time_limit": 1}])
def test_current_contract_only_accepts_game_count(settings):
    with pytest.raises(ValueError, match="最多局数"):
        validate_bot_profile(Profile(continuous_settings=settings))


@pytest.mark.parametrize("text,games", [("1", 1), (" 031 ", 31), (str(10**100), 10**100)])
def test_game_count_text_uses_unbounded_python_integer(text, games):
    assert parse_max_games(text) == games


@pytest.mark.parametrize("text", ["", " ", "0", "-1", "+2", "1.5", "abc", "1_000", "1e3", "五"])
def test_invalid_game_count_text_has_chinese_reason(text):
    with pytest.raises(ValueError, match="大于0的整数"):
        parse_max_games(text)


@pytest.mark.parametrize("version", [1, 3, 4, 5, 6])
def test_old_duration_is_removed_in_memory_and_original_is_backed_up_once(tmp_path, version):
    profile = asdict(Profile())
    profile.update(play_model="wp", continuous_policy="balanced", threshold=0.91)
    profile["continuous_settings"] = {"max_games": 27, "time_limit": 1}
    original = json.dumps({"version": version, "profile": profile}, ensure_ascii=False)
    target = tmp_path / "settings.json"
    target.write_text(original, encoding="utf8")
    store = Store(tmp_path)
    assert store.profile.continuous_settings == {"max_games": 27}
    assert target.read_text(encoding="utf8") == original
    backup = tmp_path / f"settings.v{version}.bak.json"
    assert not backup.exists()
    store.save()
    assert backup.read_text(encoding="utf8") == original
    saved = json.loads(target.read_text(encoding="utf8"))
    assert saved["version"] == 7
    assert saved["profile"] == profile | {"continuous_settings": {"max_games": 27}}
    store.profile.continuous_settings["max_games"] = 10**100
    store.save()
    assert backup.read_text(encoding="utf8") == original


@pytest.mark.parametrize("games", [31, 2**31, 2**64, 10**100])
def test_cli_large_target_only_overrides_current_launch(tmp_path, monkeypatch, games):
    from doudizhu_lab.cli import app

    store = Store(tmp_path)
    store.save()
    before = store.settings_path.read_bytes()
    profiles = []
    monkeypatch.setattr(sys, "argv", ["ddz-lab", "--data-dir", str(tmp_path), "--max-games", str(games)])
    monkeypatch.setattr(app, "QApplication", Mock(return_value=Mock()))
    monkeypatch.setattr(app, "MainWindow", lambda config, **kw: profiles.append(config.profile) or Mock())
    app.main()
    assert profiles[0].continuous_settings == {"max_games": games}
    assert store.settings_path.read_bytes() == before


@pytest.mark.parametrize("text", ["", "0", "-1", "1.5", "abc"])
def test_cli_invalid_target_is_rejected_before_gui_or_data_access(monkeypatch, capsys, text):
    from doudizhu_lab.cli import app

    application = Mock()
    monkeypatch.setattr(sys, "argv", ["ddz-lab", "--max-games", text])
    monkeypatch.setattr(app, "QApplication", application)
    with pytest.raises(SystemExit) as exc:
        app.main()
    assert exc.value.code == 2 and "大于0的整数" in capsys.readouterr().err
    application.assert_not_called()


@pytest.mark.parametrize("legacy", [["1"], ["14401"], ["旧值"], ["1", "--time-limit", "2"]])
def test_cli_old_duration_is_ignored_and_warns_once(tmp_path, monkeypatch, capsys, legacy):
    from doudizhu_lab.cli import app

    original = json.dumps(
        {"version": 6, "profile": {"continuous_settings": {"max_games": 7, "time_limit": 1}}}
    )
    target = tmp_path / "settings.json"
    target.write_text(original, encoding="utf8")
    profiles = []
    monkeypatch.setattr(sys, "argv", ["ddz-lab", "--data-dir", str(tmp_path), "--time-limit", *legacy])
    monkeypatch.setattr(app, "QApplication", Mock(return_value=Mock()))
    monkeypatch.setattr(app, "MainWindow", lambda config, **kw: profiles.append(config.profile) or Mock())
    app.main()
    assert profiles[0].continuous_settings == {"max_games": 7}
    assert target.read_text(encoding="utf8") == original
    assert capsys.readouterr().err.count("--time-limit 已弃用并忽略") == 1


def test_cli_help_only_advertises_game_target(monkeypatch, capsys):
    from doudizhu_lab.cli import app

    monkeypatch.setattr(sys, "argv", ["ddz-lab", "--help"])
    with pytest.raises(SystemExit) as exc:
        app.main()
    output = capsys.readouterr().out
    assert exc.value.code == 0 and "--max-games" in output and "--time-limit" not in output
