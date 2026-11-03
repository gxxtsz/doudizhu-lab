"""11月3日下午私人故障图的固定回归入口；源图片仅留在显式指定的本机目录。"""

import json
import sqlite3
from contextlib import closing
from types import SimpleNamespace

import pytest

from doudizhu_lab.domain.bot_types import ObservationSnapshot
from doudizhu_lab.domain.models import Frame
from doudizhu_lab.domain.tracker import GameState
from doudizhu_lab.storage.config import read_rgb
from doudizhu_lab.storage.replay import profile_from_dict
from doudizhu_lab.vision.cards import Recognizer
from doudizhu_lab.vision.controls import ControlRecognizer
from doudizhu_lab.vision.table import MarkerLibrary, TableRecognizer
from tests.support import local_data_root


def recognize(name, fid):
    folder = local_data_root() / "sessions" / name
    if not (folder / "evidence" / (fid + ".png")).exists():
        pytest.skip("未提供指定的私人故障样本")
    with closing(sqlite3.connect((folder / "session.sqlite3").as_uri() + "?mode=ro", uri=True)) as db:
        profile = profile_from_dict(
            json.loads(db.execute("SELECT value FROM meta WHERE key='profile'").fetchone()[0])
        )
    frame = Frame(
        "offline", read_rgb(folder / "evidence" / (fid + ".png")), 0, fid, 0, "replay", fid, "regression", 1
    )
    table = TableRecognizer(
        Recognizer(folder / "templates/templates", include_bundled=False),
        MarkerLibrary(folder / "templates/markers"),
    )
    observation, recognition = table.observe(frame, profile)
    return folder, profile, frame, observation, recognition


@pytest.mark.local_data
@pytest.mark.parametrize(
    "name,fid,expected",
    [
        ("20261103_133552_594545", "3017c974-22943", ["J", "9", "BJ"]),
        ("20261103_143626_6cbfcd", "a972d490-5124", ["5", "BJ", "9"]),
        ("20261103_143626_6cbfcd", "a972d490-9251", ["8", "BJ", "4"]),
        ("20261103_145717_5db1ff", "4d7a26b0-3004", ["Q", "8", "BJ"]),
    ],
)
def test_frozen_templates_read_all_three_bottom_cards(name, fid, expected):
    *_, observation, recognition = recognize(name, fid)
    assert observation.bottom == expected


@pytest.mark.local_data
def test_double_joker_highlight_has_known_unselected_edges():
    folder, profile, frame, observation, recognition = recognize("20261103_145717_5db1ff", "4d7a26b0-25398")
    snap = ObservationSnapshot.create(frame, recognition, observation, GameState(), [])
    result = ControlRecognizer(folder / "templates/controls").enrich(snap, profile)
    assert not result.control_reason
    assert len(result.hand) == 20 and all(c.selected is False for c in result.hand)


@pytest.mark.local_data
@pytest.mark.parametrize(
    "name,fid,card",
    [
        ("20261103_153036_3918af", "569e43a5-9727", "9"),
        ("20261103_145717_5db1ff", "4d7a26b0-29189", "10"),
    ],
)
def test_single_remaining_card_still_has_pixel_location(name, fid, card):
    folder, profile, frame, observation, recognition = recognize(name, fid)
    assert observation.hand == [card]
    snap = ObservationSnapshot.create(frame, recognition, observation, GameState(), [])
    result = ControlRecognizer(folder / "templates/controls").enrich(snap, profile)
    assert not result.control_reason and result.hand[0].rank == card


@pytest.mark.local_data
def test_historical_database_copy_releases_real_windows_file_handle(tmp_path):
    """只复制旧故障数据库到临时目录，验证收尾后真实文件可删除，不改写私人原库。"""
    from doudizhu_lab.runtime.bot import BotRuntime, InputGate
    from doudizhu_lab.storage.session import SessionStore

    source = local_data_root() / "sessions/20261103_123813_604b40/session.sqlite3"
    if not source.is_file():
        pytest.skip("未提供早期数据库占用故障资料")
    folder = tmp_path / "sessions/history-copy"
    folder.mkdir(parents=True)
    target = folder / "session.sqlite3"
    with closing(sqlite3.connect(source.as_uri() + "?mode=ro", uri=True)) as original:
        with closing(sqlite3.connect(target)) as copied:
            original.backup(copied)
    bot = BotRuntime(tmp_path, InputGate(), agent=SimpleNamespace(model_id="本机只读回归"), threaded=False)
    bot.attach(SimpleNamespace())
    bot._audit_session = SessionStore(folder)
    bot.attach(None)
    assert bot._audit_session is None
    target.unlink()
    assert not target.exists() and source.exists()


@pytest.mark.local_data
def test_legacy_missing_actions_keep_original_incomplete_results():
    """旧资料只读重放；切换第二版策略不能倒写当时不完整的历史或成绩。"""
    from tools.regress_afternoon import verify_legacy_gaps

    assert verify_legacy_gaps(local_data_root())["通过"]
