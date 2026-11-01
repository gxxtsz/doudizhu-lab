"""本机旧录像的状态回归：长连对、聊天遮挡与下一局独立恢复。"""

import hashlib
import json

import pytest

from doudizhu_lab.domain.tracker import GameTracker
from doudizhu_lab.storage.truth import validate_truth
from doudizhu_lab.vision.table import TableObservation
from tests.support import local_data_root


@pytest.mark.local_data
def test_reviewed_games_and_obscured_game_keep_trusted_boundary():
    root = local_data_root() / "reliability-regression/paced-session"
    manifest = root / "manifest.json"
    if not manifest.exists():
        pytest.skip("本机旧录像状态回归数据不随源码提交")
    meta = json.loads(manifest.read_text(encoding="utf-8"))
    path = root / "observations.jsonl"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == meta["sha256"]
    tracker = GameTracker(meta["seat_order"], config=meta["tracker_settings"])
    states, actions = {}, {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            update = tracker.observe(TableObservation(**json.loads(line)))
            states[update.state.game_id] = update.state
            for event in update.events:
                if event.kind in ("play", "pass") and event.confirmed:
                    actions.setdefault(event.game_id, []).append(event)
    games = [state for state in states.values() if state.game_id]
    assert len(games) == 3
    assert [(g.turn, g.complete, g.pending) for g in games] == [
        (39, True, False),
        (2, False, True),
        (37, True, False),
    ]
    first_actions = actions[games[0].game_id]
    assert first_actions[0].cards == ["6", "6", "7", "7", "8", "8", "9", "9"]
    assert sum(e.kind == "pass" and e.seat == "self" for e in first_actions) == 6
    assert games[2].remaining == {"self": 0, "left": 1, "right": 12}


@pytest.mark.local_data
def test_expanded_manual_labels_have_222_closed_actions_and_keep_unknown_windows():
    folder = local_data_root() / "sessions/20261031_204318_5839df/annotations"
    path = folder / "development-reviewed-seven-v2.json"
    if not path.exists():
        pytest.skip("本机人工开发标注不随源码提交")
    truth = json.loads(path.read_text(encoding="utf-8"))
    assert not truth["independent_review"] and len(truth["events"]) == 222
    assert validate_truth(truth, require_review=True)["valid"]
    partial = json.loads((folder / "development-obscured-game07.json").read_text(encoding="utf-8"))
    assert not partial["complete_trusted"] and partial["games"][0]["incomplete"]
    assert len(partial["events"]) == 34 and len(partial["unknown_windows"]) == 2
