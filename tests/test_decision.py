"""tests for src/mi_race/decisions.py"""
import json
from datetime import datetime, timedelta

import numpy as np
import pytest

from mi_race import __version__
from mi_race.decisions import LOG_FILE, DecisionLog, DecisionLogError, read_log


def answers(*given):
    """a fake terminal: the answers and how they were given, in order"""
    queue = list(given)
    return lambda text, yes, accept_defaults: queue.pop(0)


def clock(start=datetime(2026, 11, 19, 14, 0, 0)):
    """a fake clock: one minute later at every call"""
    calls = []
    def now():
        calls.append(None)
        return start + timedelta(minutes=len(calls) - 1)
    return now


def test_a_decision_is_saved_with_its_evidence_proposal_answer_and_time(tmp_path):
    log = DecisionLog(tmp_path / "out", "search", "configs/example.json", answer=answers((True, "Enter")), now=clock())
    assert log.ask("threshold", "Set the threshold to 0.05 bits?", evidence={"sd_bits": 0.0259, "pairs": 20},
                   proposal={"search.threshold": 0.05}) is True
    saved = json.loads((tmp_path / "out" / LOG_FILE).read_text(encoding="utf-8"))
    assert saved == {"sessions": [{
        "command": "search", "config_file": "configs/example.json", "mi_race": __version__,
        "started": "2026-11-19T14:00:00",
        "decisions": [{"step": "threshold", "question": "Set the threshold to 0.05 bits?",
                       "evidence": {"sd_bits": 0.0259, "pairs": 20}, "proposal": {"search.threshold": 0.05},
                       "default": True, "answer": True, "how": "Enter", "time": "2026-11-19T14:01:00"}]}]}


def test_the_log_is_saved_after_every_answer(tmp_path):
    log = DecisionLog(tmp_path, "search", "c.json", answer=answers((True, "typed"), (False, "typed")))
    log.ask("threshold", "first?")
    assert len(read_log(tmp_path / LOG_FILE)[0]["decisions"]) == 1          # already on disk before the second
    log.ask("final", "second?", yes=False)
    decisions = read_log(tmp_path / LOG_FILE)[0]["decisions"]
    assert [(d["step"], d["default"], d["answer"]) for d in decisions] == [("threshold", True, True), ("final", False, False)]
    assert not (tmp_path / (LOG_FILE + ".tmp")).exists()


def test_the_default_taken_without_a_question_is_logged_as_such(tmp_path):
    seen = []
    def answer(text, yes, accept_defaults):
        seen.append(accept_defaults)
        return yes, "--yes"
    log = DecisionLog(tmp_path, "search", "c.json", accept_defaults=True, answer=answer)
    assert log.ask("final", "Evaluate once on the final datasets?", yes=False) is False
    assert seen == [True] and log.decisions[0]["how"] == "--yes"


def test_earlier_sessions_are_kept(tmp_path):
    DecisionLog(tmp_path, "search", "c.json", answer=answers((True, "Enter"))).ask("threshold", "first session?")
    DecisionLog(tmp_path, "final", "c.json", answer=answers((False, "typed"))).ask("final", "second session?")
    assert [s["command"] for s in read_log(tmp_path / LOG_FILE)] == ["search", "final"]


def test_a_session_that_asked_nothing_leaves_no_trace(tmp_path):
    DecisionLog(tmp_path, "search", "c.json", answer=answers())
    assert read_log(tmp_path / LOG_FILE) == [] and not (tmp_path / LOG_FILE).exists()


def test_numpy_values_are_saved_as_plain_json(tmp_path):
    log = DecisionLog(tmp_path, "search", "c.json", answer=answers((True, "Enter")))
    log.ask("check", "Keep it?", evidence={"gain": np.float32(0.25), "higher": np.int64(4), "better": np.bool_(True),
                                           "codebook": np.array([[50, 0], [0, 50]])})
    assert read_log(tmp_path / LOG_FILE)[0]["decisions"][0]["evidence"] == {
        "gain": 0.25, "higher": 4, "better": True, "codebook": [[50, 0], [0, 50]]}


@pytest.mark.parametrize("content", ["not json", "[1, 2]", '{"runs": []}', '{"sessions": 3}'])
def test_a_broken_log_stops_before_any_question_and_is_never_written_over(tmp_path, content):
    (tmp_path / LOG_FILE).write_text(content, encoding="utf-8")
    with pytest.raises(DecisionLogError, match="decision log"):
        DecisionLog(tmp_path, "search", "c.json", answer=answers((True, "Enter")))
    assert (tmp_path / LOG_FILE).read_text(encoding="utf-8") == content