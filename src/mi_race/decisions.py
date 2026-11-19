"""decisions: the log of a guided command. Every question the tool asks goes into output.dir/decisions.json,
with the evidence it showed, what a yes changes, the default, the answer, how it was given and when.
The file is written again after every answer, so an interrupted session keeps what was decided,
and earlier sessions are kept, so the report can show every decision behind a result (step 7)"""

import json
from datetime import datetime
from pathlib import Path

from mi_race import __version__, ui

LOG_FILE = "decisions.json"


class DecisionLogError(ValueError):
    """a decisions.json that cannot be read: it is never written over"""


def _plain(value):
    """numpy numbers and arrays as plain JSON values"""
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"a {type(value).__name__} cannot go into the decision log")


def read_log(path) -> list:
    """the sessions saved in a decisions.json, oldest first (none if there is no file yet)"""
    path = Path(path)
    if not path.exists():
        return []
    try:
        sessions = json.loads(path.read_text(encoding="utf-8"))["sessions"]
    except (json.JSONDecodeError, UnicodeDecodeError, KeyError, TypeError) as problem:
        raise DecisionLogError(f"{path} is not a decision log ({problem}): move it away to start a new one")
    if not isinstance(sessions, list):
        raise DecisionLogError(f"{path} is not a decision log: 'sessions' is not a list")
    return sessions


class DecisionLog:
    """the decisions of one session of a guided command, added to output_dir/decisions.json.
    A session that asked nothing leaves no trace. answer and now can be replaced in tests"""

    def __init__(self, output_dir, command: str, config_file, accept_defaults: bool = False,
                 answer=ui.answer, now=datetime.now):
        self.path = Path(output_dir) / LOG_FILE
        self.earlier = read_log(self.path)              # read first: a broken file stops the command before it asks
        self.accept_defaults, self.answer, self.now = accept_defaults, answer, now
        self.session = {"command": command, "config_file": str(config_file), "mi_race": __version__,
                        "started": self._stamp(), "decisions": []}

    def _stamp(self) -> str:
        return self.now().isoformat(timespec="seconds")

    def ask(self, step: str, text: str, evidence: dict | None = None, proposal: dict | None = None,
            yes: bool = True) -> bool:
        """ask a yes/no question (yes: the default), log it and save the log. evidence: the numbers shown
        before the question; proposal: what a yes changes. Returns the answer"""
        answer, how = self.answer(text, yes, self.accept_defaults)
        self.session["decisions"].append({"step": step, "question": text, "evidence": evidence or {},
                                          "proposal": proposal or {}, "default": yes, "answer": answer,
                                          "how": how, "time": self._stamp()})
        self.save()
        return answer

    @property
    def decisions(self) -> list:
        return self.session["decisions"]

    def save(self) -> None:
        """the earlier sessions and this one, written to a temporary file first: an interruption while
        writing never leaves a broken log"""
        text = json.dumps({"sessions": [*self.earlier, self.session]}, indent=2, ensure_ascii=False,
                          default=_plain) + "\n"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(self.path.name + ".tmp")
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(self.path)