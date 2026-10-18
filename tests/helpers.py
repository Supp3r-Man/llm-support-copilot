import itertools
import json
import urllib.error
import urllib.request

from copilot.models import ToolCall, Turn, Usage

_ids = itertools.count(1)


def call(name, **args):
    return Turn(None, [ToolCall(f"c{next(_ids)}", name, args)], Usage(10, 5), "tool_use")


def say(text):
    return Turn(text, [], Usage(10, 5), "end_turn")


class ScriptedModel:
    """Plays back turns (or callables of messages -> Turn); the last step repeats forever."""

    name = "scripted"

    def __init__(self, steps):
        self.steps, self.n, self.seen = list(steps), 0, []

    def generate(self, messages, tools):
        self.seen.append([dict(m) for m in messages])
        step = self.steps[min(self.n, len(self.steps) - 1)]
        self.n += 1
        return step(messages) if callable(step) else step


def http(url, method="GET", body=None, headers=None, raw=None, timeout=15):
    """Returns (status, headers, text). Never raises on HTTP error statuses."""
    data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
    hdrs = {"Content-Type": "application/json", **(headers or {})}
    req = urllib.request.Request(url, data=data, headers=hdrs, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.headers, r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.headers, e.read().decode()
