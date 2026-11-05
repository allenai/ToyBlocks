"""Run the FF planner (``ff_planner/``: FF-v2.3modified, vendored from TAMP/planners) on PDDL text and parse its plan."""

from __future__ import annotations

import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from blocksuite.vecgen.symbolic.pddl import Fact, Problem, render_problem

_PLAN_MARKER = "found legal plan as follows"
_STATES_MARKER = "*** Plan 1 States ***"


def domain_without_actions(domain_path: Path, actions: list[str], out_dir: Path) -> Path:
    """Copy of ``domain_path`` in ``out_dir`` with the ``(:action NAME ...)`` blocks of
    ``actions`` removed, so FF can never plan them (e.g. pick-and-place-only datagen).
    Parentheses inside ``;`` comments are ignored."""
    text = Path(domain_path).read_text()
    lower = text.lower()
    for name in actions:
        key = f"(:action {name.lower()}"
        start = 0
        while True:
            i = lower.find(key, start)
            if i < 0:
                break
            nxt = lower[i + len(key) : i + len(key) + 1]
            if nxt and not nxt.isspace():  # a longer action name with this prefix
                start = i + len(key)
                continue
            line_start = lower.rfind("\n", 0, i) + 1
            if ";" in lower[line_start:i]:  # inside a comment
                start = i + len(key)
                continue
            depth, j, in_comment = 0, i, False
            while j < len(text):
                ch = text[j]
                if in_comment:
                    in_comment = ch != "\n"
                elif ch == ";":
                    in_comment = True
                elif ch == "(":
                    depth += 1
                elif ch == ")":
                    depth -= 1
                    if depth == 0:
                        break
                j += 1
            if depth != 0:
                raise ValueError(f"unbalanced (:action {name} ...) in {domain_path}")
            text = text[:i] + text[j + 1 :]
            lower = text.lower()
            start = i
    out = Path(out_dir) / Path(domain_path).name
    out.write_text(text)
    return out


def default_ff_binary() -> Path:
    """The vendored ``ff_planner/ff`` (``make`` in ``ff_planner/`` rebuilds it)."""
    vendored = Path(__file__).resolve().parent / "ff_planner" / "ff"
    if not vendored.exists():
        raise FileNotFoundError(f"FF binary not found at {vendored}; run `make` in {vendored.parent}")
    return vendored


@dataclass
class FFResult:
    ok: bool
    actions: list[Fact]
    seconds: float
    error: str = ""
    # ``states[i]``: the true facts before ``actions[i]`` (see parse_ff_states)
    states: list[tuple[str, ...]] = field(default_factory=list)


def parse_ff_output(text: str) -> list[Fact] | None:
    """Action lines after the plan marker, as lowercase ``Fact(action, args)``."""
    if _PLAN_MARKER not in text:
        return None
    body = text.split(_PLAN_MARKER, 1)[1]
    actions: list[Fact] = []
    for line in body.splitlines():
        s = line.strip()
        if s.startswith("step"):
            s = s[4:].strip()
        if ":" not in s:
            if actions and not s:
                break
            continue
        idx, rest = s.split(":", 1)
        if not idx.strip().isdigit():
            if actions:
                break
            continue
        toks = rest.split()
        if not toks:
            continue
        actions.append(Fact(toks[0].lower(), tuple(t.lower() for t in toks[1:])))
    return actions


def parse_ff_states(text: str) -> list[tuple[str, ...]]:
    """FF's ``*** Plan 1 States ***`` list: the true facts before each plan action (state i
    precedes action i), as sorted lowercase ``(pred arg ...)`` strings, the props TAMP's
    OutputParser reads into ``current_hl_state``. Empty when FF printed no states."""
    if _STATES_MARKER not in text:
        return []
    states: list[tuple[str, ...]] = []
    for block in text.split(_STATES_MARKER, 1)[1].split("State")[1:]:
        props: set[str] = set()
        for line in block.splitlines():
            s = line.strip()
            if not s or s.upper().startswith("NOT-"):
                continue
            words = s.replace("(", " ").replace(")", " ").replace(",", " ").lower().split()
            props.add(f"({' '.join(words)})")
        if props:
            states.append(tuple(sorted(props)))
    return states


def run_ff(
    domain_path: Path, problem_text: str, ff: Path | None = None, timeout_s: float = 60.0
) -> FFResult:
    import time

    ff = ff or default_ff_binary()
    domain_path = Path(domain_path).resolve()
    t0 = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="vecgen_ff_") as tmp:
        prob = Path(tmp) / "problem.pddl"
        prob.write_text(problem_text)
        try:
            cp = subprocess.run(
                [str(ff), "-o", str(domain_path), "-f", str(prob)],
                capture_output=True,
                text=True,
                timeout=timeout_s,
                cwd=tmp,
            )
        except subprocess.TimeoutExpired:
            return FFResult(False, [], time.monotonic() - t0, "timeout")
    actions = parse_ff_output(cp.stdout)
    dt = time.monotonic() - t0
    if actions is None:
        return FFResult(False, [], dt, (cp.stdout[-400:] + cp.stderr[-400:]).strip())
    return FFResult(True, actions, dt, states=parse_ff_states(cp.stdout))


def plan_problems(
    domain_path: Path,
    problem: Problem,
    inits: list[list[Fact]],
    ff: Path | None = None,
    workers: int = 8,
) -> list[FFResult]:
    """Run FF once per init (one env each) on a thread pool."""
    texts = [render_problem(problem, init) for init in inits]
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        return list(pool.map(lambda t: run_ff(domain_path, t, ff), texts))
