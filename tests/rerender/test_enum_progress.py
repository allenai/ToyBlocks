"""    python tests/rerender/test_enum_progress.py

The cold enumeration scan must never go silent, at any source size.

enumerate_source_episodes walks every problem x split of the SOURCE -- O(problems) weka
round-trips -- before the orchestrator can print a plan. The old rule printed progress every 250
problems and ONLY when the source had >=500 of them, so a 397-problem source (sep12) emitted
nothing at all for the entire scan: 12 min of total silence after the last startup line. That is
indistinguishable from a wedged orchestrator, which is exactly what the code comment there warns
invites someone to kill a healthy run -- and it cost 15 min of investigation on 2026-09-14.

The scan now announces itself before starting, reports on BOTH a problem-count step and a 30s
timer (so a small-but-slow source still reports), and always prints a final line.
"""
import sys
from pathlib import Path

_SRC = (Path(__file__).resolve().parents[2] / "scripts" / "rerender" / "render_sources.py").read_text()


def _step_rule():
    """The step expression AS WRITTEN in render_sources, not a copy of it.

    Re-typing `max(1, min(250, n // 20))` here would pass no matter what the module does; the
    point of this file is to check the module's own rule, so pull it out and evaluate it.
    """
    import re
    m = re.search(r"_step = (max\(1, min\(\d+, _n // \d+\)\))", _SRC)
    if not m:
        raise SystemExit("FAIL: could not find the _step rule in render_sources.py -- if it "
                         "was rewritten, update this test to match the new shape.")
    expr = m.group(1)
    return lambda n: eval(expr, {"max": max, "min": min}, {"_n": n})  # noqa: S307


def old(n): return 250 if n >= 500 else 0      # the rule that left sep12 silent
new = _step_rule()

print(f"{'problems':>9} {'old step':>9} {'old lines':>10} {'new step':>9} {'new lines':>10}")
for n in (1, 12, 50, 100, 200, 371, 397, 499, 500, 1200, 2400, 10000):
    o, nw = old(n), new(n)
    ol = (n // o) if o else 0
    nl = n // nw
    flag = "  <-- sep12: silent under the old rule" if n == 397 else ""
    print(f"{n:>9} {o:>9} {ol:>10} {nw:>9} {nl:>10}{flag}")

fails=[]
for n in (1, 12, 50, 100, 200, 371, 397, 499, 500, 1200, 2400, 10000):
    if new(n) < 1: fails.append(f"step<1 at n={n}")
    if n >= 20 and n // new(n) < 10: fails.append(f"<10 progress lines at n={n}")
    if new(n) > 250: fails.append(f"step>250 at n={n}")
# every source, however small, must still get the start and end lines
# The cadence is only half of it: the scan must also say it STARTED and that it finished.
for needle, why in (('"[enumerate] cache miss -- cold scan of', "no scan-start announcement"),
                    ("_now - _last >= 30", "no time-based progress (a slow small source stays silent)"),
                    ('f"[enumerate] scanned {_n}/{_n} problems', "no unconditional final line")):
    if needle.strip('"\'') not in _SRC and needle not in _SRC:
        fails.append(why)
print("\nscan announces start, reports on count AND time, prints a final line: "
      f"{'yes' if not fails else 'NO -- ' + '; '.join(fails)}")
print("FAIL: "+", ".join(fails)) if fails else print("\nALL PASS")
if __name__ == "__main__":
    sys.exit(1 if fails else 0)


def test_main():
    assert not fails
