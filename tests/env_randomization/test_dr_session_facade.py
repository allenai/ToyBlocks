"""DRSession must be pure delegation: same functions, same order, no logic of its own.

The one method that re-implements anything is kit_args -- rerender builds that string inline
before launch, so the facade cannot delegate to it. This pins the two to the same output over
every checked-in flag yaml. Pure: no Isaac, no GPU.
"""
import glob
import inspect

import yaml

from blocksuite.domain_randomization.session import DRSession

YAMLS = sorted(glob.glob("src/blocksuite/domain_randomization/flags/rerender_flags_*.yaml"))


def _rerender_inline_kit_args(flags):
    """Verbatim copy of rerender.py's pre-launch block, kept here as the oracle."""
    cs = flags.get("carb_settings")
    if isinstance(cs, dict) and cs:
        def kv(k, v):
            v = ("true" if v else "false") if isinstance(v, bool) else v
            return f"--/{str(k).lstrip('/')}={v}"
        return " ".join(kv(k, v) for k, v in cs.items())
    return ""


def test_kit_args_matches_rerender_on_every_checked_in_yaml():
    assert YAMLS, "no flag yamls found - run from the repo root"
    checked = 0
    for p in YAMLS:
        try:
            flags = yaml.safe_load(open(p)) or {}
        except Exception:                                     # noqa: BLE001
            continue
        assert DRSession.kit_args(flags) == _rerender_inline_kit_args(flags), p
        checked += 1
    assert checked == len(YAMLS), f"only {checked} of {len(YAMLS)} yamls parsed"


def test_kit_args_is_pure_and_handles_absence():
    assert DRSession.kit_args(None) == ""
    assert DRSession.kit_args({}) == ""
    assert DRSession.kit_args({"carb_settings": {}}) == ""
    assert DRSession.kit_args({"carb_settings": {"/a/b": True}}) == "--/a/b=true"
    assert DRSession.kit_args({"carb_settings": {"a/b": False}}) == "--/a/b=false"


def test_facade_holds_no_sampling_logic():
    """Every method should be a thin delegation. A method that grows a loop or an rng is a sign
    behaviour is leaking out of interpreter.py into the facade."""
    for name, fn in inspect.getmembers(DRSession, inspect.isfunction):
        if name.startswith("_") or name == "kit_args":
            continue
        src = inspect.getsource(fn)
        assert "default_rng" not in src, f"{name} samples"
        assert "\n    for " not in src, f"{name} loops"


def test_seeding_default_is_prefix_free():
    """dagger streams episodes, so the default must be a mode sample_episodes accepts."""
    s = DRSession({}, salt=1)
    assert s.seeding == "hash"
