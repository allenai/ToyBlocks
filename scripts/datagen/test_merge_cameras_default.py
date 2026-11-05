"""    python scripts/datagen/test_merge_cameras_default.py

The pipeline's --merge-cameras default must keep the sidecars --render-extra produces.

The default used to be an explicit six-key list (front/gripper/left x images+depth). Naming
cameras explicitly makes merge_success drop everything else -- merge_success.py:1082 is
`if keep_all_cameras or key in keep_video_keys` -- so a run launched with

    --render-extra="--extra_rgb 224 --extra_depth rg8_224 --depth_gop 1000"

rendered observation.images.camera:<cam>_224 and observation.depth.camera:<cam>_rg8_224 and then
merged a dataset without them. Those are the streams HVLA trains on. merge_success's own
docstring records the same incident ("which is how a merged dataset ended up without the 224x224
depth streams the render had produced"), and the sep16_flip_left_right_play run was queued to
repeat it.

The old list also named left_camera, which --render-skip-cameras left_camera never renders, so
the default simultaneously asked for three keys that cannot exist and dropped two that do.
"""
import argparse
import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PIPELINE = REPO / "scripts" / "datagen" / "run_datagen_pipeline.py"
MERGE = REPO / "src" / "toyblocks" / "vecgen" / "recording" / "merge.py"


def _default_merge_cameras() -> list[str]:
    """Parse the default out of the argparse call without importing the pipeline (heavy deps)."""
    src = PIPELINE.read_text()
    i = src.index('"--merge-cameras"')
    j = src.index("default=", i)
    k = src.index("help=", j)
    return eval(src[j + len("default="):k].strip().rstrip(","))  # noqa: S307 - our own literal


def _load_merge_helpers():
    """The merge module (numpy/pyarrow only at import; lerobot is imported lazily)."""
    spec = importlib.util.spec_from_file_location("merge_under_test", MERGE)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # dataclasses resolve their module through sys.modules
    spec.loader.exec_module(mod)
    return mod


SIDECARS = [
    "observation.images.camera:front_camera_224",
    "observation.images.camera:gripper_camera_224",
    "observation.depth.camera:front_camera_rg8_224",
    "observation.depth.camera:gripper_camera_rg8_224",
]
BASE = [
    "observation.images.camera:front_camera",
    "observation.images.camera:gripper_camera",
    "observation.depth.camera:front_camera",
    "observation.depth.camera:gripper_camera",
]


def test_default_is_all():
    print("A. the pipeline default asks for every video feature")
    got = _default_merge_cameras()
    assert got == ["all"], f"--merge-cameras default is {got!r}, expected ['all']"
    print(f"   default = {got}")


def test_all_keeps_sidecars_and_explicit_list_drops_them():
    print("B. 'all' keeps the sidecars; the old explicit list drops them")
    ms = _load_merge_helpers()
    features = {k: {"dtype": "video"} for k in BASE + SIDECARS}
    features["observation.state"] = {"dtype": "float32"}

    def kept(cameras):
        keep_all = ms.wants_all_cameras(cameras)
        keys = [] if keep_all else ms.normalize_camera_keys(cameras)
        return [k for k, v in features.items()
                if v.get("dtype") != "video" or keep_all or k in keys]

    with_all = kept(["all"])
    for k in SIDECARS:
        assert k in with_all, f"'all' dropped {k}"
    print(f"   'all' keeps {len([k for k in with_all if k != 'observation.state'])} video keys "
          f"including all {len(SIDECARS)} sidecars")

    old_default = BASE + [
        "observation.images.camera:left_camera",
        "observation.depth.camera:left_camera",
    ]
    with_list = kept(old_default)
    dropped = [k for k in SIDECARS if k not in with_list]
    assert dropped == SIDECARS, f"expected the old list to drop every sidecar, dropped {dropped}"
    print(f"   old explicit list drops all {len(dropped)}: "
          f"{[k.split(':')[-1] for k in dropped]}")


def test_star_is_also_all():
    print("C. '*' is accepted as well as 'all'")
    ms = _load_merge_helpers()
    assert ms.wants_all_cameras(["*"]) and ms.wants_all_cameras(["ALL"])
    assert not ms.wants_all_cameras(["front_camera"])
    print("   '*' and 'ALL' -> all; a bare camera name -> explicit")


def _merge_invocations(scripts=("merge_datasets.py", "merge_success.py")):
    """Every place in the repo that spells out a `script` command line.

    Returns (path, line_no, the command text that follows the script name). Covers scripts and
    docs alike: a documented command is a real regression path because people copy-paste it.
    """
    out = []
    for sub in ("scripts", "docs"):
        root = REPO / sub
        if not root.is_dir():
            continue
        for f in sorted(root.rglob("*")):
            if f.suffix not in (".py", ".sh", ".md", ".yaml", ".yml") or not f.is_file():
                continue
            if f.name == Path(__file__).name:
                continue
            try:
                text = f.read_text()
            except (UnicodeDecodeError, OSError):
                continue
            for n, line in enumerate(text.splitlines(), 1):
                if not any(s in line for s in scripts):
                    continue
                # the flags may continue onto following lines (list literal, \ continuation)
                tail = "\n".join(text.splitlines()[n - 1:n + 11])
                out.append((f.relative_to(REPO), n, tail))
    return out


def test_every_merge_invocation_asks_for_all_cameras():
    """The durable guard: a new call site that forgets --cameras fails here, not in a dataset.

    run_tamper_pipeline.py shelled out with no --cameras at all and inherited merge_success's
    named-list default, which on a rendered dataset matches nothing at all -- bare names become
    observation.images.<cam> while render writes observation.images.camera:<cam>. The upstream
    default is 'all' now, but a pinned older TAMP still has the old one, so each call site says it.
    """
    print("D. every merge invocation in the repo asks for all cameras")
    sites = _merge_invocations()
    assert sites, "found no merge invocations -- the scanner is broken"
    bad = []
    for path, line, tail in sites:
        # A bare mention in prose (no flags at all) is not an invocation.
        if "--input_dirs" not in tail:
            continue
        if "--cameras" not in tail:
            bad.append(f"{path}:{line} passes no --cameras")
            continue
        # Literal 'all', or the pipeline's own configurable flag whose default test A pins.
        ok = ("--cameras all" in tail
              or '"--cameras", "all"' in tail
              or "*args.merge_cameras" in tail)
        if not ok:
            bad.append(f"{path}:{line} passes a named camera list")
    assert not bad, "merge invocations that can drop the sidecars:\n  " + "\n  ".join(bad)
    checked = [f"{p}:{n}" for p, n, t in sites if "--input_dirs" in t]
    print(f"   {len(checked)} invocation(s) checked: {', '.join(checked)}")




if __name__ == "__main__":
    test_default_is_all()
    test_all_keeps_sidecars_and_explicit_list_drops_them()
    test_star_is_also_all()
    test_every_merge_invocation_asks_for_all_cameras()
    print("\nPASS")
