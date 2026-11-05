"""Load / compare / write helper for the policy-input goldens.

Named ``golden_harness`` rather than ``goldens`` because the committed JSON lives
in ``tests/serving/goldens/`` -- a module and a data directory of the same name in
one package resolves in favour of the module, which is the kind of quiet
resolution nobody should have to know about.

Built on the ``FieldDiff`` / ``SceneMismatchReport`` / :func:`compare_array` /
:func:`compare_key_set` / :func:`malformed_golden` vocabulary in
:mod:`toyblocks.eval.scene_verification` -- the same one the rollout golden
uses -- so there is one comparison framework in the repo rather than a second
one living in ``tests/``.

Two ideas carry the design:

* **Identity gates the numbers.** A golden holds an ``identity`` block (fixture
  digest, config fixture, transform class, state width, depth keys, image size,
  interpolate device) and a ``body`` of numbers. :func:`compare_identity`
  short-circuits the body compare on a mismatch, so a failure is never ambiguous
  about *which* of "the input changed" and "the arithmetic changed" happened --
  which is the difference between reviewing a new baseline and reverting a bug.
* **A write is refused from a failing run.** The golden-independent invariants
  the caller checks first (the ``wire == uint8/255`` identity, all-zero ptmaps,
  normalized state inside its band) land in the same report, and
  :func:`write_golden` asserts the report is clean before overwriting. Otherwise
  ``--write-policy-input-golden`` would launder a real regression into the
  baseline.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from toyblocks.eval.scene_verification import (
    FieldDiff,
    SceneMismatchReport,
    compare_array,
    compare_key_set,
    malformed_golden,
)

GOLDEN_DIR = Path(__file__).parent / "goldens"

REGEN_COMMAND = (
    "pytest tests/serving/test_policy_input_goldens.py --write-policy-input-golden"
)

IDENTITY = "identity"
BODY = "body"


def golden_path(name: str) -> Path:
    return GOLDEN_DIR / f"{name}.json"


def new_report(name: str) -> SceneMismatchReport:
    """A report whose header names the golden rather than a dataset episode.

    ``SceneMismatchReport`` was written for the dataset comparison, so its
    ``dataset_path`` / ``episode_index`` fields are repurposed here: the golden
    file is what this run is being diffed against, and there is no episode.
    """
    return SceneMismatchReport(dataset_path=str(golden_path(name)), episode_index=-1)


def load_golden(name: str) -> dict[str, Any]:
    """The committed golden, or a failure naming the command that writes it."""
    path = golden_path(name)
    if not path.is_file():
        raise FileNotFoundError(
            f"no policy-input golden at {path}. It is generated from the same pure "
            f"functions this test calls, so write it with:\n    {REGEN_COMMAND}\n"
            "then review the diff and commit it."
        )
    with open(path) as f:
        return json.load(f)


def write_golden(name: str, observed: dict[str, Any], report: SceneMismatchReport) -> Path:
    """Overwrite ``name``'s golden from this run, or refuse to.

    The refusal is the point: ``report`` already carries the golden-independent
    invariants, so a run that broke one of them cannot quietly become the new
    baseline.
    """
    assert report.ok, (
        "refusing to write the policy-input golden from a run that already "
        f"failed its own invariants:\n{report.format_console()}"
    )
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    path = golden_path(name)
    path.write_text(json.dumps(observed, indent=2, sort_keys=True) + "\n")
    return path


def compare_exact(
    name: str, observed: Any, golden: Any, report: SceneMismatchReport,
    *, hard: bool = True,
) -> None:
    """Diff two non-numeric values (a digest, a dtype, a shape) as equality.

    Kept apart from :func:`compare_array` because there is no tolerance to
    consume: a sha256 that differs by one byte differs completely, so reporting a
    margin would invite reading it as "nearly right".

    ``hard=False`` reports the difference as a warn line instead of failing the
    run, for a digest over a value that is only *nearly* reproducible -- a
    floating-point kernel's output. The digest still earns its place there: it
    says the bytes moved, which is the first thing worth knowing when the
    tolerant checks beside it start drifting.
    """
    same = observed == golden
    report.diffs.append(
        FieldDiff(
            name=name,
            passed=same,
            hard=hard,
            dataset_value=golden,
            sim_value=observed,
            margin=None if same else float("inf"),
        )
    )


def identity(
    *,
    obs_digest: str,
    config: str,
    stats: str,
    transform_class: str,
    state_dim: int,
    depth_keys: list[str],
    include_ptmap: bool,
    image_size: list[int],
    image_interpolate_device: str,
    action_mode: str,
) -> dict[str, Any]:
    """Everything that has to be equal before the numbers mean anything.

    ``image_interpolate_device`` is in here because it is pinned to ``"cpu"``
    for every golden: the resize output differs between CUDA and CPU kernels, so
    a golden computed on a GPU box would fail on the CPU lane for a reason that
    has nothing to do with the code under test.
    """
    return {
        "obs_digest": obs_digest,
        "config": config,
        "stats": stats,
        "transform_class": transform_class,
        "state_dim": int(state_dim),
        "depth_keys": list(depth_keys),
        "include_ptmap": bool(include_ptmap),
        "image_size": list(image_size),
        "image_interpolate_device": image_interpolate_device,
        "action_mode": action_mode,
    }


def compare_identity(
    observed: dict[str, Any], golden: dict[str, Any], report: SceneMismatchReport
) -> bool:
    """Diff the identity blocks; ``False`` means skip the body compare.

    Reported per field rather than as one blob so the console line says
    ``golden_identity[obs_digest]`` or ``golden_identity[state_dim]`` -- the
    field name *is* the diagnosis.
    """
    if IDENTITY not in golden:
        malformed_golden(
            IDENTITY,
            report,
            remedy=f"Re-record it with: {REGEN_COMMAND}",
        )
        return False
    gold = golden[IDENTITY]
    compare_key_set("golden_identity[fields]", observed, gold, report)
    ok = True
    for key in sorted(observed):
        if key not in gold:
            ok = False
            continue
        compare_exact(f"golden_identity[{key}]", observed[key], gold[key], report)
        ok = ok and observed[key] == gold[key]
    return ok and sorted(observed) == sorted(gold)


def compare_section(
    section: str,
    observed: dict[str, Any],
    golden: dict[str, Any],
    report: SceneMismatchReport,
) -> dict[str, Any] | None:
    """The golden's ``body[section]``, or a structured failure if it has none."""
    body = golden.get(BODY)
    if not isinstance(body, dict) or section not in body:
        malformed_golden(
            f"{BODY}.{section}",
            report,
            remedy=f"Re-record it with: {REGEN_COMMAND}",
        )
        return None
    return body[section]


def compare_exact_array(
    name: str, observed: Any, golden: Any, report: SceneMismatchReport
) -> None:
    """Bit-exact numeric compare: same pure function, same input, same numbers.

    ``tol=0.0`` is deliberate rather than lazy for everything reached by plain
    float32 arithmetic or by integer index math (the state blocks, the
    intrinsics rescale, the nearest-neighbour depth resize): those reproduce
    byte-for-byte across platforms, so any non-zero difference is a code change
    and a tolerance would only decide how large a silent behaviour change is
    allowed to be.

    The bilinear RGB resize is the one step that does not qualify and is
    compared with a tolerance instead; see ``test_policy_input_goldens.py``.
    """
    compare_array(name, observed, golden, 0.0, report)


def describe_wire(wire: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Per-key dtype and shape of an emitted wire dict, as JSON.

    ``task_name`` and friends are lists, not arrays, so they are described by
    length and element type -- dropping them would leave the golden silent about
    the one key whose *type* the server validates differently.
    """
    out: dict[str, dict[str, Any]] = {}
    for key in sorted(wire):
        value = wire[key]
        if isinstance(value, np.ndarray):
            out[key] = {"dtype": str(value.dtype), "shape": list(value.shape)}
        elif isinstance(value, (list, tuple)):
            elem = {type(v).__name__ for v in value}
            out[key] = {"dtype": f"list[{'|'.join(sorted(elem)) or 'empty'}]",
                        "shape": [len(value)]}
        else:
            out[key] = {"dtype": type(value).__name__, "shape": []}
    return out


def compare_described(
    name: str,
    observed: dict[str, dict[str, Any]],
    golden: dict[str, dict[str, Any]],
    report: SceneMismatchReport,
) -> None:
    """Diff two :func:`describe_wire` maps: key set first, then each entry."""
    compare_key_set(f"{name}[keys]", observed, golden, report)
    for key in sorted(set(observed) & set(golden)):
        compare_exact(f"{name}[{key}].dtype", observed[key]["dtype"], golden[key]["dtype"], report)
        compare_exact(f"{name}[{key}].shape", observed[key]["shape"], golden[key]["shape"], report)
