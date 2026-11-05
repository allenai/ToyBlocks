"""Evaluation log for ``vlagent_rcs_client``: run provenance + per-command verdicts.

The log is JSON Lines -- one JSON object per line, appended as the operator
grades each command from the web GUI, so a crashed run keeps everything already
written. The first line of a file is a ``kind="run"`` header recording what the
run was made of (blocksuite/hvla commits, policy checkpoint, stereo source, PD
stiffness); every later line is a ``kind="verdict"`` record naming the command
that was graded and the parquet fragments holding it.

The file lives next to the recordings, in the checkpoint's own folder::

    <checkpoint dir>/<cfg.record_path>/eval_log_<YYYY-MM-DD>.json

which keeps a policy, the episodes it produced and their grades together.
"""

from __future__ import annotations

import datetime
import json
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from blocksuite.serving.sim2real.rollout.hardware_config import INCLUDE_DEPTH, INCLUDE_RIGHT
from blocksuite.serving.sim2real.rollout.inference_config import InferenceConfig

logger = logging.getLogger(__name__)

# .../blocksuite/src/blocksuite/serving/sim2real/rollout/eval_log.py -> blocksuite repo root.
BLOCKSUITE_ROOT = Path(__file__).resolve().parents[5]
HVLA_ROOT = BLOCKSUITE_ROOT / "maniflow-tb"

EVAL_LOG_PREFIX = "eval_log"


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

def _git(repo: Path, *args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), *args],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        logger.debug("git %s failed in %s", args, repo, exc_info=True)
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip()


def git_revision(repo: Path) -> dict[str, Any] | None:
    """``{commit, branch, dirty}`` for ``repo``, or None when it is not a checkout.

    ``dirty`` means tracked files differ from the commit, i.e. the code that ran
    is not exactly what ``commit`` names.
    """
    if not repo.is_dir():
        return None
    commit = _git(repo, "rev-parse", "HEAD")
    if commit is None:
        return None
    status = _git(repo, "status", "--porcelain", "--untracked-files=no")
    return {
        "path": str(repo),
        "commit": commit,
        "branch": _git(repo, "rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(status),
    }


def stereo_source(cfg: InferenceConfig) -> dict[str, Any]:
    """Which depth source the policy is being fed, and how it is configured."""
    if cfg.use_ffstereo:
        source = "ffstereo"
    elif cfg.include_depth:
        source = "zed_depth"
    else:
        source = "none"
    info: dict[str, Any] = {
        "source": source,
        "include_depth": cfg.include_depth,
        "include_right": INCLUDE_RIGHT,
        "hardware_include_depth": INCLUDE_DEPTH,
    }
    if cfg.use_ffstereo:
        info.update(
            {
                "host": cfg.ffstereo_host,
                "port": cfg.ffstereo_port,
                "clip_m": cfg.ffstereo_clip_m,
                "clip_min_m": cfg.ffstereo_clip_min_m,
                "max_depth_m": cfg.max_depth_m,
            }
        )
    return info


def stiffness_profile(env: Any) -> dict[str, Any] | None:
    """PD gains the env was built with (see ``env_setup.get_env``), or None."""
    profile = getattr(env, "stiffness_profile", None)
    return dict(profile) if isinstance(profile, dict) else None


def run_metadata(
    cfg: InferenceConfig,
    env: Any,
    *,
    policy_checkpoint: str,
    record_dir: Path,
) -> dict[str, Any]:
    """The ``kind="run"`` header: everything needed to place a set of verdicts."""
    return {
        "kind": "run",
        "time": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "blocksuite_commit": git_revision(BLOCKSUITE_ROOT),
        "hvla_commit": git_revision(HVLA_ROOT),
        "policy_checkpoint": policy_checkpoint or None,
        # The HL checkpoint is half of what produced a verdict when Molmo is
        # driving, so a run header without it cannot place its episodes.
        "molmo_checkpoint": (
            (cfg.molmo_checkpoint or None) if getattr(cfg, "use_molmo", False) else None
        ),
        "vlagents": {
            "host": cfg.vlagents_host,
            "port": cfg.vlagents_port,
            "model": cfg.vlagents_model,
        },
        "stereo": stereo_source(cfg),
        "stiffness": stiffness_profile(env),
        "record_dir": str(record_dir),
        "instruction": cfg.instruction,
        "point_goal_format": cfg.point_goal_format,
        "fps": cfg.fps,
        "n_action_steps": cfg.n_action_steps,
        "temporal_agg": cfg.temporal_agg,
        "tam": cfg.tam,
    }


# ---------------------------------------------------------------------------
# Output location
# ---------------------------------------------------------------------------

def checkpoint_dir(cfg: InferenceConfig, checkpoint_hint: str | None) -> Path | None:
    """Folder holding the checkpoint, from the same sources as ``point_goal_format``.

    ``cfg.model_config_path`` wins over the path the vlagents server reports, and
    either may be a config file, the ``*.safetensors`` file, or the directory
    itself. Returns None when nothing resolves on this machine -- a server running
    on another host reports a path that does not exist here.
    """
    source = cfg.model_config_path or checkpoint_hint
    if not source:
        return None
    candidate = Path(source).expanduser()
    directory = candidate if candidate.is_dir() else candidate.parent
    return directory if directory.is_dir() else None


def eval_output_dir(cfg: InferenceConfig, checkpoint_hint: str | None) -> Path | None:
    """``<checkpoint dir>/<cfg.record_path>``, created on demand.

    Returns None when the checkpoint folder cannot be resolved or written to, so
    callers keep recording wherever they already were instead of failing a run
    over a logging path.
    """
    base = checkpoint_dir(cfg, checkpoint_hint)
    if base is None:
        return None
    out_dir = base / Path(cfg.record_path).name
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        logger.warning(
            "cannot write recordings next to the checkpoint (%s); keeping %s",
            out_dir,
            cfg.record_path,
            exc_info=True,
        )
        return None
    return out_dir


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------

class EvalLogger:
    """JSON Lines eval log in ``directory``, one file per day.

    Append-only in normal use. The one exception is
    :meth:`amend_last_verdict`, which corrects a mis-graded command in place
    rather than letting two contradictory lines stand for one skill.
    """

    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)
        date = datetime.date.today().isoformat()
        self.path = self.directory / f"{EVAL_LOG_PREFIX}_{date}.json"

    def append(self, record: dict[str, Any]) -> bool:
        """Write one JSON object as a line. Never raises: a failed grade must not
        take the robot down mid-eval."""
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a") as handle:
                handle.write(json.dumps(record, default=str))
                handle.write("\n")
        except OSError:
            logger.exception("could not append to eval log %s", self.path)
            return False
        return True

    def amend_last_verdict(self, success: bool) -> dict[str, Any] | None:
        """Correct the most recent verdict line's ``success`` value in place.

        For the operator who graded a skill, then realised they graded it
        wrong. Appending a second line would leave two contradictory records of
        one command, so the line is rewritten and stamped with ``amended_at`` /
        ``amended_from`` -- the correction is visible, not silent.

        The rebuilt file is renamed over the original, so an interrupted
        amendment leaves the previous log intact rather than a half-written
        one. Returns the amended record, or None when there was nothing to
        amend or the rewrite failed. Never raises.
        """
        try:
            lines = self.path.read_text().splitlines()
        except FileNotFoundError:
            # Nothing has been written yet; not an error worth a traceback.
            logger.warning("no eval log at %s to amend", self.path)
            return None
        except OSError:
            logger.exception("could not read eval log %s to amend it", self.path)
            return None

        target = None
        for index in range(len(lines) - 1, -1, -1):
            try:
                record = json.loads(lines[index])
            except json.JSONDecodeError:
                continue  # hand-edited or truncated line: leave it verbatim
            if record.get("kind") == "verdict":
                target = (index, record)
                break
        if target is None:
            logger.warning("no verdict in %s to amend", self.path)
            return None

        index, record = target
        record["amended_from"] = bool(record.get("success"))
        record["success"] = bool(success)
        record["amended_at"] = (
            datetime.datetime.now().astimezone().isoformat(timespec="seconds")
        )
        lines[index] = json.dumps(record, default=str)

        try:
            fd, tmp_path = tempfile.mkstemp(
                dir=self.directory, prefix=self.path.name, suffix=".tmp"
            )
        except OSError:
            logger.exception("could not create a temp file beside %s", self.path)
            return None
        try:
            with os.fdopen(fd, "w") as handle:
                handle.write("\n".join(lines) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_path, self.path)
        except OSError:
            logger.exception("could not rewrite eval log %s", self.path)
            Path(tmp_path).unlink(missing_ok=True)
            return None
        return record
