"""Interactive goal GUI utilities for vlagent_rcs_client."""

from __future__ import annotations

import io
import json
import logging
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from time import sleep
from typing import Any
from urllib.parse import urlparse

import cv2
import numpy as np
from PIL import Image

from blocksuite.serving.sim2real.rollout.camera_utils import (
    GUI_DEPTH_CAMERAS,
    INTERACTIVE_GUI_CAMERAS,
    POINT_PICK_CAMERA,
)
from blocksuite.serving.sim2real.rollout.ffstereo_server import (
    colorize_depth_mm,
    depth_color_range,
)
from blocksuite.serving.sim2real.rollout.hardware_config import INCLUDE_DEPTH
from blocksuite.serving.sim2real.rollout.goal_utils import (
    action_requires_point,
    allowed_goal_actions_display,
    build_display_heatmap,
    format_language_goal,
    goal_action_buttons,
    is_valid_goal_action,
    overlay_heatmap_on_rgb,
    policy_space_uv,
    src_frame_size_for_point,
)
from blocksuite.serving.sim2real.rollout.inference import InferenceController
from blocksuite.serving.sim2real.rollout.inference_config import POINT_GOAL_FORMATS

logger = logging.getLogger(__name__)

GUI_PREVIEW_WIDTH = 640
WEB_GUI_PORT = 8765

GUI_CONTROL_COMMANDS: tuple[tuple[str, str, str], ...] = (
    ("Stop (q)", "q", "Stop episode and reset"),
    ("Success (s)", "s", "Mark success and reset"),
    ("Go Home (h)", "h", "Moving the robot to its home pose"),
    ("Reload (o)", "o", "Reload vlagent_rcs_client.json"),
)
GUI_COMMAND_KEYS = frozenset(cmd for _label, cmd, _desc in GUI_CONTROL_COMMANDS)


def camera_accepts_point_pick(cam_name: str, cfg: Any | None = None) -> bool:
    if cam_name != POINT_PICK_CAMERA:
        return False
    # When Molmo2 HL is running it picks the target itself, so a stray click
    # would silently overwrite the model's heatmap with a human one. Lock the
    # camera panel into view-only mode while ``use_molmo`` is on.
    if cfg is not None and getattr(cfg, "use_molmo", False):
        return False
    return True


def is_depth_camera(cam_name: str) -> bool:
    return cam_name.endswith("_depth")


def gui_camera_names(cfg: Any) -> list[str]:
    """Camera panels to render in the browser GUI, in render order.

    Order is chosen so the browser's 2-column grid lays out as::

        front_depth | gripper_depth   (top row, when depth is present)
        front_rgb   | gripper_rgb     (bottom row, always)

    RGB panels are always shown. Depth panels are prepended whenever depth is
    actually being captured or synthesized:

    * ``hardware_config.INCLUDE_DEPTH = True`` — ZED emits a native depth
      stream alongside RGB (visualized here, independent of whether the
      policy consumes it via ``InferenceConfig.include_depth``).
    * ``cfg.use_ffstereo`` — Fast-FoundationStereo produces depth from the
      L/R pair every inference step.
    """
    names: list[str] = []
    if INCLUDE_DEPTH or getattr(cfg, "use_ffstereo", False):
        names.extend(GUI_DEPTH_CAMERAS)
    names.extend(INTERACTIVE_GUI_CAMERAS)
    return names


def submit_gui_command(controller: InferenceController, command: str) -> str:
    """Queue a terminal-style control command from the GUI."""
    command = command.strip().lower()
    labels = {cmd: desc for _label, cmd, desc in GUI_CONTROL_COMMANDS}
    if command not in labels:
        raise ValueError(f"Unknown command {command!r}")
    controller.submit_command(command)
    if controller._cfg.dry_run:
        return f"{labels[command]} (dry-run: no robot motion)"
    return labels[command]


def grade_running_command(controller: InferenceController, success: bool) -> str:
    """Queue a grade for the command that is running, if there is one.

    Returns the sentence describing what was graded, or ``""`` when nothing was
    -- the first command of a run has no previously run skill behind it, and a
    command already graded by "Stop (success/failed)" must not be graded a
    second time here. Either way the grade is dropped rather than written twice
    or misfiled against the next command.
    """
    pending = controller.claim_verdict(success)
    if not pending:
        return ""
    seconds, _running = controller.command_runtime
    verdict = "success" if success else "failure"
    return f"Graded {pending!r} as {verdict} after {seconds:.1f}s."


def stop_with_verdict(controller: InferenceController, success: bool) -> str:
    """Grade the running command, then reset the episode.

    The verdict is queued first so the inference loop records the episode uuid
    and step range *before* the reset discards them. Success also marks the
    recorded episode successful (the ``s`` command); failure just stops it.
    """
    graded = grade_running_command(controller, success)
    if not graded:
        # Nothing is running and nothing is awaiting a grade: the operator
        # already graded this skill with the other button and is correcting it.
        # Only this branch skips the stop -- there is no episode to reset, and
        # an episode that IS running must still be stopped even when its grade
        # has nowhere to go.
        amended = controller.claim_amendment(success)
        if amended:
            verdict = "success" if success else "failure"
            return f"Corrected the grade for {amended!r} to {verdict}."
        graded = "Nothing to grade yet (no command has run)."
    controller.submit_command("s" if success else "q")
    # The prompt belongs to the episode that just ended; leaving it on screen
    # reads as if it were still driving the robot.
    if getattr(controller._cfg, "use_molmo", False):
        controller.clear_molmo_prompt()
    if controller._cfg.dry_run:
        return f"{graded} (dry-run: no robot motion)"
    return f"{graded} Episode stopped and reset."


def submit_goal_from_parts(
    controller: InferenceController,
    goal: str,
    point: tuple[str, float, float] | None,
    *,
    record: bool = False,
) -> str:
    goal = goal.strip()
    if not is_valid_goal_action(goal):
        allowed = allowed_goal_actions_display()
        logger.warning(
            "skipping invalid goal %r; allowed actions: %s", goal, allowed
        )
        raise ValueError(f"Invalid action goal {goal!r}. Allowed: {allowed}")
    if controller._cfg.point_goal_format == "action_goal":
        if action_requires_point(goal) and point is None:
            raise ValueError(f"'{goal}' requires a point (u,v)")
        submit_point = point
        u = v = None
        if point is not None:
            src_frame_size = src_frame_size_for_point(
                point, controller.get_latest_frames()
            )
            policy_uv = policy_space_uv(point, src_frame_size)
            if policy_uv is not None:
                u, v = policy_uv
    else:
        submit_point = None
        u = v = None
    if controller._cfg.include_heatmap and point is not None:
        submit_point = point
    formatted = format_language_goal(
        goal, controller._cfg.point_goal_format, u=u, v=v
    )
    controller.submit_language_goal(
        goal, submit_point, auto_start=True, record=record
    )
    return formatted


# Hot pink marker in RGB (#FF69B4).
_POINT_MARKER_RGB = (255, 105, 180)


def draw_point_marker(rgb: np.ndarray, x: float, y: float) -> np.ndarray:
    """Return a copy of ``rgb`` with a click marker drawn at ``(x, y)``."""
    marked = np.asarray(rgb).copy()
    if marked.ndim != 3:
        return marked
    cx, cy = int(round(x)), int(round(y))
    # OpenCV drawing ops expect BGR buffers; convert temporarily.
    marked_bgr = cv2.cvtColor(marked, cv2.COLOR_RGB2BGR)
    marker_bgr = (_POINT_MARKER_RGB[2], _POINT_MARKER_RGB[1], _POINT_MARKER_RGB[0])
    cv2.drawMarker(
        marked_bgr,
        (cx, cy),
        color=marker_bgr,
        markerType=cv2.MARKER_CROSS,
        markerSize=24,
        thickness=2,
    )
    cv2.circle(marked_bgr, (cx, cy), 10, marker_bgr, 2)
    return cv2.cvtColor(marked_bgr, cv2.COLOR_BGR2RGB)


def encode_frame_jpeg(
    cam_name: str,
    data: np.ndarray,
    point: tuple[str, float, float] | None,
    *,
    target_sigma: float = 0.8,
    depth_clip_m: float = 0.0,
) -> bytes:
    frame = decorate_camera_frame(
        cam_name,
        data,
        point,
        target_sigma=target_sigma,
        depth_clip_m=depth_clip_m,
    )
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    h, w = frame.shape[:2]
    scale = GUI_PREVIEW_WIDTH / max(w, 1)
    if scale < 1.0:
        frame = cv2.resize(
            frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA
        )
    buffer = io.BytesIO()
    Image.fromarray(frame).save(buffer, format="JPEG", quality=85)
    return buffer.getvalue()


def colorize_depth_for_gui(
    cam_name: str, depth: np.ndarray, *, depth_clip_m: float = 0.0
) -> np.ndarray:
    """Turn a 2D uint16 mm depth map into an HxWx3 uint8 RGB image for display."""
    depth_mm = np.asarray(depth)
    if depth_mm.dtype != np.uint16:
        depth_mm = depth_mm.astype(np.uint16)
    vmin_m, vmax_m = depth_color_range(depth_mm, clip_m=depth_clip_m)
    return colorize_depth_mm(depth_mm, vmin_m, vmax_m, label=cam_name)


def decorate_camera_frame(
    cam_name: str,
    data: np.ndarray,
    point: tuple[str, float, float] | None,
    *,
    target_sigma: float = 0.8,
    depth_clip_m: float = 0.0,
) -> np.ndarray:
    """Draw click marker + display heatmap on RGB; colorize depth for viewing."""
    arr = np.asarray(data)
    # 2D depth: colorize (view-only, no click marker / heatmap overlay).
    if is_depth_camera(cam_name) or arr.ndim == 2:
        return colorize_depth_for_gui(cam_name, arr, depth_clip_m=depth_clip_m)

    frame = arr
    if point is not None and point[0] == cam_name:
        frame = draw_point_marker(frame, point[1], point[2])
        if cam_name == POINT_PICK_CAMERA:
            height, width = frame.shape[:2]
            display_hm = build_display_heatmap(
                point[1],
                point[2],
                width,
                height,
                target_sigma=target_sigma,
            )
            frame = overlay_heatmap_on_rgb(frame, display_hm)
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return frame


class _QuietThreadingHTTPServer(ThreadingHTTPServer):
    """``ThreadingHTTPServer`` that stays quiet when a browser hangs up.

    The GUI re-requests every camera frame several times a second, so a
    browser routinely abandons an in-flight response -- a new ``img.src``
    supersedes the last one, the page is reloaded, the tab is closed -- and
    the write fails with ``BrokenPipeError``/``ConnectionResetError``. That is
    normal client behaviour rather than a server fault, so it is dropped
    instead of printing a traceback per aborted frame. Every other exception
    still reaches the default handler.
    """

    def handle_error(self, request: Any, client_address: Any) -> None:
        if isinstance(sys.exc_info()[1], (BrokenPipeError, ConnectionResetError)):
            return
        super().handle_error(request, client_address)


class WebGoalGui:
    """Browser GUI served at http://localhost:<port>."""

    _HTML = """<!DOCTYPE html>
<html><head>
<meta charset="utf-8"><title>VLAgent Goal Conditioning</title>
<style>
  body {
    font-family: sans-serif; margin: 16px; background: #1a1a1a; color: #eee;
    /* Room for the fixed checkpoint footer so it never covers the controls. */
    padding-bottom: 72px;
  }
  #ckpt-footer {
    position: fixed; left: 0; right: 0; bottom: 0;
    background: #101010; border-top: 1px solid #444;
    padding: 8px 16px; font-family: monospace; font-size: 13px;
    color: #aaa; white-space: nowrap; overflow-x: auto;
  }
  #ckpt-footer .k { color: #777; }
  #ckpt-footer .v { color: #cfe; }
  #ckpt-footer div { white-space: nowrap; }
  /* Two-column grid so paired panels (front | gripper) line up: depth on top
     when present, RGB on the bottom. Capped so each cell hits ~640px on wide
     screens without stretching further. */
  .row {
    display: grid; grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 16px; max-width: 1320px;
  }
  .cam { text-align: center; min-width: 0; }
  .cam img { width: 100%; max-width: 640px; height: auto; object-fit: contain; border: 2px solid #444; }
  .cam img.pickable { cursor: crosshair; }
  .cam img.view-only { cursor: default; opacity: 0.95; }
  input[type=text] { width: 420px; font-size: 16px; padding: 6px; }
  /* One button style everywhere: the action-goal buttons set the look, the
     control buttons match it. */
  button {
    font-size: 14px; padding: 8px 14px; margin: 0;
    background: #2b2b2b; color: #eee;
    border: 1px solid #555; border-radius: 4px; cursor: pointer;
  }
  button:hover { background: #383838; }
  button.selected {
    background: #9cf; color: #10222e; border-color: #9cf; font-weight: bold;
  }
  /* Buttons that write a verdict into the eval log. */
  button.grade-ok { border-color: #4c8; color: #cfe9d8; }
  button.grade-ok:hover { background: #24382e; }
  button.grade-bad { border-color: #c66; color: #f2d5d5; }
  button.grade-bad:hover { background: #3a2626; }
  .rate-label { color: #888; font-size: 13px; margin-left: 4px; }
  /* Running success rate, beside the record checkbox. */
  #success-rate { font-family: monospace; font-size: 13px; color: #888; }
  #success-rate .v { color: #cfe; }
  #timer {
    font-family: monospace; font-size: 15px; color: #ccc;
    border: 1px solid #444; border-radius: 4px; padding: 6px 10px;
  }
  /* Ticking up: the command is still running. */
  #timer.running { color: #9cf; border-color: #9cf; }
  /* Horizontal tape of graded commands, newest first. Scrolls sideways rather
     than wrapping, so the controls below never move as it fills up. */
  #tape {
    flex: 1; min-width: 0; display: flex; gap: 6px; align-items: center;
    overflow-x: auto; overflow-y: hidden; white-space: nowrap;
    padding: 4px 2px; scrollbar-width: thin;
  }
  #tape:empty::after {
    content: "no commands graded yet"; color: #666; font-size: 13px;
  }
  .tick {
    flex: none; font-family: monospace; font-size: 13px;
    padding: 4px 8px; border: 1px solid #555; border-radius: 4px;
    background: #222; color: #ccc;
  }
  .tick .verdict { font-weight: bold; margin-right: 4px; }
  .tick .secs { color: #888; margin-left: 6px; }
  .tick.ok { border-color: #4c8; }
  .tick.ok .verdict { color: #4c8; }
  .tick.bad { border-color: #c66; }
  .tick.bad .verdict { color: #c66; }
  /* Newly arrived entries flash once so a grade is visibly registered. */
  .tick.fresh { animation: tick-in 0.9s ease-out; }
  @keyframes tick-in {
    from { background: #3a3a3a; transform: translateX(-6px); }
    to { background: #222; transform: none; }
  }
  #status { margin: 10px 0; min-height: 1.2em; color: #9cf; }
  #point { color: #ccc; }
  label { margin-right: 12px; }
  .examples { font-size: 13px; color: #aaa; max-width: 900px; margin: 4px 0 10px 0; }
  /* Button rows: flex gap replaces the per-button margins. */
  .controls {
    display: flex; flex-wrap: wrap; gap: 6px; align-items: center;
    margin-top: 8px;
  }
  #action-buttons { display: flex; flex-wrap: wrap; gap: 6px; max-width: 900px; }
  .molmo-box {
    background: #22303c; border: 1px solid #38556b; border-radius: 6px;
    padding: 12px; margin: 12px 0; max-width: 900px;
  }
  .molmo-box h3 { margin: 0; color: #9cf; }
  /* Title on the left, Start on the right of the panel's top row. */
  .molmo-head {
    display: flex; align-items: center; gap: 12px; margin-bottom: 8px;
  }
  /* The instruction box takes the slack between the title and Start. */
  .molmo-head input[type=text] { flex: 1; min-width: 0; width: auto; }
  .molmo-out { font-family: monospace; white-space: pre-wrap; color: #cfe; }
  .hidden { display: none; }
</style></head><body>
<h2>VLAgent Goal Conditioning</h2>
<p id="status">Pick an action goal, click a point on front_rgb (action_goal only), then Submit (success) or Submit (failed) to grade the command that ran and send this one.</p>
<p id="point">Point: none</p>
<div class="row" id="cameras"></div>

<div id="molmo-panel" class="molmo-box hidden">
  <div class="molmo-head">
    <h3>Molmo2 HL (vision-language)</h3>
    <input id="molmo-prompt" type="text" placeholder="HL instruction"
           onkeydown="if (event.key === 'Enter') submitMolmoPrompt();">
    <button onclick="submitMolmoPrompt()">Start</button>
  </div>
  <p><b>Last Molmo2 output</b> (updates ~2 Hz):</p>
  <div>
    <span>current prompt:</span> <span id="molmo-active-prompt" class="molmo-out">-</span><br>
    <span>prev_action_str:</span> <span id="molmo-prev-action" class="molmo-out">-</span><br>
    <span>model prompt:</span> <span id="molmo-model-prompt" class="molmo-out">-</span><br>
    <span>raw:</span> <span id="molmo-raw" class="molmo-out">-</span><br>
    <span>language:</span> <span id="molmo-language" class="molmo-out">-</span><br>
    <span>target (u,v):</span> <span id="molmo-target" class="molmo-out">-</span><br>
    <span>LL instruction sent:</span> <span id="molmo-instruction" class="molmo-out">-</span>
  </div>
</div>

<p>
  <label>Prompt format:</label>
  <label><input type="radio" name="fmt" value="action_only" checked> action_only</label>
  <label><input type="radio" name="fmt" value="action_goal"> action_goal (+ u,v)</label>
</p>
<div id="action-buttons"></div>
<div class="examples">Exactly one action goal is sent per submit.</div>
<p class="controls">
  <button id="submit-ok" class="grade-ok" onclick="submitGoal(true)">Submit (success)</button>
  <button id="submit-bad" class="grade-bad" onclick="submitGoal(false)">Submit (failed)</button>
  <span class="rate-label">grade the command that ran, then send the selected one:</span>
</p>
<p class="controls">
  <button class="grade-ok" onclick="stopRated(true)">Stop (success)</button>
  <button class="grade-bad" onclick="stopRated(false)">Stop (failed)</button>
  <span class="rate-label">grade the command that ran, then stop and reset:</span>
</p>
<div class="controls">
  <span id="timer">Skill time: idle</span>
  <div id="tape" title="previous commands this run, newest first"></div>
</div>
<p class="controls">
  <button onclick="clearPoint()">Clear Point</button>
  <button onclick="sendCommand('h')" title="Move the robot home; stop the episode first">Go Home (h)</button>
  <button onclick="sendCommand('o')">Reload (o)</button>
  <button onclick="shutdown()">Quit (x)</button>
  <label><input id="record" type="checkbox"> Record episode</label>
  <span id="success-rate" title="graded commands this run">success rate: <span class="v">-</span></span>
</p>
<div id="ckpt-footer">
  <div><span class="k">checkpoint:</span> <span class="v" id="ckpt-path">-</span></div>
  <div id="hl-ckpt-row" style="display:none"><span class="k">HL checkpoint:</span> <span class="v" id="hl-ckpt-path">-</span></div>
  <div><span class="k">eval log:</span> <span class="v" id="eval-log">-</span></div>
</div>
<script>
let selected = null;
let cameraNames = [];
let selectedGoal = null;
let pointPickCamera = "front_rgb";
let useMolmo = false;
// The command last sent to the robot, as a commandKey() string. The Submit
// buttons stay highlighted only while the staged command still equals it, so
// picking another action or another point visibly un-commits them.
let submittedKey = null;
// Run timer, anchored on the last /api/timer answer and ticked locally between
// polls so the readout moves at display rate rather than at poll rate.
let timerSeconds = 0;
let timerRunning = false;
let timerAnchorMs = Date.now();
// Skill the timer refers to; labels the box in place of a generic word. Empty
// only before the first command, where "Skill time" stands in.
let timerSkill = "";
function currentFormat() {
  const radio = document.querySelector('input[name="fmt"]:checked');
  return radio ? radio.value : "";
}
function commandKey() {
  // Everything that changes the prompt actually sent: action, point, format.
  const point = selected ? [selected.camera, selected.u, selected.v] : null;
  return JSON.stringify([selectedGoal, point, currentFormat()]);
}
function refreshSubmitHighlight() {
  const sent = submittedKey !== null && submittedKey === commandKey();
  for (const id of ["submit-ok", "submit-bad"]) {
    const btn = document.getElementById(id);
    if (btn) btn.classList.toggle("selected", sent);
  }
}
function buildCameraPanels(cams) {
  const row = document.getElementById("cameras");
  row.innerHTML = "";
  cameraNames = [];
  for (const c of cams) {
    cameraNames.push(c.name);
    const wrap = document.createElement("div");
    wrap.className = "cam";
    const label = document.createElement("div");
    let suffix = "";
    if (c.kind === "depth") suffix = " (depth, view only)";
    else if (!c.pickable) suffix = " (view only)";
    label.textContent = c.name + suffix;
    const img = document.createElement("img");
    img.id = c.name;
    if (c.pickable) {
      img.className = "pickable";
      img.addEventListener("click", (ev) => clickCam(c.name, ev));
    } else {
      img.className = "view-only";
    }
    wrap.appendChild(label);
    wrap.appendChild(img);
    row.appendChild(wrap);
  }
}
function buildActionButtons(actions) {
  const box = document.getElementById("action-buttons");
  box.innerHTML = "";
  for (const a of actions) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.textContent = a.action;
    btn.dataset.action = a.action;
    btn.title = a.requires_point
      ? "needs a point (u,v) in action_goal format"
      : "no point needed";
    btn.addEventListener("click", () => selectGoal(a.action));
    box.appendChild(btn);
  }
}
function selectGoal(action) {
  selectedGoal = action;
  for (const btn of document.querySelectorAll("#action-buttons button")) {
    btn.classList.toggle("selected", btn.dataset.action === action);
  }
  refreshSubmitHighlight();
}
function refreshCheckpoint() {
  fetch("/api/status").then(r => r.json()).then(j => {
    const path = document.getElementById("ckpt-path");
    path.textContent = (j && j.policy_checkpoint)
      ? j.policy_checkpoint
      : "waiting for the vlagents server…";
    // Only shown with --use-molmo: an HL row reading "-" on every other run
    // is a line of noise in a footer that is scanned, not read.
    const hlRow = document.getElementById("hl-ckpt-row");
    const hlPath = document.getElementById("hl-ckpt-path");
    if (j && j.molmo_checkpoint) {
      hlPath.textContent = j.molmo_checkpoint;
      hlRow.style.display = "";
    } else {
      hlRow.style.display = "none";
    }
    const log = document.getElementById("eval-log");
    log.textContent = (j && j.eval_log)
      ? j.eval_log
      : "not open yet (needs the checkpoint folder)";
  });
}
function formatDuration(seconds) {
  const total = Math.max(0, seconds);
  const mins = Math.floor(total / 60);
  const secs = total - 60 * mins;
  return mins + ":" + (secs < 10 ? "0" : "") + secs.toFixed(1);
}
// Tape entries stay in whole seconds however long a skill ran: the ticks are
// scanned side by side, and one "1:35.2" among "42" reads as the smaller
// number at a glance. Sub-second precision is noise at this size.
function formatSeconds(seconds) {
  return String(Math.round(Math.max(0, seconds)));
}
function renderTimer() {
  const el = document.getElementById("timer");
  if (!el) return;
  const shown = timerRunning
    ? timerSeconds + (Date.now() - timerAnchorMs) / 1000
    : timerSeconds;
  el.classList.toggle("running", timerRunning);
  const label = timerSkill || "Skill time";
  if (!timerRunning && shown === 0) {
    el.textContent = label + ": idle";
    return;
  }
  el.textContent = (timerRunning ? "Running" : label) + ": " + formatDuration(shown);
}
// Signature of the history last drawn: the tape is only rebuilt when it
// actually changed, so a 1 Hz poll does not churn the DOM or fight scrolling.
let tapeSignature = null;
function renderTape(history) {
  const tape = document.getElementById("tape");
  if (!tape) return;
  const signature = JSON.stringify(history);
  if (signature === tapeSignature) return;
  const hadEntries = tapeSignature !== null;
  tapeSignature = signature;
  tape.innerHTML = "";
  // Newest first: the freshest verdict is always at the left edge, where the
  // tape starts, so it needs no scrolling to read.
  const newest = history.slice().reverse();
  newest.forEach((entry, i) => {
    const tick = document.createElement("div");
    tick.className = "tick " + (entry.success ? "ok" : "bad");
    // Only the leading entry of an update flashes, and never on first paint.
    if (i === 0 && hadEntries) tick.classList.add("fresh");
    tick.title = entry.time + " — " + (entry.success ? "success" : "failure");
    const verdict = document.createElement("span");
    verdict.className = "verdict";
    verdict.textContent = entry.success ? "\u2713" : "\u2717";
    const goal = document.createElement("span");
    goal.textContent = entry.goal;
    const secs = document.createElement("span");
    secs.className = "secs";
    secs.textContent = formatSeconds(entry.duration_s || 0);
    tick.appendChild(verdict);
    tick.appendChild(goal);
    tick.appendChild(secs);
    tape.appendChild(tick);
  });
  tape.scrollLeft = 0;
}
// Running tally over every command graded this run (not just the ones still
// on the tape, which is capped).
function renderSuccessRate(successes, graded) {
  const el = document.querySelector("#success-rate .v");
  if (!el) return;
  if (!graded) { el.textContent = "-"; return; }
  const pct = Math.round((100 * successes) / graded);
  el.textContent = successes + "/" + graded + " (" + pct + "%)";
}
function refreshTimer() {
  fetch("/api/timer").then(r => r.json()).then(j => {
    timerSeconds = (j && typeof j.elapsed_s === "number") ? j.elapsed_s : 0;
    timerRunning = !!(j && j.running);
    timerAnchorMs = Date.now();
    timerSkill = (j && j.skill) || "";
    renderTape((j && j.history) || []);
    renderSuccessRate((j && j.successes) || 0, (j && j.graded) || 0);
    renderTimer();
  });
}
function refreshFrames() {
  for (const cam of cameraNames) {
    const el = document.getElementById(cam);
    if (el) el.src = "/frame/" + cam + "?t=" + Date.now();
  }
}
refreshCheckpoint();
refreshTimer();
fetch("/api/actions").then(r => r.json()).then(j => {
  const actions = (j && j.actions) || [];
  buildActionButtons(actions);
  if (actions.length) selectGoal(actions[0].action);
  // Reflect the format the client is actually configured with, so the radio
  // never silently overrides --point-goal-format on the first submit.
  if (j && j.point_goal_format) {
    const radio = document.querySelector(
      'input[name="fmt"][value="' + j.point_goal_format + '"]');
    if (radio) radio.checked = true;
  }
  // Recording is on by default (InferenceConfig.record).
  if (j && typeof j.record_default === "boolean") {
    document.getElementById("record").checked = j.record_default;
  }
  refreshSubmitHighlight();
});
for (const radio of document.querySelectorAll('input[name="fmt"]')) {
  radio.addEventListener("change", refreshSubmitHighlight);
}
// Molmo picks the skill itself, so mirror its prediction onto the action-goal
// button row: the operator reads the running skill off the same buttons they
// would have clicked by hand. Exact match first, then the longest action the
// prediction starts with, so "place on table" wins over "place" when the model
// emits extra words. No match (or nothing predicted yet) clears the row rather
// than leaving a stale skill lit.
function highlightPredictedGoal(language) {
  const text = String(language || "").toLowerCase().replace(/\\s+/g, " ").trim();
  const buttons = document.querySelectorAll("#action-buttons button");
  let match = null;
  for (const btn of buttons) {
    const action = btn.dataset.action;
    if (text === action) { match = action; break; }
    // Only on a word boundary, so "placement" is not read as "place".
    if (text.startsWith(action + " ") && (match === null || action.length > match.length)) {
      match = action;
    }
  }
  for (const btn of buttons) {
    btn.classList.toggle("selected", btn.dataset.action === match);
  }
}
function refreshMolmo() {
  fetch("/api/molmo").then(r => r.json()).then(j => {
    if (!j || !j.use_molmo) return;
    document.getElementById("molmo-active-prompt").textContent = j.prompt || "-";
    document.getElementById("molmo-prev-action").textContent = j.prev_action_str || "(none)";
    document.getElementById("molmo-model-prompt").textContent = j.model_prompt || "-";
    if (j.raw) document.getElementById("molmo-raw").textContent = j.raw;
    if (j.language) document.getElementById("molmo-language").textContent = j.language;
    highlightPredictedGoal(j.language);
    if (j.target_xy) {
      document.getElementById("molmo-target").textContent =
        "(" + j.target_xy[0].toFixed(1) + ", " + j.target_xy[1].toFixed(1) + ")";
    } else {
      document.getElementById("molmo-target").textContent = "none";
    }
    if (j.last_instruction) {
      document.getElementById("molmo-instruction").textContent = j.last_instruction;
    }
  });
}
fetch("/api/cameras").then(r => r.json()).then(j => {
  if (j && j.point_pick_camera) pointPickCamera = j.point_pick_camera;
  useMolmo = !!(j && j.use_molmo);
  buildCameraPanels(j.cameras || []);
  refreshFrames();
  setInterval(refreshFrames, 200);
  setInterval(refreshCheckpoint, 5000);
  setInterval(renderTimer, 100);
  setInterval(refreshTimer, 1000);
  if (useMolmo) {
    document.getElementById("molmo-panel").classList.remove("hidden");
    // Seed the box once with the instruction the server already has. Only
    // here, never on the 2 Hz poll, which would overwrite mid-typing.
    fetch("/api/molmo").then(r => r.json()).then(j => {
      if (j && j.prompt) document.getElementById("molmo-prompt").value = j.prompt;
    });
    setInterval(refreshMolmo, 500);
    document.getElementById("status").textContent =
      "Molmo2 HL mode: type an HL instruction and press Start; " +
      "clicking is disabled.";
    document.getElementById("point").textContent = "Point: chosen by Molmo2";
  }
});
document.addEventListener("keydown", (ev) => {
  if (ev.target && (ev.target.tagName === "INPUT" || ev.target.tagName === "TEXTAREA")) return;
  const cmd = ev.key.toLowerCase();
  if (cmd === "x") { shutdown(); return; }
  // q/s mirror the Stop (failed) / Stop (success) buttons: every stop is graded.
  if (cmd === "q") { stopRated(false); return; }
  if (cmd === "s") { stopRated(true); return; }
  if ("oh".includes(cmd)) { sendCommand(cmd); }
});
function clickCam(cam, ev) {
  if (cam !== pointPickCamera) return;
  if (useMolmo) {
    document.getElementById("status").textContent =
      "Clicking is disabled while Molmo2 HL is on.";
    return;
  }
  const img = ev.target;
  const rect = img.getBoundingClientRect();
  const u = (ev.clientX - rect.left) / rect.width;
  const v = (ev.clientY - rect.top) / rect.height;
  fetch("/api/click", {method:"POST", headers:{"Content-Type":"application/json"},
    body: JSON.stringify({camera: cam, u: u, v: v})})
    .then(r => r.json()).then(j => {
      selected = j;
      document.getElementById("point").textContent =
        "Point: " + cam + " (" + j.u.toFixed(1) + ", " + j.v.toFixed(1) + ")";
      document.getElementById("status").textContent = "Selected point on " + cam;
      refreshSubmitHighlight();
    });
}
function clearPoint() {
  fetch("/api/clear_point", {method:"POST"});
  selected = null;
  document.getElementById("point").textContent = "Point: none";
  refreshSubmitHighlight();
}
// verdict: true = the command that just ran succeeded, false = it failed.
// The server drops the grade when no command has run yet (the first command of
// a run has no previously run skill) and reports that back in j.graded.
function submitGoal(verdict) {
  const goal = selectedGoal;
  if (!goal) {
    document.getElementById("status").textContent = "Select an action goal first.";
    return;
  }
  const fmt = document.querySelector('input[name="fmt"]:checked').value;
  const record = document.getElementById("record").checked;
  fetch("/api/format", {method:"POST", headers:{"Content-Type":"application/json"},
    body: JSON.stringify({format: fmt})});
  fetch("/api/submit", {method:"POST", headers:{"Content-Type":"application/json"},
    body: JSON.stringify({goal: goal, record: record, verdict: verdict})})
    .then(r => r.json()).then(j => {
      if (j.error) { document.getElementById("status").textContent = j.error; return; }
      const suffix = record ? " — starting episode with recording." : " — starting episode.";
      const graded = j.graded
        ? j.graded + " "
        : "Nothing to grade yet (no command has run). ";
      document.getElementById("status").textContent =
        graded + "Submitted: " + j.formatted + suffix;
      submittedKey = commandKey();
      refreshSubmitHighlight();
      refreshCheckpoint();
      refreshTimer();
    });
}
function stopRated(success) {
  fetch("/api/stop", {method:"POST", headers:{"Content-Type":"application/json"},
    body: JSON.stringify({success: success})})
    .then(r => r.json()).then(j => {
      if (j.error) { document.getElementById("status").textContent = j.error; return; }
      document.getElementById("status").textContent = j.message;
      refreshCheckpoint();
      refreshTimer();
    });
}
function submitMolmoPrompt() {
  const prompt = document.getElementById("molmo-prompt").value.trim();
  if (!prompt) {
    document.getElementById("status").textContent = "Enter an HL instruction first.";
    return;
  }
  fetch("/api/molmo", {method:"POST", headers:{"Content-Type":"application/json"},
    body: JSON.stringify({prompt: prompt, auto_start: true})})
    .then(r => r.json()).then(j => {
      if (j.error) { document.getElementById("status").textContent = j.error; return; }
      document.getElementById("status").textContent = "Molmo2 HL prompt: " + j.prompt;
    });
}
function sendCommand(cmd) {
  fetch("/api/command", {method:"POST", headers:{"Content-Type":"application/json"},
    body: JSON.stringify({command: cmd})})
    .then(r => r.json()).then(j => {
      if (j.error) { document.getElementById("status").textContent = j.error; return; }
      document.getElementById("status").textContent = j.message;
      // 'o' reloads the JSON config, which can change the checkpoint.
      if (cmd === "o") refreshCheckpoint();
    });
}
function shutdown() {
  fetch("/api/shutdown", {method:"POST"});
}
</script></body></html>"""

    def __init__(self, controller: InferenceController, host: str = "0.0.0.0", port: int = WEB_GUI_PORT) -> None:
        self.controller = controller
        self.host = host
        self.port = port
        self._lock = threading.Lock()
        self._selected_point: tuple[str, float, float] | None = None
        self._server: ThreadingHTTPServer | None = None

    def _selected_point_copy(self) -> tuple[str, float, float] | None:
        with self._lock:
            return self._selected_point

    def _set_point(
        self, camera: str, u: float, v: float, *, normalized: bool = False
    ) -> tuple[float, float]:
        cfg = self.controller._cfg
        if not camera_accepts_point_pick(camera, cfg):
            if getattr(cfg, "use_molmo", False) and camera == POINT_PICK_CAMERA:
                raise ValueError(
                    "Point picking is disabled while use_molmo is on; "
                    "the Molmo2 HL model picks the target from the text prompt."
                )
            raise ValueError(
                f"Point picking is disabled for {camera}; use {POINT_PICK_CAMERA}."
            )
        if normalized:
            frames = self.controller.get_latest_frames()
            rgb = frames.get(camera)
            if rgb is not None:
                height, width = rgb.shape[:2]
                u = float(u) * width
                v = float(v) * height
        with self._lock:
            self._selected_point = (camera, u, v)
        # Freeze the frame the operator was looking at when they picked the
        # point; it rides along with every later inference as ``prev_front_rgb``.
        self.controller.capture_prev_front_rgb()
        return u, v

    def _clear_point(self) -> None:
        with self._lock:
            self._selected_point = None

    def _make_handler(self) -> type[BaseHTTPRequestHandler]:
        gui = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, _format: str, *_args: Any) -> None:
                return

            def _read_json(self) -> dict[str, Any]:
                length = int(self.headers.get("Content-Length", 0))
                if length <= 0:
                    return {}
                return json.loads(self.rfile.read(length))

            def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
                body = json.dumps(payload).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:
                path = urlparse(self.path).path
                if path in {"/", "/index.html"}:
                    body = gui._HTML.encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                if path == "/api/cameras":
                    cfg = gui.controller._cfg
                    cams: list[dict[str, Any]] = []
                    for name in gui_camera_names(cfg):
                        cams.append({
                            "name": name,
                            "kind": "depth" if is_depth_camera(name) else "rgb",
                            "pickable": camera_accepts_point_pick(name, cfg),
                        })
                    self._send_json(
                        {
                            "cameras": cams,
                            "point_pick_camera": POINT_PICK_CAMERA,
                            "use_molmo": bool(getattr(cfg, "use_molmo", False)),
                        }
                    )
                    return
                if path == "/api/status":
                    # Read off the vlagents server (see vlagents_server's
                    # exposed_checkpoint_path); empty until it answers.
                    self._send_json(
                        {
                            "policy_checkpoint": str(
                                getattr(gui.controller, "policy_checkpoint", "")
                                or ""
                            ),
                            # The HL checkpoint is config, not something the
                            # vlagents server reports, so it is read straight
                            # off cfg -- and only when Molmo is actually in
                            # the loop.
                            "molmo_checkpoint": (
                                str(
                                    getattr(
                                        gui.controller._cfg, "molmo_checkpoint", ""
                                    )
                                    or ""
                                )
                                if getattr(
                                    gui.controller._cfg, "use_molmo", False
                                )
                                else ""
                            ),
                            "eval_log": str(
                                getattr(gui.controller, "eval_log_path", "") or ""
                            ),
                        }
                    )
                    return
                if path == "/api/timer":
                    # Deliberately separate from /api/status: this is polled
                    # every second, and reading policy_checkpoint can dial the
                    # vlagents server while it is still unanswered.
                    elapsed_s, running = gui.controller.command_runtime
                    self._send_json(
                        {
                            "elapsed_s": round(elapsed_s, 3),
                            "running": running,
                            "pending_verdict": (
                                gui.controller.pending_verdict_command
                            ),
                            "skill": gui.controller.current_skill,
                            "history": gui.controller.command_history,
                            "successes": gui.controller.verdict_tally[0],
                            "graded": gui.controller.verdict_tally[1],
                        }
                    )
                    return
                if path == "/api/actions":
                    self._send_json(
                        {
                            "actions": goal_action_buttons(),
                            "point_goal_format": (
                                gui.controller._cfg.point_goal_format
                            ),
                            "record_default": bool(
                                getattr(gui.controller._cfg, "record", True)
                            ),
                        }
                    )
                    return
                if path.startswith("/frame/"):
                    cam = path.removeprefix("/frame/").split("/")[0]
                    if cam not in gui_camera_names(gui.controller._cfg):
                        self.send_error(404)
                        return
                    frames = gui.controller.get_latest_frames()
                    data = frames.get(cam)
                    if data is None:
                        self.send_error(404)
                        return
                    cfg = gui.controller._cfg
                    depth_clip_m = (
                        float(getattr(cfg, "ffstereo_clip_m", 0.0) or 0.0)
                        if getattr(cfg, "use_ffstereo", False)
                        else 0.0
                    )
                    # When Molmo2 HL is on, draw the model's picked target
                    # (from the latest inference step) instead of the human
                    # click that clicking is disabled anyway.
                    if getattr(cfg, "use_molmo", False):
                        molmo_state = gui.controller.get_molmo_state()
                        overlay_point = molmo_state.get("target_point")
                    else:
                        overlay_point = gui._selected_point_copy()
                    jpeg = encode_frame_jpeg(
                        cam,
                        data,
                        overlay_point,
                        target_sigma=cfg.target_sigma,
                        depth_clip_m=depth_clip_m,
                    )
                    self.send_response(200)
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Content-Length", str(len(jpeg)))
                    self.end_headers()
                    self.wfile.write(jpeg)
                    return
                if path == "/api/molmo":
                    state = gui.controller.get_molmo_state()
                    target_xy = state.get("target_xy")
                    payload = {
                        "use_molmo": bool(getattr(gui.controller._cfg, "use_molmo", False)),
                        "prompt": state.get("prompt", ""),
                        "raw": state.get("raw", ""),
                        "language": state.get("language", ""),
                        "target_xy": (
                            [float(target_xy[0]), float(target_xy[1])]
                            if target_xy is not None
                            else None
                        ),
                        "prev_action_str": state.get("prev_action_str", ""),
                        "model_prompt": state.get("model_prompt", ""),
                        "last_instruction": gui.controller.last_submitted_goal,
                    }
                    self._send_json(payload)
                    return
                self.send_error(404)

            def do_POST(self) -> None:
                path = urlparse(self.path).path
                data = self._read_json()
                if path == "/api/click":
                    try:
                        u_px, v_px = gui._set_point(
                            data["camera"],
                            float(data["u"]),
                            float(data["v"]),
                            normalized=True,
                        )
                    except ValueError as exc:
                        self._send_json({"error": str(exc)}, status=400)
                        return
                    self._send_json(
                        {"camera": data["camera"], "u": u_px, "v": v_px}
                    )
                    return
                if path == "/api/clear_point":
                    gui._clear_point()
                    self._send_json({"ok": True})
                    return
                if path == "/api/format":
                    gui.controller._cfg.point_goal_format = data.get(
                        "format", gui.controller._cfg.point_goal_format
                    )
                    self._send_json({"format": gui.controller._cfg.point_goal_format})
                    return
                if path == "/api/submit":
                    goal = str(data.get("goal", "")).strip()
                    if not goal:
                        self._send_json({"error": "Enter an action goal first."}, status=400)
                        return
                    point = gui._selected_point_copy()
                    record = bool(
                        data.get("record", getattr(gui.controller._cfg, "record", True))
                    )
                    # Queued before the goal so the inference loop grades the
                    # command that ran, not the one being submitted here.
                    verdict = data.get("verdict")
                    graded = (
                        grade_running_command(gui.controller, bool(verdict))
                        if verdict is not None
                        else ""
                    )
                    try:
                        formatted = submit_goal_from_parts(
                            gui.controller, goal, point, record=record
                        )
                    except ValueError as exc:
                        self._send_json({"error": str(exc)}, status=400)
                        return
                    self._send_json({"formatted": formatted, "graded": graded})
                    return
                if path == "/api/command":
                    command = str(data.get("command", "")).strip().lower()
                    try:
                        message = submit_gui_command(gui.controller, command)
                    except ValueError as exc:
                        self._send_json({"error": str(exc)}, status=400)
                        return
                    self._send_json({"message": message})
                    return
                if path == "/api/stop":
                    if "success" not in data:
                        self._send_json(
                            {"error": "stop needs success=true|false"}, status=400
                        )
                        return
                    message = stop_with_verdict(
                        gui.controller, bool(data.get("success"))
                    )
                    self._send_json({"message": message})
                    return
                if path == "/api/molmo":
                    prompt = str(data.get("prompt", "")).strip()
                    if not prompt:
                        self._send_json(
                            {"error": "Enter a Molmo2 HL prompt first."}, status=400
                        )
                        return
                    auto_start = bool(data.get("auto_start", True))
                    gui.controller.submit_molmo_prompt(prompt, auto_start=auto_start)
                    self._send_json({"prompt": prompt})
                    return
                if path == "/api/shutdown":
                    gui.controller.request_shutdown()
                    self._send_json({"ok": True})
                    return
                self.send_error(404)

        return Handler

    def run(self) -> None:
        handler = self._make_handler()
        self._server = _QuietThreadingHTTPServer((self.host, self.port), handler)
        thread = threading.Thread(
            target=self._server.serve_forever, name="web-gui", daemon=True
        )
        thread.start()
        local_url = f"http://127.0.0.1:{self.port}"
        logger.info("Web GUI listening on http://0.0.0.0:%d", self.port)
        print(
            f"\n{'=' * 60}\n"
            f"  Web GUI: {local_url}\n"
            f"  Remote SSH: ssh -L {self.port}:localhost:{self.port} user@this-host\n"
            f"  Then open http://localhost:{self.port} in your browser\n"
            f"{'=' * 60}\n",
            flush=True,
        )
        try:
            while not self.controller._shutdown_requested.is_set():
                sleep(0.2)
        except KeyboardInterrupt:
            pass
        finally:
            if self._server is not None:
                self._server.shutdown()
            self.controller.request_shutdown()


def launch_goal_gui(
    controller: InferenceController,
    *,
    gui_port: int = WEB_GUI_PORT,
) -> None:
    WebGoalGui(controller, port=gui_port).run()
