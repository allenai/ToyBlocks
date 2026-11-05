"""Batched Molmo2 high-level (HL) policy: camera frames + goal prompt -> action string.

The model side of ``sim2real/rollout/molmo2_server.py`` on the ``sim2real`` branch
(bf16 weights, greedy decode with a small token budget, the
``Previous action: {prev}. {goal}`` question), generalised to many envs per
call: every request's examples are padded into one batch by the checkpoint's own
collator and decoded by a single ``generate``, instead of one ``generate`` per
env.

The inference contract -- which cameras the HL sees, the front crop, the frame
size its ``<u, v>`` targets are in, and how the goal prompt is phrased -- is read
from the checkpoint's own training config (:func:`hl_contract_from_config`), so
serving a checkpoint cannot silently drift from the data it was trained on
(``RobotTrainDataset`` in molmo_flow's ``olmo/data/robot_datasets.py``).
"""

from __future__ import annotations

import dataclasses
import logging
import re
import time
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)

# RobotTrainDataset.hl_prompt_mode values, and the fixed prompt of "stack_ai2".
HL_PROMPT_MODES = ("stack_ai2", "stack_extract")
STACK_AI2_PROMPT = "Stack Ai2"
# RobotTrainDataset defaults, for a checkpoint whose config predates a field.
DEFAULT_HL_IMAGE_SIZE = (640, 360)
DEFAULT_MAX_NEW_TOKENS = 256
# molmo_flow-only model config keys that molmo2's schema rejects, each with its no-op value.
MOLMO_FLOW_ONLY_KEYS: dict[str, bool] = {
    "vision_backbone.use_image_augmentation": False,
    "vision_backbone.use_resize_bottleneck": False,
    "mm_preprocessor.image.single_frame": True,
}
SINGLE_IMAGE_PREFIX = "Image 1"

_ON = re.compile(r"\(on\s+(block_\S+)\s+(block_\S+)\)")
_ON_TABLE = re.compile(r"\(on_table\s+(block_\S+)\s+\S+\)")
# The last point tag and the action after it, with the label before the tag ("block X ", "table ") when there is one
_HTML_V2_ACTION = re.compile(
    r'(?:(?:block )?\S+ )?<points coords="\d+ \d+ (\d+) (\d+)[^"]*">[^<]*</points>\s*([^<]+)$'
)
# html-v2 question variants (RobotTrainDataset stack_point_prob / point_blocks / plan_prob): the steps asked
# before "point to target. predict action."
HTML_V2_STEPS: dict[str, str] = {
    "plan_stack": "point to stack blocks. plan.",
    "plan_all": "point to all blocks. plan.",
    "plan": "plan.",
    "stack": "point to stack blocks.",
    "all": "point to all blocks.",
    "target": "",  # neither: the question ends in just "point to target. predict action."
}
HTML_V2_INSTRUCTION = "point to target. predict action."


def _stack_token(block: str) -> str:
    name = block.removeprefix("block_")
    # block_l1 / block_red1 / block_y_2 / block_3_2 are duplicates; block_3 is the digit block itself.
    name = re.sub(r"_\d+$", "", name)
    name = re.sub(r"(?<=[A-Za-z])\d+$", "", name)
    return name.upper()


def stack_label(pddl_goal: str) -> str:
    """``Stack A B C`` (top to bottom) per tower in ``pddl_goal``, one tower per line.

    Copy of molmo_flow ``olmo.data.robot_datasets.stack_label`` (the
    ``hl_prompt_mode="stack_extract"`` training question when a row has no
    ``goal.hl_prompt``); keep the two in sync.
    """
    above = {support: top for top, support in _ON.findall(pddl_goal)}
    labels = []
    for base in _ON_TABLE.findall(pddl_goal):
        chain = []
        node: str | None = base
        while node is not None:
            chain.append(_stack_token(node))
            node = above.pop(node, None)
        labels.append("Stack " + " ".join(reversed(chain)))
    if above:
        raise ValueError(f"on-relations not attached to a table block: {above}")
    return "\n".join(labels)


def hl_question_with_prev_action(goal: str, prev_action: str) -> str:
    """The exact Molmo2 question string, matching TAMP ``--use_prev_action_str``.

    First call of an episode (empty prev) is the bare goal. After that the last
    raw HL output is prepended so the model sees the same
    ``Previous action: …`` prefix it was trained with.
    """
    if not prev_action:
        return goal
    return f"Previous action: {prev_action}. {goal}"


def html_v2_to_pixel(
    answer: str,
    image_size: Sequence[float] = DEFAULT_HL_IMAGE_SIZE,
    front_crop: Sequence[float] | None = None,
    point_y_offset_px: float = 0.0,
) -> tuple[str, str] | None:
    """(``action <x, y>`` in `image_size` frame pixels, previous action for the next question), or None.

    The previous action is the answer's action as the model wrote it; the next question starts with
    ``previous action: {it}. ``. `front_crop` (x0, y0, x1, y1) is the crop the model saw, in `image_size` pixels;
    its 0-1000 coordinates are relative to that crop.
    """
    answer = answer.strip()
    if answer.endswith("done"):
        return "done", "done"
    match = _HTML_V2_ACTION.search(answer)
    if match is None:
        return None
    x, y, action = int(match.group(1)), int(match.group(2)), match.group(3).strip()
    if front_crop is not None:
        x0, y0, x1, y1 = front_crop
        x1, y1 = min(x1, image_size[0]), min(y1, image_size[1])
    else:
        x0, y0, (x1, y1) = 0, 0, image_size
    y_px = y0 + y / 1000 * (y1 - y0) - point_y_offset_px  # undo training's --point_y_offset_px (targets moved down)
    return f"{action} <{x0 + x / 1000 * (x1 - x0):.0f}, {y_px:.0f}>", match.group(0)


@dataclasses.dataclass(frozen=True)
class HLContract:
    """What a Molmo2 HL checkpoint was trained to see (``RobotTrainDataset`` kwargs)."""

    #: Only the front camera (``--front_view_crops``); else front + gripper.
    front_view_only: bool = False
    #: ``(x0, y0, x1, y1)`` front-frame pixel crop, or None for the full frame.
    front_crop: tuple[int, int, int, int] | None = None
    #: ``(width, height)`` of the frames, which is also the space of the ``<u, v>`` targets.
    image_size: tuple[int, int] = DEFAULT_HL_IMAGE_SIZE
    #: How the goal prompt was phrased for rows without ``goal.hl_prompt``; None if unknown.
    prompt_mode: str | None = None
    #: html-v2 question variant (a :data:`HTML_V2_STEPS` key), or None for ``action <x, y>`` answers.
    html_v2_variant: str | None = None
    #: Feed the last HL output back as the previous action; False asks every call without one.
    use_prev_action: bool = True
    #: Pixels training moved every target down (``--point_y_offset_px``).
    point_y_offset_px: float = 0.0

    def goal_prompt(self, hl_prompt: str | None, pddl_goal: str | None) -> str:
        """The goal half of the question for one episode, as training built it.

        A recorded ``goal.hl_prompt`` wins; otherwise ``stack_extract`` derives
        ``Stack ...`` from the PDDL goal and ``stack_ai2`` uses a fixed prompt.
        """
        # html-v2 always phrases the goal from the PDDL (``stack Q Y C``), never a recorded prompt.
        if hl_prompt and not self.html_v2_variant:
            return hl_prompt
        if self.prompt_mode == "stack_ai2":
            return STACK_AI2_PROMPT
        if self.prompt_mode == "stack_extract":
            if not pddl_goal:
                raise ValueError(
                    "the HL checkpoint phrases its goal as 'Stack ...' derived from the "
                    "PDDL goal (hl_prompt_mode='stack_extract'), but this episode has no "
                    "goal.hl_prompt and no pddl_goal."
                )
            label = stack_label(pddl_goal)
            if self.html_v2_variant:
                # molmo_flow stack_instruction: "stack T teal 10", only letter blocks upper-case.
                return " ".join(["stack"] + [t if len(t) == 1 and t.isalpha() else t.lower() for t in label.split()[1:]])
            return label
        raise ValueError(
            "the HL checkpoint's config records no hl_prompt_mode, so the goal prompt "
            "it was trained on is unknown; pass an explicit prompt (--hl_prompt)."
        )

    def question(self, goal: str, prev_action: str) -> str:
        """The full Molmo2 question for ``goal`` given the env's previous action ("" for none)."""
        if not self.html_v2_variant:
            return hl_question_with_prev_action(goal, prev_action)
        prefix = f"previous action: {prev_action}. " if self.use_prev_action and prev_action else ""
        steps = HTML_V2_STEPS[self.html_v2_variant]
        return f"{prefix}{goal}. {steps + ' ' if steps else ''}{HTML_V2_INSTRUCTION}"

    def parse_output(self, text: str) -> tuple[str, str]:
        """(``action <x, y>`` HL string, previous action for the next question); raises if unparseable."""
        if not self.html_v2_variant:
            return text, text
        parsed = html_v2_to_pixel(text, self.image_size, self.front_crop, self.point_y_offset_px)
        if parsed is None:
            raise ValueError(f"Molmo2 HL answer has no html-v2 target and action: {text!r}")
        return parsed


def _robot_dataset_kwargs(data_cfg: dict[str, Any] | None) -> dict[str, Any] | None:
    """The ``RobotTrainDataset`` kwargs of a training ``data:`` section, in either mixture layout."""
    if not data_cfg:
        return None
    if data_cfg.get("dataset_kwargs"):
        return data_cfg["dataset_kwargs"]
    for mix in data_cfg.get("kwargs_mixture") or []:
        for ds in mix.get("datasets") or []:
            if str(ds.get("dataset_name", "")).startswith("robot_train"):
                return ds.get("kwargs") or {}
    for mix in data_cfg.get("multi_level_mixture") or []:
        for ds in mix.get("datasets") or []:
            inner = ds.get("dataset") or {}
            if str(inner.get("_name", "")).endswith("RobotTrain"):
                return inner
    return None


def hl_contract_from_config(cfg: dict[str, Any]) -> HLContract:
    """Read the HL inference contract off a checkpoint's ``config.yaml`` (as a dict)."""
    kwargs = _robot_dataset_kwargs(cfg.get("data"))
    if kwargs is None:
        logger.warning(
            "HL checkpoint config has no robot_train dataset kwargs; assuming front + "
            "gripper frames, no crop, %s targets, and an explicit goal prompt.",
            DEFAULT_HL_IMAGE_SIZE,
        )
        return HLContract()
    crop = kwargs.get("front_crop")
    size = kwargs.get("image_size")
    plan = float(kwargs.get("plan_prob") or 0) > 0
    # Block points come first when stack_point_prob > 0; point_blocks picks which blocks ("stack" by default).
    blocks: str | None = None
    if float(kwargs.get("stack_point_prob") or 0) > 0:
        blocks = "all" if kwargs.get("point_blocks") == "all" else "stack"
    return HLContract(
        front_view_only=bool(kwargs.get("front_view_only", False)),
        front_crop=tuple(int(v) for v in crop) if crop else None,  # type: ignore[arg-type]
        image_size=tuple(int(v) for v in size) if size else DEFAULT_HL_IMAGE_SIZE,  # type: ignore[arg-type]
        prompt_mode=kwargs.get("hl_prompt_mode"),
        html_v2_variant=((f"plan_{blocks}" if blocks else "plan") if plan else (blocks or "target"))
        if kwargs.get("target_format") == "html-v2"
        else None,
        point_y_offset_px=float(kwargs.get("point_y_offset_px") or 0.0),
    )


def _ensure_single_image_prefix(preprocessor: Any) -> None:
    """Start a lone image's tokens with "Image 1", as in training.

    molmo_flow's multi-image preprocessor already does; molmo2's only tags images when there are several.
    """
    prefix = np.asarray(preprocessor.image_preprocessor.tokenizer.encode(SINGLE_IMAGE_PREFIX))
    blank = np.zeros((*DEFAULT_HL_IMAGE_SIZE[::-1], 3), np.uint8)
    if np.array_equal(preprocessor([blank])[0].tokens[: len(prefix)], prefix):
        return

    class WithSingleImagePrefix(type(preprocessor)):
        def __call__(self, images: Sequence[np.ndarray], *args: Any, **kwargs: Any) -> list[Any]:
            out = super().__call__(images, *args, **kwargs)
            if len(images) == 1:
                out[0].tokens = np.concatenate([prefix, out[0].tokens])
            return out

    preprocessor.__class__ = WithSingleImagePrefix


def to_hwc_uint8(img: Any) -> np.ndarray:
    """One camera frame as HWC uint8 [0, 255] (accepts CHW and float [0, 1])."""
    if torch.is_tensor(img):
        img = img.detach().cpu().numpy()
    arr = np.asarray(img)
    if arr.ndim == 3 and arr.shape[0] == 3 and arr.shape[2] != 3:
        arr = arr.transpose(1, 2, 0)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"expected one HxWx3 RGB frame, got shape {arr.shape}")
    if arr.dtype != np.uint8:
        arr = (arr * 255) if arr.max() <= 1.01 else arr
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


class Molmo2HLPolicy:
    """Molmo2 HL model, batched: one ``generate`` for every example of a call.

    Args:
        checkpoint_path: olmo checkpoint dir (``config.yaml`` + weights).
        device: torch device.
        max_new_tokens: decode budget; HL outputs are ``"pick <409, 210>"``-sized.
        max_batch: examples per ``generate``; larger calls are split.
        use_prev_action: feed each env's last HL output back as the previous action.
        html_v2_variant: overrides the html-v2 question variant read from the config (a
            :data:`HTML_V2_STEPS` key); None keeps the config's.
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cuda",
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        max_batch: int = 64,
        use_prev_action: bool = True,
        html_v2_variant: str | None = None,
    ) -> None:
        # molmo2's collator keys its dtype map on ``np.bool``, which NumPy 2 defines as
        # ``np.bool_``; ToyBlocks pins NumPy 1.26 (torch / Isaac), where the alias was removed.
        if not hasattr(np, "bool"):
            np.bool = np.bool_  # type: ignore[attr-defined]
        from olmo.models.model_config import get_model_types
        from olmo.train.checkpointer import load_model_state
        from olmo.util import resource_path
        from omegaconf import OmegaConf

        cfg = OmegaConf.load(resource_path(checkpoint_path, "config.yaml"))
        self.contract = hl_contract_from_config(
            OmegaConf.to_container(cfg, resolve=False)  # type: ignore[arg-type]
        )
        for key, noop in MOLMO_FLOW_ONLY_KEYS.items():
            parent, leaf = key.rsplit(".", 1)
            if OmegaConf.select(cfg.model, parent, default={}).pop(leaf, noop) != noop:
                raise ValueError(f"HL checkpoint needs model.{key}={noop!r} to load without molmo_flow")
        model_cls = get_model_types()[cfg.model.get("model_name", "molmo")]
        model_cfg = OmegaConf.to_object(
            OmegaConf.merge(OmegaConf.structured(model_cls), model_cls.update_legacy_settings(cfg.model))
        )
        self.contract = dataclasses.replace(
            self.contract,
            use_prev_action=use_prev_action,
            html_v2_variant=html_v2_variant or self.contract.html_v2_variant,
        )
        t0 = time.perf_counter()
        with torch.device("meta"):
            model = model_cfg.build_model()
        # bf16 weights (sim2real molmo2_server): half the memory of the fp32
        # checkpoint and faster decode; the forward ran under bf16 autocast anyway.
        # Cast on the meta device so the fp32 copy is never allocated (molmo_flow's
        # molmoact inference wrapper loads the same way).
        model.to(torch.bfloat16)
        model.to_empty(device=device)
        load_model_state(checkpoint_path, model)
        # Loading can leave tensors on the CPU (or in fp32) for some checkpoints, e.g. a VideoOlmo's
        # new_embedding; move everything onto the device in bf16 once more. A no-op when already there.
        model.to(device=device, dtype=torch.bfloat16)
        model.eval()
        self.model = model
        self.preprocessor = model_cfg.build_preprocessor(for_inference=True, is_training=False)
        _ensure_single_image_prefix(self.preprocessor.preprocessor.multi_image_preprocessor)
        self.collator = model_cfg.build_collator(
            self.preprocessor.get_output_shapes(), pad_mode=None, include_metadata=False
        )
        self.tokenizer = self.preprocessor.preprocessor.text_preprocessor.tokenizer
        self.device = device
        self.max_new_tokens = int(max_new_tokens)
        self.max_batch = int(max_batch)
        logger.info(
            "Molmo2 HL loaded in %.1fs (%s, %s, contract=%s)",
            time.perf_counter() - t0,
            type(model).__name__,
            checkpoint_path,
            self.contract,
        )

    def images_for(self, front: Any, gripper: Any | None) -> list[np.ndarray]:
        """The image list one example feeds the model, per :attr:`contract`."""
        front = to_hwc_uint8(front)
        crop = self.contract.front_crop
        if crop is not None:
            x0, y0, x1, y1 = crop
            front = front[y0:y1, x0:x1]
        if self.contract.front_view_only:
            return [np.ascontiguousarray(front)]
        if gripper is None:
            raise ValueError(
                "this HL checkpoint was trained on front + gripper frames but no "
                "gripper frame was given."
            )
        return [np.ascontiguousarray(front), to_hwc_uint8(gripper)]

    def generate(self, images: Sequence[list[np.ndarray]], questions: Sequence[str]) -> list[str]:
        """Raw HL strings for ``len(questions)`` examples (``images[i]`` from :meth:`images_for`)."""
        if len(images) != len(questions):
            raise ValueError(f"{len(images)} image lists for {len(questions)} questions")
        out: list[str] = []
        for start in range(0, len(questions), self.max_batch):
            out += self._generate_batch(
                images[start : start + self.max_batch], questions[start : start + self.max_batch]
            )
        return out

    def _generate_batch(self, images: Sequence[list[np.ndarray]], questions: Sequence[str]) -> list[str]:
        t0 = time.perf_counter()
        examples = [
            self.preprocessor(
                dict(image=list(ims), question=q, style="user_qa" if self.contract.html_v2_variant else "demo")
            )
            for ims, q in zip(images, questions)
        ]
        for ex in examples:
            ex.pop("metadata", None)
        # The collator right-pads every example to the batch's longest (token
        # padding -1, which generate() masks) and names the tokens input_ids.
        batch = self.collator(examples)
        batch = {k: v.to(self.device) for k, v in batch.items() if torch.is_tensor(v)}
        t1 = time.perf_counter()
        with torch.inference_mode(), torch.autocast(
            torch.device(self.device).type, enabled=True, dtype=torch.bfloat16
        ):
            result = self.model.generate(batch, max_steps=self.max_new_tokens)
        token_ids = result.token_ids[:, 0].tolist()
        t2 = time.perf_counter()
        eos = self.tokenizer.eos_token_id
        texts = []
        for row in token_ids:
            # Padding after an early EOS is -1 (or EOS); keep the decoded prefix only.
            row = [t for t in row if t >= 0]
            if eos in row:
                row = row[: row.index(eos)]
            texts.append(self.tokenizer.decode(row))
        logger.info(
            "Molmo2 HL batch of %d: preprocess %.0fms, generate %.0fms",
            len(examples),
            1000 * (t1 - t0),
            1000 * (t2 - t1),
        )
        return texts
