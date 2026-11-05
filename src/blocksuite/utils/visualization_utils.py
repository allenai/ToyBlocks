import os
import textwrap
from typing import Optional

import imageio
import numpy as np
import tqdm
import torch


# Default colormap for heatmap overlays. "turbo" is matplotlib's perceptually
# smoother drop-in for jet, giving a cleaner heat-glow look than a hand-rolled
# rainbow. Swap for any matplotlib colormap name (e.g. "inferno", "magma").
HEATMAP_COLORMAP = "turbo"


def _apply_colormap(x: np.ndarray, cmap: str = HEATMAP_COLORMAP) -> np.ndarray:
    """Map intensities in ``[0, 1]`` (HxW) to RGB floats in ``[0, 1]`` (HxWx3)
    using a matplotlib colormap."""
    import matplotlib

    x = np.clip(x, 0.0, 1.0)
    rgb = matplotlib.colormaps[cmap](x)[..., :3]  # drop alpha
    return np.asarray(rgb, dtype=np.float32)


def overlay_heatmap_and_text(
    frame: np.ndarray,
    heatmap: Optional[np.ndarray] = None,
    text: Optional[str] = None,
    *,
    alpha: float = 0.5,
) -> np.ndarray:
    """Blend a conditioning ``heatmap`` (``HEATMAP_COLORMAP``) onto an RGB ``frame`` and draw ``text``.

    Args:
        frame: ``HxWx3`` uint8 RGB image (the camera view the heatmap is aligned to).
        heatmap: 2D float array (any resolution; resized to the frame). Values are
            normalized by their own max, so a Gaussian peak reads as full color and
            an all-zero map (language-only step) leaves the frame untouched.
        text: HL language instruction drawn top-left over a dark band. Skipped when
            empty.
        alpha: peak opacity of the colored heatmap overlay.

    Returns an ``HxWx3`` uint8 RGB image.
    """
    from PIL import Image, ImageDraw, ImageFont

    frame = np.ascontiguousarray(np.asarray(frame)[..., :3]).astype(np.uint8)
    H, W = frame.shape[:2]
    out = frame

    if heatmap is not None:
        hm = np.asarray(heatmap, dtype=np.float32)
        hm_max = float(hm.max()) if hm.size else 0.0
        if hm_max > 0.0:
            hm = hm / hm_max
            hm_img = Image.fromarray((np.clip(hm, 0.0, 1.0) * 255).astype(np.uint8)).resize(
                (W, H), Image.BILINEAR
            )
            hm_resized = np.asarray(hm_img, dtype=np.float32) / 255.0
            color = _apply_colormap(hm_resized) * 255.0
            a = (hm_resized * float(alpha))[..., None]
            out = np.clip((1.0 - a) * frame.astype(np.float32) + a * color, 0, 255).astype(np.uint8)

    if text:
        img = Image.fromarray(out)
        draw = ImageDraw.Draw(img)
        font = ImageFont.load_default()
        # Wrap on an estimate of characters-per-line for the frame width.
        char_w = max(1, draw.textlength("x", font=font))
        wrapped = textwrap.fill(text, width=max(8, int(W / char_w) - 1))
        x0, y0 = 3, 2
        bbox = draw.multiline_textbbox((x0, y0), wrapped, font=font)
        draw.rectangle([0, 0, W, bbox[3] + 2], fill=(0, 0, 0))
        draw.multiline_text((x0, y0), wrapped, fill=(255, 255, 255), font=font)
        out = np.asarray(img)

    return out


def overlay_hl_input_debug(
    frame: np.ndarray,
    *,
    prev_action: str,
    prompt: str,
    hl_output: str,
) -> np.ndarray:
    """Burn the exact Molmo2 previous-action / prompt / output onto ``frame``.

    ``prev_action`` is the string that was prepended as ``Previous action:``
    (empty on the first HL call of an episode). ``prompt`` is the full question
    Molmo2 received. ``hl_output`` is the model's latest raw string.
    """
    from PIL import Image, ImageDraw, ImageFont

    frame = np.ascontiguousarray(np.asarray(frame)[..., :3]).astype(np.uint8)
    _, W = frame.shape[:2]
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    char_w = max(1, draw.textlength("x", font=font))
    wrap = max(8, int(W / char_w) - 1)
    prev_label = prev_action if prev_action else "(none — first HL call of episode)"
    lines = [
        f"PREV: {prev_label}",
        f"PROMPT: {prompt}" if prompt else "PROMPT: (none)",
        f"OUT: {hl_output}" if hl_output else "OUT: (none)",
    ]
    wrapped = "\n".join(textwrap.fill(line, width=wrap) for line in lines)
    x0, y0 = 3, 2
    bbox = draw.multiline_textbbox((x0, y0), wrapped, font=font)
    draw.rectangle([0, 0, W, bbox[3] + 2], fill=(0, 0, 0))
    draw.multiline_text((x0, y0), wrapped, fill=(255, 255, 255), font=font)
    return np.asarray(img)


def images_to_video(
    images: list[np.ndarray],
    output_dir: str,
    video_name: str,
    fps: int = 10,
    quality: Optional[float] = 9,
    verbose: bool = True,
    **kwargs,
):
    r"""Calls imageio to run FFMPEG on a list of images. For more info on
    parameters, see https://imageio.readthedocs.io/en/stable/format_ffmpeg.html
    Args:
        images: The list of images. Images should be HxWx3 in RGB order.
        output_dir: The folder to put the video in.
        video_name: The name for the video.
        fps: Frames per second for the video. Not all values work with FFMPEG,
            use at your own risk.
        quality: Default is 5. Uses variable bit rate. Highest quality is 10,
            lowest is 0.  Set to None to prevent variable bitrate flags to
            FFMPEG so you can manually specify them using output_params
            instead. Specifying a fixed bitrate using ‘bitrate’ disables
            this parameter.
    References:
        https://github.com/facebookresearch/habitat-lab/blob/main/habitat/utils/visualizations/utils.py
    """
    assert 0 <= quality <= 10
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    video_name = video_name.replace(" ", "_").replace("\n", "_") + ".mp4"
    output_path = os.path.join(output_dir, video_name)
    writer = imageio.get_writer(output_path, fps=fps, quality=quality, **kwargs)
    if verbose:
        print(f"Video created: {output_path}")
        images_iter = tqdm.tqdm(images)
    else:
        images_iter = images
    for im in images_iter:
        writer.append_data(im)
    writer.close()

def tile_images(images: list[np.ndarray | torch.Tensor], nrows=1) -> np.ndarray | torch.Tensor:
    """
    Tile multiple images to a single image comprised of nrows and an appropriate number of columns to fit all the images.
    The images can also be batched (e.g. of shape (B, H, W, C)), but give images must all have the same batch size.

    if nrows is 1, images can be of different sizes. If nrows > 1, they must all be the same size.
    """
    # Sort images in descending order of vertical height
    batched = False
    if len(images[0].shape) == 4:
        batched = True
    if nrows == 1:
        images = sorted(images, key=lambda x: x.shape[0 + batched], reverse=True)

    columns = []
    if batched:
        max_h = images[0].shape[1] * nrows
        cur_h = 0
        cur_w = images[0].shape[2]
    else:
        max_h = images[0].shape[0] * nrows
        cur_h = 0
        cur_w = images[0].shape[1]

    # Arrange images in columns from left to right
    column = []
    for im in images:
        if cur_h + im.shape[0 + batched] <= max_h and cur_w == im.shape[1 + batched]:
            column.append(im)
            cur_h += im.shape[0 + batched]
        else:
            columns.append(column)
            column = [im]
            cur_h, cur_w = im.shape[0 + batched : 2 + batched]
    columns.append(column)

    # Tile columns
    total_width = sum(x[0].shape[1 + batched] for x in columns)

    is_torch = False
    if torch is not None:
        is_torch = isinstance(images[0], torch.Tensor)

    output_shape = (max_h, total_width, 3)
    if batched:
        output_shape = (images[0].shape[0], max_h, total_width, 3)
    if is_torch:
        output_image = torch.zeros(output_shape, dtype=images[0].dtype)
    else:
        output_image = np.zeros(output_shape, dtype=images[0].dtype)
    cur_x = 0
    for column in columns:
        cur_w = column[0].shape[1 + batched]
        next_x = cur_x + cur_w
        if is_torch:
            column_image = torch.concatenate(column, dim=0 + batched)
        else:
            column_image = np.concatenate(column, axis=0 + batched)
        cur_h = column_image.shape[0 + batched]
        output_image[..., :cur_h, cur_x:next_x, :] = column_image
        cur_x = next_x
    return output_image