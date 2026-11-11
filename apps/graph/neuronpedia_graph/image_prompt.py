"""Image + text prompts for attribution graphs.

The webapp sends the prompt as text, rendered through the chat template like any other graph
prompt, plus the image as base64. This module puts the image into that prompt the way the model's
own chat template would, runs the model's processor, and returns what circuit-tracer needs: the
token ids (image placeholders included) and the non-token forward inputs, which
``ReplacementModel.multimodal_inputs`` merges in. It also describes where the image sits in the
prompt, for the graph metadata the UI draws the patch grid from.

Where the image goes is read off the chat template rather than hardcoded per family: the template
is rendered with and without an image ahead of the same text, and the difference is the
placeholder (``<start_of_image>`` on Gemma 3) and the point it goes in at (right after the user
turn's opening). A prompt that does not open that way, such as a raw completion prompt, gets the
image at the very start, after the BOS.
"""

from __future__ import annotations

import base64
import binascii
import io
import math
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import torch
from PIL import Image, UnidentifiedImageError

MAX_IMAGE_BYTES = 20 * 1024 * 1024
# Attribution over an image prompt holds every active feature at every image position (256 on
# Gemma 3, ~570k features with top-64 transcoders), so a batch that is fine for text runs out of
# GPU memory; requests are capped at this.
IMAGE_MAX_BATCH_SIZE = int(os.getenv("IMAGE_MAX_BATCH_SIZE", "2"))
# Attribution time grows with the feature nodes it traces back from; at 500 an image graph on
# gemma-3-4b-it takes about 3.5 minutes on one 24 GB GPU, inside the webapp's 5-minute request window.
IMAGE_MAX_FEATURE_NODES = int(os.getenv("IMAGE_MAX_FEATURE_NODES", "500"))
# Pixel cap for processors that size the image token count to the image (see `load_processor`).
IMAGE_MAX_PIXELS = int(os.getenv("IMAGE_MAX_PIXELS", str(448 * 448)))

_PROBE_TEXT = "\x00probe\x00"


class ImageInputError(ValueError):
    """The request's image cannot be used; reported to the caller as a 400."""


@dataclass(frozen=True)
class ImagePrompt:
    """An image + text prompt, ready for circuit-tracer."""

    input_ids: torch.Tensor  # 1-D, image placeholders included
    forward_kwargs: dict[str, torch.Tensor]  # pixel_values and the family's companions
    image_token_id: int
    image_positions: list[int]
    grid_rows: int
    grid_cols: int

    @property
    def text_token_count(self) -> int:
        return int(self.input_ids.shape[0]) - len(self.image_positions)

    def metadata(self, image_base64: str) -> dict[str, Any]:
        """``metadata.image_input`` of the graph JSON; the webapp draws the patch grid from it."""
        return {
            "image_base64": image_base64,
            "image_token_id": self.image_token_id,
            "image_positions": self.image_positions,
            "grid_rows": self.grid_rows,
            "grid_cols": self.grid_cols,
        }


def decode_image(image_base64: str) -> Image.Image:
    """Decode a base64 image (optionally a ``data:`` URL) to RGB."""
    data = image_base64.split(",", 1)[1] if image_base64.startswith("data:") else image_base64
    try:
        raw = base64.b64decode(data, validate=True)
    except (binascii.Error, ValueError) as e:
        raise ImageInputError("image_base64 is not valid base64") from e
    if len(raw) > MAX_IMAGE_BYTES:
        raise ImageInputError(f"image is {len(raw)} bytes; the limit is {MAX_IMAGE_BYTES}")
    try:
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except (UnidentifiedImageError, OSError) as e:
        raise ImageInputError("image_base64 is not a readable image") from e


@lru_cache(maxsize=2)
def load_processor(model_id: str) -> Any:
    """The model's processor.

    Dynamic-resolution processors (Qwen-VL: one token per 28x28 block after its 2x2 merge) are
    capped at ``IMAGE_MAX_PIXELS``; uncapped, a photo becomes over a thousand tokens and runs past
    the token limit. The default, 448x448, is 256 tokens: the same as Gemma 3's fixed count.
    """
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(model_id)
    if getattr(getattr(processor, "image_processor", None), "merge_size", None):
        processor = AutoProcessor.from_pretrained(model_id, max_pixels=IMAGE_MAX_PIXELS)
    if getattr(processor, "image_processor", None) is None or getattr(processor, "image_token_id", None) is None:
        raise ImageInputError(f"{model_id} has no image processor; image input needs a vision-language model")
    return processor


def image_insertion(processor: Any) -> tuple[str, str]:
    """``(prefix, placeholder)``: the template puts ``placeholder`` right after ``prefix``."""

    def render(content: list[dict[str, str]]) -> str:
        return processor.apply_chat_template([{"role": "user", "content": content}], tokenize=False)

    with_image = render([{"type": "image"}, {"type": "text", "text": _PROBE_TEXT}])
    text_only = render([{"type": "text", "text": _PROBE_TEXT}])
    prefix_len = len(os.path.commonprefix([with_image, text_only]))
    suffix_len = len(os.path.commonprefix([with_image[::-1], text_only[::-1]]))
    placeholder = with_image[prefix_len : len(with_image) - suffix_len]
    if not placeholder or _PROBE_TEXT in placeholder:
        raise ImageInputError("could not find where this model's chat template puts an image")
    return with_image[:prefix_len], placeholder


def insert_image(prompt: str, prefix: str, placeholder: str, bos: str | None) -> str:
    """``prompt`` with ``placeholder`` where the chat template would put an image."""
    if prompt.startswith(prefix):
        return prefix + placeholder + prompt[len(prefix) :]
    # The rendered template may open without the BOS the prompt carries.
    if bos and prompt.startswith(bos + prefix):
        head = bos + prefix
        return head + placeholder + prompt[len(head) :]
    if bos and prompt.startswith(bos):
        return bos + placeholder + prompt[len(bos) :]
    return placeholder + prompt


def _grid(n_positions: int, grid_thw: torch.Tensor | None = None, merge_size: int = 1) -> tuple[int, int]:
    """Rows and columns of the patch grid the image tokens tile.

    Qwen-VL states it (``image_grid_thw``, before its ``merge_size`` spatial merge), and the grid
    can be any aspect ratio. Without it, the tokens are taken as a square (Gemma 3's 16x16), and a
    count that is not one is laid out as a single row.
    """
    if grid_thw is not None:
        _, h, w = (int(v) for v in grid_thw[0])
        rows, cols = h // merge_size, w // merge_size
        if rows * cols == n_positions:
            return rows, cols
    side = math.isqrt(n_positions)
    return (side, side) if side * side == n_positions else (1, n_positions)


def build_image_prompt(
    model_id: str,
    prompt: str,
    image_base64: str,
    *,
    device: torch.device | str,
    dtype: torch.dtype,
    bos: str | None,
) -> ImagePrompt:
    """Put the image into ``prompt`` (already chat-templated, BOS included) and run the processor."""
    image = decode_image(image_base64)
    processor = load_processor(model_id)
    prefix, placeholder = image_insertion(processor)
    text = insert_image(prompt, prefix, placeholder, bos)
    batch = processor(text=[text], images=[image], add_special_tokens=False, return_tensors="pt")
    input_ids = batch["input_ids"][0].to(device)
    forward_kwargs = {
        key: (value.to(device, dtype) if value.is_floating_point() else value.to(device))
        for key, value in batch.items()
        if key not in ("input_ids", "attention_mask")
    }
    image_token_id = int(processor.image_token_id)
    image_positions = (input_ids == image_token_id).nonzero().flatten().tolist()
    if not image_positions:
        raise ImageInputError("the processor placed no image tokens in the prompt")
    merge_size = int(getattr(processor.image_processor, "merge_size", 1) or 1)
    rows, cols = _grid(len(image_positions), batch.get("image_grid_thw"), merge_size)
    return ImagePrompt(
        input_ids=input_ids,
        forward_kwargs=forward_kwargs,
        image_token_id=image_token_id,
        image_positions=image_positions,
        grid_rows=rows,
        grid_cols=cols,
    )
