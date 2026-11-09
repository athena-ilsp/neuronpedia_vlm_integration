"""Image + text prompts for vision-language models.

The single place that turns ``(prompt text, image)`` into what a forward pass and the response
need: the token ids, their display strings, the extra forward inputs (``pixel_values`` and, per
architecture, ``token_type_ids`` / ``image_grid_thw``) and where the image's placeholder tokens
landed. Endpoints call :func:`build_image_prompt` and hand ``forward_kwargs`` to the capture in
``engine_adapter``; nothing else in the server knows about images.

**The prompt is chat-formatted when an image is present.** The image goes into one user turn
ahead of the text, rendered by the model's own processor chat template. That is the only
model-agnostic way to get the image placeholder right (Gemma 3 writes ``<start_of_image>``,
Qwen-VL ``<|vision_start|><|image_pad|><|vision_end|>``), and it is the format VLM-trained
SAEs/transcoders see. Text-only requests are untouched: they keep the raw-prompt convention the
rest of the server uses.

**Eager backend only.** The image enters through the HF forward's keyword inputs, which only the
in-process engine passes; on any other backend this refuses with :class:`BackendUnsupported`
before any work is done.
"""

from __future__ import annotations

import base64
import binascii
import io
import logging
import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import torch
from interp_engine import EagerModel, InterpModel
from PIL import Image, UnidentifiedImageError
from transformers import AutoProcessor

from neuronpedia_inference.engine_adapter import BackendUnsupported, backend_name
from neuronpedia_inference.schemas import ActivationImage

logger = logging.getLogger(__name__)

#: Ceiling on the decoded image, in bytes. Requests arrive as base64 JSON from the webapp; this
#: keeps one request from holding hundreds of MB of host memory before the processor resizes it.
MAX_IMAGE_BYTES = 20 * 1024 * 1024

#: Processor outputs that are not forward kwargs: the ids are passed as tokens, and capture takes
#: the mask through its own argument (a single unpadded prompt needs none).
_NOT_FORWARD_KWARGS = frozenset({"input_ids", "attention_mask"})


class MultimodalInputError(ValueError):
    """A request's image cannot be used: bad encoding, too large, or a model that takes no images.

    Endpoints return it as a 400 -- it is always the caller's input, never a server fault.
    """


@dataclass(frozen=True)
class ImagePrompt:
    """Everything an endpoint needs from one image + text prompt."""

    #: ``[seq]`` token ids, image placeholder tokens expanded.
    tokens: torch.Tensor
    #: One display string per token in ``tokens``.
    str_tokens: list[str]
    #: Extra keyword inputs for the HF forward, already on the model's device and dtype.
    forward_kwargs: dict[str, torch.Tensor]
    #: The placeholder token every image position carries (e.g. ``<image_soft_token>``).
    image_token: str
    #: Indices into ``tokens`` of the image positions, in order.
    image_token_positions: list[int]
    #: Patch grid the image positions tile, row-major, when the processor states or implies one.
    grid_rows: int | None
    grid_cols: int | None
    #: Indices into ``tokens`` of the user's text, in order. Empty with no text, and also when the
    #: text did not tokenize to the same ids inside the template (see :func:`_locate_text`).
    text_token_positions: list[int]
    #: True when there was text but :func:`_locate_text` could not find its ids in the prompt.
    has_unlocated_text: bool = False

    def template_positions(self) -> list[int]:
        """Positions the chat template added: BOS, turn markers, role names, image delimiters.

        Everything that is neither an image placeholder nor the user's text. Empty when the text
        could not be located, so a caller masking these never masks the user's own tokens.
        """
        if self.has_unlocated_text:
            return []
        content = set(self.image_token_positions) | set(self.text_token_positions)
        return [i for i in range(len(self.str_tokens)) if i not in content]


def decode_image(image_base64: str) -> Image.Image:
    """Decode a base64 image (a bare string or a ``data:image/...;base64,`` URL) to RGB."""
    payload = image_base64.split(",", 1)[1] if image_base64.startswith("data:") else image_base64
    # 4 base64 chars carry 3 bytes, so this bounds the decoded size before allocating it.
    if len(payload) * 3 // 4 > MAX_IMAGE_BYTES:
        raise MultimodalInputError(f"Image is larger than {MAX_IMAGE_BYTES // (1024 * 1024)} MiB.")
    try:
        raw = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as e:
        raise MultimodalInputError("Image is not valid base64.") from e
    try:
        with Image.open(io.BytesIO(raw)) as img:
            return img.convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as e:
        raise MultimodalInputError(f"Could not read the image: {e}") from e


def _locate_text(token_ids: list[int], text_ids: list[int], start: int) -> list[int]:
    """Positions of ``text_ids`` as one contiguous run in ``token_ids`` at or after ``start``.

    The last match wins, since the user's text follows the image and any system preamble. Empty
    when there is no exact match -- a template can merge the text's first token with the
    whitespace before it -- in which case the caller reports the text as unlocated rather than
    guessing at its span.
    """
    n = len(text_ids)
    if n == 0:
        return []
    for begin in range(len(token_ids) - n, start - 1, -1):
        if token_ids[begin : begin + n] == text_ids:
            return list(range(begin, begin + n))
    return []


@lru_cache(maxsize=2)
def _processor(hf_model_id: str) -> Any:
    """The model's processor, loaded once per process on the first image request."""
    logger.info("Loading processor for %s (first image request)", hf_model_id)
    return AutoProcessor.from_pretrained(hf_model_id)


def _image_grid(processor: Any, batch: dict[str, Any], n_image_tokens: int) -> tuple[int | None, int | None]:
    """Rows and columns of the patch grid the image tokens tile.

    Qwen-VL states it (``image_grid_thw``, before the spatial merge). Gemma 3 states nothing but
    always emits a square ``image_seq_length``. Anything else gets ``None`` rather than a guess.
    """
    grid_thw = batch.get("image_grid_thw")
    if grid_thw is not None:
        t, h, w = (int(v) for v in grid_thw[0])
        merge = int(getattr(getattr(processor, "image_processor", None), "merge_size", 1) or 1)
        if t == 1 and (h // merge) * (w // merge) == n_image_tokens:
            return h // merge, w // merge
        return None, None
    side = math.isqrt(n_image_tokens)
    if side > 0 and side * side == n_image_tokens:
        return side, side
    return None, None


def build_image_prompt(model: InterpModel, text: str, image_base64: str) -> ImagePrompt:
    """Tokenize ``text`` with one image for ``model``: one chat user turn, image first.

    Raises :class:`BackendUnsupported` off the eager backend and :class:`MultimodalInputError`
    for an unusable image or a model whose processor takes no images.
    """
    if not isinstance(model, EagerModel):
        raise BackendUnsupported(f"Image input needs the eager backend, and this instance runs {backend_name(model)}.")
    image = decode_image(image_base64)
    processor = _processor(model.hf_model_id)
    if getattr(processor, "image_processor", None) is None:
        raise MultimodalInputError(f"{model.hf_model_id} does not take image input.")
    image_token_id = getattr(processor, "image_token_id", None)
    if image_token_id is None:
        raise MultimodalInputError(f"{model.hf_model_id}'s processor names no image token.")

    content: list[dict[str, str]] = [{"type": "image"}]
    if text:
        content.append({"type": "text", "text": text})
    try:
        chat = processor.apply_chat_template(
            [{"role": "user", "content": content}], add_generation_prompt=False, tokenize=False
        )
    except ValueError as e:  # transformers raises ValueError when there is no chat template
        raise MultimodalInputError(f"{model.hf_model_id} has no chat template to place an image in.") from e

    # The template already carries the BOS, so the tokenizer must not add another.
    batch = dict(processor(text=[chat], images=[image], add_special_tokens=False, return_tensors="pt"))
    tokens = batch["input_ids"][0]
    forward_kwargs = {
        key: (value.to(model.device, model.dtype) if value.is_floating_point() else value.to(model.device))
        for key, value in batch.items()
        if key not in _NOT_FORWARD_KWARGS and isinstance(value, torch.Tensor)
    }
    positions = (tokens == image_token_id).nonzero().flatten().tolist()
    rows, cols = _image_grid(processor, batch, len(positions))
    text_ids = processor.tokenizer(text, add_special_tokens=False)["input_ids"] if text else []
    text_positions = _locate_text(tokens.tolist(), text_ids, start=positions[-1] + 1 if positions else 0)
    if text_ids and not text_positions:
        logger.warning("Could not locate the prompt text inside the chat template; template masking is off")
    return ImagePrompt(
        tokens=tokens.to(model.device),
        str_tokens=model.to_str_tokens(tokens),
        forward_kwargs=forward_kwargs,
        image_token=processor.tokenizer.convert_ids_to_tokens(image_token_id),
        image_token_positions=positions,
        grid_rows=rows,
        grid_cols=cols,
        text_token_positions=text_positions,
        has_unlocated_text=bool(text_ids) and not text_positions,
    )


@dataclass(frozen=True)
class PreparedPrompt:
    """A tokenized prompt, text-only or with an image, in the shape every activation endpoint uses."""

    tokens: torch.Tensor
    str_tokens: list[str]
    #: Set when the prompt carries an image.
    image: ImagePrompt | None = None

    @property
    def forward_kwargs(self) -> dict[str, torch.Tensor] | None:
        """The image's extra forward inputs for ``capture_cache_async``; None for text."""
        return None if self.image is None else self.image.forward_kwargs


def prepare_prompt(model: InterpModel, text: str, image_base64: str | None, *, prepend_bos: bool) -> PreparedPrompt:
    """Tokenize ``text`` the way the endpoints always have, or with ``image_base64`` via the processor.

    Text-only keeps each endpoint's own BOS convention (``prepend_bos``, and any BOS the caller
    already put in ``text``); an image prompt is chat-formatted, which carries its own BOS.
    """
    if image_base64 is not None:
        image = build_image_prompt(model, text, image_base64)
        return PreparedPrompt(tokens=image.tokens, str_tokens=image.str_tokens, image=image)
    tokens = model.to_tokens(text, prepend_bos=prepend_bos, truncate=False)[0]
    return PreparedPrompt(tokens=tokens, str_tokens=model.to_str_tokens(text, prepend_bos=prepend_bos))


def activation_image(image: ImagePrompt, dropped_leading: int = 0) -> ActivationImage:
    """The response's ``image`` block, positions re-indexed for a response that dropped leading tokens.

    Endpoints that strip the BOS from ``tokens`` (``/activation/single``) pass ``dropped_leading=1``
    so every position still indexes the ``tokens`` they return.
    """

    def shift(positions: list[int]) -> list[int]:
        return [p - dropped_leading for p in positions if p >= dropped_leading]

    return ActivationImage(
        image_token=image.image_token,
        token_positions=shift(image.image_token_positions),
        grid_rows=image.grid_rows,
        grid_cols=image.grid_cols,
        template_token_positions=shift(image.template_positions()),
    )
