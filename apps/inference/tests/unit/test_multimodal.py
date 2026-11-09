"""Image + text prompts (`neuronpedia_inference.multimodal`) and their eager-only capture path.

CPU only and weight-free: decoding, grid inference and the refusals. The end-to-end path with a
real vision-language model is exercised by the GPU integration test.
"""

import asyncio
import base64
import io
from types import SimpleNamespace
from typing import cast

import pytest
import torch
from interp_engine import InterpModel
from PIL import Image

from neuronpedia_inference import multimodal
from neuronpedia_inference.engine_adapter import BackendUnsupported, capture_cache_async
from neuronpedia_inference.multimodal import MultimodalInputError, build_image_prompt, decode_image


def _png_base64(size: tuple[int, int] = (8, 6), mode: str = "RGBA") -> str:
    buf = io.BytesIO()
    Image.new(mode, size, (10, 20, 30, 255) if mode == "RGBA" else (10, 20, 30)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


class _NotEager:
    """A non-eager backend as far as the image path can tell."""

    def describe(self) -> SimpleNamespace:
        return SimpleNamespace(backend="vllm")


def test_decode_image_returns_rgb_from_a_bare_string_and_a_data_url():
    data = _png_base64()
    for payload in (data, f"data:image/png;base64,{data}"):
        img = decode_image(payload)
        assert img.mode == "RGB"
        assert img.size == (8, 6)


@pytest.mark.parametrize(
    ("payload", "match"),
    [
        ("not base64 !!!", "not valid base64"),
        (base64.b64encode(b"definitely not an image").decode(), "Could not read the image"),
    ],
)
def test_decode_image_refuses_unusable_input_as_a_client_error(payload: str, match: str):
    with pytest.raises(MultimodalInputError, match=match):
        decode_image(payload)


def test_decode_image_refuses_an_oversized_payload_before_decoding_it(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(multimodal, "MAX_IMAGE_BYTES", 16)
    with pytest.raises(MultimodalInputError, match="larger than"):
        decode_image(_png_base64(size=(64, 64)))


def test_a_client_error_is_a_value_error_so_existing_handlers_still_catch_it():
    assert issubclass(MultimodalInputError, ValueError)


def test_image_grid_is_square_when_the_processor_states_none_and_the_count_is_square():
    # Gemma 3: no grid in the batch, a fixed 256 = 16 x 16 placeholders.
    assert multimodal._image_grid(SimpleNamespace(), {}, 256) == (16, 16)
    assert multimodal._image_grid(SimpleNamespace(), {}, 255) == (None, None)
    assert multimodal._image_grid(SimpleNamespace(), {}, 0) == (None, None)


def test_image_grid_follows_image_grid_thw_after_the_spatial_merge():
    # Qwen-VL: the processor states (t, h, w) in patches before a 2 x 2 merge.
    processor = SimpleNamespace(image_processor=SimpleNamespace(merge_size=2))
    batch = {"image_grid_thw": torch.tensor([[1, 28, 40]])}
    assert multimodal._image_grid(processor, batch, 14 * 20) == (14, 20)
    # A count that disagrees with the stated grid gets no grid rather than a wrong one.
    assert multimodal._image_grid(processor, batch, 100) == (None, None)


def test_build_image_prompt_refuses_off_the_eager_backend_before_decoding():
    with pytest.raises(BackendUnsupported, match="eager backend.*vllm"):
        build_image_prompt(cast(InterpModel, _NotEager()), "hi", "not even an image")


def test_capture_refuses_forward_kwargs_off_the_eager_backend():
    with pytest.raises(BackendUnsupported, match="eager backend"):
        asyncio.run(
            capture_cache_async(
                cast(InterpModel, _NotEager()),
                torch.tensor([1, 2, 3]),
                ["blocks.0.hook_resid_post"],
                {"pixel_values": torch.zeros(1, 3, 4, 4)},
            )
        )
