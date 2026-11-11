"""Image + text graph prompts: decoding, where the image goes, and the metadata the UI reads."""

import base64
import io
import os

import pytest
from PIL import Image

from neuronpedia_graph.image_prompt import (
    ImageInputError,
    _grid,
    decode_image,
    image_insertion,
    insert_image,
    load_processor,
)

GEMMA_PREFIX = "<bos><start_of_turn>user\n"


def _png_base64() -> str:
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), (255, 0, 0)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def test_decode_image_reads_plain_and_data_url_base64():
    raw = _png_base64()
    assert decode_image(raw).size == (8, 8)
    assert decode_image(f"data:image/png;base64,{raw}").mode == "RGB"


@pytest.mark.parametrize("bad", ["not base64!!", base64.b64encode(b"not an image").decode()])
def test_decode_image_refuses_what_is_not_an_image(bad):
    with pytest.raises(ImageInputError):
        decode_image(bad)


def test_the_image_goes_after_the_user_turn_opening():
    prompt = f"{GEMMA_PREFIX}The red shape is a<end_of_turn>\n<start_of_turn>model\n"
    out = insert_image(prompt, GEMMA_PREFIX, "<start_of_image>", "<bos>")
    assert out == f"{GEMMA_PREFIX}<start_of_image>The red shape is a<end_of_turn>\n<start_of_turn>model\n"


def test_a_template_rendered_without_bos_still_matches_a_prompt_with_one():
    out = insert_image("<bos><start_of_turn>user\nHi", "<start_of_turn>user\n", "<img>", "<bos>")
    assert out == "<bos><start_of_turn>user\n<img>Hi"


def test_a_raw_prompt_gets_the_image_right_after_the_bos():
    assert insert_image("<bos>The red shape is a", GEMMA_PREFIX, "<img>", "<bos>") == "<bos><img>The red shape is a"
    assert insert_image("The red shape is a", GEMMA_PREFIX, "<img>", None) == "<img>The red shape is a"


def test_grid_is_square_when_it_can_be():
    assert _grid(256) == (16, 16)
    assert _grid(10) == (1, 10)


def test_grid_follows_qwen_grid_thw_after_the_merge():
    import torch

    # A 1920x1080 photo on Qwen2.5-VL: 24x42 patches, 2x2-merged into 12x21 tokens.
    assert _grid(252, torch.tensor([[1, 24, 42]]), merge_size=2) == (12, 21)


@pytest.mark.skipif(not os.getenv("HF_TOKEN"), reason="google/gemma-3-4b-it is gated; set HF_TOKEN")
def test_gemma_3_template_puts_start_of_image_after_the_user_turn():
    prefix, placeholder = image_insertion(load_processor("google/gemma-3-4b-it"))
    assert prefix == GEMMA_PREFIX
    assert placeholder == "<start_of_image>"
