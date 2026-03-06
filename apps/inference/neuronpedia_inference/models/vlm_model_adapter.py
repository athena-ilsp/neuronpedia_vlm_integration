"""VLM change: Adapter that wraps stock Gemma3ForConditionalGeneration to match
the HookedTransformer interface expected by Neuronpedia's existing endpoints.

Uses standard PyTorch forward hooks to capture activations — no HookedGemma3
dependency, works with any transformers version, supports images via pixel_values.

Captured hook points per layer (matching the names your SAEs/transcoders were trained on):
  - language_model.model.layers.{i}.hook_mlp_out  (MLP output — SAE training target)
  - language_model.model.layers.{i}.hook_mlp_in   (MLP input  — transcoder input)
"""

import base64
import io
import logging
from types import SimpleNamespace

import torch
from PIL import Image

logger = logging.getLogger(__name__)



class VLMModelAdapter:
    """VLM change: Wraps stock Gemma3ForConditionalGeneration to expose the
    HookedTransformer-like interface that existing Neuronpedia endpoints expect.

    Registers PyTorch forward hooks on each transformer layer's MLP to capture:
      - hook_mlp_out: output of the MLP (what SAEs are trained on)
      - hook_mlp_in:  input to the MLP (what transcoders use as input)

    Works with standard transformers Gemma3ForConditionalGeneration — no hooked
    model variant required. Image inputs are supported via set_image().
    """

    def __init__(self, vlm_model, processor):
        self.vlm_model = vlm_model
        self.processor = processor
        self.tokenizer = processor.tokenizer

        # VLM change: expose config attributes matching HookedTransformer.cfg
        text_config = vlm_model.config.text_config
        self.cfg = SimpleNamespace(
            n_layers=text_config.num_hidden_layers,
            d_head=text_config.head_dim,
            n_heads=text_config.num_attention_heads,
            n_key_value_heads=text_config.num_key_value_heads,
            tokenizer_prepends_bos=True,
            default_prepend_bos=True,
        )

        # VLM change: store current image for the next run_with_cache call
        self._current_pixel_values = None
        self._current_token_type_ids = None
        self._pil_image = None  # VLM change: PIL image needed by processor for tokenization

        # VLM change: build a map of hook_name -> MLP module for each layer.
        # In transformers 4.57: Gemma3ForConditionalGeneration
        #   .model          -> Gemma3Model
        #   .language_model -> Gemma3TextModel  (has .layers directly)
        self._layers = vlm_model.model.language_model.layers
        logger.info("VLMModelAdapter: found %d transformer layers", len(self._layers))

    @property
    def device(self):
        return next(self.vlm_model.parameters()).device

    def set_image(self, image_base64: str | None):
        """VLM change: Store an image for the next run_with_cache call. Pass None to clear."""
        if image_base64:
            image_bytes = base64.b64decode(image_base64)
            pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            self._pil_image = pil_image
            self._current_pixel_values = None  # will be set in to_tokens
        else:
            self._current_pixel_values = None
            self._current_token_type_ids = None
            self._pil_image = None

    def _wrap_in_chat_template(self, text):
        """VLM change: Wrap user text in the Gemma3 chat template to match SAE training format.

        The SAE was trained on chat-formatted data produced by processor.apply_chat_template().
        Using raw text causes a massive activation spike at position 1 that drowns out real features.
        """
        # Strip any existing BOS token — the chat template will add its own
        bos = self.tokenizer.bos_token or ""
        if text.startswith(bos):
            text = text[len(bos):]
        # Build a minimal user-turn message
        messages = [{"role": "user", "content": [{"type": "text", "text": text}]}]
        chat_text = self.processor.apply_chat_template(
            messages, add_generation_prompt=False, tokenize=False
        )
        return chat_text

    def to_tokens(self, text, prepend_bos=True, truncate=False):
        """VLM change: Tokenize text, returning (1, seq_len) tensor on model device.

        Wraps text in the Gemma3 chat template (matching SAE training format).
        If an image was provided via set_image(), image tokens are inserted.
        """
        if self._pil_image is not None:
            # VLM change: build chat-formatted text with image, using the full processor
            # exactly like SAE training (VLMActivationsStore.get_batch_tokens).
            bos = self.tokenizer.bos_token or ""
            raw_text = text[len(bos):] if text.startswith(bos) else text
            messages = [{"role": "user", "content": [
                {"type": "text", "text": raw_text},
                {"type": "image", "image": self._pil_image},
            ]}]
            chat_text = self.processor.apply_chat_template(
                messages, add_generation_prompt=False, tokenize=False
            )
            # Use the full processor to tokenize text + process image together
            result = self.processor(
                text=[chat_text], images=[self._pil_image],
                return_tensors="pt", padding=False, add_special_tokens=False
            )
            self._current_pixel_values = result["pixel_values"].to(str(self.device))
            if "token_type_ids" in result:
                self._current_token_type_ids = result["token_type_ids"].to(self.device)
            return result["input_ids"].to(self.device)

        # VLM change: text-only — wrap in chat template to match SAE training format.
        # We pass add_special_tokens=False because apply_chat_template already adds it.
        chat_text = self._wrap_in_chat_template(text)
        result = self.processor(text=[chat_text], return_tensors="pt", padding=False, add_special_tokens=False)
        tokens = result["input_ids"]
        return tokens.to(self.device)

    def to_str_tokens(self, text, prepend_bos=True):
        """VLM change: Return list of string tokens.

        If pixel_values are set, uses to_tokens() so image patch tokens are included.
        """
        token_ids = self.to_tokens(text, prepend_bos=prepend_bos)[0]
        return [self.tokenizer.decode([tid.item()]) for tid in token_ids]

    def run_with_cache(self, tokens, stop_at_layer=None, **kwargs):
        """VLM change: Run forward pass and return (output, cache) like HookedTransformer.

        Registers PyTorch hooks before the forward pass to capture MLP inputs and outputs
        for every layer. Hooks are removed immediately after the forward pass.

        stop_at_layer: if set, only hooks up to that layer are registered (saves compute).
        pixel_values: automatically included if set_image() was called beforehand.
        """
        cache: dict[str, torch.Tensor] = {}
        handles = []

        # VLM change: register hooks only up to stop_at_layer (or all layers)
        max_layer = stop_at_layer if stop_at_layer is not None else self.cfg.n_layers
        layers = self._layers

        for i in range(min(max_layer, len(layers))):
            layer = layers[i]
            mlp_out_name = f"language_model.model.layers.{i}.hook_mlp_out"
            mlp_in_name = f"language_model.model.layers.{i}.hook_mlp_in"

            # VLM change: hook MLP output (SAE target)
            def make_out_hook(name):
                def hook(module, input, output):
                    cache[name] = output if isinstance(output, torch.Tensor) else output[0]
                return hook

            # VLM change: hook MLP input (transcoder input) via pre-hook
            def make_in_hook(name):
                def hook(module, input):
                    # input is a tuple; first element is the hidden state
                    cache[name] = input[0] if isinstance(input, tuple) else input
                return hook

            handles.append(layer.mlp.register_forward_hook(make_out_hook(mlp_out_name)))
            handles.append(layer.mlp.register_forward_pre_hook(make_in_hook(mlp_in_name)))

        # VLM change: run stock forward pass — works with or without pixel_values
        # Ensure input_ids is 2D (batch, seq_len) as required by Gemma3ForConditionalGeneration
        tokens_2d = tokens if tokens.dim() == 2 else tokens.unsqueeze(0)
        run_kwargs: dict = {"input_ids": tokens_2d}
        if self._current_pixel_values is not None:
            run_kwargs["pixel_values"] = self._current_pixel_values
            # VLM change: use token_type_ids from the processor if available,
            # otherwise compute from image token positions.
            if self._current_token_type_ids is not None:
                run_kwargs["token_type_ids"] = self._current_token_type_ids
            else:
                image_token_id = self.processor.image_token_id  # 262144 for Gemma3
                run_kwargs["token_type_ids"] = (tokens_2d == image_token_id).long()
        else:
            # VLM change: text-only — all zeros (no image tokens)
            run_kwargs["token_type_ids"] = torch.zeros_like(tokens_2d)
        # pass any extra kwargs (e.g. attention_mask)
        run_kwargs.update(kwargs)

        with torch.no_grad():
            output = self.vlm_model(**run_kwargs)

        # VLM change: always remove hooks after forward pass
        for h in handles:
            h.remove()

        return output, cache