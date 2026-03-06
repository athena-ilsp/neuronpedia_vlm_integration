import logging
import sys
from types import SimpleNamespace
from typing import Any

import torch

from neuronpedia_inference.saes.base import BaseSAE

logger = logging.getLogger(__name__)


class VlmSAEWrapper:
    """Wraps the custom VLM SparseAutoencoder to match the sae-lens SAE interface
    expected by Neuronpedia's existing activation endpoints."""

    def __init__(self, sae, hook_name: str):
        self.inner = sae
        self.W_enc = sae.W_enc  # [d_in, d_sae]
        self.W_dec = sae.W_dec  # [d_sae, d_out]
        self.device = sae.device
        self.dtype = sae.dtype
        self.cfg = SimpleNamespace(
            metadata=SimpleNamespace(
                hook_name=hook_name,
                neuronpedia_id=None,
                prepend_bos=False,
            )
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode activations to feature activations.

        The VLM SAE forward returns (sae_out, feature_acts, loss, mse, l1, ghost).
        We only need feature_acts for inference.
        """
        with torch.no_grad():
            _, feature_acts, *_ = self.inner(x)
        return feature_acts

    def to(self, device: str) -> "VlmSAEWrapper":
        self.inner = self.inner.to(device)
        self.W_enc = self.inner.W_enc
        self.W_dec = self.inner.W_dec
        self.device = self.inner.device
        return self

    def eval(self) -> "VlmSAEWrapper":
        self.inner.eval()
        return self

    def fold_W_dec_norm(self) -> None:
        # No-op: custom VLM SAEs don't use this normalization
        pass


class VlmSAE(BaseSAE):
    @staticmethod
    def load(path: str, device: str, dtype: str) -> tuple[Any, str]:
        """Load a VLM SAE from a .pt checkpoint file.

        Args:
            path: Path to the .pt checkpoint file
            device: Target device
            dtype: Data type string (unused, SAE uses its own dtype from config)

        Returns:
            (VlmSAEWrapper, hook_name) tuple
        """
        # Import here to avoid circular/missing imports when VLM is not configured
        from sae_training.sparse_autoencoder import SparseAutoencoder

        logger.info(f"Loading VLM SAE from: {path}")
        sae = SparseAutoencoder.from_pretrained(path, device="cpu")
        sae.to(device)
        sae.eval()

        hook_name = sae.cfg.hook_point
        wrapper = VlmSAEWrapper(sae, hook_name)

        logger.info(
            f"Loaded VLM SAE: d_in={sae.d_in}, d_sae={sae.d_sae}, "
            f"hook={hook_name}, topk={sae.use_topk}, "
            f"transcoder={sae.cfg.is_transcoder}"
        )

        return wrapper, hook_name
