"""Utilities for collecting a model's final response representation.

The collector deliberately uses a forward hook on the final normalization
layer instead of ``output_hidden_states=True``.  The latter retains every
transformer layer and is prohibitively expensive for long PPO rollouts.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Any

import torch
from torch import nn


_FINAL_NORM_SUFFIXES = (
    "model.norm",
    "model.final_layernorm",
    "model.final_layer_norm",
    "model.norm_f",
    "transformer.ln_f",
    "transformer.final_layernorm",
    "transformer.final_layer_norm",
    "transformer.norm_f",
    "gpt_neox.final_layer_norm",
    "decoder.final_layer_norm",
)


def find_final_norm_module(model: nn.Module) -> tuple[str, nn.Module]:
    """Find the final transformer normalization layer in common HF models."""

    named_modules = list(model.named_modules())
    for suffix in _FINAL_NORM_SUFFIXES:
        matches = [
            (name, module)
            for name, module in named_modules
            if name == suffix or name.endswith(f".{suffix}")
        ]
        if matches:
            # PEFT/FSDP wrappers can expose aliases.  The shortest qualified
            # name is the closest match to the model-level final norm.
            return min(matches, key=lambda item: len(item[0]))

    available_tail = [name for name, _ in named_modules if "norm" in name.lower()][-12:]
    raise RuntimeError(
        "Could not locate the model's final normalization layer for rubric-probe hidden-state collection. "
        f"Last normalization-like modules: {available_tail}"
    )


def _tensor_from_hook_output(output: Any) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)) and output and torch.is_tensor(output[0]):
        return output[0]
    raise TypeError(f"Final normalization hook returned unsupported output type: {type(output)!r}")


class LastHiddenStateCapture(AbstractContextManager):
    """Temporarily capture the output of a model's final normalization layer."""

    def __init__(self, final_norm: nn.Module):
        self.final_norm = final_norm
        self.hidden: torch.Tensor | None = None
        self._handle = None

    def __enter__(self) -> "LastHiddenStateCapture":
        def capture(_module, _inputs, output):
            self.hidden = _tensor_from_hook_output(output).detach()

        self._handle = self.final_norm.register_forward_hook(capture)
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def take(self) -> torch.Tensor:
        if self.hidden is None:
            raise RuntimeError("The final normalization hook did not observe a hidden state during model forward.")
        hidden = self.hidden
        self.hidden = None
        return hidden


def pool_response_hidden(
    response_hidden: torch.Tensor,
    response_mask: torch.Tensor,
    *,
    inplace_mask: bool = False,
    allow_empty: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return last-valid-token and masked-mean representations.

    Args:
        response_hidden: Final-layer states shaped ``[batch, response_tokens, hidden]``.
        response_mask: Valid response-token mask shaped ``[batch, response_tokens]``.
        inplace_mask: Mask padding positions in ``response_hidden`` in-place.  This is
            useful directly after inference to avoid another sequence-sized allocation.
        allow_empty: Return zero vectors for zero-token rows.  PPO uses this for
            synthetic divisor-padding rows, which are filtered before saving.

    Returns:
        ``(last, mean)`` tensors, each shaped ``[batch, hidden]`` and in the
        same dtype as ``response_hidden``.
    """

    if response_hidden.ndim != 3:
        raise ValueError(f"response_hidden must be rank 3, got shape={tuple(response_hidden.shape)}")
    if response_mask.ndim != 2 or response_mask.shape != response_hidden.shape[:2]:
        raise ValueError(
            "response_mask must match the first two response_hidden dimensions, "
            f"got hidden={tuple(response_hidden.shape)} mask={tuple(response_mask.shape)}"
        )

    valid_mask = response_mask.to(device=response_hidden.device, dtype=torch.bool)
    token_counts = valid_mask.sum(dim=-1)
    empty_rows = token_counts <= 0
    if torch.any(empty_rows) and not allow_empty:
        bad_rows = torch.nonzero(token_counts <= 0, as_tuple=False).squeeze(-1).tolist()
        raise ValueError(f"Cannot pool empty responses; zero-token rows={bad_rows}")

    positions = torch.arange(response_hidden.shape[1], device=response_hidden.device).unsqueeze(0)
    last_positions = positions.masked_fill(~valid_mask, -1).max(dim=-1).values.clamp_min(0)
    row_positions = torch.arange(response_hidden.shape[0], device=response_hidden.device)
    last = response_hidden[row_positions, last_positions].clone()
    last[empty_rows] = 0

    masked_hidden = response_hidden if inplace_mask else response_hidden.clone()
    masked_hidden.masked_fill_(~valid_mask.unsqueeze(-1), 0)
    # dtype=float32 only changes the accumulator/result; it does not materialize
    # a float32 copy of the full token-level hidden tensor.
    summed = masked_hidden.sum(dim=1, dtype=torch.float32)
    mean = (summed / token_counts.clamp_min(1).unsqueeze(-1).to(dtype=torch.float32)).to(response_hidden.dtype)
    return last, mean
