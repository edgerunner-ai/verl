# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch

WeightUpdate = tuple[str, torch.Tensor]

_LM_HEAD_SUFFIX = "lm_head.weight"


def ensure_tied_embed_aliases(
    weights: list[WeightUpdate],
    *,
    include_language_model_alias: bool | None = None,
    tie_word_embeddings: bool = True,
) -> list[WeightUpdate]:
    """Duplicate tied ``lm_head`` tensors as ``embed_tokens`` in the same batch.

    Newer vLLM ``AutoWeightsLoader`` skips ``lm_head.weight`` when it is tied to
    ``embed_tokens``, then errors if that embed name is missing from the *same*
    ``load_weights`` call. FSDP2 often emits only ``lm_head`` (or puts the two
    names in different IPC buckets). Megatron-Bridge already exports the embed
    name this check wants.

    Untied models (NemotronH, ``tie_word_embeddings=False``) already send a
    real ``embeddings`` tensor. vLLM renames that to ``embed_tokens``, and an
    extra alias is a second copy of ``lm_head`` loaded into the same parameter
    afterwards, so the input embeddings become the output matrix.

    Gemma-4 multimodal vLLM looks up ``language_model.model.embed_tokens``.
    Emit that alias only when the live module has a ``language_model`` child
    (or the batch already uses the prefix). Injecting it into a text model
    such as NemotronH makes ``AutoWeightsLoader`` raise on a missing module.
    """
    if not tie_word_embeddings:
        return weights
    if include_language_model_alias is None:
        include_language_model_alias = any(name.startswith("language_model.") for name, _ in weights)
    names = {name for name, _ in weights}
    extra: list[WeightUpdate] = []

    def _add(alias: str, tensor: torch.Tensor) -> None:
        if alias not in names:
            extra.append((alias, tensor))
            names.add(alias)

    for name, tensor in weights:
        if name.endswith(_LM_HEAD_SUFFIX):
            prefix = name[: -len(_LM_HEAD_SUFFIX)]
            _add(f"{prefix}model.embed_tokens.weight", tensor)
            if prefix in ("", "language_model."):
                _add("model.embed_tokens.weight", tensor)
            if include_language_model_alias:
                _add("language_model.model.embed_tokens.weight", tensor)
        elif name.endswith("model.embed_tokens.weight"):
            if name == "model.embed_tokens.weight" and include_language_model_alias:
                _add("language_model.model.embed_tokens.weight", tensor)
            elif name == "language_model.model.embed_tokens.weight":
                _add("model.embed_tokens.weight", tensor)

    if not extra:
        return weights
    return list(weights) + extra


def split_buffer_updates(
    model: torch.nn.Module, weights: list[WeightUpdate]
) -> tuple[list[WeightUpdate], list[WeightUpdate], dict[str, torch.Tensor]]:
    """Split incoming weight updates into parameter and buffer updates.

    Returns the parameter updates, the buffer updates, and the model's
    ``named_buffers`` map so callers can reuse it without re-iterating.
    """
    named_buffers = dict(model.named_buffers())
    param_updates, buffer_updates = [], []
    for name, tensor in weights:
        if name in named_buffers:
            buffer_updates.append((name, tensor))
        else:
            param_updates.append((name, tensor))
    return param_updates, buffer_updates, named_buffers


@torch.no_grad()
def apply_buffer_updates(
    model: torch.nn.Module,
    buffer_updates: list[WeightUpdate],
    named_buffers: dict[str, torch.Tensor] | None = None,
) -> int:
    """Copy updated buffer tensors into the target model in-place."""
    if not buffer_updates:
        return 0

    if named_buffers is None:
        named_buffers = dict(model.named_buffers())
    loaded = 0
    for name, tensor in buffer_updates:
        if name not in named_buffers:
            continue

        target = named_buffers[name]
        if target.shape != tensor.shape:
            raise ValueError(
                f"Buffer shape mismatch for {name}: expected {tuple(target.shape)}, got {tuple(tensor.shape)}"
            )

        source = tensor.to(device=target.device, dtype=target.dtype, non_blocking=False)
        target.copy_(source, non_blocking=False)
        loaded += 1

    return loaded
