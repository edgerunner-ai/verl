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

import importlib.util
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_weight_update_utils():
    module_path = _REPO_ROOT / "verl/workers/rollout/vllm_rollout/weight_update_utils.py"
    spec = importlib.util.spec_from_file_location("weight_update_utils_tied", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ensure_tied_embed_aliases = _load_weight_update_utils().ensure_tied_embed_aliases


def test_ensure_tied_embed_aliases_duplicates_gemma4_mm_lm_head():
    head = torch.ones(3, 2)
    out = ensure_tied_embed_aliases([("language_model.lm_head.weight", head)])
    names = [n for n, _ in out]
    assert "language_model.lm_head.weight" in names
    assert "language_model.model.embed_tokens.weight" in names
    assert "model.embed_tokens.weight" in names
    aliased = dict(out)
    assert aliased["language_model.model.embed_tokens.weight"] is head


def test_ensure_tied_embed_aliases_noop_when_embed_already_present():
    t = torch.ones(2, 2)
    weights = [
        ("language_model.lm_head.weight", t),
        ("language_model.model.embed_tokens.weight", t),
    ]
    out = ensure_tied_embed_aliases(weights)
    assert [n for n, _ in out].count("language_model.model.embed_tokens.weight") == 1
