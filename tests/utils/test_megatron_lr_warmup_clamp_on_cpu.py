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

"""CPU tests for Megatron warmup clamping (no megatron-core required)."""

import importlib.util
import sys
import types
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_clamp():
    megatron = types.ModuleType("megatron")
    core = types.ModuleType("megatron.core")
    optimizer = types.ModuleType("megatron.core.optimizer")
    scheduler = types.ModuleType("megatron.core.optimizer_param_scheduler")
    optimizer.OptimizerConfig = object
    optimizer.get_megatron_optimizer = lambda *a, **k: None
    scheduler.OptimizerParamScheduler = object
    core.optimizer = optimizer
    core.optimizer_param_scheduler = scheduler
    megatron.core = core

    logger = types.ModuleType("verl.utils.logger")
    logger.print_rank_0 = lambda *a, **k: None
    dtypes = types.ModuleType("verl.utils.torch_dtypes")

    class _PrecisionType:
        pass

    dtypes.PrecisionType = _PrecisionType

    fakes = {
        "megatron": megatron,
        "megatron.core": core,
        "megatron.core.optimizer": optimizer,
        "megatron.core.optimizer_param_scheduler": scheduler,
        "verl.utils.logger": logger,
        "verl.utils.torch_dtypes": dtypes,
    }
    saved = {name: sys.modules.get(name) for name in fakes}
    sys.modules.update(fakes)
    try:
        path = _REPO_ROOT / "verl/utils/megatron/optimizer.py"
        spec = importlib.util.spec_from_file_location("megatron_optimizer_under_test", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module._clamp_megatron_lr_warmup_steps
    finally:
        for name, prev in saved.items():
            if prev is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = prev


_clamp = _load_clamp()


def test_clamp_warmup_when_ge_decay():
    warmup, decay = _clamp(20, 10)
    assert decay == 10
    assert warmup == 9


def test_clamp_noop_when_warmup_below_decay():
    warmup, decay = _clamp(20, 60)
    assert (warmup, decay) == (20, 60)
