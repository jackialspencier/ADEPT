# Copyright 2025 the LlamaFactory team / ADEPT baseline compat.
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

"""Shim removed HuggingFace transformers.utils helpers for newer transformers.

ADEPT vendors an older LLaMA-Factory that imports symbols no longer exported by
transformers>=5. Keeping one shared env (llm-baselines) requires these shims.
"""

from __future__ import annotations

import transformers.utils as _utils


def _is_torch_sdpa_available() -> bool:
    import torch

    return hasattr(torch.nn.functional, "scaled_dot_product_attention")


def _is_safetensors_available() -> bool:
    try:
        import safetensors  # noqa: F401

        return True
    except ImportError:
        return False


def _is_torch_fx_available() -> bool:
    try:
        import torch.fx  # noqa: F401

        return True
    except ImportError:
        return False


_SHIMS = {
    "is_torch_sdpa_available": _is_torch_sdpa_available,
    "is_safetensors_available": _is_safetensors_available,
    "is_torch_fx_available": _is_torch_fx_available,
}

for _name, _fn in _SHIMS.items():
    if not hasattr(_utils, _name):
        setattr(_utils, _name, _fn)

# Also expose on transformers.utils.import_utils when present.
try:
    import transformers.utils.import_utils as _import_utils

    for _name, _fn in _SHIMS.items():
        if not hasattr(_import_utils, _name):
            setattr(_import_utils, _name, _fn)
except Exception:
    pass
