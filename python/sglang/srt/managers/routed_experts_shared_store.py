# Copyright 2023-2024 SGLang Team
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
# ==============================================================================
"""Ray shared-store helpers for routed_experts metadata."""

from __future__ import annotations

import logging
from typing import List, Optional

import pybase64
import torch

logger = logging.getLogger(__name__)

_ROUTED_EXPERTS_SHARED_STORE = None
_ROUTED_EXPERTS_SHARED_STORE_NAME = "shared_store"
_ROUTED_EXPERTS_SHARED_STORE_NAMESPACE = "sglang"


class _RoutedExpertsSharedStore:
    def __init__(self):
        self._data = {}

    def put(self, data):
        import ray

        ref = ray.put(data)
        key = ref.hex()
        self._data[key] = ref
        return key

    def get(self, key):
        import ray

        ref = self._data.pop(key)
        return ray.get(ref)

    def clear(self):
        import ray

        all_data = list(self._data.values())
        if len(all_data) > 0:
            ray.internal.free(all_data, local_only=False)
        self._data.clear()


def _lazy_get_routed_experts_shared_store():
    global _ROUTED_EXPERTS_SHARED_STORE
    if _ROUTED_EXPERTS_SHARED_STORE is not None:
        return _ROUTED_EXPERTS_SHARED_STORE

    import ray

    if not ray.is_initialized():
        ray.init(address="auto", ignore_reinit_error=True)

    try:
        _ROUTED_EXPERTS_SHARED_STORE = ray.get_actor(
            _ROUTED_EXPERTS_SHARED_STORE_NAME,
            namespace=_ROUTED_EXPERTS_SHARED_STORE_NAMESPACE,
        )
    except ValueError:
        try:
            actor_cls = ray.remote(num_cpus=0)(_RoutedExpertsSharedStore)
            _ROUTED_EXPERTS_SHARED_STORE = actor_cls.options(
                name=_ROUTED_EXPERTS_SHARED_STORE_NAME,
                namespace=_ROUTED_EXPERTS_SHARED_STORE_NAMESPACE,
                lifetime="detached",
            ).remote()
        except ray.exceptions.ActorAlreadyExistsError:
            _ROUTED_EXPERTS_SHARED_STORE = ray.get_actor(
                _ROUTED_EXPERTS_SHARED_STORE_NAME,
                namespace=_ROUTED_EXPERTS_SHARED_STORE_NAMESPACE,
            )
    return _ROUTED_EXPERTS_SHARED_STORE


def encode_routed_experts(routed_experts_tensor: torch.Tensor) -> str:
    try:
        import ray

        store = _lazy_get_routed_experts_shared_store()
        return ray.get(store.put.remote(routed_experts_tensor.numpy()))
    except Exception:
        logger.exception(
            "Failed to put routed_experts into Ray shared store; falling back to base64."
        )
        return pybase64.b64encode(routed_experts_tensor.numpy().tobytes()).decode(
            "utf-8"
        )


def encode_routed_experts_per_request(
    data_list: Optional[List[Optional[torch.Tensor]]],
) -> Optional[List[Optional[str]]]:
    if data_list is None:
        return None
    return [
        encode_routed_experts(item) if item is not None else None for item in data_list
    ]
