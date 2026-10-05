"""Flower adapter (skipped unless ``flwr`` is installed)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

pytest.importorskip("flwr")

from ppfl.experiments.common import init_model  # noqa: E402
from ppfl.federated.client import IoTClient  # noqa: E402
from ppfl.federated.flower_adapter import FlowerIoTClient, ndarrays_to_vector, vector_to_ndarrays  # noqa: E402
from ppfl.utils.serialization import flatten_state_dict  # noqa: E402


def test_numpy_client_roundtrip(tiny_cfg, tiny_dataset):
    vector, specs = flatten_state_dict(init_model(tiny_cfg, tiny_dataset.input_dim).state_dict())
    assert np.allclose(ndarrays_to_vector(vector_to_ndarrays(vector, specs)), vector, atol=1e-7)
    client = IoTClient(tiny_dataset.clients[0], tiny_cfg, tiny_dataset.input_dim, torch.device("cpu"))
    fc = FlowerIoTClient(client, specs, vector)
    params = fc.get_parameters({})
    new_params, n, metrics = fc.fit(params, {"round": 1, "proximal_mu": 0.0})
    assert n == client.num_train_samples and "train_loss" in metrics
    assert [p.shape for p in new_params] == [p.shape for p in params]
    assert not np.allclose(ndarrays_to_vector(new_params), vector)
    loss, n_val, _ = fc.evaluate(new_params, {})
    assert loss > 0 and n_val > 0
