"""FedAvg / FedProx aggregation, client/server behaviour and plain-vs-secure equivalence."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ppfl.experiments.common import init_model
from ppfl.federated.client import IoTClient
from ppfl.federated.fedavg import FedAvg, build_contribution, decode_aggregate, weighted_average
from ppfl.federated.fedprox import FedProx, ProximalTerm, build_strategy
from ppfl.federated.server import FederatedServer
from ppfl.models.autoencoder import Autoencoder
from ppfl.training.trainer import train_autoencoder
from ppfl.utils.config import TrainingConfig
from ppfl.utils.serialization import flatten_state_dict, unflatten_state_dict

from conftest import tiny_config


def test_fedavg_aggregation_is_sample_weighted_average():
    rng = np.random.default_rng(0)
    global_vec = rng.normal(size=50)
    locals_ = [global_vec + rng.normal(size=50) for _ in range(3)]
    counts, losses = [10, 30, 60], [0.5, 0.2, 0.1]
    summed = np.sum([build_contribution(w - global_vec, n, l) for w, n, l in zip(locals_, counts, losses)], axis=0)
    agg = decode_aggregate(summed)
    new = FedAvg().apply(global_vec, agg)
    assert np.allclose(new, weighted_average(locals_, counts))
    assert agg.total_weight == 100
    assert agg.mean_loss == pytest.approx((5 + 6 + 6) / 100)


def test_server_learning_rate_scales_update():
    agg = decode_aggregate(build_contribution(np.ones(4), 5, 0.0))
    assert np.allclose(FedAvg(server_learning_rate=0.5).apply(np.zeros(4), agg), 0.5)


def test_decode_rejects_empty_aggregate():
    with pytest.raises(ValueError):
        decode_aggregate(np.zeros(5))


def test_state_dict_roundtrip():
    model = Autoencoder(9, [7], 3)
    vec, specs = flatten_state_dict(model.state_dict())
    restored = unflatten_state_dict(vec, specs)
    for k, v in model.state_dict().items():
        assert torch.equal(v, restored[k])


def test_proximal_term_value_and_gradient():
    model = Autoencoder(4, [3], 2)
    prox = ProximalTerm.from_model(model, mu=0.5)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)  # every parameter moved by +1
    params = list(model.parameters())
    n = sum(p.numel() for p in params)
    assert float(prox.penalty(params)) == pytest.approx(0.25 * n)
    for p in params:
        p.grad = None
    prox.penalty(params).backward()
    autograd = [p.grad.clone() for p in params]
    for p in params:
        p.grad = torch.zeros_like(p)
    prox.add_gradient_(params)
    for a, p in zip(autograd, params):
        assert torch.allclose(a, p.grad) and torch.allclose(p.grad, torch.full_like(p, 0.5))


def test_fedprox_keeps_local_model_closer_to_global():
    rng = np.random.default_rng(0)
    X = rng.normal(loc=3.0, size=(800, 6)).astype(np.float32)
    distances = {}
    for mu in (0.0, 5.0):
        torch.manual_seed(0)
        model = Autoencoder(6, [5], 2)
        start, _ = flatten_state_dict(model.state_dict())
        train_autoencoder(model, X, epochs=3, cfg=TrainingConfig(batch_size=32, learning_rate=1e-2),
                          device=torch.device("cpu"), generator=torch.Generator().manual_seed(0), proximal_mu=mu)
        end, _ = flatten_state_dict(model.state_dict())
        distances[mu] = np.linalg.norm(end - start)
    assert distances[5.0] < 0.7 * distances[0.0]


def test_strategy_factory_and_instructions():
    assert isinstance(build_strategy("fedavg", 0.1, 1.0), FedAvg)
    prox = build_strategy("fedprox", 0.1, 1.0)
    assert isinstance(prox, FedProx) and prox.client_instructions()["proximal_mu"] == 0.1
    assert FedAvg().client_instructions()["proximal_mu"] == 0.0
    with pytest.raises(ValueError):
        build_strategy("fedsgd", 0.0, 1.0)


def _run_rounds(cfg, dataset, rounds=2):
    clients = [IoTClient(c, cfg, dataset.input_dim, torch.device("cpu")) for c in dataset.clients]
    server = FederatedServer(init_model(cfg, dataset.input_dim), build_strategy(cfg.federated.strategy, cfg.federated.mu, 1.0), cfg)
    summaries = [server.run_round(r, clients) for r in range(1, rounds + 1)]
    return server, clients, summaries


def test_server_round_equals_weighted_average_of_client_models(tiny_cfg, tiny_dataset):
    cfg = tiny_cfg.copy()
    clients = [IoTClient(c, cfg, tiny_dataset.input_dim, torch.device("cpu")) for c in tiny_dataset.clients]
    server = FederatedServer(init_model(cfg, tiny_dataset.input_dim), FedAvg(), cfg)
    g0 = server.global_vector.copy()
    summary = server.run_round(1, clients)
    # Re-run each client's local training independently and average by hand.
    locals_, counts = [], []
    for c in clients:
        model = c._model_from(g0, server.specs)
        from ppfl.utils.seed import torch_generator

        train_autoencoder(model, c._train_X, epochs=cfg.federated.local_epochs, cfg=cfg.training, device=torch.device("cpu"),
                          generator=torch_generator(cfg.experiment.seed, "client-shuffle", c.client_id, 1))
        locals_.append(flatten_state_dict(model.state_dict())[0])
        counts.append(c.num_train_samples)
    assert np.allclose(server.global_vector, weighted_average(locals_, counts), atol=1e-6)
    assert summary.participants == [0, 1, 2] and summary.comm.upload_bytes > 0 and summary.comm.download_bytes > 0


def test_secure_aggregation_matches_plain_aggregation(tmp_path, tiny_cfg, tiny_dataset):
    plain_cfg = tiny_cfg.copy()
    secure_cfg = tiny_cfg.copy()
    secure_cfg.secure_aggregation.enabled = True
    secure_cfg.secure_aggregation.deterministic = True
    plain, _, _ = _run_rounds(plain_cfg, tiny_dataset)
    secure, _, sums = _run_rounds(secure_cfg, tiny_dataset)
    # Only fixed-point quantisation (2^-24 per value) separates the two.
    assert np.allclose(plain.global_vector, secure.global_vector, atol=1e-5)
    assert sums[0].comm.secagg_overhead_bytes > 0


def test_unmasked_updates_are_never_released_with_secagg(tiny_cfg, tiny_dataset):
    cfg = tiny_cfg.copy()
    cfg.secure_aggregation.enabled = True
    client = IoTClient(tiny_dataset.clients[0], cfg, tiny_dataset.input_dim, torch.device("cpu"))
    server = FederatedServer(init_model(cfg, tiny_dataset.input_dim), FedAvg(), cfg)
    client.fit(server.global_vector, server.specs, 1, {"proximal_mu": 0.0})
    with pytest.raises(PermissionError):
        client.plain_message()


def test_secure_round_with_dropout_uses_survivors_only(tmp_path, tiny_dataset):
    cfg = tiny_config(tmp_path, "partition.num_clients=3", "secure_aggregation.enabled=true",
                      "secure_aggregation.dropout_rate=0.9", "secure_aggregation.threshold_fraction=0.5",
                      "secure_aggregation.deterministic=true")
    server, clients, sums = _run_rounds(cfg, tiny_dataset, rounds=1)
    s = sums[0]
    assert len(s.dropped) == 1  # at most n - t = 3 - 2 clients may drop
    survivors = [c for c in clients if c.client_id not in s.dropped]
    assert s.aggregate.total_weight == sum(c.num_train_samples for c in survivors)


def test_client_selection_fraction(tiny_cfg, tiny_dataset):
    cfg = tiny_cfg.copy()
    cfg.federated.fraction_fit = 0.5
    cfg.federated.min_fit_clients = 2
    clients = [IoTClient(c, cfg, tiny_dataset.input_dim, torch.device("cpu")) for c in tiny_dataset.clients]
    server = FederatedServer(init_model(cfg, tiny_dataset.input_dim), FedAvg(), cfg)
    sel = server.select_clients(clients, 1)
    assert len(sel) == 2 and sel == server.select_clients(clients, 1)  # deterministic per round
