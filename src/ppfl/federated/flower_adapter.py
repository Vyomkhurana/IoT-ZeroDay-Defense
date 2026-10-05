"""Optional Flower (flwr) integration.

The research results in this project come from the native simulation engine
(:mod:`ppfl.federated.server`), because secure aggregation needs exact control over
which messages the server observes. This adapter runs the *same* :class:`IoTClient`
(local autoencoder training, FedProx term, DP-SGD) under Flower's FedAvg / FedProx
strategies over real gRPC connections on localhost, demonstrating that the client
logic is framework-agnostic and deployable.

Scope: FedAvg / FedProx + client-side DP-SGD. Secure aggregation is *not* wired into
the Flower path (Flower provides its own SecAgg+ implementation through client mods
in its Message API); use the native engine for SecAgg experiments.

Requires ``pip install "flwr>=1.10"``. Flower's Ray-based simulation backend is not
needed: the server and clients communicate over gRPC in one process (client threads).
"""

from __future__ import annotations

import threading
import time
from typing import Any

import numpy as np
import torch

from ppfl.data.federated_dataset import FederatedDataset, build_federated_dataset
from ppfl.detection.anomaly_score import reconstruction_errors
from ppfl.evaluation.evaluator import evaluate_detector
from ppfl.experiments.common import EvalContext, init_model
from ppfl.federated.client import IoTClient
from ppfl.utils.config import Config
from ppfl.utils.environment import resolve_device
from ppfl.utils.logging import get_logger
from ppfl.utils.serialization import TensorSpec, flatten_state_dict

LOGGER = get_logger(__name__)

try:  # pragma: no cover - exercised only when flwr is installed
    import flwr
    from flwr.client import NumPyClient
    from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
    from flwr.server import ServerConfig, start_server
    from flwr.server.strategy import FedAvg as FlowerFedAvg
    from flwr.server.strategy import FedProx as FlowerFedProx

    FLOWER_AVAILABLE = True
except ImportError:  # pragma: no cover
    FLOWER_AVAILABLE = False
    NumPyClient = object  # type: ignore[assignment,misc]


def _require_flower() -> None:
    if not FLOWER_AVAILABLE:
        raise ImportError('The Flower adapter needs flwr: pip install "flwr>=1.10"')


def vector_to_ndarrays(vector: np.ndarray, specs: list[TensorSpec]) -> list[np.ndarray]:
    out, offset = [], 0
    for s in specs:
        out.append(vector[offset : offset + s.numel].reshape(s.shape).astype(np.float32))
        offset += s.numel
    return out


def ndarrays_to_vector(arrays: list[np.ndarray]) -> np.ndarray:
    return np.concatenate([np.asarray(a, dtype=np.float64).reshape(-1) for a in arrays])


class FlowerIoTClient(NumPyClient):  # type: ignore[misc,valid-type]
    """Exposes an :class:`IoTClient` through Flower's ``NumPyClient`` interface."""

    def __init__(self, client: IoTClient, specs: list[TensorSpec], initial_vector: np.ndarray) -> None:
        self.client = client
        self.specs = specs
        self._latest = initial_vector

    def get_parameters(self, config: dict[str, Any]) -> list[np.ndarray]:
        return vector_to_ndarrays(self._latest, self.specs)

    def fit(self, parameters: list[np.ndarray], config: dict[str, Any]) -> tuple[list[np.ndarray], int, dict[str, Any]]:
        global_vector = ndarrays_to_vector(parameters)
        stats = self.client.fit(global_vector, self.specs, int(config.get("round", 0)), {"proximal_mu": float(config.get("proximal_mu", 0.0))})
        # The outbox holds [n * delta, n, n * loss]; Flower expects plain local weights + n.
        contribution = self.client._outbox
        assert contribution is not None
        self._latest = global_vector + contribution[:-2] / stats.num_samples
        metrics = {"train_loss": float(stats.train_loss), "compute_time_s": float(stats.compute_time_s)}
        if stats.epsilon is not None:
            metrics["epsilon"] = float(stats.epsilon)
        return vector_to_ndarrays(self._latest, self.specs), stats.num_samples, metrics

    def evaluate(self, parameters: list[np.ndarray], config: dict[str, Any]) -> tuple[float, int, dict[str, Any]]:
        model = self.client._model_from(ndarrays_to_vector(parameters), self.specs)
        errors = reconstruction_errors(model, self.client._val_benign, device=self.client.device)
        return float(errors.mean()) if len(errors) else 0.0, int(len(errors)), {}


def _weighted(metrics: list[tuple[int, dict[str, Any]]]) -> dict[str, Any]:
    total = sum(n for n, _ in metrics) or 1
    keys = set().union(*(m.keys() for _, m in metrics)) if metrics else set()
    out = {k: sum(n * m.get(k, 0.0) for n, m in metrics) / total for k in keys if k != "epsilon"}
    eps = [m["epsilon"] for _, m in metrics if "epsilon" in m]
    if eps:
        out["epsilon_max"] = max(eps)
    return out


def build_flower_strategy(cfg: Config, initial_vector: np.ndarray, specs: list[TensorSpec], num_clients: int, store: dict[str, Any]):
    """FedAvg / FedProx Flower strategy that also records the latest global weights in ``store``."""
    _require_flower()
    f = cfg.federated
    base = FlowerFedProx if f.strategy == "fedprox" else FlowerFedAvg

    class RecordingStrategy(base):  # type: ignore[misc,valid-type]
        def aggregate_fit(self, server_round, results, failures):  # noqa: ANN001
            params, metrics = super().aggregate_fit(server_round, results, failures)
            if params is not None:
                store["vector"] = ndarrays_to_vector(parameters_to_ndarrays(params))
                store.setdefault("history", []).append({"round": server_round, **metrics})
                LOGGER.info("flower round %d | %s", server_round, {k: round(v, 5) for k, v in metrics.items()})
            return params, metrics

    kwargs: dict[str, Any] = {
        "fraction_fit": f.fraction_fit,
        "fraction_evaluate": 1.0,
        "min_fit_clients": max(1, min(f.min_fit_clients, num_clients)),
        "min_evaluate_clients": num_clients,
        "min_available_clients": num_clients,
        "initial_parameters": ndarrays_to_parameters(vector_to_ndarrays(initial_vector, specs)),
        "on_fit_config_fn": lambda r: {"round": r, "proximal_mu": f.mu if f.strategy == "fedprox" else 0.0},
        "fit_metrics_aggregation_fn": _weighted,
        "evaluate_metrics_aggregation_fn": _weighted,
    }
    if f.strategy == "fedprox":
        kwargs["proximal_mu"] = f.mu
    return RecordingStrategy(**kwargs)


def run_flower(cfg: Config, address: str = "127.0.0.1:8089", dataset: FederatedDataset | None = None) -> dict[str, Any]:
    """Run FedAvg/FedProx with Flower over localhost gRPC and evaluate the final model."""
    _require_flower()
    if cfg.secure_aggregation.enabled:
        raise ValueError("Secure aggregation is only implemented in the native engine (scripts/train.py)")
    device = resolve_device(cfg.experiment.device)
    dataset = dataset or build_federated_dataset(cfg)
    model = init_model(cfg, dataset.input_dim)
    vector, specs = flatten_state_dict(model.state_dict())
    clients = [IoTClient(c, cfg, dataset.input_dim, device) for c in dataset.clients]
    store: dict[str, Any] = {"vector": vector}
    strategy = build_flower_strategy(cfg, vector, specs, len(clients), store)

    def client_thread(c: IoTClient) -> None:
        flwr.client.start_client(server_address=address, client=FlowerIoTClient(c, specs, vector).to_client(), insecure=True)

    threads = [threading.Thread(target=client_thread, args=(c,), daemon=True) for c in clients]
    start = time.perf_counter()

    def launch_clients() -> None:
        time.sleep(2.0)  # let the gRPC server bind first
        for t in threads:
            t.start()

    threading.Thread(target=launch_clients, daemon=True).start()
    start_server(server_address=address, config=ServerConfig(num_rounds=cfg.federated.rounds), strategy=strategy)
    for t in threads:
        t.join(timeout=30)
    elapsed = time.perf_counter() - start

    final = init_model(cfg, dataset.input_dim)
    final.load_state_dict({k: torch.as_tensor(v) for k, v in zip(model.state_dict(), vector_to_ndarrays(store["vector"], specs))})
    final.to(device)
    ctx = EvalContext(cfg, dataset, device)
    threshold = ctx.histogram_threshold(final)
    ev = evaluate_detector(ctx.scorer(final), dataset, threshold.value, per_client=False)
    return {
        "framework": f"flwr {flwr.__version__}",
        "strategy": cfg.federated.strategy,
        "rounds": cfg.federated.rounds,
        "history": store.get("history", []),
        "summary": ev.summary,
        "threshold": threshold.value,
        "wall_time_s": elapsed,
    }
