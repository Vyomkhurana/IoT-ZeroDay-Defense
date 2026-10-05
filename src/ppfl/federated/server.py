"""Federated server: client selection, aggregation (plain or secure) and global model updates.

What the server observes:

* plain FL     — each selected client's contribution vector ``[n * delta, n, n * loss]``;
* secure FL    — public keys, encrypted share ciphertexts, *masked* vectors, and the
                 Shamir shares needed to unmask the **sum** only.

In both cases it never receives raw traffic. With secure aggregation, even the
per-client sample counts and losses are only revealed in aggregate, because they are
part of the masked vector.
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
import torch

from ppfl.detection.threshold import ThresholdResult, ThresholdSelector
from ppfl.federated.client import ClientRoundStats, IoTClient
from ppfl.federated.fedavg import AggregateUpdate, FedAvg, decode_aggregate
from ppfl.models.autoencoder import Autoencoder
from ppfl.security.secure_aggregation import FixedPointEncoder, SecureAggregator
from ppfl.utils.config import Config
from ppfl.utils.logging import get_logger
from ppfl.utils.seed import derive_seed, numpy_rng
from ppfl.utils.serialization import flatten_state_dict, payload_nbytes, unflatten_state_dict

LOGGER = get_logger(__name__)


@dataclass
class CommunicationStats:
    upload_bytes: int = 0
    download_bytes: int = 0
    secagg_overhead_bytes: int = 0
    update_payload_bytes: int = 0  # size of one client's model-update message
    by_phase: dict[str, int] = field(default_factory=dict)

    @property
    def total_bytes(self) -> int:
        return self.upload_bytes + self.download_bytes


@dataclass
class RoundSummary:
    round: int
    participants: list[int]
    dropped: list[int]
    aggregate: AggregateUpdate
    client_stats: list[ClientRoundStats]
    comm: CommunicationStats
    wall_time_s: float


class FederatedServer:
    def __init__(self, model: Autoencoder, strategy: FedAvg, cfg: Config) -> None:
        self.cfg = cfg
        self.strategy = strategy
        self.global_vector, self.specs = flatten_state_dict(model.state_dict())
        sa = cfg.secure_aggregation
        self.secure = sa.enabled
        self.aggregator = SecureAggregator(sa.threshold_fraction, FixedPointEncoder(sa.fractional_bits)) if sa.enabled else None

    # ---------------------------------------------------------------- model access
    def global_state(self) -> dict[str, torch.Tensor]:
        return unflatten_state_dict(self.global_vector, self.specs)

    def global_model(self, input_dim: int) -> Autoencoder:
        model = Autoencoder.from_config(input_dim, self.cfg.model)
        model.load_state_dict(self.global_state())
        return model

    def broadcast_nbytes(self) -> int:
        """Bytes of the global-model message sent to each selected client (float32 weights)."""
        state = {k: v.float().numpy() for k, v in self.global_state().items()}
        return payload_nbytes({"state": state, "instructions": self.strategy.client_instructions()})

    # ---------------------------------------------------------------- selection
    def select_clients(self, clients: Sequence[IoTClient], round_idx: int) -> list[IoTClient]:
        f = self.cfg.federated
        k = max(min(f.min_fit_clients, len(clients)), int(round(f.fraction_fit * len(clients))))
        if k >= len(clients):
            return list(clients)
        idx = numpy_rng(self.cfg.experiment.seed, "selection", round_idx).choice(len(clients), size=k, replace=False)
        return [clients[i] for i in sorted(idx)]

    # ---------------------------------------------------------------- aggregation
    def _secagg_rng(self, purpose: str, round_idx: int, client_id: int) -> random.Random | None:
        if not self.cfg.secure_aggregation.deterministic:
            return None  # OS CSPRNG
        return random.Random(derive_seed(self.cfg.experiment.seed, "secagg", purpose, round_idx, client_id))

    def aggregate_outboxes(
        self, clients: Sequence[IoTClient], round_idx: int, purpose: str, allow_dropout: bool
    ) -> tuple[np.ndarray, list[int], CommunicationStats]:
        """Sum the clients' outbox vectors, securely if configured. Returns (sum, dropped ids, comm)."""
        comm = CommunicationStats()
        if not self.secure:
            messages = [c.plain_message() for c in clients]
            comm.update_payload_bytes = payload_nbytes(messages[0])
            comm.upload_bytes = sum(payload_nbytes(m) for m in messages)
            comm.by_phase["update"] = comm.upload_bytes
            return np.sum([m.astype(np.float64) for m in messages], axis=0), [], comm

        assert self.aggregator is not None
        dropped: list[int] = []
        rate = self.cfg.secure_aggregation.dropout_rate
        if allow_dropout and rate > 0:
            rng = numpy_rng(self.cfg.experiment.seed, "dropout", purpose, round_idx)
            candidates = [c.client_id for c in clients if rng.random() < rate]
            max_drop = len(clients) - self.aggregator.threshold_for(len(clients))
            dropped = candidates[:max(0, max_drop)]
        endpoints = [
            c.secure_endpoint(self.aggregator, len(clients), round_idx, self._secagg_rng(purpose, round_idx, c.client_id))
            for c in clients
        ]
        total, stats, _ = self.aggregator.aggregate(endpoints, dropped)
        comm.update_payload_bytes = stats.bytes_by_phase.get("masked_input", 0) // max(1, len(clients) - len(dropped))
        comm.upload_bytes = stats.bytes_up
        comm.download_bytes = stats.bytes_down
        comm.secagg_overhead_bytes = stats.bytes_up + stats.bytes_down - stats.bytes_by_phase.get("masked_input", 0)
        comm.by_phase = dict(stats.bytes_by_phase)
        return total, dropped, comm

    # ---------------------------------------------------------------- one round
    def run_round(self, round_idx: int, clients: Sequence[IoTClient], on_client_done: Callable[[ClientRoundStats], None] | None = None) -> RoundSummary:
        start = time.perf_counter()
        selected = self.select_clients(clients, round_idx)
        instructions = self.strategy.client_instructions()
        stats = []
        for client in selected:
            s = client.fit(self.global_vector, self.specs, round_idx, instructions)
            stats.append(s)
            if on_client_done is not None:
                on_client_done(s)
        total, dropped, comm = self.aggregate_outboxes(selected, round_idx, "update", allow_dropout=True)
        comm.download_bytes += self.broadcast_nbytes() * len(selected)
        comm.by_phase["broadcast"] = self.broadcast_nbytes() * len(selected)
        aggregate = decode_aggregate(total)
        self.global_vector = self.strategy.apply(self.global_vector, aggregate)
        return RoundSummary(
            round=round_idx,
            participants=[c.client_id for c in selected],
            dropped=dropped,
            aggregate=aggregate,
            client_stats=stats,
            comm=comm,
            wall_time_s=time.perf_counter() - start,
        )

    # ---------------------------------------------------------------- threshold
    def calibrate_threshold(
        self, clients: Sequence[IoTClient], selector: ThresholdSelector, round_idx: int, noise_epsilon: float | None
    ) -> tuple[ThresholdResult, CommunicationStats]:
        """Federated threshold from (securely) aggregated validation-score histograms."""
        for c in clients:
            c.prepare_histograms(self.global_vector, self.specs, selector, round_idx, noise_epsilon)
        total, _, comm = self.aggregate_outboxes(clients, round_idx, "histogram", allow_dropout=False)
        comm.download_bytes += self.broadcast_nbytes() * len(clients)
        result = selector.from_histogram_vector(np.clip(total, 0, None))
        result.details["histogram_dp_epsilon"] = noise_epsilon
        result.details["secure_aggregation"] = self.secure
        return result, comm
