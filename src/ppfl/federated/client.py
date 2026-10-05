"""Simulated IoT gateway participating in federated training.

An :class:`IoTClient` owns its local traffic and never exposes it. Per round it

1. receives the global model and strategy instructions (e.g. FedProx ``mu``),
2. trains the autoencoder locally on its benign traffic (DP-SGD if enabled),
3. prepares its contribution vector ``[n * delta, n, n * loss]`` in a private outbox,
4. releases that vector either in the clear (plain FL) or only in masked form through
   a secure-aggregation endpoint (secure FL).

For threshold calibration it releases only bin counts of its validation anomaly
scores (optionally Laplace-noised), again either in the clear or masked.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from ppfl.data.federated_dataset import ClientData
from ppfl.detection.anomaly_score import reconstruction_errors
from ppfl.detection.threshold import ThresholdSelector
from ppfl.federated.fedavg import build_contribution
from ppfl.models.autoencoder import Autoencoder
from ppfl.privacy.differential_privacy import DPSGDEngine, laplace_mechanism
from ppfl.security.secure_aggregation import SecAggClient, SecureAggregator
from ppfl.training.trainer import train_autoencoder
from ppfl.utils.config import Config
from ppfl.utils.seed import numpy_rng, torch_generator
from ppfl.utils.serialization import TensorSpec, flatten_state_dict, unflatten_state_dict


@dataclass
class ClientRoundStats:
    """Experimenter-side measurements of one local round (not sent to the server)."""

    client_id: int
    round: int
    num_samples: int
    train_loss: float
    epoch_losses: list[float] = field(default_factory=list)
    compute_time_s: float = 0.0
    steps: int = 0
    epsilon: float | None = None


class IoTClient:
    def __init__(self, data: ClientData, cfg: Config, input_dim: int, device: torch.device) -> None:
        self.client_id = data.client_id
        self.cfg = cfg
        self.input_dim = input_dim
        self.device = device
        self._data = data
        benign = cfg.data.benign_label
        self._train_X = data.training_matrix(
            benign, cfg.data.train_contamination, numpy_rng(cfg.experiment.seed, "contamination", self.client_id)
        )
        self._val_benign = data.val.X[data.val.labels == benign]
        self._val_attack = data.val.X[data.val.labels != benign]  # known attacks only (unseen are never in val)
        self._outbox: np.ndarray | None = None
        self.history: list[ClientRoundStats] = []
        self.dp_engine: DPSGDEngine | None = None
        if cfg.privacy.enabled:
            f = cfg.federated
            steps_per_round = f.local_epochs * int(np.ceil(len(self._train_X) / cfg.training.batch_size))
            self.dp_engine = DPSGDEngine(
                cfg.privacy,
                num_samples=len(self._train_X),
                batch_size=cfg.training.batch_size,
                planned_steps=f.rounds * steps_per_round,
                seed=cfg.experiment.seed,
                client_id=self.client_id,
            )

    # ------------------------------------------------------------------ properties
    @property
    def num_train_samples(self) -> int:
        return int(len(self._train_X))

    @property
    def data(self) -> ClientData:
        """Local data access for *experimenter-side* evaluation only."""
        return self._data

    def epsilon(self) -> float | None:
        return self.dp_engine.epsilon() if self.dp_engine else None

    # ------------------------------------------------------------------ model utils
    def _model_from(self, global_vector: np.ndarray, specs: list[TensorSpec]) -> Autoencoder:
        model = Autoencoder.from_config(self.input_dim, self.cfg.model)
        model.load_state_dict(unflatten_state_dict(global_vector, specs))
        return model.to(self.device)

    # ------------------------------------------------------------------ training
    def fit(self, global_vector: np.ndarray, specs: list[TensorSpec], round_idx: int, instructions: dict[str, Any]) -> ClientRoundStats:
        """Local training; the resulting contribution stays in the private outbox."""
        start = time.perf_counter()
        model = self._model_from(global_vector, specs)
        result = train_autoencoder(
            model,
            self._train_X,
            epochs=self.cfg.federated.local_epochs,
            cfg=self.cfg.training,
            device=self.device,
            generator=torch_generator(self.cfg.experiment.seed, "client-shuffle", self.client_id, round_idx),
            proximal_mu=float(instructions.get("proximal_mu", 0.0)),
            dp_engine=self.dp_engine,
        )
        local_vector, _ = flatten_state_dict(model.state_dict())
        self._outbox = build_contribution(local_vector - global_vector, self.num_train_samples, result.final_loss)
        stats = ClientRoundStats(
            client_id=self.client_id,
            round=round_idx,
            num_samples=self.num_train_samples,
            train_loss=result.final_loss,
            epoch_losses=result.epoch_losses,
            compute_time_s=time.perf_counter() - start,
            steps=result.num_steps,
            epsilon=self.epsilon(),
        )
        self.history.append(stats)
        return stats

    # ------------------------------------------------------------------ threshold calibration
    def validation_scores(self, global_vector: np.ndarray, specs: list[TensorSpec]) -> tuple[np.ndarray, np.ndarray]:
        model = self._model_from(global_vector, specs)
        bs = self.cfg.evaluation.batch_size
        return (
            reconstruction_errors(model, self._val_benign, bs, self.device),
            reconstruction_errors(model, self._val_attack, bs, self.device),
        )

    def prepare_histograms(
        self, global_vector: np.ndarray, specs: list[TensorSpec], selector: ThresholdSelector, round_idx: int, noise_epsilon: float | None
    ) -> None:
        """Put ``[benign_hist | attack_hist]`` of local validation scores in the outbox."""
        benign, attack = self.validation_scores(global_vector, specs)
        hist = selector.histograms(benign, attack)
        if noise_epsilon is not None:
            hist = laplace_mechanism(hist, noise_epsilon, numpy_rng(self.cfg.experiment.seed, "hist-noise", self.client_id, round_idx))
        self._outbox = hist

    # ------------------------------------------------------------------ release
    def plain_message(self) -> np.ndarray:
        """Release the outbox in the clear (plain FL). Transmitted as float32."""
        if self._outbox is None:
            raise RuntimeError("Nothing to send: call fit() or prepare_histograms() first")
        if self.cfg.secure_aggregation.enabled:
            raise PermissionError("Secure aggregation is enabled: unmasked updates are never released")
        return self._outbox.astype(np.float32)

    def secure_endpoint(self, aggregator: SecureAggregator, num_parties: int, round_id: int, rng: random.Random | None) -> SecAggClient:
        """Release the outbox only through a secure-aggregation endpoint (masked)."""
        if self._outbox is None:
            raise RuntimeError("Nothing to send: call fit() or prepare_histograms() first")
        endpoint = aggregator.make_client(self.client_id, len(self._outbox), num_parties, round_id, rng)
        endpoint.set_input(self._outbox, num_parties)
        return endpoint
