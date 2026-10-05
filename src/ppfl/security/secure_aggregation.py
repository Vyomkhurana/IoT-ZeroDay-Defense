"""Pairwise-masking secure aggregation with dropout recovery (research prototype).

Follows the structure of Bonawitz et al., "Practical Secure Aggregation for
Privacy-Preserving Machine Learning" (CCS 2017), in the honest-but-curious server
setting. The goal: the server learns ``sum_{u in U2} x_u`` for the surviving clients
``U2`` but no individual ``x_u``.

Protocol (one aggregation)
--------------------------
0. **AdvertiseKeys** — every client ``u`` generates two Diffie-Hellman key pairs:
   ``(c_u^SK, c_u^PK)`` for encrypting point-to-point messages and
   ``(s_u^SK, s_u^PK)`` for deriving pairwise masks. The server broadcasts all public keys.
1. **ShareKeys** — ``u`` draws a self-mask seed ``b_u`` and Shamir-shares both ``s_u^SK``
   and ``b_u`` (threshold ``t``) among all clients. Each share is encrypted for its
   recipient (key from ``c``-agreement) and routed through the server, which only sees
   ciphertexts.
2. **MaskedInputCollection** — ``u`` encodes its vector in fixed point modulo ``2^64``
   and sends ``y_u = x_u + PRG(b_u) + sum_{v != u} sign(u, v) * PRG(s_uv)``,
   with ``s_uv = KDF(DH(s_u^SK, s_v^PK))`` and ``sign = +1 if u < v else -1``.
   Pairwise masks cancel in the sum. Clients may drop out before this step.
3. **Unmasking** — for each surviving ``u`` the clients reveal shares of ``b_u``; for
   each dropped ``u`` they reveal shares of ``s_u^SK``. A client never reveals both
   for the same ``u``. With ``>= t`` responses the server removes the self masks of
   survivors and the dangling pairwise masks of dropped clients.

Building blocks
---------------
* Key agreement: finite-field Diffie-Hellman over the RFC 3526 2048-bit MODP group,
  shared secret hashed with SHA-256.
* PRG: SHAKE-256 (an extendable-output function) expanded to ``uint64`` words.
* Share encryption: SHAKE-256 keystream + HMAC-SHA256 (encrypt-then-MAC).
* Secret sharing: Shamir over the Mersenne prime field ``2^521 - 1``.
* Encoding: fixed point with ``fractional_bits`` bits, two's complement modulo ``2^64``.

Limitations — this is a faithful *simulation* for research, **not** production
cryptography: no authenticated public-key infrastructure / signatures (so no
protection against an actively malicious server, which could e.g. lie about
dropouts), no constant-time arithmetic, a single process plays every party, and the
key/seed generation can be made deterministic for tests.
"""

from __future__ import annotations

import hashlib
import hmac
import math
import random
import secrets
import time
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from ppfl.utils.serialization import payload_nbytes

# RFC 3526, group 14 (2048-bit MODP), generator 2.
MODP_2048_PRIME = int(
    "FFFFFFFFFFFFFFFFC90FDAA22168C234C4C6628B80DC1CD129024E088A67CC74020BBEA63B139B22514A08798E3404DD"
    "EF9519B3CD3A431B302B0A6DF25F14374FE1356D6D51C245E485B576625E7EC6F44C42E9A637ED6B0BFF5CB6F406B7ED"
    "EE386BFB5A899FA5AE9F24117C4B1FE649286651ECE45B3DC2007CB8A163BF0598DA48361C55D39A69163FA8FD24CF5F"
    "83655D23DCA3AD961C62F356208552BB9ED529077096966D670C354E4ABC9804F1746C08CA18217C32905E462E36CE3B"
    "E39E772C180E86039B2783A2EC07A28FB5C55DF06F4C52C9DE2BCBF6955817183995497CEA956AE515D2261898FA0510"
    "15728E5A8AACAA68FFFFFFFFFFFFFFFF",
    16,
)
MODP_GENERATOR = 2
SHAMIR_PRIME = 2**521 - 1  # Mersenne prime; larger than every 256-bit secret
_SECRET_BITS = 256
_SHARE_BYTES = (SHAMIR_PRIME.bit_length() + 7) // 8


class SecureAggregationError(RuntimeError):
    """Protocol failure (too few clients, integrity failure, overflow...)."""


# --------------------------------------------------------------------------------------
# Primitives
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class FixedPointEncoder:
    """Real vectors <-> elements of Z_{2^64} (two's complement fixed point)."""

    fractional_bits: int = 24

    @property
    def scale(self) -> float:
        return float(2**self.fractional_bits)

    def max_abs_value(self, num_parties: int) -> float:
        """Largest |x| such that the sum over ``num_parties`` cannot overflow."""
        return (2**63 - 1) / (self.scale * max(num_parties, 1))

    def encode(self, x: np.ndarray, num_parties: int = 1) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if not np.all(np.isfinite(x)):
            raise SecureAggregationError("Cannot encode non-finite values")
        limit = self.max_abs_value(num_parties)
        if x.size and np.max(np.abs(x)) >= limit:
            raise SecureAggregationError(
                f"Value {np.max(np.abs(x)):.3g} exceeds the fixed-point range ±{limit:.3g} for "
                f"{num_parties} parties; lower secure_aggregation.fractional_bits"
            )
        return np.round(x * self.scale).astype(np.int64).view(np.uint64)

    def decode(self, v: np.ndarray) -> np.ndarray:
        return np.asarray(v, dtype=np.uint64).view(np.int64).astype(np.float64) / self.scale


def prg_mask(seed: bytes, length: int, label: bytes = b"") -> np.ndarray:
    """Expand a seed into ``length`` pseudo-random uint64 words with SHAKE-256."""
    if length == 0:
        return np.zeros(0, dtype=np.uint64)
    stream = hashlib.shake_256(b"ppfl-secagg-prg|" + label + b"|" + seed).digest(8 * length)
    return np.frombuffer(stream, dtype="<u8").astype(np.uint64)


def int_to_bytes(value: int, length: int | None = None) -> bytes:
    length = length or max(1, (value.bit_length() + 7) // 8)
    return value.to_bytes(length, "big")


class DiffieHellman:
    """Finite-field DH over the RFC 3526 2048-bit group."""

    @staticmethod
    def keypair(rng: random.Random | None = None) -> tuple[int, int]:
        sk = (rng.getrandbits(_SECRET_BITS) if rng else secrets.randbits(_SECRET_BITS)) | 1
        return sk, pow(MODP_GENERATOR, sk, MODP_2048_PRIME)

    @staticmethod
    def agree(sk: int, peer_pk: int) -> bytes:
        if not 1 < peer_pk < MODP_2048_PRIME - 1:
            raise SecureAggregationError("Invalid Diffie-Hellman public key")
        shared = pow(peer_pk, sk, MODP_2048_PRIME)
        return hashlib.sha256(b"ppfl-dh|" + int_to_bytes(shared, 256)).digest()


class Shamir:
    """t-out-of-n Shamir secret sharing over GF(2^521 - 1)."""

    prime = SHAMIR_PRIME

    @classmethod
    def split(cls, secret: int, xs: Sequence[int], threshold: int, rng: random.Random | None = None) -> dict[int, int]:
        if not 0 <= secret < cls.prime:
            raise ValueError("Secret out of field range")
        if not 1 <= threshold <= len(xs):
            raise ValueError(f"Invalid threshold {threshold} for {len(xs)} shares")
        if any(x % cls.prime == 0 for x in xs):
            raise ValueError("Share x-coordinates must be non-zero")
        draw = (lambda: rng.randrange(cls.prime)) if rng else (lambda: secrets.randbelow(cls.prime))
        coeffs = [secret] + [draw() for _ in range(threshold - 1)]
        shares = {}
        for x in xs:
            acc = 0
            for c in reversed(coeffs):  # Horner
                acc = (acc * x + c) % cls.prime
            shares[x] = acc
        return shares

    @classmethod
    def reconstruct(cls, shares: dict[int, int]) -> int:
        if not shares:
            raise ValueError("No shares supplied")
        p = cls.prime
        secret = 0
        items = list(shares.items())
        for i, (xi, yi) in enumerate(items):
            num, den = 1, 1
            for j, (xj, _) in enumerate(items):
                if i != j:
                    num = (num * -xj) % p
                    den = (den * (xi - xj)) % p
            secret = (secret + yi * num * pow(den, p - 2, p)) % p
        return secret


def _keystream_encrypt(key: bytes, nonce: bytes, plaintext: bytes) -> bytes:
    stream = hashlib.shake_256(b"ppfl-enc|" + key + b"|" + nonce).digest(len(plaintext))
    body = bytes(a ^ b for a, b in zip(plaintext, stream))
    tag = hmac.new(key, nonce + body, hashlib.sha256).digest()
    return body + tag


def _keystream_decrypt(key: bytes, nonce: bytes, ciphertext: bytes) -> bytes:
    body, tag = ciphertext[:-32], ciphertext[-32:]
    if not hmac.compare_digest(tag, hmac.new(key, nonce + body, hashlib.sha256).digest()):
        raise SecureAggregationError("Share ciphertext failed authentication")
    stream = hashlib.shake_256(b"ppfl-enc|" + key + b"|" + nonce).digest(len(body))
    return bytes(a ^ b for a, b in zip(body, stream))


# --------------------------------------------------------------------------------------
# Messages (everything the server ever receives)
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class AdvertisedKeys:
    client_id: int
    channel_public_key: int
    mask_public_key: int


@dataclass(frozen=True)
class EncryptedShare:
    sender: int
    recipient: int
    ciphertext: bytes


@dataclass(frozen=True)
class MaskedInput:
    client_id: int
    vector: np.ndarray  # uint64, looks uniformly random to the server


@dataclass(frozen=True)
class UnmaskingResponse:
    client_id: int
    self_mask_shares: dict[int, int]  # shares of b_u for surviving u
    key_shares: dict[int, int]  # shares of s_u^SK for dropped u


# --------------------------------------------------------------------------------------
# Client
# --------------------------------------------------------------------------------------


class SecAggClient:
    """One participant. Holds its private input; only protocol messages leave it."""

    def __init__(
        self,
        client_id: int,
        vector_length: int,
        threshold: int,
        round_id: int,
        encoder: FixedPointEncoder,
        rng: random.Random | None = None,
    ) -> None:
        self.client_id = client_id
        self.vector_length = vector_length
        self.threshold = threshold
        self.round_id = round_id
        self.encoder = encoder
        self._rng = rng
        self._input: np.ndarray | None = None
        self._c_sk, self._c_pk = DiffieHellman.keypair(rng)
        self._s_sk, self._s_pk = DiffieHellman.keypair(rng)
        self._b = rng.getrandbits(_SECRET_BITS) if rng else secrets.randbits(_SECRET_BITS)
        self._roster: dict[int, AdvertisedKeys] = {}
        self._received: dict[int, tuple[int, int]] = {}  # sender -> (s_sk share, b share)

    def set_input(self, vector: np.ndarray, num_parties: int) -> None:
        if vector.shape != (self.vector_length,):
            raise SecureAggregationError(f"Input length {vector.shape} != {self.vector_length}")
        self._input = self.encoder.encode(vector, num_parties)

    # Round 0
    def advertise_keys(self) -> AdvertisedKeys:
        return AdvertisedKeys(self.client_id, self._c_pk, self._s_pk)

    def _nonce(self, sender: int, recipient: int) -> bytes:
        return f"r{self.round_id}|{sender}->{recipient}".encode()

    # Round 1
    def share_keys(self, roster: dict[int, AdvertisedKeys]) -> list[EncryptedShare]:
        if self.client_id not in roster or len(roster) < self.threshold:
            raise SecureAggregationError("Roster too small or missing this client; aborting")
        self._roster = dict(roster)
        xs = [cid + 1 for cid in sorted(roster)]
        sk_shares = Shamir.split(self._s_sk, xs, self.threshold, self._rng)
        b_shares = Shamir.split(self._b, xs, self.threshold, self._rng)
        out = []
        for cid in sorted(roster):
            plain = int_to_bytes(sk_shares[cid + 1], _SHARE_BYTES) + int_to_bytes(b_shares[cid + 1], _SHARE_BYTES)
            if cid == self.client_id:
                self._received[cid] = (sk_shares[cid + 1], b_shares[cid + 1])
                continue
            key = DiffieHellman.agree(self._c_sk, roster[cid].channel_public_key)
            out.append(EncryptedShare(self.client_id, cid, _keystream_encrypt(key, self._nonce(self.client_id, cid), plain)))
        return out

    def receive_shares(self, packets: list[EncryptedShare]) -> None:
        for pkt in packets:
            if pkt.recipient != self.client_id or pkt.sender not in self._roster:
                raise SecureAggregationError("Share routed to the wrong client")
            key = DiffieHellman.agree(self._c_sk, self._roster[pkt.sender].channel_public_key)
            plain = _keystream_decrypt(key, self._nonce(pkt.sender, self.client_id), pkt.ciphertext)
            self._received[pkt.sender] = (
                int.from_bytes(plain[:_SHARE_BYTES], "big"),
                int.from_bytes(plain[_SHARE_BYTES:], "big"),
            )

    # Round 2
    def masked_input(self) -> MaskedInput:
        if self._input is None:
            raise SecureAggregationError("set_input() must be called before masked_input()")
        y = self._input + prg_mask(int_to_bytes(self._b, 32), self.vector_length, b"self")
        for cid, keys in self._roster.items():
            if cid == self.client_id:
                continue
            mask = prg_mask(DiffieHellman.agree(self._s_sk, keys.mask_public_key), self.vector_length, b"pair")
            y = y + mask if self.client_id < cid else y - mask  # uint64 arithmetic wraps mod 2^64
        return MaskedInput(self.client_id, y)

    # Round 3
    def unmasking_response(self, survivors: Sequence[int], dropped: Sequence[int]) -> UnmaskingResponse:
        survivors, dropped = set(survivors), set(dropped)
        if survivors & dropped:
            raise SecureAggregationError("A client cannot be both surviving and dropped")
        if len(survivors) < self.threshold:
            raise SecureAggregationError("Fewer surviving clients than the threshold; aborting")
        if not (survivors | dropped) <= set(self._roster):
            raise SecureAggregationError("Unknown client ids in unmasking request")
        return UnmaskingResponse(
            self.client_id,
            {u: self._received[u][1] for u in survivors if u in self._received},
            {u: self._received[u][0] for u in dropped if u in self._received},
        )


# --------------------------------------------------------------------------------------
# Server
# --------------------------------------------------------------------------------------


@dataclass
class SecAggStats:
    num_clients: int
    num_dropped: int
    threshold: int
    bytes_up: int = 0
    bytes_down: int = 0
    bytes_by_phase: dict[str, int] = field(default_factory=dict)
    server_time_s: float = 0.0

    def add(self, phase: str, up: int = 0, down: int = 0) -> None:
        self.bytes_up += up
        self.bytes_down += down
        self.bytes_by_phase[phase] = self.bytes_by_phase.get(phase, 0) + up + down


class SecAggServer:
    """Aggregator. ``transcript`` records every message it receives (for auditing/tests)."""

    def __init__(self, vector_length: int, threshold: int, encoder: FixedPointEncoder) -> None:
        self.vector_length = vector_length
        self.threshold = threshold
        self.encoder = encoder
        self.transcript: list[tuple[str, int, object]] = []
        self._roster: dict[int, AdvertisedKeys] = {}

    def collect_keys(self, adverts: list[AdvertisedKeys]) -> dict[int, AdvertisedKeys]:
        self._roster = {a.client_id: a for a in adverts}
        if len(self._roster) < self.threshold:
            raise SecureAggregationError(f"Only {len(self._roster)} clients advertised keys; threshold is {self.threshold}")
        self.transcript += [("keys", a.client_id, a) for a in adverts]
        return dict(self._roster)

    def route_shares(self, packets: list[EncryptedShare]) -> dict[int, list[EncryptedShare]]:
        self.transcript += [("share", p.sender, p) for p in packets]
        routed: dict[int, list[EncryptedShare]] = {cid: [] for cid in self._roster}
        for p in packets:
            routed[p.recipient].append(p)
        return routed

    def unmask(self, masked: list[MaskedInput], responses: list[UnmaskingResponse]) -> np.ndarray:
        """Remove all masks and return the decoded sum of surviving clients' inputs."""
        survivors = sorted(m.client_id for m in masked)
        dropped = sorted(set(self._roster) - set(survivors))
        self.transcript += [("masked_input", m.client_id, m) for m in masked]
        self.transcript += [("unmask", r.client_id, r) for r in responses]
        if len(responses) < self.threshold:
            raise SecureAggregationError(f"Only {len(responses)} unmasking responses; threshold is {self.threshold}")

        total = np.zeros(self.vector_length, dtype=np.uint64)
        for m in masked:
            total = total + m.vector
        for u in survivors:
            shares = {r.client_id + 1: r.self_mask_shares[u] for r in responses if u in r.self_mask_shares}
            b_u = Shamir.reconstruct(dict(list(shares.items())[: self.threshold]))
            total = total - prg_mask(int_to_bytes(b_u, 32), self.vector_length, b"self")
        for u in dropped:
            shares = {r.client_id + 1: r.key_shares[u] for r in responses if u in r.key_shares}
            s_sk_u = Shamir.reconstruct(dict(list(shares.items())[: self.threshold]))
            if pow(MODP_GENERATOR, s_sk_u, MODP_2048_PRIME) != self._roster[u].mask_public_key:
                raise SecureAggregationError(f"Reconstructed key of dropped client {u} is inconsistent")
            for v in survivors:
                mask = prg_mask(DiffieHellman.agree(s_sk_u, self._roster[v].mask_public_key), self.vector_length, b"pair")
                # y_v contained +mask if v < u else -mask; remove it.
                total = total - mask if v < u else total + mask
        return self.encoder.decode(total)


# --------------------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------------------


class SecureAggregator:
    """Runs one full protocol instance over a set of :class:`SecAggClient` endpoints.

    In the simulation every party lives in one process; this class plays the role of
    the network: it carries messages between the clients and the :class:`SecAggServer`
    and measures their serialized size. The server object only ever receives the
    message types defined above.
    """

    def __init__(self, threshold_fraction: float, encoder: FixedPointEncoder) -> None:
        self.threshold_fraction = threshold_fraction
        self.encoder = encoder

    def threshold_for(self, num_clients: int) -> int:
        return max(2, min(num_clients, math.ceil(self.threshold_fraction * num_clients)))

    def make_client(self, client_id: int, vector_length: int, num_clients: int, round_id: int, rng: random.Random | None) -> SecAggClient:
        return SecAggClient(client_id, vector_length, self.threshold_for(num_clients), round_id, self.encoder, rng)

    def aggregate(self, clients: Sequence[SecAggClient], dropouts: Sequence[int] = ()) -> tuple[np.ndarray, SecAggStats, SecAggServer]:
        if len(clients) < 2:
            raise SecureAggregationError("Secure aggregation needs at least two clients")
        start = time.perf_counter()
        n = len(clients)
        length = clients[0].vector_length
        threshold = clients[0].threshold
        stats = SecAggStats(num_clients=n, num_dropped=len(dropouts), threshold=threshold)
        server = SecAggServer(length, threshold, self.encoder)
        by_id = {c.client_id: c for c in clients}

        adverts = [c.advertise_keys() for c in clients]
        roster = server.collect_keys(adverts)
        stats.add("advertise_keys", up=sum(map(payload_nbytes, adverts)), down=n * payload_nbytes(roster))

        packets = [p for c in clients for p in c.share_keys(roster)]
        routed = server.route_shares(packets)
        for cid, inbox in routed.items():
            by_id[cid].receive_shares(inbox)
        size = sum(map(payload_nbytes, packets))
        stats.add("share_keys", up=size, down=size)

        survivors = [c for c in clients if c.client_id not in set(dropouts)]
        if len(survivors) < threshold:
            raise SecureAggregationError(f"{len(survivors)} survivors < threshold {threshold}")
        masked = [c.masked_input() for c in survivors]
        stats.add("masked_input", up=sum(payload_nbytes(m) for m in masked))

        survivor_ids = [c.client_id for c in survivors]
        dropped_ids = sorted(set(by_id) - set(survivor_ids))
        request = (survivor_ids, dropped_ids)
        responses = [c.unmasking_response(*request) for c in survivors]
        stats.add("unmasking", up=sum(map(payload_nbytes, responses)), down=len(survivors) * payload_nbytes(request))

        total = server.unmask(masked, responses)
        stats.server_time_s = time.perf_counter() - start
        return total, stats, server
