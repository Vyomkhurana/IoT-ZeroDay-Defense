"""Secure aggregation: mask cancellation, dropout recovery, what the server sees."""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from ppfl.models.autoencoder import Autoencoder
from ppfl.security.secure_aggregation import (
    DiffieHellman,
    EncryptedShare,
    FixedPointEncoder,
    SecureAggregationError,
    SecureAggregator,
    Shamir,
    prg_mask,
)
from ppfl.utils.serialization import flatten_state_dict, unflatten_state_dict

ENC = FixedPointEncoder(24)


def _aggregate(vectors, dropouts=(), threshold_fraction=0.6, deterministic=True):
    agg = SecureAggregator(threshold_fraction, ENC)
    n = len(vectors)
    clients = [agg.make_client(i, len(vectors[0]), n, 7, random.Random(i) if deterministic else None) for i in range(n)]
    for c, v in zip(clients, vectors):
        c.set_input(v, n)
    return agg.aggregate(clients, dropouts)


def test_masks_cancel_three_clients():
    rng = np.random.default_rng(0)
    U1, U2, U3 = (rng.normal(size=1000) for _ in range(3))
    total, stats, _ = _aggregate([U1, U2, U3])
    assert np.allclose(total, U1 + U2 + U3, atol=3 * 2**-24)
    assert stats.num_dropped == 0 and stats.bytes_up > 0


@pytest.mark.parametrize("dropouts", [(1,), (0, 4), (5, 2)])
def test_dropout_recovery(dropouts):
    rng = np.random.default_rng(1)
    vectors = [rng.normal(scale=10, size=300) for _ in range(6)]
    total, stats, _ = _aggregate(vectors, dropouts, threshold_fraction=0.5)
    expected = sum(v for i, v in enumerate(vectors) if i not in dropouts)
    assert np.allclose(total, expected, atol=1e-5)
    assert stats.num_dropped == len(dropouts)


def test_too_many_dropouts_abort():
    vectors = [np.ones(10)] * 4
    with pytest.raises(SecureAggregationError):
        _aggregate(vectors, dropouts=(0, 1, 2), threshold_fraction=0.75)


def test_server_only_sees_masked_inputs():
    rng = np.random.default_rng(2)
    vectors = [rng.normal(size=500) for _ in range(4)]
    _, _, server = _aggregate(vectors)
    masked = [msg for kind, _, msg in server.transcript if kind == "masked_input"]
    assert len(masked) == 4
    for m in masked:
        decoded = ENC.decode(m.vector)
        assert not np.allclose(decoded, vectors[m.client_id], atol=1.0)
        # masked words are ~uniform over 2^64: decoded magnitudes are astronomically large
        assert np.median(np.abs(decoded)) > 1e9
    kinds = {kind for kind, _, _ in server.transcript}
    assert kinds == {"keys", "share", "masked_input", "unmask"}
    shares = [msg for kind, _, msg in server.transcript if kind == "share"]
    assert all(isinstance(s, EncryptedShare) for s in shares)


def test_unmasking_never_reveals_both_secrets_of_a_client():
    vectors = [np.ones(20)] * 5
    _, _, server = _aggregate(vectors, dropouts=(3,), threshold_fraction=0.6)
    for kind, _, resp in server.transcript:
        if kind == "unmask":
            assert not set(resp.self_mask_shares) & set(resp.key_shares)
            assert 3 in resp.key_shares and 3 not in resp.self_mask_shares


def test_state_dict_aggregation():
    models = []
    for seed in range(3):
        torch.manual_seed(seed)
        models.append(Autoencoder(10, [8], 3))
    vecs = [flatten_state_dict(m.state_dict())[0] for m in models]
    specs = flatten_state_dict(models[0].state_dict())[1]
    total, _, _ = _aggregate(vecs)
    mean_state = unflatten_state_dict(total / 3, specs)
    for name, tensor in mean_state.items():
        manual = sum(m.state_dict()[name].double() for m in models) / 3
        assert torch.allclose(tensor.double(), manual, atol=1e-6)


def test_fixed_point_encoding_roundtrip_and_overflow():
    x = np.array([-3.5, 0.0, 1e-7, 123456.789])
    assert np.allclose(ENC.decode(ENC.encode(x)), x, atol=2**-24)
    # modular sums of encodings decode to sums (negative values via two's complement)
    assert np.allclose(ENC.decode(ENC.encode(x) + ENC.encode(-2 * x)), -x, atol=1e-6)
    with pytest.raises(SecureAggregationError):
        ENC.encode(np.array([1e12]), num_parties=10)
    with pytest.raises(SecureAggregationError):
        ENC.encode(np.array([np.nan]))


def test_shamir_threshold_reconstruction():
    rng = random.Random(0)
    secret = rng.getrandbits(256)
    shares = Shamir.split(secret, xs=list(range(1, 8)), threshold=4, rng=rng)
    for subset in ([1, 2, 3, 4], [7, 5, 3, 2], [1, 3, 5, 7, 6]):
        assert Shamir.reconstruct({x: shares[x] for x in subset}) == secret
    assert Shamir.reconstruct({x: shares[x] for x in [1, 2, 3]}) != secret  # below threshold
    with pytest.raises(ValueError):
        Shamir.split(secret, xs=[0, 1, 2], threshold=2)


def test_diffie_hellman_agreement_and_prg_determinism():
    sk_a, pk_a = DiffieHellman.keypair(random.Random(1))
    sk_b, pk_b = DiffieHellman.keypair(random.Random(2))
    assert DiffieHellman.agree(sk_a, pk_b) == DiffieHellman.agree(sk_b, pk_a)
    seed = DiffieHellman.agree(sk_a, pk_b)
    assert np.array_equal(prg_mask(seed, 100), prg_mask(seed, 100))
    assert not np.array_equal(prg_mask(seed, 100, b"x"), prg_mask(seed, 100, b"y"))
    with pytest.raises(SecureAggregationError):
        DiffieHellman.agree(sk_a, 1)


def test_tampered_share_is_detected():
    agg = SecureAggregator(0.6, ENC)
    clients = [agg.make_client(i, 5, 3, 0, random.Random(i)) for i in range(3)]
    roster = {c.client_id: c.advertise_keys() for c in clients}
    packets = clients[0].share_keys(roster)
    for c in clients[1:]:
        c.share_keys(roster)
    bad = packets[0]
    tampered = EncryptedShare(bad.sender, bad.recipient, bytes([bad.ciphertext[0] ^ 1]) + bad.ciphertext[1:])
    with pytest.raises(SecureAggregationError, match="authentication"):
        clients[bad.recipient].receive_shares([tampered])


def test_deterministic_mode_reproducible_and_random_mode_correct():
    rng = np.random.default_rng(3)
    vectors = [rng.normal(size=50) for _ in range(3)]
    _, _, s1 = _aggregate(vectors, deterministic=True)
    _, _, s2 = _aggregate(vectors, deterministic=True)
    m1 = [m.vector for k, _, m in s1.transcript if k == "masked_input"]
    m2 = [m.vector for k, _, m in s2.transcript if k == "masked_input"]
    assert all(np.array_equal(a, b) for a, b in zip(m1, m2))
    total, _, _ = _aggregate(vectors, deterministic=False)  # OS CSPRNG keys
    assert np.allclose(total, sum(vectors), atol=1e-6)
