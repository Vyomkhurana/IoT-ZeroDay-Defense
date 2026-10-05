"""Secure aggregation (pairwise masking with dropout recovery; research prototype)."""

from ppfl.security.secure_aggregation import (
    FixedPointEncoder,
    SecAggClient,
    SecAggServer,
    SecAggStats,
    SecureAggregationError,
    SecureAggregator,
    Shamir,
)

__all__ = [
    "FixedPointEncoder",
    "SecAggClient",
    "SecAggServer",
    "SecAggStats",
    "SecureAggregationError",
    "SecureAggregator",
    "Shamir",
]
