# src/experiments/registry.py
"""Registry for all experiments to avoid circular imports."""

from .bias import BIAS
from .reasoning import REASONING
from .reranker import RERANKER
from .safety import SAFETY

EXPERIMENTS = {
    "bias": BIAS,
    "reasoning": REASONING,
    "reranker": RERANKER,
    "safety": SAFETY
}

