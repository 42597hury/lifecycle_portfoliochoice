"""Throwaway config file for verify_arbitrage.py — tests the
   y_1-bump + drop-ret-to-K=3 budget-conserving variant.

Usage:
    python verify_arbitrage.py scripts/scratch/_arbitrage_check_y1_lobatto.py
"""
from configs._canonical_jax import BASE_CONFIG, CANONICAL_DISC

base_config = BASE_CONFIG

disc_config = CANONICAL_DISC._replace(
    n_state_quad_nodes=(3, 3, 3, 5),
    n_ret_nodes_1d=(3, 3),
    state_lobatto_Z=(None, None, None, 2.93),
    ret_lobatto_Z=None,
)
