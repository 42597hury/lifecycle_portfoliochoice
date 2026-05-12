"""Bootstrap inference for the lifecycle VAR system.

End-to-end stationary block bootstrap of the annual primitive panel,
propagating uncertainty through inflation AR(1), proxy real yields,
the 2-var construction VAR, EH yield, lambda-loaded returns, and the
final lifecycle VAR. Paired draws across lambda.
"""
from __future__ import annotations

from .donor_panel import DonorPanel, build_donor_panel
from .stationary_block import stationary_block_indices

__all__ = ["DonorPanel", "build_donor_panel", "stationary_block_indices"]
