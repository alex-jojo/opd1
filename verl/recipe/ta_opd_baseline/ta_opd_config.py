"""Configuration extensions for the isolated TA-OPD baseline recipe."""

from dataclasses import dataclass, field
from typing import Any

from verl.workers.config import PolicyLossConfig


@dataclass
class TAOPDPolicyLossConfig(PolicyLossConfig):
    """Policy loss configuration extended with TA-OPD-specific options."""

    ta_opd: dict[str, Any] = field(default_factory=dict)
