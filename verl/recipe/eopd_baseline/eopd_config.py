"""Configuration extensions for the isolated EOPD baseline recipe."""

from dataclasses import dataclass, field
from typing import Any

from verl.workers.config import PolicyLossConfig


@dataclass
class EOPDPolicyLossConfig(PolicyLossConfig):
    """Policy loss configuration extended with EOPD-specific options."""

    eopd: dict[str, Any] = field(default_factory=dict)
