import dataclasses
from typing import Literal


@dataclasses.dataclass(frozen=True)
class IllibMoeVelocityExpertConfig:
    cond_dim: int = 512
    num_experts: int = 4
    top_k: int | None = None
    expert_hidden_dim: int = 512
    expert_hidden_depth: int = 2
    gating_hidden_dim: int = 256
    gating_hidden_depth: int = 1
    activation: Literal["silu", "relu", "gelu"] = "silu"
    gating_temperature: float = 1.0


@dataclasses.dataclass(frozen=True)
class PytorchActionExpertConfig:
    name: Literal["gemma_token", "il_moe_velocity"] = "gemma_token"
    il_moe_velocity: IllibMoeVelocityExpertConfig = dataclasses.field(default_factory=IllibMoeVelocityExpertConfig)

