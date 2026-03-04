import abc
from typing import Any

import torch
from torch import nn


class ActionExpert(nn.Module, abc.ABC):
    @abc.abstractmethod
    def encode_prefix(
        self,
        *,
        model: nn.Module,
        images: list[torch.Tensor],
        img_masks: list[torch.Tensor],
        lang_tokens: torch.Tensor,
        lang_masks: torch.Tensor,
    ) -> dict[str, Any]: ...

    @abc.abstractmethod
    def compute_velocity_train(
        self,
        *,
        model: nn.Module,
        images: list[torch.Tensor],
        img_masks: list[torch.Tensor],
        lang_tokens: torch.Tensor,
        lang_masks: torch.Tensor,
        state: torch.Tensor,
        x_t: torch.Tensor,
        time: torch.Tensor,
    ) -> torch.Tensor: ...

    @abc.abstractmethod
    def compute_velocity_infer(
        self,
        *,
        model: nn.Module,
        prefix_ctx: dict[str, Any],
        state: torch.Tensor,
        x_t: torch.Tensor,
        time: torch.Tensor,
    ) -> torch.Tensor: ...
