from __future__ import annotations

from typing import Any

import torch
from torch import nn

from openpi.models_pytorch.action_experts.base import ActionExpert


class IllibMoeVelocityFieldExpert(ActionExpert):
    def __init__(
        self,
        *,
        vlm_width: int,
        state_dim: int,
        action_dim: int,
        cond_dim: int = 512,
        num_experts: int = 4,
        top_k: int | None = None,
        expert_hidden_dim: int = 512,
        expert_hidden_depth: int = 2,
        gating_hidden_dim: int = 256,
        gating_hidden_depth: int = 1,
        activation: str = "silu",
        gating_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        try:
            from il_lib.nn.flow_matching.moe_velocity_field import ActionBlockMoEVelocityField
            from il_lib.nn.flow_matching.moe_velocity_field import ActionBlockSpec
        except Exception as e:  # noqa: BLE001
            raise ImportError(
                "il_lib is required for pytorch_action_expert='il_moe_velocity'. "
                "Install it in the runtime environment (e.g. pip install -e /path/to/il_lib)."
            ) from e

        self._cond_dim = int(cond_dim)
        self._vlm_width = int(vlm_width)
        self._state_dim = int(state_dim)
        self._action_dim = int(action_dim)

        self.prefix_proj = nn.Linear(self._vlm_width, self._cond_dim)
        self.state_proj = nn.Linear(self._state_dim, self._cond_dim)

        action_blocks = [ActionBlockSpec(name="all", start=0, dim=self._action_dim)]
        self.velocity_field = ActionBlockMoEVelocityField(
            action_dim=self._action_dim,
            action_blocks=action_blocks,
            cond_dim=self._cond_dim,
            num_experts=int(num_experts),
            top_k=top_k,
            expert_hidden_dim=int(expert_hidden_dim),
            expert_hidden_depth=int(expert_hidden_depth),
            gating_hidden_dim=int(gating_hidden_dim),
            gating_hidden_depth=int(gating_hidden_depth),
            activation=str(activation),
            gating_temperature=float(gating_temperature),
        )

    def _pool_prefix(self, *, prefix_out: torch.Tensor, prefix_pad_masks: torch.Tensor) -> torch.Tensor:
        mask = prefix_pad_masks.to(dtype=prefix_out.dtype).unsqueeze(-1)
        denom = mask.sum(dim=1).clamp(min=1.0)
        pooled = (prefix_out * mask).sum(dim=1) / denom
        return pooled

    def _encode_prefix_cond(
        self,
        *,
        model: nn.Module,
        images: list[torch.Tensor],
        img_masks: list[torch.Tensor],
        lang_tokens: torch.Tensor,
        lang_masks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        att_2d_masks = model.make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        att_2d_masks_4d = model._prepare_attention_masks_4d(att_2d_masks)

        prefix_output = model.paligemma_with_expert.paligemma.language_model.forward(
            inputs_embeds=prefix_embs,
            attention_mask=att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=None,
            use_cache=False,
            adarms_cond=None,
        )
        prefix_out = prefix_output.last_hidden_state
        pooled = self._pool_prefix(prefix_out=prefix_out, prefix_pad_masks=prefix_pad_masks)
        prefix_cond = self.prefix_proj(pooled.to(dtype=torch.float32))
        return prefix_cond, prefix_pad_masks

    def encode_prefix(
        self,
        *,
        model: nn.Module,
        images: list[torch.Tensor],
        img_masks: list[torch.Tensor],
        lang_tokens: torch.Tensor,
        lang_masks: torch.Tensor,
    ) -> dict[str, Any]:
        prefix_cond, prefix_pad_masks = self._encode_prefix_cond(
            model=model, images=images, img_masks=img_masks, lang_tokens=lang_tokens, lang_masks=lang_masks
        )
        return {"prefix_cond": prefix_cond, "prefix_pad_masks": prefix_pad_masks}

    def _build_cond(self, *, prefix_cond: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        return prefix_cond + self.state_proj(state.to(dtype=torch.float32))

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
    ) -> torch.Tensor:
        prefix_cond, _ = self._encode_prefix_cond(
            model=model, images=images, img_masks=img_masks, lang_tokens=lang_tokens, lang_masks=lang_masks
        )
        cond = self._build_cond(prefix_cond=prefix_cond, state=state)
        return self.velocity_field(x=x_t.to(dtype=torch.float32), t=time.to(dtype=torch.float32), cond=cond)

    def compute_velocity_infer(
        self,
        *,
        model: nn.Module,
        prefix_ctx: dict[str, Any],
        state: torch.Tensor,
        x_t: torch.Tensor,
        time: torch.Tensor,
    ) -> torch.Tensor:
        prefix_cond = prefix_ctx["prefix_cond"]
        cond = self._build_cond(prefix_cond=prefix_cond, state=state)
        return self.velocity_field(x=x_t.to(dtype=torch.float32), t=time.to(dtype=torch.float32), cond=cond)

