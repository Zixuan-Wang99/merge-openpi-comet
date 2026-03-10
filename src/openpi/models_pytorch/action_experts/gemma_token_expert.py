from typing import Any

import torch
from torch import nn

from openpi.models_pytorch.action_experts.base import ActionExpert


class GemmaTokenExpert(ActionExpert):
    def encode_prefix(
        self,
        *,
        model: nn.Module,
        images: list[torch.Tensor],
        img_masks: list[torch.Tensor],
        lang_tokens: torch.Tensor,
        lang_masks: torch.Tensor,
    ) -> dict[str, Any]:
        prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        prefix_att_2d_masks = model.make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        prefix_att_2d_masks_4d = model._prepare_attention_masks_4d(prefix_att_2d_masks)

        model.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "sdpa" 
        _, past_key_values = model.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )
        return {
            "prefix_pad_masks": prefix_pad_masks,
            "past_key_values": past_key_values,
        }

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
        prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = model.embed_suffix(state, x_t, time)
        if (
            model.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
        att_2d_masks = model.make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        att_2d_masks_4d = model._prepare_attention_masks_4d(att_2d_masks)

        (_, suffix_out), _ = model.paligemma_with_expert.forward(
            attention_mask=att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )
        suffix_out = suffix_out[:, -model.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        return model.action_out_proj(suffix_out)
    
    # for subtask: returns velocity + prefix_output + prefix_pad_masks
    def compute_velocity_train_with_prefix(
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
        token_ar_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute velocity and return prefix hidden states for subtask CE loss.

        This method differs from compute_velocity_train in two ways:
        1. Returns (v_t, prefix_output, prefix_pad_masks) instead of just v_t.
        2. When token_ar_mask is provided, injects causal attention on the subtask
           portion of the prefix text tokens. This matches the reference JAX
           implementation's Pi05Subtask.compute_loss.

        Args:
            model: The PI0Pytorch or VLM2 model.
            images, img_masks, lang_tokens, lang_masks, state, x_t, time:
                Standard training inputs.
            token_ar_mask: int Tensor [B, max_token_len] with 0=bidirectional,
                1=causal. Produced by TokenizeSubtaskTraining. If None, all
                prefix text tokens use bidirectional attention (standard Pi0.5).

        Returns:
            v_t: Predicted velocity [B, action_horizon, action_dim]
            prefix_output: VLM prefix hidden states [B, prefix_S, hidden_dim] (after RMSNorm)
            prefix_pad_masks: Prefix padding masks [B, prefix_S]
        """

        prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = model.embed_suffix(state, x_t, time)

        if (
            model.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        # prefix_att_masks is [B, prefix_S] where prefix = [img_tokens..., text_tokens].
        # By default, all are 0 (bidirectional). When token_ar_mask is provided,
        # we replace the text portion's att_mask with token_ar_mask so that subtask
        # tokens use causal (autoregressive) attention.
        if token_ar_mask is not None:
            B = prefix_att_masks.shape[0]
            prefix_S = prefix_att_masks.shape[1]
            num_text = lang_tokens.shape[1]  # = max_token_len
            num_img = prefix_S - num_text

            # Image tokens: bidirectional (0)
            img_ar = torch.zeros(B, num_img, dtype=prefix_att_masks.dtype, device=prefix_att_masks.device)
            # Text tokens: use token_ar_mask (0=bidir, 1=causal)
            text_ar = token_ar_mask.to(dtype=prefix_att_masks.dtype, device=prefix_att_masks.device)
            # Replace prefix att_masks
            prefix_att_masks = torch.cat([img_ar, text_ar], dim=1)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
        att_2d_masks = model.make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        att_2d_masks_4d = model._prepare_attention_masks_4d(att_2d_masks)

        (prefix_output, suffix_out), _ = model.paligemma_with_expert.forward(
            attention_mask=att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )
        suffix_out = suffix_out[:, -model.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        v_t = model.action_out_proj(suffix_out)
        return v_t, prefix_output, prefix_pad_masks

    def compute_velocity_infer(
        self,
        *,
        model: nn.Module,
        prefix_ctx: dict[str, Any],
        state: torch.Tensor,
        x_t: torch.Tensor,
        time: torch.Tensor,
    ) -> torch.Tensor:
        prefix_pad_masks = prefix_ctx["prefix_pad_masks"]
        past_key_values = prefix_ctx["past_key_values"]

        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = model.embed_suffix(state, x_t, time)
        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
        suffix_att_2d_masks = model.make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1
        full_att_2d_masks_4d = model._prepare_attention_masks_4d(full_att_2d_masks)

        model.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "sdpa"  # noqa: SLF001
        outputs_embeds, _ = model.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )

        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -model.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        return model.action_out_proj(suffix_out)

