"""
All Sundial-related components are inlined here to keep the model definition self-contained.
"""

from __future__ import annotations

import math
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    Cache,
    DynamicCache,
    GenerationMixin,
    LogitsProcessorList,
    PreTrainedModel,
    PretrainedConfig,
    StoppingCriteriaList,
)
from transformers.activations import ACT2FN
from transformers.generation import EosTokenCriteria, validate_stopping_criteria
from transformers.generation.utils import (
    GenerateDecoderOnlyOutput,
    GenerateEncoderDecoderOutput,
    GenerateNonBeamOutput,
    GenerateOutput,
    GenerationConfig,
)
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from transformers.modeling_outputs import MoeCausalLMOutputWithPast, MoeModelOutputWithPast
from transformers.utils import ModelOutput

from models.coms_rope import apply_rotary_pos_emb, RotaryEmbeddingExt


class FlowLoss(nn.Module):
    """Flow Loss"""

    def __init__(self, target_channels: int, z_channels: int, depth: int, width: int, num_sampling_steps: int):
        super().__init__()
        self.in_channels = target_channels
        self.net = SimpleMLPAdaLN(
            in_channels=target_channels,
            model_channels=width,
            out_channels=target_channels,
            z_channels=z_channels,
            num_res_blocks=depth,
        )
        self.num_sampling_steps = num_sampling_steps

    def forward(
        self,
        target: torch.Tensor,
        z: torch.Tensor,
        mask: torch.Tensor | None = None,
        mask_y: torch.Tensor | None = None,
    ) -> torch.Tensor:
        noise = torch.randn_like(target)
        t = torch.rand(target.shape[0], device=target.device)

        noised_target = t[:, None] * target + (1 - t[:, None]) * noise

        predict_v = self.net(noised_target, t * 1000, z)

        weights = 1.0 / \
            torch.arange(1, self.in_channels + 1,
                         dtype=torch.float32, device=target.device)
        if mask_y is not None:
            loss = (mask_y * weights * (predict_v - target) ** 2).sum(dim=-1)
        else:
            loss = (weights * (predict_v - target) ** 2).sum(dim=-1)

        if mask is not None:
            loss = (loss * mask).sum() / mask.sum()
        return loss.mean()

    def sample(self, z: torch.Tensor, num_samples: int = 1) -> torch.Tensor:
        z = z.repeat(num_samples, 1)
        noise = torch.randn(z.shape[0], self.in_channels).to(z.device)
        x = noise
        dt = 1.0 / self.num_sampling_steps
        for i in range(self.num_sampling_steps):
            t = (torch.ones((x.shape[0])) * i /
                 self.num_sampling_steps).to(x.device)
            pred = self.net(x, t * 1000, z)
            x = x + (pred - noise) * dt
        x = x.reshape(num_samples, -1, self.in_channels).transpose(0, 1)
        return x


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale) + shift


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
            device=t.device
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class ResBlock(nn.Module):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels

        self.in_ln = nn.LayerNorm(channels, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels, bias=True),
            nn.SiLU(),
            nn.Linear(channels, channels, bias=True),
        )

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(channels, 3 * channels, bias=True))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(
            y).chunk(3, dim=-1)
        h = modulate(self.in_ln(x), shift_mlp, scale_mlp)
        h = self.mlp(h)
        return x + gate_mlp * h


class FinalLayer(nn.Module):
    """
    The final layer adopted from DiT.
    """

    def __init__(self, model_channels: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(
            model_channels, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(model_channels, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(
            model_channels, 2 * model_channels, bias=True))

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class SimpleMLPAdaLN(nn.Module):
    """
    The MLP for Diffusion Loss.
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param z_channels: channels in the condition.
    :param num_res_blocks: number of residual blocks per downsample.
    """

    def __init__(self, in_channels: int, model_channels: int, out_channels: int, z_channels: int, num_res_blocks: int):
        super().__init__()

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks

        self.time_embed = TimestepEmbedder(model_channels)
        self.cond_embed = nn.Linear(z_channels, model_channels)

        self.input_proj = nn.Linear(in_channels, model_channels)

        res_blocks = []
        for _ in range(num_res_blocks):
            res_blocks.append(
                ResBlock(
                    model_channels,
                )
            )

        self.res_blocks = nn.ModuleList(res_blocks)
        self.final_layer = FinalLayer(model_channels, out_channels)

        self.initialize_weights()

    def initialize_weights(self) -> None:
        def _basic_init(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        nn.init.normal_(self.time_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_embed.mlp[2].weight, std=0.02)

        for block in self.res_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x: torch.Tensor, t: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Apply the model to an input batch.
        :param x: an [N x C] Tensor of inputs.
        :param t: a 1-D batch of timesteps.
        :param c: conditioning from AR transformer.
        :return: an [N x C] Tensor of outputs.
        """
        x = self.input_proj(x)
        t = self.time_embed(t)
        c = self.cond_embed(c)
        y = t + c

        for block in self.res_blocks:
            x = block(x, y)

        return self.final_layer(x, y)


class TSGenerationMixin(GenerationMixin):
    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        prefix_allowed_tokens_fn: Optional[Callable[[
            int, torch.Tensor], List[int]]] = None,
        synced_gpus: Optional[bool] = None,
        assistant_model: Optional["PreTrainedModel"] = None,
        streamer: Optional["BaseStreamer"] = None,
        negative_prompt_ids: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        revin: Optional[bool] = True,
        num_samples: Optional[int] = 1,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        if revin:
            means = inputs.mean(dim=-1, keepdim=True)
            stdev = inputs.std(dim=-1, keepdim=True, unbiased=False) + 1e-5
            inputs = (inputs - means) / stdev
        outputs = super().generate(
            inputs=inputs,
            generation_config=generation_config,
            logits_processor=logits_processor,
            stopping_criteria=stopping_criteria,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            synced_gpus=synced_gpus,
            assistant_model=assistant_model,
            streamer=streamer,
            negative_prompt_ids=negative_prompt_ids,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            num_samples=num_samples,
            **kwargs,
        )
        if revin:
            stdev = stdev.unsqueeze(1).repeat(1, num_samples, 1)
            means = means.unsqueeze(1).repeat(1, num_samples, 1)
            outputs = (outputs * stdev) + means
        return outputs

    def _greedy_search(
        self,
        input_ids: torch.Tensor,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        output_logits: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        synced_gpus: bool = False,
        streamer: Optional["BaseStreamer"] = None,
        **model_kwargs,
    ) -> Union[GenerateNonBeamOutput, torch.Tensor]:
        input_ids = input_ids.to(self.device)
        batch_size, cur_len = input_ids.shape
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        if max_length is not None:
            warnings.warn(
                "`max_length` is deprecated in this function, use"
                " `stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length=max_length)])` instead.",
                UserWarning,
            )
            stopping_criteria = validate_stopping_criteria(
                stopping_criteria, max_length)
        pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
        if eos_token_id is not None:
            stopping_criteria.append(
                EosTokenCriteria(eos_token_id=eos_token_id))
        else:
            eos_token_id = [criteria.eos_token_id.tolist(
            ) for criteria in stopping_criteria if hasattr(criteria, "eos_token_id")]
            eos_token_id = eos_token_id[0] if eos_token_id else None
            if eos_token_id is None and self.generation_config.eos_token_id is not None:
                eos_token_id = self.generation_config.eos_token_id
                stopping_criteria.append(
                    EosTokenCriteria(eos_token_id=eos_token_id))

        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
        output_attentions = output_attentions if output_attentions is not None else self.generation_config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate if return_dict_in_generate is not None else self.generation_config.return_dict_in_generate
        )

        raw_logits = () if (return_dict_in_generate and output_logits) else None
        scores = () if (return_dict_in_generate and output_scores) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (
            return_dict_in_generate and output_hidden_states) else None

        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get(
                "attentions") if output_attentions else None
            encoder_hidden_states = model_kwargs["encoder_outputs"].get(
                "hidden_states") if output_hidden_states else None

        if "inputs_embeds" in model_kwargs:
            cur_len = model_kwargs["inputs_embeds"].shape[1]
        this_peer_finished = False
        unfinished_sequences = torch.ones(
            batch_size, dtype=torch.long, device=input_ids.device)
        model_kwargs["cache_position"] = torch.arange(
            cur_len, device=input_ids.device)
        true_seq_len = (cur_len + self.config.input_token_len -
                        1) // self.config.input_token_len
        model_kwargs["attention_mask"] = model_kwargs["attention_mask"][:, -true_seq_len:]
        max_length = stopping_criteria.max_length
        generate_results = None
        while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
            model_inputs = self.prepare_inputs_for_generation(
                input_ids, **model_kwargs)

            input_length = input_ids.shape[1]

            outputs = self(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                max_output_length=max_length - input_length,
            )

            if synced_gpus and this_peer_finished:
                continue
            next_token_logits = outputs.logits

            next_tokens_scores = logits_processor(input_ids, next_token_logits)

            if return_dict_in_generate:
                if output_scores:
                    scores += (next_tokens_scores,)
                if output_logits:
                    raw_logits += (next_token_logits,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (
                            outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,) if self.config.is_encoder_decoder else (
                            outputs.hidden_states,)
                    )

            next_tokens = next_tokens_scores

            if eos_token_id is not None:
                next_tokens = next_tokens * unfinished_sequences + \
                    pad_token_id * (1 - unfinished_sequences)

            horizon_length = next_tokens.shape[-1] // self.config.input_token_len

            past_key_values = model_kwargs.get("past_key_values")
            if past_key_values is None:
                generate_results = next_tokens
            else:
                generate_results = torch.cat(
                    [generate_results, next_tokens], dim=-1)
            input_ids = torch.cat(
                [input_ids, next_tokens.median(dim=1)[0]], dim=-1)

            if streamer is not None:
                streamer.put(next_tokens.cpu())
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                horizon_length=horizon_length,
                is_encoder_decoder=self.config.is_encoder_decoder,
            )
            unfinished_sequences = unfinished_sequences & ~stopping_criteria(
                input_ids, scores)
            this_peer_finished = unfinished_sequences.max() == 0

        if input_ids.shape[-1] > max_length:
            input_ids = input_ids[:, :max_length]

        if streamer is not None:
            streamer.end()

        if return_dict_in_generate:
            if self.config.is_encoder_decoder:
                return GenerateEncoderDecoderOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
            else:
                return GenerateDecoderOnlyOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
        else:
            return generate_results[:, :, : (max_length - cur_len)]

    def _update_model_kwargs_for_generation(
        self,
        outputs: ModelOutput,
        model_kwargs: Dict[str, Any],
        horizon_length: int = 1,
        is_encoder_decoder: bool = False,
        standardize_cache_format: bool = False,
    ) -> Dict[str, Any]:
        model_kwargs["past_key_values"] = self._extract_past_from_model_output(
            outputs, standardize_cache_format=standardize_cache_format
        )
        if getattr(outputs, "state", None) is not None:
            model_kwargs["state"] = outputs.state

        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = torch.cat(
                [token_type_ids, token_type_ids[:, -1].unsqueeze(-1)], dim=-1)

        if not is_encoder_decoder:
            if "attention_mask" in model_kwargs:
                attention_mask = model_kwargs["attention_mask"]
                model_kwargs["attention_mask"] = torch.cat(
                    [attention_mask, attention_mask.new_ones((attention_mask.shape[0], horizon_length))], dim=-1
                )
        else:
            if "decoder_attention_mask" in model_kwargs:
                decoder_attention_mask = model_kwargs["decoder_attention_mask"]
                model_kwargs["decoder_attention_mask"] = torch.cat(
                    [decoder_attention_mask, decoder_attention_mask.new_ones(
                        (decoder_attention_mask.shape[0], horizon_length))],
                    dim=-1,
                )

        if "cache_position" in model_kwargs and model_kwargs["cache_position"] is not None:
            model_kwargs["cache_position"] = model_kwargs["cache_position"][-1:] + horizon_length

        return model_kwargs


class SundialConfig(PretrainedConfig):
    model_type = "sundial"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        input_token_len: int = 16,
        hidden_size: int = 768,
        intermediate_size: int = 3072,
        output_token_lens: List[int] = [720],
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        hidden_act: str = "silu",
        use_cache: bool = True,
        rope_theta: int = 10000,
        dropout_rate: float = 0.1,
        initializer_range: float = 0.02,
        max_position_embeddings: int = 10000,
        flow_loss_depth: int = 3,
        num_sampling_steps: int = 50,
        diffusion_batch_mul: int = 4,
        **kwargs,
    ):
        self.input_token_len = input_token_len
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.output_token_lens = output_token_lens
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.dropout_rate = dropout_rate
        self.initializer_range = initializer_range
        self.max_position_embeddings = max_position_embeddings
        self.flow_loss_depth = flow_loss_depth
        self.num_sampling_steps = num_sampling_steps
        self.diffusion_batch_mul = diffusion_batch_mul

        super().__init__(**kwargs)


def standardize_w_mask(x: torch.Tensor, mask: torch.Tensor, min_std: float = 1e-3) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Normalize x using only observed entries (mask==1). Returns normalized x, mean, and std for de-normalization.
    """
    obs_count = torch.clamp(mask.sum(dim=1, keepdim=True), min=1e-6)
    means = ((x * mask).sum(dim=1, keepdim=True) / obs_count)
    var = (((x - means) ** 2) * mask).sum(dim=1, keepdim=True) / obs_count
    stdev = torch.sqrt(var)
    stdev = torch.where(stdev > min_std, stdev,
                        torch.tensor(1.0, device=x.device))
    x_norm = ((x - means) / stdev) * mask
    return x_norm, means, stdev


VALID_OBSERVED_MISSING_POLICIES = {"mask_aware", "seq_mean_impute"}


def model_observed_mask(
    observed_mask: torch.Tensor,
    policy: str,
    in_window_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Return the mask the backbone should see for observed history tokens.

    ``observed_mask`` remains the source of truth for normalization, text
    anchoring, loss, and evaluation. ``seq_mean_impute`` only changes which
    in-window history slots the frozen backbone/tokenizer treats as valid.
    """
    if policy not in VALID_OBSERVED_MISSING_POLICIES:
        raise ValueError(
            f"Unsupported observed_missing_policy: {policy}. "
            f"Expected one of {sorted(VALID_OBSERVED_MISSING_POLICIES)}."
        )
    if policy == "mask_aware":
        return observed_mask
    if in_window_mask is None:
        in_window_mask = torch.ones_like(observed_mask, dtype=observed_mask.dtype)
    return in_window_mask.to(device=observed_mask.device, dtype=observed_mask.dtype)


class SundialPatchEmbedding(nn.Module):
    def __init__(self, config: SundialConfig):
        super().__init__()
        self.dropout = nn.Dropout(config.dropout_rate)
        self.hidden_layer = nn.Linear(
            config.input_token_len * 2, config.intermediate_size)
        self.act = ACT2FN[config.hidden_act]
        self.output_layer = nn.Linear(
            config.intermediate_size, config.hidden_size)
        self.residual_layer = nn.Linear(
            config.input_token_len * 2, config.hidden_size)
        self.input_token_len = config.input_token_len

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if mask is None:
            mask = torch.ones_like(x, dtype=torch.float32)
        input_length = x.shape[-1]
        padding_length = (self.input_token_len - (input_length %
                          self.input_token_len)) % self.input_token_len
        x = F.pad(x, (padding_length, 0))
        mask = F.pad(mask, (padding_length, 0))
        x = x.unfold(dimension=-1, size=self.input_token_len,
                     step=self.input_token_len)
        mask = mask.unfold(
            dimension=-1, size=self.input_token_len, step=self.input_token_len)

        x = torch.cat([x, mask], dim=-1)
        hid = self.act(self.hidden_layer(x))
        out = self.dropout(self.output_layer(hid))
        res = self.residual_layer(x)
        out = out + res
        return out


class SundialAttention(nn.Module):
    def __init__(self, config: SundialConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.attention_dropout = config.dropout_rate
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.rotary_emb = RotaryEmbeddingExt(
            self.head_dim, max_position_embeddings=config.max_position_embeddings)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        n_text_tokens: int = 0,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        nt = n_text_tokens if (n_text_tokens > 0 and hasattr(self, "k_proj_text")) else 0
        if nt > 0:
            key_states = torch.cat(
                [self.k_proj_text(hidden_states[:, :nt]),
                 self.k_proj(hidden_states[:, nt:])],
                dim=1,
            )
            value_states = torch.cat(
                [self.v_proj_text(hidden_states[:, :nt]),
                 self.v_proj(hidden_states[:, nt:])],
                dim=1,
            )
        else:
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value.get_usable_length(
                kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(
            value_states, seq_len=kv_seq_len, position_ids=position_ids)
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx)

        attn_output = F.scaled_dot_product_attention(
            query_states, key_states, value_states, attention_mask, dropout_p=(
                self.attention_dropout if self.training else 0.0)
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        attn_weights = None
        if output_attentions:
            attn_weights = torch.einsum(
                "bnqd,bnkd->bnqk", query_states, key_states)

        return attn_output, attn_weights, past_key_value


class SundialMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, hidden_act: str):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(
            self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(
            self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(
            self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(hidden_state)) * self.up_proj(hidden_state))


class SundialDecoderLayer(nn.Module):
    def __init__(self, config: SundialConfig, layer_idx: int):
        super().__init__()
        self.self_attn = SundialAttention(config, layer_idx)

        self.ffn_layer = SundialMLP(
            hidden_size=config.hidden_size, intermediate_size=config.intermediate_size, hidden_act=config.hidden_act
        )
        self.norm1 = torch.nn.LayerNorm(config.hidden_size)
        self.norm2 = torch.nn.LayerNorm(config.hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        n_text_tokens: int = 0,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, Optional[torch.FloatTensor], Optional[torch.FloatTensor]]:
        residual = hidden_states

        hidden_states = self.norm1(hidden_states)

        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            n_text_tokens=n_text_tokens,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.ffn_layer(hidden_states)
        hidden_states = residual + hidden_states

        if not output_attentions:
            self_attn_weights = None

        if not use_cache:
            present_key_value = None
        return hidden_states, self_attn_weights, present_key_value


class SundialPreTrainedModel(PreTrainedModel):
    config_class = SundialConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["SundialDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = False
    _supports_cache_class = True

    def _init_weights(self, module: nn.Module) -> None:
        std = self.config.initializer_range
        if isinstance(module, torch.nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, torch.nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class SundialModel(SundialPreTrainedModel):
    def __init__(self, config: SundialConfig):
        super().__init__(config)
        self.embed_layer = SundialPatchEmbedding(config)
        self.layers = nn.ModuleList([SundialDecoderLayer(
            config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        self.norm = torch.nn.LayerNorm(config.hidden_size)
        self.gradient_checkpointing = False

    def forward(
        self,
        input_ids: torch.FloatTensor | None = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        n_text_tokens: int = 0,
    ) -> Union[Tuple, MoeModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape

        if inputs_embeds is None:
            inputs_embeds = self.embed_layer(input_ids)
            seq_length = inputs_embeds.shape[1]

        if self.gradient_checkpointing and self.training:
            if use_cache:
                use_cache = False

        past_key_values_length = 0

        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(
                    past_key_values)
            past_key_values_length = past_key_values.get_usable_length(
                seq_length)

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device)
            position_ids = position_ids.view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length)
            if not torch.is_floating_point(position_ids):
                position_ids = position_ids.long()

        if attention_mask is not None and attention_mask.dim() == 4:
            pass
        else:
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
                sliding_window=None,
            )

        hidden_states = inputs_embeds

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    n_text_tokens,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    n_text_tokens=n_text_tokens,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

            if use_cache:
                next_decoder_cache = layer_outputs[2]

        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = next_decoder_cache.to_legacy_cache(
            ) if use_legacy_cache else next_decoder_cache

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return MoeModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class SundialForPrediction(SundialPreTrainedModel, TSGenerationMixin):
    def __init__(self, config: SundialConfig):
        super().__init__(config)
        self.config = config
        self.model = SundialModel(self.config)
        self.flow_loss = FlowLoss(
            self.config.output_token_lens[-1],
            self.config.hidden_size,
            self.config.flow_loss_depth,
            self.config.hidden_size,
            self.config.num_sampling_steps,
        )
        self.post_init()

    def set_decoder(self, decoder: nn.Module) -> None:
        self.model = decoder

    def get_decoder(self) -> nn.Module:
        return self.model

    def forward(
        self,
        input_ids: torch.FloatTensor | None = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.FloatTensor] = None,
        loss_masks: Optional[torch.FloatTensor] = None,
        mask_y: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        max_output_length: Optional[int] = None,
        revin: Optional[bool] = False,
        num_samples: Optional[int] = 1,
    ) -> Union[Tuple, MoeCausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if revin:
            means = input_ids.mean(1, keepdim=True).detach()
            stdev = input_ids.std(dim=1, keepdim=True, unbiased=False).detach()
            stdev = torch.where(stdev > 1e-2, stdev,
                                torch.tensor(1.0, device=input_ids.device))
            input_ids = (input_ids - means) / stdev
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0] if not return_dict else outputs.last_hidden_state
        predictions = None

        loss = None
        if labels is not None:
            if revin:
                labels = (labels - means) / stdev
            output_token_len = self.config.output_token_lens[-1]
            seq_len = hidden_states.shape[1] * self.config.input_token_len
            labels = labels[:, : seq_len -
                            self.config.input_token_len + output_token_len]
            shift_labels = labels.unfold(
                dimension=-1, size=output_token_len, step=self.config.input_token_len)

            bsz, L, _ = shift_labels.shape
            shift_labels = shift_labels.reshape(
                bsz * L, -1).repeat(self.config.diffusion_batch_mul, 1)
            hidden_states = hidden_states.reshape(
                bsz * L, -1).repeat(self.config.diffusion_batch_mul, 1)
            loss_masks = loss_masks.reshape(
                bsz * L).repeat(self.config.diffusion_batch_mul)
            mask_y = mask_y.repeat(L * self.config.diffusion_batch_mul, 1)

            loss = self.flow_loss(
                shift_labels, hidden_states, loss_masks, mask_y)
        else:
            if max_output_length is None:
                output_token_len = self.config.output_token_lens[0]
                max_output_length = output_token_len
            else:
                output_token_len = self.config.output_token_lens[0]
                for h in self.config.output_token_lens[1:]:
                    if h > max_output_length:
                        break
                    else:
                        output_token_len = h

            bsz = hidden_states.shape[0]
            hidden_states = hidden_states[:, -1, :]
            predictions = self.flow_loss.sample(hidden_states, num_samples)
            if output_token_len > max_output_length:
                predictions = predictions[:, :, :max_output_length]
            if revin:
                predictions = predictions * stdev + means
        if not return_dict:
            output = (predictions,) + outputs[1:]
            return (loss) + output if loss is not None else output

        return MoeCausalLMOutputWithPast(
            loss=loss,
            logits=predictions,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.Tensor,
        past_key_values: Cache | Tuple | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        revin: bool = False,
        num_samples: int = 1,
        **kwargs,
    ) -> Dict[str, Any]:
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                cache_length = past_key_values.get_seq_length()
                if isinstance(past_key_values, DynamicCache):
                    past_length = past_key_values.seen_tokens
                else:
                    past_length = cache_length

                max_cache_length = past_key_values.get_max_length()
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]
                max_cache_length = None

            if attention_mask is not None and attention_mask.shape[1] > (input_ids.shape[1] // self.config.input_token_len):
                input_ids = input_ids[:, -(attention_mask.shape[1] -
                                           past_length) * self.config.input_token_len:]
            elif past_length < (input_ids.shape[1] // self.config.input_token_len):
                input_ids = input_ids[:, past_length *
                                      self.config.input_token_len:]

            if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + (input_ids.shape[1] // self.config.input_token_len) > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -
                                            (input_ids.shape[1] // self.config.input_token_len):]

        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "revin": revin,
                "num_samples": num_samples,
            }
        )
        return model_inputs
