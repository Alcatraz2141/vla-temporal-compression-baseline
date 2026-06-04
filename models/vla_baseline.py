from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torchvision import models

from memory.multiscale_memory import MultiScaleMemory


def _build_resnet18(pretrained: bool) -> tuple[nn.Module, int]:
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    backbone = models.resnet18(weights=weights)
    feature_dim = backbone.fc.in_features
    backbone.fc = nn.Identity()
    return backbone, feature_dim


def _build_resnet50(pretrained: bool) -> tuple[nn.Module, int]:
    weights = models.ResNet50_Weights.DEFAULT if pretrained else None
    backbone = models.resnet50(weights=weights)
    feature_dim = backbone.fc.in_features
    backbone.fc = nn.Identity()
    return backbone, feature_dim


class BehaviorCloningAgent(nn.Module):
    """Non-VLM BC baseline: ResNet-50 + proprio MLP + action MLP."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        T_action: int,
        pretrained_vision: bool = True,
        hidden_dim: int = 512,
    ) -> None:
        super().__init__()
        self.T_action = T_action
        self.action_dim = action_dim
        self.vision, vision_dim = _build_resnet50(pretrained_vision)
        self.state_encoder = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.head = nn.Sequential(
            nn.Linear(vision_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, T_action * action_dim),
        )

    def forward(self, images: torch.Tensor, states: torch.Tensor, **_: Any) -> torch.Tensor:
        # Use the latest observation to reproduce the simple single-step BC baseline.
        image = images[:, -1]
        state = states[:, -1]
        features = torch.cat([self.vision(image), self.state_encoder(state)], dim=-1)
        return self.head(features).view(image.size(0), self.T_action, self.action_dim)


class BaselineVLA(nn.Module):
    """Short-context VLA-style BC model with per-timestep image/state tokens."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        T_obs: int,
        T_action: int,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 4,
        dropout: float = 0.1,
        pretrained_vision: bool = True,
        vision_encoder: str = "resnet18",
        use_memory: bool = False,
        state_hidden_dim: int = 128,
        action_hidden_dim: int = 256,
        use_action_history: bool = False,
        use_language: bool = False,
        language_vocab_size: int = 1024,
    ) -> None:
        super().__init__()
        self.T_obs = T_obs
        self.T_action = T_action
        self.action_dim = action_dim
        self.use_action_history = use_action_history
        self.use_language = use_language
        if vision_encoder != "resnet18":
            raise ValueError("BaselineVLA currently supports vision_encoder=resnet18.")
        self.vision, vision_dim = _build_resnet18(pretrained_vision)
        self.vision_proj = nn.Linear(vision_dim, d_model)
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, state_hidden_dim),
            nn.ReLU(),
            nn.Linear(state_hidden_dim, d_model),
        )
        if self.use_action_history:
            self.action_encoder = nn.Sequential(nn.Linear(action_dim, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        else:
            self.action_encoder = None
        self.language_embedding = nn.Embedding(language_vocab_size, d_model) if self.use_language else None
        self.fusion = nn.Linear(d_model * (3 if self.use_action_history else 2), d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, T_obs, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.memory = MultiScaleMemory(d_model) if use_memory else None
        self.action_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, action_hidden_dim),
            nn.GELU(),
            nn.Linear(action_hidden_dim, T_action * action_dim),
        )

    def forward(
        self,
        images: torch.Tensor,
        states: torch.Tensor,
        actions: torch.Tensor | None = None,
        language_ids: torch.Tensor | None = None,
        past_memory: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        bsz, timesteps, channels, height, width = images.shape
        flat_images = images.view(bsz * timesteps, channels, height, width)
        if flat_images.is_cuda:
            flat_images = flat_images.contiguous(memory_format=torch.channels_last)
        vision_tokens = self.vision_proj(self.vision(flat_images)).view(bsz, timesteps, -1)
        state_tokens = self.state_encoder(states)
        pieces = [vision_tokens, state_tokens]
        if self.use_action_history:
            if actions is None:
                actions = torch.zeros(bsz, timesteps, self.action_dim, dtype=states.dtype, device=states.device)
            pieces.append(self.action_encoder(actions))
        tokens = self.fusion(torch.cat(pieces, dim=-1))
        if self.language_embedding is not None:
            if language_ids is None:
                language_ids = torch.zeros(bsz, dtype=torch.long, device=states.device)
            tokens = tokens + self.language_embedding(language_ids).unsqueeze(1)
        tokens = tokens + self.pos_embedding[:, :timesteps]
        tokens = self.transformer(tokens)
        if self.memory is not None:
            tokens, _ = self.memory(tokens, past_memory)
        pooled = tokens[:, -1]
        return self.action_head(pooled).view(bsz, self.T_action, self.action_dim)


class RT1StyleBaseline(nn.Module):
    """RT-1-inspired baseline using CNN tokens + causal Transformer policy.

    This is not an exact reproduction of Google's RT-1 training recipe. It is a
    controlled PyTorch baseline with RT-1-like ingredients: per-timestep visual
    tokens, proprioceptive tokens, temporal embeddings, causal attention, and
    chunked action prediction.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        T_obs: int,
        T_action: int,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 4,
        dropout: float = 0.1,
        pretrained_vision: bool = True,
        action_hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.T_obs = T_obs
        self.T_action = T_action
        self.action_dim = action_dim
        self.vision, vision_dim = _build_resnet18(pretrained_vision)
        self.image_token = nn.Linear(vision_dim, d_model)
        self.state_token = nn.Sequential(
            nn.Linear(state_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.type_embedding = nn.Parameter(torch.zeros(1, 2, d_model))
        self.time_embedding = nn.Parameter(torch.zeros(1, T_obs, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.action_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, action_hidden_dim),
            nn.GELU(),
            nn.Linear(action_hidden_dim, T_action * action_dim),
        )

    def forward(self, images: torch.Tensor, states: torch.Tensor, **_: Any) -> torch.Tensor:
        bsz, timesteps, channels, height, width = images.shape
        flat_images = images.view(bsz * timesteps, channels, height, width)
        if flat_images.is_cuda:
            flat_images = flat_images.contiguous(memory_format=torch.channels_last)
        image_tokens = self.image_token(self.vision(flat_images)).view(bsz, timesteps, -1)
        state_tokens = self.state_token(states)

        tokens = torch.stack([image_tokens, state_tokens], dim=2)
        tokens = tokens + self.type_embedding[:, None, :, :]
        tokens = tokens + self.time_embedding[:, :timesteps, None, :]
        tokens = tokens.flatten(1, 2)
        causal_mask = torch.triu(
            torch.ones(tokens.size(1), tokens.size(1), device=tokens.device, dtype=torch.bool),
            diagonal=1,
        )
        tokens = self.transformer(tokens, mask=causal_mask)
        pooled = tokens[:, -1]
        return self.action_head(pooled).view(bsz, self.T_action, self.action_dim)


class ACTChunkedBaseline(nn.Module):
    """ACT-style chunked behavior cloning baseline with learned action queries."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        T_obs: int,
        T_action: int,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 4,
        dropout: float = 0.1,
        pretrained_vision: bool = True,
        vision_encoder: str = "resnet18",
        state_hidden_dim: int = 128,
        action_hidden_dim: int = 256,
        use_action_history: bool = True,
        use_language: bool = True,
        language_vocab_size: int = 1024,
        input_modalities: str = "vision_state_action",
        use_phase: bool = False,
        phase_vocab_size: int = 4,
        use_object_signals: bool = False,
        use_event_memory: bool = False,
        memory_chunk_size: int = 8,
        max_memory_tokens: int = 8,
        gate_type: str = "event",
    ) -> None:
        super().__init__()
        self.T_obs = T_obs
        self.T_action = T_action
        self.action_dim = action_dim
        self.use_action_history = use_action_history
        self.use_language = use_language
        self.use_phase = use_phase
        self.use_object_signals = use_object_signals
        self.use_event_memory = use_event_memory
        self.memory_chunk_size = max(int(memory_chunk_size), 1)
        self.max_memory_tokens = max(int(max_memory_tokens), 1)
        if gate_type not in {"event", "age_based"}:
            raise ValueError("gate_type must be 'event' or 'age_based'.")
        self.gate_type = gate_type
        self.input_modalities = input_modalities
        allowed_modalities = {"vision_state_action", "vision_state", "state_action", "state"}
        if self.input_modalities not in allowed_modalities:
            raise ValueError(f"input_modalities must be one of {sorted(allowed_modalities)}.")
        self.use_vision = self.input_modalities in {"vision_state_action", "vision_state"}
        self.use_state = self.input_modalities in {"vision_state_action", "vision_state", "state_action", "state"}
        self.use_action_history = self.use_action_history and self.input_modalities in {"vision_state_action", "state_action"}
        if self.use_vision and vision_encoder != "resnet18":
            raise ValueError("ACTChunkedBaseline currently supports vision_encoder=resnet18.")

        if self.use_vision:
            self.vision, vision_dim = _build_resnet18(pretrained_vision)
            self.vision_proj = nn.Linear(vision_dim, d_model)
        else:
            self.vision = None
            self.vision_proj = None
        if self.use_state:
            self.state_encoder = nn.Sequential(
                nn.Linear(state_dim, state_hidden_dim),
                nn.ReLU(),
                nn.Linear(state_hidden_dim, d_model),
            )
        else:
            self.state_encoder = None
        if self.use_action_history:
            self.action_encoder = nn.Sequential(nn.Linear(action_dim, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        else:
            self.action_encoder = None
        self.language_embedding = nn.Embedding(language_vocab_size, d_model) if self.use_language else None
        self.phase_embedding = nn.Embedding(phase_vocab_size, d_model) if self.use_phase else None
        self.secured_embedding = nn.Embedding(2, d_model) if self.use_object_signals else None
        self.placement_ready_embedding = nn.Embedding(2, d_model) if self.use_object_signals else None
        fusion_inputs = int(self.use_vision) + int(self.use_state) + int(self.use_action_history)
        if fusion_inputs < 1:
            raise ValueError("ACTChunkedBaseline needs at least one input modality.")
        self.context_fusion = nn.Linear(d_model * fusion_inputs, d_model)
        self.context_pos_embedding = nn.Parameter(torch.zeros(1, T_obs, d_model))
        self.memory_pos_embedding = nn.Parameter(torch.zeros(1, self.max_memory_tokens, d_model))
        self.memory_pool_query = nn.Parameter(torch.randn(d_model) * 0.02)
        self.memory_gate = nn.Sequential(
            nn.Linear(d_model * 3 + 1, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )
        self.query_embedding = nn.Parameter(torch.randn(1, T_action, d_model) * 0.02)
        self.query_pos_embedding = nn.Parameter(torch.zeros(1, T_action, d_model))
        self.query_type_embedding = nn.Parameter(torch.zeros(1, 1, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        self.action_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, action_hidden_dim),
            nn.GELU(),
            nn.Linear(action_hidden_dim, action_dim),
        )

    def _encode_context_tokens(
        self,
        images: torch.Tensor,
        states: torch.Tensor,
        actions: torch.Tensor | None = None,
        language_ids: torch.Tensor | None = None,
        phase_ids: torch.Tensor | None = None,
        secured_ids: torch.Tensor | None = None,
        placement_ready_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bsz, timesteps = states.shape[:2]
        pieces = []
        if self.use_vision:
            _, _, channels, height, width = images.shape
            flat_images = images.view(bsz * timesteps, channels, height, width)
            if flat_images.is_cuda:
                flat_images = flat_images.contiguous(memory_format=torch.channels_last)
            pieces.append(self.vision_proj(self.vision(flat_images)).view(bsz, timesteps, -1))
        if self.use_state:
            pieces.append(self.state_encoder(states))
        if self.use_action_history:
            if actions is None:
                actions = torch.zeros(bsz, timesteps, self.action_dim, dtype=states.dtype, device=states.device)
            pieces.append(self.action_encoder(actions))

        context = self.context_fusion(torch.cat(pieces, dim=-1))
        if self.language_embedding is not None:
            if language_ids is None:
                language_ids = torch.zeros(bsz, dtype=torch.long, device=states.device)
            context = context + self.language_embedding(language_ids).unsqueeze(1)
        if self.phase_embedding is not None:
            if phase_ids is None:
                phase_ids = torch.zeros(bsz, timesteps, dtype=torch.long, device=states.device)
            context = context + self.phase_embedding(phase_ids.clamp(0, self.phase_embedding.num_embeddings - 1))
        if self.use_object_signals:
            if secured_ids is None:
                secured_ids = torch.zeros(bsz, timesteps, dtype=torch.long, device=states.device)
            if placement_ready_ids is None:
                placement_ready_ids = torch.zeros(bsz, timesteps, dtype=torch.long, device=states.device)
            context = context + self.secured_embedding(secured_ids.clamp(0, 1))
            context = context + self.placement_ready_embedding(placement_ready_ids.clamp(0, 1))
        return context

    def _event_memory_tokens(self, tokens: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, timesteps, dim = tokens.shape
        chunk_count = max((timesteps + self.memory_chunk_size - 1) // self.memory_chunk_size, 1)
        padded_steps = chunk_count * self.memory_chunk_size
        pad = padded_steps - timesteps
        if pad:
            tokens = torch.cat([tokens, tokens.new_zeros(bsz, pad, dim)], dim=1)
            mask = torch.cat([mask, torch.zeros(bsz, pad, dtype=torch.bool, device=mask.device)], dim=1)

        chunk_tokens = tokens.view(bsz, chunk_count, self.memory_chunk_size, dim)
        chunk_mask = mask.view(bsz, chunk_count, self.memory_chunk_size)

        attn_logits = torch.einsum("bcsd,d->bcs", chunk_tokens, self.memory_pool_query)
        attn_logits = attn_logits.masked_fill(~chunk_mask, -1e4)
        attn = torch.softmax(attn_logits, dim=-1) * chunk_mask.to(tokens.dtype)
        attn = attn / attn.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        summaries = torch.einsum("bcs,bcsd->bcd", attn, chunk_tokens)

        pair_mask = chunk_mask[:, :, 1:] & chunk_mask[:, :, :-1]
        pair_counts = pair_mask.sum(dim=-1).clamp_min(1).to(tokens.dtype).unsqueeze(-1)
        pair_mask_f = pair_mask.to(tokens.dtype).unsqueeze(-1)
        deltas = (chunk_tokens.diff(dim=2).abs() * pair_mask_f).sum(dim=2) / pair_counts
        chunk_mean = (chunk_tokens * chunk_mask.to(tokens.dtype).unsqueeze(-1)).sum(dim=2)
        chunk_mean = chunk_mean / chunk_mask.sum(dim=-1).clamp_min(1).to(tokens.dtype).unsqueeze(-1)
        age = torch.linspace(0.0, 1.0, chunk_count, device=tokens.device).view(1, chunk_count, 1).expand(bsz, -1, -1)
        if self.gate_type == "age_based":
            gate_score = age.clamp_min(1.0 / max(chunk_count, 1))
        else:
            gate_score = torch.sigmoid(self.memory_gate(torch.cat([summaries, deltas, chunk_mean, age], dim=-1)))
        memory_tokens = summaries * gate_score
        memory_mask = chunk_mask.any(dim=-1)
        if memory_tokens.size(1) > self.max_memory_tokens:
            memory_tokens = memory_tokens[:, -self.max_memory_tokens :]
            memory_mask = memory_mask[:, -self.max_memory_tokens :]
        empty_memory = ~memory_mask.any(dim=1)
        if empty_memory.any():
            memory_tokens = memory_tokens.clone()
            memory_mask = memory_mask.clone()
            memory_tokens[empty_memory, 0] = 0
            memory_mask[empty_memory, 0] = True
        return memory_tokens, memory_mask

    def forward(
        self,
        images: torch.Tensor,
        states: torch.Tensor,
        actions: torch.Tensor | None = None,
        older_obs: torch.Tensor | None = None,
        older_states: torch.Tensor | None = None,
        older_actions: torch.Tensor | None = None,
        recent_mask: torch.Tensor | None = None,
        older_mask: torch.Tensor | None = None,
        language_ids: torch.Tensor | None = None,
        recent_phase_ids: torch.Tensor | None = None,
        target_phase_ids: torch.Tensor | None = None,
        recent_secured_ids: torch.Tensor | None = None,
        target_secured_ids: torch.Tensor | None = None,
        recent_placement_ready_ids: torch.Tensor | None = None,
        target_placement_ready_ids: torch.Tensor | None = None,
        **_: Any,
    ) -> torch.Tensor:
        bsz, timesteps = states.shape[:2]
        context = self._encode_context_tokens(
            images,
            states,
            actions,
            language_ids=language_ids,
            phase_ids=recent_phase_ids,
            secured_ids=recent_secured_ids,
            placement_ready_ids=recent_placement_ready_ids,
        )
        context = context + self.context_pos_embedding[:, :timesteps]
        context_mask = recent_mask.to(torch.bool) if recent_mask is not None else torch.ones(bsz, timesteps, dtype=torch.bool, device=states.device)

        if self.use_event_memory and older_obs is not None and older_states is not None:
            older_context = self._encode_context_tokens(
                older_obs,
                older_states,
                older_actions,
                language_ids=language_ids,
            )
            if older_mask is None:
                older_mask = torch.ones(older_context.shape[:2], dtype=torch.bool, device=older_context.device)
            memory_tokens, memory_mask = self._event_memory_tokens(older_context, older_mask.to(torch.bool))
            memory_tokens = memory_tokens + self.memory_pos_embedding[:, : memory_tokens.size(1)]
            context = torch.cat([memory_tokens, context], dim=1)
            context_mask = torch.cat([memory_mask, context_mask], dim=1)
        memory = self.encoder(context, src_key_padding_mask=~context_mask)

        queries = self.query_embedding.expand(bsz, -1, -1) + self.query_pos_embedding + self.query_type_embedding
        if self.language_embedding is not None:
            queries = queries + self.language_embedding(language_ids).unsqueeze(1)
        if self.phase_embedding is not None:
            if target_phase_ids is None:
                target_phase_ids = torch.zeros(bsz, self.T_action, dtype=torch.long, device=states.device)
            target_phase_ids = target_phase_ids.clamp(0, self.phase_embedding.num_embeddings - 1)
            queries = queries + self.phase_embedding(target_phase_ids)
        if self.use_object_signals:
            if target_secured_ids is None:
                target_secured_ids = torch.zeros(bsz, self.T_action, dtype=torch.long, device=states.device)
            if target_placement_ready_ids is None:
                target_placement_ready_ids = torch.zeros(bsz, self.T_action, dtype=torch.long, device=states.device)
            queries = queries + self.secured_embedding(target_secured_ids.clamp(0, 1))
            queries = queries + self.placement_ready_embedding(target_placement_ready_ids.clamp(0, 1))
        decoded = self.decoder(queries, memory)
        return self.action_head(decoded)


class DiffusionPolicyBaseline(nn.Module):
    """Small conditional diffusion policy over action chunks.

    This keeps the observation encoder close to ACT, but replaces direct action
    regression with denoising over a full action chunk.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        T_obs: int,
        T_action: int,
        d_model: int = 192,
        n_layers: int = 3,
        n_heads: int = 4,
        dropout: float = 0.1,
        pretrained_vision: bool = True,
        vision_encoder: str = "resnet18",
        state_hidden_dim: int = 128,
        action_hidden_dim: int = 192,
        use_action_history: bool = True,
        use_language: bool = True,
        language_vocab_size: int = 1024,
        diffusion_steps: int = 32,
        sample_stochastic: bool = False,
        sample_init: str = "zeros",
    ) -> None:
        super().__init__()
        if vision_encoder != "resnet18":
            raise ValueError("DiffusionPolicyBaseline currently supports vision_encoder=resnet18.")
        self.T_obs = T_obs
        self.T_action = T_action
        self.action_dim = action_dim
        self.use_action_history = use_action_history
        self.use_language = use_language
        self.diffusion_steps = int(diffusion_steps)
        if sample_init not in {"zeros", "normal"}:
            raise ValueError("sample_init must be 'zeros' or 'normal'.")
        self.sample_stochastic = sample_stochastic
        self.sample_init = sample_init

        self.vision, vision_dim = _build_resnet18(pretrained_vision)
        self.vision_proj = nn.Linear(vision_dim, d_model)
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, state_hidden_dim),
            nn.ReLU(),
            nn.Linear(state_hidden_dim, d_model),
        )
        self.action_encoder = nn.Sequential(nn.Linear(action_dim, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.language_embedding = nn.Embedding(language_vocab_size, d_model) if self.use_language else None
        self.context_fusion = nn.Linear(d_model * (3 if self.use_action_history else 2), d_model)
        self.context_pos_embedding = nn.Parameter(torch.zeros(1, T_obs, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.noisy_action_proj = nn.Linear(action_dim, d_model)
        self.action_pos_embedding = nn.Parameter(torch.zeros(1, T_action, d_model))
        self.diffusion_time_embedding = nn.Embedding(self.diffusion_steps, d_model)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.denoiser = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        self.noise_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, action_hidden_dim),
            nn.GELU(),
            nn.Linear(action_hidden_dim, action_dim),
        )

        betas = torch.linspace(1e-4, 2e-2, self.diffusion_steps)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("sqrt_alpha_bars", torch.sqrt(alpha_bars))
        self.register_buffer("sqrt_one_minus_alpha_bars", torch.sqrt(1.0 - alpha_bars))

    def _encode_context(
        self,
        images: torch.Tensor,
        states: torch.Tensor,
        actions: torch.Tensor | None = None,
        language_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bsz, timesteps, channels, height, width = images.shape
        flat_images = images.view(bsz * timesteps, channels, height, width)
        if flat_images.is_cuda:
            flat_images = flat_images.contiguous(memory_format=torch.channels_last)
        pieces = [
            self.vision_proj(self.vision(flat_images)).view(bsz, timesteps, -1),
            self.state_encoder(states),
        ]
        if self.use_action_history:
            if actions is None:
                actions = torch.zeros(bsz, timesteps, self.action_dim, dtype=states.dtype, device=states.device)
            pieces.append(self.action_encoder(actions))
        context = self.context_fusion(torch.cat(pieces, dim=-1))
        if self.language_embedding is not None:
            if language_ids is None:
                language_ids = torch.zeros(bsz, dtype=torch.long, device=states.device)
            context = context + self.language_embedding(language_ids).unsqueeze(1)
        context = context + self.context_pos_embedding[:, :timesteps]
        return self.encoder(context)

    def _predict_noise(self, memory: torch.Tensor, noisy_actions: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        tokens = self.noisy_action_proj(noisy_actions)
        tokens = tokens + self.action_pos_embedding[:, : noisy_actions.size(1)]
        tokens = tokens + self.diffusion_time_embedding(timesteps).unsqueeze(1)
        return self.noise_head(self.denoiser(tokens, memory))

    def diffusion_loss(
        self,
        images: torch.Tensor,
        states: torch.Tensor,
        actions: torch.Tensor | None,
        target_actions: torch.Tensor,
        target_mask: torch.Tensor,
        language_ids: torch.Tensor | None = None,
        gripper_transition: torch.Tensor | None = None,
        placement_window: torch.Tensor | None = None,
        gripper_weight: float = 1.0,
        placement_weight: float = 1.0,
        **_: Any,
    ) -> torch.Tensor:
        del gripper_transition
        bsz = target_actions.size(0)
        memory = self._encode_context(images, states, actions, language_ids)
        timesteps = torch.randint(0, self.diffusion_steps, (bsz,), device=target_actions.device)
        noise = torch.randn_like(target_actions)
        view_shape = (bsz,) + (1,) * (target_actions.ndim - 1)
        noisy = (
            self.sqrt_alpha_bars[timesteps].view(view_shape) * target_actions
            + self.sqrt_one_minus_alpha_bars[timesteps].view(view_shape) * noise
        )
        pred_noise = self._predict_noise(memory, noisy, timesteps)

        loss = (pred_noise - noise).pow(2)
        dim_weights = torch.ones(target_actions.size(-1), dtype=loss.dtype, device=loss.device)
        if target_actions.size(-1) >= 7 and gripper_weight != 1.0:
            dim_weights[-1] = gripper_weight
        step_weights = target_mask.to(loss.dtype)
        if placement_window is not None and placement_weight != 1.0:
            placement = placement_window.to(device=loss.device, dtype=loss.dtype)
            step_weights = torch.where(placement > 0, torch.full_like(step_weights, placement_weight), step_weights)
        weights = step_weights.unsqueeze(-1) * dim_weights.view(1, 1, -1)
        return (loss * weights).sum() / weights.sum().clamp_min(1.0)

    @torch.no_grad()
    def sample_actions(
        self,
        images: torch.Tensor,
        states: torch.Tensor,
        actions: torch.Tensor | None = None,
        language_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        memory = self._encode_context(images, states, actions, language_ids)
        if self.sample_init == "normal":
            sample = torch.randn(images.size(0), self.T_action, self.action_dim, dtype=states.dtype, device=states.device)
        else:
            sample = torch.zeros(images.size(0), self.T_action, self.action_dim, dtype=states.dtype, device=states.device)
        for step in range(self.diffusion_steps - 1, -1, -1):
            timestep = torch.full((images.size(0),), step, dtype=torch.long, device=states.device)
            pred_noise = self._predict_noise(memory, sample, timestep)
            alpha = self.alphas[step]
            alpha_bar = self.alpha_bars[step]
            beta = self.betas[step]
            sample = (sample - beta / torch.sqrt(1.0 - alpha_bar) * pred_noise) / torch.sqrt(alpha)
            if step > 0 and self.sample_stochastic:
                sample = sample + torch.sqrt(beta) * torch.randn_like(sample)
        return sample

    def forward(
        self,
        images: torch.Tensor,
        states: torch.Tensor,
        actions: torch.Tensor | None = None,
        language_ids: torch.Tensor | None = None,
        **_: Any,
    ) -> torch.Tensor:
        return self.sample_actions(images, states, actions, language_ids)


class EventGatedMemoryVLA(nn.Module):
    """Long-horizon policy with cheap event-gated summaries over older context."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        T_action: int,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 4,
        dropout: float = 0.1,
        pretrained_vision: bool = True,
        chunk_size: int = 8,
        max_memory_tokens: int = 8,
        gate_type: str = "event",
        query_type: str = "concat",
        action_hidden_dim: int = 256,
        use_language: bool = False,
        language_vocab_size: int = 1024,
    ) -> None:
        super().__init__()
        if gate_type not in {"event", "age_based"}:
            raise ValueError("gate_type must be 'event' or 'age_based'.")
        if query_type not in {"concat", "cross_attention"}:
            raise ValueError("query_type must be 'concat' or 'cross_attention'.")
        self.T_action = T_action
        self.action_dim = action_dim
        self.chunk_size = int(chunk_size)
        self.max_memory_tokens = int(max_memory_tokens)
        self.gate_type = gate_type
        self.query_type = query_type
        self.use_language = use_language

        self.vision, vision_dim = _build_resnet18(pretrained_vision)
        self.vision_proj = nn.Linear(vision_dim, d_model)
        self.action_encoder = nn.Sequential(nn.Linear(action_dim, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.state_encoder = nn.Sequential(nn.Linear(state_dim, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.language_embedding = nn.Embedding(language_vocab_size, d_model) if self.use_language else None
        self.step_fusion = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.pool_query = nn.Parameter(torch.randn(d_model) * 0.02)
        self.gate = nn.Sequential(
            nn.Linear(d_model * 4 + 1, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.memory_attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.action_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, action_hidden_dim),
            nn.GELU(),
            nn.Linear(action_hidden_dim, T_action * action_dim),
        )

    def _encode_steps(self, obs: torch.Tensor, actions: torch.Tensor, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, timesteps, channels, height, width = obs.shape
        flat_obs = obs.view(bsz * timesteps, channels, height, width)
        if flat_obs.is_cuda:
            flat_obs = flat_obs.contiguous(memory_format=torch.channels_last)
        visual = self.vision_proj(self.vision(flat_obs)).view(bsz, timesteps, -1)
        action = self.action_encoder(actions)
        state = self.state_encoder(states)
        step = self.step_fusion(torch.cat([visual, action, state], dim=-1))
        return step, visual, action, state

    def _chunk_memory(
        self,
        tokens: torch.Tensor,
        visual: torch.Tensor,
        action: torch.Tensor,
        state: torch.Tensor,
        mask: torch.Tensor,
        language_token: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, timesteps, dim = tokens.shape
        chunk_count = max((timesteps + self.chunk_size - 1) // self.chunk_size, 1)
        padded_steps = chunk_count * self.chunk_size
        pad = padded_steps - timesteps
        if pad:
            tokens = torch.cat([tokens, tokens.new_zeros(bsz, pad, dim)], dim=1)
            visual = torch.cat([visual, visual.new_zeros(bsz, pad, dim)], dim=1)
            action = torch.cat([action, action.new_zeros(bsz, pad, dim)], dim=1)
            state = torch.cat([state, state.new_zeros(bsz, pad, dim)], dim=1)
            mask = torch.cat([mask, torch.zeros(bsz, pad, dtype=torch.bool, device=mask.device)], dim=1)

        view_shape = (bsz, chunk_count, self.chunk_size, dim)
        chunk_tokens = tokens.view(view_shape)
        chunk_visual = visual.view(view_shape)
        chunk_action = action.view(view_shape)
        chunk_state = state.view(view_shape)
        chunk_mask = mask.view(bsz, chunk_count, self.chunk_size)

        attn_logits = torch.einsum("bcsd,d->bcs", chunk_tokens, self.pool_query)
        attn_logits = attn_logits.masked_fill(~chunk_mask, -1e4)
        attn = torch.softmax(attn_logits, dim=-1) * chunk_mask.to(tokens.dtype)
        attn = attn / attn.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        summaries = torch.einsum("bcs,bcsd->bcd", attn, chunk_tokens)

        pair_mask = chunk_mask[:, :, 1:] & chunk_mask[:, :, :-1]
        pair_counts = pair_mask.sum(dim=-1).clamp_min(1).to(tokens.dtype).unsqueeze(-1)
        pair_mask_f = pair_mask.to(tokens.dtype).unsqueeze(-1)
        delta_visual = (chunk_visual.diff(dim=2).abs() * pair_mask_f).sum(dim=2) / pair_counts
        delta_action = (chunk_action.diff(dim=2).abs() * pair_mask_f).sum(dim=2) / pair_counts
        delta_state = (chunk_state.diff(dim=2).abs() * pair_mask_f).sum(dim=2) / pair_counts
        age = torch.linspace(0.0, 1.0, chunk_count, device=tokens.device).view(1, chunk_count, 1).expand(bsz, -1, -1)
        if self.gate_type == "age_based":
            gate_score = age.clamp_min(1.0 / max(chunk_count, 1))
        else:
            if language_token is None:
                language_features = delta_visual.new_zeros(bsz, chunk_count, dim)
            else:
                language_features = language_token.expand(-1, chunk_count, -1)
            gate_input = torch.cat([delta_visual, delta_action, delta_state, language_features, age], dim=-1)
            gate_score = torch.sigmoid(self.gate(gate_input))
        memory_tokens = summaries * gate_score
        memory_mask = chunk_mask.any(dim=-1)
        if memory_tokens.size(1) > self.max_memory_tokens:
            memory_tokens = memory_tokens[:, -self.max_memory_tokens :]
            memory_mask = memory_mask[:, -self.max_memory_tokens :]
        return memory_tokens, memory_mask

    def forward(
        self,
        recent_obs: torch.Tensor,
        recent_actions: torch.Tensor,
        recent_states: torch.Tensor,
        older_obs: torch.Tensor,
        older_actions: torch.Tensor,
        older_states: torch.Tensor,
        recent_mask: torch.Tensor | None = None,
        older_mask: torch.Tensor | None = None,
        language_ids: torch.Tensor | None = None,
        **_: Any,
    ) -> torch.Tensor:
        recent_tokens, _, _, _ = self._encode_steps(recent_obs, recent_actions, recent_states)
        older_tokens, older_visual, older_action, older_state = self._encode_steps(older_obs, older_actions, older_states)
        language_token = None
        if self.language_embedding is not None:
            if language_ids is None:
                language_ids = torch.zeros(recent_obs.size(0), dtype=torch.long, device=recent_obs.device)
            language_token = self.language_embedding(language_ids).unsqueeze(1)
            recent_tokens = recent_tokens + language_token
            older_tokens = older_tokens + language_token
        if recent_mask is None:
            recent_mask = torch.ones(recent_tokens.shape[:2], dtype=torch.bool, device=recent_tokens.device)
        if older_mask is None:
            older_mask = torch.ones(older_tokens.shape[:2], dtype=torch.bool, device=older_tokens.device)

        memory_tokens, memory_mask = self._chunk_memory(
            older_tokens,
            older_visual,
            older_action,
            older_state,
            older_mask,
            language_token=language_token,
        )
        empty_memory = ~memory_mask.any(dim=1)
        if empty_memory.any():
            memory_mask = memory_mask.clone()
            memory_tokens = memory_tokens.clone()
            memory_mask[empty_memory, 0] = True
            memory_tokens[empty_memory, 0] = 0
        if self.query_type == "cross_attention":
            recent_encoded = self.transformer(recent_tokens, src_key_padding_mask=~recent_mask)
            query = recent_encoded[:, -1:].contiguous()
            attended, _ = self.memory_attention(query, memory_tokens, memory_tokens, key_padding_mask=~memory_mask)
            pooled = query.squeeze(1) + attended.squeeze(1)
        else:
            tokens = torch.cat([memory_tokens, recent_tokens], dim=1)
            mask = torch.cat([memory_mask, recent_mask], dim=1)
            encoded = self.transformer(tokens, src_key_padding_mask=~mask)
            pooled = encoded[:, -1]
        return self.action_head(pooled).view(recent_obs.size(0), self.T_action, self.action_dim)


class OctoBaseline(nn.Module):
    """Adapter placeholder for evaluating an open-source Octo checkpoint.

    Octo models are normally distributed through the Octo/JAX ecosystem rather
    than as a small PyTorch module. This wrapper makes the intended evaluation
    surface explicit without pretending weights are locally available.
    """

    def __init__(self, checkpoint: str | None = None) -> None:
        super().__init__()
        self.checkpoint = checkpoint

    def forward(self, *_: Any, **__: Any) -> torch.Tensor:
        raise NotImplementedError(
            "OctoBaseline requires installing the official Octo inference stack and providing a checkpoint. "
            "Use BaselineVLA or BehaviorCloningAgent for the local PyTorch baselines."
        )


def build_model(config: dict[str, Any], state_dim: int, action_dim: int) -> nn.Module:
    model_cfg = config["model"]
    data_cfg = config["data"]
    baseline = model_cfg.get("baseline", "sliding_window")
    T_obs = int(data_cfg.get("K_recent", data_cfg.get("T_obs", 4))) if data_cfg.get("source") == "episode" else int(data_cfg["T_obs"])
    if baseline == "no_temporal":
        T_obs = 1
    if baseline == "larger_window":
        T_obs = max(T_obs, int(data_cfg["T_obs"]) * 2)
    T_action = int(data_cfg.get("H_action", data_cfg.get("T_action", 16)))

    if baseline == "bc_resnet50":
        return BehaviorCloningAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            T_action=T_action,
            pretrained_vision=bool(model_cfg.get("pretrained_vision", True)),
            hidden_dim=int(model_cfg.get("action_hidden_dim", 256)),
        )
    if baseline == "octo":
        return OctoBaseline(model_cfg.get("octo_checkpoint"))
    if baseline == "rt1_style":
        return RT1StyleBaseline(
            state_dim=state_dim,
            action_dim=action_dim,
            T_obs=T_obs,
            T_action=T_action,
            d_model=int(model_cfg.get("d_model", 256)),
            n_layers=int(model_cfg.get("n_layers", 4)),
            n_heads=int(model_cfg.get("n_heads", 4)),
            dropout=float(model_cfg.get("dropout", 0.1)),
            pretrained_vision=bool(model_cfg.get("pretrained_vision", True)),
            action_hidden_dim=int(model_cfg.get("action_hidden_dim", 256)),
        )
    if baseline in {"act_chunked", "event_gated_act"}:
        memory_cfg = config.get("memory", {})
        return ACTChunkedBaseline(
            state_dim=state_dim,
            action_dim=action_dim,
            T_obs=T_obs,
            T_action=T_action,
            d_model=int(model_cfg.get("d_model", 256)),
            n_layers=int(model_cfg.get("n_layers", 4)),
            n_heads=int(model_cfg.get("n_heads", 4)),
            dropout=float(model_cfg.get("dropout", 0.1)),
            pretrained_vision=bool(model_cfg.get("pretrained_vision", True)),
            vision_encoder=model_cfg.get("vision_encoder", "resnet18"),
            state_hidden_dim=int(model_cfg.get("state_hidden_dim", 128)),
            action_hidden_dim=int(model_cfg.get("action_hidden_dim", 256)),
            use_action_history=bool(model_cfg.get("use_action_history", True)),
            use_language=bool(model_cfg.get("use_language", True)),
            language_vocab_size=int(model_cfg.get("language_vocab_size", 1024)),
            input_modalities=str(model_cfg.get("input_modalities", "vision_state_action")),
            use_phase=bool(model_cfg.get("use_phase", False)),
            phase_vocab_size=int(model_cfg.get("phase_vocab_size", 4)),
            use_object_signals=bool(model_cfg.get("use_object_signals", False)),
            use_event_memory=baseline == "event_gated_act",
            memory_chunk_size=int(memory_cfg.get("chunk_size", 8)),
            max_memory_tokens=int(memory_cfg.get("max_memory_tokens", 8)),
            gate_type=memory_cfg.get("gate_type", "event"),
        )
    if baseline == "diffusion_policy":
        return DiffusionPolicyBaseline(
            state_dim=state_dim,
            action_dim=action_dim,
            T_obs=T_obs,
            T_action=T_action,
            d_model=int(model_cfg.get("d_model", 192)),
            n_layers=int(model_cfg.get("n_layers", 3)),
            n_heads=int(model_cfg.get("n_heads", 4)),
            dropout=float(model_cfg.get("dropout", 0.1)),
            pretrained_vision=bool(model_cfg.get("pretrained_vision", True)),
            vision_encoder=model_cfg.get("vision_encoder", "resnet18"),
            state_hidden_dim=int(model_cfg.get("state_hidden_dim", 128)),
            action_hidden_dim=int(model_cfg.get("action_hidden_dim", 192)),
            use_action_history=bool(model_cfg.get("use_action_history", True)),
            use_language=bool(model_cfg.get("use_language", True)),
            language_vocab_size=int(model_cfg.get("language_vocab_size", 1024)),
            diffusion_steps=int(model_cfg.get("diffusion_steps", 32)),
            sample_stochastic=bool(model_cfg.get("sample_stochastic", False)),
            sample_init=str(model_cfg.get("sample_init", "zeros")),
        )
    if baseline == "event_gated_memory":
        memory_cfg = config.get("memory", {})
        return EventGatedMemoryVLA(
            state_dim=state_dim,
            action_dim=action_dim,
            T_action=T_action,
            d_model=int(model_cfg.get("d_model", 256)),
            n_layers=int(model_cfg.get("n_layers", 4)),
            n_heads=int(model_cfg.get("n_heads", 4)),
            dropout=float(model_cfg.get("dropout", 0.1)),
            pretrained_vision=bool(model_cfg.get("pretrained_vision", True)),
            chunk_size=int(memory_cfg.get("chunk_size", 8)),
            max_memory_tokens=int(memory_cfg.get("max_memory_tokens", 8)),
            gate_type=memory_cfg.get("gate_type", "event"),
            query_type=memory_cfg.get("query_type", "concat"),
            action_hidden_dim=int(model_cfg.get("action_hidden_dim", 256)),
            use_language=bool(model_cfg.get("use_language", False)),
            language_vocab_size=int(model_cfg.get("language_vocab_size", 1024)),
        )
    if baseline not in {"sliding_window", "no_temporal", "larger_window"}:
        raise ValueError(f"Unknown baseline: {baseline}")
    return BaselineVLA(
        state_dim=state_dim,
        action_dim=action_dim,
        T_obs=T_obs,
        T_action=T_action,
        d_model=int(model_cfg.get("d_model", 256)),
        n_layers=int(model_cfg.get("n_layers", 4)),
        n_heads=int(model_cfg.get("n_heads", 4)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        pretrained_vision=bool(model_cfg.get("pretrained_vision", True)),
        vision_encoder=model_cfg.get("vision_encoder", "resnet18"),
        use_memory=bool(model_cfg.get("use_memory", False)),
        state_hidden_dim=int(model_cfg.get("state_hidden_dim", 128)),
        action_hidden_dim=int(model_cfg.get("action_hidden_dim", 256)),
        use_action_history=bool(model_cfg.get("use_action_history", False)),
        use_language=bool(model_cfg.get("use_language", False)),
        language_vocab_size=int(model_cfg.get("language_vocab_size", 1024)),
    )
