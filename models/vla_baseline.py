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
    ) -> None:
        super().__init__()
        self.T_obs = T_obs
        self.T_action = T_action
        self.action_dim = action_dim
        if vision_encoder != "resnet18":
            raise ValueError("BaselineVLA currently supports vision_encoder=resnet18.")
        self.vision, vision_dim = _build_resnet18(pretrained_vision)
        self.vision_proj = nn.Linear(vision_dim, d_model)
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, state_hidden_dim),
            nn.ReLU(),
            nn.Linear(state_hidden_dim, d_model),
        )
        self.fusion = nn.Linear(d_model * 2, d_model)
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
        past_memory: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        bsz, timesteps, channels, height, width = images.shape
        flat_images = images.view(bsz * timesteps, channels, height, width)
        if flat_images.is_cuda:
            flat_images = flat_images.contiguous(memory_format=torch.channels_last)
        vision_tokens = self.vision_proj(self.vision(flat_images)).view(bsz, timesteps, -1)
        state_tokens = self.state_encoder(states)
        tokens = self.fusion(torch.cat([vision_tokens, state_tokens], dim=-1))
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
    T_obs = 1 if baseline == "no_temporal" else int(data_cfg["T_obs"])
    if baseline == "larger_window":
        T_obs = max(T_obs, int(data_cfg["T_obs"]) * 2)

    if baseline == "bc_resnet50":
        return BehaviorCloningAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            T_action=int(data_cfg["T_action"]),
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
            T_action=int(data_cfg["T_action"]),
            d_model=int(model_cfg.get("d_model", 256)),
            n_layers=int(model_cfg.get("n_layers", 4)),
            n_heads=int(model_cfg.get("n_heads", 4)),
            dropout=float(model_cfg.get("dropout", 0.1)),
            pretrained_vision=bool(model_cfg.get("pretrained_vision", True)),
            action_hidden_dim=int(model_cfg.get("action_hidden_dim", 256)),
        )
    if baseline not in {"sliding_window", "no_temporal", "larger_window"}:
        raise ValueError(f"Unknown baseline: {baseline}")
    return BaselineVLA(
        state_dim=state_dim,
        action_dim=action_dim,
        T_obs=T_obs,
        T_action=int(data_cfg["T_action"]),
        d_model=int(model_cfg.get("d_model", 256)),
        n_layers=int(model_cfg.get("n_layers", 4)),
        n_heads=int(model_cfg.get("n_heads", 4)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        pretrained_vision=bool(model_cfg.get("pretrained_vision", True)),
        vision_encoder=model_cfg.get("vision_encoder", "resnet18"),
        use_memory=bool(model_cfg.get("use_memory", False)),
        state_hidden_dim=int(model_cfg.get("state_hidden_dim", 128)),
        action_hidden_dim=int(model_cfg.get("action_hidden_dim", 256)),
    )
