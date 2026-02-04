from abc import ABC, abstractmethod
from typing import Optional, Dict
import math

import torch
from torch import nn

from synthesisers.binary_diffusion_tabular import TASK


__all__ = ["BaseModel", "SimpleTableGenerator"]


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Residual(nn.Module):
    """Residual layer with timestep embedding."""

    def __init__(
        self, i: int, o: int, time_emb_dim: Optional[int] = None, use_bias: bool = True
    ):
        super(Residual, self).__init__()
        self.fc = nn.Linear(i, o)
        self.bn = nn.BatchNorm1d(o)
        self.relu = nn.ReLU()
        self.use_bias = use_bias

        # Timestep embedding MLP
        if time_emb_dim is not None:
            self.mlp = nn.Sequential(
                nn.SiLU(), nn.Linear(time_emb_dim, o * 2, bias=use_bias)
            )
        else:
            self.mlp = None

    def forward(
        self, x: torch.Tensor, time_emb: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        out = self.fc(x)
        out = self.bn(out)
        out = self.relu(out)

        # Apply timestep embedding if available
        if self.mlp is not None and time_emb is not None:
            time_emb = self.mlp(time_emb)
            scale, shift = time_emb.chunk(2, dim=1)
            out = out * scale + shift

        return torch.cat([out, x], dim=1)


class BaseModel(nn.Module, ABC):

    def __init__(
        self,
        data_dim: int,
        out_dim: int,
    ):
        super().__init__()
        self.data_dim = data_dim
        self.out_dim = out_dim

    @classmethod
    @abstractmethod
    def from_config(cls, config: Dict) -> "BaseModel":
        pass

    @property
    @abstractmethod
    def config(self) -> Dict:
        pass

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        pass


class SimpleTableGenerator(BaseModel):
    """Simple denoiser model for table generation

    Model works with 1d signals of fixed size"""

    def __init__(
        self,
        data_dim: int,
        dim: int,
        n_res_blocks: int,
        out_dim: int,
        task: TASK,
        conditional: bool = False,
        n_classes: int = 0,
        classifier_free_guidance: bool = False,
    ):
        """
        Args:
            data_dim: dimension of data
            dim: internal dimensionality
            n_res_blocks: number of residual blocks
            out_dim: number of output dimensions
            task: task to generate data for. Options: classification, regression
            conditional: if True, generative model is conditional
            n_classes: number of classes for classification
            classifier_free_guidance: if True, classifier free guidance is used during sampling
        """

        if task not in ["classification", "regression"]:
            raise ValueError(f"Invalid task: {task}")

        if task == "classification" and conditional and n_classes <= 0:
            raise ValueError("n_classes must be greater than 0 for classification")

        super(SimpleTableGenerator, self).__init__(data_dim, out_dim)

        self.dim = dim
        self.n_res_blocks = n_res_blocks
        self.n_classes = n_classes
        self.classifier_free_guidance = classifier_free_guidance
        self.conditional = conditional
        self.task = task

        time_dim = dim
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim, bias=True),
            nn.GELU(),
            nn.Linear(time_dim, time_dim, bias=True),
        )

        if self.task == "classification":
            if self.conditional:
                if n_classes > 0:
                    if self.classifier_free_guidance:
                        self.class_emb = nn.Linear(n_classes, dim, bias=True)
                    else:
                        self.class_emb = nn.Embedding(n_classes, dim)
        else:
            if self.conditional:
                # Regression task
                self.cond_emb = nn.Linear(1, dim, bias=True)

        self.data_proj = nn.Linear(self.data_dim, dim, bias=True)

        item = dim
        self.blocks = nn.ModuleList([])
        for _ in range(n_res_blocks):
            self.blocks.append(Residual(dim, item, time_emb_dim=time_dim))
            dim += item

        self.out = nn.Linear(dim, self.out_dim, bias=True)

    @property
    def config(self) -> Dict:
        """Returns model configuration in dictionary

        Returns:
            dict: model configuration with the following keys:
                  data_dim: dimension of data
                  dim: internal dimensionality
                  n_res_blocks: number of residual blocks
                  out_dim: number of output dimensions
                  task: task to generate data for. Options: classification, regression
                  conditional: if True, generative model is conditional
                  n_classes: number of classes for classification
                  classifier_free_guidance: if True, classifier free guidance is used during sampling
        """

        return {
            "data_dim": self.data_dim,
            "dim": self.dim,
            "n_res_blocks": self.n_res_blocks,
            "out_dim": self.out_dim,
            "task": self.task,
            "conditional": self.conditional,
            "n_classes": self.n_classes,
            "classifier_free_guidance": self.classifier_free_guidance,
        }

    @classmethod
    def from_config(cls, config: Dict) -> "SimpleTableGenerator":
        return cls(**config)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """Run denoising step for table generation

        Args:
            x: noisy input data (BS, data_dim)
            t: timesteps (BS,)
            y: conditional input data (BS,) or (BS, n_classes) if classifier free guidance
            *args:
            **kwargs:

        Returns:
            torch.Tensor: denoised data
        """

        t = self.time_mlp(t)

        if y is not None and hasattr(self, "cond_emb"):
            y = self.cond_emb(y)
            t = t + y

        if y is not None and hasattr(self, "class_emb"):
            y = self.class_emb(y)
            t = t + y

        x = self.data_proj(x)

        for block in self.blocks:
            x = block(x, t)
        return self.out(x)
