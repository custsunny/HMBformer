from typing import Tuple
import torch
import torch.nn as nn
from wenet.utils.class_utils import WENET_ACTIVATION_CLASSES


class ConvolutionalSpatialGatingUnit(torch.nn.Module):
    """Convolutional Spatial Gating Unit (CSGU)."""

    def __init__(
            self,
            size: int,
            kernel_size: int,
            dropout_rate: float,
            use_linear_after_conv: bool,
            gate_activation: str,
            causal: bool = True,
    ):
        super().__init__()

        n_channels = size // 2
        self.norm = nn.LayerNorm(n_channels)

        if causal:
            padding = 0
            self.lorder = kernel_size - 1
        else:
            assert (kernel_size - 1) % 2 == 0
            padding = (kernel_size - 1) // 2
            self.lorder = 0

        self.conv = torch.nn.Conv1d(
            n_channels,
            n_channels,
            kernel_size,
            1,
            padding,
            groups=n_channels,
        )
        if use_linear_after_conv:
            self.linear = torch.nn.Linear(n_channels, n_channels)
        else:
            self.linear = None

        if gate_activation == "identity":
            self.act = torch.nn.Identity()
        else:
            self.act = WENET_ACTIVATION_CLASSES[gate_activation]()

        self.dropout = torch.nn.Dropout(dropout_rate)
        self.espnet_initialization_fn()

    def espnet_initialization_fn(self):
        torch.nn.init.normal_(self.conv.weight, std=1e-3)
        torch.nn.init.ones_(self.conv.bias)
        if self.linear is not None:
            torch.nn.init.normal_(self.linear.weight, std=1e-3)
            torch.nn.init.ones_(self.linear.bias)

    def forward(
            self, x: torch.Tensor, x_a: torch.Tensor, cache: torch.Tensor = torch.zeros((0, 0, 0))
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        x_r, x_g = x.chunk(2, dim=-1)
        x_g = x_g.transpose(1, 2)

        if self.lorder > 0:
            if cache.size(2) == 0:
                x_g = nn.functional.pad(x_g, (self.lorder, 0), 'constant', 0.0)
            else:
                assert cache.size(0) == x_g.size(0)
                assert cache.size(1) == x_g.size(1)
                x_g = torch.cat((cache, x_g), dim=2)
            assert (x_g.size(2) > self.lorder)
            new_cache = x_g[:, :, -self.lorder:]
        else:
            new_cache = torch.zeros((0, 0, 0), dtype=x_g.dtype, device=x_g.device)

        x_g = x_g.transpose(1, 2)
        x_g = self.norm(x_g)
        x_g = self.conv(x_g.transpose(1, 2)).transpose(1, 2)
        if self.linear is not None:
            x_g = self.linear(x_g)

        x_g = self.act(x_g)
        x_g = x_g.transpose(1, 2)
        x_a = x_a.transpose(1, 2)
        x_g = torch.cat((x_g, x_a), dim=1)
        x_g = nn.functional.glu(x_g, dim=1)
        x_g = x_g.transpose(1, 2)

        out = x_r * x_g
        out = self.dropout(out)

        # 数值检查
        if torch.isnan(out).any() or torch.isinf(out).any():
            raise ValueError("NaN or Inf found in output tensor.")

        return out, new_cache


class ConvolutionalSpatialGatingUnitNormal(torch.nn.Module):
    """Convolutional Spatial Gating Unit (CSGU)."""

    def __init__(
            self,
            size: int,
            kernel_size: int,
            dropout_rate: float,
            use_linear_after_conv: bool,
            gate_activation: str,
            causal: bool = True,
    ):
        super().__init__()

        n_channels = size // 2
        self.norm = nn.LayerNorm(n_channels)

        if causal:
            padding = 0
            self.lorder = kernel_size - 1
        else:
            assert (kernel_size - 1) % 2 == 0
            padding = (kernel_size - 1) // 2
            self.lorder = 0

        self.conv = torch.nn.Conv1d(
            n_channels,
            n_channels,
            kernel_size,
            1,
            padding,
            groups=n_channels,
        )
        if use_linear_after_conv:
            self.linear = torch.nn.Linear(n_channels, n_channels)
        else:
            self.linear = None

        if gate_activation == "identity":
            self.act = torch.nn.Identity()
        else:
            self.act = WENET_ACTIVATION_CLASSES[gate_activation]()

        self.dropout = torch.nn.Dropout(dropout_rate)
        self.espnet_initialization_fn()

    def espnet_initialization_fn(self):
        torch.nn.init.normal_(self.conv.weight, std=1e-3)
        torch.nn.init.ones_(self.conv.bias)
        if self.linear is not None:
            torch.nn.init.normal_(self.linear.weight, std=1e-3)
            torch.nn.init.ones_(self.linear.bias)

    def forward(
            self, x: torch.Tensor, cache: torch.Tensor = torch.zeros((0, 0, 0))
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        x_r, x_g = x.chunk(2, dim=-1)
        x_g = x_g.transpose(1, 2)

        if self.lorder > 0:
            if cache.size(2) == 0:
                x_g = nn.functional.pad(x_g, (self.lorder, 0), 'constant', 0.0)
            else:
                assert cache.size(0) == x_g.size(0)
                assert cache.size(1) == x_g.size(1)
                x_g = torch.cat((cache, x_g), dim=2)
            assert (x_g.size(2) > self.lorder)
            new_cache = x_g[:, :, -self.lorder:]
        else:
            new_cache = torch.zeros((0, 0, 0), dtype=x_g.dtype, device=x_g.device)

        x_g = x_g.transpose(1, 2)
        x_g = self.norm(x_g)
        x_g = self.conv(x_g.transpose(1, 2)).transpose(1, 2)
        if self.linear is not None:
            x_g = self.linear(x_g)

        x_g = self.act(x_g)
        out = x_r * x_g
        out = self.dropout(out)

        # 数值检查
        if torch.isnan(out).any() or torch.isinf(out).any():
            raise ValueError("NaN or Inf found in output tensor.")

        return out, new_cache


class ConvolutionalGatingMLP(torch.nn.Module):
    """Convolutional Gating MLP (cgMLP)."""

    def __init__(
            self,
            size: int,
            linear_units: int,
            kernel_size: int,
            dropout_rate: float,
            use_linear_after_conv: bool,
            gate_activation: str,
            causal: bool = True,
    ):
        super().__init__()

        self.channel_proj1 = torch.nn.Sequential(
            torch.nn.Linear(size, linear_units), torch.nn.GELU())
        self.csgu = ConvolutionalSpatialGatingUnit(
            size=linear_units,
            kernel_size=kernel_size,
            dropout_rate=dropout_rate,
            use_linear_after_conv=use_linear_after_conv,
            gate_activation=gate_activation,
            causal=causal,
        )
        self.csgu_nor = ConvolutionalSpatialGatingUnitNormal(
            size=linear_units,
            kernel_size=kernel_size,
            dropout_rate=dropout_rate,
            use_linear_after_conv=use_linear_after_conv,
            gate_activation=gate_activation,
            causal=causal,
        )
        self.channel_proj2 = torch.nn.Linear(linear_units // 2, size)
        self.channel_proj3 = torch.nn.Linear(size, linear_units // 2)

    def forward(
            self,
            x: torch.Tensor,
            x1: torch.Tensor,
            mask: torch.Tensor,
            cache: torch.Tensor = torch.zeros((0, 0, 0))
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        xs_pad = x

        xs_pad = self.channel_proj1(xs_pad)
        x1 = self.channel_proj3(x1)

        xs_pad_1, new_cnn_cache = self.csgu(xs_pad, x1, cache)
        xs_pad_n, _ = self.csgu_nor(xs_pad, cache)

        xs_pad_1 = self.channel_proj2(xs_pad_1)
        xs_pad_n = self.channel_proj2(xs_pad_n)

        out = xs_pad_1
        out_n = xs_pad_n

        # 数值检查
        if torch.isnan(out).any() or torch.isinf(out).any():
            raise ValueError("NaN or Inf found in output tensor.")

        return out, out_n, new_cnn_cache
