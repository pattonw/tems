import math
import warnings
from collections.abc import Sequence
from typing import Any

import torch
from torch.nn import Conv1d, Conv2d, Conv3d
from torch.nn.modules.conv import _ConvNd

from .tem import ContextAwareModule
from .utils import crop


class ConvPass(ContextAwareModule):
    """
    The ConvPass class wraps a series of convolutional layers with a limited set
    of arguments. It is designed to be used with the UNet class and provides satisfies
    the `ContextAwareModule` interface.

    :param dims: the number of dimensions (1, 2, or 3)
    :param in_channels: the number of input channels
    :param out_channels: the number of output channels
    :param kernel_sizes: a sequence of kernel sizes for each convolutional layer
    :param activation: the activation function to use after each convolutional layer
    :param padding: the padding mode to use for the convolutional layers
    :param residual: add residual connection from input to output
    :param initialize_weights: init conv weights based on activation
    """

    def __init__(
        self,
        dims: int,
        in_channels: int = 1,
        out_channels: int = 1,
        kernel_sizes: Sequence[Sequence[int] | int] = (3, 3),
        activation: type[torch.nn.Module] | None = torch.nn.ReLU,
        padding: str = "valid",
        residual: bool = False,
        initialize_weights: bool = True,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.residual = residual

        self._dims = dims
        self._context = torch.tensor((0,) * dims)
        self._equivariant_step = torch.tensor((1,) * dims)

        layers: list[torch.nn.Module] = []

        conv: Any = {
            1: Conv1d,
            2: Conv2d,
            3: Conv3d,
        }[dims]

        for kernel_size in kernel_sizes:
            if isinstance(kernel_size, int):
                _kernel_size = (kernel_size,) * dims
            else:
                _kernel_size = tuple(kernel_size)

            conv_layer = conv(
                in_channels,
                out_channels,
                _kernel_size,
                padding=padding,
            )

            if initialize_weights:
                self._initialize_conv(
                    conv_layer=conv_layer,
                    kernel_size=_kernel_size,
                    in_channels=in_channels,
                    activation=activation,
                )

            layers.append(conv_layer)

            if padding == "valid":
                self._context += torch.tensor(_kernel_size) - 1

            in_channels = out_channels

            if activation is not None:
                layers.append(activation())

        self.conv_pass = torch.nn.Sequential(*layers)

        self.residual_layer = (
            torch.nn.Identity()
            if (not residual or self.in_channels == self.out_channels)
            else conv(
                self.in_channels,
                self.out_channels,
                kernel_size=1,
                padding=0,
            )
        )

    def _initialize_conv(
        self,
        conv_layer: _ConvNd,
        kernel_size: Sequence[int],
        in_channels: int,
        activation: type[torch.nn.Module] | None,
    ) -> None:
        assert conv_layer.bias is not None
        torch.nn.init.zeros_(conv_layer.bias)

        if activation is torch.nn.ReLU:
            torch.nn.init.kaiming_uniform_(
                conv_layer.weight,
                nonlinearity="relu",
            )
        elif activation is torch.nn.LeakyReLU:
            torch.nn.init.kaiming_uniform_(
                conv_layer.weight,
                nonlinearity="leaky_relu",
            )
        elif activation is torch.nn.Sigmoid or activation is torch.nn.Tanh:
            torch.nn.init.xavier_uniform_(conv_layer.weight)
        elif activation is torch.nn.Identity:
            warnings.warn(
                "Using Identity activation with the ConvPass module is assumed "
                "to be a test case. The convolutional layer will be initialized "
                "with constants.",
                stacklevel=2,
            )
            constant = 1.0 / math.prod(kernel_size) / in_channels
            torch.nn.init.constant_(conv_layer.weight, constant)

    @property
    def context(self) -> torch.Tensor:
        """
        The context of the ConvPass module.
        Set to zero if the padding is "same".
        Set to the sum of the [kernel_size - 1 for kernel_size in kernel_sizes]
        if the padding is "valid".
        """
        return self._context

    @property
    def equivariant_step(self) -> torch.Tensor:
        """
        Always 1 for ConvPass since we don't yet support strided convolutions.
        """
        return self._equivariant_step

    @property
    def min_input_shape(self) -> torch.Tensor:
        """
        Simply 1 + `self.context`.
        """
        return self.min_output_shape + self._context

    @property
    def min_output_shape(self) -> torch.Tensor:
        """
        Always 1
        """
        return torch.tensor((1,) * self.dims)

    @property
    def dims(self) -> int:
        """
        The number of dimensions of the ConvPass module.
        """
        return self._dims

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply a series of convolutions to the input tensor.

        :param x: the input tensor
        """
        y = self.conv_pass(x)

        if self.residual:
            residual = self.residual_layer(x)
            residual = crop(
                residual,
                torch.tensor(y.shape[2:], device=y.device),
            )
            return y + residual

        return y
