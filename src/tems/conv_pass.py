from typing import Any, Sequence

import torch
from torch.nn import Conv1d, Conv2d, Conv3d

from .tem import ContextAwareModule


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
    """

    _context: torch.Tensor
    _equivariant_step: torch.Tensor
    _dims: int

    def __init__(
        self,
        dims: int,
        in_channels: int = 1,
        out_channels: int = 1,
        kernel_sizes: Sequence[Sequence[int] | int] = (3, 3),
        activation: type[torch.nn.Module] = torch.nn.ReLU,
        padding: str = "valid",
    ):
        super().__init__()

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
            # if isinstance(kernel_size, int):
            #     _kernel_size = (kernel_size,) * dims
            # else:
            #     _kernel_size = tuple(kernel_size)
            conv_layer = conv(
                in_channels,
                out_channels,
                kernel_size,
                padding=padding,
            )
            # initialize the weights properly
            assert conv_layer.bias is not None
            torch.nn.init.zeros_(conv_layer.bias)
            if activation is torch.nn.ReLU:
                torch.nn.init.kaiming_uniform_(conv_layer.weight, nonlinearity="relu")
            elif activation is torch.nn.LeakyReLU:
                torch.nn.init.kaiming_uniform_(
                    conv_layer.weight, nonlinearity="leaky_relu"
                )
            elif activation is torch.nn.Sigmoid or activation is torch.nn.Tanh:
                torch.nn.init.xavier_uniform_(conv_layer.weight)
            elif activation is torch.nn.Identity:
                torch.nn.init.constant_(conv_layer.weight, 1.0)
            layers.append(conv_layer)

            if padding == "valid":
                self._context += torch.tensor(kernel_size) - 1

            in_channels = out_channels

            if activation is not None:
                layers.append(activation())

        self.conv_pass = torch.nn.Sequential(*layers)

    @property
    def context(self) -> torch.Tensor:
        """
        The context of the ConvPass module.
        Set to zero if the padding is "same".
        Set to the sum of the [kernel_size - 1 for kernel_size in kernel_sizes] if the padding is "valid".
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
        return self.conv_pass(x)
