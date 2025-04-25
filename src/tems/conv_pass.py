from typing import Any, Sequence

import torch
from torch.nn import Conv1d, Conv2d, Conv3d

from .tem import ContextAwareModule


class ConvPass(ContextAwareModule):
    _context: torch.Tensor
    _invariant_step: torch.Tensor
    _dims: int

    def __init__(
        self,
        dims: int,
        in_channels: int = 1,
        out_channels: int = 1,
        kernel_sizes: Sequence[Sequence[int] | int] = (3, 3),
        activation: type[torch.nn.Module] = torch.nn.ReLU,
        padding="valid",
    ):
        super().__init__()

        self._dims = dims
        self._context = torch.tensor((0,) * dims)
        self._invariant_step = torch.tensor((1,) * dims)

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
        return self._context

    @property
    def invariant_step(self) -> torch.Tensor:
        return self._invariant_step

    @property
    def min_input_shape(self) -> torch.Tensor:
        return self.min_output_shape + self._context

    @property
    def min_output_shape(self) -> torch.Tensor:
        return torch.tensor((1,) * self.dims)

    @property
    def dims(self) -> int:
        return self._dims

    def forward(self, x):
        return self.conv_pass(x)
