from typing import Sequence

import torch

from .tem import ContextAwareModule


class Downsample(ContextAwareModule):
    """
    The Downsample class wraps a MaxPool layer with a limited set of arguments.
    It is designed to be used with the UNet class and provides satisfies
    the `ContextAwareModule` interface.

    :param dims: the number of dimensions (1, 2, or 3)
    :param downsample_factor: the downsample factor for each dimension
    """

    _context: torch.Tensor
    _equivariant_step: torch.Tensor
    _dims: int

    def __init__(self, dims: int, downsample_factor: Sequence[int] | int):
        super().__init__()

        self._dims = dims
        self._context = torch.tensor((0,) * dims)

        pool = {
            1: torch.nn.MaxPool1d,
            2: torch.nn.MaxPool2d,
            3: torch.nn.MaxPool3d,
        }[dims]

        if isinstance(downsample_factor, int):
            _downsample_factor = (downsample_factor,) * dims
        else:
            _downsample_factor = tuple(downsample_factor)

        self.down = pool(
            _downsample_factor,
            stride=_downsample_factor,
            padding=(0,) * dims,
        )

        self._equivariant_step = self._context + torch.tensor(_downsample_factor)

    @property
    def context(self) -> torch.Tensor:
        """
        The context is always 0 for downsampling.
        """
        return self._context

    @property
    def equivariant_step(self) -> torch.Tensor:
        """
        The invariant step is the downsample factor for each dimension.
        """
        return self._equivariant_step

    @property
    def min_input_shape(self) -> torch.Tensor:
        """
        `min_input_shape` is equal to the downsample factor for each dimension.
        """
        return self.min_output_shape * self.equivariant_step

    @property
    def min_output_shape(self) -> torch.Tensor:
        """
        `min_output_shape` is just 1 for each dimension.
        """
        return torch.tensor((1,) * self.dims)

    @property
    def dims(self) -> int:
        """
        The number of dimensions (1, 2, or 3).
        """
        return self._dims

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the downsample operation to the input tensor.

        :param x: the input tensor
        """
        for d in range(1, self.dims + 1):
            if x.size()[-d] % self.equivariant_step[-d] != 0:
                raise RuntimeError(
                    "Can not downsample shape %s with factor %s, mismatch "
                    "in spatial dimension %d"
                    % (x.size(), self.equivariant_step, self.dims - d)
                )

        return self.down(x)


class Upsample(ContextAwareModule):
    """
    The Upsample class wraps a Upsample layer with a limited set of arguments.
    It is designed to be used with the UNet class and satisfies
    the `ContextAwareModule` interface.

    :param dims: the number of dimensions
    :param scale_factor: the upsample factor
    :param mode: the upsample mode (nearest, bilinear, etc.)
    """

    _dims: int

    def __init__(
        self,
        dims: int,
        scale_factor: Sequence[int] | int,
        mode: str = "nearest",
    ):
        super().__init__()

        self._dims = dims
        self._equivariant_step = torch.tensor((1,) * self.dims) / torch.tensor(
            scale_factor
        )
        scale_factor = (
            tuple(scale_factor) if not isinstance(scale_factor, int) else scale_factor
        )
        self.up = torch.nn.Upsample(scale_factor=scale_factor, mode=mode)

    @property
    def context(self) -> torch.Tensor:
        """
        `context` is always 0 for upsampling.
        """
        return torch.tensor((0,) * self.dims)

    @property
    def equivariant_step(self) -> torch.Tensor:
        """
        The invariant step is the inverse of the upsample factor for each dimension.
        A upsample factor of 4 means each pixel shift in the input shifts the output by 4 pixels.
        """
        return self._equivariant_step

    @property
    def min_input_shape(self) -> torch.Tensor:
        """
        `min_input_shape` is equal to 1 for each dimension.
        """
        return torch.tensor((1,) * self.dims)

    @property
    def min_output_shape(self) -> torch.Tensor:
        """
        `min_output_shape` is equal to the upsample factor for each dimension.
        """
        return (self.min_input_shape / self.equivariant_step).int()

    @property
    def dims(self) -> int:
        """
        The number of dimensions.
        """
        return self._dims

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the upsample operation to the input tensor.

        :param x: the input tensor
        """
        return self.up(x)
