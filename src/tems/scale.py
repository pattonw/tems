from typing import Sequence

import torch

from .tem import ContextAwareModule


class Downsample(ContextAwareModule):
    _context: torch.Tensor
    _invariant_step: torch.Tensor
    _dims: int

    def __init__(self, dims: int, downsample_factor: Sequence[int] | int):
        super().__init__()

        self._dims = dims
        self._context = torch.tensor((0,) * dims)

        pool = {
            2: torch.nn.MaxPool2d,
            3: torch.nn.MaxPool3d,
            4: torch.nn.MaxPool3d,  # only 3D pooling, even for 4D input
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

        self._invariant_step = self._context + torch.tensor(_downsample_factor)

    @property
    def context(self) -> torch.Tensor:
        return self._context

    @property
    def invariant_step(self) -> torch.Tensor:
        return self._invariant_step

    @property
    def min_input_shape(self) -> torch.Tensor:
        return self.min_output_shape * self.invariant_step

    @property
    def min_output_shape(self) -> torch.Tensor:
        return torch.tensor((1,) * self.dims)

    @property
    def dims(self) -> int:
        return self._dims

    def forward(self, x):
        for d in range(1, self.dims + 1):
            if x.size()[-d] % self.invariant_step[-d] != 0:
                raise RuntimeError(
                    "Can not downsample shape %s with factor %s, mismatch "
                    "in spatial dimension %d"
                    % (x.size(), self.invariant_step, self.dims - d)
                )

        return self.down(x)


class Upsample(ContextAwareModule):
    _dims: int

    def __init__(
        self,
        dims: int,
        scale_factor: Sequence[int] | int,
        mode: str = "nearest",
    ):
        super().__init__()

        self._dims = dims
        self._invariant_step = torch.tensor((1,) * self.dims) / torch.tensor(
            scale_factor
        )
        scale_factor = (
            tuple(scale_factor) if not isinstance(scale_factor, int) else scale_factor
        )
        self.up = torch.nn.Upsample(scale_factor=scale_factor, mode=mode)

    @property
    def context(self) -> torch.Tensor:
        return torch.tensor((0,) * self.dims)

    @property
    def invariant_step(self) -> torch.Tensor:
        return self._invariant_step

    @property
    def min_input_shape(self) -> torch.Tensor:
        return torch.tensor((1,) * self.dims)

    @property
    def min_output_shape(self) -> torch.Tensor:
        return (self.min_input_shape / self.invariant_step).int()

    @property
    def dims(self) -> int:
        return self._dims

    def forward(self, x):
        return self.up(x)
