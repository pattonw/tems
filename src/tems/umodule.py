import torch

from .conv_pass import ConvPass
from .scale import Downsample, Upsample
from .tem import ContextAwareModule


class UModule(ContextAwareModule):
    equivariance_context: torch.Tensor
    _dims: int
    _invariant_step: torch.Tensor

    def __init__(
        self,
        in_conv_pass: ConvPass,
        downsample: Downsample,
        lower_block: ContextAwareModule,
        upsample: Upsample,
        out_conv_pass: ConvPass,
        _equivariance_context: torch.Tensor | None = None,
    ):
        super().__init__()
        self._dims = in_conv_pass.dims
        self._invariant_step = downsample.invariant_step * lower_block.invariant_step
        assert torch.equal(
            downsample.invariant_step * upsample.invariant_step,
            torch.tensor((1,) * self.dims),
        ), "Down and Up sampling must have the same scale factor"

        self.equivariance_context = (
            _equivariance_context.long()
            if _equivariance_context is not None
            else torch.tensor((0,) * self.dims)
        )
        assert (
            self.dims
            == downsample.dims
            == lower_block.dims
            == upsample.dims
            == out_conv_pass.dims
        ), "All modules must have the same number of dimensions"

        self.in_conv_pass = in_conv_pass
        self.downsample = downsample
        self.lower_block = lower_block
        self.upsample = upsample
        self.out_conv_pass = out_conv_pass

    @property
    def dims(self) -> int:
        return self._dims

    @property
    def context(self) -> torch.Tensor:
        base_context = (
            self.in_conv_pass.context
            + self.downsample.invariant_step * self.lower_block.context
            + self.out_conv_pass.context
        )

        if self.training:
            return base_context
        else:
            return base_context + self.equivariance_context

    @property
    def invariant_step(self) -> torch.Tensor:
        return self._invariant_step

    @property
    def min_input_shape(self) -> torch.Tensor:
        # Some details about the lower block
        lower_block_input_shape = self.lower_block.min_input_shape
        lower_block_context = self.lower_block.context
        lower_block_output_shape = lower_block_input_shape - lower_block_context

        # Upsample the lower block output shape and subtract our out conv context
        min_lower_output = (
            lower_block_output_shape / self.upsample.invariant_step
        ).long()
        min_out = min_lower_output - self.out_conv_pass.context

        # min_out could be negative. We want it to be at least [1,1]
        min_expansion = torch.tensor([1] * self.dims) - min_out
        min_expansion[min_expansion < 0] = 0

        # we must round this value up to the next multiple of the invariant step
        min_expansion = (
            torch.ceil(min_expansion / self.invariant_step) * self.invariant_step
        ).long()

        # whats the minimum input shape of the lower block scaled by the downsample step
        min_lower_input = lower_block_input_shape * self.downsample.invariant_step

        # now we just add the min_expansion term, and the context from the input conv pass
        min_input_shape = min_lower_input + min_expansion + self.in_conv_pass.context
        return min_input_shape

    @property
    def min_output_shape(self) -> torch.Tensor:
        return self.min_input_shape - self.context

    def set_equivariance_context(self, equivariance_context: torch.Tensor):
        self.equivariance_context = equivariance_context

    @torch.jit.export
    def crop(self, x: torch.Tensor, shape: torch.Tensor) -> torch.Tensor:
        """Center-crop x to match spatial dimensions given by shape."""

        x_shape = x.size()[2:]
        offset = (torch.tensor(x_shape) - shape) // 2
        for i, (o, s) in enumerate(zip(offset, shape)):
            x = torch.slice_copy(x, i + 2, o.item(), o.item() + s)
        return x

    def forward(self, x):
        # simple processing
        f_in = self.in_conv_pass(x)
        g_in = self.downsample(f_in)
        g_out = self.lower_block(g_in)
        f_in_up = self.upsample(g_out)

        # crop f_in and f_in_up to ensure translation equivariance
        target_shape = torch.tensor(f_in_up.size()[-self.dims :])
        if not self.training:
            target_shape = target_shape - self.equivariance_context
        f_in = self.crop(f_in, target_shape)
        f_in_up = self.crop(f_in_up, target_shape)
        f_in_cat = torch.cat([f_in, f_in_up], dim=1)

        # final conv pass
        y = self.out_conv_pass(f_in_cat)
        return y
