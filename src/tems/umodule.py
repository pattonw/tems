import torch

from .conv_pass import ConvPass
from .scale import Downsample, Upsample
from .tem import ContextAwareModule


class UModule(ContextAwareModule):
    """
    The UModule class is an abstraction of a single layer
    of a UNet. It involves an input conv pass, a downsample,
    a lower block, an upsample, and an output conv pass.
    It is designed to be used with the UNet class and satisfies
    the `ContextAwareModule` interface.

    :param in_conv_pass: the input conv pass
    :param downsample: the downsample layer
    :param lower_block: the lower block
    :param upsample: the upsample layer
    :param out_conv_pass: the output
    """

    equivariance_context: torch.Tensor
    _dims: int
    _equivariant_step: torch.Tensor

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
        self._equivariant_step = downsample.equivariant_step * lower_block.equivariant_step
        assert torch.equal(
            downsample.equivariant_step * upsample.equivariant_step,
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
        """
        The number of dimensions (1, 2, or 3).
        """
        return self._dims

    @property
    def context(self) -> torch.Tensor:
        """
        `in_conv_pass.context` + `downsample.equivariant_step * lower_block.context`
        + `out_conv_pass.context` + (Optional `equivariance_context`)

        The equivariance context is only added during evaluation and is used to make
        sure the network is translation equivariant for easy blockwise processing
        without tiling artifacts.
        """
        base_context = (
            self.in_conv_pass.context
            + self.downsample.equivariant_step * self.lower_block.context
            + self.out_conv_pass.context
        )

        if self.training:
            return base_context
        else:
            return base_context + self.equivariance_context

    @property
    def equivariant_step(self) -> torch.Tensor:
        """
        The invariant step is the product of the downsample factors.
        """
        return self._equivariant_step

    @property
    def min_input_shape(self) -> torch.Tensor:
        """
        The minimum input shape that this module can accept.
        """
        # Some details about the lower block
        lower_block_input_shape = self.lower_block.min_input_shape
        lower_block_context = self.lower_block.context
        lower_block_output_shape = lower_block_input_shape - lower_block_context

        # Upsample the lower block output shape and subtract our out conv context
        min_lower_output = (
            lower_block_output_shape / self.upsample.equivariant_step
        ).long()
        min_out = min_lower_output - self.out_conv_pass.context

        # min_out could be negative. We want it to be at least [1,1]
        min_expansion = torch.tensor([1] * self.dims) - min_out
        min_expansion[min_expansion < 0] = 0

        # we must round this value up to the next multiple of the invariant step
        min_expansion = (
            torch.ceil(min_expansion / self.equivariant_step) * self.equivariant_step
        ).long()

        # whats the minimum input shape of the lower block scaled by the downsample step
        min_lower_input = lower_block_input_shape * self.downsample.equivariant_step

        # now we just add the min_expansion term, and the context from the input conv pass
        min_input_shape = min_lower_input + min_expansion + self.in_conv_pass.context
        return min_input_shape

    @property
    def min_output_shape(self) -> torch.Tensor:
        """
        The minimum output shape that this module can produce.
        `min_input_shape` - `context`
        """
        return self.min_input_shape - self.context

    def set_equivariance_context(self, equivariance_context: torch.Tensor):
        """
        Set the equivariance context to be used during evaluation.

        :param equivariance_context: the equivariance context per dimension
        """
        self.equivariance_context = equivariance_context

    @torch.jit.export
    def crop(self, x: torch.Tensor, shape: torch.Tensor) -> torch.Tensor:
        """
        Center-crop x to match spatial dimensions given by shape.

        :param x: the input tensor
        :param shape: the target shape
        """

        x_shape = x.size()[2:]
        offset = (torch.tensor(x_shape) - shape) // 2
        for i, (o, s) in enumerate(zip(offset, shape)):
            x = torch.slice_copy(x, i + 2, o.item(), o.item() + s)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the UModule to the input tensor.

        :param x: the input tensor
        """
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
