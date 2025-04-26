from typing import Sequence

import torch

from .conv_pass import ConvPass
from .scale import Downsample, Upsample
from .tem import ContextAwareModule
from .umodule import UModule


class UNet(ContextAwareModule):
    """
    A UNet implementation with helper functions to calculate the
    minimum input and output shapes of the network, along with the
    context and appropriate step size for translation equivariance.

    This UNet is also fully scriptable with `torch.jit.script` making
    it easy to share and deploy.

    :param dims: the number of dimensions
    :param bottleneck: the bottleneck module
    :param levels: a sequence of tuples containing the left conv pass,
        downsample, upsample, and right conv pass for each level
    """

    _dims: int

    def __init__(
        self,
        dims: int,
        bottleneck: ContextAwareModule,
        levels: Sequence[tuple[ConvPass, Downsample, Upsample, ConvPass]],
    ):
        super().__init__()

        self._dims = dims

        head_module: ContextAwareModule | None = None
        for left, down, up, right in reversed(levels):
            head_module = UModule(
                in_conv_pass=left,
                downsample=down,
                lower_block=head_module if head_module is not None else bottleneck,
                upsample=up,
                out_conv_pass=right,
            )
        assert head_module is not None, "0 level UNet not supported"

        self.head_module = head_module

        # handle cropping to ensure translation equivariance
        # get all downsampling factors
        downsampling = [torch.tensor((1,) * dims)] + [
            downsample.equivariant_step for _, downsample, _, _ in levels
        ]
        stacked_downsampling = torch.stack(downsampling)
        layer_downsampling = torch.cumprod(stacked_downsampling, dim=0)
        total_downsampling = torch.prod(stacked_downsampling, dim=0)

        # get all layer output_sizes
        output_shape = head_module.min_output_shape

        # invariant output_shape: round output_shape down to the next multiple of total_downsampling
        invariant_output_shape = (
            torch.floor(output_shape / total_downsampling) * total_downsampling
        )
        to_crop = output_shape - invariant_output_shape

        crop_amounts: list[torch.Tensor] = []
        for downsample_factor in reversed(layer_downsampling[:-1]):
            can_crop = to_crop % (downsample_factor * 2) == 0
            crop_amount = (to_crop / downsample_factor) * can_crop
            crop_amounts = [crop_amount] + crop_amounts
            to_crop = to_crop - crop_amount * downsample_factor
        stacked_crop_amounts: torch.Tensor = torch.stack(crop_amounts)

        for crop_amount in stacked_crop_amounts:
            assert isinstance(head_module, UModule)
            head_module.set_equivariance_context(crop_amount.long())
            head_module = head_module.lower_block

    @property
    def dims(self) -> int:
        """
        The number of dimensions (1, 2, or 3).
        """
        return self._dims

    @property
    def context(self) -> torch.Tensor:
        """
        The context of the UNet.
        """
        return self.head_module.context

    @property
    def equivariant_step(self) -> torch.Tensor:
        """
        The invariant step is the product of all downsampling factors.
        """
        return self.head_module.equivariant_step

    @property
    def min_input_shape(self) -> torch.Tensor:
        """
        The minimum input shape that this module can accept.
        """
        return self.head_module.min_input_shape

    @property
    def min_output_shape(self) -> torch.Tensor:
        """
        The minimum output shape that this module can produce.
        """
        return self.head_module.min_output_shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the UNet to the input tensor.

        :param x: the input tensor
        """
        return self.head_module(x)

    @classmethod
    def funlib_api(
        cls,
        dims: int,
        in_channels: int,
        num_fmaps: int,
        fmap_inc_factor: int,
        downsample_factors: Sequence[Sequence[int] | int],
        kernel_size_down: Sequence[Sequence[Sequence[int] | int]] | None = None,
        kernel_size_up: Sequence[Sequence[Sequence[int] | int]] | None = None,
        activation: str = "ReLU",
        num_fmaps_out: int | None = None,
        num_heads: int = 1,
        constant_upsample: bool = False,
        padding: str = "valid",
    ):
        """
        A helper method to match the API of the funlib UNet as closely as possible.
        This method is helpful to creating a UNet with a more compact API whereas
        the default constructor makes you define every layer explicitly.

        :param dims: the number of dimensions
        :param in_channels: the number of input channels
        :param num_fmaps: the number of feature maps
        :param fmap_inc_factor: the factor by which to increase the number of feature maps
        :param downsample_factors: the downsample factors for each level
        :param kernel_size_down: the kernel size for the downsample convolutions
        :param kernel_size_up: the kernel size for the upsample convolutions
        :param activation: the activation function to use
        :param num_fmaps_out: the number of output feature maps
        :param num_heads: the number of heads to use
        :param constant_upsample: whether to use constant upsampling
        :param padding: the padding mode to use. Supported values are "valid" and "same".
        """
        _activation: type[torch.nn.Module] = getattr(torch.nn, activation)
        kernel_size_up = (
            kernel_size_up
            if kernel_size_up is not None
            else [(3, 3)] * len(downsample_factors)
        )
        kernel_size_down = (
            kernel_size_down
            if kernel_size_down is not None
            else [(3, 3)] * len(downsample_factors)
        )
        if (
            kernel_size_down is not None
            and len(kernel_size_down) == len(downsample_factors) + 1
        ):
            bottleneck_kernel = kernel_size_down[-1]
        else:
            bottleneck_kernel = [3, 3]

        layers = []
        for i, (kernel_down, scale_factor, kernel_up) in enumerate(
            zip(
                kernel_size_down,
                downsample_factors,
                kernel_size_up,
            )
        ):
            layers.append(
                (
                    ConvPass(
                        dims,
                        in_channels
                        if i == 0
                        else num_fmaps * fmap_inc_factor ** (i - 1),
                        num_fmaps * fmap_inc_factor**i,
                        kernel_sizes=kernel_down,
                        activation=_activation,
                        padding=padding,
                    ),
                    Downsample(dims, scale_factor),
                    Upsample(dims, scale_factor),
                    ConvPass(
                        dims,
                        num_fmaps * fmap_inc_factor**i
                        + num_fmaps * fmap_inc_factor ** (i + 1),
                        num_fmaps * fmap_inc_factor**i,
                        kernel_sizes=kernel_up,
                        activation=_activation,
                        padding=padding,
                    ),
                )
            )
        bottleneck = ConvPass(
            dims,
            num_fmaps * fmap_inc_factor ** (len(downsample_factors) - 1),
            num_fmaps * fmap_inc_factor ** len(downsample_factors),
            kernel_sizes=bottleneck_kernel,
            activation=_activation,
            padding=padding,
        )
        return UNet(dims, bottleneck, layers)
