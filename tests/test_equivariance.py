from itertools import product

import numpy as np
import pytest
import torch
from funlib.geometry import Coordinate, Roi
from funlib.persistence.arrays import Array
from torch.nn.modules.conv import _ConvNd

from tems import (
    UNet,
)

# Every value this test pushes through the network is an integer, and the
# network runs in float64, so as long as no partial sum exceeds 2**53 every
# convolution, pooling and upsampling step is exact: the blockwise and the
# whole-array paths must then agree bit for bit, whatever algorithm the
# convolution picks for a given input shape. The same test in float32 with
# real-valued weights compares outputs that differ in the sixth significant
# figure for some draws, which is not the contract under test.
EXACT_INTEGER_LIMIT = 2**53


def gcd(a: int, b: int):
    """
    Compute the greatest common divisor of two integers.
    """
    while b:
        a, b = b, a % b
    return a


def gcd_coord(self: Coordinate, other: Coordinate):
    """
    Compute the greatest common divisor of two coordinates.
    """
    return Coordinate(*[gcd(a, b) for a, b in zip(self, other, strict=False)])


def integer_weights(unet: torch.nn.Module, rng: np.random.RandomState) -> None:
    """
    Replace every convolution's weights and biases with values in {-1, 0, 1}
    and switch the network to float64, so the forward pass is exact integer
    arithmetic. A forward hook on each convolution asserts that its partial
    sums cannot leave the exactly representable range.
    """
    unet.double()
    for module in unet.modules():
        if not isinstance(module, _ConvNd):
            continue
        assert module.bias is not None
        with torch.no_grad():
            module.weight.copy_(
                torch.from_numpy(rng.randint(-1, 2, size=module.weight.shape))
            )
            module.bias.copy_(
                torch.from_numpy(rng.randint(-1, 2, size=module.bias.shape))
            )
        module.register_forward_hook(_assert_exact)


def _assert_exact(conv: _ConvNd, inputs: tuple, output: torch.Tensor):
    (x,) = inputs
    assert conv.bias is not None
    l1_per_output = conv.weight.abs().flatten(1).sum(dim=1).max()
    bound = x.abs().max() * l1_per_output + conv.bias.abs().max()
    assert x.dtype == torch.float64
    assert bound < EXACT_INTEGER_LIMIT, (
        f"partial sums may reach {bound:.3g}; float64 is no longer exact"
    )


@pytest.mark.parametrize("eval_mode", [True, False])
@pytest.mark.parametrize(
    "downsample_factors",
    [
        [[2, 1], [2, 1], [2, 1], [2, 1]],
        [[2, 1], [3, 1], [4, 1], [2, 1]],
        [[2, 1], [4, 1], [3, 1], [5, 1]],
    ],
)
def test_translation_equivariance(eval_mode, downsample_factors):
    in_channels, out_channels = 1, 1
    unet = UNet.funlib_api(
        dims=2,
        in_channels=in_channels,
        num_fmaps=out_channels,
        fmap_inc_factor=1,
        downsample_factors=downsample_factors,
        activation="Identity",
        initialize_weights=False,
    )
    # RandomState's stream is frozen across numpy versions, so the weights and
    # the input are the same on every machine and every run.
    rng = np.random.RandomState(0)
    integer_weights(unet, rng)
    if eval_mode:
        unet = unet.eval()
    min_output_shape = Coordinate(unet.min_output_shape)
    context = Coordinate(unet.context) // 2
    downsampling = Coordinate(unet.equivariant_step)
    blocks = Coordinate(1, 1) + downsampling / gcd_coord(
        downsampling, min_output_shape
    ) * Coordinate(1, 0)

    in_array = Array(
        rng.randint(0, 16, size=(1, in_channels, *(min_output_shape * blocks))),
        voxel_size=(1,) * len(min_output_shape),
    )
    blockwise_out_array = Array(
        np.zeros((1, out_channels, *(min_output_shape * blocks))),
        voxel_size=(1,) * len(min_output_shape),
    )
    full_out_array = Array(
        np.zeros((1, out_channels, *(min_output_shape * blocks))),
        voxel_size=(1,) * len(min_output_shape),
    )

    # process blockwise
    roi = Roi((0, 0), min_output_shape)

    for i, j in product(range(blocks[0]), range(blocks[1])):
        shift = Coordinate(i, j) * min_output_shape
        block_roi = roi + shift
        in_data = in_array.to_ndarray(block_roi.grow(context, context))
        out_data = unet(torch.from_numpy(in_data).double()).detach().cpu().numpy()
        blockwise_out_array[block_roi] = out_data

    full_out_array[full_out_array.roi] = (
        unet(
            torch.from_numpy(
                in_array.to_ndarray(in_array.roi.grow(context, context))
            ).double()
        )
        .detach()
        .cpu()
        .numpy()
    )

    a, b = blockwise_out_array[:], full_out_array[:]
    if eval_mode:
        np.testing.assert_array_equal(
            a, b, err_msg="Blockwise and in-memory output data do not match"
        )
    else:
        # Documented in the README: the extra crops that make the network
        # translation equivariant are only applied in eval mode, so while
        # training, blockwise processing is not the same as the in-memory pass.
        assert not np.array_equal(a, b), (
            "Blockwise and in-memory output data match in train mode"
        )
