from itertools import product

import numpy as np
import pytest
import torch
from funlib.geometry import Coordinate, Roi
from funlib.persistence.arrays import Array

from tems import (
    UNet,
)


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
    return Coordinate(*[gcd(a, b) for a, b in zip(self, other)])


Coordinate.gcd = gcd_coord


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
    )
    if eval_mode:
        unet = unet.eval()
    min_output_shape = Coordinate(unet.min_output_shape)
    context = Coordinate(unet.context) // 2
    downsampling = Coordinate(unet.equivariant_step)
    blocks = Coordinate(1, 1) + downsampling / downsampling.gcd(
        min_output_shape
    ) * Coordinate(1, 0)

    in_array = Array(
        np.arange(
            (
                in_channels
                * min_output_shape[0]
                * min_output_shape[1]
                * blocks[0]
                * blocks[1]
            )
        ).reshape(1, in_channels, *(min_output_shape * blocks)),
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
        out_data = unet(torch.from_numpy(in_data).float()).detach().cpu().numpy()
        blockwise_out_array[block_roi] = out_data

    full_out_array[full_out_array.roi] = (
        unet(
            torch.from_numpy(
                in_array.to_ndarray(in_array.roi.grow(context, context))
            ).float()
        )
        .detach()
        .cpu()
        .numpy()
    )

    a, b = blockwise_out_array[:], full_out_array[:]
    if not eval_mode:
        with pytest.raises(AssertionError):
            assert np.allclose(a, b), "Blockwise and in-memory output data do not match"
    else:
        assert np.allclose(a, b), "Blockwise and in-memory output data do not match"
