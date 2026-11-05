import math

import pytest
import torch

from tems import ConvPass


def test_identity_activation_initializes_constant_weights():
    """
    With an Identity activation the initializer has no nonlinearity to match,
    so it warns and sets every convolution to an averaging kernel with zero bias.
    """
    in_channels, kernel_sizes = 2, [(3, 3), (5, 5)]
    with pytest.warns(UserWarning, match="Identity activation"):
        conv_pass = ConvPass(
            dims=2,
            in_channels=in_channels,
            out_channels=1,
            kernel_sizes=kernel_sizes,
            activation=torch.nn.Identity,
            initialize_weights=True,
        )

    convs = [m for m in conv_pass.modules() if isinstance(m, torch.nn.Conv2d)]
    assert len(convs) == len(kernel_sizes)
    for conv, kernel_size in zip(convs, kernel_sizes, strict=True):
        constant = 1.0 / math.prod(kernel_size) / conv.in_channels
        assert torch.equal(conv.weight, torch.full_like(conv.weight, constant))
        assert conv.bias is not None
        assert torch.equal(conv.bias, torch.zeros_like(conv.bias))
