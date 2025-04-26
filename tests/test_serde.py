import pytest
import torch

from tems import (
    ConvPass,
    Downsample,
    UModule,
    UNet,
    Upsample,
)


def identity(module: torch.nn.Module, tmp_path) -> torch.nn.Module:
    return module


def save_load(module: torch.nn.Module, tmp_path) -> torch.nn.Module:
    torch.save(module, tmp_path / "module.pt")
    return torch.load(tmp_path / "module.pt", weights_only=False)


def jit_script(module: torch.nn.Module, tmp_path) -> torch.nn.Module:
    module = torch.jit.script(module)
    return module


def jit_save_load(module: torch.nn.Module, tmp_path) -> torch.nn.Module:
    module = torch.jit.script(module)
    torch.jit.save(module, tmp_path / "module.jit")
    return torch.jit.load(tmp_path / "module.jit")


@pytest.mark.parametrize(
    "in_channels, out_channels, dims, kernel_sizes, activation, padding",
    [
        (1, 1, 2, [3], torch.nn.ReLU, "valid"),
        (2, 2, 3, [(3, 3, 3)], torch.nn.ReLU, "same"),
        (4, 4, 3, ((5, 5, 5), (3, 3, 3)), torch.nn.LeakyReLU, "valid"),
        (8, 8, 2, [7, [3, 3]], torch.nn.Sigmoid, "same"),
    ],
)
@pytest.mark.parametrize("serde", [identity, save_load, jit_script, jit_save_load])
def test_convpass(
    in_channels, out_channels, dims, kernel_sizes, activation, padding, tmp_path, serde
):
    # generate a few versions of the model (serialized/deserialized, jit, etc)
    conv_pass = ConvPass(
        dims=dims,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_sizes=kernel_sizes,
        activation=activation,
        padding=padding,
    )
    conv_pass_loaded = serde(conv_pass, tmp_path)

    # generate input data
    in_data = torch.rand(1, in_channels, *(conv_pass.context + 5))

    # compute output data
    out_data = conv_pass(in_data)
    out_data_loaded = conv_pass_loaded(in_data)

    # check that all versions of the model give the same output
    assert torch.equal(out_data, out_data_loaded)

    assert torch.equal(
        torch.tensor(in_data.shape[2:]) - torch.tensor(out_data.shape[2:]),
        conv_pass.context,
    ), (in_data.shape, out_data.shape, conv_pass.context)


@pytest.mark.parametrize(
    "dims, downsample_factor, invalid_input",
    [
        (3, (2, 3, 4), True),
        (2, 2, True),
        (3, 3, True),
        (2, [3, 1], True),
        (3, [2, 3, 4], False),
        (2, 2, False),
        (3, 3, False),
        (2, (3, 1), False),
    ],
)
@pytest.mark.parametrize("serde", [identity, save_load, jit_script, jit_save_load])
def test_downsample(dims, downsample_factor, invalid_input, serde, tmp_path):
    # build a few versions of the model (serialized/deserialized, jit, etc)
    downsample = Downsample(dims=dims, downsample_factor=downsample_factor)
    downsample_loaded = serde(downsample, tmp_path)

    # get the invariant step
    equivariant_step = downsample.equivariant_step
    if invalid_input:
        # generate invalid input data
        in_data = torch.rand(1, 1, *(equivariant_step + 1))

        # check that the models raise errors
        with pytest.raises(RuntimeError):
            downsample(in_data)
        with pytest.raises((RuntimeError, torch.jit.Error)):
            downsample_loaded(in_data)
    else:
        # generate valid input data
        in_data = torch.rand(1, 1, *(equivariant_step * 4))

        # process data
        out_data = downsample(in_data)
        out_data_loaded = downsample_loaded(in_data)

        # check results equal
        assert torch.equal(out_data, out_data_loaded)


@pytest.mark.parametrize(
    "dims, scale_factor, mode",
    [
        (1, 2, "nearest"),
        (3, 3, "trilinear"),
        (2, (4, 2), "bilinear"),
        (3, [2, 4, 3], "nearest"),
    ],
)
@pytest.mark.parametrize("serde", [identity, save_load, jit_script, jit_save_load])
def test_upsample(dims, scale_factor, mode, serde, tmp_path):
    # build a few versions of the model (serialized/deserialized, jit, etc)
    upsample = Upsample(
        dims=dims,
        scale_factor=scale_factor,
        mode=mode,
    )
    upsample_loaded = serde(upsample, tmp_path)

    # create in data
    in_data = torch.rand(1, 1, *((3,) * dims))

    # process data
    out_data = upsample(in_data)
    out_data_loaded = upsample_loaded(in_data)

    # check results equal
    assert torch.equal(out_data, out_data_loaded)


@pytest.mark.parametrize(
    "in_shape, in_conv_pass, downsample, lower_block, upsample, out_conv_pass, equivariance_context",
    [
        (
            (14, 20),
            ConvPass(dims=2, kernel_sizes=[3]),
            Downsample(dims=2, downsample_factor=(2, 3)),
            ConvPass(dims=2, kernel_sizes=[3]),
            Upsample(dims=2, scale_factor=(2, 3)),
            ConvPass(dims=2, in_channels=2, kernel_sizes=[3]),
            (2, 4),
        ),
    ],
)
@pytest.mark.parametrize("serde", [identity, save_load, jit_script, jit_save_load])
def test_umodule(
    in_shape,
    in_conv_pass,
    downsample,
    lower_block,
    upsample,
    out_conv_pass,
    equivariance_context,
    serde,
    tmp_path,
):
    # a few versions of the model (serialized/deserialized, jit, etc)
    umodule = UModule(
        in_conv_pass=in_conv_pass,
        downsample=downsample,
        lower_block=lower_block,
        upsample=upsample,
        out_conv_pass=out_conv_pass,
        _equivariance_context=torch.tensor(equivariance_context),
    )
    umodule_loaded = serde(umodule, tmp_path)

    # in data
    min_input_shape = umodule.min_input_shape
    min_output_shape = min_input_shape - umodule.context
    in_data = torch.rand(1, 1, *(min_input_shape))

    # out data versions
    out_data = umodule(in_data)
    out_data_loaded = umodule_loaded(in_data)

    # check that the data is the shape we expect
    assert out_data.shape[2:] == tuple(min_output_shape)

    # check that all versions of the model give the same output
    assert torch.equal(out_data, out_data_loaded)

    # check eval mode
    umodule = umodule.eval()
    umodule_loaded = umodule_loaded.eval()

    # generate input data
    in_data = torch.rand(1, 1, *(in_shape))

    # generate eval mode data
    out_data = umodule(in_data)
    out_data_loaded = umodule_loaded(in_data)

    # check that all versions of the model give the same output
    assert torch.equal(out_data, out_data_loaded)

    # check eval mode generated data with expected shape
    expected_output_shape = (
        torch.tensor(in_shape)
        - in_conv_pass.context
        - lower_block.context * downsample.equivariant_step
        - out_conv_pass.context
        - umodule.equivariance_context
    )
    assert out_data.shape[2:] == tuple(expected_output_shape)


@pytest.mark.parametrize(
    "dims, bottleneck, levels",
    [
        (
            2,
            ConvPass(2),
            [
                (
                    ConvPass(2),
                    Downsample(2, 2),
                    Upsample(2, 2),
                    ConvPass(2, in_channels=2),
                ),
                (
                    ConvPass(2),
                    Downsample(2, 2),
                    Upsample(2, 2),
                    ConvPass(2, in_channels=2),
                ),
                (
                    ConvPass(2),
                    Downsample(2, 2),
                    Upsample(2, 2),
                    ConvPass(2, in_channels=2),
                ),
                (
                    ConvPass(2),
                    Downsample(2, 2),
                    Upsample(2, 2),
                    ConvPass(2, in_channels=2),
                ),
            ],
        ),
    ],
)
@pytest.mark.parametrize("serde", [identity, save_load, jit_script, jit_save_load])
def test_unet(dims, bottleneck, levels, serde, tmp_path):
    unet = UNet(dims=dims, bottleneck=bottleneck, levels=levels)
    unet_loaded = serde(unet, tmp_path)

    # generate input data
    in_data = torch.rand(1, 1, *(unet.min_input_shape))

    # process data
    out_data = unet(in_data)
    out_data_loaded = unet_loaded(in_data)

    # check that all versions of the model give the same output
    assert torch.equal(out_data, out_data_loaded)

    # check eval mode
    unet = unet.eval()
    unet_loaded = unet_loaded.eval()

    # generate input data
    in_data = torch.rand(1, 1, *(unet.min_input_shape))

    # generate eval mode data
    out_data = unet(in_data)
    out_data_loaded = unet_loaded(in_data)

    # check that all versions of the model give the same output
    assert torch.equal(out_data, out_data_loaded)
