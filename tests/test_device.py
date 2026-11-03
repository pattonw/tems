import pytest
import torch

from tems import ConvPass, UNet

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@cuda
@pytest.mark.parametrize("script", [False, True])
@pytest.mark.parametrize("in_channels", [1, 2])
def test_residual_convpass_cuda(script, in_channels):
    conv_pass = ConvPass(2, in_channels=in_channels, out_channels=2, residual=True)
    if script:
        conv_pass = torch.jit.script(conv_pass)
    x = torch.rand(1, in_channels, 10, 10)
    expected = conv_pass(x)
    out = conv_pass.cuda()(x.cuda())
    assert out.device.type == "cuda"
    assert torch.allclose(out.cpu(), expected, atol=1e-5)


@cuda
def test_residual_unet_cuda():
    unet = UNet.funlib_api(
        dims=2,
        in_channels=1,
        num_fmaps=2,
        fmap_inc_factor=2,
        downsample_factors=[[2, 2]],
        residuals=True,
    ).eval()
    x = torch.rand(1, 1, *unet.min_input_shape.tolist())
    expected = unet(x)
    out = unet.cuda()(x.cuda())
    assert torch.allclose(out.cpu(), expected, atol=1e-5)
