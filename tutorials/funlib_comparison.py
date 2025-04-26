# %% [markdown]
# # Funlib API comparison
# This notebook compares the `UNet` implementation in `funlib.learn.torch`
# with the one in `tems`.
#
# We specifically compare
# - the constructors to show that it is a very easy migration.
# - funlib.UNet fails `torch.jit.script`.
# - funlib.UNet crops unnecessarily aggressively to maintain translation equivariance.

# %% [markdown]
# ## Setup

# %%
import torch
from funlib.learn.torch.models import UNet as FunlibUNet

from tems import UNet

# Here we define a set of downsampling factors for demonstration purposes.
# I use 2D downsampling factors to simplify compute and because the
# funlib UNet does not support 1D.
downsample_factors = [
    [[2, 1], [2, 1], [2, 1], [2, 1]],
    [[2, 1], [3, 1], [4, 1], [2, 1]],
    [[2, 1], [4, 1], [3, 1], [5, 1]],
]
# The extra input is necessary because the funlib UNet crops more aggressively
# than the tems UNet, thus has a larger `min_input_shape`. These were found via manual
# guess and check. Figuring out the appropriate input shape for the funlib UNet is not
# always easy.
extra_inputs = [
    (16, 0),
    (48 * 2, 0),
    (120 * 2, 0),
]


# %%
def build_unet(downsample_factors):
    return UNet.funlib_api(
        dims=2,
        in_channels=1,
        num_fmaps=1,
        fmap_inc_factor=1,
        downsample_factors=downsample_factors,
        activation="Identity",
    )


def build_funlib_unet(downsample_factors):
    return FunlibUNet(
        in_channels=1,
        num_fmaps=1,
        fmap_inc_factor=1,
        downsample_factors=downsample_factors,
        kernel_size_down=[[[3, 3], [3, 3]]] * (len(downsample_factors) + 1),
        kernel_size_up=[[[3, 3], [3, 3]]] * len(downsample_factors),
        activation="Identity",
    )


# %% [markdown]
# ## Jit scripting

# %%
unet = build_unet(downsample_factors[0])
try:
    torch.jit.script(unet)
    print("Successfully scripted tems.UNet")
except RuntimeError:
    print("Failed to script tems.UNet")

# %%
funlib_unet = build_funlib_unet(downsample_factors[0])
try:
    torch.jit.script(funlib_unet)
    print("Successfully scripted funlib.UNet")
except RuntimeError:
    print("Failed to script funlib.UNet")


# %% [markdown]
# ## Cropping

# %%
def test_unet_comparison(tems_unet: UNet, funlib_unet: FunlibUNet, input_shape):
    in_data = torch.rand(1, 1, *(input_shape))
    print("Input shape:", list(in_data.shape[2:]))
    tems_out_data = tems_unet(in_data)
    tems_out_training = tems_unet.train()(in_data)
    funlib_out_data = funlib_unet(in_data)
    print("Output shape tems (Training):", list(tems_out_training.shape[2:]))
    print("Output shape tems (Translation Equivariant):", list(tems_out_data.shape[2:]))
    print(
        "Output shape funlib (Translation Equivariant):",
        list(funlib_out_data.shape[2:]),
    )


# %%

for downsample_factor, extra_input in zip(downsample_factors, extra_inputs):
    unet = build_unet(downsample_factor).eval()
    input_shape = unet.min_input_shape + torch.tensor(extra_input)
    funlib_unet = build_funlib_unet(downsample_factor).eval()

    print("Total downsampling factor:", unet.invariant_step)
    test_unet_comparison(unet, funlib_unet, input_shape)
    print()

# %% [markdown]
# How are we cropping less and maintaining translation equivariance?
# Because we compute the necessary input/output shapes of every component
# in the UNet, we can check what the globally minimal crop is, and then apply
# only that crop at whatever point in the network it is most efficient.
# The funlib UNet builds layer by layer and identifies whatever crop is needed
# to keep each level of the UNet translation equivariant. This is overkill.
