[![tests](https://github.com/pattonw/tems/actions/workflows/tests.yaml/badge.svg)](https://github.com/pattonw/tems/actions/workflows/tests.yaml)
[![codecov](https://codecov.io/gh/pattonw/tems/branch/main/graph/badge.svg?token=YOUR_TOKEN)](https://codecov.io/gh/pattonw/tems)
[![ruff](https://github.com/pattonw/tems/actions/workflows/ruff.yaml/badge.svg)](https://github.com/pattonw/tems/actions/workflows/ruff.yaml)
[![mypy](https://github.com/pattonw/tems/actions/workflows/mypy.yaml/badge.svg)](https://github.com/pattonw/tems/actions/workflows/mypy.yaml)

[![pypi](https://img.shields.io/pypi/v/tems.svg)](https://pypi.python.org/pypi/tems)

# Translation Equivariant ModelS (tems)

See the docs [here](https://pattonw.github.io/tems)

## Available Models
- UNet

## Why use these Models
- It is surprisingly annoying to make a torch modules that are compatible with `torch.jit.script` for easy packaging and sharing. All models in this library are tested for serlializaiton/deserialization with `torch.jit.script`
- All models in this library have some helpful properties:
    - `min_input_shape` - the minimum amount of data that can be passed through a model
    - `min_output_shape` - the output shape when given data with shape: `min_input_shape`
    - `context` - (`input_shape` - `output_shape`)
    - `equivariant_step` - The minimum value by which you must increase the input shape to be able to process without error
- All models are translation equivariant **when in eval mode and with padding="valid"**
    - To maintain translation equivariance, we sometimes need to crop more aggressively. This is only done in `eval` mode since it is assumed you will not be processing data blockwise during training. Note that `min_input_shape`, `min_output_shape`, and `context` will thus all change when the model is switched between modes.
    - This behaviour is explicitly tested and shown to be true s.t. while training, blockwise processing is not the same as full in memmory operations, wheras while in eval mode, this identity holds.
    - Crops to maintain translation equivariance are minimal to maximize the efficiency of blockwise processing with these models.
- Nice simple pure torch implementation. No other dependencies.