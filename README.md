[![tests](https://github.com/pattonw/tems/actions/workflows/tests.yaml/badge.svg)](https://github.com/pattonw/tems/actions/workflows/tests.yaml)
[![lint](https://github.com/pattonw/tems/actions/workflows/lint.yaml/badge.svg)](https://github.com/pattonw/tems/actions/workflows/lint.yaml)
[![docs](https://github.com/pattonw/tems/actions/workflows/docs.yaml/badge.svg)](https://pattonw.github.io/tems)
[![publish](https://github.com/pattonw/tems/actions/workflows/publish.yaml/badge.svg)](https://pypi.org/project/tems/)

# Translation Equivariant ModelS (TEMS)

**Documentation:** https://pattonw.github.io/tems

## Install

```bash
pip install tems    # or: uv add tems
```

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

## Contributing

The commands below are the ones CI runs, so a green local run means a green PR.

```bash
git clone https://github.com/pattonw/tems
cd tems
uv sync                                             # Python, dependencies, dev tools (from uv.lock)
uv pip install -e .                                 # the package itself (no build backend yet)
uv run pytest                                       # tests with coverage
uv run --only-group dev ruff check . && uv run --only-group dev ruff format --check .
uv run ty check                                     # types
# docs: see .github/workflows/docs.yaml (notebooks are converted before sphinx-build)
```

`uv.lock` pins the development environment; the `lock-upgrade` workflow refreshes it monthly and opens a pull request.
