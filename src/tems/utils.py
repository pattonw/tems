import torch


def crop(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Center-crop x to match spatial dimensions given by y.

    :param x: the input tensor
    :param y: the target shape
    """
    x_shape = x.size()[2:]
    offset = (torch.tensor(x_shape) - y) // 2
    for i, (o, s) in enumerate(zip(offset, y)):
        x = torch.slice_copy(x, i + 2, o.item(), o.item() + s)
    return x
