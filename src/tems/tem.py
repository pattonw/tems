from abc import ABC, abstractmethod

import torch


class ContextAwareModule(ABC, torch.nn.Module):
    @property
    @abstractmethod
    def context(self) -> torch.Tensor: ...

    @property
    @abstractmethod
    def equivariant_step(self) -> torch.Tensor: ...

    @property
    @abstractmethod
    def min_input_shape(self) -> torch.Tensor: ...

    @property
    @abstractmethod
    def min_output_shape(self) -> torch.Tensor: ...

    @property
    @abstractmethod
    def dims(self) -> int: ...
