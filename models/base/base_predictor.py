import pandas as pd
import torch
import abc


class BasePredictor(abc.ABC):
    tasks = None

    @abc.abstractmethod
    @classmethod
    def load(cls) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    @classmethod
    def run(
            cls, 
            df: pd.api.typing.DataFrameGroupBy
    ) -> tuple[tuple[list[torch.DoubleTensor], ...], list[torch.LongTensor], list[torch.DoubleTensor]]:
        raise NotImplementedError

    @abc.abstractmethod
    @classmethod
    def run_sensitivity(
            cls, 
            df: pd.api.typing.DataFrameGroupBy
    ) -> tuple[tuple[list[torch.DoubleTensor], ...], list[torch.DoubleTensor]]:
        raise NotImplementedError
    