import pandas as pd
import torch
import abc


class BasePredictor(abc.ABC):
    tasks = None

    @classmethod
    @abc.abstractmethod
    def load(cls) -> None:
        '''
        Set tasks and other predictor-specific attributes.
        Must be called before run() or run_sensitivity().
        '''
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def run(
            cls, 
            df: pd.api.typing.DataFrameGroupBy
    ) -> tuple[tuple[list[torch.DoubleTensor], ...], list[torch.LongTensor], list[torch.DoubleTensor], int]:
        '''
        Run the predictor on the given DataFrameGroupBy.

        Parameters:
        df (pd.api.typing.DataFrameGroupBy): DataFrameGroupBy with the following columns:
            - 'peptide': Peptide sequences.
            - 'label': Binary labels.
            - 'log50k' (optional): Log50k values.
        
        Returns:
        tuple[tuple[list[torch.DoubleTensor], ...], list[torch.LongTensor], list[torch.DoubleTensor], int]: 
            - Tuple of predictions for each task.
            - List of labels.
            - List of log50k values.
            - Total runtime in nanoseconds.
        '''
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def run_sensitivity(
            cls, 
            df: pd.api.typing.DataFrameGroupBy
    ) -> tuple[tuple[list[torch.DoubleTensor], ...], list[torch.DoubleTensor]]:
        '''
        Run the predictor on the given DataFrameGroupBy for sensitivity analysis.

        Parameters:
        df (pd.api.typing.DataFrameGroupBy): DataFrameGroupBy with the following columns:
            - 'peptide1': Peptide sequences.
            - 'peptide2': Peptide sequences.
            - 'label': Binary labels.
            - 'log50k' (optional): Log50k values.
        
        Returns:
        tuple[tuple[list[torch.DoubleTensor], ...], list[torch.DoubleTensor]]: 
            - Tuple of predictions for each task.
            - List of log50k values.
        '''
        raise NotImplementedError
    