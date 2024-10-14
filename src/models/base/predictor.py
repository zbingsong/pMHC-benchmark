import pandas as pd
import pandas.api.typing as pd_typing
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
    def run_retrieval(
            cls, 
            df: pd_typing.DataFrameGroupBy
    ) -> tuple[tuple[dict[str, dict[str, torch.DoubleTensor]], ...], dict[str, dict[str, torch.DoubleTensor]], dict[str, dict[str, torch.DoubleTensor]], int]:
        '''
        Run the predictor on the given DataFrameGroupBy.

        Parameters:
        df (pd.api.typing.DataFrameGroupBy): DataFrameGroupBy with the following columns:
            - 'peptide': Peptide sequences.
            - 'label': Binary labels.
            - 'log50k' (optional): Log50k values.
        
        Returns:
        tuple[tuple[dict[str, dict[str, torch.DoubleTensor]], ...], dict[str, dict[str, torch.DoubleTensor]], dict[str, dict[str, torch.DoubleTensor]], int]:
            - Retrieval result, where metrics are computed per MHC and per peptide length.
                - Tuple of predictions for each task, where each prediction is a dict (key=MHC names) of dicts (key=peptide length).
                - Labels as a dict (key=MHC names) of dicts (key=peptide length).
                - Log50k values as a dict (key=MHC names) of dicts (key=peptide length), if available; otherwise the second layer dicts are empty.
                - Total runtime in nanoseconds.
        '''
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def run_sensitivity(
            cls, 
            grouped_df: pd_typing.DataFrameGroupBy
    ) -> tuple[tuple[dict[str, dict[str, torch.DoubleTensor]], ...], dict[str, dict[str, torch.DoubleTensor]]]:
        '''
        Run the predictor on the given DataFrameGroupBy for sensitivity analysis.

        Parameters:
        grouped_df (pd.api.typing.DataFrameGroupBy): DataFrameGroupBy with the following columns:
            - 'peptide1': Peptide sequences.
            - 'peptide2': Peptide sequences.
            - 'label': Binary labels.
            - 'log50k' (optional): Log50k values.
        
        Returns:
        tuple[tuple[dict[str, dict[str, torch.DoubleTensor]], ...], dict[str, dict[str, torch.DoubleTensor]]]:
            - Sensitivity result, where metrics are computed per MHC and per peptide length.
                - Tuple of predictions for each task, where each prediction is a dict (key=MHC names) of dicts (key=peptide length).
                - Log50k values as a dict (key=MHC names) of dicts (key=peptide length).
        '''
        raise NotImplementedError
    