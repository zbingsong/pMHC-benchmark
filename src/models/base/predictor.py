import pandas as pd
import torch

import abc
import dataclasses


@dataclasses.dataclass
class PredictorConfigs(object):
    '''
    Dataclass for loading predictor, same configs for all predictors
    '''
    temp_dir: str
    # log_dir: str
    

class BasePredictor(abc.ABC):
    tasks = None

    @classmethod
    @abc.abstractmethod
    def load(cls, predictor_configs: PredictorConfigs) -> None:
        '''
        Set tasks and other predictor-specific attributes.
        Must be called before run() or run_sensitivity().
        '''
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def run_retrieval(
            cls, 
            df: pd.DataFrame
    ) -> tuple[tuple[dict[str, dict[str, torch.DoubleTensor]], ...], dict[str, dict[str, torch.LongTensor]], dict[str, dict[str, torch.DoubleTensor]], int]:
        '''
        Run the predictor on the given DataFrame.

        Parameters:
        df (pd.DataFrame): DataFrame with the following columns:
            - 'mhc_name': MHC names
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
    def run_sq(
            cls, 
            df: pd.DataFrame
    ) -> tuple[tuple[dict[str, dict[str, torch.DoubleTensor]], ...], dict[str, dict[str, torch.LongTensor]], dict[str, dict[str, torch.DoubleTensor]], int]:
        '''
        Run the predictor on the given DataFrame, from a square dataset (i.e. every MHC is paired with every peptide).

        Parameters:
        df (pd.DataFrame): DataFrame with the following columns:
            - 'mhc_name': MHC names.
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
            df: pd.DataFrame
    ) -> tuple[tuple[dict[str, dict[str, torch.DoubleTensor]], ...], dict[str, dict[str, torch.DoubleTensor]]]:
        '''
        Run the predictor on the given DataFrame for sensitivity analysis.

        Parameters:
        df (pd.DataFrame): DataFrame with the following columns:
            - 'peptide1': Peptide sequences.
            - 'peptide2': Peptide sequences.
            - 'label': Binary labels, or 'log50k': Log50k values.
        
        Returns:
        tuple[tuple[dict[str, dict[str, torch.DoubleTensor]], ...], dict[str, dict[str, torch.DoubleTensor]]]:
            - Sensitivity result, where metrics are computed per MHC and per peptide length.
                - Tuple of predictions for each task, where each prediction is a dict (key=MHC names) of dicts (key=peptide length) containg differences in predictions.
                - Label or Log50k values as a dict (key=MHC names) of dicts (key=peptide length) containg differences in labels or log50ks, depending on input dataframe.
        '''
        raise NotImplementedError
