from mhcflurry import Class1PresentationPredictor
import pandas as pd
import torch

import time

from . import BasePredictor


class MHCflurryPredictor(BasePredictor):
    tasks = None
    _predictor = None
    _log50k_base = None

    @classmethod
    def load(cls) -> None:
        cls.tasks = ['EL', 'BA']
        cls._predictor = Class1PresentationPredictor.load()
        cls._log50k_base = torch.log(torch.tensor(50000, dtype=torch.double))

    @classmethod
    def run_retrieval(
            cls,
            grouped_df: pd.api.typing.DataFrameGroupBy
    ) -> tuple[tuple[list[torch.DoubleTensor], ...], list[torch.LongTensor], list[torch.DoubleTensor], int]:
        affinity_preds = []
        presentation_preds = []
        labels = []
        log50ks = []
        times = []
        for mhc_name, group in grouped_df:
            group = group.reset_index(drop=True)
            peptides = group['peptide'].tolist()
            try:
                start_time = time.time_ns()
                result_df = cls._predictor.predict(peptides, [mhc_name], verbose=0, include_affinity_percentile=True)
                end_time = time.time_ns()
            except Exception as e:
                print(e)
                continue
            times.append(end_time - start_time)
            affinity_preds.append(1 - torch.log(torch.tensor(result_df['affinity'].tolist(), dtype=torch.double)) / cls._log50k_base)
            presentation_preds.append(torch.tensor(result_df['presentation_score'].tolist(), dtype=torch.double))
            labels.append(torch.tensor(group['label'].tolist(), dtype=torch.long))
            if 'log50k' in group.columns:
                log50ks.append(torch.tensor(group['log50k'].tolist(), dtype=torch.double))
        return (presentation_preds, affinity_preds), labels, log50ks, sum(times)

    @classmethod
    def run_sensitivity(
            cls,
            df: pd.api.typing.DataFrameGroupBy
    ) -> tuple[tuple[torch.DoubleTensor, torch.DoubleTensor], torch.DoubleTensor]:
        affinity_preds1 = []
        affinity_preds2 = []
        presentation_preds1 = []
        presentation_preds2 = []
        log50ks1 = []
        log50ks2 = []
        for mhc_name, group in df:
            group = group.reset_index(drop=True)
            peptides1 = group['peptide1'].tolist()
            peptides2 = group['peptide2'].tolist()
            try:
                result_df1 = cls._predictor.predict(peptides1, [mhc_name], verbose=0, include_affinity_percentile=True)
                result_df2 = cls._predictor.predict(peptides2, [mhc_name], verbose=0, include_affinity_percentile=True)
            except Exception as e:
                print(e)
                continue
            affinity_preds1.append(1 - torch.log(torch.tensor(result_df1['affinity'].tolist(), dtype=torch.double)) / cls._log50k_base)
            affinity_preds2.append(1 - torch.log(torch.tensor(result_df2['affinity'].tolist(), dtype=torch.double)) / cls._log50k_base)
            presentation_preds1.append(torch.tensor(result_df1['presentation_score'].tolist(), dtype=torch.double))
            presentation_preds2.append(torch.tensor(result_df2['presentation_score'].tolist(), dtype=torch.double))
            log50ks1.append(torch.tensor(group['log50k1'].tolist(), dtype=torch.double))
            log50ks2.append(torch.tensor(group['log50k2'].tolist(), dtype=torch.double))
        affinity_preds_diff = torch.cat(affinity_preds1) - torch.cat(affinity_preds2)
        presentation_preds_diff = torch.cat(presentation_preds1) - torch.cat(presentation_preds2)
        log50ks_diff = torch.cat(log50ks1) - torch.cat(log50ks2)
        return (presentation_preds_diff, affinity_preds_diff), log50ks_diff
