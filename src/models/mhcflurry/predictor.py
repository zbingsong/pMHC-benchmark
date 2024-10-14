from mhcflurry import Class1PresentationPredictor
import pandas.api.typing as pd_typing
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
            grouped_df: pd_typing.DataFrameGroupBy
    ) -> tuple[tuple[list[torch.DoubleTensor], ...], dict[str, dict[str, torch.LongTensor]], dict[str, dict[str, torch.DoubleTensor]], int]:
        affinity_preds = {}
        presentation_preds = {}
        labels = {}
        log50ks = {}
        times = []

        for mhc_name, group in grouped_df:
            affinity_pred = {}
            presentation_pred = {}
            label = {}
            log50k = {}
            group = group.reset_index(drop=True)
            grouped_by_len = group.groupby(group['peptide'].str.len())

            for length, subgroup in grouped_by_len:
                peptides = subgroup['peptide'].tolist()
                start_time = time.time_ns()
                result_df = cls._predictor.predict(peptides, [mhc_name], verbose=0, include_affinity_percentile=True)
                end_time = time.time_ns()
                affinity_pred[length] = 1 - torch.log(torch.tensor(result_df['affinity'].tolist(), dtype=torch.double)) / cls._log50k_base
                presentation_pred[length] = torch.tensor(result_df['presentation_score'].tolist(), dtype=torch.double)
                label[length] = torch.tensor(subgroup['label'].tolist(), dtype=torch.long)
                if 'log50k' in subgroup.columns:
                    log50k[length] = torch.tensor(subgroup['log50k'].tolist(), dtype=torch.double)
                times.append(end_time - start_time)

            affinity_preds[mhc_name] = affinity_pred
            presentation_preds[mhc_name] = presentation_pred
            labels[mhc_name] = label
            log50ks[mhc_name] = log50k

        return (presentation_preds, affinity_preds), labels, log50ks, sum(times)

    @classmethod
    def run_sensitivity(
            cls,
            df: pd_typing.DataFrameGroupBy
    ) -> tuple[tuple[dict[str, dict[str, torch.DoubleTensor]], ...], dict[str, dict[str, torch.DoubleTensor]]]:
        affinity_preds_diff = {}
        presentation_preds_diff = {}
        log50ks_diff = {}

        for mhc_name, group in df:
            affinity_pred_diff = {}
            presentation_pred_diff = {}
            log50k_diff = {}
            group = group.reset_index(drop=True)
            grouped_by_len = group.groupby(group['peptide1'].str.len())

            for length, subgroup in grouped_by_len:
                peptides1 = subgroup['peptide1'].tolist()
                peptides2 = subgroup['peptide2'].tolist()
                try:
                    result_df1 = cls._predictor.predict(peptides1, [mhc_name], verbose=0, include_affinity_percentile=True)
                    result_df2 = cls._predictor.predict(peptides2, [mhc_name], verbose=0, include_affinity_percentile=True)
                except Exception as e:
                    print(e)
                    continue
                affinity_pred1 = 1 - torch.log(torch.tensor(result_df1['affinity'].tolist(), dtype=torch.double)) / cls._log50k_base
                affinity_pred2 = 1 - torch.log(torch.tensor(result_df2['affinity'].tolist(), dtype=torch.double)) / cls._log50k_base
                presentation_pred1 = torch.tensor(result_df1['presentation_score'].tolist(), dtype=torch.double)
                presentation_pred2 = torch.tensor(result_df2['presentation_score'].tolist(), dtype=torch.double)
                log50k1 = torch.tensor(subgroup['log50k1'].tolist(), dtype=torch.double)
                log50k2 = torch.tensor(subgroup['log50k2'].tolist(), dtype=torch.double)

                affinity_pred_diff[length] = affinity_pred1 - affinity_pred2
                presentation_pred_diff[length] = presentation_pred1 - presentation_pred2
                log50k_diff[length] = log50k1 - log50k2

        affinity_preds_diff[mhc_name] = affinity_pred_diff
        presentation_preds_diff[mhc_name] = presentation_pred_diff
        log50ks_diff[mhc_name] = log50k_diff

        return (presentation_preds_diff, affinity_preds_diff), log50ks_diff
