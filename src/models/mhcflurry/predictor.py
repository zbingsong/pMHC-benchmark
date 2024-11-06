from mhcflurry import Class1PresentationPredictor
import pandas as pd
import torch

import time
import json
import os
import pathlib
import sys
import io

from . import BasePredictor, SuppressStdout


class MHCflurryPredictor(BasePredictor):
    tasks = None
    _predictor = None
    _log50k_base = None
    _unknown_peptide = None
    _exclude_mhc_prefix = None

    @classmethod
    def load(cls) -> None:
        cls.tasks = ['EL', 'BA']
        cls._predictor = Class1PresentationPredictor.load()
        cls._log50k_base = torch.log(torch.tensor(50000, dtype=torch.double))
        cls._exclude_mhc_prefix = ('H2-Qa2', 'Ceat-', 'BoLA-A', 'BoLA-D', 'BoLA-H', 'BoLA-J', 'BoLA-T', 'Mamu-A2', 'Mamu-A7', 'Mamu-A11')
        curr_dir = pathlib.Path(__file__).parent
        with open(f'{curr_dir}/configs.json', 'r') as f:
            configs = json.load(f)
            cls._unknown_peptide = os.path.expanduser(configs['unknown_peptide'])

    @classmethod
    def run_retrieval(
            cls,
            df: pd.DataFrame
    ) -> tuple[tuple[list[torch.DoubleTensor], ...], dict[str, dict[str, torch.LongTensor]], dict[str, dict[str, torch.DoubleTensor]], int]:
        df = df.groupby('mhc_name')

        affinity_preds = {}
        presentation_preds = {}
        labels = {}
        log50ks = {}
        times = []

        for mhc_name, group in df:
            if mhc_name.startswith(cls._exclude_mhc_prefix):
                print(f'Excluded MHC: {mhc_name}')
                continue   
            
            if mhc_name.startswith('Eqca-'):
                mhc_name = mhc_name[:-5] + '*' + mhc_name[-4:]
            elif mhc_name.startswith('Mamu-A1:'):
                mhc_name = 'Mamu-A*' + mhc_name[-4:-2] + ':' + mhc_name[-2:]
                if mhc_name.startswith(('Mamu-A*1', 'Mamu-A*2')):
                    print(f'MHC {mhc_name} not supported')
                    continue
            elif mhc_name.startswith('Mamu-B:'):
                mhc_name = 'Mamu-B*' + mhc_name[-4:-2] + ':' + mhc_name[-2:]

            print(f'Predicting for MHC: {mhc_name}')

            affinity_pred = {}
            presentation_pred = {}
            label = {}
            log50k = {}
            group = group.reset_index(drop=True)
            group = group[~group['peptide'].str.contains(r'[BJOUXZ]', regex=True)]
            grouped_by_len = group.groupby(group['peptide'].str.len())

            for length, subgroup in grouped_by_len:
                if length > 15:
                    print(f'Peptide length {length} is too long for MHC {mhc_name}')
                    continue

                peptides = subgroup['peptide'].tolist()
                formatted_mhc_name = mhc_name
                if mhc_name.startswith('BoLA-'):
                    # replace the first 0 with *
                    formatted_mhc_name = mhc_name.replace('0', '*', 1)

                start_time = time.time_ns()
                with SuppressStdout():
                    result_df = cls._predictor.predict(peptides, [formatted_mhc_name], verbose=0, include_affinity_percentile=True)
                end_time = time.time_ns()

                affinity_pred[length] = 1.0 - torch.log(torch.tensor(result_df['affinity'].tolist(), dtype=torch.double)) / cls._log50k_base
                presentation_pred[length] = 100.0 - torch.tensor(result_df['presentation_percentile'].tolist(), dtype=torch.double)
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
    def run_sq(
            cls, 
            df: pd.DataFrame
    ) -> tuple[tuple[dict[str, dict[str, torch.DoubleTensor]], ...], dict[str, dict[str, torch.LongTensor]], dict[str, dict[str, torch.DoubleTensor]], int]:
        return cls.run_retrieval(df)

    @classmethod
    def run_sensitivity(
            cls,
            df: pd.DataFrame
    ) -> tuple[tuple[dict[str, dict[str, torch.DoubleTensor]], ...], dict[str, dict[str, torch.DoubleTensor]]]:
        if_ba = 'log50k1' in df.columns

        df = df.groupby('mhc_name')

        affinity_preds_diff = {}
        presentation_preds_diff = {}
        labels_diff = {}
        log50ks_diff = {}

        for mhc_name, group in df:
            affinity_pred_diff = {}
            presentation_pred_diff = {}
            label_diff = {}
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
                affinity_pred_diff[length] = affinity_pred1 - affinity_pred2
                presentation_pred1 = torch.tensor(result_df1['presentation_score'].tolist(), dtype=torch.double)
                presentation_pred2 = torch.tensor(result_df2['presentation_score'].tolist(), dtype=torch.double)
                presentation_pred_diff[length] = presentation_pred1 - presentation_pred2
                if if_ba:
                    log50k1 = torch.tensor(subgroup['log50k1'].tolist(), dtype=torch.double)
                    log50k2 = torch.tensor(subgroup['log50k2'].tolist(), dtype=torch.double)
                    log50k_diff[length] = log50k1 - log50k2
                else:
                    label1 = torch.tensor(subgroup['label1'].tolist(), dtype=torch.long)
                    label2 = torch.tensor(subgroup['label2'].tolist(), dtype=torch.long)
                    label_diff[length] = label1 - label2

        affinity_preds_diff[mhc_name] = affinity_pred_diff
        presentation_preds_diff[mhc_name] = presentation_pred_diff
        labels_diff[mhc_name] = label_diff
        log50ks_diff[mhc_name] = log50k_diff

        if if_ba:
            return (presentation_preds_diff, affinity_preds_diff), log50ks_diff
        else:
            return (presentation_preds_diff, affinity_preds_diff), labels_diff
