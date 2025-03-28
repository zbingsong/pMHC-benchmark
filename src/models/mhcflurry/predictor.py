from mhcflurry import Class1PresentationPredictor
import pandas as pd
import torch
import time
import typing
from . import BasePredictor, PredictorConfigs, SuppressStdout


class MHCflurryPredictor(BasePredictor):
    tasks = None
    _predictor = None
    _log50k_base = None
    _exclude_mhc_prefix = None

    # @typing.override
    @classmethod
    def load(cls, predictor_configs: PredictorConfigs) -> None:
        cls.tasks = ['EL', 'BA']
        cls._predictor = Class1PresentationPredictor.load()
        cls._log50k_base = torch.log(torch.tensor(50000, dtype=torch.double))
        cls._exclude_mhc_prefix = ('H2-Qa2', 'Ceat-', 'BoLA-A', 'BoLA-D', 'BoLA-H', 'BoLA-J', 'BoLA-T', 'Mamu-A2', 'Mamu-A7', 'Mamu-A11')

    # @typing.override
    @classmethod
    def run_retrieval(
            cls,
            df: pd.DataFrame
    ) -> tuple[tuple[dict[str, dict[str, torch.DoubleTensor]], ...], dict[str, dict[str, torch.LongTensor]], dict[str, dict[str, torch.DoubleTensor]], int, int]:
        df, num_skipped = cls._filter(df)
        if len(df) == 0:
            print('No valid peptides')
            return ({},), {}, {}, 0, num_skipped
        df = df.groupby('mhc_name')

        affinity_preds = {}
        presentation_preds = {}
        labels = {}
        log50ks = {}
        times = []

        for mhc_name, group in df:
            formatted_mhc_name = cls.__format_mhc(mhc_name)
            if formatted_mhc_name.startswith(('Mamu-A*1', 'Mamu-A*2')):
                print(f'MHC {mhc_name} not supported, skip {len(group)} peptide')
                num_skipped += len(group)
                continue
            
            group.reset_index(drop=True, inplace=True)

            affinity_pred = {}
            presentation_pred = {}
            label = {}
            log50k = {}

            grouped_by_len = group.groupby(group['peptide'].str.len())

            for length, subgroup in grouped_by_len:
                peptides = subgroup['peptide'].tolist()

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

        return (presentation_preds, affinity_preds), labels, log50ks, sum(times), num_skipped
    
    # @typing.override
    @classmethod
    def run_sq(
            cls, 
            df: pd.DataFrame
    ) -> tuple[tuple[dict[str, dict[str, torch.DoubleTensor]], ...], dict[str, dict[str, torch.LongTensor]], dict[str, dict[str, torch.DoubleTensor]], int, int]:
        return cls.run_retrieval(df)

    # @typing.override
    @classmethod
    def run_sensitivity(
            cls,
            df: pd.DataFrame
    ) -> tuple[tuple[dict[str, torch.DoubleTensor], ...], dict[str, torch.DoubleTensor]]:
        if_ba = 'log50k1' in df.columns
        df, _ = cls._filter_sensitivity(df)
        if len(df) == 0:
            print('No valid peptides')
            return ({},), {}
        df = df.groupby('mhc_name')

        affinity_preds_diff = {}
        presentation_preds_diff = {}
        labels_diff = {}
        log50ks_diff = {}

        for mhc_name, group in df:
            formatted_mhc_name = cls.__format_mhc(mhc_name)
            if formatted_mhc_name.startswith(('Mamu-A*1', 'Mamu-A*2')):
                print(f'MHC {mhc_name} not supported, skip {len(group)} peptide')
                continue
            group.reset_index(drop=True, inplace=True)
            
            affinity_pred_diff = None
            presentation_pred_diff = None
            label_diff = None
            log50k_diff = None

            peptides1 = group['peptide1'].tolist()
            peptides2 = group['peptide2'].tolist()
            try:
                result_df1 = cls._predictor.predict(peptides1, [formatted_mhc_name], verbose=0, include_affinity_percentile=True)
                result_df2 = cls._predictor.predict(peptides2, [formatted_mhc_name], verbose=0, include_affinity_percentile=True)
            except Exception as e:
                print(e)
                continue
            affinity_pred1 = 1 - torch.log(torch.tensor(result_df1['affinity'].tolist(), dtype=torch.double)) / cls._log50k_base
            affinity_pred2 = 1 - torch.log(torch.tensor(result_df2['affinity'].tolist(), dtype=torch.double)) / cls._log50k_base
            affinity_pred_diff = affinity_pred1 - affinity_pred2
            presentation_pred1 = torch.tensor(result_df1['presentation_score'].tolist(), dtype=torch.double)
            presentation_pred2 = torch.tensor(result_df2['presentation_score'].tolist(), dtype=torch.double)
            presentation_pred_diff = presentation_pred1 - presentation_pred2
            if if_ba:
                log50k1 = torch.tensor(group['log50k1'].tolist(), dtype=torch.double)
                log50k2 = torch.tensor(group['log50k2'].tolist(), dtype=torch.double)
                log50k_diff = log50k1 - log50k2
            else:
                label1 = torch.tensor(group['label1'].tolist(), dtype=torch.long)
                label2 = torch.tensor(group['label2'].tolist(), dtype=torch.long)
                label_diff = label1 - label2

            affinity_preds_diff[mhc_name] = affinity_pred_diff
            presentation_preds_diff[mhc_name] = presentation_pred_diff
            labels_diff[mhc_name] = label_diff
            log50ks_diff[mhc_name] = log50k_diff

        if if_ba:
            return (presentation_preds_diff, affinity_preds_diff), log50ks_diff
        else:
            return (presentation_preds_diff, affinity_preds_diff), labels_diff

    # @typing.override
    @classmethod
    def _filter(cls, df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
        filtered = df
        filtered = filtered[~filtered['mhc_name'].str.startswith(cls._exclude_mhc_prefix)]
        filtered = filtered[~filtered['peptide'].str.contains(r'[BJOUXZ]', regex=True)]
        filtered = filtered[filtered['peptide'].str.len() <= 15]
        filtered = filtered[filtered['peptide'].str.len() >= 8]
        if len(df) != len(filtered):
            filtered = filtered.reset_index(drop=True)
            print('Skipped peptides: ', len(df) - len(filtered))
        return filtered, len(df) - len(filtered)
    
    # @typing.override
    @classmethod
    def _filter_sensitivity(cls, df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
        filtered = df
        filtered = filtered[~filtered['mhc_name'].str.startswith(cls._exclude_mhc_prefix)]
        filtered = filtered[~filtered['peptide1'].str.contains(r'[BJOUXZ]', regex=True)]
        filtered = filtered[filtered['peptide1'].str.len() <= 15]
        filtered = filtered[filtered['peptide1'].str.len() >= 8]
        filtered = filtered[~filtered['peptide2'].str.contains(r'[BJOUXZ]', regex=True)]
        filtered = filtered[filtered['peptide2'].str.len() <= 15]
        filtered = filtered[filtered['peptide2'].str.len() >= 8]
        if len(df) != len(filtered):
            filtered = filtered.reset_index(drop=True)
            print('Skipped peptides: ', len(df) - len(filtered))
        return filtered, len(df) - len(filtered)
    
    @classmethod
    def __format_mhc(cls, mhc_name: str) -> str:
        if mhc_name.startswith('Eqca-'):
            return mhc_name[:-5] + '*' + mhc_name[-4:]
        elif mhc_name.startswith('Mamu-A1:'):
            return 'Mamu-A*' + mhc_name[-4:-2] + ':' + mhc_name[-2:]
        elif mhc_name.startswith('Mamu-B:'):
            return 'Mamu-B*' + mhc_name[-4:-2] + ':' + mhc_name[-2:]
        elif mhc_name.startswith('BoLA-'):
            return mhc_name.replace('0', '*', 1)
        return mhc_name
    