import pandas as pd
import pandas.core.groupby.generic as pd_typing
import torch

import os
import subprocess
import json
import time
import pathlib

from . import BasePredictor


class MHCfoveaPredictor(BasePredictor):
    tasks = None
    _exe_dir = None
    _unknown_mhc = None
    _wd = None

    @classmethod
    def load(cls) -> None:
        cls.tasks = ['EL']
        cls._wd = os.getcwd()
        curr_dir = pathlib.Path(__file__).parent
        with open(f'{curr_dir}/configs.json', 'r') as f:
            configs = json.load(f)
            cls._exe_dir = os.path.expanduser(configs['exe_dir'])
            cls._unknown_mhc = os.path.expanduser(configs['unknown_mhc'])

    @classmethod
    def run_retrieval(
            cls,
            df: pd_typing.DataFrameGroupBy
    ) -> tuple[tuple[dict[str, dict[str, torch.DoubleTensor]], ...], dict[str, dict[str, torch.LongTensor]], dict[str, dict[str, torch.DoubleTensor]], int]:
        preds = {}
        labels = {}
        log50ks = {}
        times = []

        for mhc_name, group in df:
            if not mhc_name.startswith(('HLA-A', 'HLA-B', 'HLA-C')):
                print(f'Unknown MHC name: {mhc_name}')
                if cls._unknown_mhc == 'ignore':
                    continue
                elif cls._unknown_mhc == 'error':
                    raise ValueError(f'Unknown MHC name: {mhc_name}')
            
            pred = {}
            label = {}
            log50k = {}
            # peptide should contain none of B, J, O, U, X, Z
            group = group[~group['peptide'].str.contains(r'[BJOUXZ]', regex=True)]
            group = group[group['peptide'].str.len() <= 16]
            group = group.reset_index(drop=True)
            if len(group) == 0:
                print(f'No valid peptides for {mhc_name}')
                continue
            if ':' in mhc_name:
                mhc_formatted = mhc_name[4] + '*' + mhc_name[5:]
            else:
                mhc_formatted = mhc_name[4] + '*' + mhc_name[5:7] + ':' + mhc_name[7:]

            print(f'Running retrieval for {mhc_name}')

            input_df = pd.DataFrame({'sequence': group['peptide'], 'mhc': [mhc_formatted] * len(group)})
            input_df.to_csv('peptides.csv', index=False)
                    
            start_time = time.time_ns()
            run_result = subprocess.run(['env/bin/python', 'mhcfovea/predictor.py', f'{cls._wd}/peptides.csv', 'results'], cwd=cls._exe_dir)
            end_time = time.time_ns()
            assert run_result.returncode == 0
            times.append(end_time - start_time)
            try:
                result_df = pd.read_csv(f'{cls._exe_dir}/results/prediction.csv')
                assert len(result_df) == len(group), f'Length mismatch: {len(result_df)} vs {len(group)} for {mhc_name}'
            except Exception as e:
                print(mhc_name, ' failed')
                raise e
            
            result_df['label'] = group['label']
            if 'log50k' in group.columns:
                result_df['log50k'] = group['log50k']
            # print(result_df.columns)
            grouped_by_len = result_df.groupby(result_df['sequence'].str.len())
            for length, subgroup in grouped_by_len:
                # print(subgroup.columns)
                pred[length] = torch.tensor((1 - subgroup['%rank']).tolist(), dtype=torch.double)
                label[length] = torch.tensor(subgroup['label'].tolist(), dtype=torch.long)
                if 'log50k' in subgroup.columns:
                    log50k[length] = torch.tensor(subgroup['log50k'].tolist(), dtype=torch.double)
            
            preds[mhc_name] = pred
            labels[mhc_name] = label
            log50ks[mhc_name] = log50k

        if os.path.exists('peptides.csv'):
            os.remove('peptides.csv')
        return (preds,), labels, log50ks, sum(times)
            
    @classmethod
    def run_sensitivity(
            cls,
            df: pd_typing.DataFrameGroupBy
    ) -> tuple[tuple[dict[str, dict[str, torch.DoubleTensor]], ...], dict[str, dict[str, torch.DoubleTensor]]]:
        preds_diff = {}
        log50ks_diff = {}

        for mhc_name, group in df:
            if not mhc_name.startswith('HLA-'):
                print(f'Unknown MHC name: {mhc_name}')
                if cls._unknown_mhc == 'ignore':
                    continue
                elif cls._unknown_mhc == 'error':
                    raise ValueError(f'Unknown MHC name: {mhc_name}')
                
            pred_diff = {}
            log50k_diff = {}
            group = group[~group['peptide1'].str.contains(r'[BJOUXZ]', regex=True)]
            group = group[~group['peptide2'].str.contains(r'[BJOUXZ]', regex=True)]
            group = group[group['peptide1'].str.len() <= 16]
            group = group[group['peptide2'].str.len() <= 16]
            group = group.reset_index(drop=True)
            if len(group) == 0:
                print(f'No valid peptides for {mhc_name}')
                continue
            mhc_formatted = mhc_name[4] + '*' + mhc_name[5:]

            peptides1 = group['peptide1'].tolist()
            peptides2 = group['peptide2'].tolist()
            input_df = pd.DataFrame({'sequence': peptides1 + peptides2, 'mhc': [mhc_formatted] * (len(peptides1) + len(peptides2))})
            input_df.to_csv('peptides.csv', index=False)

            run_result = subprocess.run(['env/bin/python', 'mhcfovea/predictor.py', f'{cls._wd}/peptides.csv', 'results'], cwd=cls._exe_dir)
            assert run_result.returncode == 0
            try:
                result_df = pd.read_csv(f'{cls._exe_dir}/results/prediction.csv')
                assert len(result_df) == 2 * len(group), f'Length mismatch: {len(result_df)} vs {2 * len(group)} for {mhc_name}'
            except Exception as e:
                print(mhc_name, ' failed')
                raise e
            
            result_df1 = result_df.iloc[:len(group)]
            result_df2 = result_df.iloc[len(group):]
            parallel_df = pd.DataFrame({'length': group['peptide1'].str.len(), 'pred1': 1 - result_df1['%rank'], 'pred2': 1 - result_df2['%rank'], 'log50k1': group['log50k1'], 'log50k2': group['log50k2']})
            grouped_by_len = parallel_df.groupby(parallel_df['length'])
            for length, subgroup in grouped_by_len:
                pred1 = torch.tensor(subgroup['pred1'].tolist(), dtype=torch.double)
                pred2 = torch.tensor(subgroup['pred2'], dtype=torch.double)
                log50k1 = torch.tensor(subgroup['log50k1'].tolist(), dtype=torch.double)
                log50k2 = torch.tensor(subgroup['log50k2'].tolist(), dtype=torch.double)
                pred_diff[length] = pred1 - pred2
                log50k_diff[length] = log50k1 - log50k2

        preds_diff[mhc_name] = pred_diff
        log50ks_diff[mhc_name] = log50k_diff

        os.remove('peptides.csv')
        return (preds_diff,), log50ks_diff
