import pandas as pd
import torch

import os
import subprocess
import json
import time
import pathlib

from . import BasePredictor, SuppressStdout


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
            df: pd.DataFrame
    ) -> tuple[tuple[dict[str, dict[str, torch.DoubleTensor]], ...], dict[str, dict[str, torch.LongTensor]], dict[str, dict[str, torch.DoubleTensor]], int]:
        preds = {}
        labels = {}
        log50ks = {}
        times = []

        filtered = df
        filtered = filtered[filtered['mhc_name'].str.startswith(('HLA-A', 'HLA-B', 'HLA-C'))]
        filtered = filtered[~filtered['peptide'].str.contains(r'[BJOUXZ]', regex=True)]
        if len(df) != len(filtered):
            filtered = filtered.reset_index(drop=True)
            print('Skipped peptides: ', len(df) - len(filtered))
        if len(filtered) == 0:
            print('No valid peptides')
            return ({},), {}, {}, 0

        df = filtered
        input_df = pd.DataFrame({'sequence': df['peptide'], 'mhc': df['mhc_name'].transform(cls._format_mhc)})
        input_df.to_csv('peptides_mhcfovea.csv', index=False)
                
        start_time = time.time_ns()
        with SuppressStdout():
            run_result = subprocess.run(['env/bin/python', 'mhcfovea/predictor.py', f'{cls._wd}/peptides_mhcfovea.csv', 'results'], cwd=cls._exe_dir, stdout=subprocess.DEVNULL)
        end_time = time.time_ns()
        assert run_result.returncode == 0
        times.append(end_time - start_time)
        try:
            result_df = pd.read_csv(f'{cls._exe_dir}/results/prediction.csv')
            assert len(result_df) == len(df), f'Length mismatch: {len(result_df)} vs {len(df)}'
        except Exception as e:
            raise e
        
        result_df['label'] = df['label']
        if 'log50k' in df.columns:
            result_df['log50k'] = df['log50k']
        result_df['mhc'] = 'HLA-' + result_df['mhc'].str.replace('*', '')

        for mhc_name, group in result_df.groupby('mhc'):
            pred = {}
            label = {}
            log50k = {}
            # print(result_df.columns)
            grouped_by_len = group.groupby(group['sequence'].str.len())
            for length, subgroup in grouped_by_len:
                # print(subgroup.columns)
                pred[length] = 100.0 - torch.tensor(subgroup['%rank'].tolist(), dtype=torch.double)
                label[length] = torch.tensor(subgroup['label'].tolist(), dtype=torch.long)
                if 'log50k' in subgroup.columns:
                    log50k[length] = torch.tensor(subgroup['log50k'].tolist(), dtype=torch.double)
        
            preds[mhc_name] = pred
            labels[mhc_name] = label
            log50ks[mhc_name] = log50k

        if os.path.exists('peptides_mhcfovea.csv'):
            os.remove('peptides_mhcfovea.csv')
        return (preds,), labels, log50ks, sum(times)
    
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
        preds_diff = {}
        log50ks_diff = {}

        filtered = df
        filtered = filtered[filtered['mhc_name'].str.startswith(('HLA-A', 'HLA-B', 'HLA-C'))]
        filtered = filtered[~filtered['peptide1'].str.contains(r'[BJOUXZ]', regex=True)]
        filtered = filtered[~filtered['peptide2'].str.contains(r'[BJOUXZ]', regex=True)]
        if len(df) != len(filtered):
            filtered = filtered.reset_index(drop=True)
            print('Skipped peptides: ', len(df) - len(filtered))
        if len(filtered) == 0:
            print('No valid peptides')
            return ({},), 0

        input_df = pd.DataFrame({'sequence': pd.concat([df['peptide1'], df['peptide2']]), 'mhc': pd.concat([df['mhc_name'].transform(cls._format_mhc), df['mhc_name'].transform(cls._format_mhc)])})
        input_df.to_csv('peptides_mhcfovea.csv', index=False)
            
        with SuppressStdout():
            run_result = subprocess.run(['env/bin/python', 'mhcfovea/predictor.py', f'{cls._wd}/peptides_mhcfovea.csv', 'results'], cwd=cls._exe_dir, stdout=subprocess.DEVNULL)
        assert run_result.returncode == 0
        try:
            result_df = pd.read_csv(f'{cls._exe_dir}/results/prediction.csv')
            assert len(result_df) == 2 * len(df), f'Length mismatch: {len(result_df)} vs {2 * len(df)}'
        except Exception as e:
            raise e
        
        result_df1 = result_df.iloc[:len(df)].reset_index(drop=True)
        result_df2 = result_df.iloc[len(df):].reset_index(drop=True)
        result_df1.rename(columns={'%rank': '%rank1'}, inplace=True)
        result_df1['log50k1'] = df['log50k1']
        result_df1['%rank2'] = result_df2['%rank']
        result_df1['log50k2'] = df['log50k2']
        result_df1['mhc'] = 'HLA-' + result_df1['mhc'].str.replace('*', '')
        # result_df1 now has columns: peptide1, %rank1, log50k1, peptide2, %rank2, log50k2, mhc

        for mhc_name, group in result_df1.groupby('mhc'):
            pred_diff = {}
            log50k_diff = {}
            grouped_by_len = group.groupby(group['peptide1'].str.len())
            for length, subgroup in grouped_by_len:
                pred1 = 100.0 - torch.tensor(subgroup['%rank1'].tolist(), dtype=torch.double)
                pred2 = 100.0 - torch.tensor(subgroup['%rank2'].tolist(), dtype=torch.double)
                log50k1 = torch.tensor(subgroup['log50k1'].tolist(), dtype=torch.double)
                log50k2 = torch.tensor(subgroup['log50k2'].tolist(), dtype=torch.double)
                pred_diff[length] = pred1 - pred2
                log50k_diff[length] = log50k1 - log50k2

            preds_diff[mhc_name] = pred_diff
            log50ks_diff[mhc_name] = log50k_diff

        if os.path.exists('peptides_mhcfovea.csv'):
            os.remove('peptides_mhcfovea.csv')
        return (preds_diff,), log50ks_diff
    
    @classmethod
    def _format_mhc(cls, mhc: str) -> str:
        if ':' in mhc:
            return mhc[4] + '*' + mhc[5:]
        else:
            return mhc[4] + '*' + mhc[5:7] + ':' + mhc[7:]
