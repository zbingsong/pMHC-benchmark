import pandas as pd
import pandas.core.groupby.generic as pd_typing
import torch

import os
import subprocess
import json
import time
import pathlib

from . import BasePredictor


class TransPHLAPredictor(BasePredictor):
    tasks = None
    _exe_dir = None
    _unknown_mhc = None
    _unknown_peptide = None
    _wd = None

    @classmethod
    def load(cls) -> None:
        cls.tasks = ['Mix']
        cls._wd = os.getcwd()
        curr_dir = pathlib.Path(__file__).parent
        with open(f'{curr_dir}/configs.json', 'r') as f:
            configs = json.load(f)
            cls._exe_dir = os.path.expanduser(configs['exe_dir'])
            cls._unknown_mhc = os.path.expanduser(configs['unknown_mhc'])
            cls._unknown_peptide = os.path.expanduser(configs['unknown_peptide'])

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
            if not mhc_name.startswith('HLA-'):
                print(f'Unknown MHC name: {mhc_name}')
                if cls._unknown_mhc == 'ignore':
                    continue
                elif cls._unknown_mhc == 'error':
                    raise ValueError(f'Unknown MHC name: {mhc_name}')
                
            pred = {}
            label = {}
            log50k = {}
            # peptide should contain none of B, J, O, U, X, Z
            group = group[~group['peptide'].str.contains(r'[BJOUXZ]', regex=True)].reset_index(drop=True)
            if len(group) == 0:
                print(f'No valid peptides for {mhc_name}')
                continue
            grouped_by_len = group.groupby(group['peptide'].str.len())

            for length, subgroup in grouped_by_len:
                if length >= 15:
                    if cls._unknown_peptide == 'ignore':
                        continue
                    elif cls._unknown_peptide == 'error':
                        raise ValueError(f'Peptide for {mhc_name} is too long: {length}')
                print(f'Running {mhc_name} with {length} length peptides, number of peptides: {len(subgroup)}')
                    
                with open('mhcs.fasta', 'w') as mhc_f, open('peptides.fasta', 'w') as peptide_f:
                    for row in subgroup.itertuples():
                        mhc_f.write(f'>{row.mhc_name}\n{row.mhc_seq}\n')
                        peptide_f.write(f'>{row.peptide}\n{row.peptide}\n')

                start_time = time.time_ns()
                run_result = subprocess.run(['../env/bin/python', 'pHLAIformer.py', '--peptide_file', f'{cls._wd}/peptides.fasta', '--HLA_file', f'{cls._wd}/mhcs.fasta', '--output_dir', 'results', '--threshold', '0.5'], cwd=cls._exe_dir)
                end_time = time.time_ns()
                assert run_result.returncode == 0
                times.append(end_time - start_time)

                try:
                    result_df = pd.read_csv(f'{cls._exe_dir}/results/predict_results.csv')
                    assert len(result_df) == len(subgroup), f'Length mismatch: {len(result_df)} != {len(subgroup)} for {mhc_name}'
                except Exception as e:
                    print(mhc_name, ' failed')
                    raise e
                
                pred[length] = torch.tensor(result_df['y_prob'].tolist(), dtype=torch.double)
                label[length] = torch.tensor(subgroup['label'].tolist(), dtype=torch.long)
                if 'log50k' in subgroup.columns:
                    log50k[length] = torch.tensor(subgroup['log50k'].tolist(), dtype=torch.double)
            
            preds[mhc_name] = pred
            labels[mhc_name] = label
            log50ks[mhc_name] = log50k

        if os.path.exists('mhcs.fasta'):
            os.remove('mhcs.fasta')
            os.remove('peptides.fasta')
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
            group = group.reset_index(drop=True)
            if len(group) == 0:
                print(f'No valid peptides for {mhc_name}')
                continue
            grouped_by_len = group.groupby(group['peptide1'].str.len())

            for length, subgroup in grouped_by_len:
                if length > 14:
                    if cls._unknown_peptide == 'ignore':
                        continue
                    elif cls._unknown_peptide == 'error':
                        raise ValueError(f'Peptide for {mhc_name} is too long: {length}')
                print(f'Running {mhc_name} with {length} length peptides, number of peptides: {len(subgroup)}')
                    
                with open('mhcs.fasta', 'w') as mhc_f, open('peptides.fasta', 'w') as peptide_f:
                    for row in subgroup.itertuples():
                        mhc_f.write(f'>{row.mhc_name}\n{row.mhc_seq}\n>{row.mhc_name}\n{row.mhc_seq}\n')
                        peptide_f.write(f'>{row.peptide1}\n{row.peptide1}\n>{row.peptide2}\n{row.peptide2}\n')
                wd = os.getcwd()

                run_result = subprocess.run(['../env/bin/python', 'pHLAIformer.py', '--peptide_file', f'{wd}/peptides.fasta', '--HLA_file', f'{wd}/mhcs.fasta', '--output_dir', 'results', '--threshold', '0.5'], cwd=cls._exe_dir, stdout=subprocess.DEVNULL)
                assert run_result.returncode == 0

                try:
                    result_df = pd.read_csv(f'{cls._exe_dir}/results/predict_results.csv')
                    assert len(result_df) == 2 * len(subgroup), f'Length mismatch: {len(result_df)} != {len(subgroup)}'
                except Exception as e:
                    print(mhc_name, ' failed')
                    raise e
                
                pred1 = torch.tensor(result_df['y_prob'].tolist()[::2], dtype=torch.double)
                pred2 = torch.tensor(result_df['y_prob'].tolist()[1::2], dtype=torch.double)
                log50k1 = torch.tensor(subgroup['log50k1'].tolist(), dtype=torch.double)
                log50k2 = torch.tensor(subgroup['log50k2'].tolist(), dtype=torch.double)

                pred_diff[length] = pred1 - pred2
                log50k_diff[length] = log50k1 - log50k2

        preds_diff[mhc_name] = pred_diff
        log50ks_diff[mhc_name] = log50k_diff

        if os.path.exists('mhcs.fasta'):
            os.remove('mhcs.fasta')
            os.remove('peptides.fasta')
        return (preds_diff,), log50ks_diff
