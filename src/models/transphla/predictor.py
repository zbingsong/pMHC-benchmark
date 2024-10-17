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

    @classmethod
    def load(cls) -> None:
        cls.tasks = ['Mix']
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
            group = group.reset_index(drop=True)
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
                        mhc_f.write(f'>{row.Index}\n{row.mhc_name}\n')
                        peptide_f.write(f'>{row.Index}\n{row.peptide}\n')
                wd = os.getcwd()

                start_time = time.time_ns()
                run_result = subprocess.run(['../env/bin/python', 'pHLAIformer.py', '--peptide_file', f'{wd}/peptides.fasta', '--HLA_file', f'{wd}/mhcs.fasta', '--threshold', '0.5', '--cut_peptide', 'False', '--output_dir', 'results', '--output_attention', 'False', '--output_heatmap', 'True', '--output_mutation', 'True'], cwd=cls._exe_dir)
                end_time = time.time_ns()
                assert run_result.returncode == 0
                times.append(end_time - start_time)

                try:
                    result_df = pd.read_csv(f'{cls._exe_dir}/results/predict_results.txt')
                    assert len(result_df) == len(subgroup)
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

        os.remove('peptides.txt')
        return (preds,), labels, log50ks, sum(times)
            
    @classmethod
    def run_sensitivity(
            cls,
            df: pd_typing.DataFrameGroupBy
    ) -> tuple[tuple[dict[str, dict[str, torch.DoubleTensor]], ...], dict[str, dict[str, torch.DoubleTensor]]]:
        preds_diff = {}
        log50ks_diff = {}

        for mhc_name, group in df:
            pred_diff = {}
            log50k_diff = {}
            group = group.reset_index(drop=True)
            grouped_by_len = group.groupby(group['peptide1'].str.len())

            for length, subgroup in grouped_by_len:
                with open('mhcs.fasta', 'w') as mhc_f, open('peptides.fasta', 'w') as peptide_f:
                    for row in subgroup.itertuples():
                        mhc_f.write(f'>{row.Index}\n{row.mhc_name}\n>{row.Index}\n{row.mhc_name}\n')
                        peptide_f.write(f'>{row.Index}\n{row.peptide1}\n>{row.Index}\n{row.peptide2}\n')
                wd = os.getcwd()

                run_result = subprocess.run(['env/bin/python', 'pHLAIformer.py', '--peptide_file', f'{wd}/peptides.fasta', '--HLA_file', f'{wd}/mhcs.fasta', '--threshold', '0.5', '--cut_peptide', 'False', '--output_dir', 'results', '--output_attention', 'False', '--output_heatmap', 'True', '--output_mutation', 'True'], cwd=cls._exe_dir)
                assert run_result.returncode == 0

                try:
                    result_df = pd.read_csv(f'{cls._exe_dir}/results/predict_results.txt')
                    assert len(result_df) == len(subgroup)
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

        os.remove('peptides.txt')
        return (preds_diff,), log50ks_diff
