import pandas as pd
import pandas.core.groupby.generic as pd_typing
import torch

import os
import subprocess
import json
import time
import pathlib

from . import BasePredictor


class MixMHCpred30Predictor(BasePredictor):
    tasks = None
    _executable = None
    _unknown_mhc = None

    @classmethod
    def load(cls) -> None:
        cls.tasks = ['EL']
        curr_dir = pathlib.Path(__file__).parent
        with open(f'{curr_dir}/configs.json', 'r') as f:
            configs = json.load(f)
            cls._executable = os.path.expanduser(configs['exe_path'])
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
            
            print(f'Running retrieval for {mhc_name}')

            pred = {}
            label = {}
            log50k = {}
            # peptide should contain none of B, J, O, U, X, Z
            filtered = group[~group['peptide'].str.contains(r'[BJOUXZ]', regex=True)]
            filtered = filtered[filtered['peptide'].str.len() <= 14]
            filtered = filtered.reset_index(drop=True)
            if len(group) != len(filtered):
                print('Skipped peptides:', len(group) - len(filtered))
            group = filtered
            if len(group) == 0:
                print(f'No valid peptides for {mhc_name}')
                continue

            mhc_formatted = mhc_name[4:].replace(':', '')

            peptides = group['peptide'].tolist()
            with open(f'peptides.fasta', 'w') as f:
                for peptide in peptides:
                    f.write(f'>{peptide}\n{peptide}\n')
                    
            start_time = time.time_ns()
            run_result = subprocess.run([cls._executable, '-i', 'peptides.fasta', '-o', 'result.tsv', '-a', mhc_formatted], stdout=subprocess.DEVNULL)
            end_time = time.time_ns()
            assert run_result.returncode == 0
            times.append(end_time - start_time)
            try:
                result_df = pd.read_csv('result.tsv', sep='\t', skiprows=list(range(11)))
            except Exception as e:
                print(mhc_formatted, ' failed')
                raise e
            result_df['label'] = group['label']
            if 'log50k' in group.columns:
                result_df['log50k'] = group['log50k']
            grouped_by_len = result_df.groupby(result_df['Peptide'].str.len())
            for length, subgroup in grouped_by_len:
                pred[length] = torch.tensor((1 - subgroup['%Rank_bestAllele']).tolist(), dtype=torch.double)
                label[length] = torch.tensor(subgroup['label'].tolist(), dtype=torch.long)
                if 'log50k' in subgroup.columns:
                    log50k[length] = torch.tensor(subgroup['log50k'].tolist(), dtype=torch.double)
            
            preds[mhc_name] = pred
            labels[mhc_name] = label
            log50ks[mhc_name] = log50k

        if os.path.exists('peptides.txt'):
            os.remove('peptides.txt')
            os.remove('result.tsv')
        return (preds,), labels, log50ks, sum(times)
    
    @classmethod
    def run_sq(
            cls, 
            df: pd_typing.DataFrameGroupBy
    ) -> tuple[tuple[dict[str, dict[str, torch.DoubleTensor]], ...], dict[str, dict[str, torch.LongTensor]], dict[str, dict[str, torch.DoubleTensor]], int]:
        return cls.run_retrieval(df)
            
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
            mhc_formatted = mhc_name[4:].replace(':', '')

            for length, subgroup in grouped_by_len:
                if length > 14:
                    continue
                peptides1 = subgroup['peptide1'].tolist()
                with open(f'peptides.fasta', 'w') as f:
                    for peptide in peptides1:
                        f.write(f'>{peptide}\n{peptide}\n')

                run_result = subprocess.run([cls._executable, '-i', 'peptides.fasta', '-o', 'result.tsv', '-a', mhc_formatted])
                assert run_result.returncode == 0
                result_df = pd.read_csv('result.tsv', sep='\t', skiprows=list(range(11)))
                pred1 = torch.tensor((1 - result_df['%Rank_bestAllele']).tolist(), dtype=torch.double)
                log50k1 = torch.tensor(subgroup['log50k1'].tolist(), dtype=torch.double)

                peptides2 = subgroup['peptide2'].tolist()
                with open(f'peptides.fasta', 'w') as f:
                    for peptide in peptides2:
                        f.write(f'>{peptide}\n{peptide}\n')
                run_result = subprocess.run([cls._executable, '-i', 'peptides.fasta', '-o', 'result.tsv', '-a', mhc_formatted])
                assert run_result.returncode == 0
                result_df = pd.read_csv('result.tsv', sep='\t', skiprows=list(range(11)))
                pred2 = torch.tensor((1 - result_df['%Rank_bestAllele']).tolist(), dtype=torch.double)
                log50k2 = torch.tensor(subgroup['log50k2'].tolist(), dtype=torch.double)

                pred_diff[length] = pred1 - pred2
                log50k_diff[length] = log50k1 - log50k2

        preds_diff[mhc_name] = pred_diff
        log50ks_diff[mhc_name] = log50k_diff

        os.remove('peptides.fasta')
        os.remove('result.tsv')
        return (preds_diff,), log50ks_diff
