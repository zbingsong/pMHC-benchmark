import pandas as pd
import pandas.api.typing as pd_typing
import torch

import os
import subprocess
import json
import time
import pathlib

from . import BasePredictor


class MixMHCpredPredictor(BasePredictor):
    tasks = None
    _executable = None

    @classmethod
    def load(cls) -> None:
        cls.tasks = ['EL']
        curr_dir = pathlib.Path(__file__).parent
        with open(f'{curr_dir}/configs.json', 'r') as f:
            configs = json.load(f)
            cls._executable = os.path.expanduser(configs['exe_path'])

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
            pred = {}
            label = {}
            log50k = {}
            group = group.reset_index(drop=True)
            grouped_by_len = group.groupby(group['peptide'].str.len())

            for length, subgroup in grouped_by_len:
                peptides = subgroup['peptide'].tolist()
                with open(f'peptides.fasta', 'w') as f:
                    for peptide in peptides:
                        f.write(f'>{peptide}\n{peptide}\n')
                mhc_formatted = mhc_name[4:].replace(':', '')
                start_time = time.time_ns()
                run_result = subprocess.run([cls._executable, '-i', 'peptides.fasta', '-o', 'result.tsv', '-a', mhc_formatted])
                end_time = time.time_ns()
                assert run_result.returncode == 0
                times.append(end_time - start_time)
                try:
                    result_df = pd.read_csv('result.tsv', sep='\t', skiprows=list(range(11)))
                except:
                    print(mhc_name, ' failed')
                pred[length] = torch.tensor((1 - result_df['%Rank_bestAllele']).tolist(), dtype=torch.double)
                label[length] = torch.tensor(subgroup['label'].tolist(), dtype=torch.long)
                if 'log50k' in subgroup.columns:
                    log50k[length] = torch.tensor(subgroup['log50k'].tolist(), dtype=torch.double)
            
            preds[mhc_name] = pred
            labels[mhc_name] = label
            log50ks[mhc_name] = log50k

        os.remove('peptides.fasta')
        os.remove('result.tsv')
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
                peptides1 = subgroup['peptide1'].tolist()
                with open(f'peptides.fasta', 'w') as f:
                    for peptide in peptides1:
                        f.write(f'>{peptide}\n{peptide}\n')
                mhc_formatted = mhc_name[4:].replace(':', '')
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
