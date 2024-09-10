import pandas as pd
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
        cls.tasks = ['BA']
        curr_dir = pathlib.Path(__file__).parent
        with open(f'{curr_dir}/configs.json', 'r') as f:
            configs = json.load(f)
            cls._executable = os.path.expanduser(configs['exe_path'])

    @classmethod
    def run(
            cls,
            df: pd.api.typing.DataFrameGroupBy
    ) -> tuple[tuple[list[torch.DoubleTensor]], list[torch.LongTensor], list[torch.DoubleTensor], int]:
        preds = []
        labels = []
        log50ks = []
        times = []
        for mhc_name, group in df:
            group = group.reset_index(drop=True)
            peptides = group['peptide'].tolist()
            with open(f'peptides.fasta', 'w') as f:
                for peptide in peptides:
                    f.write(f'>{peptide}\n{peptide}\n')
            mhc_formatted = mhc_name[4:].replace(':', '')
            start_time = time.time_ns()
            run_result = subprocess.run([cls._executable, '-i', 'peptides.fasta', '-o', 'result.tsv', '-a', mhc_formatted])
            end_time = time.time_ns()
            assert run_result.returncode == 0
            times.append(end_time - start_time)
            result_df = pd.read_csv('result.tsv', sep='\t', skiprows=list(range(11)))
            preds.append(torch.tensor((1 - result_df['%Rank_bestAllele']).tolist(), dtype=torch.double))
            labels.append(torch.tensor(group['label'].tolist(), dtype=torch.int))
            if 'log50k' in group.columns:
                log50ks.append(torch.tensor(group['log50k'].tolist(), dtype=torch.double))
        os.remove('peptides.fasta')
        os.remove('result.tsv')
        return (preds,), labels, log50ks, sum(times)
            
    @classmethod
    def run_sensitivity(
            cls,
            df: pd.api.typing.DataFrameGroupBy
    ) -> tuple[tuple[torch.DoubleTensor], torch.DoubleTensor]:
        preds1 = []
        preds2 = []
        log50ks1 = []
        log50ks2 = []
        for mhc, group in df:
            group = group.reset_index(drop=True)
            peptides1 = group['peptide1'].tolist()
            peptides2 = group['peptide2'].tolist()
            with open(f'peptides.fasta', 'w') as f:
                for peptide in peptides1:
                    f.write(f'>{peptide}\n{peptide}\n')
            mhc_formatted = mhc[4:].replace(':', '')
            run_result = subprocess.run([cls._executable, '-i', 'peptides.fasta', '-o', 'result.tsv', '-a', mhc_formatted])
            assert run_result.returncode == 0
            result_df = pd.read_csv('result.tsv', sep='\t', skiprows=list(range(11)))
            preds1.append(torch.tensor((1 - result_df['%Rank_bestAllele']).tolist(), dtype=torch.double))
            log50ks1.append(torch.tensor(group['log50k1'].tolist(), dtype=torch.double))
            with open(f'peptides.fasta', 'w') as f:
                for peptide in peptides2:
                    f.write(f'>{peptide}\n{peptide}\n')
            run_result = subprocess.run([cls._executable, '-i', 'peptides.fasta', '-o', 'result.tsv', '-a', mhc_formatted])
            assert run_result.returncode == 0
            result_df = pd.read_csv('result.tsv', sep='\t', skiprows=list(range(11)))
            preds2.append(torch.tensor((1 - result_df['%Rank_bestAllele']).tolist(), dtype=torch.double))
            log50ks2.append(torch.tensor(group['log50k2'].tolist(), dtype=torch.double))
        preds_diff = torch.cat(preds1) - torch.cat(preds2)
        log50ks_diff = torch.cat(log50ks1) - torch.cat(log50ks2)
        os.remove('peptides.fasta')
        os.remove('result.tsv')
        return (preds_diff,), log50ks_diff
