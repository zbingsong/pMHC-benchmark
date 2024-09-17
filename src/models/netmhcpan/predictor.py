import pandas as pd
import torch

import subprocess
import os
import json
import time
import pathlib

from . import BasePredictor


class NetMHCpanPredictor(BasePredictor):
    tasks = None
    _executable = None

    @classmethod
    def load(cls) -> None:
        cls.tasks = ['BA', 'EL']
        curr_dir = pathlib.Path(__file__).parent
        with open(f'{curr_dir}/configs.json', 'r') as f:
            configs = json.load(f)
            cls._executable = os.path.expanduser(configs['exe_path'])

    @classmethod
    def run_retrieval(
            cls,
            df: pd.api.typing.DataFrameGroupBy
    ) -> tuple[tuple[tuple[torch.DoubleTensor, ...], torch.LongTensor, int], tuple[tuple[list[torch.DoubleTensor], ...], list[torch.LongTensor], list[torch.DoubleTensor], int]]:
        BA_preds = []
        EL_preds = []
        labels = []
        log50ks = []
        times = []
        for mhc_name, group in df:
            group = group.reset_index(drop=True)
            peptides = group['peptide'].tolist()
            with open('peptides.txt', 'w') as file:
                for peptide in peptides:
                    file.write(peptide + '\n')
            start_time = time.time_ns()
            run_result = subprocess.run([cls._executable, '-p', 'peptides.txt', '-a', mhc_name, '-l', '8,9,10,11,12,13,14', '-BA', '-xls', '-xlsfile', 'out.tsv'])
            end_time = time.time_ns()
            assert run_result.returncode == 0
            result_df = pd.read_csv('out.tsv', sep='\t', skiprows=[0])
            BA_preds.append(100 - torch.tensor(result_df['BA_Rank'].tolist(), dtype=torch.double))
            EL_preds.append(100 - torch.tensor(result_df['EL_Rank'].tolist(), dtype=torch.double))
            labels.append(torch.tensor(group['label'].tolist(), dtype=torch.long))
            if 'log50k' in group.columns:
                log50ks.append(torch.tensor(group['log50k'].tolist(), dtype=torch.double))
            assert len(result_df) == len(group)
            times.append(end_time - start_time)
        os.remove('out.tsv')
        os.remove('peptides.txt')
        return (BA_preds, EL_preds), labels, log50ks, sum(times)

    @classmethod
    def run_sensitivity(
            cls,
            df: pd.api.typing.DataFrameGroupBy
    ) -> tuple[tuple[torch.DoubleTensor], torch.DoubleTensor]:
        BA_preds1 = []
        BA_preds2 = []
        EL_preds1 = []
        EL_preds2 = []
        log50ks1 = []
        log50ks2 = []
        for mhc_name, group in df:
            group = group.reset_index(drop=True)
            peptides1 = group['peptide1'].tolist()
            peptides2 = group['peptide2'].tolist()
            with open('peptides1.txt', 'w') as file:
                for peptide in peptides1:
                    file.write(peptide + '\n')
            with open('peptides2.txt', 'w') as file:
                for peptide in peptides2:
                    file.write(peptide + '\n')
            process1 = subprocess.Popen([cls._executable, '-p', 'peptides1.txt', '-a', mhc_name, '-l', '8,9,10,11,12,13,14', '-BA', '-xls', '-xlsfile', 'out1.tsv'], stdout=subprocess.DEVNULL)
            process2 = subprocess.Popen([cls._executable, '-p', 'peptides2.txt', '-a', mhc_name, '-l', '8,9,10,11,12,13,14', '-BA', '-xls', '-xlsfile', 'out2.tsv'], stdout=subprocess.DEVNULL)
            process1.wait()
            process2.wait()
            assert process1.returncode == 0 and process2.returncode == 0
            result_df1 = pd.read_csv('out1.tsv', sep='\t', skiprows=[0])
            result_df2 = pd.read_csv('out2.tsv', sep='\t', skiprows=[0])
            BA_preds1.append(100 - torch.tensor(result_df1['BA_Rank'].tolist(), dtype=torch.double))
            BA_preds2.append(100 - torch.tensor(result_df2['BA_Rank'].tolist(), dtype=torch.double))
            EL_preds1.append(100 - torch.tensor(result_df1['EL_Rank'].tolist(), dtype=torch.double))
            EL_preds2.append(100 - torch.tensor(result_df2['EL_Rank'].tolist(), dtype=torch.double))
            log50ks1.append(torch.tensor(group['log50k1'].tolist(), dtype=torch.double))
            log50ks2.append(torch.tensor(group['log50k2'].tolist(), dtype=torch.double))
        BA_preds_diff = torch.cat(BA_preds1) - torch.cat(BA_preds2)
        EL_preds_diff = torch.cat(EL_preds1) - torch.cat(EL_preds2)
        log50ks_diff = torch.cat(log50ks1) - torch.cat(log50ks2)
        os.remove('out1.tsv')
        os.remove('out2.tsv')
        os.remove('peptides1.txt')
        os.remove('peptides2.txt')
        return (BA_preds_diff, EL_preds_diff), log50ks_diff
