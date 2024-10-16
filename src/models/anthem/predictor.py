import pandas.api.typing as pd_typing
import torch

import os
import subprocess
import json
import time
import pathlib
import glob
import shutil

from . import BasePredictor


class AnthemPredictor(BasePredictor):
    tasks = None
    _exe_dir = None
    _unknown_mhc = None

    @classmethod
    def load(cls) -> None:
        cls.tasks = ['Mix']
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
                peptides = subgroup['peptide'].tolist()
                with open('peptides.txt', 'w') as f:
                    for peptide in peptides:
                        f.write(f'{peptide}\n')
                mhc_formatted = mhc_name[:5] + '*' + mhc_name[5:]
                wd = os.getcwd()

                start_time = time.time_ns()
                run_result = subprocess.run(['env/bin/python', 'sware_b_main.py', '--length', str(length), '--HLA', mhc_formatted, '--mode', 'prediction', '--peptide_file', f'{wd}/peptides.txt'], cwd=cls._exe_dir)
                end_time = time.time_ns()
                assert run_result.returncode == 0
                times.append(end_time - start_time)

                # Anthem creates a new directory for each run, so we need to find the result file
                files = glob.iglob(f'{cls._exe_dir}/*')
                latest_dir = max(files, key=os.path.getctime)
                try:
                    with open(f'{latest_dir}/length_{length}_prediction_result.txt', 'r') as f:
                        for _ in range(5):
                            f.readline()
                        result = []
                        for line in f:
                            if line.startswith('-'):
                                break
                            cells = line.strip().split()
                            result.append(float(cells[-1]))
                except Exception as e:
                    print(mhc_name, ' failed')
                    raise e
                pred[length] = torch.tensor(result, dtype=torch.double)
                label[length] = torch.tensor(subgroup['label'].tolist(), dtype=torch.long)
                if 'log50k' in subgroup.columns:
                    log50k[length] = torch.tensor(subgroup['log50k'].tolist(), dtype=torch.double)
                
                shutil.rmtree(latest_dir)
            
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
                peptides1 = subgroup['peptide1'].tolist()
                peptides2 = subgroup['peptide2'].tolist()
                with open(f'peptides.txt', 'w') as f:
                    for peptide in peptides1:
                        f.write(f'{peptide}\n')
                    for peptide in peptides2:
                        f.write(f'{peptide}\n')
                mhc_formatted = mhc_name[:5] + '*' + mhc_name[5:]
                wd = os.getcwd()

                run_result = subprocess.run(['env/bin/python', 'sware_b_main.py', '--length', str(length), '--HLA', mhc_formatted, '--mode', 'prediction', '--peptide_file', f'{wd}/peptides.txt'], cwd=cls._exe_dir)
                assert run_result.returncode == 0

                # Anthem creates a new directory for each run, so we need to find the result file
                files = glob.iglob(f'{cls._exe_dir}/*')
                latest_dir = max(files, key=os.path.getctime)
                try:
                    with open(f'{latest_dir}/length_{length}_prediction_result.txt', 'r') as f:
                        for _ in range(5):
                            f.readline()
                        result = []
                        for line in f:
                            if line.startswith('-'):
                                break
                            cells = line.strip().split()
                            result.append(float(cells[-1]))
                except Exception as e:
                    print(mhc_name, ' failed')
                    raise e
                
                pred1 = torch.tensor(result[:len(result)//2], dtype=torch.double)
                pred2 = torch.tensor(result[len(result)//2:], dtype=torch.double)
                log50k1 = torch.tensor(subgroup['log50k1'].tolist(), dtype=torch.double)
                log50k2 = torch.tensor(subgroup['log50k2'].tolist(), dtype=torch.double)
                shutil.rmtree(latest_dir)

                pred_diff[length] = pred1 - pred2
                log50k_diff[length] = log50k1 - log50k2

        preds_diff[mhc_name] = pred_diff
        log50ks_diff[mhc_name] = log50k_diff

        os.remove('peptides.txt')
        return (preds_diff,), log50ks_diff
