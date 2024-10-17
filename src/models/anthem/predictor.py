import pandas.core.groupby.generic as pd_typing
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
            # peptide should contain none of B, J, O, U, X, Z
            group = group[~group['peptide'].str.contains(r'[BJOUXZ]', regex=True)].reset_index(drop=True)
            grouped_by_len = group.groupby(group['peptide'].str.len())

            mhc_formatted = mhc_name
            if ':' not in mhc_name:
                mhc_formatted = mhc_formatted[:-2] + ':' + mhc_formatted[-2:]
            if '*' not in mhc_name:
                mhc_formatted = mhc_formatted[:5] + '*' + mhc_formatted[5:]

            for length, subgroup in grouped_by_len:
                peptides = subgroup['peptide'].tolist()
                with open('peptides.txt', 'w') as f:
                    for peptide in peptides:
                        f.write(f'{peptide}\n')
                wd = os.getcwd()

                start_time = time.time_ns()
                run_result = subprocess.run(['env/bin/python', 'sware_b_main.py', '--HLA', mhc_formatted, '--mode', 'prediction', '--peptide_file', f'{wd}/peptides.txt'], cwd=cls._exe_dir)
                end_time = time.time_ns()

                try:
                    assert run_result.returncode == 0
                except Exception as e:
                    print(f'Error running Anthem for {mhc_formatted} with length {length}')
                    if cls._unknown_mhc == 'ignore':
                        continue
                    elif cls._unknown_mhc == 'error':
                        raise e
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
                            line = line.strip()
                            if line.startswith('-'):
                                break
                            cells = line.split()
                            result.append(float(cells[-1]))
                        assert len(result) == len(subgroup), f'Length mismatch: {len(result)} vs {len(subgroup)} for {mhc_name} with length {length}'
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
            group = group[~group['peptide'].str.contains(r'[BJOUXZ]', regex=True)].reset_index(drop=True)
            grouped_by_len = group.groupby(group['peptide1'].str.len())

            mhc_formatted = mhc_name
            if ':' not in mhc_name:
                mhc_formatted = mhc_formatted[:-2] + ':' + mhc_formatted[-2:]
            if '*' not in mhc_name:
                mhc_formatted = mhc_formatted[:5] + '*' + mhc_formatted[5:]

            for length, subgroup in grouped_by_len:
                with open(f'peptides.txt', 'w') as f:
                    for row in subgroup.itertuples():
                        f.write(f'{row.peptide1}\n{row.peptide2}\n')
                wd = os.getcwd()

                run_result = subprocess.run(['env/bin/python', 'sware_b_main.py', '--HLA', mhc_formatted, '--mode', 'prediction', '--peptide_file', f'{wd}/peptides.txt'], cwd=cls._exe_dir)
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
                            line = line.strip()
                            if line.startswith('-'):
                                break
                            cells = line.split()
                            result.append(float(cells[-1]))
                        assert len(result) == len(subgroup), f'Length mismatch: {len(result)} vs {len(subgroup)} for {mhc_name} with length {length}'
                except Exception as e:
                    print(mhc_name, ' failed')
                    raise e
                
                pred1 = torch.tensor(result[::2], dtype=torch.double)
                pred2 = torch.tensor(result[1::2], dtype=torch.double)
                log50k1 = torch.tensor(subgroup['log50k1'].tolist(), dtype=torch.double)
                log50k2 = torch.tensor(subgroup['log50k2'].tolist(), dtype=torch.double)
                shutil.rmtree(latest_dir)

                pred_diff[length] = pred1 - pred2
                log50k_diff[length] = log50k1 - log50k2

        preds_diff[mhc_name] = pred_diff
        log50ks_diff[mhc_name] = log50k_diff

        os.remove('peptides.txt')
        return (preds_diff,), log50ks_diff
