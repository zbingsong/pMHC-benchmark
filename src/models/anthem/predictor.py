import pandas as pd
import torch
import os
import subprocess
import time
import glob
import shutil
import typing

from . import BasePredictor, PredictorConfigs


class AnthemPredictor(BasePredictor):
    tasks = None
    _exe_dir = None
    _temp_dir = None
    _wd = None

    # @typing.override
    @classmethod
    def load(cls, predictor_configs: PredictorConfigs) -> None:
        cls._temp_dir = predictor_configs.temp_dir
        cls.tasks = ['Mix']
        cls._wd = os.getcwd()
        cls._exe_dir = '~/repo/Anthem'

    # @typing.override
    @classmethod
    def run_retrieval(
            cls,
            df: pd.DataFrame
    ) -> tuple[tuple[dict[str, dict[str, torch.DoubleTensor]], ...], dict[str, dict[str, torch.LongTensor]], dict[str, dict[str, torch.DoubleTensor]], int, int]:
        df, num_skipped = cls._filter(df)
        if len(df) == 0:
            print('No valid peptides')
            return ({},), {}, {}, 0, num_skipped
        df = df.groupby('mhc_name')
        
        preds = {}
        labels = {}
        log50ks = {}
        times = []

        for mhc_name, group in df:
            group.reset_index(drop=True, inplace=True)

            pred = {}
            label = {}
            log50k = {}

            grouped_by_len = group.groupby(group['peptide'].str.len())

            mhc_formatted = cls.__format_mhc(mhc_name)

            for length, subgroup in grouped_by_len:
                peptides = subgroup['peptide'].tolist()
                with open(f'{cls._temp_dir}/peptides_anthem.txt', 'w') as f:
                    for peptide in peptides:
                        f.write(f'{peptide}\n')

                start_time = time.time_ns()
                run_result = subprocess.run(['env/bin/python', 'sware_b_main.py', '--HLA', mhc_formatted, '--mode', 'prediction', '--peptide_file', f'{cls._wd}/{cls._temp_dir}/peptides_anthem.txt'], cwd=cls._exe_dir)
                end_time = time.time_ns()

                try:
                    assert run_result.returncode == 0
                except Exception as e:
                    print(f'Error running Anthem for {mhc_formatted} with length {length}')
                    num_skipped += len(subgroup)
                    continue
                times.append(end_time - start_time)

                # Anthem may drop some peptides, so need to keep track of the returned peptides
                returned_peptides = set()
                # Anthem creates a new directory for each run, so we need to find the result file
                files = glob.iglob(f'{cls._exe_dir}/*')
                latest_dir = max(files, key=os.path.getctime)
                with open(f'{latest_dir}/length_{length}_prediction_result.txt', 'r') as f:
                    for _ in range(5):
                        f.readline()
                    result = []
                    for line in f:
                        line = line.strip()
                        if line.startswith('-'):
                            break
                        cells = line.split()
                        returned_peptides.add(cells[0])
                        result.append(float(cells[-1]))
                    # assert len(result) == len(subgroup), f'Length mismatch: {len(result)} vs {len(subgroup)} for {mhc_name} with length {length}'
                
                num_skipped += len(subgroup) - len(returned_peptides)
                filtered_indices = subgroup['peptide'].isin(returned_peptides)
                pred[length] = torch.tensor(result, dtype=torch.double)
                label[length] = torch.tensor(subgroup[filtered_indices]['label'].tolist(), dtype=torch.long)
                if 'log50k' in subgroup.columns:
                    log50k[length] = torch.tensor(subgroup[filtered_indices]['log50k'].tolist(), dtype=torch.double)
                
                shutil.rmtree(latest_dir)
            
            preds[mhc_name] = pred
            labels[mhc_name] = label
            log50ks[mhc_name] = log50k

        # os.remove('peptides_anthem.txt')
        # print(f'Skipped {num_skipped} peptides')
        return (preds,), labels, log50ks, sum(times), num_skipped
    
    # @typing.override
    @classmethod
    def run_sq(
            cls, 
            df: pd.DataFrame
    ) -> tuple[tuple[dict[str, dict[str, torch.DoubleTensor]], ...], dict[str, dict[str, torch.LongTensor]], dict[str, dict[str, torch.DoubleTensor]], int, int]:
        return cls.run_retrieval(df)
    
    # @typing.override
    @classmethod
    def run_sensitivity(
            cls,
            df: pd.DataFrame
    ) -> tuple[tuple[dict[str, torch.DoubleTensor], ...], dict[str, torch.DoubleTensor]]:
        if_ba = 'log50k1' in df.columns
        df, num_skipped = cls._filter_sensitivity(df)
        if len(df) == 0:
            print('No valid peptides')
            return ({},), {}
        df = df.groupby('mhc_name')
        
        preds_diff = {}
        labels_diff = {}
        log50ks_diff = {}

        for mhc_name, group in df:
            group.reset_index(drop=True, inplace=True)
            
            pred_diff = []
            label_diff = []
            log50k_diff = []

            grouped_by_len = group.groupby(group['peptide1'].str.len())

            mhc_formatted = cls.__format_mhc(mhc_name)

            for length, subgroup in grouped_by_len:
                with open(f'{cls._temp_dir}/peptides_anthem.txt', 'w') as f:
                    for row in subgroup.itertuples():
                        f.write(f'{row.peptide1}\n{row.peptide2}\n')

                run_result = subprocess.run(['env/bin/python', 'sware_b_main.py', '--HLA', mhc_formatted, '--mode', 'prediction', '--peptide_file', f'{cls._wd}/{cls._temp_dir}/peptides_anthem.txt'], cwd=cls._exe_dir)
                try:
                    assert run_result.returncode == 0
                except Exception as e:
                    print(f'Error running Anthem for {mhc_formatted} with length {length}')
                    num_skipped += len(subgroup)
                    continue

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
                        assert len(result) == 2 * len(subgroup), f'Length mismatch: {len(result)} vs {len(subgroup)} for {mhc_name} with length {length}'
                except Exception as e:
                    print(mhc_name, ' failed')
                    raise e
                
                pred1 = torch.tensor(result[::2], dtype=torch.double)
                pred2 = torch.tensor(result[1::2], dtype=torch.double)
                pred_diff.append(pred1 - pred2)
                if if_ba:
                    log50k1 = torch.tensor(subgroup['log50k1'].tolist(), dtype=torch.double)
                    log50k2 = torch.tensor(subgroup['log50k2'].tolist(), dtype=torch.double)
                    log50k_diff.append(log50k1 - log50k2)
                else:
                    label1 = torch.tensor(subgroup['label1'].tolist(), dtype=torch.long)
                    label2 = torch.tensor(subgroup['label2'].tolist(), dtype=torch.long)
                    label_diff.append(label1 - label2)
                
                shutil.rmtree(latest_dir)

            if len(pred_diff) == 0:
                continue
            preds_diff[mhc_name] = torch.cat(pred_diff)
            if if_ba:
                log50ks_diff[mhc_name] = torch.cat(log50k_diff)
            else:
                labels_diff[mhc_name] = torch.cat(label_diff)

        print(f'Skipped {num_skipped} peptides')

        # os.remove('peptides_anthem.txt')
        if if_ba:
            return (preds_diff,), log50ks_diff
        else:
            return (preds_diff,), labels_diff

    # @typing.override
    @classmethod
    def _filter(cls, df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
        filtered = df
        filtered = filtered[~filtered['peptide'].str.contains(r'[BJOUXZ]', regex=True)]
        filtered = filtered[filtered['mhc_name'].str.startswith('HLA-')]
        filtered = filtered[filtered['peptide'].str.len() <= 14]
        filtered = filtered[filtered['peptide'].str.len() >= 8]
        if len(df) != len(filtered):
            filtered = filtered.reset_index(drop=True)
            print('Skipped peptides: ', len(df) - len(filtered))
        return filtered, len(df) - len(filtered)
    
    # @typing.override
    @classmethod
    def _filter_sensitivity(cls, df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
        filtered = df
        filtered = filtered[~filtered['peptide1'].str.contains(r'[BJOUXZ]', regex=True)]
        filtered = filtered[~filtered['peptide2'].str.contains(r'[BJOUXZ]', regex=True)]
        filtered = filtered[filtered['mhc_name'].str.startswith('HLA-')]
        filtered = filtered[filtered['peptide1'].str.len() <= 14]
        filtered = filtered[filtered['peptide1'].str.len() >= 8]
        filtered = filtered[filtered['peptide2'].str.len() <= 14]
        filtered = filtered[filtered['peptide2'].str.len() >= 8]
        if len(df) != len(filtered):
            filtered = filtered.reset_index(drop=True)
            print('Skipped peptides: ', len(df) - len(filtered))
        return filtered, len(df) - len(filtered)
    
    @classmethod
    def __format_mhc(cls, mhc_name: str) -> str:
        mhc_formatted = mhc_name
        if ':' not in mhc_name:
            mhc_formatted = mhc_formatted[:-2] + ':' + mhc_formatted[-2:]
        if '*' not in mhc_name:
            mhc_formatted = mhc_formatted[:5] + '*' + mhc_formatted[5:]
        return mhc_formatted
    