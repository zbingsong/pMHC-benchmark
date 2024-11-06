import pandas as pd
import torch

import os
import subprocess
import json
import time
import pathlib

from . import BasePredictor, PredictorConfigs


class TransPHLAPredictor(BasePredictor):
    tasks = None
    _exe_dir = None
    _temp_dir = None
    _unknown_mhc = None
    _unknown_peptide = None
    _wd = None

    @classmethod
    def load(cls, predictor_configs: PredictorConfigs) -> None:
        cls._temp_dir = predictor_configs.temp_dir
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
            df: pd.DataFrame
    ) -> tuple[tuple[dict[str, dict[str, torch.DoubleTensor]], ...], dict[str, dict[str, torch.LongTensor]], dict[str, dict[str, torch.DoubleTensor]], int]:       
        preds = {}
        labels = {}
        log50ks = {}
        times = []

        filtered = df
        filtered = filtered[filtered['mhc_name'].str.startswith(('HLA-A', 'HLA-B', 'HLA-C'))]
        filtered = filtered[~filtered['peptide'].str.contains(r'[BJOUXZ]', regex=True)]
        filtered = filtered[filtered['peptide'].str.len() <= 15]
        if len(df) != len(filtered):
            filtered = filtered.reset_index(drop=True)
            print('Skipped peptides: ', len(df) - len(filtered))
        if len(filtered) == 0:
            print('No valid peptides')
            return ({},), {}, {}, 0
        
        df = filtered
        with open(f'{cls._temp_dir}/peptides_transphla.fasta', 'w') as peptide_f, open(f'{cls._temp_dir}/mhcs_transphla.fasta', 'w') as mhc_f:
            for row in df.itertuples():
                peptide_f.write(f'>{row.peptide}\n{row.peptide}\n')
                mhc_f.write(f'>{row.mhc_name}\n{row.mhc_seq}\n')

        start_time = time.time_ns()
        run_result = subprocess.run(['../env/bin/python', 'pHLAIformer.py', '--peptide_file', f'{cls._wd}/{cls._temp_dir}/peptides_transphla.fasta', '--HLA_file', f'{cls._wd}/{cls._temp_dir}/mhcs_transphla.fasta', '--output_dir', 'results', '--threshold', '0.5'], cwd=cls._exe_dir)
        end_time = time.time_ns()
        assert run_result.returncode == 0
        times.append(end_time - start_time)

        try:
            result_df = pd.read_csv(f'{cls._exe_dir}/results/predict_results.csv')
            assert len(result_df) == len(df), f'Length mismatch: {len(result_df)} != {len(df)}'
        except Exception as e:
            raise e
        
        result_df['label'] = df['label']
        if 'log50k' in df.columns:
            result_df['log50k'] = df['log50k']
        
        for mhc_name, group in result_df.groupby('HLA'):
            pred = {}
            label = {}
            log50k = {}
            grouped_by_len = group.groupby(group['peptide'].str.len())
            for length, subgroup in grouped_by_len:
                pred[length] = torch.tensor(subgroup['y_prob'].tolist(), dtype=torch.double)
                label[length] = torch.tensor(subgroup['label'].tolist(), dtype=torch.long)
                if 'log50k' in subgroup.columns:
                    log50k[length] = torch.tensor(subgroup['log50k'].tolist(), dtype=torch.double)
                
            preds[mhc_name] = pred
            labels[mhc_name] = label
            log50ks[mhc_name] = log50k

        # if os.path.exists('mhcs_transphla.fasta'):
        #     os.remove('mhcs_transphla.fasta')
        #     os.remove('peptides_transphla.fasta')
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
        if_ba = 'log50k1' in df.columns
        
        preds_diff = {}
        labels_diff = {}
        log50ks_diff = {}

        filtered = df
        filtered = filtered[filtered['mhc_name'].str.startswith(('HLA-A', 'HLA-B', 'HLA-C'))]
        filtered = filtered[~filtered['peptide1'].str.contains(r'[BJOUXZ]', regex=True)]
        filtered = filtered[~filtered['peptide2'].str.contains(r'[BJOUXZ]', regex=True)]
        filtered = filtered[filtered['peptide1'].str.len() <= 15]
        filtered = filtered[filtered['peptide2'].str.len() <= 15]
        if len(df) != len(filtered):
            filtered = filtered.reset_index(drop=True)
            print('Skipped peptides: ', len(df) - len(filtered))
        if len(filtered) == 0:
            print('No valid peptides')
            return ({},), 0
        
        df = filtered
        with open(f'{cls._temp_dir}/peptides_transphla.fasta', 'w') as peptide_f, open(f'{cls._temp_dir}/mhcs_transphla.fasta', 'w') as mhc_f:
            for row in df.itertuples():
                peptide_f.write(f'>{row.peptide1}\n{row.peptide1}\n>{row.peptide2}\n{row.peptide2}\n')
                mhc_f.write(f'>{row.mhc_name}\n{row.mhc_seq}\n>{row.mhc_name}\n{row.mhc_seq}\n')

        run_result = subprocess.run(['../env/bin/python', 'pHLAIformer.py', '--peptide_file', f'{cls._wd}/{cls._temp_dir}/peptides_transphla.fasta', '--HLA_file', f'{cls._wd}/{cls._temp_dir}/mhcs_transphla.fasta', '--output_dir', 'results', '--threshold', '0.5'], cwd=cls._exe_dir)
        assert run_result.returncode == 0

        try:
            result_df = pd.read_csv(f'{cls._exe_dir}/results/predict_results.csv')
            assert len(result_df) == 2 * len(df), f'Length mismatch: {len(result_df)} != {2 * len(df)}'
        except Exception as e:
            raise e
        
        result_df1 = result_df.iloc[::2].reset_index(drop=True)
        result_df2 = result_df.iloc[1::2].reset_index(drop=True)
        result_df1.rename(columns={'y_prob': 'y_prob1'}, inplace=True)
        result_df1['y_prob2'] = result_df2['y_prob']
        if if_ba:
            result_df1['log50k1'] = df['log50k1']
            result_df1['log50k2'] = df['log50k2']
        else:
            result_df1['label1'] = df['label1']
            result_df1['label2'] = df['label2']

        for mhc_name, group in result_df1.groupby('HLA'):
            pred_diff = {}
            label_diff = {}
            log50k_diff = {}
            grouped_by_len = group.groupby(group['peptide1'].str.len())
            for length, subgroup in grouped_by_len:
                pred1 = torch.tensor(subgroup['y_prob1'].tolist(), dtype=torch.double)
                pred2 = torch.tensor(subgroup['y_prob2'].tolist(), dtype=torch.double)
                pred_diff[length] = pred1 - pred2
                if if_ba:
                    log50k1 = torch.tensor(subgroup['log50k1'].tolist(), dtype=torch.double)
                    log50k2 = torch.tensor(subgroup['log50k2'].tolist(), dtype=torch.double)
                    log50k_diff[length] = log50k1 - log50k2
                else:
                    label1 = torch.tensor(subgroup['label1'].tolist(), dtype=torch.long)
                    label2 = torch.tensor(subgroup['label2'].tolist(), dtype=torch.long)
                    label_diff[length] = label1 - label2

            preds_diff[mhc_name] = pred_diff
            labels_diff[mhc_name] = label_diff
            log50ks_diff[mhc_name] = log50k_diff

        # if os.path.exists('mhcs_transphla.fasta'):
        #     os.remove('mhcs_transphla.fasta')
        #     os.remove('peptides_transphla.fasta')
        if if_ba:
            return (preds_diff,), log50ks_diff
        else:
            return (preds_diff,), labels_diff
