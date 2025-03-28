import pandas as pd
import torch

import os
import subprocess
import json
import time
import pathlib
import typing

from . import BasePredictor, PredictorConfigs


class TransPHLAPredictor(BasePredictor):
    tasks = None
    _exe_dir = None
    _temp_dir = None
    _wd = None
    _max_batch_size = 50000 # tested on my computer only

    # @typing.override
    @classmethod
    def load(cls, predictor_configs: PredictorConfigs) -> None:
        cls._temp_dir = predictor_configs.temp_dir
        cls.tasks = ['Mix']
        cls._wd = os.getcwd()
        cls._exe_dir = '~/repo/TransPHLA-AOMP/TransPHLA-AOMP'

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
        
        preds = {}
        labels = {}
        log50ks = {}
        times = []

        result_dfs = []
        # divide dataframe into batches, test each batch and finally concatenate the results
        for i in range(0, len(df), cls._max_batch_size):
            batch_df = df.iloc[i:i+cls._max_batch_size]
            with open(f'{cls._temp_dir}/peptides_transphla.fasta', 'w') as peptide_f, open(f'{cls._temp_dir}/mhcs_transphla.fasta', 'w') as mhc_f:
                for row in batch_df.itertuples():
                    peptide_f.write(f'>{row.peptide}\n{row.peptide}\n')
                    mhc_f.write(f'>{row.mhc_name}\n{row.mhc_seq}\n')

            start_time = time.time_ns()
            run_result = subprocess.run(['../env/bin/python', 'pHLAIformer.py', '--peptide_file', f'{cls._wd}/{cls._temp_dir}/peptides_transphla.fasta', '--HLA_file', f'{cls._wd}/{cls._temp_dir}/mhcs_transphla.fasta', '--output_dir', 'results', '--threshold', '0.5'], cwd=cls._exe_dir)
            end_time = time.time_ns()
            assert run_result.returncode == 0
            times.append(end_time - start_time)

            try:
                result_df = pd.read_csv(f'{cls._exe_dir}/results/predict_results.csv')
                assert len(result_df) == len(batch_df), f'Length mismatch: {len(result_df)} != {len(batch_df)}'
            except Exception as e:
                raise e
            result_dfs.append(result_df)
        
        result_df = pd.concat(result_dfs, ignore_index=True)
        result_df['label'] = df['label']
        if 'log50k' in df.columns:
            result_df['log50k'] = df['log50k']
        
        for mhc_name, group in result_df.groupby('HLA'):
            pred = {}
            label = {}
            log50k = {}
            group.reset_index(drop=True, inplace=True)
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
        df, _ = cls._filter_sensitivity(df)
        if len(df) == 0:
            print('No valid peptides')
            return ({},), {}
        
        preds_diff = {}
        labels_diff = {}
        log50ks_diff = {}

        result_dfs1 = []
        result_dfs2 = []
        # divide dataframe into batches, test each batch and finally concatenate the results
        for i in range(0, len(df), cls._max_batch_size):
            batch_df = df.iloc[i:i+cls._max_batch_size]
            with open(f'{cls._temp_dir}/peptides_transphla.fasta', 'w') as peptide_f, open(f'{cls._temp_dir}/mhcs_transphla.fasta', 'w') as mhc_f:
                for row in batch_df.itertuples():
                    peptide_f.write(f'>{row.peptide1}\n{row.peptide1}\n>{row.peptide2}\n{row.peptide2}\n')
                    mhc_f.write(f'>{row.mhc_name}\n{row.mhc_seq}\n>{row.mhc_name}\n{row.mhc_seq}\n')

            run_result = subprocess.run(['../env/bin/python', 'pHLAIformer.py', '--peptide_file', f'{cls._wd}/{cls._temp_dir}/peptides_transphla.fasta', '--HLA_file', f'{cls._wd}/{cls._temp_dir}/mhcs_transphla.fasta', '--output_dir', 'results', '--threshold', '0.5'], cwd=cls._exe_dir)
            assert run_result.returncode == 0

            try:
                result_df = pd.read_csv(f'{cls._exe_dir}/results/predict_results.csv')
                assert len(result_df) == 2 * len(batch_df), f'Length mismatch: {len(result_df)} != {2 * len(batch_df)}'
            except Exception as e:
                raise e
            result_df1 = result_df.iloc[::2].reset_index(drop=True)
            result_df2 = result_df.iloc[1::2].reset_index(drop=True)
            result_dfs1.append(result_df1)
            result_dfs2.append(result_df2)
        
        result_df1 = pd.concat(result_dfs1, ignore_index=True)
        result_df2 = pd.concat(result_dfs2, ignore_index=True)
        result_df1.rename(columns={'y_prob': 'y_prob1'}, inplace=True)
        result_df1['y_prob2'] = result_df2['y_prob']
        if if_ba:
            result_df1['log50k1'] = df['log50k1']
            result_df1['log50k2'] = df['log50k2']
        else:
            result_df1['label1'] = df['label1']
            result_df1['label2'] = df['label2']

        for mhc_name, group in result_df1.groupby('HLA'):
            pred_diff = None
            label_diff = None
            log50k_diff = None

            pred1 = torch.tensor(group['y_prob1'].tolist(), dtype=torch.double)
            pred2 = torch.tensor(group['y_prob2'].tolist(), dtype=torch.double)
            pred_diff = pred1 - pred2
            if if_ba:
                log50k1 = torch.tensor(group['log50k1'].tolist(), dtype=torch.double)
                log50k2 = torch.tensor(group['log50k2'].tolist(), dtype=torch.double)
                log50k_diff = log50k1 - log50k2
            else:
                label1 = torch.tensor(group['label1'].tolist(), dtype=torch.long)
                label2 = torch.tensor(group['label2'].tolist(), dtype=torch.long)
                label_diff = label1 - label2

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

    # @typing.override
    @classmethod
    def _filter(cls, df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
        filtered = df
        filtered = filtered[filtered['mhc_name'].str.startswith(('HLA-A', 'HLA-B', 'HLA-C'))]
        filtered = filtered[~filtered['peptide'].str.contains(r'[BJOUXZ]', regex=True)]
        filtered = filtered[filtered['peptide'].str.len() <= 15]
        filtered = filtered[filtered['peptide'].str.len() >= 8]
        if len(df) != len(filtered):
            filtered = filtered.reset_index(drop=True)
            print('Skipped peptides: ', len(df) - len(filtered))
        return filtered, len(df) - len(filtered)
    
    # @typing.override
    @classmethod
    def _filter_sensitivity(cls, df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
        filtered = df
        filtered = filtered[filtered['mhc_name'].str.startswith(('HLA-A', 'HLA-B', 'HLA-C'))]
        filtered = filtered[~filtered['peptide1'].str.contains(r'[BJOUXZ]', regex=True)]
        filtered = filtered[~filtered['peptide2'].str.contains(r'[BJOUXZ]', regex=True)]
        filtered = filtered[filtered['peptide1'].str.len() <= 15]
        filtered = filtered[filtered['peptide1'].str.len() >= 8]
        filtered = filtered[filtered['peptide2'].str.len() <= 15]
        filtered = filtered[filtered['peptide2'].str.len() >= 8]
        if len(df) != len(filtered):
            filtered = filtered.reset_index(drop=True)
            print('Skipped peptides: ', len(df) - len(filtered))
        return filtered, len(df) - len(filtered)
    