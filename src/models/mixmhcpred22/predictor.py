import pandas as pd
import torch

import os
import subprocess
import json
import time
import pathlib
import typing

from . import BasePredictor, PredictorConfigs


class MixMHCpred22Predictor(BasePredictor):
    tasks = None
    _temp_dir = None
    _executable = None
    _unknown_mhc = None

    @typing.override
    @classmethod
    def load(cls, predictor_configs: PredictorConfigs) -> None:
        cls._temp_dir = predictor_configs.temp_dir
        cls.tasks = ['EL']
        curr_dir = pathlib.Path(__file__).parent
        with open(f'{curr_dir}/configs.json', 'r') as f:
            configs = json.load(f)
            cls._executable = os.path.expanduser(configs['exe_path'])
            cls._unknown_mhc = os.path.expanduser(configs['unknown_mhc'])

    @typing.override
    @classmethod
    def run_retrieval(
            cls,
            df: pd.DataFrame
    ) -> tuple[tuple[dict[str, dict[str, torch.DoubleTensor]], ...], dict[str, dict[str, torch.LongTensor]], dict[str, dict[str, torch.DoubleTensor]], int]:
        df = cls._filter(df)
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

            mhc_formatted = mhc_name[4:].replace(':', '')

            peptides = group['peptide'].tolist()
            with open(f'{cls._temp_dir}/peptides_mixmhcpred22.fasta', 'w') as f:
                for peptide in peptides:
                    f.write(f'>{peptide}\n{peptide}\n')
                    
            start_time = time.time_ns()
            run_result = subprocess.run([cls._executable, '-i', f'{cls._temp_dir}/peptides_mixmhcpred22.fasta', '-o', f'{cls._temp_dir}/result_mixmhcpred22.tsv', '-a', mhc_formatted], stdout=subprocess.DEVNULL)
            end_time = time.time_ns()
            assert run_result.returncode == 0
            times.append(end_time - start_time)
            try:
                result_df = pd.read_csv(f'{cls._temp_dir}/result_mixmhcpred22.tsv', sep='\t', skiprows=list(range(11)))
                assert len(result_df) == len(peptides), f'Peptide numbers mismatch: {len(result_df)} vs {len(peptides)}'
            except Exception as e:
                print(mhc_formatted, ' failed')
                raise e
            result_df['label'] = group['label']
            if 'log50k' in group.columns:
                result_df['log50k'] = group['log50k']
            grouped_by_len = result_df.groupby(result_df['Peptide'].str.len())
            for length, subgroup in grouped_by_len:
                pred[length] = torch.tensor((100.0 - subgroup['%Rank_bestAllele']).tolist(), dtype=torch.double)
                label[length] = torch.tensor(subgroup['label'].tolist(), dtype=torch.long)
                if 'log50k' in subgroup.columns:
                    log50k[length] = torch.tensor(subgroup['log50k'].tolist(), dtype=torch.double)
            
            preds[mhc_name] = pred
            labels[mhc_name] = label
            log50ks[mhc_name] = log50k

        # if os.path.exists('peptides_mixmhcpred22.fasta'):
        #     os.remove('peptides_mixmhcpred22.fasta')
        #     os.remove('result_mixmhcpred22.tsv')
        return (preds,), labels, log50ks, sum(times)
    
    @typing.override
    @classmethod
    def run_sq(
            cls,
            df: pd.DataFrame
    ) -> tuple[tuple[dict[str, dict[str, torch.DoubleTensor]], ...], dict[str, dict[str, torch.LongTensor]], dict[str, dict[str, torch.DoubleTensor]], int]:
        df = cls._filter(df)
        if len(df) == 0:
            print('No valid peptides')
            return ({},), {}, {}, 0
        df = df.groupby('mhc_name')
        
        preds = {}
        labels = {}
        log50ks = {}
        times = []
        
        mhcs = [mhc_name for mhc_name in df.groups.keys()]
        mhcs_formatted = [mhc_name[4:].replace(':', '') for mhc_name in mhcs]

        group_len = df.get_group(mhcs[0]).shape[0]
        for _, group in df:
            assert group.shape[0] == group_len, f'Peptide numbers mismatch: {group.shape[0]} vs {group_len}'

        peptides = df.get_group(mhcs[0])['peptide'].tolist()

        with open(f'{cls._temp_dir}/peptides_mixmhcpred22.fasta', 'w') as f:
            for peptide in peptides:
                f.write(f'>{peptide}\n{peptide}\n')
                
        start_time = time.time_ns()
        run_result = subprocess.run([cls._executable, '-i', f'{cls._temp_dir}/peptides_mixmhcpred22.fasta', '-o', f'{cls._temp_dir}/result_mixmhcpred22.tsv', '-a', ','.join(mhcs_formatted)], stdout=subprocess.DEVNULL)
        end_time = time.time_ns()
        assert run_result.returncode == 0
        times.append(end_time - start_time)
        try:
            result_df = pd.read_csv(f'{cls._temp_dir}/result_mixmhcpred22.tsv', sep='\t', skiprows=list(range(11)))
        except Exception as e:
            print('sqaure dataframe failed')
            raise e
        
        for mhc_name, mhc_name_formatted in zip(mhcs, mhcs_formatted):
            group = df.get_group(mhc_name)
            group.reset_index(drop=True, inplace=True)
            result_key = f'%Rank_{mhc_name_formatted}'
            result_df['label'] = group['label']
            if 'log50k' in group.columns:
                result_df['log50k'] = group['log50k']
            grouped_by_len = result_df.groupby(result_df['Peptide'].str.len())
            pred = {}
            label = {}
            log50k = {}
            for length, subgroup in grouped_by_len:
                pred[length] = torch.tensor((100.0 - subgroup[result_key]).tolist(), dtype=torch.double)
                label[length] = torch.tensor(subgroup['label'].tolist(), dtype=torch.long)
                if 'log50k' in subgroup.columns:
                    log50k[length] = torch.tensor(subgroup['log50k'].tolist(), dtype=torch.double)
            preds[mhc_name] = pred
            labels[mhc_name] = label
            log50ks[mhc_name] = log50k

        # if os.path.exists('peptides_mixmhcpred22.fasta'):
        #     os.remove('peptides_mixmhcpred22.fasta')
        #     os.remove('result_mixmhcpred22.tsv')
        return (preds,), labels, log50ks, sum(times)
    
    @typing.override
    @classmethod
    def run_sensitivity(
            cls,
            df: pd.DataFrame
    ) -> tuple[tuple[dict[str, torch.DoubleTensor], ...], dict[str, torch.DoubleTensor]]:
        if_ba = 'log50k1' in df.columns
        df = cls._filter_sensitivity(df)
        if len(df) == 0:
            print('No valid peptides')
            return ({},), {}
        df = df.groupby('mhc_name')
        
        preds_diff = {}
        labels_diff = {}
        log50ks_diff = {}

        for mhc_name, group in df:
            group.reset_index(drop=True, inplace=True)

            pred_diff = None
            label_diff = None
            log50k_diff = None

            mhc_formatted = mhc_name[4:].replace(':', '')

            peptides1 = group['peptide1'].tolist()
            peptides2 = group['peptide2'].tolist()
            with open(f'{cls._temp_dir}/peptides_mixmhcpred22.fasta', 'w') as f:
                for peptide in peptides1:
                    f.write(f'>{peptide}\n{peptide}\n')
                for peptide in peptides2:
                    f.write(f'>{peptide}\n{peptide}\n')

            run_result = subprocess.run([cls._executable, '-i', f'{cls._temp_dir}/peptides_mixmhcpred22.fasta', '-o', f'{cls._temp_dir}/result_mixmhcpred22.tsv', '-a', mhc_formatted], stdout=subprocess.DEVNULL)
            assert run_result.returncode == 0
            result_df = pd.read_csv(f'{cls._temp_dir}/result_mixmhcpred22.tsv', sep='\t', skiprows=list(range(11)))
            result_df1 = result_df.iloc[:len(peptides1)].reset_index(drop=True)
            result_df2 = result_df.iloc[len(peptides1):].reset_index(drop=True)
            if if_ba:
                result_df1['log50k'] = group['log50k1']
                result_df2['log50k'] = group['log50k2']
            else:
                result_df1['label'] = group['label1']
                result_df2['label'] = group['label2']

            pred1 = torch.tensor((100.0 - result_df1['%Rank_bestAllele']).tolist(), dtype=torch.double)
            pred2 = torch.tensor((100.0 - result_df2['%Rank_bestAllele']).tolist(), dtype=torch.double)
            pred_diff = pred1 - pred2
            if if_ba:
                log50k1 = torch.tensor(result_df1['log50k'].tolist(), dtype=torch.double)
                log50k2 = torch.tensor(result_df2['log50k'].tolist(), dtype=torch.double)
                log50k_diff = log50k1 - log50k2
            else:
                label1 = torch.tensor(result_df1['label'].tolist(), dtype=torch.long)
                label2 = torch.tensor(result_df2['label'].tolist(), dtype=torch.long)
                label_diff = label1 - label2

            preds_diff[mhc_name] = pred_diff
            labels_diff[mhc_name] = label_diff
            log50ks_diff[mhc_name] = log50k_diff

        # os.remove('peptides_mixmhcpred22.fasta')
        # os.remove('result_mixmhcpred22.tsv')
        if if_ba:
            return (preds_diff,), log50ks_diff
        else:
            return (preds_diff,), labels_diff

    @classmethod
    def _filter(cls, df: pd.DataFrame) -> pd.DataFrame:
        filtered = df
        filtered = filtered[filtered['mhc_name'].str.startswith(('HLA-A', 'HLA-B', 'HLA-C'))]
        filtered = filtered[~filtered['peptide'].str.contains(r'[BJOUXZ]', regex=True)]
        filtered = filtered[filtered['peptide'].str.len() <= 14]
        filtered = filtered[filtered['peptide'].str.len() >= 8]
        if len(df) != len(filtered):
            filtered = filtered.reset_index(drop=True)
            print('Skipped peptides: ', len(df) - len(filtered))
        return filtered
    
    @classmethod
    def _filter_sensitivity(cls, df: pd.DataFrame) -> pd.DataFrame:
        filtered = df
        filtered = filtered[filtered['mhc_name'].str.startswith(('HLA-A', 'HLA-B', 'HLA-C'))]
        filtered = filtered[~filtered['peptide1'].str.contains(r'[BJOUXZ]', regex=True)]
        filtered = filtered[~filtered['peptide2'].str.contains(r'[BJOUXZ]', regex=True)]
        filtered = filtered[filtered['peptide1'].str.len() <= 14]
        filtered = filtered[filtered['peptide1'].str.len() >= 8]
        filtered = filtered[filtered['peptide2'].str.len() <= 14]
        filtered = filtered[filtered['peptide2'].str.len() >= 8]
        if len(df) != len(filtered):
            filtered = filtered.reset_index(drop=True)
            print('Skipped peptides: ', len(df) - len(filtered))
        return filtered
    