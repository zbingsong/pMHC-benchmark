import pandas as pd
import torch

import os
import subprocess
import json
import time
import pathlib

from . import BasePredictor, PredictorConfigs


class MixMHCpred30Predictor(BasePredictor):
    tasks = None
    _temp_dir = None
    _executable = None
    _unknown_mhc = None

    @classmethod
    def load(cls, predictor_configs: PredictorConfigs) -> None:
        cls._temp_dir = predictor_configs.temp_dir
        cls.tasks = ['EL']
        curr_dir = pathlib.Path(__file__).parent
        with open(f'{curr_dir}/configs.json', 'r') as f:
            configs = json.load(f)
            cls._executable = os.path.expanduser(configs['exe_path'])
            cls._unknown_mhc = os.path.expanduser(configs['unknown_mhc'])

    @classmethod
    def run_retrieval(
            cls,
            df: pd.DataFrame
    ) -> tuple[tuple[dict[str, dict[str, torch.DoubleTensor]], ...], dict[str, dict[str, torch.LongTensor]], dict[str, dict[str, torch.DoubleTensor]], int]:
        df = df.groupby('mhc_name')

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
            with open(f'{cls._temp_dir}/peptides_mixmhcpred30.fasta', 'w') as f:
                for peptide in peptides:
                    f.write(f'>{peptide}\n{peptide}\n')

            start_time = time.time_ns()
            run_result = subprocess.run([cls._executable, '-i', f'{cls._temp_dir}/peptides_mixmhcpred30.fasta', '-o', f'{cls._temp_dir}/result_mixmhcpred30.tsv', '-a', mhc_formatted], stdout=subprocess.DEVNULL)
            end_time = time.time_ns()
            assert run_result.returncode == 0
            times.append(end_time - start_time)
            try:
                result_df = pd.read_csv(f'{cls._temp_dir}/result_mixmhcpred30.tsv', sep='\t', skiprows=list(range(11)))
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

        # if os.path.exists('peptides_mixmhcpred30.txt'):
        #     os.remove('peptides_mixmhcpred30.txt')
        #     os.remove('result_mixmhcpred30.tsv')
        return (preds,), labels, log50ks, sum(times)
    
    @classmethod
    def run_sq(
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
        filtered = filtered[filtered['peptide'].str.len() <= 14]
        if len(df) != len(filtered):
            filtered = filtered.reset_index(drop=True)
            print('Skipped peptides: ', len(df) - len(filtered))
        if len(filtered) == 0:
            print('No valid peptides')
            return ({},), {}, {}, 0
        
        df = filtered.groupby('mhc_name')
        mhcs = [mhc_name for mhc_name in df.groups.keys()]
        mhcs_formatted = [mhc_name[4:].replace(':', '') for mhc_name in mhcs]

        group_len = df.get_group(mhcs[0]).shape[0]
        for _, group in df:
            assert group.shape[0] == group_len, f'Peptide numbers mismatch: {group.shape[0]} vs {group_len}'

        peptides = df.get_group(mhcs[0])['peptide'].tolist()

        with open(f'{cls._temp_dir}/peptides_mixmhcpred30.fasta', 'w') as f:
            for peptide in peptides:
                f.write(f'>{peptide}\n{peptide}\n')
                
        start_time = time.time_ns()
        run_result = subprocess.run([cls._executable, '-i', f'{cls._temp_dir}peptides_mixmhcpred30.fasta', '-o', f'{cls._temp_dir}/result_mixmhcpred30.tsv', '-a', ','.join(mhcs_formatted)], stdout=subprocess.DEVNULL)
        end_time = time.time_ns()
        assert run_result.returncode == 0
        times.append(end_time - start_time)
        try:
            result_df = pd.read_csv(f'{cls._temp_dir}/result_mixmhcpred30.tsv', sep='\t', skiprows=list(range(11)))
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
                pred[length] = torch.tensor((1 - subgroup[result_key]).tolist(), dtype=torch.double)
                label[length] = torch.tensor(subgroup['label'].tolist(), dtype=torch.long)
                if 'log50k' in subgroup.columns:
                    log50k[length] = torch.tensor(subgroup['log50k'].tolist(), dtype=torch.double)
            preds[mhc_name] = pred
            labels[mhc_name] = label
            log50ks[mhc_name] = log50k

        # if os.path.exists('peptides_mixmhcpred30.fasta'):
        #     os.remove('peptides_mixmhcpred30.fasta')
        #     os.remove('result_mixmhcpred30.tsv')
        return (preds,), labels, log50ks, sum(times)
            
    @classmethod
    def run_sensitivity(
            cls,
            df: pd.DataFrame
    ) -> tuple[tuple[dict[str, dict[str, torch.DoubleTensor]], ...], dict[str, dict[str, torch.DoubleTensor]]]:
        if_ba = 'log50k1' in df.columns
        
        df = df.groupby('mhc')
        
        preds_diff = {}
        labels_diff = {}
        log50ks_diff = {}

        for mhc_name, group in df:
            if not mhc_name.startswith('HLA-'):
                print(f'Unknown MHC name: {mhc_name}')
                if cls._unknown_mhc == 'ignore':
                    continue
                elif cls._unknown_mhc == 'error':
                    raise ValueError(f'Unknown MHC name: {mhc_name}')
                
            pred_diff = {}
            label_diff = {}
            log50k_diff = {}

            filtered = group
            filtered = filtered[~filtered['peptide1'].str.contains(r'[BJOUXZ]', regex=True)]
            filtered = filtered[~filtered['peptide2'].str.contains(r'[BJOUXZ]', regex=True)]
            filtered = filtered[filtered['peptide1'].str.len() <= 14]
            filtered = filtered[filtered['peptide2'].str.len() <= 14]
            filtered = filtered.reset_index(drop=True)
            if len(group) != len(filtered):
                print('Skipped peptides:', len(group) - len(filtered))
            if len(group) == 0:
                print(f'No valid peptides for {mhc_name}')
                continue
            mhc_formatted = mhc_name[4:].replace(':', '')

            peptides1 = group['peptide1'].tolist()
            peptides2 = group['peptide2'].tolist()
            with open(f'{cls._temp_dir}/peptides_mixmhcpred30.fasta', 'w') as f:
                for peptide in peptides1:
                    f.write(f'>{peptide}\n{peptide}\n')
                for peptide in peptides2:
                    f.write(f'>{peptide}\n{peptide}\n')

            run_result = subprocess.run([cls._executable, '-i', f'{cls._temp_dir}/peptides_mixmhcpred30.fasta', '-o', f'{cls._temp_dir}/result_mixmhcpred30.tsv', '-a', mhc_formatted])
            assert run_result.returncode == 0
            result_df = pd.read_csv(f'{cls._temp_dir}/result_mixmhcpred30.tsv', sep='\t', skiprows=list(range(11)))
            result_df1 = result_df.iloc[:len(peptides1)].reset_index(drop=True)
            result_df2 = result_df.iloc[len(peptides1):].reset_index(drop=True)
            if if_ba:
                result_df1['log50k'] = group['log50k1']
                result_df2['log50k'] = group['log50k2']
            else:
                result_df1['label'] = group['label1']
                result_df2['label'] = group['label2']

            grouped_by_len1 = result_df1.groupby(result_df1['Peptide'].str.len())
            grouped_by_len2 = result_df2.groupby(result_df2['Peptide'].str.len())

            for length in grouped_by_len1.groups.keys():
                subgroup1 = grouped_by_len1.get_group(length)
                subgroup2 = grouped_by_len2.get_group(length)
                pred1 = torch.tensor((1 - subgroup1['%Rank_bestAllele']).tolist(), dtype=torch.double)
                pred2 = torch.tensor((1 - subgroup2['%Rank_bestAllele']).tolist(), dtype=torch.double)
                pred_diff[length] = pred1 - pred2
                if if_ba:
                    log50k1 = torch.tensor(subgroup1['log50k'].tolist(), dtype=torch.double)
                    log50k2 = torch.tensor(subgroup2['log50k'].tolist(), dtype=torch.double)
                    log50k_diff[length] = log50k1 - log50k2
                else:
                    label1 = torch.tensor(subgroup1['label'].tolist(), dtype=torch.long)
                    label2 = torch.tensor(subgroup2['label'].tolist(), dtype=torch.long)
                    label_diff[length] = label1 - label2

            preds_diff[mhc_name] = pred_diff
            labels_diff[mhc_name] = label_diff
            log50ks_diff[mhc_name] = log50k_diff

        # os.remove('peptides_mixmhcpred30.fasta')
        # os.remove('result_mixmhcpred30.tsv')
        if if_ba:
            return (preds_diff,), log50ks_diff
        else:
            return (preds_diff,), labels_diff
