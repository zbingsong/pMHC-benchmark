import pandas as pd
import torch

import subprocess
import os
import json
import time
import pathlib

from . import BasePredictor, PredictorConfigs


class NetMHCpanPredictor(BasePredictor):
    tasks = None
    _temp_dir = None
    _executable = None

    @classmethod
    def load(cls, predictor_configs: PredictorConfigs) -> None:
        cls._temp_dir = predictor_configs.temp_dir
        cls.tasks = ['BA', 'EL']
        curr_dir = pathlib.Path(__file__).parent
        with open(f'{curr_dir}/configs.json', 'r') as f:
            configs = json.load(f)
            cls._executable = os.path.expanduser(configs['exe_path'])

    @classmethod
    def run_retrieval(
            cls,
            df: pd.DataFrame
    ) -> tuple[tuple[dict[str, dict[str, torch.DoubleTensor]], ...], dict[str, dict[str, torch.LongTensor]], dict[str, dict[str, torch.DoubleTensor]], int]:
        df = df.groupby('mhc_name')

        BA_preds = {}
        EL_preds = {}
        labels = {}
        log50ks = {}
        times = []

        for mhc_name, group in df:
            BA_pred = {}
            EL_pred = {}
            label = {}
            log50k = {}
            group = group.reset_index(drop=True)
            grouped_by_len = group.groupby(group['peptide'].str.len())

            for length, subgroup in grouped_by_len:
                # if length > 14:
                #     continue
                peptides = subgroup['peptide'].tolist()
                with open(f'{cls._temp_dir}/peptides_netmhcpan.txt', 'w') as file:
                    for peptide in peptides:
                        file.write(peptide + '\n')
                start_time = time.time_ns()
                run_result = subprocess.run([cls._executable, '-p', f'{cls._temp_dir}/peptides_netmhcpan.txt', '-a', mhc_name, '-l', str(length), '-BA', '-xls', '-xlsfile', f'{cls._temp_dir}/out_netmhcpan.tsv'], stdout=subprocess.DEVNULL)
                end_time = time.time_ns()
                assert run_result.returncode == 0

                result_df = pd.read_csv(f'{cls._temp_dir}/out_netmhcpan.tsv', sep='\t', skiprows=[0])
                assert len(result_df) == len(subgroup), f'Length mismatch: {len(result_df)} != {len(subgroup)} for {mhc_name}'
                BA_pred[length] = 100 - torch.tensor(result_df['BA_Rank'].tolist(), dtype=torch.double)
                EL_pred[length] = 100 - torch.tensor(result_df['EL_Rank'].tolist(), dtype=torch.double)
                label[length] = torch.tensor(subgroup['label'].tolist(), dtype=torch.long)
                if 'log50k' in subgroup.columns:
                    log50k[length] = torch.tensor(subgroup['log50k'].tolist(), dtype=torch.double)
                times.append(end_time - start_time)
            
            BA_preds[mhc_name] = BA_pred
            EL_preds[mhc_name] = EL_pred
            labels[mhc_name] = label
            log50ks[mhc_name] = log50k

        # os.remove('out_netmhcpan.tsv')
        # os.remove('peptides_netmhcpan.txt')
        return (BA_preds, EL_preds), labels, log50ks, sum(times)
    
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
        
        df = df.groupby('mhc_name')

        BA_preds_diff = {}
        EL_preds_diff = {}
        labels_diff = {}
        log50ks_diff = {}

        for mhc_name, group in df:
            BA_pred_diff = {}
            EL_pred_diff = {}
            label_diff = {}
            log50k_diff = {}
            group = group.reset_index(drop=True)
            grouped_by_len = group.groupby(group['peptide1'].str.len())

            for length, subgroup in grouped_by_len:
                peptides1 = subgroup['peptide1'].tolist()
                peptides2 = subgroup['peptide2'].tolist()
                with open(f'{cls._temp_dir}/peptides_netmhcpan.txt', 'w') as file:
                    for peptide in peptides1:
                        file.write(peptide + '\n')
                    for peptide in peptides2:
                        file.write(peptide + '\n')
                run_result = subprocess.run([cls._executable, '-p', f'{cls._temp_dir}/peptides_netmhcpan.txt', '-a', mhc_name, '-l', str(length), '-BA', '-xls', '-xlsfile', f'{cls._temp_dir}/out_netmhcpan.tsv'], stdout=subprocess.DEVNULL)
                assert run_result.returncode == 0

                result_df = pd.read_csv(f'{cls._temp_dir}/out_netmhcpan.tsv', sep='\t', skiprows=[0])
                assert len(result_df) == len(subgroup), f'Length mismatch: {len(result_df)} != {2 * len(subgroup)} for {mhc_name}'
                
                result_df1 = result_df.iloc[:len(subgroup)]
                result_df2 = result_df.iloc[len(subgroup):]
                BA_pred1 = 100 - torch.tensor(result_df1['BA_Rank'].tolist(), dtype=torch.double)
                BA_pred2 = 100 - torch.tensor(result_df2['BA_Rank'].tolist(), dtype=torch.double)
                BA_pred_diff[length] = BA_pred1 - BA_pred2
                EL_pred1 = 100 - torch.tensor(result_df1['EL_Rank'].tolist(), dtype=torch.double)
                EL_pred2 = 100 - torch.tensor(result_df2['EL_Rank'].tolist(), dtype=torch.double)
                EL_pred_diff[length] = EL_pred1 - EL_pred2
                if if_ba:
                    log50k1 = torch.tensor(subgroup['log50k1'].tolist(), dtype=torch.double)
                    log50k2 = torch.tensor(subgroup['log50k2'].tolist(), dtype=torch.double)
                    log50k_diff[length] = log50k1 - log50k2
                else:
                    label1 = torch.tensor(subgroup['label1'].tolist(), dtype=torch.long)
                    label2 = torch.tensor(subgroup['label2'].tolist(), dtype=torch.long)
                    label_diff[length] = label1 - label2
            
            BA_preds_diff[mhc_name] = BA_pred_diff
            EL_preds_diff[mhc_name] = EL_pred_diff
            labels_diff[mhc_name] = label_diff
            log50ks_diff[mhc_name] = log50k_diff

        # os.remove('out1_netmhcpan.tsv')
        # os.remove('out2_netmhcpan.tsv')
        # os.remove('peptides1_netmhcpan.txt')
        # os.remove('peptides2_netmhcpan.txt')
        if if_ba:
            return (BA_preds_diff, EL_preds_diff), log50ks_diff
        else:
            return (BA_preds_diff, EL_preds_diff), labels_diff
