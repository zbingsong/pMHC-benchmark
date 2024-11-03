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
                with open('peptides_netmhcpan.txt', 'w') as file:
                    for peptide in peptides:
                        file.write(peptide + '\n')
                start_time = time.time_ns()
                run_result = subprocess.run([cls._executable, '-p', 'peptides_netmhcpan.txt', '-a', mhc_name, '-l', str(length), '-BA', '-xls', '-xlsfile', 'out_netmhcpan.tsv'], stdout=subprocess.DEVNULL)
                end_time = time.time_ns()
                assert run_result.returncode == 0

                result_df = pd.read_csv('out_netmhcpan.tsv', sep='\t', skiprows=[0])
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

        os.remove('out_netmhcpan.tsv')
        os.remove('peptides_netmhcpan.txt')
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
        df = df.groupby('mhc_name')

        BA_preds_diff = {}
        EL_preds_diff = {}
        log50ks_diff = {}

        for mhc_name, group in df:
            BA_pred_diff = {}
            EL_pred_diff = {}
            log50k_diff = {}
            group = group.reset_index(drop=True)
            grouped_by_len = group.groupby(group['peptide1'].str.len())

            for length, subgroup in grouped_by_len:
                peptides1 = subgroup['peptide1'].tolist()
                peptides2 = subgroup['peptide2'].tolist()
                with open('peptides1_netmhcpan.txt', 'w') as file:
                    for peptide in peptides1:
                        file.write(peptide + '\n')
                with open('peptides2_netmhcpan.txt', 'w') as file:
                    for peptide in peptides2:
                        file.write(peptide + '\n')
                process1 = subprocess.Popen([cls._executable, '-p', 'peptides1_netmhcpan.txt', '-a', mhc_name, '-l', str(length), '-BA', '-xls', '-xlsfile', 'out1_netmhcpan.tsv'], stdout=subprocess.DEVNULL)
                process2 = subprocess.Popen([cls._executable, '-p', 'peptides2_netmhcpan.txt', '-a', mhc_name, '-l', str(length), '-BA', '-xls', '-xlsfile', 'out2_netmhcpan.tsv'], stdout=subprocess.DEVNULL)
                process1.wait()
                process2.wait()
                assert process1.returncode == 0 and process2.returncode == 0

                result_df1 = pd.read_csv('out1_netmhcpan.tsv', sep='\t', skiprows=[0])
                result_df2 = pd.read_csv('out2_netmhcpan.tsv', sep='\t', skiprows=[0])
                assert len(result_df1) == len(subgroup), f'Length mismatch: {len(result_df1)} != {len(subgroup)} for {mhc_name}'
                assert len(result_df2) == len(subgroup), f'Length mismatch: {len(result_df2)} != {len(subgroup)} for {mhc_name}'
                
                BA_pred1 = 100 - torch.tensor(result_df1['BA_Rank'].tolist(), dtype=torch.double)
                BA_pred2 = 100 - torch.tensor(result_df2['BA_Rank'].tolist(), dtype=torch.double)
                EL_pred1 = 100 - torch.tensor(result_df1['EL_Rank'].tolist(), dtype=torch.double)
                EL_pred2 = 100 - torch.tensor(result_df2['EL_Rank'].tolist(), dtype=torch.double)
                log50k1 = torch.tensor(subgroup['log50k1'].tolist(), dtype=torch.double)
                log50k2 = torch.tensor(subgroup['log50k2'].tolist(), dtype=torch.double)

                BA_pred_diff[length] = BA_pred1 - BA_pred2
                EL_pred_diff[length] = EL_pred1 - EL_pred2
                log50k_diff[length] = log50k1 - log50k2
            
            BA_preds_diff[mhc_name] = BA_pred_diff
            EL_preds_diff[mhc_name] = EL_pred_diff
            log50ks_diff[mhc_name] = log50k_diff

        os.remove('out1_netmhcpan.tsv')
        os.remove('out2_netmhcpan.tsv')
        os.remove('peptides1_netmhcpan.txt')
        os.remove('peptides2_netmhcpan.txt')
        return (BA_preds_diff, EL_preds_diff), log50ks_diff
