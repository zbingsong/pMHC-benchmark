import pandas as pd
import torch
import os
import subprocess
import time
import typing
from . import BasePredictor, PredictorConfigs


class MHCSeqNet2Predictor(BasePredictor):
    '''
    Make sure you start the MHCSeqNet2 docker container before using this model!
    '''
    tasks = None
    _exe_dir = None
    _temp_dir = None
    _wd = None

    # @typing.override
    @classmethod
    def load(cls, predictor_configs: PredictorConfigs) -> None:
        cls._temp_dir = predictor_configs.temp_dir
        cls.tasks = ['EL']
        cls._wd = os.getcwd()
        cls._exe_dir = '~/repo/MHCSeqNet2'

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

        # df = df[187504:189758]
        peptides = df['peptide']
        mhcs = df['mhc_name'].str[:5] + '*' + df['mhc_name'].str[5:]
        input_df = pd.DataFrame({'Allele': mhcs, 'Peptide': peptides})
        input_path = os.path.join(cls._exe_dir, 'resources/datasets/input.csv')
        # Must include the index column, or will get KeyError: Allele
        input_df.to_csv(input_path)
        
        start_time = time.time_ns()
        run_result = subprocess.run('docker exec -w /home/bingo/repo/MHCSeqNet2 mhcseqnet2-dev-mhcseqnet2-1 python mhctool.py --MODE CSV --CSV_PATH "resources/datasets/input.csv" --PEPTIDE_COLUMN_NAME Peptide --ALLELE_COLUMN_NAME Allele --LOG_UNKNOW --LOG_UNKNOW_PATH "resources/unknow.log" --USE_ENSEMBLE --ALLELE_MAPPER_PATH resources/allele_mapper --OUTPUT_DIRECTORY "resources/outputs/output.csv" --TEMP_FILE_PATH "/tmp/mhcseqnet2_raw.csv"', shell=True, stdout=subprocess.DEVNULL)
        end_time = time.time_ns()
        assert run_result.returncode == 0
        times.append(end_time - start_time)
        try:
            output_path = os.path.join(cls._exe_dir, 'resources/outputs/output.csv')
            result_df = pd.read_csv(output_path, index_col=0)
            if len(result_df) != len(df):
                pass
            assert len(result_df) == len(df), f'Length mismatch: {len(result_df)} vs {len(df)}' 
        except Exception as e:
            raise e

        result_df['label'] = df['label']
        result_df['mhc'] = df['mhc']
        result_df['len'] = result_df['Peptide'].str.len()
        if 'log50k' in df.columns:
            result_df['log50k'] = df['log50k']

        for mhc_name, group in result_df.groupby('mhc'):
            pred = {}
            label = {}
            log50k = {}
            # print(result_df.columns)
            group.reset_index(drop=True, inplace=True)
            grouped_by_len = group.groupby('len')
            for length, subgroup in grouped_by_len:
                # print(subgroup.columns)
                # MHCSeqNet2 produces a score in (0, 1) where higher score => better epitope
                pred[length] = torch.tensor(subgroup['Prediction'].to_numpy(), dtype=torch.double)
                label[length] = torch.tensor(subgroup['label'].to_numpy(), dtype=torch.long)
                if 'log50k' in subgroup.columns:
                    log50k[length] = torch.tensor(subgroup['log50k'].to_numpy(), dtype=torch.double)
        
            preds[mhc_name] = pred
            labels[mhc_name] = label
            log50ks[mhc_name] = log50k

        # if os.path.exists('peptides_mhcfovea.csv'):
        #     os.remove('peptides_mhcfovea.csv')
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
        # if_ba = 'log50k1' in df.columns
        # df, _ = cls._filter_sensitivity(df)
        # if len(df) == 0:
        #     print('No valid peptides')
        #     return ({},), {}
        
        # preds_diff = {}
        # labels_diff = {}
        # log50ks_diff = {}

        # input_df = pd.DataFrame({'mhc': pd.concat([df['mhc_name'].transform(cls.__format_mhc), df['mhc_name'].transform(cls.__format_mhc)]), 'pep': pd.concat([df['peptide1'], df['peptide2']])})
        # input_df.to_csv(f'{cls._temp_dir}/bigmhc_in.csv', index=False)

        # run_result = subprocess.run([f'{cls._exe_dir}/env/bin/python', f'{cls._exe_dir}/src/predict.py', f'-i={cls._temp_dir}/bigmhc.csv', '-m=el', '-d=0', f'-o={cls._temp_dir}/bigmhc_out.csv'], stdout=subprocess.DEVNULL)
        # assert run_result.returncode == 0
            
        # result_df = pd.read_csv(f'{cls._temp_dir}/bigmhc_out.csv')
        # assert len(result_df) == 2 * len(df), f'Length mismatch: {len(result_df)} vs {2 * len(df)}'
        
        # result_df1 = result_df.iloc[:len(df)].reset_index(drop=True)
        # result_df2 = result_df.iloc[len(df):].reset_index(drop=True)
        # result_df1.rename(columns={'BigMHC_EL': 'BigMHC_EL1'}, inplace=True)
        # result_df1['BigMHC_EL2'] = result_df2['BigMHC_EL']
        # if if_ba:
        #     result_df1['log50k1'] = df['log50k1']
        #     result_df1['log50k2'] = df['log50k2']
        # else:
        #     result_df1['label1'] = df['label1']
        #     result_df1['label2'] = df['label2']

        # for mhc_name, group in result_df1.groupby('mhc'):
        #     pred_diff = None
        #     label_diff = None
        #     log50k_diff = None

        #     pred1 = 100.0 - torch.tensor(group['BigMHC_EL1'].tolist(), dtype=torch.double)
        #     pred2 = 100.0 - torch.tensor(group['BigMHC_EL2'].tolist(), dtype=torch.double)
        #     pred_diff = pred1 - pred2
        #     if if_ba:
        #         log50k1 = torch.tensor(group['log50k1'].tolist(), dtype=torch.double)
        #         log50k2 = torch.tensor(group['log50k2'].tolist(), dtype=torch.double)
        #         log50k_diff = log50k1 - log50k2
        #     else:
        #         label1 = torch.tensor(group['label1'].tolist(), dtype=torch.long)
        #         label2 = torch.tensor(group['label2'].tolist(), dtype=torch.long)
        #         label_diff = label1 - label2

        #     preds_diff[mhc_name] = pred_diff
        #     labels_diff[mhc_name] = label_diff
        #     log50ks_diff[mhc_name] = log50k_diff

        # # if os.path.exists('peptides_mhcfovea.csv'):
        # #     os.remove('peptides_mhcfovea.csv')
        # if if_ba:
        #     return (preds_diff,), log50ks_diff
        # else:
        #     return (preds_diff,), labels_diff
        pass
    
    # @typing.override
    @classmethod
    def _filter(cls, df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
        filtered = df
        filtered = filtered[filtered['mhc_name'].str.startswith(('HLA-A', 'HLA-B', 'HLA-C'))]
        filtered = filtered[~filtered['peptide'].str.contains(r'[UZ]', regex=True)]
        filtered = filtered[filtered['peptide'].str.len() <= 15]
        # filtered = filtered[filtered['peptide'].str.len() >= 8]
        if len(df) != len(filtered):
            filtered = filtered.reset_index(drop=True)
            print('Skipped peptides: ', len(df) - len(filtered))
        return filtered, len(df) - len(filtered)
    
    # @typing.override
    @classmethod
    def _filter_sensitivity(cls, df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
        filtered = df
        filtered = filtered[filtered['mhc_name'].str.startswith(('HLA-A', 'HLA-B', 'HLA-C'))]
        filtered = filtered[~filtered['peptide1'].str.contains(r'[UZ]', regex=True)]
        filtered = filtered[~filtered['peptide2'].str.contains(r'[UZ]', regex=True)]
        filtered = filtered[filtered['peptide1'].str.len() <= 15]
        # filtered = filtered[filtered['peptide1'].str.len() >= 8]
        filtered = filtered[filtered['peptide2'].str.len() <= 15]
        # filtered = filtered[filtered['peptide2'].str.len() >= 8]
        if len(df) != len(filtered):
            filtered = filtered.reset_index(drop=True)
            print('Skipped peptides: ', len(df) - len(filtered))
        return filtered, len(df) - len(filtered)
    
    @classmethod
    def __format_mhc(cls, mhc: str) -> str:
        return mhc
