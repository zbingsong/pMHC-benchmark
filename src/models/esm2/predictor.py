import pandas as pd
import torch
import esm

import os
import json
import time
import pathlib
import typing

from . import BasePredictor, PredictorConfigs


class Model:
    def __init__(self, weight_path: str, device: str | int) -> None:
        '''
        Uses bf16
        '''
        self.model, alphabet = esm.pretrained.load_model_and_alphabet(weight_path)
        self.model = self.model.to(torch.bfloat16)
        self.model = self.model.to(device)
        self.tokenizer = alphabet.get_batch_converter()
        self.device = device
    
    def __call__(self, sequences: str | list[str]) -> torch.FloatTensor:
        '''
        Output embeddings are moved to cpu before return
        '''
        if isinstance(sequences, str):
            sequences = [sequences]
        data = [('', seq) for seq in sequences]
        _, _, input_ids = self.tokenizer(data)
        input_ids = input_ids.to(self.device)
        hidden: dict[str, torch.FloatTensor] = self.model(input_ids, repr_layers=[self.model.num_layers])
        repr = hidden['representations'][self.model.num_layers] # shape (batch_size, seq_len, hidden_size)
        out = repr[:, 0] # shape (batch_size, hidden_size)
        out = torch.nn.functional.normalize(out, p=2, dim=-1).cpu()
        return out


class ESM2Predictor(BasePredictor):
    tasks = None
    _predictor = None
    _temp_dir = None
    _batch_size=None

    # @typing.override
    @classmethod
    def load(cls, predictor_configs: PredictorConfigs) -> None:
        cls._temp_dir = predictor_configs.temp_dir
        cls.tasks = ['EL']
        curr_dir = pathlib.Path(__file__).parent
        with open(f'{curr_dir}/configs.json', 'r') as f:
            configs = json.load(f)
            weight_path = os.path.expanduser(configs['weight_path'])
            device = configs['device']
            cls._batch_size = configs['batch_size']
        cls._predictor = Model(weight_path, device)

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
        grouped_df = df.groupby('mhc_name')
        
        preds = {}
        labels = {}
        log50ks = {}
        times = []

        unique_peptides = df['peptide'].unique()
        should_do_lookup = (len(unique_peptides) < len(df) // 1.5) # too many repeated peptides, so build a lookup table
        
        with torch.no_grad():
            start_time = time.time_ns()

            unique_mhcs = df['mhc_seq'].unique()
            mhc_embeddings: torch.FloatTensor = cls._predictor(unique_mhcs.tolist()).float()
            mhc_to_embedding = {mhc: embedding for mhc, embedding in zip(unique_mhcs, mhc_embeddings)}
            if should_do_lookup:
                peptide_embedding_list: list[torch.FloatTensor] = []
                for i in range(0, len(unique_peptides), cls._batch_size):
                    peptide_embedding: torch.FloatTensor = cls._predictor(unique_peptides[i:i+cls._batch_size].tolist()).float()
                    peptide_embedding_list.append(peptide_embedding)
                peptide_embeddings: torch.FloatTensor = torch.cat(peptide_embedding_list, dim=0)
                peptide_to_embedding = {peptide: embedding for peptide, embedding in zip(unique_peptides, peptide_embeddings)}

            end_time = time.time_ns()
            times.append(end_time - start_time)

            for i, (mhc_name, group) in enumerate(grouped_df):
                group.reset_index(drop=True, inplace=True)
                pred = {}
                label = {}
                log50k = {}
                start_time = time.time_ns()

                mhc: str = group['mhc_seq'].iloc[0]
                mhc_tensor = mhc_to_embedding[mhc].unsqueeze(0)
                if should_do_lookup:
                    peptide_tensors = torch.stack([peptide_to_embedding[peptide] for peptide in group['peptide']], dim=0)
                else:
                    peptides = group['peptide']
                    peptide_embedding_list: list[torch.FloatTensor] = []
                    for i in range(0, len(peptides), cls._batch_size):
                        peptide_embedding: torch.FloatTensor = cls._predictor(peptides[i:i+cls._batch_size].tolist()).float()
                        peptide_embedding_list.append(peptide_embedding)
                    peptide_tensors = torch.cat(peptide_embedding_list, dim=0)
                similarities: torch.FloatTensor = torch.nn.functional.cosine_similarity(mhc_tensor, peptide_tensors, dim=-1)

                end_time = time.time_ns()
                times.append(end_time - start_time)

                group['preds'] = similarities.tolist()
                grouped_by_len = group.groupby(group['peptide'].str.len())
                for length, subgroup in grouped_by_len:
                    pred[length] = torch.tensor(subgroup['preds'].tolist(), dtype=torch.double)
                    label[length] = torch.tensor(subgroup['label'].tolist(), dtype=torch.long)
                    if 'log50k' in subgroup.columns:
                        log50k[length] = torch.tensor(subgroup['log50k'].tolist(), dtype=torch.double)
                
                preds[mhc_name] = pred
                labels[mhc_name] = label
                log50ks[mhc_name] = log50k

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

        unique_mhc_seqs = df['mhc_seq'].unique()

        with torch.no_grad():
            mhc_embeddings: torch.FloatTensor = cls._predictor(unique_mhc_seqs.tolist()).float()
            # shape of value: (dim_latent,)
            mhc_seq_to_embedding = {mhc: embedding for mhc, embedding in zip(unique_mhc_seqs, mhc_embeddings)}
            # shape (df_len, dim_latent)
            mhc_features: torch.FloatTensor = torch.stack([mhc_seq_to_embedding[mhc] for mhc in df['mhc_seq']], dim=0)
            
            peptides_wt = df['peptide1']
            peptides_mut = df['peptide2']
            peptide_wt_embedding_list = []
            peptide_mut_embedding_list = []
            for i in range(0, len(peptides_wt), cls._batch_size):
                peptide_wt_feature: torch.FloatTensor = cls._predictor(peptides_wt[i:i+cls._batch_size].tolist()).float()
                peptide_mut_feature: torch.FloatTensor = cls._predictor(peptides_mut[i:i+cls._batch_size].tolist()).float()
                peptide_wt_embedding_list.append(peptide_wt_feature)
                peptide_mut_embedding_list.append(peptide_mut_feature)
            peptide_wt_features: torch.FloatTensor = torch.cat(peptide_wt_embedding_list, dim=0)
            peptide_mut_features: torch.FloatTensor = torch.cat(peptide_mut_embedding_list, dim=0)
            assert mhc_features.size() == peptide_wt_features.size() == peptide_mut_features.size(), 'Sensitivity: Shapes of features do not match'
            # shape (df_len,)
            similarities_wt: torch.FloatTensor = torch.nn.functional.cosine_similarity(mhc_features, peptide_wt_features, dim=-1)
            similarities_mut: torch.FloatTensor = torch.nn.functional.cosine_similarity(mhc_features, peptide_mut_features, dim=-1)
            df['predictions_diff'] = similarities_wt - similarities_mut
            if if_ba:
                # shape (df_len,)
                log50k1: torch.FloatTensor = torch.tensor(df['log50k1'].tolist(), dtype=torch.float)
                log50k2: torch.FloatTensor = torch.tensor(df['log50k2'].tolist(), dtype=torch.float)
                df['log50ks_diff'] = log50k1 - log50k2
            else:
                label1: torch.IntTensor = torch.tensor(df['label1'].tolist(), dtype=torch.int)
                label2: torch.IntTensor = torch.tensor(df['label2'].tolist(), dtype=torch.int)
                df['labels_diff'] = label1 - label2

        grouped_df = df.groupby('mhc_name')
        if if_ba:
            for mhc_name, group in grouped_df:
                preds_diff[mhc_name] = group['predictions_diff']
                log50ks_diff[mhc_name] = group['log50ks_diff']
            return (preds_diff,), log50ks_diff
        else:
            for mhc_name, group in grouped_df:
                preds_diff[mhc_name] = group['predictions_diff']
                labels_diff[mhc_name] = group['labels_diff']
            return (preds_diff,), labels_diff

    # @typing.override
    @classmethod
    def _filter(cls, df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
        return df, 0
    
    # @typing.override
    @classmethod
    def _filter_sensitivity(cls, df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
        return df, 0
    