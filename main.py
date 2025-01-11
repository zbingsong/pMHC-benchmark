import pandas as pd

import json
import argparse
import enum
import pathlib

import src


CONFIGS_DIR = 'configs'


class Predictors(enum.Enum):
    '''
    Enum class for model functions

    Function parameters:
    - df: pd.api.typing.DataFrameGroupBy
        - DataFrame grouped by MHC names
        - Contain m MHCs where each MHC corresponds to n peptides (n may vary)

    Function return:
    - tuple of 3 elements
        - tuple[list[torch.DoubleTensor], ...]
            - Variable length tuple of lists of torch.DoubleTensor
            - Each list (of length m) contains predictions for one task
            - Each tensor has length n
        - list[torch.LongTensor]
            - List of labels as tensors with length m
            - Each tensor has length n
        - list[torch.DoubleTensor]
            - List of log50k values as tensors with length m if this DataFrame has "log50k" column, or 0 if not
            - Each tensor (if list has a positive length) has length n
    '''
    MHCFLURRY = src.MHCflurryPredictor()
    MIXMHCPRED22 = src.MixMHCpred22Predictor()
    MIXMHCPRED30 = src.MixMHCpred30Predictor()
    NETMHCPAN = src.NetMHCpanPredictor()
    ANTHEM = src.AnthemPredictor()
    TRANSPHLA = src.TransPHLAPredictor()
    MHCFOVEA = src.MHCfoveaPredictor()
    ESM2 = src.ESM2Predictor()


def main(model_name: str):
    '''
    Main function to test models

    Parameters:
    model_name (str): name of the model to test
    '''
    if model_name.upper() not in Predictors.__members__:
        raise ValueError(f'Invalid model name: {model_name}')

    predictor: src.BasePredictor = Predictors[model_name.upper()].value

    with open(f'{CONFIGS_DIR}/configs.json', 'r') as f:
        configs = json.load(f)
        filelist_path = f'{CONFIGS_DIR}/{configs["filelist_path"]}'
        filelist_sq_path = f'{CONFIGS_DIR}/{configs["filelist_sq_path"]}'
        filelist_sensitivity_el_path = f'{CONFIGS_DIR}/{configs["filelist_sensitivity_el_path"]}'
        filelist_sensitivity_ba_path = f'{CONFIGS_DIR}/{configs["filelist_sensitivity_ba_path"]}'
        data_dir = configs['data_dir']
        output_dir = configs['output_dir']
        temp_dir = configs['temp_dir']
    
    predictor_configs = src.PredictorConfigs(temp_dir)
    predictor.load(predictor_configs)

    filenames = []
    with open(filelist_path, 'r') as f:
        for line in f:
            if line.strip() != '' and line[0] != '#':
                filenames.append(line.strip())

    filenames_sq = []
    with open(filelist_sq_path, 'r') as f:
        for line in f:
            if line.strip() != '' and line[0] != '#':
                filenames_sq.append(line.strip())

    filenames_sensitivity_el = []
    with open(filelist_sensitivity_el_path, 'r') as f:
        for line in f:
            if line.strip() != '' and line[0] != '#':
                filenames_sensitivity_el.append(line.strip())
    
    filenames_sensitivity_ba = []
    with open(filelist_sensitivity_ba_path, 'r') as f:
        for line in f:
            if line.strip() != '' and line[0] != '#':
                filenames_sensitivity_ba.append(line.strip())
    
    pathlib.Path(f'{output_dir}/{model_name}').mkdir(parents=True, exist_ok=True)

    for filename in filenames:
        df = pd.read_csv(f'{data_dir}/{filename}')
        df = df.astype({'label': int})
        if_reg = False
        if 'log50k' in df.columns:
            df = df.astype({'log50k': float})
            if_reg = True
        # start_time = time.time_ns()
        predictions, labels, log50ks, time_taken, num_skipped = predictor.run_retrieval(df)
        if time_taken == 0:
            print(f'{model_name} is not compatible with any data in {filename}')
            continue
        # end_time = time.time_ns()
        for prediction, task in zip(predictions, predictor.tasks):
            name = f'{output_dir}/{model_name}/{task}_{filename[:-4]}'
            src.test_retrieval(prediction, labels, time_taken, num_skipped, name)
            if if_reg:
                src.test_regression(prediction, log50ks, name)

    # sqaure dataset
    for filename in filenames_sq:
        df = pd.read_csv(f'{data_dir}/{filename}')
        df = df.astype({'label': int})
        predictions, labels, log50ks, time_taken, num_skipped = predictor.run_sq(df)
        if time_taken == 0:
            print(f'{model_name} is not compatible with any data in {filename}')
            continue
        for prediction, task in zip(predictions, predictor.tasks):
            name = f'{output_dir}/{model_name}/{task}_{filename[:-4]}'
            src.test_retrieval(prediction, labels, time_taken, num_skipped, name)

    # test eluted ligand sensitivity
    for filename in filenames_sensitivity_el:
        df = pd.read_csv(f'{data_dir}/{filename}')
        df = df.astype({'label1': int, 'label2': int})
        prediction_diffs, label_diff = predictor.run_sensitivity(df)
        if len(label_diff) == 0:
            print(f'{model_name} is not compatible with any data in {filename}')
            continue
        for prediction_diff, task in zip(prediction_diffs, predictor.tasks):
            name = f'{output_dir}/{model_name}/{task}_{filename[:-4]}'
            src.test_sensitivity_el(prediction_diff, label_diff, name)

    # test binding affinity sensitivity
    for filename in filenames_sensitivity_ba:
        df = pd.read_csv(f'{data_dir}/{filename}')
        df = df.astype({'log50k1': float, 'log50k2': float})
        prediction_diffs, log50k_diff = predictor.run_sensitivity(df)
        if len(log50k_diff) == 0:
            print(f'{model_name} is not compatible with any data in {filename}')
            continue
        for prediction_diff, task in zip(prediction_diffs, predictor.tasks):
            name = f'{output_dir}/{model_name}/{task}_{filename[:-4]}'
            src.test_sensitivity_ba(prediction_diff, log50k_diff, name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, required=True, help='Model name')
    args = parser.parse_args()
    main(args.model)
