import pandas as pd

import json
import argparse
import enum
import pathlib
import time

import src


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
    MIXMHCPRED = src.MixMHCpredPredictor()
    NETMHCPAN = src.NetMHCpanPredictor()


def main(model_name: str):
    '''
    Main function to test models

    Parameters:
    model_name (str): name of the model to test
    '''
    if model_name.upper() not in Predictors.__members__:
        raise ValueError(f'Invalid model name: {model_name}')
    
    start_time = time.time_ns()

    predictor: src.BasePredictor = Predictors[model_name.upper()].value
    predictor.load()

    with open('configs.json', 'r') as f:
        configs = json.load(f)
        filelist_path = configs['filelist_path']
        data_dir = configs['data_dir']
        output_dir = configs['output_dir']

    filenames = []
    with open(filelist_path, 'r') as f:
        for line in f:
            if line.strip() != '' and line[0] != '#':
                filenames.append(line.strip())
    
    pathlib.Path(f'{output_dir}/{model_name}').mkdir(parents=True, exist_ok=True)

    for filename in filenames:
        df = pd.read_csv(f'{data_dir}/{filename}')
        df = df.astype({'label': int})
        if 'log50k' in df.columns:
            df = df.astype({'log50k': float})
        df = df.groupby('mhc_name')
        predictions, labels, log50ks, time_taken = predictor.run(df)
        for prediction, task in zip(predictions, predictor.tasks):
            with open(f'{output_dir}/{model_name}/{task}_{filename[:-4]}.txt', 'w') as file:
                src.test_retrieval(prediction, labels, time_taken, file)
                if task == 'BA':
                    src.test_regression(prediction, log50ks, file)

    # test sensitivity
    df = pd.read_csv(f'{data_dir}/pairs.csv')
    df = df.astype({'label1': int, 'label2': int, 'log50k1': float, 'log50k2': float})
    prediction_diffs, log50k_diffs = predictor.run_sensitivity(df)
    for prediction_diff, log50k_diff, task in zip(prediction_diffs, log50k_diffs, predictor.tasks):
        with open(f'{output_dir}/{model_name}/{task}_sensitivity.txt', 'w') as file:
            src.test_sensitivity(prediction_diff, log50k_diff, file)

    end_time = time.time_ns()
    with open(f'{output_dir}/{model_name}/time.txt', 'w') as file:
        file.write(f'Time elapsed: {(end_time - start_time)} ns\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, required=True, help='Model name')
    args = parser.parse_args()
    main(args.model)
