import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import collections

from . import metrics


RETRIEVAL_KS = [1, 3, 5, 10, 20] # for precision@k
ALPHAS = [20, 85, 150] # for BEDROC, 20 is suggested by authors, 85 is used by DrugCLIP
PROPORTIONS = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2] # for EF

TEST_RETRIEVAL_TEMPLATE = '''Test Retrieval by MHC:
retrieval k: {}
alphas: {}
proportions: {}
positives: {}
totals: {}
auroc pooled: {:.6f}
auroc averaged by MHC: {:.6f}
auprc pooled: {:.6f}
auprc averaged by MHC: {:.6f}
precision@k pooled: {}
precision@k averaged by MHC: {}
bedroc pooled: {}
bedroc averaged by MHC: {}
enrichment_factor pooled: {}
enrichment_factor averaged by MHC: {}
auac pooled: {}
auac averaged by MHC: {}
time: {} nanoseconds
skipped rows: {}
'''

TEST_SENSITIVITY_EL_TEMPLATE = '''Test Sensitivity EL:
accuracy: {:.4f}
precision: {:.4f}
recall: {:.4f}
f1_score: {:.4f}
'''

TEST_SENSITIVITY_BA_TEMPLATE = '''Test Sensitivity BA:
accuracy: {:.4f}
precision: {:.4f}
recall: {:.4f}
f1_score: {:.4f}
pearson_corrcoef: {:.6f}
spearman_corrcoef: {:.6f}
'''


def _plot_predictions(predictions: torch.DoubleTensor, output_filename: str) -> None:
    '''
    Parameters:
    predictions (torch.FloatTensor): tensor of shape (n, ), containing the cosine similarities between MHC and peptide pairs
    '''
    # Plot histogram of test predictions
    plt.hist(predictions.float(), bins=200)
    plt.xlabel('EL Score')
    plt.ylabel('Frequency')
    plt.savefig(f'{output_filename}_predictions.png')
    plt.close()


def _plot_sensitivity_el(
        predictions_diff: torch.DoubleTensor,
        binary_labels_diff: torch.LongTensor,
        output_filename: str
) -> None:
    # flip the sign of the predictions whose corresponding labels are 0
    data = torch.where(binary_labels_diff == 0, -predictions_diff, predictions_diff).numpy()
    plt.figure(figsize=(8, 8))
    plt.violinplot(data, showmeans=False, showextrema=False)
    # draw a horizontal line at y=0
    plt.axhline(y=0, color='r', linestyle='--', linewidth=1)
    quartile1, medians, quartile3 = np.percentile(data, [25, 50, 75])
    whiskers_min = np.min(data[data > quartile1 - 1.5 * (quartile3 - quartile1)])
    whiskers_max = np.max(data[data < quartile3 + 1.5 * (quartile3 - quartile1)])
    plt.vlines(1, whiskers_min, whiskers_max, linestyle='-', linewidth=1)
    plt.vlines(1, quartile1, quartile3, linestyle='-', linewidth=15)
    plt.hlines(medians, 0.9, 1.1, linestyle='-', linewidth=1)
    plt.hlines([whiskers_min, whiskers_max], 0.975, 1.025, linestyle='-', linewidth=1)
    plt.ylabel('Difference in EL Score')
    plt.tight_layout()
    plt.savefig(f'{output_filename}.png')
    plt.close()


def _plot_sensitivity_ba(
        predictions_diff: torch.DoubleTensor, 
        log50ks_diff: torch.DoubleTensor, 
        output_filename: str
) -> None:
    plt.figure(figsize=(8, 8))
    plt.plot(predictions_diff.numpy(), log50ks_diff.numpy(), '.')
    plt.xlabel('Predicted difference in BA')
    plt.ylabel('Measured difference in BA log50k')
    plt.savefig(f'{output_filename}.png')
    plt.close()


def _compute_metrics(
        output_df: pd.DataFrame,
        preds: torch.DoubleTensor,
        labs: torch.LongTensor,
        mhc_name: str,
        suffix: str | int,
) -> None:
    sorted_preds, sorted_indices = torch.sort(preds, descending=True)
    sorted_labs = labs[sorted_indices]
    auroc = metrics.compute_binary_auroc(sorted_preds, sorted_labs)
    output_df.loc[f'auroc_{suffix}', mhc_name] = auroc.item()
    auprc = metrics.compute_binary_auprc(sorted_preds, sorted_labs)
    output_df.loc[f'auprc_{suffix}', mhc_name] = auprc.item()
    for k in RETRIEVAL_KS:
        precision_at_k = metrics.compute_retrieval_precision(sorted_labs, top_k=k)
        output_df.loc[f'precision@{k}_{suffix}', mhc_name] = precision_at_k.item()
    for alpha in ALPHAS:
        bedroc = metrics.compute_bedroc(sorted_labs, alpha=alpha)
        output_df.loc[f'bedroc_{alpha}_{suffix}', mhc_name] = bedroc.item()
    for proportion in PROPORTIONS:
        enrichment_factor = metrics.compute_ef(sorted_labs, proportion=proportion)
        output_df.loc[f'enrichment_factor_{proportion}_{suffix}', mhc_name] = enrichment_factor.item()
        auac = metrics.compute_auac(sorted_labs, proportion=proportion)
        output_df.loc[f'auac_{proportion}_{suffix}', mhc_name] = auac.item()


def _compute_averged_metrics(
        output_df: pd.DataFrame,
) -> dict[str, float | np.ndarray]:
    filtered_df = output_df[output_df.index.str.endswith('overall_by_mhc')]
    filtered_df = filtered_df.drop('overall_by_len', axis=1)
    averages = filtered_df.mean(axis=1)
    precision_at_k = np.array([averages[f'precision@{k}_overall_by_mhc'] for k in RETRIEVAL_KS])
    bedroc = np.array([averages[f'bedroc_{alpha}_overall_by_mhc'] for alpha in ALPHAS])
    auac = np.array([averages[f'auac_{proportion}_overall_by_mhc'] for proportion in PROPORTIONS])
    enrichment_factors = np.array([averages[f'enrichment_factor_{proportion}_overall_by_mhc'] for proportion in PROPORTIONS])
    result = {
        'auroc': averages['auroc_overall_by_mhc'],
        'auprc': averages['auprc_overall_by_mhc'],
        'precision_at_k': precision_at_k,
        'bedroc': bedroc,
        'auac': auac,
        'enrichment_factors': enrichment_factors
    }
    return result


def test_retrieval(
        predictions: dict[str, dict[str, torch.DoubleTensor]], 
        labels: dict[str, dict[str, torch.LongTensor]], 
        time_taken: int,
        num_skipped: int,
        output_filename: str,
) -> None:
    # if predictions of an mhc is empty, drop it
    predictions = {k: v for k, v in predictions.items() if len(v) > 0}
    # find the max and min peptide length for building dataframes
    min_pep_len = 50
    max_pep_len = 0
    for preds in predictions.values():
        min_len = min(preds.keys())
        max_len = max(preds.keys())
        min_pep_len = min(min_pep_len, min_len)
        max_pep_len = max(max_pep_len, max_len)

    columns = sorted(list(predictions.keys())) + ['overall_by_len']
    indices_per_test = list(range(min_pep_len, max_pep_len + 1)) + ['overall_by_mhc']
    tests = ['auroc', 'auprc'] + [f'bedroc_{alpha}' for alpha in ALPHAS] + [f'auac_{proportion}' for proportion in PROPORTIONS] + [f'precision@{k}' for k in RETRIEVAL_KS] + [f'enrichment_factor_{proportion}' for proportion in PROPORTIONS]
    indices = [f'{test}_{index}' for test in tests for index in indices_per_test]

    num_positive_peptides: int = 0
    num_total_peptides: int = 0
    # columns are MHC names and 'overall_by_len', indices are metrics, where each metric has a row for each peotide length and 'overall_by_mhc'
    output_df = pd.DataFrame(columns=columns, index=indices, dtype=float)

    predictions_by_length = collections.defaultdict(list)
    labels_by_length = collections.defaultdict(list)
    # for computing metrics for all data
    all_predictions = []
    all_labels = []

    for mhc_name in predictions.keys():
        pred_dict = predictions[mhc_name]
        lab_dict = labels[mhc_name]
        preds_list = []
        labs_list = []

        # for each sequence length, calculate metrics
        for seq_length in pred_dict.keys():
            preds = pred_dict[seq_length]
            labs = lab_dict[seq_length]
            preds_list.append(preds)
            labs_list.append(labs)
            num_positive_peptides += labs.sum().item()
            num_total_peptides += labs.size(0)

            _compute_metrics(output_df, preds, labs, mhc_name, seq_length)
            predictions_by_length[seq_length].append(preds)
            labels_by_length[seq_length].append(labs)
        
        # concatenate predictions and labels for all sequence lengths, calculate metrics for each MHC
        preds = torch.cat(preds_list)
        labs = torch.cat(labs_list)
        _compute_metrics(output_df, preds, labs, mhc_name, 'overall_by_mhc')

    # concatenate predictions and labels for all MHCs, calculate metrics for each peptide length
    for length in range(min_pep_len, max_pep_len + 1):
        # if have predictions for this length
        if len(predictions_by_length[length]) > 0:
            preds = torch.cat(predictions_by_length[length])
            labs = torch.cat(labels_by_length[length])
            all_predictions.append(preds)
            all_labels.append(labs)
            _compute_metrics(output_df, preds, labs, 'overall_by_len', length)
        else:
            predictions_by_length[length] = torch.tensor([], dtype=torch.double)
            labels_by_length[length] = torch.tensor([], dtype=torch.long)
    
    total_preds = torch.cat(all_predictions)
    total_labs = torch.cat(all_labels)
    _compute_metrics(output_df, total_preds, total_labs, 'overall_by_len', 'overall_by_mhc')

    output_df.to_csv(f'{output_filename}_breakdown.csv')
    _plot_predictions(total_preds, output_filename)

    auroc = output_df.loc['auroc_overall_by_mhc', 'overall_by_len']
    auprc = output_df.loc['auprc_overall_by_mhc', 'overall_by_len']
    precision_at_k = np.array([output_df.loc[f'precision@{k}_overall_by_mhc', 'overall_by_len'] for k in RETRIEVAL_KS])
    bedroc = np.array([output_df.loc[f'bedroc_{alpha}_overall_by_mhc', 'overall_by_len'] for alpha in ALPHAS])
    auac = np.array([output_df.loc[f'auac_{proportion}_overall_by_mhc', 'overall_by_len'] for proportion in PROPORTIONS])
    enrichment_factors = np.array([output_df.loc[f'enrichment_factor_{proportion}_overall_by_mhc', 'overall_by_len'] for proportion in PROPORTIONS])

    averaged_metrics = _compute_averged_metrics(output_df)

    with open(f'{output_filename}.txt', 'w') as output_file:
        output_file.write(TEST_RETRIEVAL_TEMPLATE.format(
            RETRIEVAL_KS,
            ALPHAS,
            PROPORTIONS,
            num_positive_peptides,
            num_total_peptides,
            auroc,
            averaged_metrics['auroc'],
            auprc,
            averaged_metrics['auprc'],
            np.array2string(precision_at_k, precision=6, separator=', '),
            np.array2string(averaged_metrics['precision_at_k'], precision=6, separator=', '),
            np.array2string(bedroc, precision=6, separator=', '),
            np.array2string(averaged_metrics['bedroc'], precision=6, separator=', '),
            np.array2string(enrichment_factors, precision=6, separator=', '),
            np.array2string(averaged_metrics['enrichment_factors'], precision=6, separator=', '),
            np.array2string(auac, precision=6, separator=', '),
            np.array2string(averaged_metrics['auac'], precision=6, separator=', '),
            time_taken,
            num_skipped
        ))


def test_sensitivity_el(
        predictions_diff: dict[str, torch.DoubleTensor], 
        labels_diff: dict[str, torch.DoubleTensor],
        output_filename: str,
) -> None:
    preds = []
    labs = []
    for mhc_name in predictions_diff.keys():
        preds.append(predictions_diff[mhc_name])
        labs.append(labels_diff[mhc_name])
    predictions_diff = torch.cat(preds)
    labels_diff = torch.cat(labs)
    assert predictions_diff.size(0) == labels_diff.size(0), f'{predictions_diff.size()} != {labels_diff.size()}'

    # Treat this as a binary classification problem, where positive differences are considered positive examples
    binary_predictions_diff = (predictions_diff > 0).int()
    binary_labels_diff = (labels_diff > 0).int()
    accuracy = metrics.compute_binary_accuracy(binary_predictions_diff, binary_labels_diff)
    precision = metrics.compute_binary_precision(binary_predictions_diff, binary_labels_diff)
    recall = metrics.compute_binary_recall(binary_predictions_diff, binary_labels_diff)
    f1_score = metrics.compute_binary_f1(binary_predictions_diff, binary_labels_diff)
    
    with open(f'{output_filename}.txt', 'w') as output_file:
        output_file.write(TEST_SENSITIVITY_EL_TEMPLATE.format(
            accuracy, 
            precision, 
            recall, 
            f1_score
        ))

    _plot_sensitivity_el(predictions_diff, binary_labels_diff, output_filename)


def test_sensitivity_ba(
        predictions_diff: dict[str, dict[str, torch.DoubleTensor]], 
        log50ks_diff: dict[str, dict[str, torch.DoubleTensor]],
        output_filename: str,
) -> None:
    preds = []
    logs = []
    for mhc_name in predictions_diff.keys():
        pred_dict = predictions_diff[mhc_name]
        log_dict = log50ks_diff[mhc_name]
        for length in pred_dict.keys():
            preds.append(pred_dict[length])
            logs.append(log_dict[length])
    predictions_diff = torch.cat(preds)
    log50ks_diff = torch.cat(logs)
    assert predictions_diff.size(0) == log50ks_diff.size(0), f'{predictions_diff.size()} != {log50ks_diff.size()}'

    # Treat this as a binary classification problem, where positive differences are considered positive examples
    binary_predictions_diff = (predictions_diff > 0).int()
    binary_log50ks_diff = (log50ks_diff > 0).int()
    accuracy = metrics.compute_binary_accuracy(binary_predictions_diff, binary_log50ks_diff)
    precision = metrics.compute_binary_precision(binary_predictions_diff, binary_log50ks_diff)
    recall = metrics.compute_binary_recall(binary_predictions_diff, binary_log50ks_diff)
    f1_score = metrics.compute_binary_f1(binary_predictions_diff, binary_log50ks_diff)

    # Treat this as a regression problem
    pearson_corrcoef = metrics.compute_pcc(predictions_diff, log50ks_diff)
    spearman_corrcoef = metrics.compute_srcc(predictions_diff, log50ks_diff)
    
    with open(f'{output_filename}.txt', 'w') as output_file:
        output_file.write(TEST_SENSITIVITY_BA_TEMPLATE.format(
            accuracy, 
            precision, 
            recall, 
            f1_score, 
            pearson_corrcoef, 
            spearman_corrcoef
        ))

    _plot_sensitivity_ba(predictions_diff, log50ks_diff, output_filename)


def test_regression(
        predictions: dict[str, dict[str, torch.DoubleTensor]], 
        log50ks: dict[str, dict[str, torch.DoubleTensor]], 
        output_filename: str,
) -> None:
    preds = []
    logs = []
    for mhc_name in predictions.keys():
        pred_dict = predictions[mhc_name]
        log_dict = log50ks[mhc_name]
        for length in pred_dict.keys():
            preds.append(pred_dict[length])
            logs.append(log_dict[length])
    
    predictions = torch.cat(preds)
    log50ks = torch.cat(logs)

    pearson_corrcoefs = metrics.compute_pcc(predictions, log50ks)
    spearman_corrcoefs = metrics.compute_srcc(predictions, log50ks)

    with open(f'{output_filename}.txt', 'a') as output_file:
        output_file.write('Test Regression:\npearson_corrcoef: {:.6f}\nspearman_corrcoef: {:.6f}\n'
            .format(
                pearson_corrcoefs, 
                spearman_corrcoefs
            )
        )
