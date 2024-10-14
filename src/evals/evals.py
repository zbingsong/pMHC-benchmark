import torch
import torchmetrics.functional.classification
import torchmetrics.functional.retrieval
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from . import compute_binary_auroc, compute_binary_auprc, compute_bedroc, compute_retrieval_auroc, compute_retrieval_auprc, compute_ef


RETRIEVAL_KS = [1, 5, 10, 25, 50, 100]
ALPHAS = [0.5, 1, 2, 5, 10, 20]
PEPTIDE_LENGTHS = [8, 9, 10, 11, 12, 13, 14]

TEST_RETRIEVAL_TEMPLATE = '''
Test Retrieval by MHC:
retrieval k: {}
alphas: {}
positives: {:.4f} +/- {:.4f}
totals: {:.4f} +/- {:.4f}
Averaged by MHC:
auroc: {:.6f} +/- {:.6f}
auprc: {:.6f} +/- {:.6f}
bedroc: {:.6f} +/- {:.6f}
Retrieval by MHC:
precision@k:
{}
std:
{}
auroc@k:
{}
std:
{}
auprc@k:
{}
std:
{}
enrichment_factors:
{}
std:
{}
BEDROC:
{}
std:
{}
time: {} nanoseconds

'''

TEST_SENSITIVITY_TEMPLATE = '''
Test Sensitivity:
accuracy: {:.4f}
precision: {:.4f}
recall: {:.4f}
f1_score: {:.4f}
pearson_corrcoef: {:.6f}
spearman_corrcoef: {:.6f}

'''


def _plot_similarities(predictions: torch.FloatTensor, name: str) -> None:
    '''
    Parameters:
    predictions (torch.FloatTensor): tensor of shape (n, ), containing the cosine similarities between MHC and peptide pairs
    '''
    # Plot histogram of test predictions
    plt.hist(predictions.float(), bins=200, range=(-1, 1))
    plt.xlabel('EL Score')
    plt.ylabel('Frequency')
    plt.savefig(name)
    plt.close()


def test_retrieval(
        predictions: dict[str, dict[str, torch.DoubleTensor]], 
        labels: dict[str, dict[str, torch.DoubleTensor]], 
        time_taken: int,
        output_filename: str,
) -> None:
    n = len(predictions)
    num_positive_peptides = torch.zeros(n, dtype=torch.long)
    num_total_peptides = torch.zeros(n, dtype=torch.long)

    aurocs = torch.zeros(n, dtype=torch.double)
    auprcs = torch.zeros(n, dtype=torch.double)
    bedrocs = torch.zeros(n, dtype=torch.double)
    
    precision_at_ks = torch.zeros(n, len(RETRIEVAL_KS), dtype=torch.double)
    auroc_at_ks = torch.zeros(n, len(RETRIEVAL_KS), dtype=torch.double)
    auprc_at_ks = torch.zeros(n, len(RETRIEVAL_KS), dtype=torch.double)
    enrichment_factors = torch.zeros(n, len(ALPHAS), dtype=torch.double)
    bedroc_at_ks = torch.zeros(n, len(ALPHAS), dtype=torch.double)

    auroc_df = pd.DataFrame(columns=[predictions.keys()], index=PEPTIDE_LENGTHS + ['overall_len'], dtype=float)
    auroc_df.insert(len(predictions.keys()), 'overall_mhc', float('nan'))
    auprc_df = pd.DataFrame(columns=[predictions.keys()], index=PEPTIDE_LENGTHS + ['overall_len'], dtype=float)
    auprc_df.insert(len(predictions.keys()), 'overall_mhc', float('nan'))

    predictions_by_length = {l: [] for l in PEPTIDE_LENGTHS}
    labels_by_length = {l: [] for l in PEPTIDE_LENGTHS}

    predictions_all = torch.cat(predictions)
    _plot_similarities(predictions_all, f'{output_filename}_predictions.png')

    for i, mhc_name in enumerate(predictions.keys()):
        pred_dict = predictions[mhc_name]
        lab_dict = labels[mhc_name]

        for seq_length in pred_dict.keys():
            preds = pred_dict[seq_length]
            labs = lab_dict[seq_length]
            sorted_preds, sorted_indices = torch.sort(preds, descending=True)
            sorted_labs = labs[sorted_indices]
            auroc = compute_binary_auroc(sorted_preds, sorted_labs)
            auprc = compute_binary_auprc(sorted_preds, sorted_labs)
            auroc_df.loc[seq_length, mhc_name] = auroc.item()
            auprc_df.loc[seq_length, mhc_name] = auprc.item()
            predictions_by_length[seq_length].append(preds)
            labels_by_length[seq_length].append(labs)
        
        preds = torch.cat([pred_dict[seq_length] for seq_length in PEPTIDE_LENGTHS])
        labs = torch.cat([lab_dict[seq_length] for seq_length in PEPTIDE_LENGTHS])
        sorted_preds, sorted_indices = torch.sort(preds, descending=True)
        sorted_labs = labs[sorted_indices]

        aurocs[i] = compute_binary_auroc(sorted_preds, sorted_labs)
        auprcs[i] = compute_binary_auprc(sorted_preds, sorted_labs)
        bedrocs[i] = compute_bedroc(sorted_labs, 100.0)
        num_positive_peptides[i] = sorted_labs.sum()
        num_total_peptides[i] = sorted_labs.size(0)
        auroc_df.loc['overall_len', mhc_name] = aurocs[i].item()
        auprc_df.loc['overall_len', mhc_name] = auprcs[i].item()

        for j, k in enumerate(RETRIEVAL_KS):
            precision_at_ks[i, j] = torchmetrics.functional.retrieval.retrieval_precision(sorted_preds, sorted_labs, top_k=k)
            auroc_at_ks[i, j] = compute_retrieval_auroc(sorted_preds, sorted_labs, top_k=k)
            auprc_at_ks[i, j] = compute_retrieval_auprc(sorted_preds, sorted_labs, top_k=k)
        for j, alpha in enumerate(ALPHAS):
            enrichment_factors[i, j] = compute_ef(sorted_labs, alpha=alpha)
            bedroc_at_ks[i, j] = compute_bedroc(sorted_labs, alpha=alpha)

    num_positive_peptides = num_positive_peptides.float()
    num_total_peptides = num_total_peptides.float()

    for length in PEPTIDE_LENGTHS:
        if len(predictions_by_length[length]) > 0:
            preds = torch.cat(predictions_by_length[length])
            labs = torch.cat(labels_by_length[length])
            predictions_by_length[length] = preds
            labels_by_length[length] = labs
            sorted_preds, sorted_indices = torch.sort(preds, descending=True)
            sorted_labs = labs[sorted_indices]
            auroc_df.loc[length, 'overall_mhc'] = compute_binary_auroc(sorted_preds, sorted_labs).item()
            auprc_df.loc[length, 'overall_mhc'] = compute_binary_auprc(sorted_preds, sorted_labs).item()
        else:
            predictions_by_length[length] = torch.tensor([])
            labels_by_length[length] = torch.tensor([])
    
    total_preds = torch.cat([predictions_by_length[length] for length in PEPTIDE_LENGTHS])
    total_labs = torch.cat([labels_by_length[length] for length in PEPTIDE_LENGTHS])
    sorted_preds, sorted_indices = torch.sort(total_preds, descending=True)
    sorted_labs = total_labs[sorted_indices]
    auroc_df.loc['overall_len', 'overall_mhc'] = compute_binary_auroc(sorted_preds, sorted_labs).item()
    auprc_df.loc['overall_len', 'overall_mhc'] = compute_binary_auprc(sorted_preds, sorted_labs).item()

    with open(f'{output_filename}.txt', 'w') as output_file:
        output_file.write(TEST_RETRIEVAL_TEMPLATE.format(
            RETRIEVAL_KS,
            ALPHAS,
            num_positive_peptides.mean(),
            num_positive_peptides.std(),
            num_total_peptides.mean(),
            num_total_peptides.std(),
            aurocs.mean(), 
            aurocs.std(),
            auprcs.mean(),
            auprcs.std(),
            bedrocs.mean(),
            bedrocs.std(),
            np.array2string(precision_at_ks.mean(dim=0).numpy(), precision=6, separator=','),
            np.array2string(precision_at_ks.std(dim=0).numpy(), precision=6, separator=','),
            np.array2string(auroc_at_ks.mean(dim=0).numpy(), precision=6, separator=','),
            np.array2string(auroc_at_ks.std(dim=0).numpy(), precision=6, separator=','),
            np.array2string(auprc_at_ks.mean(dim=0).numpy(), precision=6, separator=','),
            np.array2string(auprc_at_ks.std(dim=0).numpy(), precision=6, separator=','),
            np.array2string(enrichment_factors.mean(dim=0).numpy(), precision=6, separator=','),
            np.array2string(enrichment_factors.std(dim=0).numpy(), precision=6, separator=','),
            np.array2string(bedroc_at_ks.mean(dim=0).numpy(), precision=6, separator=','),
            np.array2string(bedroc_at_ks.std(dim=0).numpy(), precision=6, separator=','),
            time_taken
        ))
    
    auroc_df.to_csv(f'{output_filename}_auroc.csv')
    auprc_df.to_csv(f'{output_filename}_auprc.csv')


def test_sensitivity(
        predictions_diff: torch.DoubleTensor, 
        log50ks_diff: torch.DoubleTensor,
        plot_filename: str,
        output_filename: str,
) -> None:
    # Treat this as a binary classification problem, where positive differences are considered positive examples
    binary_predictions_diff = (predictions_diff > 0).int()
    binary_log50ks_diff = (log50ks_diff > 0).int()
    accuracy = torchmetrics.functional.classification.binary_accuracy(binary_predictions_diff, binary_log50ks_diff)
    precision = torchmetrics.functional.classification.binary_precision(binary_predictions_diff, binary_log50ks_diff)
    recall = torchmetrics.functional.classification.binary_recall(binary_predictions_diff, binary_log50ks_diff)
    f1_score = torchmetrics.functional.classification.binary_f1_score(binary_predictions_diff, binary_log50ks_diff)

    # Treat this as a regression problem
    pearson_corrcoef = torchmetrics.functional.pearson_corrcoef(predictions_diff, log50ks_diff)
    spearman_corrcoef = torchmetrics.functional.spearman_corrcoef(predictions_diff, log50ks_diff)
    
    with open(f'{output_filename}.txt', 'w') as output_file:
        output_file.write(TEST_SENSITIVITY_TEMPLATE.format(
            accuracy, 
            precision, 
            recall, 
            f1_score, 
            pearson_corrcoef, 
            spearman_corrcoef
        ))

    plt.plot(predictions_diff.numpy(), log50ks_diff.numpy(), '.')
    plt.xlabel('Predicted difference in BA')
    plt.ylabel('Measured difference in BA log50k')
    plt.savefig(f'{plot_filename}.png')
    plt.close()


def test_regression(
        predictions: list[torch.DoubleTensor], 
        log50ks: list[torch.DoubleTensor], 
        output_filename: str,
) -> None:
    n = len(predictions)
    pearson_corrcoefs = torch.zeros(n, dtype=torch.double)
    spearman_corrcoefs = torch.zeros(n, dtype=torch.double)

    for i, (preds, log50k) in enumerate(zip(predictions, log50ks)):
        pearson_corrcoefs[i] = torchmetrics.functional.pearson_corrcoef(preds, log50k)
        spearman_corrcoefs[i] = torchmetrics.functional.spearman_corrcoef(preds, log50k)

    with open(f'{output_filename}.txt', 'a') as output_file:
        output_file.write('\nTest Regression:\npearson_corrcoef: {:.6f}\nspearman_corrcoef: {:.6f}\n'
            .format(
                pearson_corrcoefs.mean(), 
                spearman_corrcoefs.mean()
            )
        )
