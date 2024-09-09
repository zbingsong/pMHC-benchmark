import torch
import torchmetrics.functional.classification
import torchmetrics.functional.retrieval
import numpy as np
import matplotlib.pyplot as plt

import io
import typing
import math


RETRIEVAL_KS = [1, 5, 10, 25, 50, 100]
ALPHAS = [0.5, 1, 2, 5, 10, 20]

TEST_RETRIEVAL_TEMPLATE = '''
Test Retrieval by MHC:
positives: {:.4f} +/- {:.4f}
totals: {:.4f} +/- {:.4f}
auroc_by_mhc: {:.6f} +/- {:.6f}
auprc_by_mhc: {:.6f} +/- {:.6f}
auac_by_mhc: {:.6f} +/- {:.6f}
bedrocs_by_mhc: {:.6f} +/- {:.6f}
precision@k:
{}
std:
{}
recall@k:
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
AUAC@k:
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


def _compute_ef(
        sorted_labels: torch.LongTensor, 
        alpha: float=100.0
) -> torch.DoubleTensor:
    '''
    EF_alpha = (NTB_alpha) / (NTB_total * alpha)
    where NTB_alpha is the number of true binders in the top alpha fraction of the predictions, and NTB_total is the total number of true binders in the dataset.

    Parameters:
    sorted_labels (torch.LongTensor): tensor of shape (n, ), containing the true labels sorted by predicted score
    alpha (float): proportion of positive samples to consider, in the range (0, 100]
    - warning: at small alpha, there may be significant precision issues
    
    Returns:
    torch.DoubleTensor: scalar tensor representing enrichment factor at alpha
    '''
    alpha /= 100
    n = sorted_labels.sum().long()
    if n == 0:
        return torch.tensor(0, dtype=torch.double)
    N = sorted_labels.size(0)
    k = math.ceil(alpha * N)
    num_true_binders_alpha = sorted_labels[:k].sum().long()
    return num_true_binders_alpha / (n * alpha)


def _compute_auac(sorted_labels: torch.LongTensor, alpha: float=100.0) -> torch.DoubleTensor:
    '''
    AUAC = \int_0^1 F_a(x) dx, where F_a(x) is the fraction of true binders in the top x fraction of the predictions.
    Discrete: AUAC = \sum_{i=0}^N TP_i / i, where TP_i is the number of true positives in the top i predictions.

    Parameters:
    sorted_labels (torch.LongTensor): tensor of shape (n, ), containing the true labels sorted by predicted score
    
    Returns:
    torch.DoubleTensor: scalar tensor representing area under the accumulation curve
    '''
    n = math.ceil(alpha * sorted_labels.size(0) / 100)
    if n == 0:
        return torch.tensor(0, dtype=torch.double)
    sorted_labels = sorted_labels[:n]
    true_positives = sorted_labels.cumsum(0)
    return (true_positives / torch.arange(1, n + 1).float()).mean()


def _compute_rie(
        sorted_labels: torch.LongTensor,
        alpha: torch.DoubleTensor
) -> torch.DoubleTensor:
    '''
    RIE = \frac{\frac{1}{n} \sum_{i=1}^n \exp(-\alpha r_i / N)}{\frac{1}{N} \left( \frac{1 - \exp(-\alpha)}{\exp(\alpha / N) - 1} \right)}
    where n is the number of true positives, N is the total number of predictions (both positive and negative), and r_i is the rank of the i-th true positive in the predictions (1-indexed).

    Parameters:
    sorted_labels (torch.LongTensor): tensor of shape (N, ), containing the true labels sorted by predicted score
    alpha (torch.DoubleTensor): porportion of early recognition we care about, in the range (0, 100]
    - warning: at small alpha, there may be significant precision issues
    
    Returns:
    torch.DoubleTensor: scalar tensor representing robust initial enhancement
    '''
    n = sorted_labels.sum().long()
    if n == 0:
        return torch.tensor(0, dtype=torch.double)
    N = sorted_labels.size(0)
    R_alpha = torch.tensor([n / N], dtype=torch.double)
    positive_ranks = torch.nonzero(sorted_labels).squeeze(1) + 1
    rie = (torch.exp(-alpha * positive_ranks / N)).sum() \
        / (R_alpha * (1 - torch.exp(-alpha)) / (torch.exp(alpha / N) - 1))
    return rie


def _compute_bedroc(
        sorted_labels: torch.LongTensor,
        alpha: float=1
) -> torch.DoubleTensor:
    '''
    BEDROC = RIE * \frac{}{}
    where n is the number of true positives, N is the total number of predictions (both positive and negative), and r_i is the rank of the i-th true positive in the predictions (1-indexed).

    Parameters:
    sorted_labels (torch.LongTensor): tensor of shape (N, ), containing the true labels sorted by predicted score
    alpha (float): porportion of early recognition we care about, in the range (0, 100]
    - warning: at small alpha, there may be significant precision issues
    
    Returns:
    torch.DoubleTensor: scalar tensor representing Boltzmann-Enhanced Discrimination of Receiver Operating Characteristic
    '''
    n = sorted_labels.sum().long()
    N = sorted_labels.size(0)
    if n == 0:
        return torch.tensor(0, dtype=torch.double)
    elif n == N:
        return torch.tensor(1, dtype=torch.double)
    R_alpha = torch.tensor([n / N], dtype=torch.double)
    alpha = torch.tensor([alpha], dtype=torch.double)
    RIE = _compute_rie(sorted_labels, alpha)
    BEDROC = RIE * (R_alpha * torch.sinh(alpha / 2)) / (torch.cosh(alpha / 2) - torch.cosh(alpha / 2 - alpha * R_alpha)) \
        + 1 / (1 - torch.exp(alpha * (1 - R_alpha)))
    return BEDROC


def _compute_binary_auroc(
        predictions: torch.DoubleTensor, 
        labels: torch.LongTensor
) -> torch.DoubleTensor:
    label_sum = labels.sum()
    # if labels are all negative or all positive, return 1
    if label_sum == 0 or label_sum == labels.size(0):
        return torch.tensor(1, dtype=torch.double)
    # otherwise, compute AUROC normally
    else:
        return torchmetrics.functional.classification.binary_auroc(predictions, labels)
    

def _compute_binary_auprc(
        predictions: torch.DoubleTensor, 
        labels: torch.LongTensor
) -> torch.DoubleTensor:
    label_sum = labels.sum()
    # if labels are all negative, return 0
    if label_sum == 0:
        return torch.tensor(0, dtype=torch.double)
    # if labels are all positive, return 1
    elif label_sum == labels.size(0):
        return torch.tensor(1, dtype=torch.double)
    # otherwise, compute AUPRC normally
    else:
        return torchmetrics.functional.classification.binary_average_precision(predictions, labels)


def _compute_retrieval_auroc(
        sorted_predictions: torch.DoubleTensor, 
        sorted_labels: torch.LongTensor,
        top_k: int=100
) -> torch.DoubleTensor:
    label_sum = sorted_labels[:top_k].sum()
    # if all labels are negative, return 0
    if label_sum == 0:
        return torch.tensor(0, dtype=torch.double)
    # if all labels are positive, return 1
    elif label_sum == min(top_k, sorted_labels.size(0)):
        return torch.tensor(1, dtype=torch.double)
    # otherwise, compute AUROC normally
    else:
        return torchmetrics.functional.classification.binary_auroc(sorted_predictions[:top_k], sorted_labels[:top_k])


def _compute_retrieval_auprc(
        sorted_predictions: torch.DoubleTensor, 
        sorted_labels: torch.LongTensor,
        top_k: int=100
) -> torch.DoubleTensor:
    label_sum = sorted_labels[:top_k].sum()
    # if all labels are negative, return 0
    if label_sum == 0:
        return torch.tensor(0, dtype=torch.double)
    # if all labels are positive, return 1
    elif label_sum == min(top_k, sorted_labels.size(0)):
        return torch.tensor(1, dtype=torch.double)
    # otherwise, compute AUPRC normally
    else:
        return torchmetrics.functional.classification.binary_average_precision(sorted_predictions[:top_k], sorted_labels[:top_k])


def test_retrieval(
        predictions: list[torch.DoubleTensor], 
        labels: list[torch.LongTensor], 
        output_file: io.TextIOWrapper
) -> None:
    n = len(predictions)
    num_positive_peptides = torch.zeros(n, dtype=torch.long)
    num_total_peptides = torch.zeros(n, dtype=torch.long)

    aurocs = torch.zeros(n, dtype=torch.double)
    auprcs = torch.zeros(n, dtype=torch.double)
    auacs = torch.zeros(n, dtype=torch.double)
    bedrocs = torch.zeros(n, dtype=torch.double)
    
    precision_at_ks = torch.zeros(n, len(RETRIEVAL_KS), dtype=torch.double)
    recall_at_ks = torch.zeros(n, len(RETRIEVAL_KS), dtype=torch.double)
    auroc_at_ks = torch.zeros(n, len(RETRIEVAL_KS), dtype=torch.double)
    auprc_at_ks = torch.zeros(n, len(RETRIEVAL_KS), dtype=torch.double)
    auac_at_ks = torch.zeros(n, len(RETRIEVAL_KS), dtype=torch.double)
    enrichment_factors = torch.zeros(n, len(ALPHAS), dtype=torch.double)
    bedroc_at_ks = torch.zeros(n, len(ALPHAS), dtype=torch.double)

    for i, (preds, labs) in enumerate(zip(predictions, labels)):
        sorted_preds, sorted_indices = torch.sort(preds, descending=True)
        sorted_labs = labs[sorted_indices]
        num_positive_peptides[i] = sorted_labs.sum()
        num_total_peptides[i] = sorted_labs.size(0)
        aurocs[i] = _compute_binary_auroc(sorted_preds, sorted_labs)
        auprcs[i] = _compute_binary_auprc(sorted_preds, sorted_labs)
        auacs[i] = _compute_auac(sorted_labs, 100.0)
        bedrocs[i] = _compute_bedroc(sorted_labs, 100.0)
        for j, k in enumerate(RETRIEVAL_KS):
            precision_at_ks[i, j] = torchmetrics.functional.retrieval.retrieval_precision(sorted_preds, sorted_labs, top_k=k)
            recall_at_ks[i, j] = torchmetrics.functional.retrieval.retrieval_recall(sorted_preds, sorted_labs, top_k=k)
            auroc_at_ks[i, j] = _compute_retrieval_auroc(sorted_preds, sorted_labs, top_k=k)
            auprc_at_ks[i, j] = _compute_retrieval_auprc(sorted_preds, sorted_labs, top_k=k)
        for j, alpha in enumerate(ALPHAS):
            auac_at_ks[i, j] = _compute_auac(sorted_labs, alpha)
            enrichment_factors[i, j] = _compute_ef(sorted_labs, alpha)
            bedroc_at_ks[i, j] = _compute_bedroc(sorted_labs, alpha)

    num_positive_peptides = num_positive_peptides.float()
    num_total_peptides = num_total_peptides.float()

    if output_file is not None:
        output_file.write(TEST_RETRIEVAL_TEMPLATE.format(
            num_positive_peptides.mean(),
            num_positive_peptides.std(),
            num_total_peptides.mean(),
            num_total_peptides.std(),
            aurocs.mean(), 
            aurocs.std(),
            auprcs.mean(),
            auprcs.std(),
            auacs.mean(),
            auacs.std(),
            bedrocs.mean(),
            bedrocs.std(),
            np.array2string(precision_at_ks.mean(dim=0).numpy(), precision=6, separator=','),
            np.array2string(precision_at_ks.std(dim=0).numpy(), precision=6, separator=','),
            np.array2string(recall_at_ks.mean(dim=0).numpy(), precision=6, separator=','),
            np.array2string(recall_at_ks.std(dim=0).numpy(), precision=6, separator=','),
            np.array2string(auroc_at_ks.mean(dim=0).numpy(), precision=6, separator=','),
            np.array2string(auroc_at_ks.std(dim=0).numpy(), precision=6, separator=','),
            np.array2string(auprc_at_ks.mean(dim=0).numpy(), precision=6, separator=','),
            np.array2string(auprc_at_ks.std(dim=0).numpy(), precision=6, separator=','),
            np.array2string(auac_at_ks.mean(dim=0).numpy(), precision=6, separator=','),
            np.array2string(auac_at_ks.std(dim=0).numpy(), precision=6, separator=','),
            np.array2string(enrichment_factors.mean(dim=0).numpy(), precision=6, separator=','),
            np.array2string(enrichment_factors.std(dim=0).numpy(), precision=6, separator=','),
            np.array2string(bedroc_at_ks.mean(dim=0).numpy(), precision=6, separator=','),
            np.array2string(bedroc_at_ks.std(dim=0).numpy(), precision=6, separator=',')
        ))


def test_sensitivity(
        predictions_diff: torch.DoubleTensor, 
        log50ks_diff: torch.DoubleTensor,
        plot_filename: str='sensitivity',
        output_file: typing.Optional[io.TextIOWrapper]=None,
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
        output_file: io.TextIOWrapper
) -> tuple[torch.DoubleTensor, torch.DoubleTensor]:
    n = len(predictions)
    pearson_corrcoefs = torch.zeros(n, dtype=torch.double)
    spearman_corrcoefs = torch.zeros(n, dtype=torch.double)

    for i, (preds, log50k) in enumerate(zip(predictions, log50ks)):
        pearson_corrcoefs[i] = torchmetrics.functional.pearson_corrcoef(preds, log50k)
        spearman_corrcoefs[i] = torchmetrics.functional.spearman_corrcoef(preds, log50k)

    if output_file is not None:
        output_file.write('\nTest Regression:\npearson_corrcoef: {:.6f}\nspearman_corrcoef: {:.6f}\n'
            .format(
                pearson_corrcoefs.mean(), 
                spearman_corrcoefs.mean()
            )
        )
    
    return pearson_corrcoefs.mean(), spearman_corrcoefs.mean()
