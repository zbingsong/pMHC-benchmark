import torch
import sklearn.metrics
import scipy.stats

import math


def compute_ef(
        sorted_labels: torch.LongTensor, 
        proportion: float=0.05
) -> torch.DoubleTensor:
    '''
    EF_alpha = (NTB_alpha) / (NTB_total * alpha)
    where NTB_alpha is the number of true binders in the top alpha fraction of the predictions, and NTB_total is the total number of true binders in the dataset.

    Parameters:
    sorted_labels (torch.LongTensor): tensor of shape (n, ), containing the true labels sorted by predicted score
    proportion (float): proportion of positive samples to consider, in the range (0, 1] (alpha in the equation above)
    - warning: at small alpha, there may be significant precision issues
    
    Returns:
    torch.DoubleTensor: scalar tensor representing enrichment factor at alpha
    '''
    n = sorted_labels.sum().long()
    if n == 0:
        return torch.tensor(0, dtype=torch.double)
    N = sorted_labels.size(0)
    k = math.ceil(proportion * N)
    num_true_binders_alpha = sorted_labels[:k].sum().long()
    return num_true_binders_alpha / (n * proportion)


def compute_auac(
        sorted_labels: torch.LongTensor, 
        proportion: float=0.05
) -> torch.DoubleTensor:
    r'''
    AUAC = \int_0^1 F_a(x) dx, where F_a(x) is the fraction of true binders in the top x fraction of the predictions.
    Discrete: AUAC = \sum_{i=0}^N \frac{TP_i}{i}, where TP_i is the number of true positives in the top i predictions.

    Parameters:
    sorted_labels (torch.LongTensor): tensor of shape (n, ), containing the true labels sorted by predicted score
    proportion (float): proportion of positive samples to consider, in the range (0, 1] (x in the equation above)
    
    Returns:
    torch.DoubleTensor: scalar tensor representing area under the accumulation curve
    '''
    n = math.ceil(proportion * sorted_labels.size(0))
    if n == 0:
        return torch.tensor(0, dtype=torch.double)
    sorted_labels = sorted_labels[:n]
    true_positives = sorted_labels.cumsum(0)
    return (true_positives / torch.arange(1, n + 1).float()).mean()


def _compute_rie(
        sorted_labels: torch.LongTensor,
        alpha: torch.DoubleTensor=20.0
) -> torch.DoubleTensor:
    r'''
    Robust Initial Enhancement (RIE): https://doi.org/10.1021/ci0100144
    Implementation based on: https://doi.org/10.1021/ci600426e, equation 19
    RIE = \frac{\frac{1}{n} \sum_{i=1}^n \exp(-\alpha r_i / N)}{\frac{1}{N} \left( \frac{1 - \exp(-\alpha)}{\exp(\alpha / N) - 1} \right)}
    where n is the number of true positives, N is the total number of predictions (both positive and negative), and r_i is the rank of the i-th true positive in the predictions (1-indexed).

    Parameters:
    sorted_labels (torch.LongTensor): tensor of shape (N, ), containing the true labels sorted by predicted score
    alpha (torch.DoubleTensor): hyperparameter controlling the strength of the enhancement
    - greater than 0
    - usually take the value of 20
    - higher alpha values result in stronger enhancement, similar to smaller proportion in EF and AUAC; alpha is similar to 1 / proportion (https://doi.org/10.1186/1758-2946-5-26)
    - warning: at small alpha, there may be significant precision issues
    
    Returns:
    torch.DoubleTensor: scalar tensor representing robust initial enhancement
    '''
    n = sorted_labels.sum().long()
    if n == 0:
        return torch.tensor(0, dtype=torch.double)
    N = sorted_labels.size(0)
    R_alpha = (n / N).to(torch.double)
    positive_ranks = torch.nonzero(sorted_labels).squeeze(1) + 1
    rie = (torch.exp(-alpha * positive_ranks / N)).sum() / (R_alpha * (1 - torch.exp(-alpha)) / (torch.exp(alpha / N) - 1))
    return rie


def compute_bedroc(
        sorted_labels: torch.LongTensor,
        alpha: float=20.0
) -> torch.DoubleTensor:
    r'''
    Boltzmann-Enhanced Discrimination of Receiver Operating Characteristic (BEDROC): https://doi.org/10.1021/ci600426e
    BEDROC = RIE * \frac{R_{\alpha} \sinh (\alpha / 2)}{\cosh (\alpha / 2) - \cosh (\alpha / 2 - \alpha R_{\alpha})} + \frac{1}{1 - \exp(\alpha (1 - R_{\alpha})}
    where n is the number of true positives, N is the total number of predictions (both positive and negative), and r_i is the rank of the i-th true positive in the predictions (1-indexed).
    BEDROC scales RIE to [0, 1].

    Parameters:
    sorted_labels (torch.LongTensor): tensor of shape (N, ), containing the true labels sorted by predicted score
    alpha (float): hyperparameter controlling the strength of the enhancement
    - greater than 0
    - usually take the value of 20
    - higher alpha values result in stronger enhancement, similar to smaller proportion in EF and AUAC; alpha is similar to 1 / proportion (https://doi.org/10.1186/1758-2946-5-26)
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
    R_alpha = (n / N).to(torch.double)
    alpha = torch.tensor(alpha, dtype=torch.double)
    RIE = _compute_rie(sorted_labels, alpha)
    BEDROC = RIE * (R_alpha * torch.sinh(alpha / 2)) / (torch.cosh(alpha / 2) - torch.cosh(alpha / 2 - alpha * R_alpha)) + 1 / (1 - torch.exp(alpha * (1 - R_alpha)))
    return BEDROC


def compute_binary_auroc(
        predictions: torch.DoubleTensor, 
        labels: torch.LongTensor
) -> torch.DoubleTensor:
    label_sum = labels.sum()
    # if labels are all negative or all positive, return 1
    if label_sum == 0 or label_sum == labels.size(0):
        return torch.tensor(1, dtype=torch.double)
    # otherwise, compute AUROC normally
    else:
        predictions /= predictions.max()
        predictions = predictions.numpy()
        labels = labels.numpy()
        return torch.tensor(sklearn.metrics.roc_auc_score(labels, predictions), dtype=torch.double)
    

def compute_binary_auprc(
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
        predictions /= predictions.max()
        predictions = predictions.numpy()
        labels = labels.numpy()
        return torch.tensor(sklearn.metrics.average_precision_score(labels, predictions), dtype=torch.double)
    

def compute_binary_accuracy(
        binary_predictions: torch.LongTensor,
        binary_labels: torch.LongTensor
) -> torch.DoubleTensor:
    return torch.tensor(sklearn.metrics.accuracy_score(binary_labels, binary_predictions), dtype=torch.double)


def compute_binary_precision(
        binary_predictions: torch.LongTensor,
        binary_labels: torch.LongTensor
) -> torch.DoubleTensor:
    return torch.tensor(sklearn.metrics.precision_score(binary_labels, binary_predictions), dtype=torch.double)


def compute_binary_recall(
        binary_predictions: torch.LongTensor,
        binary_labels: torch.LongTensor
) -> torch.DoubleTensor:
    return torch.tensor(sklearn.metrics.recall_score(binary_labels, binary_predictions), dtype=torch.double)


def compute_binary_f1(
        binary_predictions: torch.LongTensor,
        binary_labels: torch.LongTensor
) -> torch.DoubleTensor:
    return torch.tensor(sklearn.metrics.f1_score(binary_labels, binary_predictions), dtype=torch.double)


def compute_pcc(
        predictions: torch.DoubleTensor,
        targets: torch.DoubleTensor
) -> torch.DoubleTensor:
    result = scipy.stats.pearsonr(predictions.numpy(), targets.numpy())
    return torch.tensor(result.statistic, dtype=torch.double)


def compute_srcc(
        predictions: torch.DoubleTensor,
        targets: torch.DoubleTensor
) -> torch.DoubleTensor:
    result = scipy.stats.spearmanr(predictions.numpy(), targets.numpy())
    return torch.tensor(result.statistic, dtype=torch.double)


def compute_retrieval_precision(
        sorted_labels: torch.LongTensor,
        top_k: int=100
) -> torch.DoubleTensor:
    label_sum = sorted_labels[:top_k].sum()
    return label_sum.to(torch.double) / top_k


# def compute_retrieval_auroc(
#         sorted_predictions: torch.DoubleTensor, 
#         sorted_labels: torch.LongTensor,
#         top_k: int=100
# ) -> torch.DoubleTensor:
#     label_sum = sorted_labels[:top_k].sum()
#     # if all labels are negative, return 0
#     if label_sum == 0:
#         return torch.tensor(0, dtype=torch.double)
#     # if all labels are positive, return 1
#     elif label_sum == min(top_k, sorted_labels.size(0)):
#         return torch.tensor(1, dtype=torch.double)
#     # otherwise, compute AUROC normally
#     else:
#         return torch.tensor(sklearn.metrics.roc_auc_score(sorted_labels[:top_k], sorted_predictions[:top_k]), dtype=torch.double)


# def compute_retrieval_auprc(
#         sorted_predictions: torch.DoubleTensor, 
#         sorted_labels: torch.LongTensor,
#         top_k: int=100
# ) -> torch.DoubleTensor:
#     label_sum = sorted_labels[:top_k].sum()
#     # if all labels are negative, return 0
#     if label_sum == 0:
#         return torch.tensor(0, dtype=torch.double)
#     # if all labels are positive, return 1
#     elif label_sum == min(top_k, sorted_labels.size(0)):
#         return torch.tensor(1, dtype=torch.double)
#     # otherwise, compute AUPRC normally
#     else:
#         return torch.tensor(sklearn.metrics.average_precision_score(sorted_labels[:top_k], sorted_predictions[:top_k]), dtype=torch.double)
