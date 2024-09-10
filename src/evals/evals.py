import torch
import torchmetrics.functional.classification
import torchmetrics.functional.retrieval
import numpy as np
import matplotlib.pyplot as plt
import io
import typing

from . import compute_binary_auroc, compute_binary_auprc, compute_auac, compute_bedroc, compute_retrieval_auroc, compute_retrieval_auprc, compute_ef


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
Time taken: {} nanoseconds

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


def test_retrieval(
        predictions: list[torch.DoubleTensor], 
        labels: list[torch.LongTensor], 
        time_taken: int,
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
        aurocs[i] = compute_binary_auroc(sorted_preds, sorted_labs)
        auprcs[i] = compute_binary_auprc(sorted_preds, sorted_labs)
        auacs[i] = compute_auac(sorted_labs, 100.0)
        bedrocs[i] = compute_bedroc(sorted_labs, 100.0)
        for j, k in enumerate(RETRIEVAL_KS):
            precision_at_ks[i, j] = torchmetrics.functional.retrieval.retrieval_precision(sorted_preds, sorted_labs, top_k=k)
            recall_at_ks[i, j] = torchmetrics.functional.retrieval.retrieval_recall(sorted_preds, sorted_labs, top_k=k)
            auroc_at_ks[i, j] = compute_retrieval_auroc(sorted_preds, sorted_labs, top_k=k)
            auprc_at_ks[i, j] = compute_retrieval_auprc(sorted_preds, sorted_labs, top_k=k)
        for j, alpha in enumerate(ALPHAS):
            auac_at_ks[i, j] = compute_auac(sorted_labs, alpha)
            enrichment_factors[i, j] = compute_ef(sorted_labs, alpha)
            bedroc_at_ks[i, j] = compute_bedroc(sorted_labs, alpha)

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
            np.array2string(bedroc_at_ks.std(dim=0).numpy(), precision=6, separator=','),
            time_taken
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
