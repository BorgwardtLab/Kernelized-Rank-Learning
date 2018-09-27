import argparse
import numpy as np
import os
from misc import DOCETAXEL, BORTEZOMIB
from sklearn.metrics import roc_auc_score
from misc import NDCG, get_median_trick_gamma
from scipy.stats import mannwhitneyu


def clinical_results(drug_name, pred_prefix, test_y, binary_test_y, lambdas, gamma, k, nfolds):

    best_ndcg = None
    for Lambda in lambdas:
        pred_fn = '{}pred_k{}_l{}_g{}.npz'.format(pred_prefix, k, Lambda, gamma)
        pred = np.load(pred_fn)
        if len(lambdas) > 1:
            ndcgs = []
            for i in range(nfolds):
                fold_fn = '{}fold{}_k{}_l{}_g{}.npz'.format(pred_prefix, i, k, Lambda, gamma)
                fold = np.load(fold_fn)
                ndcgs.extend(NDCG(fold['true_Y'], fold['pred_Y'], k))
            ndcg = np.mean(ndcgs)
            print 'CV, NDCG@{} = {:.4f}, lambda = {}, gamma = {}'.format(k, ndcg, Lambda, gamma)
        else:
            print 'Skipping cross-validation because there is only one lambda to choose from.'
            ndcg = None
        if best_ndcg is None or ndcg > best_ndcg:
            best_ndcg = ndcg
            best_lambda = Lambda
            # Calculate ranks from predictions
            if 'drug_names' in pred:
                drug_names = np.array([x.lower() for x in pred['drug_names']])
                drug_index = np.nonzero(drug_names == drug_name.lower())[0][0]
            else:
                # This is just for backwards compability
                # The original code assumed that the drug of interest is at position 0
                drug_index = 0
            drug_ranks = []
            for pred_rank_scores in pred['test_Y_pred']:
                sorted_idx = np.argsort(pred_rank_scores)[::-1]
                rank = np.nonzero(sorted_idx == drug_index)[0][0] + 1
                drug_ranks.append(rank)
            drug_ranks = np.array(drug_ranks)

    if 'drug_names' not in pred:
        print '\n!!!! RUNNING A LEGACY VERSION !!!!!'
        print '! Assuming that the drug of interest is at position 0 in pred[\'test_pred_Y\'] !'
        print '!!!! RUNNING A LEGACY VERSION !!!!!\n'

    np.savez('{}results'.format(pred_prefix), drug_ranks=drug_ranks, test_y=test_y, binary_test_y=binary_test_y)

    if len(lambdas) > 1:
        print 'BEST CV, NDCG@{} = {:.4f}, lambda = {}, gamma = {}'.format(k, best_ndcg, best_lambda, gamma)
    print '\n', pred_prefix
    print 'AUROC = {}'.format(roc_auc_score(binary_test_y, -1 * drug_ranks))
    print 'Wilcoxon rank sum test p-val = {}'.format(mannwhitneyu(drug_ranks[binary_test_y == 0], drug_ranks[binary_test_y == 1])[1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_fn', type=str)
    parser.add_argument('-d', '--drug', type=str, required=True)
    parser.add_argument('-p', '--prefix', type=str, required=True)
    parser.add_argument('-l', '--lambdas', nargs='+', type=float, required=True)
    parser.add_argument('-g', '--gamma', type=float)
    parser.add_argument('-k', '--ndcg_k', type=int, default=10)
    parser.add_argument('-n', '--nfolds', type=int, default=3)
    args = parser.parse_args()

    data = np.load(args.data_fn)
    test_y = data['test_y']
    binary_test_y = data['binary_test_y']

    clinical_results(drug_name=args.drug, pred_prefix=args.prefix, test_y=test_y, binary_test_y=binary_test_y,
                     lambdas=args.lambdas, gamma=args.gamma, k=args.ndcg_k, nfolds=args.nfolds)



if __name__ == '__main__':
    main()
