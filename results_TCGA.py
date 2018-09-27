import argparse
import numpy as np
import pandas as pd
import scipy
import scipy.io
import itertools
import os
import sys
import copy
import collections
from misc import LAPATINIB, VELIPARIB, OLAPARIB1, OLAPARIB2, TALAZOPARIB, RUCAPARIB, RUXOLITINIB
from misc import DRUG_NAMES, DRUG_IDS
from misc import PARP_INHIBS, TNBC_DRUGS
from misc import intersect_index, NDCG


ER, PR, HER2, CHR17, BRCA_GERM, JAK2_RPPA = 'ER', 'PR', 'HER2', 'CHR17', 'BRCA_germline', 'JAK2_RPPA'
NEGATIVE, POSITIVE, INDETERMINATE, EQUIVOCAL = 'negative', 'positive', 'indeterminate', 'equivocal'
PARPi = 'PARPi'

def get_acc(subtype, wins_array):
    return np.sum(wins_array[subtype]) / float(np.sum(subtype))


def tcga_results(brca_df, pred_prefix, test_ids, lambdas, gamma, k, nfolds, verbose=True):
    brca_samples = np.array(brca_df.index)
    brca_samples = np.array(brca_df.index)

    brca_samples = np.array(brca_df.index)
    merged = intersect_index(test_ids, brca_samples)
    test_keep_index = np.array(merged['index1'].values, dtype=np.int)
    brca_keep_index = np.array(merged['index2'].values, dtype=np.int)

    her2_pos = np.array(brca_df.loc[:, HER2])[brca_keep_index] == POSITIVE
    her2_neg = np.array(brca_df.loc[:, HER2])[brca_keep_index] == NEGATIVE
    chr17_pos = np.array([~np.isnan(x) and x >= 2 for x in np.array(brca_df.loc[:, CHR17])[brca_keep_index]])
    er_neg = np.array(brca_df.loc[:, ER])[brca_keep_index] == NEGATIVE
    pr_neg = np.array(brca_df.loc[:, PR])[brca_keep_index] == NEGATIVE
    mbrca_pos = np.array(brca_df.loc[:, BRCA_GERM])[brca_keep_index] == True
    jak2_pos = np.array([~np.isnan(x) and x >= 0 for x in np.array(brca_df.loc[:, JAK2_RPPA])[brca_keep_index]])
    tnbc = her2_neg & ~chr17_pos & er_neg & pr_neg
    tnbc_mbrca_pos = tnbc & mbrca_pos
    tnbc_jak2_pos = tnbc & jak2_pos

    best_ndcg, drug_ranks = None, None
    for Lambda in lambdas:
        if len(lambdas) > 1:
            ndcgs = []
            for i in range(nfolds):
                fold = np.load('{}fold{}_k{}_l{}_g{}.npz'.format(pred_prefix, i, k, Lambda, gamma))
                ndcgs.extend(NDCG(fold['true_Y'], fold['pred_Y'], k))
            ndcgs = np.array(ndcgs)
            ndcg = np.mean(ndcgs[~np.isnan(ndcgs)])
            if verbose:
                print 'CV, NDCG@{} = {:.4f}, lambda = {}, gamma = {}'.format(k, ndcg, Lambda, gamma)
        else:
            if verbose:
                print 'Skipping cross-validation because there is only one lambda to choose from.'
            ndcg = None

        if best_ndcg is None or ndcg > best_ndcg:
            best_ndcg = ndcg
            best_lambda = Lambda

    pred_fn = '{}pred_k{}_l{}_g{}.npz'.format(pred_prefix, k, best_lambda, gamma)
    pred = np.load(pred_fn)
    test_Y_pred = pred['test_Y_pred']
    test_Y_pred = test_Y_pred[test_keep_index]

    drug_indexes = {}
    if 'drug_names' in pred:
        drug_names = np.array([x.lower() for x in pred['drug_names']])
        for drug in [LAPATINIB] + TNBC_DRUGS:
            drug_indexes[drug] = np.nonzero(drug_names == DRUG_NAMES[drug].lower())[0][0]
    else:
        # This is just for backwards compability
        # The original code assumed that lapatinib is at position 0
        # And the TNBC drugs follow in a pre-specified order
        for drug_index, drug in enumerate([LAPATINIB] + TNBC_DRUGS):
            drug_indexes[drug] = drug_index

    lap_wins = {PARPi: np.ones((test_Y_pred.shape[0],), dtype=bool)}
    acc_her2_pos, acc_tnbc_mbrca_pos, acc_tnbc_jak2_pos = {}, {}, {}
    lapatinib_index = drug_indexes[LAPATINIB]
    for drug in TNBC_DRUGS:
        drug_index = drug_indexes[drug]
        # data consistency check
        assert not np.any(test_Y_pred[:, lapatinib_index] == test_Y_pred[:, drug_index]) or np.sum(
            test_Y_pred[:lapatinib_index]) == 0
        lap_wins[drug] = test_Y_pred[:, lapatinib_index] > test_Y_pred[:, drug_index]
        if drug in PARP_INHIBS:
            lap_wins[PARPi] = lap_wins[PARPi] & lap_wins[drug]
        acc_her2_pos[drug] = get_acc(her2_pos, lap_wins[drug])
        acc_tnbc_mbrca_pos[drug] = 1 - get_acc(tnbc_mbrca_pos, lap_wins[drug])
        acc_tnbc_jak2_pos[drug] = 1 - get_acc(tnbc_jak2_pos, lap_wins[drug])
    acc_her2_pos[PARPi] = get_acc(her2_pos, lap_wins[PARPi])
    acc_tnbc_mbrca_pos[PARPi] = 1 - get_acc(tnbc_mbrca_pos, lap_wins[PARPi])

    if verbose and 'drug_names' not in pred:
        print '\n!!!! RUNNING A LEGACY VERSION !!!!!'
        print '! Assuming that lapatinib is at position 0 and TNBC drugs follow in a pre-specified order in pred[\'test_pred_Y\'] !'
        print '!!!! RUNNING A LEGACY VERSION !!!!!\n'

    assert test_Y_pred.shape[0] == test_ids[test_keep_index].shape[0]
    assert test_Y_pred.shape[0] == her2_pos.shape[0]
    np.savez('{}results'.format(pred_prefix), test_Y_pred=test_Y_pred, test_ids=test_ids[test_keep_index],
             her2_pos=her2_pos, tnbc_mbrca_pos=tnbc_mbrca_pos, tnbc_jak2_pos=tnbc_jak2_pos, **lap_wins)

    if verbose and len(lambdas) > 1:
        print 'BEST CV, NDCG@{} = {:.4f}, lambda = {}, gamma = {}'.format(k, best_ndcg, best_lambda, gamma)



    print 'HER2+:', np.sum(her2_pos)
    print 'TNBC mBRCA:', np.sum(tnbc_mbrca_pos)
    print 'TNBC JAK2+:', np.sum(tnbc_jak2_pos)

    FULL_LINE = '------------------------------------------------------------------'
    print '\n', FULL_LINE
    print '| {:<23} | {:<10} | {:<10} | {:<10} |'.format('Recommendation', 'HER2+', 'TNBC mBRCA', 'TNBC JAK2+')
    print FULL_LINE
    for drug in [PARPi] + PARP_INHIBS + [RUXOLITINIB]:
        if drug == RUXOLITINIB:
            print '| lapatinib > {:<11} | {:<10} | {:<10} | {:<10} |'.format(drug, '{:.2f}'.format(acc_her2_pos[drug]), '---',
                                                             '{:.2f}'.format(1 - acc_tnbc_jak2_pos[drug]))
        else:
            print '| lapatinib > {:<11} | {:<10} | {:<10} | {:<10} |'.format(drug, '{:.2f}'.format(acc_her2_pos[drug]),
                                                                 '{:.2f}'.format(1 - acc_tnbc_mbrca_pos[drug]), '---')
    print FULL_LINE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_fn', type=str)
    parser.add_argument('-p', '--prefix', type=str, required=True)
    parser.add_argument('-l', '--lambdas', nargs='+', type=float, required=True)
    parser.add_argument('-g', '--gamma', type=float)
    parser.add_argument('-k', '--ndcg_k', type=int, default=10)
    parser.add_argument('-n', '--nfolds', type=int, default=3)
    parser.add_argument('-b', '--brca_df', type=str, default='data/TCGA_BRCA_clinical.csv.gz')
    args = parser.parse_args()

    data = np.load(args.data_fn)
    test_ids = data['test_ids']
    assert data['test_X'].shape[0] == test_ids.shape[0]

    brca_df = pd.read_csv(args.brca_df, index_col=0)
    assert len(brca_df.index) == len(set(brca_df.index))


    tcga_results(brca_df=brca_df, pred_prefix=args.prefix, test_ids=test_ids,
                 lambdas=args.lambdas, gamma=args.gamma, k=args.ndcg_k, nfolds=args.nfolds)



if __name__ == '__main__':
    main()
