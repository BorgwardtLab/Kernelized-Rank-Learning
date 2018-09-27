import argparse
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from KRL import KRL
from misc import NDCG, get_median_trick_gamma


def clinical_KRL(k, Lambda, gamma, nfolds, fold, train_X, train_Y, test_X, drug_ids, drug_names, njobs, seed, output_prefix):

    if fold != -1:
        assert fold in range(nfolds)
        kfold = KFold(n_splits=nfolds, shuffle=True, random_state=seed)
        for i, (train_index, test_index) in enumerate(kfold.split(train_X)):
            if i == fold:
                pred_Y = KRL.KRL(train_X[train_index], train_Y[train_index], train_X[test_index], k=k, Lambda=Lambda, gamma=gamma, njobs=njobs)
                np.savez('{}fold{}_k{}_l{}_{}.npz'.format(output_prefix, fold, k, Lambda, gamma if gamma == 'linear' else 'g{}'.format(gamma)),
                     pred_Y=pred_Y, true_Y=train_Y[test_index],
                     nfolds=nfolds, fold=fold,
                     Lambda=Lambda, gamma=gamma, k=k,
                     drug_ids=drug_ids, drug_names=drug_names, seed=seed)
                ndcg = np.mean(NDCG(train_Y[test_index], pred_Y, k))
                print '{}-fold, fold {}, NDCG@{} = {} for lambda = {} and gamma = {}'.format(nfolds, fold, k, ndcg, Lambda, gamma)
    else:
        test_Y_pred = KRL.KRL(train_X, train_Y, test_X, k=k, Lambda=Lambda, gamma=gamma, njobs=njobs)
        np.savez('{}pred_k{}_l{}_{}.npz'.format(output_prefix, k, Lambda, gamma if gamma == 'linear' else 'g{}'.format(gamma)),
                 test_Y_pred=test_Y_pred,
                 nfolds=nfolds, fold=fold,
                 Lambda=Lambda, gamma=gamma, k=k,
                 drug_ids=drug_ids, drug_names=drug_names, seed=seed)
        print 'Test prediction saved: l = {} lambda = {} and gamma = {}'.format(k, Lambda, gamma)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('data_fn', type=str)
    parser.add_argument('-o', '--prefix', required=True)
    parser.add_argument('-l', '--Lambda', type=float, required=True)
    parser.add_argument('-g', '--gamma', type=str, default='median-trick', help='\'linear\' for linear kernel; float for RBF kernel; \'median-trick\' for RBF kernel with the median trick gamma')
    parser.add_argument('-f', '--fold', type=int, default=-1, help='Set --fold to -1 for prediction on the test set')
    parser.add_argument('-k', '--ndcg_k', type=int, default=10)
    parser.add_argument('-j', '--njobs', type=int, default=1)
    parser.add_argument('-n', '--nfolds', type=int, default=3)
    parser.add_argument('-s', '--seed', type=int, default=10000)
    parser.add_argument('--few_drugs', action='store_true')
    args = parser.parse_args()

    if args.gamma != 'linear' and args.gamma != 'median-trick':
        try:
            args.gamma = float(args.gamma)
        except:
            parser.error('--gamma accepts the following values: \'linear\' for linear kernel; float for RBF kernel; \'median-trick\' for RBF kernel with the median trick gamma')

    print 'Loading the data {}'.format(args.data_fn)
    data = np.load(args.data_fn)
    drug_ids = data['drug_ids']
    drug_names = data['drug_names']
    train_X = data['train_X']
    train_Y = data['train_Y']
    test_X = data['test_X']

    if args.few_drugs:
        from misc import DOCETAXEL, BORTEZOMIB
        from misc import LAPATINIB, VELIPARIB, OLAPARIB1, OLAPARIB2, TALAZOPARIB, RUCAPARIB, RUXOLITINIB
        from misc import DRUG_NAMES, DRUG_IDS
        drug_names_lower = np.array([x.lower() for x in drug_names])
        drug_indexes, drug_names, drug_ids = [], [], []
        for i, drug in enumerate(DRUG_NAMES):
            drug_indexes.append(np.nonzero(drug_names_lower == DRUG_NAMES[drug].lower())[0][0])
            drug_names.append(DRUG_NAMES[drug])
            drug_ids.append(DRUG_IDS[drug])
        train_Y = train_Y[:,drug_indexes]

    if args.gamma == 'median-trick':
        args.gamma = get_median_trick_gamma(train_X)
        print 'Median-trick gamma:', args.gamma

    clinical_KRL(k=args.ndcg_k, Lambda=args.Lambda, gamma=args.gamma, nfolds=args.nfolds, fold=args.fold,
                 train_X=train_X, train_Y=train_Y, test_X=test_X,
                 drug_ids=drug_ids, drug_names=drug_names,
                 njobs=args.njobs, seed=args.seed, output_prefix=args.prefix)


if __name__ == '__main__':
    main()
