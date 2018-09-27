#!/usr/bin/env python

import argparse
import sys
import itertools
import numpy as np
import scipy.io as sio
import traceback
import yaml
import os

from misc import NDCG, Precision, Rank, Percentile, non_nan_mean
from misc import FULL, SAMPLE, KEEPK
from misc import GEX, WES, CNV, METH
from misc import KRL, LKRL, KBMTL, KRR, RF, EN
from misc import BASELINES
from misc import PARAM_STR, RANK_STR, PERCENTILE_STR, NDCG_STR, PRECISION_STR, DELIM

def nsort(l, desc=True):
    '''
    sorts a list of strings numerically 
    '''
    return sorted(l, key=lambda x: float(x), reverse=desc)

def float2str(number):
    return '{}'.format(number)


def safe_load(filename, mat=False):
    try:
        result = np.load(filename) if not mat else sio.loadmat(filename)
    except Exception:
        result = None
        print >> sys.stderr, 'ERROR:', filename
        traceback.print_exc()
    return result


def is_edge(param, param_range):
    return nsort(param_range).index(param) in [0, len(param_range) - 1]


def param_to_str(param_tuples, delim=' '):
    return delim.join(map(lambda x: '{}{}{}'.format(x[0], delim, 'E' if is_edge(x[0], x[1]) else '.'), param_tuples))


def params_info(method, fold, param_tuples, ratio, seed, cv_metric, cv_k, delim=' '):
    if cv_metric is Precision:
        cv_metric_str = PRECISION_STR
    elif cv_metric is NDCG:
        cv_metric_str = NDCG_STR
    else:
        cv_metric_str = 'UNKNOWN'

    return delim.join([PARAM_STR, method, 'fold{}'.format(fold), '{:g}'.format(ratio), str(seed), cv_metric_str, str(cv_k), param_to_str(param_tuples, delim)])


def result_info(metric, results, method, seed, ratio, k_eval, debug=False, delim=' '):
    return delim.join([metric, method, str(seed), '{:g}'.format(ratio), str(k_eval),
                      float2str(non_nan_mean(results)),
                      delim.join(map(float2str, results)) if not debug else ''])


def ndcg_info(results, method, seed, ratio, k_eval, debug=False, delim=' '):
    return result_info(NDCG_STR, results, method, seed, ratio, k_eval, debug, delim)


def precision_info(results, method, seed, ratio, k_eval, debug=False, delim=' '):
    return result_info(PRECISION_STR, results, method, seed, ratio, k_eval, debug, delim)


def percentile_info(percentile, i, method, seed, ratio, k_perc, delim=' '):
    return delim.join([PERCENTILE_STR, method, str(seed), '{:g}'.format(ratio), str(k_perc),
                       str(i + 1), float2str(percentile)])


def rank_info(relative_ranks, method, seed, ratio, debug=False, delim=' '):
    return delim.join([RANK_STR, method, str(seed), '{:g}'.format(ratio),
                       float2str(non_nan_mean(relative_ranks)),
                       delim.join(map(str, relative_ranks)) if not debug else ''])


def get_result_filename(method, analysis, data_name, seed, fold1, fold2=-1, keepk=None, ratio=None, params=None):
    if analysis == FULL: assert ratio == 1
    directory_method = '{}/results/{}/{}/{}'.format(os.getcwd(), data_name, analysis, method)

    if method in BASELINES:
        assert fold2 == -1 and params is None
        if analysis == FULL:
            filename = '{}/{}_{}_seed{}_cv{}.npz'.format(directory_method, method, analysis, seed, fold1)
        elif analysis == SAMPLE:
            filename = '{}/{}_{}_seed{}_cv{}_ratio{:g}.npz'.format(directory_method, method, analysis, seed, fold1, ratio)
        elif analysis == KEEPK:
            filename = '{}/{}_{}_seed{}_cv{}_ratio{:g}_keep{}.npz'.format(directory_method, method, analysis, seed, fold1, ratio, keepk)
        else:
            raise ValueError('unknown analysis')

    elif method == KBMTL:
        assert len(params) == 3
        if analysis == FULL:
            filename = '{}/{}_{}_seed{}_cv{}{}_Alpha{:g}_Beta{:g}_Gamma{:g}.mat'.format(directory_method, method, analysis, seed, fold1, ('.{}'.format(fold2)) if fold2 != -1 else '', params[0], params[1], params[2])
        elif analysis == SAMPLE:
            filename = '{}/{}_{}_seed{}_cv{}{}_ratio{:g}_Alpha{:g}_Beta{:g}_Gamma{:g}.mat'.format(directory_method, method, analysis, seed, fold1, ('.{}'.format(fold2)) if fold2 != -1 else '', ratio, params[0], params[1], params[2])
        elif analysis == KEEPK:
            filename = '{}/{}_{}_seed{}_cv{}{}_ratio{:g}_keep{}_Alpha{:g}_Beta{:g}_Gamma{:g}.mat'.format(directory_method, method, analysis, seed, fold1, ('.{}'.format(fold2)) if fold2 != -1 else '', ratio, keepk, params[0], params[1], params[2])
        else:
            raise ValueError('unknown analysis')

    elif method == LKRL:
        assert len(params) == 1
        if analysis == FULL:
            filename = '{}/{}_{}_seed{}_cv{}{}_Lambda{:g}.npz'.format(directory_method, method, analysis, seed, fold1, ('.{}'.format(fold2)) if fold2 != -1 else '', params[0])
        elif analysis == SAMPLE:
            filename = '{}/{}_{}_seed{}_cv{}{}_ratio{:g}_Lambda{:g}.npz'.format(directory_method, method, analysis, seed, fold1, ('.{}'.format(fold2)) if fold2 != -1 else '', ratio, params[0])
        elif analysis == KEEPK:
            filename = '{}/{}_{}_seed{}_cv{}{}_ratio{:g}_keep{}_Lambda{:g}.npz'.format(directory_method, method, analysis, seed, fold1, ('.{}'.format(fold2)) if fold2 != -1 else '', ratio, keepk, params[0])
        else:
            raise ValueError('unknown analysis')

    elif method == KRL:
        assert len(params) == 2
        if analysis == FULL:
            filename = '{}/{}_{}_seed{}_cv{}{}_Lambda{:g}_Gamma{:g}.npz'.format(directory_method, method, analysis, seed, fold1, ('.{}'.format(fold2)) if fold2 != -1 else '', params[0], params[1])
        elif analysis == SAMPLE:
            filename = '{}/{}_{}_seed{}_cv{}{}_ratio{:g}_Lambda{:g}_Gamma{:g}.npz'.format(directory_method, method, analysis, seed, fold1, ('.{}'.format(fold2)) if fold2 != -1 else '', ratio, params[0], params[1])
        elif analysis == KEEPK:
            filename = '{}/{}_{}_seed{}_cv{}{}_ratio{:g}_keep{}_Lambda{:g}_Gamma{:g}.npz'.format(directory_method, method, analysis, seed, fold1, ('.{}'.format(fold2)) if fold2 != -1 else '', ratio, keepk, params[0], params[1])
        else:
            raise ValueError('unknown analysis')

    else:
        raise ValueError('unknown method')

    return filename


def update_metric(current_value, result, cv_metric, cv_k):
    return (current_value + non_nan_mean(cv_metric(result['Y_true'], result['Y_pred'], cv_k))) \
            if result is not None and ~np.isnan(current_value) else np.nan


def get_result(method, analysis, data_name, fold1, num_folds, keepk, ratio, seed, cv_metric, cv_ks, krl_lambdas, krl_gammas, lkrl_lambdas, kbmtl_alphas, kbmtl_betas, kbmtl_gammas):
    results = {}
    for cv_k in cv_ks:
        if method in BASELINES:
            filename = get_result_filename(method, analysis, data_name, seed, fold1, -1, keepk, ratio, None)
            result = safe_load(filename)

        elif method == KBMTL:
            metric10i = np.zeros([len(kbmtl_alphas), len(kbmtl_betas), len(kbmtl_gammas)])
            for fold2 in range(num_folds):
                for i, a in enumerate(kbmtl_alphas):
                    for j, b in enumerate(kbmtl_betas):
                        for k, g in enumerate(kbmtl_gammas):
                            filename_i = get_result_filename(method, analysis, data_name, seed, fold1, fold2, keepk, ratio, [a, b, g])
                            result_i = safe_load(filename_i, mat=True)
                            try:
                                metric10i[i, j, k] = update_metric(metric10i[i, j, k], result_i, cv_metric, cv_k)
                            except Exception:
                                print >> sys.stderr, 'ERROR:', filename_i
                                traceback.print_exc()
            param_idx = np.where(metric10i == np.max(metric10i))
            a, b, g = kbmtl_alphas[param_idx[0][0]], kbmtl_betas[param_idx[1][0]], kbmtl_gammas[param_idx[2][0]]
            print params_info(method, fold1, [(a, kbmtl_alphas), (b, kbmtl_betas), (g, kbmtl_gammas)], ratio, seed, cv_metric, cv_k, delim=DELIM)
            filename = get_result_filename(method, analysis, data_name, seed, fold1, -1, keepk, ratio, [a, b, g])
            result = safe_load(filename, mat=True)

        elif method == LKRL:
            metric10i = np.zeros([len(lkrl_lambdas)])
            for fold2 in range(num_folds):
                for i, l in enumerate(lkrl_lambdas):
                    filename_i = get_result_filename(method, analysis, data_name, seed, fold1, fold2, keepk, ratio, [l])
                    result_i = safe_load(filename_i)
                    metric10i[i] = update_metric(metric10i[i], result_i, cv_metric, cv_k)
            param_idx = np.where(metric10i == np.max(metric10i))
            l = lkrl_lambdas[param_idx[0][0]]
            print params_info(method, fold1, [(l, lkrl_lambdas)], ratio, seed, cv_metric, cv_k, delim=DELIM)
            filename = get_result_filename(method, analysis, data_name, seed, fold1, -1, keepk, ratio, [l])
            result = safe_load(filename)

        elif method == KRL:
            metric10i = np.zeros([len(krl_lambdas), len(krl_gammas)])
            for fold2 in range(num_folds):
                for i, l in enumerate(krl_lambdas):
                    for j, g in enumerate(krl_gammas):
                        filename_i = get_result_filename(method, analysis, data_name, seed, fold1, fold2, keepk, ratio, [l, g])
                        result_i = safe_load(filename_i)
                        metric10i[i, j] = update_metric(metric10i[i, j], result_i, cv_metric, cv_k)
            param_idx = np.where(metric10i == np.max(metric10i))
            l, g = krl_lambdas[param_idx[0][0]], krl_gammas[param_idx[1][0]]
            print params_info(method, fold1, [(l, krl_lambdas), (g, krl_gammas)], ratio, seed, cv_metric, cv_k, delim=DELIM)
            filename = get_result_filename(method, analysis, data_name, seed, fold1, -1, keepk, ratio, [l, g])
            result = safe_load(filename)
        else:
            raise ValueError('unknown method')

        if result is None:
            raise ValueError('could not find suitable result files for {}'.format(method))

        results[cv_k] = result

    return results


def main():

    print 'Calculating evaluation metrics from the predictions...'

    config_file = sys.argv[1]
    with open(config_file, 'r') as f:
        config = yaml.load(f)

    analysis = config['analysis']
    assert analysis in [FULL, SAMPLE, KEEPK], 'Unknown analysis type {} specified in the config file'.format(analysis)
    data_name = config['data']
    assert data_name in [GEX, WES, CNV, METH], 'Unknown data type {} specified in the config file'.format(data_name)
    methods = config['methods']
    for method in methods:
        assert method in [KRL, LKRL, KBMTL, KRR, RF, EN], 'Unknown method {} specified in the config file'.format(method)

    num_folds = config['cv']
    seeds = np.array(config['seeds'], dtype=int)
    sample_ratios = np.array(config['sample_ratios'], dtype=float)
    keepk_ratios = np.array(config['keepk_ratios'], dtype=float)
    keepk = config['keepk']
    k_evals = np.array(config['k_evals'], dtype=int)

    krl_k = config['krl_k']
    krl_lambdas = np.array(config['krl_lambdas'], dtype=float)
    krl_gammas = np.array(config['krl_gammas'], dtype=float)
    lkrl_k = config['lkrl_k']
    lkrl_lambdas = np.array(config['lkrl_lambdas'], dtype=float)
    kbmtl_alphas = np.array(config['kbmtl_alphas'], dtype=float)
    kbmtl_betas = np.array(config['kbmtl_betas'], dtype=float)
    kbmtl_gammas = np.array(config['kbmtl_gammas'], dtype=float)

    debug = False
    foldwise = analysis == FULL
    ratio_range = sample_ratios if analysis == SAMPLE else keepk_ratios
    cv_metric = NDCG
    single_cv_k = False
    k_percs = [1]
    k_opt = keepk if analysis == KEEPK and krl_k > keepk else krl_k
    cv_ks = set(list(k_evals) + [k_opt])
    rank_type = 'best_prediction'
    rank = True
    perc = True

    for seed in seeds:
        for ratio in ratio_range:
            abs_ranks, relative_ranks = {}, {}
            for method in methods:
                abs_ranks[method], relative_ranks[method] = [], []
                foldwise_ndcgk = {}
                foldwise_precisionk = {}
                percentiles = {}
                for fold1 in range(num_folds):
                    try:
                        results = get_result(method, analysis, data_name, fold1, num_folds, keepk, ratio, seed, cv_metric, cv_ks, krl_lambdas, krl_gammas, lkrl_lambdas, kbmtl_alphas, kbmtl_betas, kbmtl_gammas)
                    except Exception as e:
                        results = None
                        print >> sys.stderr, 'ERROR:', method, analysis, data_name, fold1, keepk, ratio, seed, cv_metric, cv_ks
                        traceback.print_exc()
                        break # if any of the folds is missing, there is no point to continue

                    for k in k_evals:
                        Y_true = results[k]['Y_true'] if not single_cv_k else results[iter(cv_ks).next()]['Y_true']
                        Y_pred = results[k]['Y_pred'] if not single_cv_k else results[iter(cv_ks).next()]['Y_pred']
                        if k not in foldwise_ndcgk: foldwise_ndcgk[k] = []
                        if k not in foldwise_precisionk: foldwise_precisionk[k] = []
                        foldwise_ndcgk[k].append(NDCG(Y_true, Y_pred, k))
                        foldwise_precisionk[k].append(Precision(Y_true, Y_pred, k))

                    for k in k_percs:
                        # do not re-optimise hyper-parameters for every k_perc, use args.k_opt
                        Y_true = results[k_opt]['Y_true'] if not single_cv_k else results[iter(cv_ks).next()]['Y_true']
                        Y_pred = results[k_opt]['Y_pred'] if not single_cv_k else results[iter(cv_ks).next()]['Y_pred']
                        if k not in percentiles: percentiles[k] = []
                        percentiles[k].extend(Percentile(Y_true, Y_pred, k))

                    # do not re-optimise hyper-parameters for method ranking, use args.k_opt
                    Y_true = results[k_opt]['Y_true'] if not single_cv_k else results[iter(cv_ks).next()]['Y_true']
                    Y_pred = results[k_opt]['Y_pred'] if not single_cv_k else results[iter(cv_ks).next()]['Y_pred']
                    abs_ranks[method].extend(Rank(Y_true, Y_pred, rank_type))

                if results is None:
                    del abs_ranks[method]
                    del relative_ranks[method]
                    print >> sys.stderr, 'SKIPPING method {}, ratio {:g}, seed {}'.format(method, ratio, seed) 
                    continue # some of the folds were missing, continue with the next method

                for k in k_evals:
                    if foldwise:
                        ndcg_result = [non_nan_mean(fold) for fold in foldwise_ndcgk[k]]
                        precision_result = [non_nan_mean(fold) for fold in foldwise_precisionk[k]]
                    else:
                        ndcg_result = list(itertools.chain.from_iterable(foldwise_ndcgk[k]))
                        precision_result = list(itertools.chain.from_iterable(foldwise_precisionk[k]))
                    print ndcg_info(ndcg_result, method, seed, ratio, k, debug=debug, delim=DELIM)
                    print precision_info(precision_result, method, seed, ratio, k, debug=debug, delim=DELIM)

                if perc:
                    for k in k_percs:
                        for p_tuple in percentiles[k]:
                            assert len(p_tuple) == k
                            for i, p in enumerate(p_tuple):
                                print percentile_info(p, i, method, seed, ratio, k, delim=DELIM)
                            if debug:
                                print '...'
                                break

            if rank and len(abs_ranks) > 0:
                size = len(abs_ranks.itervalues().next())
                assert np.all([size == len(abs_ranks[method]) for method in abs_ranks.keys()])
                for i in range(size):
                    this_patient = sorted([abs_ranks[method][i] for method in abs_ranks.keys()])
                    for method in abs_ranks.keys():
                        relative_ranks[method].append(this_patient.index(abs_ranks[method][i]) + 1)
                for method in abs_ranks.keys():
                    print rank_info(relative_ranks[method], method, seed, ratio, debug=debug, delim=DELIM)

    print 'Finished.'

if __name__ == "__main__":
    main()
