#!/usr/bin/env python

import os
import sys
import yaml
import argparse
import numpy as np
import scipy.io as sio 
import IPython as ip

from misc import read_FULL, read_KEEPK, read_SAMPLE
from misc import FULL, SAMPLE, KEEPK
from misc import GEX, WES, CNV, METH
from misc import KRL as KRL_STR
from misc import LKRL as LKRL_STR
from misc import KBMTL as KBMTL_STR
from misc import KRR as KRR_STR
from misc import RF as RF_STR
from misc import EN as EN_STR
from results import get_result_filename
from baseline.baseline import KRR, EN, RF
from KRL.KRL import KRL 


def main():
    config_file = sys.argv[1]
    with open(config_file, 'r') as f:
        config = yaml.load(f)

    seeds = np.array(config['seeds'], dtype=int)
    analysis = config['analysis']
    assert analysis in [FULL, SAMPLE, KEEPK], 'Unknown analysis type {} specified in the config file'.format(analysis)
    data_name = config['data']
    assert data_name in [GEX, WES, CNV, METH], 'Unknown data type {} specified in the config file'.format(data_name)
    cv = config['cv']
    sample_ratios = np.array(config['sample_ratios'], dtype=float)
    keepk_ratios = np.array(config['keepk_ratios'], dtype=float)
    keepk = config['keepk']
    krr_alphas = np.array(config['krr_alphas'], dtype=float)
    krr_gammas = np.array(config['krr_gammas'], dtype=float)
    rf_nestimators = np.array(config['rf_nestimators'], dtype=int)
    en_alphas = np.array(config['en_alphas'], dtype=float)
    en_l1ratios = np.array(config['en_l1ratios'], dtype=float)
    krl_k = config['krl_k']
    krl_lambdas = np.array(config['krl_lambdas'], dtype=float)
    krl_gammas = np.array(config['krl_gammas'], dtype=float)
    lkrl_k = config['lkrl_k']
    lkrl_lambdas = np.array(config['lkrl_lambdas'], dtype=float)
    njobs = config['njobs']
    verbose = config['verbose']
    methods = config['methods']
    for method in methods:
        assert method in [KRL_STR, LKRL_STR, KBMTL_STR, KRR_STR, RF_STR, EN_STR], 'Unknown method {} specified in the config file'.format(method)

    directory = '{}/results'.format(os.getcwd())
    if not os.path.exists(directory):
        os.makedirs(directory)       

    directory = '{}/results/{}'.format(os.getcwd(), data_name)
    if not os.path.exists(directory):
        os.makedirs(directory)    

    directory = '{}/results/{}/{}'.format(os.getcwd(), data_name, analysis)
    if not os.path.exists(directory):
        os.makedirs(directory)      

    for method in methods:
        if method == KBMTL_STR:
            # KBMTL has to be run separately in Matlab
            continue

        directory_method = '{}/{}'.format(directory, method)
        if not os.path.exists(directory_method):
            os.makedirs(directory_method)       

        if analysis == FULL:
            print 'FULL: running the experiments on the full dataset...'
            for seed in seeds:
                print 'Running {}, seed {}'.format(method, seed)
                for i in range(cv):
                    X_train, X_test, Y_train, Y_test = read_FULL(data_name, seed, i)
                    if method == KRR_STR:
                        Y_pred = KRR(X_train, Y_train, X_test, krr_alphas, krr_gammas, cv, seed)
                        result_fn = get_result_filename(method, analysis, data_name, seed, i)
                        np.savez(result_fn, Y_true=Y_test, Y_pred=Y_pred)
                    elif method == EN_STR:
                        Y_pred = EN(X_train, Y_train, X_test, en_alphas, en_l1ratios, cv, seed)
                        result_fn = get_result_filename(method, analysis, data_name, seed, i)
                        np.savez(result_fn, Y_true=Y_test, Y_pred=Y_pred)
                    elif method == RF_STR:
                        Y_pred = RF(X_train, Y_train, X_test, rf_nestimators, cv, seed)
                        result_fn = get_result_filename(method, analysis, data_name, seed, i)
                        np.savez(result_fn, Y_true=Y_test, Y_pred=Y_pred)
                    elif method == KRL_STR:
                        for l in krl_lambdas:
                            for g in krl_gammas:
                                Y_pred = KRL(X_train, Y_train, X_test, krl_k, l, g, njobs, verbose)
                                result_fn = get_result_filename(method, analysis, data_name, seed, i, params=[l, g])
                                np.savez(result_fn, Y_true=Y_test, Y_pred=Y_pred)
                        for j in range(cv):
                            X_train, X_test, Y_train, Y_test = read_FULL(data_name, seed, i, j)
                            for l in krl_lambdas:
                                for g in krl_gammas:
                                    Y_pred = KRL(X_train, Y_train, X_test, krl_k, l, g, njobs, verbose)
                                    result_fn = get_result_filename(method, analysis, data_name, seed, i, fold2=j, params=[l, g])
                                    np.savez(result_fn, Y_true=Y_test, Y_pred=Y_pred)
                    elif method == LKRL_STR:
                        for l in lkrl_lambdas:
                            Y_pred = KRL(X_train, Y_train, X_test, lkrl_k, l, 'linear', njobs, verbose)
                            result_fn = get_result_filename(method, analysis, data_name, seed, i, params=[l])
                            np.savez(result_fn, Y_true=Y_test, Y_pred=Y_pred)
                        for j in range(cv):
                            X_train, X_test, Y_train, Y_test = read_FULL(data_name, seed, i, j)
                            for l in lkrl_lambdas:
                                Y_pred = KRL(X_train, Y_train, X_test, lkrl_k, l, 'linear', njobs, verbose)
                                result_fn = get_result_filename(method, analysis, data_name, seed, i, fold2=j, params=[l])
                                np.savez(result_fn, Y_true=Y_test, Y_pred=Y_pred)

        elif analysis == SAMPLE:
            print 'SAMPLE: running the experiments on the sparse training datasets...'
            for seed in seeds:
                for sr in sample_ratios:
                    print 'Running {}, seed {}, sample_ratio {:g}'.format(method, seed, sr)
                    for i in range(cv):
                        X_train, X_test, Y_train, Y_test = read_SAMPLE(data_name, seed, sr, i)
                        if method == KRR_STR:
                            Y_pred = KRR(X_train, Y_train, X_test, krr_alphas, krr_gammas, cv, seed)
                            result_fn = get_result_filename(method, analysis, data_name, seed, i, ratio=sr)
                            np.savez(result_fn, Y_true=Y_test, Y_pred=Y_pred)
                        elif method == EN_STR:
                            Y_pred = EN(X_train, Y_train, X_test, en_alphas, en_l1ratios, cv, seed)
                            result_fn = get_result_filename(method, analysis, data_name, seed, i, ratio=sr)
                            np.savez(result_fn, Y_true=Y_test, Y_pred=Y_pred)
                        elif method == RF_STR:
                            Y_pred = RF(X_train, Y_train, X_test, rf_nestimators, cv, seed)
                            result_fn = get_result_filename(method, analysis, data_name, seed, i, ratio=sr)
                            np.savez(result_fn, Y_true=Y_test, Y_pred=Y_pred)
                        elif method == KRL_STR:
                            for l in krl_lambdas:
                                for g in krl_gammas:
                                    Y_pred = KRL(X_train, Y_train, X_test, krl_k, l, g, njobs, verbose)
                                    result_fn = get_result_filename(method, analysis, data_name, seed, i, ratio=sr, params=[l, g])
                                    np.savez(result_fn, Y_true=Y_test, Y_pred=Y_pred)
                            for j in range(cv):
                                X_train, X_test, Y_train, Y_test = read_SAMPLE(data_name, seed, sr, i, j)
                                for l in krl_lambdas:
                                    for g in krl_gammas:
                                        Y_pred = KRL(X_train, Y_train, X_test, krl_k, l, g, njobs, verbose)
                                        result_fn = get_result_filename(method, analysis, data_name, seed, i, fold2=j, ratio=sr, params=[l, g])
                                        np.savez(result_fn, Y_true=Y_test, Y_pred=Y_pred)
                        elif method == LKRL_STR:
                            for l in lkrl_lambdas:
                                Y_pred = KRL(X_train, Y_train, X_test, lkrl_k, l, 'linear', njobs, verbose)
                                result_fn = get_result_filename(method, analysis, data_name, seed, i, ratio=sr, params=[l])
                                np.savez(result_fn, Y_true=Y_test, Y_pred=Y_pred)
                            for j in range(cv):
                                X_train, X_test, Y_train, Y_test = read_SAMPLE(data_name, seed, sr, i, j)
                                for l in lkrl_lambdas:
                                    Y_pred = KRL(X_train, Y_train, X_test, lkrl_k, l, 'linear', njobs, verbose)
                                    result_fn = get_result_filename(method, analysis, data_name, seed, i, fold2=j, ratio=sr, params=[l])
                                    np.savez(result_fn, Y_true=Y_test, Y_pred=Y_pred)             

        elif analysis == KEEPK:
            print 'KEEPK: running the experiments on the sparse training datasets biased towards effective therapies...'
            for seed in seeds:
                for kr in keepk_ratios:
                    print 'Running {}, seed {}, keepk {}, keepk_ratio {:g}'.format(method, seed, keepk, kr)
                    for i in range(cv):
                        X_train, X_test, Y_train, Y_test = read_KEEPK(data_name, seed, kr, keepk, i)
                        if method == KRR_STR:
                            Y_pred = KRR(X_train, Y_train, X_test, krr_alphas, krr_gammas, cv, seed)
                            result_fn = get_result_filename(method, analysis, data_name, seed, i, keepk=keepk, ratio=kr)
                            np.savez(result_fn, Y_true=Y_test, Y_pred=Y_pred)
                        elif method == EN_STR:
                            Y_pred = EN(X_train, Y_train, X_test, en_alphas, en_l1ratios, cv, seed)
                            result_fn = get_result_filename(method, analysis, data_name, seed, i, keepk=keepk, ratio=kr)
                            np.savez(result_fn, Y_true=Y_test, Y_pred=Y_pred)
                        elif method == RF_STR:
                            Y_pred = RF(X_train, Y_train, X_test, rf_nestimators, cv, seed)
                            result_fn = get_result_filename(method, analysis, data_name, seed, i, keepk=keepk, ratio=kr)
                            np.savez(result_fn, Y_true=Y_test, Y_pred=Y_pred)
                        elif method == KRL_STR:
                            for l in krl_lambdas:
                                for g in krl_gammas:
                                    Y_pred = KRL(X_train, Y_train, X_test, krl_k, l, g, njobs, verbose)
                                    result_fn = get_result_filename(method, analysis, data_name, seed, i, keepk=keepk, ratio=kr, params=[l, g])
                                    np.savez(result_fn, Y_true=Y_test, Y_pred=Y_pred)
                            for j in range(cv):
                                X_train, X_test, Y_train, Y_test = read_KEEPK(data_name, seed, kr, keepk, i, j)
                                for l in krl_lambdas:
                                    for g in krl_gammas:
                                        Y_pred = KRL(X_train, Y_train, X_test, krl_k, l, g, njobs, verbose)
                                        result_fn = get_result_filename(method, analysis, data_name, seed, i, fold2=j, keepk=keepk, ratio=kr, params=[l, g])
                                        np.savez(result_fn, Y_true=Y_test, Y_pred=Y_pred)
                        elif method == LKRL_STR:
                            for l in lkrl_lambdas:
                                Y_pred = KRL(X_train, Y_train, X_test, lkrl_k, l, 'linear', njobs, verbose)
                                result_fn = get_result_filename(method, analysis, data_name, seed, i, keepk=keepk, ratio=kr, params=[l])
                                np.savez(result_fn, Y_true=Y_test, Y_pred=Y_pred)
                            for j in range(cv):
                                X_train, X_test, Y_train, Y_test = read_KEEPK(data_name, seed, kr, keepk, i, j)
                                for l in lkrl_lambdas:
                                    Y_pred = KRL(X_train, Y_train, X_test, lkrl_k, l, 'linear', njobs, verbose)
                                    result_fn = get_result_filename(method, analysis, data_name, seed, i, fold2=j, keepk=keepk, ratio=kr, params=[l])
                                    np.savez(result_fn, Y_true=Y_test, Y_pred=Y_pred)

    print 'Finished.'

if __name__ == '__main__':
    main()
