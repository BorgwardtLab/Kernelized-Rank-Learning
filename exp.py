#!/usr/bin/env python

import os
import sys
import yaml
import argparse
import numpy as np
import scipy.io as sio 
import IPython as ip

from read_data import read_FULL, read_KEEPK, read_SAMPLE
from baseline.baseline import KRR, EN, RF
from KRL.KRL import KRL 

def main():
    config_file = sys.argv[1]
    with open(config_file, 'r') as f:
        config = yaml.load(f)

    seeds = np.array(config['seed'], dtype=int)
    analysis = config['analysis']
    data_name = config['data']
    cv = config['cv']
    sample_ratio = np.array(config['sample_ratio'], dtype=float)
    keepk_ratio = np.array(config['keepk_ratio'], dtype=float)
    keepk = config['keepk']
    krr_alpha = np.array(config['krr_alpha'], dtype=float)
    krr_gamma = np.array(config['krr_gamma'], dtype=float)
    rf_nestimators = np.array(config['rf_nestimators'], dtype=int)
    en_alpha = np.array(config['en_alpha'], dtype=float)
    en_l1ratio = np.array(config['en_l1ratio'], dtype=float)
    krl_k = config['krl_k']
    krl_lambda = np.array(config['krl_lambda'], dtype=float)
    krl_gamma = np.array(config['krl_gamma'], dtype=float)
    lkrl_k = config['lkrl_k']
    lkrl_lambda = np.array(config['lkrl_lambda'], dtype=float)
    methods = config['methods']

    directory = '%s/result' % os.getcwd()
    if not os.path.exists(directory):
        os.makedirs(directory)       

    directory = '%s/result/%s' % (os.getcwd(), data_name)
    if not os.path.exists(directory):
        os.makedirs(directory)    

    directory = '%s/result/%s/%s' % (os.getcwd(), data_name, analysis)
    if not os.path.exists(directory):
        os.makedirs(directory)      

    for method in methods:
        directory_method = '%s/%s' % (directory, method)
        if not os.path.exists(directory_method):
            os.makedirs(directory_method)       

        if analysis == 'FULL':
            for seed in seeds:
                for i in range(cv):
                    X_train, X_test, Y_train, Y_test = read_FULL(data_name, seed, i)
                    if method == 'KRR':
                        Y_pred = KRR(X_train, Y_train, X_test, krr_alpha, krr_gamma, cv, seed)
                        np.savez('%s/%s_FULL_seed%s_cv%s' % (directory_method, method, seed, i), Y_true=Y_test, Y_pred=Y_pred)
                    elif method == 'EN':
                        Y_pred = EN(X_train, Y_train, X_test, en_alpha, en_l1ratio, cv, seed)
                        np.savez('%s/%s_FULL_seed%s_cv%s' % (directory_method, method, seed, i), Y_true=Y_test, Y_pred=Y_pred)
                    elif method == 'RF':
                        Y_pred = RF(X_train, Y_train, X_test, rf_nestimators, cv, seed)
                        np.savez('%s/%s_FULL_seed%s_cv%s' % (directory_method, method, seed, i), Y_true=Y_test, Y_pred=Y_pred)
                    elif method == 'KRL':
                        for l in krl_lambda:
                            for g in krl_gamma:
                                Y_pred = KRL(X_train, Y_train, X_test, krl_k, l, g, njobs=1)
                                np.savez('%s/%s_FULL_seed%s_cv%s_Lambda%s_Gamma%s' % (directory_method, method, seed, i, l, g), Y_true=Y_test, Y_pred=Y_pred)
                    elif method == 'LKRL':
                        for l in lkrl_lambda:
                            Y_pred = KRL(X_train, Y_train, X_test, lkrl_k, lkrl_lambda, 'linear', njobs=1)
                            np.savez('%s/%s_FULL_seed%s_cv%s_Lambda%s' % (directory_method, method, seed, i, l), Y_true=Y_test, Y_pred=Y_pred)
                    for j in range(cv):
                        X_train, X_test, Y_train, Y_test = read_FULL(data_name, seed, i, j)
                        if method == 'KRL':
                            for l in krl_lambda:
                                for g in krl_gamma:
                                    Y_pred = KRL(X_train, Y_train, X_test, krl_k, l, g, njobs=1)
                                    np.savez('%s/%s_FULL_seed%s_cv%s.%s_Lambda%s_Gamma%s' % (directory_method, method, seed, i, j, l, g), Y_true=Y_test, Y_pred=Y_pred)
                        elif method == 'LKRL':
                            for l in lkrl_lambda:
                                Y_pred = KRL(X_train, Y_train, X_test, lkrl_k, lkrl_lambda, 'linear', njobs=1)
                                np.savez('%s/%s_FULL_seed%s_cv%s.%s_Lambda%s' % (directory_method, method, seed, i, j, l), Y_true=Y_test, Y_pred=Y_pred)                  

        if analysis == 'SAMPLE':
            for seed in seeds:
                for sr in sample_ratio:
                    for i in range(cv):
                        X_train, X_test, Y_train, Y_test = read_SAMPLE(data_name, seed, sr, i)
                        if method == 'KRR':
                            Y_pred = KRR(X_train, Y_train, X_test, krr_alpha, krr_gamma, cv, seed)
                            np.savez('%s/%s_SAMPLE_seed%s_cv%s_ratio%s' % (directory_method, method, seed, i, sr), Y_true=Y_test, Y_pred=Y_pred)
                        elif method == 'EN':
                            Y_pred = EN(X_train, Y_train, X_test, en_alpha, en_l1ratio, cv, seed)
                            np.savez('%s/%s_SAMPLE_seed%s_cv%s_ratio%s' % (directory_method, method, seed, i, sr), Y_true=Y_test, Y_pred=Y_pred)
                        elif method == 'RF':
                            Y_pred = RF(X_train, Y_train, X_test, rf_nestimators, cv, seed)
                            np.savez('%s/%s_SAMPLE_seed%s_cv%s_ratio%s' % (directory_method, method, seed, i, sr), Y_true=Y_test, Y_pred=Y_pred)
                        elif method == 'KRL':
                            for l in krl_lambda:
                                for g in krl_gamma:
                                    Y_pred = KRL(X_train, Y_train, X_test, krl_k, l, g, njobs=1)
                                    np.savez('%s/%s_SAMPLE_seed%s_cv%s_ratio%s_Lambda%s_Gamma%s' % (directory_method, method, seed, i, sr, l, g), Y_true=Y_test, Y_pred=Y_pred)
                        elif method == 'LKRL':
                            for l in lkrl_lambda:
                                Y_pred = KRL(X_train, Y_train, X_test, lkrl_k, lkrl_lambda, 'linear', njobs=1)
                                np.savez('%s/%s_SAMPLE_seed%s_cv%s_ratio%s_Lambda%s' % (directory_method, method, seed, sr, i, l), Y_true=Y_test, Y_pred=Y_pred)
                        for j in range(cv):
                            X_train, X_test, Y_train, Y_test = read_SAMPLE(data_name, seed, sr, i, j)
                            if method == 'KRL':
                                for l in krl_lambda:
                                    for g in krl_gamma:
                                        Y_pred = KRL(X_train, Y_train, X_test, krl_k, l, g, njobs=1)
                                        np.savez('%s/%s_SAMPLE_seed%s_cv%s.%s_ratio%s_Lambda%s_Gamma%s' % (directory_method, method, seed, i, j, sr, l, g), Y_true=Y_test, Y_pred=Y_pred)
                            elif method == 'LKRL':
                                for l in lkrl_lambda:
                                    Y_pred = KRL(X_train, Y_train, X_test, lkrl_k, lkrl_lambda, 'linear', njobs=1)
                                    np.savez('%s/%s_SAMPLE_seed%s_cv%s.%s_ratio%s_Lambda%s' % (directory_method, method, seed, i, j, sr, l), Y_true=Y_test, Y_pred=Y_pred)             

        if analysis == 'KEEPK':
            for seed in seeds:
                for kr in keepk_ratio:
                    for i in range(cv):
                        X_train, X_test, Y_train, Y_test = read_KEEPK(data_name, seed, kr, i)
                        if method == 'KRR':
                            Y_pred = KRR(X_train, Y_train, X_test, krr_alpha, krr_gamma, cv, seed)
                            np.savez('%s/%s_KEEPK_seed%s_cv%s_ratio%s' % (directory_method, method, seed, i, kr), Y_true=Y_test, Y_pred=Y_pred)
                        elif method == 'EN':
                            Y_pred = EN(X_train, Y_train, X_test, en_alpha, en_l1ratio, cv, seed)
                            np.savez('%s/%s_KEEPK_seed%s_cv%s_ratio%s' % (directory_method, method, seed, i, kr), Y_true=Y_test, Y_pred=Y_pred)
                        elif method == 'RF':
                            Y_pred = RF(X_train, Y_train, X_test, rf_nestimators, cv, seed)
                            np.savez('%s/%s_KEEPK_seed%s_cv%s_ratio%s' % (directory_method, method, seed, i, kr), Y_true=Y_test, Y_pred=Y_pred)
                        elif method == 'KRL':
                            for l in krl_lambda:
                                for g in krl_gamma:
                                    Y_pred = KRL(X_train, Y_train, X_test, krl_k, l, g, njobs=1)
                                    np.savez('%s/%s_KEEPK_seed%s_cv%s_ratio%s_Lambda%s_Gamma%s' % (directory_method, method, seed, i, kr, l, g), Y_true=Y_test, Y_pred=Y_pred)
                        elif method == 'LKRL':
                            for l in lkrl_lambda:
                                Y_pred = KRL(X_train, Y_train, X_test, lkrl_k, lkrl_lambda, 'linear', njobs=1)
                                np.savez('%s/%s_KEEPK_seed%s_cv%s_ratio%s_Lambda%s' % (directory_method, method, seed, i, kr, l), Y_true=Y_test, Y_pred=Y_pred)
                        for j in range(cv):
                            X_train, X_test, Y_train, Y_test = read_KEEPK(data_name, seed, kr, i, j)
                            if method == 'KRL':
                                for l in krl_lambda:
                                    for g in krl_gamma:
                                        Y_pred = KRL(X_train, Y_train, X_test, krl_k, l, g, njobs=1)
                                        np.savez('%s/%s_KEEPK_seed%s_cv%s.%s_ratio%s_Lambda%s_Gamma%s' % (directory_method, method, seed, i, j, kr, l, g), Y_true=Y_test, Y_pred=Y_pred)
                            elif method == 'LKRL':
                                for l in lkrl_lambda:
                                    Y_pred = KRL(X_train, Y_train, X_test, lkrl_k, lkrl_lambda, 'linear', njobs=1)
                                    np.savez('%s/%s_KEEPK_seed%s_cv%s.%s_ratio%s_Lambda%s' % (directory_method, method, seed, i, j, kr, l), Y_true=Y_test, Y_pred=Y_pred)

if __name__ == '__main__':
    main()
