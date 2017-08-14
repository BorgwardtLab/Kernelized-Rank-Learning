# Kernelized-Rank-Learning

Kernelized Rank Learning (KRL) for Personalized Drug Recommendation

## Dependencies

**Python** (2.7.12); for the KRL method: **numpy** (1.13.1), **scipy** (0.19.1), **scikit-learn** (0.18.1), **joblib** (0.11), **cvxopt** (1.1.9); for reproducing our experiments: **PyYAML** (3.12), **openpyxl** (2.4.8); for analysing the results: **Jupyter** (4.3.0), **pandas** (0.20.3), **matplotlib** (2.0.2), **seaborn** (0.7.1). All of the packages are available via **pip** (9.0.1). Tested on **Ubuntu** (16.04).

## KRL usage

The implementation of KRL is in ``KRL/KRL.py``:

    Y_pred = KRL(X_train, Y_train, X_test, k, lambda, gamma, njobs)

Input:

    X_train: n1 x p matrix of n1 samples and p feaures

    Y_train: n1 x m matrix of n1 samples and m drugs

    X_test:  n2 x p matrix of n2 samples and p feaures
    
    k:       an integer indicating the evaluation parameter k in NDCG@k used during training
    
    lambda:  a float indicating the regularization parameter
    
    gamma:   a float indicating the width of the RBF kernel or 'linear' for linear kernel
    
    njobs:   an integer indicating the number of cores that will be used for parallelization

Output:

    Y_pred:  n2 x m matrix of n2 samples and m drugs

## Scripts for reproducing our experiments

Download and preprocess the GDSC dataset:
    
    python dataset.py URLs_to_GDSC_datasets.txt
    
Edit the configuration file to specify which experiments you wish to run:
    
    vim config.yaml

Split the data for training and testing, create folds for cross-validation:

    python prepare.py config.yaml
    
Run experiments for KRL, LKRL, KRR, EN and RF:

    python exp.py config.yaml
    
Run experiments for KBMTL (https://github.com/mehmetgonen/bmtmkl). The code below was tested using Matlab R2016b. Execute this only if KBMTL was specified in the configuration file:
    
    bash prepare_KBMTL.sh
    matlab -nodisplay -nosplash -nodesktop -r "run_KBMTL"

Evaluate the results:
    
    python results.py config.yaml > results.txt

Plot the results:
    
    jupyter notebook plots.ipynb

## Analysis of the published results

All results and configuration files for our paper are located in the folder ``paper/``. To re-analyze these results, a Jupyter notebook is provided:
    
    jupyter notebook plots.ipynb


## Installation tests

Run tests and compare the results (takes approx. one hour on a desktop computer):

    bash test/run_tests.sh
    diff KEEPK_results.txt test/KEEPK_results.txt
    diff SAMPLE_results.txt test/SAMPLE_results.txt 

## Contact

Any questions can be directed to:

   * Xiao He: xiao.he [at] bsse.ethz.ch
   * Lukas Folkman: lukas.folkman [at] bsse.ethz.ch

