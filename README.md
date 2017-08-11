# Kernelized-Rank-Learning
Kernelized rank learning for personalized drug recommendation. 

## Dependencies
**Python 2.7**, modern versions of **numpy**, **scipy**, **scikit-learn**, **joblib**, **cvxopt**, **pandas**, **matplotlib**, **seaborn**, **openpyxl**, **gzip**. All of them available via **pip**.

## KRL usage
The implementation of KRL is in KRL/KRL.py

    Y_pred = KRL(X_train, Y_train, X_test, k, lambda, gamma, njobs)

Input:

    X_train: n1 x p matrix of n1 samples and p feaures

    Y_train: n1 x m matrix of n1 samples and m drugs

    X_test:  n2 x p matrix of n2 samples and p feaures
    
    k:       an integer indicating the evaluation parameter k in NDCG@k used during training
    
    lambda:  a float indicating the regularization parameter
    
    gamma:   a float indicating the width of the RBF kernel or 'linear' for linear kernel
    
    njobs:   an integer indicating the number of cores that will be used for parallization

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

## Contact
Any questions can be directed to:
   * Xiao He: xiao.he [at] bsse.ethz.ch
   * Lukas Folkman: lukas.folkman [at] bsse.ethz.ch

