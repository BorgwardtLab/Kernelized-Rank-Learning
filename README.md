# Kernelized-Rank-Learning
Kernelized rank learning for personalized drug recommendation. 

## Dependencies
**Python 2.7**, modern versions of **numpy**, **scipy**, **scikit-learn**, **joblib**, **cvxopt**. All of them available via **pip**.

## KRL Usage
The implementation of KRL is in KRL/KRL.py

    Y_pred = KRL(X_train, Y_train, X_test, k, lambda, gamma, njobs)

input:

    X_train: a n1 X p matrix of n1 samples and p feaures

    Y_train: a n1 X m matrix of n1 samples and m drugs

    X_test:  a n2 X p matrix of n2 samples and p feaures
    
    k:       an integer indicates the k in NDCG@k in training
    
    lambda:  a float indictes the regularization parameter
    
    gamma:   a float indicates the gamma in rbf kernel or 'linear' for linear kernel
    
    njobs:   an integer indicates the number of cores that will be used for parallization

Output:

    Y_pred:  a n2 X m matrix of n2 samples and m drugs

## Reproduce the experiments
Download and preprocess the dataset
    
    python dataset.py URLs_to_GDSC_datasets.txt
    
Prepare the sampled dataset    

    python prepare.py config.yaml
    
Run experiments for KRL, LKRL, KRR, EN and RF   

    python exp.py config.yaml
    
Run experiments for KBMTL
    
    bash prepare_KBMTL.sh
    matlab -nodisplay -nosplash -nodesktop -r "run_KBMTL"
    
## Contact
Any questions can be directed to:
   * Xiao He: xiao.he [at] bsse.ethz.ch
   * Lukas Folkman: lukas.folkman [at] bsse.ethz.ch

