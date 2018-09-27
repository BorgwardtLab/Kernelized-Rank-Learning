import numpy as np
import os
from misc import DOCETAXEL, BORTEZOMIB
from misc import DRUG_NAMES, DRUG_IDS
from misc import load_pRRophetic_y, load_pRRophetic_data, intersect_index

DATA_DIR = 'data'
ONLY_GDSC_R2_SAMPLES = True

# GDSC IC50 matrix
gdsc_gex = np.load(os.path.join(DATA_DIR, 'GDSC_GEX.npz'))
gdsc_X = gdsc_gex['X']
gdsc_Y = gdsc_gex['Y']
gdsc_drug_ids = gdsc_gex['drug_ids'].astype(int)
gdsc_drug_names = gdsc_gex['drug_names']
gdsc_samples = gdsc_gex['cell_names']
print 'gdsc_X:', gdsc_X.shape, 'gdsc_Y:', gdsc_Y.shape

for drug in [DOCETAXEL, BORTEZOMIB]:  
    # Geeleher et al. (2017) R scripts were used to homogenize GDSC gene expression (array-based) with TCGA gene expression (RNA-seq)
    # The data below were retrieved from breast_cancer_analysis.R call to Geeleher's pRRophetic package
    # train_* are CCLs, test_* are clinical trial patients
    train_X, _, train_samples, test_X, test_y, _ = load_pRRophetic_data(data_dir=DATA_DIR, fn_prefix='pRRophetic_{}'.format(drug))
    
    if drug == DOCETAXEL:
        # docetaxel response was measured in a binary fashion
        binary_test_y = test_y
    elif drug == BORTEZOMIB:
        # bortezomib response was measured in both binary and categorical (RECIST-like) fashion
        binary_test_y = load_pRRophetic_y(os.path.join(DATA_DIR, 'pRRophetic_{}_binary_test_y.csv.gz'.format(drug)))
        assert np.array_equal(test_y != 'PGx_Response = IE', binary_test_y != 'PGx_Responder = IE')
        filter_IE = test_y != 'PGx_Response = IE'
        test_X = test_X[filter_IE]
        binary_test_y = binary_test_y[filter_IE]
        test_y = test_y[filter_IE]
        binary_test_y[binary_test_y == 'PGx_Responder = NR'] = 0
        binary_test_y[binary_test_y == 'PGx_Responder = R'] = 1
        binary_test_y = binary_test_y.astype(int)
        test_y[test_y == 'PGx_Response = CR'] = 'CR'
        test_y[test_y == 'PGx_Response = PR'] = 'PR'
        test_y[test_y == 'PGx_Response = MR'] = 'MR'
        test_y[test_y == 'PGx_Response = NC'] = 'NC'
        test_y[test_y == 'PGx_Response = PD'] = 'PD'

    if ONLY_GDSC_R2_SAMPLES:
        # in the paper we used the intersection of GDSC release 6 and release 2 cell lines
        # see  Section 3.4.2 for details
        train_samples_r2 = np.loadtxt(os.path.join(DATA_DIR, 'pRRophetic_{}_GDSC_r2_samples.txt'.format(drug)), dtype=str)
        merged = intersect_index(np.array(map(lambda x: x.strip().lower(), train_samples)),
                                 np.array(map(lambda x: x.strip().lower(), train_samples_r2)))
        r2r6_intersect = np.array(merged['index1'].values, dtype=np.int)
        train_X = train_X[r2r6_intersect]
        train_samples = train_samples[r2r6_intersect]
        print 'Using only GDSC release 2 and release 6 intersection cell lines:', train_X.shape[0]
    
    # re-index train samples to match with GDSC release 6 IC50s
    merged = intersect_index(np.array(map(lambda x: x.strip().lower(), train_samples)),
                             np.array(map(lambda x: x.strip().lower(), gdsc_samples)))
    X_keep_index = np.array(merged['index1'].values, dtype=np.int)
    Y_keep_index = np.array(merged['index2'].values, dtype=np.int)
    train_X = train_X[X_keep_index]
    train_Y = gdsc_Y[Y_keep_index, :]
    
    # train only on cell lines for which IC50 is available for the given drug of interest
    drug_index = np.nonzero(gdsc_drug_ids == DRUG_IDS[drug])[0][0]
    train_y = train_Y[:,drug_index]
    not_nan = ~np.isnan(train_y)
    train_X = train_X[not_nan]
    train_Y = train_Y[not_nan]
    train_y = train_y[not_nan]

    # data integrity check
    assert np.all(~np.isnan(test_X))
    assert np.all(~np.isnan(train_X))
    assert np.all(~np.isnan(train_Y[:,drug_index]))
        
    print 'GDSC CCLs (training data): X {} Y {}'.format(train_X.shape, train_Y.shape)
    print '{} patients (test data): X {} y {}'.format(drug, test_X.shape, test_y.shape)
    np.savez(os.path.join(DATA_DIR, 'KRL_data_for_{}.npz'.format(drug)),
             train_X=train_X, train_Y=train_Y,
             drug_ids=gdsc_drug_ids, drug_names=gdsc_drug_names,
             test_X=test_X, test_y=test_y, binary_test_y=binary_test_y)
