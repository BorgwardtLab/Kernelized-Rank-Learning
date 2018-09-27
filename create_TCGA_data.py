import numpy as np
import os
from misc import LAPATINIB, VELIPARIB, OLAPARIB1, OLAPARIB2, TALAZOPARIB, RUCAPARIB, RUXOLITINIB
from misc import DRUG_NAMES, DRUG_IDS
from misc import PARP_INHIBS, TNBC_DRUGS
from misc import load_pRRophetic_data, intersect_index

DATA_DIR = 'data'
np.random.seed(0)

# GDSC IC50 matrix
gdsc_gex = np.load(os.path.join(DATA_DIR, 'GDSC_GEX.npz'))
gdsc_X = gdsc_gex['X']
gdsc_Y = gdsc_gex['Y']
gdsc_drug_ids = gdsc_gex['drug_ids'].astype(int)
gdsc_drug_names = gdsc_gex['drug_names']
gdsc_samples = gdsc_gex['cell_ids']
print 'gdsc_X:', gdsc_X.shape, 'gdsc_Y:', gdsc_Y.shape

# idx of the selected BRCA drugs in the GDSC IC50 matrix
drug_index = {}
for drug in [LAPATINIB] + TNBC_DRUGS:
    drug_index[drug] = np.nonzero(gdsc_drug_ids == int(DRUG_IDS[drug]))[0][0]

# Geeleher et al. (2017) R scripts were used to homogenize GDSC gene expression (array-based) with TCGA gene expression (RNA-seq)
# The data below were retrieved from breast_cancer_analysis.R call to Geeleher's pRRophetic package
# train_samples are CCLs, test_samples are TCGA patients
train_X, _, train_samples, test_X, _, test_samples = load_pRRophetic_data(data_dir=DATA_DIR, fn_prefix='pRRophetic_TCGA_BRCA', test_y_suffix=None)
merged = intersect_index(np.array(map(lambda x: x.lower().strip('x').strip(), train_samples)),
                         np.array(map(lambda x: x.lower().strip(), gdsc_samples)))
X_keep_index = np.array(merged['index1'].values, dtype=np.int)
Y_keep_index = np.array(merged['index2'].values, dtype=np.int)
train_X = train_X[X_keep_index]
train_Y = gdsc_Y[Y_keep_index, :]

# keep only primary solid tumors (01), first aliquot (A)
sample_types = np.array(map(lambda x: x[13:16], test_samples))
test_samples = np.array(map(lambda x: x.upper().replace('.', '-')[:12], test_samples))
is_tumor = sample_types == '01A'
test_X = test_X[is_tumor]
test_samples = test_samples[is_tumor]

# remove missing genes
not_nan = ~np.isnan(test_X[0])
test_X = test_X[:, not_nan]
train_X = train_X[:, not_nan]

# data integrity check
assert np.all(~np.isnan(test_X))
assert np.all(~np.isnan(train_X))
assert np.all([not np.all(np.isnan(y)) for y in train_Y])

# put the drugs of interest in the first columns of the Y matrix (just for backwards compatibility)
idx = range(train_Y.shape[1])
for i, drug in enumerate([LAPATINIB] + TNBC_DRUGS):
    idx.remove(drug_index[drug])
    idx.insert(i, drug_index[drug])
    drug_index[drug] = i
train_Y = train_Y[:,idx]

# only CCLs with more than one drug response recorded can contribute to learning to rank
not_nan = [np.sum(~np.isnan(y)) > 1 for y in train_Y]
train_Y = train_Y[not_nan]
train_X = train_X[not_nan]

print 'GDSC CCLs (training data): X {} Y {}'.format(train_X.shape, train_Y.shape)
print 'TCGA patients (test data): X {} y {}'.format(test_X.shape, test_samples.shape)
np.savez(os.path.join(DATA_DIR, 'KRL_data_for_TCGA_BRCA'),
         train_X=train_X, train_Y=train_Y,
         drug_ids=gdsc_drug_ids, drug_names=gdsc_drug_names,
         test_X=test_X, test_ids=test_samples)
