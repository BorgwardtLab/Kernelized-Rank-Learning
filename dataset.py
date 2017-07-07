#!/usr/bin/env python

import os
import sys
import urllib
import zipfile
import numpy as np
import pandas as pd
import IPython as ip
from openpyxl import load_workbook

def intersect_index(list1, list2):
    '''
    Given two lists find the index of intersect in list1

    Parameterst
    ----------
    list1: 1d numpy array
    list2: 1d numpy array
    '''
    intersect = np.intersect1d(list1, list2)
    intersect = pd.DataFrame(intersect, columns=['id'])
    list1 = np.vstack([np.arange(list1.shape[0]), list1]).T
    list1 = pd.DataFrame(list1, columns=['index1', 'id'])
    list2 = np.vstack([np.arange(list2.shape[0]), list2]).T
    list2 = pd.DataFrame(list2, columns=['index2', 'id'])    
    merged = pd.merge(list1, intersect, on='id', how='right')
    merged = pd.merge(merged, list2, on='id', how='left')

    return merged

def main():
    # Read URLs to GDSC datasets
    urls_file = sys.argv[1]
    urls = []
    with open(urls_file) as f:
        for line in f:
            if line.startswith('http://'):
                urls.append(line[:-1])


    # Create data folder
    directory = '%s/data' % os.getcwd()
    if not os.path.exists(directory):
        os.makedirs(directory)


    # Download datasets
    testfile = urllib.URLopener()
    for url in urls:
        parts = url.split('/')
        testfile.retrieve(url, '%s/%s' % (directory, parts[-1]))
        if parts[-1].endswith('.zip'):
            # Unzip the downloaded file
            zip_ref = zipfile.ZipFile('%s/%s' % (directory, parts[-1]), 'r')
            zip_ref.extractall('%s' % directory)
            zip_ref.close()


    # Read Gene expression dataset
    GEX_file = '%s/Cell_line_RMA_proc_basalExp.txt' % directory
    GEX = pd.read_csv(GEX_file, sep='\t')
    GEX_gene_symbols = np.array(GEX['GENE_SYMBOLS'], dtype='str')
    GEX = GEX.drop(['GENE_SYMBOLS', 'GENE_title'], axis=1)
    GEX_cell_ids = np.array(GEX.columns, dtype='str')
    for i, cell_id in enumerate(GEX_cell_ids):
        GEX_cell_ids[i] = cell_id[5:]
    GEX = np.array(GEX.as_matrix(), dtype=np.float).T


    # Read Exome sequencing dataset
    WES_file = '%s/CellLines_CG_BEMs/PANCAN_SEQ_BEM.txt' % directory
    WES = pd.read_csv(WES_file, sep='\t')
    WES_CG = np.array(WES['CG'], dtype='str')
    WES = WES.drop(['CG'], axis=1)
    WES_cell_ids = np.array(WES.columns, dtype='str')
    WES = np.array(WES.as_matrix(), dtype=np.int).T


    # Read Copy number dataset
    CNV_file = '%s/CellLine_CNV_BEMs/PANCAN_CNA_BEM.rdata.txt' % directory
    CNV = pd.read_csv(CNV_file, sep='\t')
    CNV_cell_ids = np.array(CNV['Unnamed: 0'], dtype='str')
    CNV = CNV.drop(['Unnamed: 0'], axis=1)
    CNV_cna = np.array(CNV.columns, dtype='str')
    CNV = np.array(CNV.as_matrix(), dtype=int)


    # Read Methylation dataset
    MET_file = '%s/METH_CELLLINES_BEMs/PANCAN.txt' % directory
    MET = pd.read_csv(MET_file, sep='\t')
    MET_met = np.array(MET['Unnamed: 0'], dtype='str')
    MET = MET.drop(['Unnamed: 0'], axis=1)
    MET_cell_ids = np.array(MET.columns, dtype='str')
    MET = np.array(MET.as_matrix(), dtype=int).T


    # Read LOG_IC50 dataset
    IC50_file = '%s/TableS4A.xlsx' % directory
    wb = load_workbook(filename=IC50_file)
    sheet = wb['TableS4A-IC50s']
    IC50_cell_ids = []
    for i in range(7, 997):
        IC50_cell_ids.append('%s' % sheet['A%s'%i].value)
    IC50_cell_ids = np.array(IC50_cell_ids, dtype='str')
    IC50_drug_ids = []
    for i, cell in enumerate(sheet[5]):
        if i > 1:
            IC50_drug_ids.append('%s' % cell.value)
    IC50_drug_ids = np.array(IC50_drug_ids, dtype='str')
    IC50 = np.ones([IC50_cell_ids.shape[0], IC50_drug_ids.shape[0]]) * np.nan
    for i in range(7, 997):    
        for j, cell in enumerate(sheet[i]):
            if j > 1:
                if cell.value != 'NA':
                    IC50[i-7, j-2] = cell.value


    # Read LOG_IC50 Threshold
    threshold_file = '%s/TableS5C.xlsx' % directory
    wb = load_workbook(filename=threshold_file)
    sheet = wb['Table-S5C binaryIC50s']
    threshold = []
    for i, cell in enumerate(sheet[7]):
        if i > 1:
            threshold.append(cell.value)
    threshold = np.array(threshold)
    drug_ids_file = '%s/TableS1F.xlsx' % directory
    wb = load_workbook(filename=drug_ids_file)
    sheet = wb['TableS1F_ScreenedCompounds']
    threshold_drug_ids = []
    for i in range(4, 269):
        threshold_drug_ids.append('%s' % sheet['B%s'%i].value)
    threshold_drug_ids = np.array(threshold_drug_ids)


    # Normalize IC50 by the threshold
    merged = intersect_index(IC50_drug_ids, threshold_drug_ids)
    IC50_keep_index = np.array(merged['index1'].as_matrix(), dtype=np.int)
    IC50_drug_ids = IC50_drug_ids[IC50_keep_index]
    IC50 = IC50[:, IC50_keep_index]
    threshold_keep_index = np.array(merged['index2'].as_matrix(), dtype=np.int)
    threshold_drug_ids = threshold_drug_ids[threshold_keep_index]
    threshold = threshold[threshold_keep_index]
    IC50_norm = - (IC50 - threshold)
    IC50_norm_min = np.min(IC50_norm[~np.isnan(IC50_norm)])
    IC50_norm = IC50_norm - IC50_norm_min


    # Save the GEX features and normalized IC50 dataset
    merged = intersect_index(GEX_cell_ids, IC50_cell_ids)
    GEX_keep_index = np.array(merged['index1'].as_matrix(), dtype=np.int)
    IC50_keep_index = np.array(merged['index2'].as_matrix(), dtype=np.int)
    GEX = GEX[GEX_keep_index]
    GEX_cell_ids = GEX_cell_ids[GEX_keep_index]
    IC50 = IC50_norm[IC50_keep_index]
    np.savez('%s/GDSC_GEX.npz' % directory, X=GEX, Y=IC50, cell_ids=GEX_cell_ids, drug_ids=IC50_drug_ids, GEX_gene_symbols=GEX_gene_symbols)


    # Save the WES features and normalized IC50 dataset
    merged = intersect_index(WES_cell_ids, IC50_cell_ids)
    WES_keep_index = np.array(merged['index1'].as_matrix(), dtype=np.int)
    IC50_keep_index = np.array(merged['index2'].as_matrix(), dtype=np.int)
    WES = WES[WES_keep_index]
    WES_cell_ids = WES_cell_ids[WES_keep_index]
    IC50 = IC50_norm[IC50_keep_index]
    np.savez('%s/GDSC_WES.npz' % directory, X=WES, Y=IC50, cell_ids=WES_cell_ids, drug_ids=IC50_drug_ids, WES_CG=WES_CG)


    # Save the CNV features and normalized IC50 dataset
    merged = intersect_index(CNV_cell_ids, IC50_cell_ids)
    CNV_keep_index = np.array(merged['index1'].as_matrix(), dtype=np.int)
    IC50_keep_index = np.array(merged['index2'].as_matrix(), dtype=np.int)
    CNV = CNV[CNV_keep_index]
    CNV_cell_ids = CNV_cell_ids[CNV_keep_index]
    IC50 = IC50_norm[IC50_keep_index]
    np.savez('%s/GDSC_CNV.npz' % directory, X=CNV, Y=IC50, cell_ids=CNV_cell_ids, drug_ids=IC50_drug_ids, CNV_cna=CNV_cna)


    # Save the MET features and normalized IC50 dataset
    merged = intersect_index(MET_cell_ids, IC50_cell_ids)
    MET_keep_index = np.array(merged['index1'].as_matrix(), dtype=np.int)
    IC50_keep_index = np.array(merged['index2'].as_matrix(), dtype=np.int)
    MET = MET[MET_keep_index]
    MET_cell_ids = MET_cell_ids[MET_keep_index]
    IC50 = IC50_norm[IC50_keep_index]
    np.savez('%s/GDSC_MET.npz' % directory, X=MET, Y=IC50, cell_ids=MET_cell_ids, drug_ids=IC50_drug_ids, MET_met=MET_met)

if __name__ == '__main__':
    main()




