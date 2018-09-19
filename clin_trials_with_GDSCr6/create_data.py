import numpy as np
import pandas as pd
import scipy
import scipy.io
import itertools
import os
import sys
import collections
from openpyxl import load_workbook


IC50_file = os.path.join('..', 'data', 'TableS4A.xlsx')
GEX_file = os.path.join('..', 'data', 'Cell_line_RMA_proc_basalExp.txt')


def no_nulls(array_like):
    return not np.any(pd.isnull(array_like))


def is_unique(array_like):
    return np.unique(array_like).shape == np.array(array_like).flatten().shape


def intersect_index(*lists):
    assert len(lists) > 1
    assert np.all([no_nulls(l) for l in lists])
    assert np.all([is_unique(l) for l in lists])
    merged_df = None
    for i, l in enumerate(lists):
        l_df = pd.DataFrame(np.stack([np.arange(len(l)), l], axis=1), columns=['index{}'.format(i), 'value'])
        merged_df = pd.merge(merged_df, l_df, on='value', how='inner') if merged_df is not None else l_df.copy()
    perm_indexes = [merged_df['index{}'.format(i)].values.astype(np.int) for i in range(len(lists))]
    assert np.all([np.array_equal(lists[0][perm_indexes[0]], lists[i][perm_indexes[i]]) for i in range(1, len(lists))])
    return perm_indexes


def main():

	# IC50
	wb = load_workbook(filename=IC50_file)
	sheet = wb['TableS4A-IC50s']

	IC50_cell_ids, IC50_cell_names = [], []
	for i in range(7, 997):
	    IC50_cell_ids.append('%s' % sheet['A%s'%i].value)
	    IC50_cell_names.append(('%s' % sheet['B%s'%i].value).strip())
	IC50_cell_ids = np.array(IC50_cell_ids, dtype='str')
	IC50_cell_names = np.array(IC50_cell_names, dtype='str')

	IC50_drug_ids, IC50_drug_names = [], []
	for i, (cell_row5, cell_row6) in enumerate(zip(sheet[5], sheet[6])):
	    if i > 1:
		IC50_drug_ids.append('%s' % cell_row5.value)
		IC50_drug_names.append(('%s' % cell_row6.value).strip())
	IC50_drug_ids = np.array(IC50_drug_ids, dtype='str')
	IC50_drug_names_lc = np.array(map(lambda x: x.lower(), IC50_drug_names))
	IC50_drug_names = np.array(IC50_drug_names, dtype='str')

	IC50 = np.ones([IC50_cell_ids.shape[0], IC50_drug_ids.shape[0]]) * np.nan
	for i in range(7, 997):    
	    for j, cell in enumerate(sheet[i]):
		if j > 1:
		    if cell.value != 'NA':
		        IC50[i-7, j-2] = cell.value
		        
	print 'IC50_cell_ids', IC50_cell_ids.shape, 'IC50_cell_names', IC50_cell_names.shape
	print 'IC50_drug_ids', IC50_drug_ids.shape, 'IC50_drug_names', IC50_drug_names.shape
	print 'IC50', IC50.shape

	# Gene expression
	GEX = pd.read_csv(GEX_file, sep='\t')
	GEX_gene_symbols = np.array(GEX['GENE_SYMBOLS'], dtype='str')
	GEX_gene_titles = np.array(GEX['GENE_title'], dtype='str')
	GEX = GEX.drop(['GENE_SYMBOLS', 'GENE_title'], axis=1)
	GEX_cell_ids = np.array(GEX.columns, dtype='str')
	for i, cell_id in enumerate(GEX_cell_ids):
	    GEX_cell_ids[i] = cell_id[5:]
	GEX = np.array(GEX.values, dtype=np.float).T

	GEX_keep_index, IC50_keep_index = intersect_index(GEX_cell_ids, IC50_cell_ids)
	GEX = GEX[GEX_keep_index]
	GEX_cell_ids = GEX_cell_ids[GEX_keep_index]
	GEX_IC50 = IC50[IC50_keep_index]

	print 'GEX_gene_symbols', GEX_gene_symbols.shape, 'GEX_gene_titles', GEX_gene_titles.shape
	print 'GEX_cell_ids', GEX_cell_ids.shape
	print 'GEX', GEX.shape
	print 'GEX_IC50', GEX_IC50.shape

	GEX_cell_names = []
	for cell_id in GEX_cell_ids:
	    cell_names = IC50_cell_names[IC50_cell_ids == cell_id]
	    assert cell_names.shape == (1,), cell_id
	    GEX_cell_names.append(cell_names[0])
	GEX_cell_names = np.array(GEX_cell_names)

	print GEX_cell_names.shape
	print np.sum(GEX_gene_symbols == 'nan'), 'missing gene symbols'

	GEX_df = pd.DataFrame(data=np.transpose(GEX[:, GEX_gene_symbols != 'nan']),
		              index=GEX_gene_symbols[GEX_gene_symbols != 'nan'],
		              columns=GEX_cell_names)
	GEX_df.to_csv(os.path.join('data', 'GEX.csv'), index_label=False)
	assert len(GEX_df.index) == len(set(GEX_df.index))
	print GEX_df.shape

	DOCETAXEL = 'docetaxel'
	BORTEZOMIB = 'bortezomib'

	for drug in [DOCETAXEL, BORTEZOMIB]:
	    drug_ids = IC50_drug_ids[np.array(map(lambda x: x.lower(), IC50_drug_names)) == drug]
	    assert drug_ids.shape[0] == 1
	    drug_id = drug_ids[0]
	    
	    drug_GEX_IC50 = GEX_IC50[:,IC50_drug_ids == drug_id].reshape((GEX_IC50.shape[0],))
	    notnan = ~np.isnan(drug_GEX_IC50)
	    drug_GEX_IC50_df = pd.DataFrame(data=drug_GEX_IC50[notnan],
		                  index=GEX_cell_names[notnan],
		                  columns=['IC50'])
	    drug_GEX_IC50_df.to_csv(os.path.join('data', '{}_GEX_IC50.csv'.format(drug)), index_label=False)
	    print drug_GEX_IC50_df.shape


if __name__ == '__main__':
    main()

