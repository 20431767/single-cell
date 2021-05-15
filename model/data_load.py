import gzip
import os
import pickle
import platform
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import scanpy as sc
from pyreadr import read_r
from scipy import io, sparse
from tqdm import tqdm

def load_preprocessed(X_count, X_unscaled, Y, cell_names, gene_names, z_normalize=True, batch=None,exp=None):
    adata_cnt = sc.AnnData(X_count)
    adata = sc.AnnData(X_unscaled)
    adata_unscaled = adata.copy()
    adata.obs['cell_groups'] = Y
    adata_cnt.obs['cell_groups'] = Y
    adata_unscaled.obs['cell_groups'] = Y

    adata.obs['cell_type'] = cell_names
    adata_cnt.obs['cell_type'] = cell_names
    adata_unscaled.obs['cell_type'] = cell_names
    if batch is not None:
        adata.obs['batch'] = batch
        adata_cnt.obs['batch'] = batch
        adata_unscaled.obs['batch'] = batch
    if exp is not None:
        adata.obs['experiment'] = exp
    cell_names = cell_names
    adata.var['gene_name'] = gene_names
    adata_cnt.var['gene_name'] = gene_names
    adata_unscaled.var['gene_name'] = gene_names
    if z_normalize:
        sc.pp.scale(adata)
    adata = adata.copy()
    return adata, adata_unscaled, adata_cnt

def preprocess(X, Y, cell_names, gene_names, quality_control=True, normalize_input=True, log1p_input=True, highly_genes=None, z_normalize=True, dataset_dir=".", generate_files=True, batch=None,exp = None):
    if generate_files:
        #idx = np.random.choice(len(X), len(X), replace=False)
        idx = np.random.choice(X.shape[0],X.shape[0],replace=False)
        # print("idx",idx)
        adata = sc.AnnData(X[idx])
        
        # print("X.shape",X.shape)
        # print("Y.shape",Y.shape)
        adata.obs['cell_groups'] = Y[idx]
        adata.obs['cell_type'] = cell_names[idx]
        if batch is not None:
            adata.obs['batch'] = batch[idx]
        if exp is not None:
            adata.obs['experiment'] = exp[idx]
        cell_names = cell_names[idx]
    adata.var['gene_name'] = gene_names
    #print("batch",adata.obs['batch'])

    if quality_control:
        sc.pp.filter_genes(adata, min_counts=10)
        #sc.pp.filter_cells(adata, min_counts=1)
    # print("adata111.X.shape:",adata.X.shape)
    adata_cnt = adata.copy()
    
    if normalize_input:
        sc.pp.normalize_per_cell(adata)
        # sc.pp.normalize_per_cell(adata, counts_per_cell_after=10**6)
    if log1p_input:
        sc.pp.log1p(adata)
 
    if highly_genes >= 0. and highly_genes <= 1.:
        sc.pp.highly_variable_genes(adata, n_top_genes = int(highly_genes * np.array(adata.X).shape[1]), subset=True)
    elif int(highly_genes) > 1 and int(highly_genes) <= np.array(adata.X).shape[1] :
        sc.pp.highly_variable_genes(adata, n_top_genes = int(highly_genes), subset=True)

    if generate_files:
        gene_names = [str(i) for i in adata.var['highly_variable'].index]
        adata_cnt = adata_cnt[:, adata.var['highly_variable'].index].copy()
    adata_unscaled = adata.copy()
    
    if z_normalize:
        sc.pp.scale(adata)
    adata = adata.copy()
    
    if generate_files:
        # format: content.form
        # e.g. (X.count.T).(name.space)
        tmp_out = adata_unscaled.to_df()
        tmp_out.index = cell_names
        tmp_out.columns = gene_names
        tmp_out.to_csv(os.path.join(dataset_dir, str(highly_genes) + "." "X.name.csv"))
        tmp_out.to_csv(os.path.join(dataset_dir, str(highly_genes) + "." "X.name.tab"), sep='\t')
        tmp_out.to_csv(os.path.join(dataset_dir, str(highly_genes) + "." "X.name.space"), sep=' ')
        tmp_out = adata_unscaled.to_df().T
        tmp_out.index = gene_names
        tmp_out.columns = cell_names
        tmp_out.to_csv(os.path.join(dataset_dir, str(highly_genes) + "." "X.T.name.csv"))
        tmp_out.to_csv(os.path.join(dataset_dir, str(highly_genes) + "." "X.T.name.tab"), sep='\t')
        tmp_out.to_csv(os.path.join(dataset_dir, str(highly_genes) + "." "X.T.name.space"), sep=' ')
        tmp_out = adata_cnt.to_df()
        tmp_out.index = cell_names
        tmp_out.columns = gene_names
        tmp_out.to_csv(os.path.join(dataset_dir, str(highly_genes) + "." "X.count.name.csv"))
        tmp_out.to_csv(os.path.join(dataset_dir, str(highly_genes) + "." "X.count.name.tab"), sep='\t')
        tmp_out.to_csv(os.path.join(dataset_dir, str(highly_genes) + "." "X.count.name.space"), sep=' ')
        tmp_out = adata_cnt.to_df().T
        tmp_out.index = gene_names
        tmp_out.columns = cell_names
        tmp_out.to_csv(os.path.join(dataset_dir, str(highly_genes) + "." "X.count.T.name.csv"))
        tmp_out.to_csv(os.path.join(dataset_dir, str(highly_genes) + "." "X.count.T.name.tab"), sep='\t')
        tmp_out.to_csv(os.path.join(dataset_dir, str(highly_genes) + "." "X.count.T.name.space"), sep=' ')
        
        np.savetxt(os.path.join(dataset_dir, str(highly_genes) + "." "X.csv"), adata_unscaled.X, delimiter=',')
        np.savetxt(os.path.join(dataset_dir, str(highly_genes) + "." "X.T.csv"), adata_unscaled.X.T, delimiter=',')
        np.savetxt(os.path.join(dataset_dir, str(highly_genes) + "." "X.tab"), adata_unscaled.X, delimiter='\t')
        np.savetxt(os.path.join(dataset_dir, str(highly_genes) + "." "X.T.tab"), adata_unscaled.X.T, delimiter='\t')
        np.savetxt(os.path.join(dataset_dir, str(highly_genes) + "." "X.space"), adata_unscaled.X, delimiter=' ')
        np.savetxt(os.path.join(dataset_dir, str(highly_genes) + "." "X.T.space"), adata_unscaled.X.T, delimiter=' ')
        np.savetxt(os.path.join(dataset_dir, str(highly_genes) + "." "X.count.csv"), adata_cnt.X, delimiter=',')
        np.savetxt(os.path.join(dataset_dir, str(highly_genes) + "." "X.count.T.csv"), adata_cnt.X.T, delimiter=',')
        np.savetxt(os.path.join(dataset_dir, str(highly_genes) + "." "X.count.tab"), adata_cnt.X, delimiter='\t')
        np.savetxt(os.path.join(dataset_dir, str(highly_genes) + "." "X.count.T.tab"), adata_cnt.X.T, delimiter='\t')
        np.savetxt(os.path.join(dataset_dir, str(highly_genes) + "." "X.count.space"), adata_cnt.X, delimiter=' ')
        np.savetxt(os.path.join(dataset_dir, str(highly_genes) + "." "X.count.T.space"), adata_cnt.X.T, delimiter=' ')
        tmp_out = adata_cnt.obs['cell_groups'].copy()
        tmp_out.index = cell_names
        tmp_out.to_csv(os.path.join(dataset_dir, str(highly_genes) + "." "Y.name.csv"))
        tmp_out.to_csv(os.path.join(dataset_dir, str(highly_genes) + "." "Y.name.tab"), sep='\t')
        tmp_out.to_csv(os.path.join(dataset_dir, str(highly_genes) + "." "Y.name.space"), sep=' ')
        np.savetxt(os.path.join(dataset_dir, str(highly_genes) + "." "Y.csv"), adata_cnt.obs['cell_groups'], delimiter=',')
        np.savetxt(os.path.join(dataset_dir, str(highly_genes) + "." "Y.tab"), adata_cnt.obs['cell_groups'], delimiter='\t')
        np.savetxt(os.path.join(dataset_dir, str(highly_genes) + "." "Y.space"), adata_cnt.obs['cell_groups'], delimiter=' ')
        if batch is not None:
            tmp_out = adata_cnt.obs['batch'].copy()
            tmp_out.index = cell_names
            tmp_out.to_csv(os.path.join(dataset_dir, str(highly_genes) + "." "batch.name.csv"))
            tmp_out.to_csv(os.path.join(dataset_dir, str(highly_genes) + "." "batch.name.tab"), sep='\t')
            tmp_out.to_csv(os.path.join(dataset_dir, str(highly_genes) + "." "batch.name.space"), sep=' ')
            np.savetxt(os.path.join(dataset_dir, str(highly_genes) + "." "batch.csv"), adata_cnt.obs['batch'], delimiter=',')
            np.savetxt(os.path.join(dataset_dir, str(highly_genes) + "." "batch.tab"), adata_cnt.obs['batch'], delimiter='\t')
            np.savetxt(os.path.join(dataset_dir, str(highly_genes) + "." "batch.space"), adata_cnt.obs['batch'], delimiter=' ')
        if exp is not None:
            tmp_out = adata_cnt.obs['experiment'].copy()
            tmp_out.index = cell_names
            tmp_out.to_csv(os.path.join(dataset_dir, str(highly_genes) + "." "experiment.name.csv"))
            tmp_out.to_csv(os.path.join(dataset_dir, str(highly_genes) + "." "experiment.name.tab"), sep='\t')
            tmp_out.to_csv(os.path.join(dataset_dir, str(highly_genes) + "." "experiment.name.space"), sep=' ')
            np.savetxt(os.path.join(dataset_dir, str(highly_genes) + "." "experiment.csv"), adata_cnt.obs['experiment'], delimiter=',',fmt = '%s')
            np.savetxt(os.path.join(dataset_dir, str(highly_genes) + "." "experiment.tab"), adata_cnt.obs['experiment'], delimiter='\t',fmt = '%s')
            np.savetxt(os.path.join(dataset_dir, str(highly_genes) + "." "experiment.space"), adata_cnt.obs['experiment'], delimiter=' ',fmt = '%s')
    return adata, adata_unscaled, adata_cnt


def load_time_course(data_dir, highly_genes=2000, generate_files=False):
    dataset_dir = 'GSE75748_sc_time_course_ec'
    if generate_files:
        sc_df = pd.read_table(os.path.join(data_dir, dataset_dir, 'GSE75748_sc_time_course_ec.csv'), index_col=0,sep=',').T
        sc_anno_df = pd.read_table(os.path.join(data_dir, dataset_dir, 'GSE75748_sc_time_course_ec.csv'),sep=',',header=None).iloc[0,1:]
        sc_df.index = sc_anno_df.index
        if sc_df.columns.duplicated().sum() != 0:
            sc_df = sc_df.groupby(sc_df.columns).sum()
        sc_expression = sc_df.values
        sc_anno = [str(a).split('.')[1].split('_')[0] for a in sc_anno_df.values]
        sc_types_set, sc_types_labels = np.unique(sc_anno, return_inverse=True)
        adata, adata_unscaled, adata_cnt = preprocess(sc_expression, sc_types_labels, sc_df.index, sc_df.columns,
                                                      normalize_input=False, highly_genes=highly_genes, 
                                                      dataset_dir=os.path.join(data_dir, dataset_dir), generate_files=generate_files)
    else:
        sc_cnt_df = pd.read_csv(os.path.join(data_dir, dataset_dir, '%s.X.count.name.csv' % str(highly_genes)), index_col=0)
        sc_unscaled = pd.read_csv(os.path.join(data_dir, dataset_dir, '%s.X.name.csv' % str(highly_genes)), index_col=0).values
        sc_types_labels = pd.read_csv(os.path.join(data_dir, dataset_dir, '%s.Y.csv' % str(highly_genes)), header=None)[0].to_numpy()
        adata, adata_unscaled, adata_cnt = load_preprocessed(sc_cnt_df.values, sc_unscaled, sc_types_labels, sc_cnt_df.index, sc_cnt_df.columns)

    return adata, adata_unscaled, adata_cnt


def load_human_progenitor(data_dir, highly_genes=2000, generate_files=False, type_level=1):
    dataset_dir = 'human_progenitor'
    if generate_files:
        sc_name = os.path.join(data_dir, dataset_dir, "GSE75748_sc_cell_type_ec.csv.gz")
        sc_df = pd.read_csv(sc_name, compression='gzip', header=0, index_col=0)
        if type_level == 1:
            sc_celltypes = np.array(list(map(lambda s: s.split('_')[0], sc_df.columns)))
        elif type_level == 2:
            sc_celltypes = np.array(list(map(lambda s: s.split('.')[0], sc_df.columns)))
        if sc_df.index.duplicated().sum() != 0:
            sc_df = sc_df.groupby(sc_df.index).sum()
        sc_expression = sc_df.T.values
        sc_types_set, sc_types_labels = np.unique(sc_celltypes, return_inverse=True)

        adata, adata_unscaled, adata_cnt = preprocess(sc_expression, sc_types_labels, sc_df.columns, sc_df.index, 
                                                    normalize_input=False, highly_genes=highly_genes, 
                                                    dataset_dir=os.path.join(data_dir, dataset_dir), generate_files=generate_files)
    else:
        sc_cnt_df = pd.read_csv(os.path.join(data_dir, dataset_dir, '%s.X.count.name.csv' % str(highly_genes)), index_col=0)
        sc_unscaled = pd.read_csv(os.path.join(data_dir, dataset_dir, '%s.X.name.csv' % str(highly_genes)), index_col=0).values
        sc_types_labels = pd.read_csv(os.path.join(data_dir, dataset_dir, '%s.Y.csv' % str(highly_genes)), header=None)[0].to_numpy()
        adata, adata_unscaled, adata_cnt = load_preprocessed(sc_cnt_df.values, sc_unscaled, sc_types_labels, sc_cnt_df.index, sc_cnt_df.columns)
    return adata, adata_unscaled, adata_cnt

def load_mouse_cortex(data_dir, highly_genes=2000, generate_files=False):
    dataset_dir = 'mouse_cortex'
    if generate_files:
        sc_df = pd.read_table(os.path.join(data_dir, dataset_dir, 'expression_mRNA_17-Aug-2014.txt'), skiprows=11, header=None, index_col=0).iloc[:, 1:]
        sc_anno_df = pd.read_table(os.path.join(data_dir, dataset_dir, 'expression_mRNA_17-Aug-2014.txt'), skiprows=7, nrows=2, index_col=1).iloc[0, 1:]
        sc_df.columns = sc_anno_df.index
        if sc_df.index.duplicated().sum() != 0:     
            sc_df = sc_df.groupby(sc_df.index).sum()
        sc_expression = sc_df.T.values
        sc_types_set, sc_types_labels = np.unique(sc_anno_df, return_inverse=True)
        adata, adata_unscaled, adata_cnt = preprocess(sc_expression, sc_types_labels, sc_df.columns, sc_df.index, 
                                                    highly_genes=highly_genes, dataset_dir=os.path.join(data_dir, dataset_dir), generate_files=generate_files)
    else:
        sc_cnt_df = pd.read_csv(os.path.join(data_dir, dataset_dir, '%s.X.count.name.csv' % str(highly_genes)), index_col=0)
        sc_unscaled = pd.read_csv(os.path.join(data_dir, dataset_dir, '%s.X.name.csv' % str(highly_genes)), index_col=0).values
        sc_types_labels = pd.read_csv(os.path.join(data_dir, dataset_dir, '%s.Y.csv' % str(highly_genes)), header=None)[0].to_numpy()
        adata, adata_unscaled, adata_cnt = load_preprocessed(sc_cnt_df.values, sc_unscaled, sc_types_labels, sc_cnt_df.index, sc_cnt_df.columns)
    
    return adata, adata_unscaled, adata_cnt

def load_human_melanoma(data_dir, highly_genes=2000, generate_files=False):
    dataset_dir = "human_melanoma"
    if generate_files:
        sc_df = pd.read_table(os.path.join(data_dir, dataset_dir, 'GSE72056_melanoma_single_cell_revised_v2.txt'), skiprows=4, header=None, index_col=0)
        sc_anno_df = pd.read_table(os.path.join(data_dir, dataset_dir, 'GSE72056_melanoma_single_cell_revised_v2.txt'), nrows=3, index_col=0).iloc[2, :]
        sc_df.columns = sc_anno_df.index
        sc_df = sc_df.loc[:, sc_anno_df != 0]
        sc_anno_df = sc_anno_df.loc[sc_anno_df != 0]
        if sc_df.index.duplicated().sum() != 0:     
            sc_df = sc_df.groupby(sc_df.index).sum()
        sc_expression = sc_df.T.values
        sc_types_set, sc_types_labels = np.unique(sc_anno_df, return_inverse=True)
        adata, adata_unscaled, adata_cnt = preprocess(sc_expression, sc_types_labels, sc_df.columns, sc_df.index, normalize_input=False, 
                                                    highly_genes=highly_genes, dataset_dir=os.path.join(data_dir, dataset_dir), generate_files=generate_files)
    else:
        sc_cnt_df = pd.read_csv(os.path.join(data_dir, dataset_dir, '%s.X.count.name.csv' % str(highly_genes)), index_col=0)
        sc_unscaled = pd.read_csv(os.path.join(data_dir, dataset_dir, '%s.X.name.csv' % str(highly_genes)), index_col=0).values
        sc_types_labels = pd.read_csv(os.path.join(data_dir, dataset_dir, '%s.Y.csv' % str(highly_genes)), header=None)[0].to_numpy()
        adata, adata_unscaled, adata_cnt = load_preprocessed(sc_cnt_df.values, sc_unscaled, sc_types_labels, sc_cnt_df.index, sc_cnt_df.columns)
    
    return adata, adata_unscaled, adata_cnt

def load_mouse_stem(data_dir, highly_genes=2000, generate_files=False):
    dataset_dir = "mouse_stem"
    if generate_files:
        # files = ["GSM1599494_ES_d0_main.csv", "GSM1599495_ES_d0_biorep_techrep1.csv", 
        #          "GSM1599496_ES_d0_biorep_techrep2.csv", "GSM1599497_ES_d2_LIFminus.csv",
        #          "GSM1599498_ES_d4_LIFminus.csv", "GSM1599499_ES_d7_LIFminus.csv"]
        files = ["GSM1599494_ES_d0_main.csv", "GSM1599497_ES_d2_LIFminus.csv",
                "GSM1599498_ES_d4_LIFminus.csv", "GSM1599499_ES_d7_LIFminus.csv"]
        sc_df = pd.read_csv(os.path.join(data_dir, dataset_dir, files[0]), header=None, index_col=0)
        sc_anno = [files[0].split("_")[2]] * len(sc_df.columns)
        for f in files[1:]:
            sc_other_df = pd.read_csv(os.path.join(data_dir, dataset_dir, f), header=None, index_col=0)
            sc_df = pd.concat([sc_df, sc_other_df], axis=0)
            sc_anno += [f.split("_")[2]] * len(sc_other_df.columns)
        sc_df.columns = list(range(len(sc_df.columns)))
        if sc_df.index.duplicated().sum() != 0:     
            sc_df = sc_df.groupby(sc_df.index).sum()
        sc_expression = sc_df.T.values
        sc_types_set, sc_types_labels = np.unique(sc_anno, return_inverse=True)
        adata, adata_unscaled, adata_cnt = preprocess(sc_expression, sc_types_labels, sc_df.columns, sc_df.index,
                                                    highly_genes=highly_genes, dataset_dir=os.path.join(data_dir, dataset_dir), generate_files=generate_files)
    else:
        sc_cnt_df = pd.read_csv(os.path.join(data_dir, dataset_dir, '%s.X.count.name.csv' % str(highly_genes)), index_col=0)
        sc_unscaled = pd.read_csv(os.path.join(data_dir, dataset_dir, '%s.X.name.csv' % str(highly_genes)), index_col=0).values
        sc_types_labels = pd.read_csv(os.path.join(data_dir, dataset_dir, '%s.Y.csv' % str(highly_genes)), header=None)[0].to_numpy()
        adata, adata_unscaled, adata_cnt = load_preprocessed(sc_cnt_df.values, sc_unscaled, sc_types_labels, sc_cnt_df.index, sc_cnt_df.columns)
    
    return adata, adata_unscaled, adata_cnt

def load_human_pancreas(data_dir, highly_genes=2000, generate_files=False):
    dataset_dir = "human_pancreas"
    print("generate_files",generate_files)
    if generate_files:
        files = ["GSM2230757_human1_umifm_counts.csv", "GSM2230758_human2_umifm_counts.csv", 
            "GSM2230759_human3_umifm_counts.csv", "GSM2230760_human4_umifm_counts.csv"]
        sc_df = pd.read_csv(os.path.join(data_dir, dataset_dir, files[0]), index_col=0)
        sc_anno_df = sc_df.iloc[:, 1]
        sc_df = sc_df.iloc[:, 2:]
        #sc_batch = [files[0].split("_")[1]] * len(sc_df.index)
        sc_batch_labels = np.zeros(sc_df.shape[0])
        for f in files[1:]:
            sc_other_df = pd.read_csv(os.path.join(data_dir, dataset_dir, f), index_col=0)
            sc_other_anno_df = sc_other_df.iloc[:, 1]
            sc_other_df = sc_other_df.iloc[:, 2:]
            sc_anno_df = pd.concat([sc_anno_df, sc_other_anno_df])
            sc_df = pd.concat([sc_df, sc_other_df])
            #sc_batch += [f.split("_")[1]] * len(sc_other_df.index)
            #print("sc_batch_labels0000",sc_batch_labels)
            #print("sc_df.shape[0]00000",sc_df.shape[0])
            sc_batch_labels = np.concatenate([sc_batch_labels,np.ones(sc_other_df.shape[0])],axis=0)
            
        if sc_df.columns.duplicated().sum() != 0:     
            sc_df = sc_df.groupby(sc_df.columns).sum()
        sc_expression = sc_df.values
        sc_types_set, sc_types_labels = np.unique(sc_anno_df, return_inverse=True)
        #sc_batch_set, sc_batch_labels = np.unique(sc_batch, return_inverse=True) #"human1_lib3.final_cell_0711"
        print("sc_batch_labels",sc_batch_labels.shape)
        print("sc_types_labels",sc_types_labels.shape)
        #print("sc_anno_df",sc_anno_df)
        adata, adata_unscaled, adata_cnt = preprocess(sc_expression, sc_types_labels, sc_df.index, sc_df.columns,
                                                    highly_genes=highly_genes, dataset_dir=os.path.join(data_dir, dataset_dir), 
                                                    generate_files=generate_files, batch=sc_batch_labels)
    else:
        sc_cnt_df = pd.read_csv(os.path.join(data_dir, dataset_dir, '%s.X.count.name.csv' % str(highly_genes)), index_col=0)
        sc_unscaled = pd.read_csv(os.path.join(data_dir, dataset_dir, '%s.X.name.csv' % str(highly_genes)), index_col=0).values
        sc_types_labels = pd.read_csv(os.path.join(data_dir, dataset_dir, '%s.Y.csv' % str(highly_genes)), header=None)[0].to_numpy()
        sc_batch_labels = pd.read_csv(os.path.join(data_dir, dataset_dir, '%s.batch.csv' % str(highly_genes)), header=None)[0].to_numpy()
        adata, adata_unscaled, adata_cnt = load_preprocessed(sc_cnt_df.values, sc_unscaled, sc_types_labels, sc_cnt_df.index, sc_cnt_df.columns, batch=sc_batch_labels)
    
    return adata, adata_unscaled, adata_cnt


def load_mouse_pancreas(data_dir, highly_genes=2000, generate_files=False):
    dataset_dir = "mouse_pancreas"
    if generate_files:
        files = ['GSM2230761_mouse1_umifm_counts.csv', 'GSM2230762_mouse2_umifm_counts.csv']
        sc_df = pd.read_csv(os.path.join(data_dir, dataset_dir, files[0]), index_col=0)
        sc_anno_df = sc_df.iloc[:, 1]
        sc_df = sc_df.iloc[:, 2:]
        sc_batch = [files[0].split("_")[1]] * len(sc_df.index)
        for f in files[1:]:
            sc_other_df = pd.read_csv(os.path.join(data_dir, dataset_dir, f), index_col=0)
            sc_other_anno_df = sc_other_df.iloc[:, 1]
            sc_other_df = sc_other_df.iloc[:, 2:]
            sc_anno_df = pd.concat([sc_anno_df, sc_other_anno_df])
            sc_df = pd.concat([sc_df, sc_other_df])
            sc_batch += [f.split("_")[1]] * len(sc_other_df.index)
        if sc_df.columns.duplicated().sum() != 0:     
            sc_df = sc_df.groupby(sc_df.columns).sum()
        sc_expression = sc_df.values
        sc_types_set, sc_types_labels = np.unique(sc_anno_df, return_inverse=True)
        sc_batch_set, sc_batch_labels = np.unique(sc_batch, return_inverse=True)  #mouse1_lib1.final_cell_0225
        adata, adata_unscaled, adata_cnt = preprocess(sc_expression, sc_types_labels, sc_df.index, sc_df.columns,
                                                    highly_genes=highly_genes, dataset_dir=os.path.join(data_dir, dataset_dir), 
                                                    generate_files=generate_files, batch=sc_batch_labels)
    else:
        sc_cnt_df = pd.read_csv(os.path.join(data_dir, dataset_dir, '%s.X.count.name.csv' % str(highly_genes)), index_col=0)
        sc_unscaled = pd.read_csv(os.path.join(data_dir, dataset_dir, '%s.X.name.csv' % str(highly_genes)), index_col=0).values
        sc_types_labels = pd.read_csv(os.path.join(data_dir, dataset_dir, '%s.Y.csv' % str(highly_genes)), header=None)[0].to_numpy()
        sc_batch_labels = pd.read_csv(os.path.join(data_dir, dataset_dir, '%s.batch.csv' % str(highly_genes)), header=None)[0].to_numpy()
        adata, adata_unscaled, adata_cnt = load_preprocessed(sc_cnt_df.values, sc_unscaled, sc_types_labels, sc_cnt_df.index, sc_cnt_df.columns, batch=sc_batch_labels)
    
    return adata, adata_unscaled, adata_cnt

def load_mouse_blastomeres(data_dir, highly_genes=4000, generate_files=False):
    dataset_dir = "mouse_blastomeres"
    if generate_files:
        sc_df = pd.read_table(os.path.join(data_dir, dataset_dir, "Goolam_et_al_2015_count_table.tsv"), index_col=0)
        sc_anno = [cell_name.split('_')[0] for cell_name in sc_df.columns]
        sc_anno = [cell_type if cell_type.endswith('cell') else cell_type + '-4cell' for cell_type in sc_anno]
        if sc_df.index.duplicated().sum() != 0:     
            sc_df = sc_df.groupby(sc_df.index).sum()
        sc_expression = sc_df.T.values
        sc_types_set, sc_types_labels = np.unique(sc_anno, return_inverse=True)
        adata, adata_unscaled, adata_cnt = preprocess(sc_expression, sc_types_labels, sc_df.columns, sc_df.index, 
                                                    highly_genes=highly_genes, dataset_dir=os.path.join(data_dir, dataset_dir), generate_files=generate_files)
    else:
        sc_cnt_df = pd.read_csv(os.path.join(data_dir, dataset_dir, '%s.X.count.name.csv' % str(highly_genes)), index_col=0)
        sc_unscaled = pd.read_csv(os.path.join(data_dir, dataset_dir, '%s.X.name.csv' % str(highly_genes)), index_col=0).values
        sc_types_labels = pd.read_csv(os.path.join(data_dir, dataset_dir, '%s.Y.csv' % str(highly_genes)), header=None)[0].to_numpy()
        adata, adata_unscaled, adata_cnt = load_preprocessed(sc_cnt_df.values, sc_unscaled, sc_types_labels, sc_cnt_df.index, sc_cnt_df.columns)
    
    return adata, adata_unscaled, adata_cnt

def load_human_cell_line(data_dir, highly_genes=5000, generate_files=False, kind="Cell_Line_FPKM"):
    dataset_dir = "human_cell_line"
    if generate_files:
        if kind == "Cell_Line_COUNT":
            file = os.path.join(data_dir, dataset_dir, "GSE81861_Cell_Line_COUNT.csv.gz")
        elif kind == "Cell_Line_FPKM":
            file = os.path.join(data_dir, dataset_dir, "GSE81861_Cell_Line_FPKM.csv.gz")
        sc_df = pd.read_csv(file, index_col=0)
        sc_anno = [cell_name.split("__")[1] for cell_name in sc_df.columns]
        if sc_df.index.duplicated().sum() != 0:     
            sc_df = sc_df.groupby(sc_df.index).sum()
        sc_expression = sc_df.T.values
        sc_types_set, sc_types_labels = np.unique(sc_anno, return_inverse=True)
        adata, adata_unscaled, adata_cnt = preprocess(sc_expression, sc_types_labels, sc_df.columns, sc_df.index, 
                                                    highly_genes=highly_genes, dataset_dir=os.path.join(data_dir, dataset_dir), generate_files=generate_files)
    else:
        sc_cnt_df = pd.read_csv(os.path.join(data_dir, dataset_dir, '%s.X.count.name.csv' % str(highly_genes)), index_col=0)
        sc_unscaled = pd.read_csv(os.path.join(data_dir, dataset_dir, '%s.X.name.csv' % str(highly_genes)), index_col=0).values
        sc_types_labels = pd.read_csv(os.path.join(data_dir, dataset_dir, '%s.Y.csv' % str(highly_genes)), header=None)[0].to_numpy()
        adata, adata_unscaled, adata_cnt = load_preprocessed(sc_cnt_df.values, sc_unscaled, sc_types_labels, sc_cnt_df.index, sc_cnt_df.columns)
    
    return adata, adata_unscaled, adata_cnt

def load_human_colorectal_cancer(data_dir, highly_genes=5000, generate_files=False, kind="NM_all_FPKM"):
    # available kind: "NM_all_COUNT" "NM_all_FPKM" "NM_epithelial_all_COUNT" "NM_epithelial_FPKM"
    #     "tumor_all_COUNT" "tumor_all_FPKM" "tumor_epithelial_all_COUNT" "tumor_epithelial_FPKM"
    dataset_dir = "human_colorectal_cancer"
    if generate_files:
        file = os.path.join(data_dir, dataset_dir, 
                            "GSE81861_CRC_" + '_'.join(kind.split('_')[:2]) + 
                            "_cells_" + kind.split('_')[2] + ".csv.gz")
        sc_df = pd.read_csv(file, index_col=0)
        sc_anno = [cell_name.split("__")[1] for cell_name in sc_df.columns]
        sc_batch= [cell_name.split("__")[1] for cell_name in sc_df.columns]
        if sc_df.index.duplicated().sum() != 0:     
            sc_df = sc_df.groupby(sc_df.index).sum()
        sc_expression = sc_df.T.values
        sc_types_set, sc_types_labels = np.unique(sc_anno, return_inverse=True)
        sc_batch_set, sc_batch_labels = np.unique(sc_batch, return_inverse=True)
        adata, adata_unscaled, adata_cnt = preprocess(sc_expression, sc_types_labels, sc_df.columns, sc_df.index, 
                                                    highly_genes=highly_genes, dataset_dir=os.path.join(data_dir, dataset_dir), 
                                                    generate_files=generate_files, batch=sc_batch_labels)
    else:
        sc_cnt_df = pd.read_csv(os.path.join(data_dir, dataset_dir, '%s.X.count.name.csv' % str(highly_genes)), index_col=0)
        sc_unscaled = pd.read_csv(os.path.join(data_dir, dataset_dir, '%s.X.name.csv' % str(highly_genes)), index_col=0).values
        sc_types_labels = pd.read_csv(os.path.join(data_dir, dataset_dir, '%s.Y.csv' % str(highly_genes)), header=None)[0].to_numpy()
        sc_batch_labels = pd.read_csv(os.path.join(data_dir, dataset_dir, '%s.batch.csv' % str(highly_genes)), header=None)[0].to_numpy()
        adata, adata_unscaled, adata_cnt = load_preprocessed(sc_cnt_df.values, sc_unscaled, sc_types_labels, sc_cnt_df.index, sc_cnt_df.columns, batch=sc_batch_labels)
    

    return adata, adata_unscaled, adata_cnt

def load_mouse_embryos(data_dir, highly_genes=2000, generate_files=False, type_level=1):
    # type_level 1: telescope
    # type_level 2: microscope
    dataset_dir = 'mouse_embryos'
    if generate_files:
        sc_df = pd.read_table(os.path.join(data_dir, dataset_dir, "GSE57249_fpkm.txt.gz"), index_col=0)
        sc_anno_df = pd.read_table(os.path.join(data_dir, dataset_dir, "mouse_embryos_annotation.txt"), index_col=0, header=None)
        if type_level == 1:
            sc_anno = [cell_name.split(' ')[0] for cell_name in sc_anno_df.iloc[:, 0]]
        elif type_level == 2:
            sc_anno = ['-'.join(cell_name.split(' ')[:-1]) for cell_name in sc_anno_df.iloc[:, 0]]
        if sc_df.index.duplicated().sum() != 0:     
            sc_df = sc_df.groupby(sc_df.index).sum()
        sc_expression = sc_df.T.values
        sc_types_set, sc_types_labels = np.unique(sc_anno, return_inverse=True)
        adata, adata_unscaled, adata_cnt = preprocess(sc_expression, sc_types_labels, sc_df.columns, sc_df.index, 
                                                    highly_genes=highly_genes, dataset_dir=os.path.join(data_dir, dataset_dir), generate_files=generate_files)
    else:
        sc_cnt_df = pd.read_csv(os.path.join(data_dir, dataset_dir, '%s.X.count.name.csv' % str(highly_genes)), index_col=0)
        sc_unscaled = pd.read_csv(os.path.join(data_dir, dataset_dir, '%s.X.name.csv' % str(highly_genes)), index_col=0).values
        sc_types_labels = pd.read_csv(os.path.join(data_dir, dataset_dir, '%s.Y.csv' % str(highly_genes)), header=None)[0].to_numpy()
        adata, adata_unscaled, adata_cnt = load_preprocessed(sc_cnt_df.values, sc_unscaled, sc_types_labels, sc_cnt_df.index, sc_cnt_df.columns)
    
    return adata, adata_unscaled, adata_cnt

def load_mouse_pancreatic_circulating_tumor(data_dir, highly_genes=3000, generate_files=False):
    dataset_dir = "mouse_pancreatic_circulating_tumor"
    if generate_files:
        sc_df = pd.read_table(os.path.join(data_dir, dataset_dir, "GSE51372_readCounts.txt.gz"), index_col=0, low_memory=False)
        sc_df = sc_df.drop(["Entrez GeneID", "uniGene", "symbol", "name", "mm9 knownGene ID"], axis=1)
        sc_anno = [cell_name.split('-')[0] for cell_name in sc_df.columns]
        if sc_df.index.duplicated().sum() != 0:     
            sc_df = sc_df.groupby(sc_df.index).sum()
        sc_expression = sc_df.T.values
        sc_types_set, sc_types_labels = np.unique(sc_anno, return_inverse=True)
        adata, adata_unscaled, adata_cnt = preprocess(sc_expression, sc_types_labels, sc_df.columns, sc_df.index, 
                                                    highly_genes=highly_genes, dataset_dir=os.path.join(data_dir, dataset_dir), generate_files=generate_files)
    else:
        sc_cnt_df = pd.read_csv(os.path.join(data_dir, dataset_dir, '%s.X.count.name.csv' % str(highly_genes)), index_col=0)
        sc_unscaled = pd.read_csv(os.path.join(data_dir, dataset_dir, '%s.X.name.csv' % str(highly_genes)), index_col=0).values
        sc_types_labels = pd.read_csv(os.path.join(data_dir, dataset_dir, '%s.Y.csv' % str(highly_genes)), header=None)[0].to_numpy()
        adata, adata_unscaled, adata_cnt = load_preprocessed(sc_cnt_df.values, sc_unscaled, sc_types_labels, sc_cnt_df.index, sc_cnt_df.columns)
    
    return adata, adata_unscaled, adata_cnt

def load_human_embryos(data_dir, highly_genes=2000, generate_files=False):
    dataset_dir = "human_embryos"
    if generate_files:
        sc_df = pd.read_csv(os.path.join(data_dir, dataset_dir, "GSE36552_RAW_expression.csv"), index_col=0)
        sc_anno_df = pd.read_csv(os.path.join(data_dir, dataset_dir, "GSE36552_RAW_metadata.txt"), index_col=0)
        sc_df = sc_df.fillna(0)
        if sc_df.columns.duplicated().sum() != 0:     
            sc_df = sc_df.groupby(sc_df.columns).sum()
        sc_expression = sc_df.values
        sc_types_set, sc_types_labels = np.unique(sc_anno_df, return_inverse=True)
        adata, adata_unscaled, adata_cnt = preprocess(sc_expression, sc_types_labels, sc_df.index, sc_df.columns, 
                                                    highly_genes=highly_genes, dataset_dir=os.path.join(data_dir, dataset_dir), generate_files=generate_files)
    else:
        sc_cnt_df = pd.read_csv(os.path.join(data_dir, dataset_dir, '%s.X.count.name.csv' % str(highly_genes)), index_col=0)
        sc_unscaled = pd.read_csv(os.path.join(data_dir, dataset_dir, '%s.X.name.csv' % str(highly_genes)), index_col=0).values
        sc_types_labels = pd.read_csv(os.path.join(data_dir, dataset_dir, '%s.Y.csv' % str(highly_genes)), header=None)[0].to_numpy()
        adata, adata_unscaled, adata_cnt = load_preprocessed(sc_cnt_df.values, sc_unscaled, sc_types_labels, sc_cnt_df.index, sc_cnt_df.columns)
        
    return adata, adata_unscaled, adata_cnt

def load_mars_skin(data_dir, highly_genes=2000, generate_files=False):
    dataset_dir = 'mars_skin'
    if generate_files:
        with open(os.path.join(data_dir, dataset_dir, dataset_dir + '.pickle'), 'rb') as in_file:
            picpik = pickle.load(in_file)
        sc_expression = picpik[0].todense()
        sc_anno = picpik[1]
        sc_types_set, sc_types_labels = np.unique(sc_anno, return_inverse=True)
        adata, adata_unscaled, adata_cnt = preprocess(sc_expression, sc_types_labels, np.arange(len(sc_anno)), picpik[2],
                                                    highly_genes=highly_genes, dataset_dir=os.path.join(data_dir, dataset_dir), generate_files=generate_files)
    else:
        sc_cnt_df = pd.read_csv(os.path.join(data_dir, dataset_dir, '%s.X.count.name.csv' % str(highly_genes)), index_col=0)
        sc_unscaled = pd.read_csv(os.path.join(data_dir, dataset_dir, '%s.X.name.csv' % str(highly_genes)), index_col=0).values
        sc_types_labels = pd.read_csv(os.path.join(data_dir, dataset_dir, '%s.Y.csv' % str(highly_genes)), header=None)[0].to_numpy()
        adata, adata_unscaled, adata_cnt = load_preprocessed(sc_cnt_df.values, sc_unscaled, sc_types_labels, sc_cnt_df.index, sc_cnt_df.columns)
        
    return adata, adata_unscaled, adata_cnt

def load_mars_limb_muscle(data_dir, highly_genes=2000, generate_files=False):
    dataset_dir = 'mars_limb_muscle'
    if generate_files:
        with open(os.path.join(data_dir, dataset_dir, dataset_dir + '.pickle'), 'rb') as in_file:
            picpik = pickle.load(in_file)
        sc_expression = picpik[0].todense()
        sc_anno = picpik[1]
        sc_types_set, sc_types_labels = np.unique(sc_anno, return_inverse=True)
        adata, adata_unscaled, adata_cnt = preprocess(sc_expression, sc_types_labels, np.arange(len(sc_anno)), picpik[2],
                                                    highly_genes=highly_genes, dataset_dir=os.path.join(data_dir, dataset_dir), generate_files=generate_files)
    else:
        sc_cnt_df = pd.read_csv(os.path.join(data_dir, dataset_dir, '%s.X.count.name.csv' % str(highly_genes)), index_col=0)
        sc_unscaled = pd.read_csv(os.path.join(data_dir, dataset_dir, '%s.X.name.csv' % str(highly_genes)), index_col=0).values
        sc_types_labels = pd.read_csv(os.path.join(data_dir, dataset_dir, '%s.Y.csv' % str(highly_genes)), header=None)[0].to_numpy()
        adata, adata_unscaled, adata_cnt = load_preprocessed(sc_cnt_df.values, sc_unscaled, sc_types_labels, sc_cnt_df.index, sc_cnt_df.columns)
        
    return adata, adata_unscaled, adata_cnt

def load_mars_spleen(data_dir, highly_genes=2000, generate_files=False):
    dataset_dir = 'mars_spleen'
    if generate_files:
        with open(os.path.join(data_dir, dataset_dir, dataset_dir + '.pickle'), 'rb') as in_file:
            picpik = pickle.load(in_file)
        sc_expression = picpik[0].todense()
        sc_anno = picpik[1]
        sc_types_set, sc_types_labels = np.unique(sc_anno, return_inverse=True)
        adata, adata_unscaled, adata_cnt = preprocess(sc_expression, sc_types_labels, np.arange(len(sc_anno)), picpik[2],
                                                    highly_genes=highly_genes, dataset_dir=os.path.join(data_dir, dataset_dir), generate_files=generate_files)
    else:
        sc_cnt_df = pd.read_csv(os.path.join(data_dir, dataset_dir, '%s.X.count.name.csv' % str(highly_genes)), index_col=0)
        sc_unscaled = pd.read_csv(os.path.join(data_dir, dataset_dir, '%s.X.name.csv' % str(highly_genes)), index_col=0).values
        sc_types_labels = pd.read_csv(os.path.join(data_dir, dataset_dir, '%s.Y.csv' % str(highly_genes)), header=None)[0].to_numpy()
        adata, adata_unscaled, adata_cnt = load_preprocessed(sc_cnt_df.values, sc_unscaled, sc_types_labels, sc_cnt_df.index, sc_cnt_df.columns)
    
    return adata, adata_unscaled, adata_cnt

def load_mars_trachea(data_dir, highly_genes=2000, generate_files=False):
    dataset_dir = 'mars_trachea'
    if generate_files:
        with open(os.path.join(data_dir, dataset_dir, dataset_dir + '.pickle'), 'rb') as in_file:
            picpik = pickle.load(in_file)
        sc_expression = picpik[0].todense()
        sc_anno = picpik[1]
        sc_types_set, sc_types_labels = np.unique(sc_anno, return_inverse=True)
        adata, adata_unscaled, adata_cnt = preprocess(sc_expression, sc_types_labels, np.arange(len(sc_anno)), picpik[2],
                                                    highly_genes=highly_genes, dataset_dir=os.path.join(data_dir, dataset_dir), generate_files=generate_files)
    else:
        sc_cnt_df = pd.read_csv(os.path.join(data_dir, dataset_dir, '%s.X.count.name.csv' % str(highly_genes)), index_col=0)
        sc_unscaled = pd.read_csv(os.path.join(data_dir, dataset_dir, '%s.X.name.csv' % str(highly_genes)), index_col=0).values
        sc_types_labels = pd.read_csv(os.path.join(data_dir, dataset_dir, '%s.Y.csv' % str(highly_genes)), header=None)[0].to_numpy()
        adata, adata_unscaled, adata_cnt = load_preprocessed(sc_cnt_df.values, sc_unscaled, sc_types_labels, sc_cnt_df.index, sc_cnt_df.columns)
    
    return adata, adata_unscaled, adata_cnt

def load_mars_tongue(data_dir, highly_genes=2000, generate_files=False):
    dataset_dir = 'mars_tongue'
    if generate_files:
        with open(os.path.join(data_dir, dataset_dir, dataset_dir + '.pickle'), 'rb') as in_file:
            picpik = pickle.load(in_file)
        sc_expression = picpik[0].todense()
        sc_anno = picpik[1]
        sc_types_set, sc_types_labels = np.unique(sc_anno, return_inverse=True)
        adata, adata_unscaled, adata_cnt = preprocess(sc_expression, sc_types_labels, np.arange(len(sc_anno)), picpik[2],
                                                    highly_genes=highly_genes, dataset_dir=os.path.join(data_dir, dataset_dir), generate_files=generate_files)
    else:
        sc_cnt_df = pd.read_csv(os.path.join(data_dir, dataset_dir, '%s.X.count.name.csv' % str(highly_genes)), index_col=0)
        sc_unscaled = pd.read_csv(os.path.join(data_dir, dataset_dir, '%s.X.name.csv' % str(highly_genes)), index_col=0).values
        sc_types_labels = pd.read_csv(os.path.join(data_dir, dataset_dir, '%s.Y.csv' % str(highly_genes)), header=None)[0].to_numpy()
        adata, adata_unscaled, adata_cnt = load_preprocessed(sc_cnt_df.values, sc_unscaled, sc_types_labels, sc_cnt_df.index, sc_cnt_df.columns)
    
    return adata, adata_unscaled, adata_cnt

def load_mars_thymus(data_dir, highly_genes=2000, generate_files=False):
    dataset_dir = 'mars_thymus'
    if generate_files:
        with open(os.path.join(data_dir, dataset_dir, dataset_dir + '.pickle'), 'rb') as in_file:
            picpik = pickle.load(in_file)
        sc_expression = picpik[0].todense()
        sc_anno = picpik[1]
        sc_types_set, sc_types_labels = np.unique(sc_anno, return_inverse=True)
        adata, adata_unscaled, adata_cnt = preprocess(sc_expression, sc_types_labels, np.arange(len(sc_anno)), picpik[2],
                                                    highly_genes=highly_genes, dataset_dir=os.path.join(data_dir, dataset_dir), generate_files=generate_files)
    else:
        sc_cnt_df = pd.read_csv(os.path.join(data_dir, dataset_dir, '%s.X.count.name.csv' % str(highly_genes)), index_col=0)
        sc_unscaled = pd.read_csv(os.path.join(data_dir, dataset_dir, '%s.X.name.csv' % str(highly_genes)), index_col=0).values
        sc_types_labels = pd.read_csv(os.path.join(data_dir, dataset_dir, '%s.Y.csv' % str(highly_genes)), header=None)[0].to_numpy()
        adata, adata_unscaled, adata_cnt = load_preprocessed(sc_cnt_df.values, sc_unscaled, sc_types_labels, sc_cnt_df.index, sc_cnt_df.columns)
    
    return adata, adata_unscaled, adata_cnt

def load_mars_bladder(data_dir, highly_genes=2000, generate_files=False):
    dataset_dir = 'mars_bladder'
    if generate_files:
        with open(os.path.join(data_dir, dataset_dir, dataset_dir + '.pickle'), 'rb') as in_file:
            picpik = pickle.load(in_file)
        sc_expression = picpik[0].todense()
        sc_anno = picpik[1]
        sc_types_set, sc_types_labels = np.unique(sc_anno, return_inverse=True)
        adata, adata_unscaled, adata_cnt = preprocess(sc_expression, sc_types_labels, np.arange(len(sc_anno)), picpik[2],
                                                    highly_genes=highly_genes, dataset_dir=os.path.join(data_dir, dataset_dir), generate_files=generate_files)
    else:
        sc_cnt_df = pd.read_csv(os.path.join(data_dir, dataset_dir, '%s.X.count.name.csv' % str(highly_genes)), index_col=0)
        sc_unscaled = pd.read_csv(os.path.join(data_dir, dataset_dir, '%s.X.name.csv' % str(highly_genes)), index_col=0).values
        sc_types_labels = pd.read_csv(os.path.join(data_dir, dataset_dir, '%s.Y.csv' % str(highly_genes)), header=None)[0].to_numpy()
        adata, adata_unscaled, adata_cnt = load_preprocessed(sc_cnt_df.values, sc_unscaled, sc_types_labels, sc_cnt_df.index, sc_cnt_df.columns)
    
    return adata, adata_unscaled, adata_cnt

def load_mars_liver(data_dir, highly_genes=2000, generate_files=False):
    dataset_dir = 'mars_liver'
    if generate_files:
        with open(os.path.join(data_dir, dataset_dir, dataset_dir + '.pickle'), 'rb') as in_file:
            picpik = pickle.load(in_file)
        sc_expression = picpik[0].todense()
        sc_anno = picpik[1]
        sc_types_set, sc_types_labels = np.unique(sc_anno, return_inverse=True)
        adata, adata_unscaled, adata_cnt = preprocess(sc_expression, sc_types_labels, np.arange(len(sc_anno)), picpik[2],
                                                    highly_genes=highly_genes, dataset_dir=os.path.join(data_dir, dataset_dir), generate_files=generate_files)
    else:
        sc_cnt_df = pd.read_csv(os.path.join(data_dir, dataset_dir, '%s.X.count.name.csv' % str(highly_genes)), index_col=0)
        sc_unscaled = pd.read_csv(os.path.join(data_dir, dataset_dir, '%s.X.name.csv' % str(highly_genes)), index_col=0).values
        sc_types_labels = pd.read_csv(os.path.join(data_dir, dataset_dir, '%s.Y.csv' % str(highly_genes)), header=None)[0].to_numpy()
        adata, adata_unscaled, adata_cnt = load_preprocessed(sc_cnt_df.values, sc_unscaled, sc_types_labels, sc_cnt_df.index, sc_cnt_df.columns)
    
    return adata, adata_unscaled, adata_cnt

def load_mars_mammary_gland(data_dir, highly_genes=2000, generate_files=False):
    dataset_dir = 'mars_mammary_gland'
    if generate_files:
        with open(os.path.join(data_dir, dataset_dir, dataset_dir + '.pickle'), 'rb') as in_file:
            picpik = pickle.load(in_file)
        sc_expression = picpik[0].todense()
        sc_anno = picpik[1]
        sc_types_set, sc_types_labels = np.unique(sc_anno, return_inverse=True)
        adata, adata_unscaled, adata_cnt = preprocess(sc_expression, sc_types_labels, np.arange(len(sc_anno)), picpik[2],
                                                    highly_genes=highly_genes, dataset_dir=os.path.join(data_dir, dataset_dir), generate_files=generate_files)
    else:
        sc_cnt_df = pd.read_csv(os.path.join(data_dir, dataset_dir, '%s.X.count.name.csv' % str(highly_genes)), index_col=0)
        sc_unscaled = pd.read_csv(os.path.join(data_dir, dataset_dir, '%s.X.name.csv' % str(highly_genes)), index_col=0).values
        sc_types_labels = pd.read_csv(os.path.join(data_dir, dataset_dir, '%s.Y.csv' % str(highly_genes)), header=None)[0].to_numpy()
        adata, adata_unscaled, adata_cnt = load_preprocessed(sc_cnt_df.values, sc_unscaled, sc_types_labels, sc_cnt_df.index, sc_cnt_df.columns)
    
    return adata, adata_unscaled, adata_cnt

# def load_resnet_datasets(data_dir, highly_genes=2000, generate_files=False):
#     dataset_dir = "data"
#     if generate_files:
#         files = ['pancreas_inDrop.txt.gz', 
#                  'pancreas_multi_celseq_expression_matrix.txt.gz',
#                  'pancreas_multi_celseq2_expression_matrix.txt.gz',
#                  #'pancreas_multi_fluidigmc1_expression_matrix.txt.gz',
#                  #'pancreas_multi_smartseq2_expression_matrix.txt.gz'
#                  ]
#         sc_df = pd.read_table(os.path.join(data_dir, dataset_dir, "pancreas", files[0]),index_col=0, low_memory=False).T
#         #sc_df.columns = sc_df.columns.str.strip()
#         gt_idx = [ i for i, s in enumerate(np.sum(sc_df != 0, axis=1))
#                if s >= 600 ]
        

#         sc_df = sc_df.iloc[gt_idx, :]
#         sc_batch = [files[0]] * len(sc_df.index)

#         for f in files[1:]:
#             sc_other_df = pd.read_table(os.path.join(data_dir, dataset_dir, "pancreas", f),index_col=0, low_memory=False).T
#             gt_idx = [ i for i, s in enumerate(np.sum(sc_other_df != 0, axis=1))
#                if s >= 600 ]
#             sc_other_df = sc_other_df.iloc[gt_idx, :]
#             sc_df = pd.concat([sc_df, sc_other_df], axis = 0)
#             sc_batch += [f] * len(sc_other_df.index)

#         sc_anno_df = pd.read_table(os.path.join(data_dir, dataset_dir, "cell_labels", "pancreas_cluster.txt"),header=None, low_memory=False)
#         sc_anno_df = sc_anno_df.loc[0:(sc_df.shape[0] - 1),:]
#         if sc_df.columns.duplicated().sum() != 0:     
#              sc_df = sc_df.groupby(sc_df.columns).sum()
#         sc_expression = sc_df.values

#         sc_types_set, sc_types_labels = np.unique(sc_anno_df, return_inverse=True)
#         sc_batch_set, sc_batch_labels = np.unique(sc_batch, return_inverse=True) 
#         adata, adata_unscaled, adata_cnt = preprocess(sc_expression, sc_types_labels, sc_df.index, sc_df.columns,
#                                                     highly_genes=highly_genes, dataset_dir=os.path.join(data_dir, dataset_dir), 
#                                                     generate_files=generate_files, batch=sc_batch_labels)
#     else:
#         sc_cnt_df = pd.read_csv(os.path.join(data_dir, dataset_dir, '%s.X.count.name.csv' % str(highly_genes)), index_col=0)
#         sc_unscaled = pd.read_csv(os.path.join(data_dir, dataset_dir, '%s.X.name.csv' % str(highly_genes)), index_col=0).values
#         sc_types_labels = pd.read_csv(os.path.join(data_dir, dataset_dir, '%s.Y.csv' % str(highly_genes)), header=None)[0].to_numpy()
#         sc_batch_labels = pd.read_csv(os.path.join(data_dir, dataset_dir, '%s.batch.csv' % str(highly_genes)), header=None)[0].to_numpy()
#         adata, adata_unscaled, adata_cnt = load_preprocessed(sc_cnt_df.values, sc_unscaled, sc_types_labels, sc_cnt_df.index, sc_cnt_df.columns, batch=sc_batch_labels)
    
#     return adata, adata_unscaled, adata_cnt

def load_simdata_datasets(data_dir, highly_genes=2000, generate_files=False):
    dataset_dir = "simulation_data"
    if generate_files:
        files = [
        'splatter_benchmark_simul1_dropout_005_b1_500_b2_900',
        'splatter_benchmark_simul2_dropout_025_b1_500_b2_900',
        'splatter_benchmark_simul3_dropout_005_b1_500_b2_450',
        'splatter_benchmark_simul4_dropout_025_b1_500_b2_450',
        'splatter_benchmark_simul5_dropout_005_b1_80_b2_400',
        'splatter_benchmark_simul6_dropout_025_b1_80_b2_400'
        ]

        sc_df = pd.read_csv(os.path.join(data_dir, dataset_dir, files[0],'counts.csv'),index_col=0, low_memory=False).T
        sc_anno = pd.read_csv(os.path.join(data_dir, dataset_dir, files[1],'cells_info.csv'), low_memory=False)
        sc_anno_df = sc_anno.iloc[:,3]
        sc_batch   = sc_anno.iloc[:,2]

        if sc_df.columns.duplicated().sum() != 0:     
             sc_df = sc_df.groupby(sc_df.columns).sum()
        sc_expression = sc_df.values

        sc_types_set, sc_types_labels = np.unique(sc_anno_df, return_inverse=True)
        sc_batch_set, sc_batch_labels = np.unique(sc_batch, return_inverse=True) 
        adata, adata_unscaled, adata_cnt = preprocess(sc_expression, sc_types_labels, sc_df.index, sc_df.columns,
                                                    highly_genes=highly_genes, dataset_dir=os.path.join(data_dir, dataset_dir), 
                                                    generate_files=generate_files, batch=sc_batch_labels)
    else:
        sc_cnt_df = pd.read_csv(os.path.join(data_dir, dataset_dir, '%s.X.count.name.csv' % str(highly_genes)), index_col=0)
        sc_unscaled = pd.read_csv(os.path.join(data_dir, dataset_dir, '%s.X.name.csv' % str(highly_genes)), index_col=0).values
        sc_types_labels = pd.read_csv(os.path.join(data_dir, dataset_dir, '%s.Y.csv' % str(highly_genes)), header=None)[0].to_numpy()
        sc_batch_labels = pd.read_csv(os.path.join(data_dir, dataset_dir, '%s.batch.csv' % str(highly_genes)), header=None)[0].to_numpy()
        adata, adata_unscaled, adata_cnt = load_preprocessed(sc_cnt_df.values, sc_unscaled, sc_types_labels, sc_cnt_df.index, sc_cnt_df.columns, batch=sc_batch_labels)
    
    return adata, adata_unscaled, adata_cnt

def load_cellbench_datasets(data_dir, highly_genes = 2000, generate_files = False):
    dataset_dir = "cellbench_kolod_pollen"
    if generate_files:
        adata = sc.read_h5ad(os.path.join(data_dir, dataset_dir, "kolod_pollen_bench.h5ad"))
        sc_expression = adata.X.astype(np.float32)
        sc_types_labels = np.array(adata.obs['ground_truth'], dtype = np.int64)
        sc_exp_labels = adata.obs['experiment']
        cells_name = adata.obs_names
        genes_name = adata.var_names
        
        adata, adata_unscaled, adata_cnt = preprocess(sc_expression, sc_types_labels, cells_name, genes_name,
                                                highly_genes=highly_genes, dataset_dir=os.path.join(data_dir, dataset_dir), 
                                                exp=sc_exp_labels, generate_files=generate_files)
    else:
        sc_cnt_df = pd.read_csv(os.path.join(data_dir, dataset_dir, '%s.X.count.name.csv' % str(highly_genes)), index_col=0)
        sc_unscaled = pd.read_csv(os.path.join(data_dir, dataset_dir, '%s.X.name.csv' % str(highly_genes)), index_col=0).values
        sc_types_labels = pd.read_csv(os.path.join(data_dir, dataset_dir, '%s.Y.csv' % str(highly_genes)), header=None)[0].to_numpy()
        sc_exp_labels = pd.read_csv(os.path.join(data_dir, dataset_dir, '%s.experiment.csv' % str(highly_genes)), header=None)[0].to_numpy()
        adata, adata_unscaled, adata_cnt = load_preprocessed(sc_cnt_df.values, sc_unscaled, sc_types_labels, sc_cnt_df.index, sc_cnt_df.columns, exp=sc_exp_labels)
    return adata, adata_unscaled, adata_cnt