#!/usr/bin/env python

import argparse
import sys
from collections import Counter

import pandas as pd
import scanpy as sc

#from nmf_network import *
from att_network import *
from data_load import *

if __name__ == "__main__":
    gpu_option= "0"
    parser = argparse.ArgumentParser()
    parser.add_argument('--dim',  type=float,default=400 ,dest='dim')
    parser.add_argument('--lr',  type=int,default=1e-4 ,dest='learning_rate')
    parser.add_argument('--epochs',  type=int,default=30 ,dest='epochs')
    parser.add_argument('--batch-size', type=int, default=256, dest='batch_size')
    parser.add_argument('--inherit-centroids', action='store_true', dest='inherit_centroids')
    parser.add_argument('--L2-normalization', action='store_true', dest='L2')
    parser.add_argument('--lambda_b',  type=float,default=0 ,dest='lambda_b')
    parser.add_argument('--lambda_c',  type=float,default=0 ,dest='lambda_c')
    parser.add_argument('--lambda_d',  type=float,default=0 ,dest='lambda_d')
    parser.add_argument('--lambda_e',  type=float,default=0 ,dest='lambda_e')
    parser.add_argument('--linkage', type=str, default="ward", dest='linkage')
    parser.add_argument('--generate-files', action='store_true', dest='generate_files')
    parser.add_argument('--hg', type=int,default=0.1 ,dest='highly_genes_num')
    parser.add_argument('--data-dir',  type=str,default="../cgi_datasets" ,dest='data_dir')
    parser.add_argument('--method',  type=str,default="KMeans" ,dest='method')
    parser.add_argument('--seed',  type=int,default=0 ,dest="seed")
    parser.add_argument('--bench-dataset',  type=str,default="human_pancreas" ,dest='dataset_dir')
    parser.add_argument('--output-dir',  type=str,default="output" ,dest='output_dir')
    parser.add_argument('--output-ari-file',  type=str,default="ari.txt" ,dest='output_ari')
    args = parser.parse_args()
    learning_rate=args.learning_rate
    batch_size = args.batch_size
    highly_genes_num = args.highly_genes_num
    dim = args.dim
    inherit_centroids = args.inherit_centroids
    L2 = args.L2
    linkage = args.linkage
    generate_files = args.generate_files
    epochs = args.epochs
    data_dir = args.data_dir
    seed = args.seed
    method = args.method
    dataset_dir = args.dataset_dir
    output_dir = args.output_dir
    output_ari = args.output_ari
    lambda_b = args.lambda_b
    lambda_c = args.lambda_c
    lambda_d = args.lambda_d
    lambda_e = args.lambda_e
    np.random.seed(seed)
    tf.compat.v1.set_random_seed(seed)
    random.seed(seed)
    
    if dataset_dir == "human_pancreas":
        adata, adata_unscaled, adata_cnt = load_human_pancreas(data_dir, highly_genes_num, generate_files)
    elif dataset_dir == "human_progenitor":
        adata, adata_unscaled, adata_cnt = load_human_progenitor(data_dir, highly_genes_num, generate_files)
    elif dataset_dir == "mouse_cortex":
        adata, adata_unscaled, adata_cnt = load_mouse_cortex(data_dir, highly_genes_num, generate_files)
    elif dataset_dir == "human_melanoma":
        adata, adata_unscaled, adata_cnt = load_human_melanoma(data_dir, highly_genes_num, generate_files)
    elif dataset_dir == "mouse_stem":
        adata, adata_unscaled, adata_cnt = load_mouse_stem(data_dir, highly_genes_num, generate_files)
    elif dataset_dir == "mouse_pancreas":
        adata, adata_unscaled, adata_cnt = load_mouse_pancreas(data_dir, highly_genes_num, generate_files)
    elif dataset_dir == "mouse_blastomeres":
        adata, adata_unscaled, adata_cnt = load_mouse_blastomeres(data_dir, highly_genes_num, generate_files)
    elif dataset_dir == "human_cell_line":
        adata, adata_unscaled, adata_cnt = load_human_cell_line(data_dir, highly_genes_num, generate_files)
    elif dataset_dir == "human_colorectal_cancer":
        adata, adata_unscaled, adata_cnt = load_human_colorectal_cancer(data_dir, highly_genes_num, generate_files)
    elif dataset_dir == "mouse_embryos":
        adata, adata_unscaled, adata_cnt = load_mouse_embryos(data_dir, highly_genes_num, generate_files)
    elif dataset_dir == "mouse_pancreatic_circulating_tumor":
        adata, adata_unscaled, adata_cnt = load_mouse_pancreatic_circulating_tumor(data_dir, highly_genes_num, generate_files)
    elif dataset_dir == "human_embryos":
        adata, adata_unscaled, adata_cnt = load_human_embryos(data_dir, highly_genes_num, generate_files)
    elif dataset_dir == "mars_skin":
        adata, adata_unscaled, adata_cnt = load_mars_skin(data_dir, highly_genes_num, generate_files)
    elif dataset_dir == "mars_limb_muscle":
        adata, adata_unscaled, adata_cnt = load_mars_limb_muscle(data_dir, highly_genes_num, generate_files)
    elif dataset_dir == "mars_spleen":
        adata, adata_unscaled, adata_cnt = load_mars_spleen(data_dir, highly_genes_num, generate_files)
    elif dataset_dir == "mars_trachea":
        adata, adata_unscaled, adata_cnt = load_mars_trachea(data_dir, highly_genes_num, generate_files)
    elif dataset_dir == "mars_tongue":
        adata, adata_unscaled, adata_cnt = load_mars_tongue(data_dir, highly_genes_num, generate_files)
    elif dataset_dir == "mars_thymus":
        adata, adata_unscaled, adata_cnt = load_mars_thymus(data_dir, highly_genes_num, generate_files)
    elif dataset_dir == "mars_bladder":
        adata, adata_unscaled, adata_cnt = load_mars_bladder(data_dir, highly_genes_num, generate_files)
    elif dataset_dir == "mars_liver":
        adata, adata_unscaled, adata_cnt = load_mars_liver(data_dir, highly_genes_num, generate_files)
    elif dataset_dir == "mars_mammary_gland":
        adata, adata_unscaled, adata_cnt = load_mars_mammary_gland(data_dir, highly_genes_num, generate_files)
    elif dataset_dir == "simulation_data":
        adata, adata_unscaled, adata_cnt = load_simdata_datasets(data_dir, highly_genes_num, generate_files)
    elif dataset_dir == "cellbench_kolod_pollen":
        adata, adata_unscaled, adata_cnt = load_cellbench_datasets(data_dir, highly_genes_num, generate_files)
    #split_percent=int(1*len(adata.X))
    split_percent = max(adata.X.shape[0],adata.X.shape[1])
    if dim <= 1 and dim > 0:
        dim = int(adata.X.shape[1] * dim)
    elif dim > 1:
        dim = int(dim)
    else:
        raise
    print("shape[1]",adata.X.shape[1])
    print("dim",dim)
    dims = [adata.X.shape[1], dim]
    tf.compat.v1.disable_eager_execution()
    cluster_num = len(np.unique(adata.obs["cell_groups"]))
    if 'batch' not in adata.obs:
        lambda_d = None
    tf.compat.v1.reset_default_graph()

    print("lambda_d:",lambda_d)
    model = Model(dataset_dir, dims, cluster_num, learning_rate, batch_size, lambda_b, lambda_c,lambda_d,lambda_e, method)

    #target_accuracy, target_ARI, annotated_target_accuracy, target_prediction_matrix = model.train(X, count_X, cellname,  pretrain_epochs, random_seed, gpu_option,unscale_X,test_X,unscale_test_X,test_count_X)
    tic = time.process_time()
    imX, Y_pred, imX_df, select_m_df, select_w_df = model.train(adata, adata_unscaled, adata_cnt, split_percent, epochs, seed, inherit_centroids, L2, linkage, gpu_option)
    toc = time.process_time()
    # with open(os.path.join(data_dir, dataset_dir, output_dir, output_ari), 'a+') as out:
    #     out.write(str(ARI) + '\n')
    cell_type, Y_target = np.unique(adata.obs["cell_groups"][:split_percent], return_inverse=True)
    with open(os.path.join(data_dir, dataset_dir, output_dir, output_ari.replace(".ari", ".cluster")), 'a+') as out:
        np.savetxt(out, Y_pred, fmt="%d")
    with open(os.path.join(data_dir, dataset_dir, output_dir, output_ari.replace(".ari", ".anno")), 'a+') as out:
        np.savetxt(out, Y_target, fmt="%d")
    with open(os.path.join(data_dir, dataset_dir, output_dir, output_ari.replace(".ari", ".time")), 'a+') as out:
        out.write(str(toc - tic) + '\n')
    imX_df.to_csv(os.path.join(data_dir, dataset_dir, output_dir, output_ari.replace(".ari", ".imx")))
    select_m_df.to_csv(os.path.join(data_dir, dataset_dir, output_dir, output_ari.replace(".ari", ".mask")))
    select_w_df.to_csv(os.path.join(data_dir, dataset_dir, output_dir, output_ari.replace(".ari", ".weight")))

    #print(imX.shape)
    #np.savetxt(os.path.join(data_dir, dataset_dir, output_dir, "imputed_X.csv"), np.squeeze(imX, axis=0), delimiter=',')










