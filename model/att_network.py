import math
import os
import pickle
import random
import sys
import time

import fastcluster
import hdbscan
import matplotlib.pyplot as plt
import numpy as np
# from imputeByGans import *
import pandas as pd
import scanpy as sc
import seaborn as sns
import tensorflow as tf
import tensorflow.keras.backend as K
from sklearn.cluster import (DBSCAN, AgglomerativeClustering, KMeans,
                             SpectralBiclustering, SpectralClustering)
from tslearn.clustering import KernelKMeans
from sklearn.decomposition import NMF, PCA
from sklearn.manifold import TSNE
from sklearn.metrics.cluster import *
from sklearn.preprocessing import normalize
# from keras.layers import GaussianNoise, Dense, Activation
from tensorflow.keras import regularizers
# import keras.backend as K
from tensorflow.keras.layers import Activation, Dense, GaussianNoise

from collections import defaultdict
from scipy.spatial import distance

# from dca.api import dca

MeanAct = lambda x: tf.clip_by_value(K.exp(x), 1e-5, 1e6)
DispAct = lambda x: tf.clip_by_value(tf.nn.softplus(x), 1e-4, 1e4)
imAct = lambda x: tf.nn.softplus(x)

#experiment
unannotated_exp = "pollen"


def Y_to_clusters(Y, cnum):
#    cnum=max(Y)+1
    clusters = []
    for i in range(cnum): clusters.append([])
    for i in range(len(Y)):
        clusters[Y[i]].append(i)
    return clusters


def get_rand_mask(mask):
    new_mask = np.zeros(shape=(mask.shape[0], mask.shape[1]))
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i][j] == 0: continue
            if np.random.uniform(0, 1, 1)[0] < 0.5:
                new_mask[i][j] = 1
    return new_mask


def get_mask(X):
    X = np.array(X)
    mask = np.zeros(shape=(X.shape[0], X.shape[1]))
    mask[X == 0] = 0
    mask[X != 0] = 1

    return mask


# define cluster accuracy
def cluster_acc(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    from sklearn.utils.linear_assignment_ import linear_assignment
    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

# transform label vector to label matrix


def label2matrix(label):
    unique_label, label = np.unique(label, return_inverse=True)
    one_hot_label = np.zeros((len(label), len(unique_label)))
    one_hot_label[np.arange(len(label)), label] = 1
    return one_hot_label


def _nan2zero(x):
    return tf.compat.v1.where(tf.math.is_nan(x), tf.zeros_like(x), x)


def _nan2inf(x):
    return tf.compat.v1.where(tf.math.is_nan(x), tf.zeros_like(x)+np.inf, x)


def _nelem(x):
    nelem = tf.reduce_sum(input_tensor=tf.cast(~tf.math.is_nan(x), tf.float32))
    return tf.cast(tf.compat.v1.where(tf.equal(nelem, 0.), 1., nelem), x.dtype)


def _reduce_mean(x):
    nelem = _nelem(x)
    x = _nan2zero(x)
    return tf.divide(tf.reduce_sum(input_tensor=x), nelem)

######Function For Remove Batch Effect ###########
def _gaussian_kernel_matrix(dist):
        """Multi-scale RBF kernel."""
        sigmas = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5,
            10, 15, 20, 25, 30, 35, 100, 1e3, 1e4, 1e5, 1e6]

        beta = 1. / (2. * (tf.expand_dims(sigmas, 1)))

        s = tf.matmul(beta, tf.reshape(dist, (1, -1)))

        return tf.reshape(tf.reduce_sum(tf.exp(-s), 0), tf.shape(dist)) / len(sigmas)

def _pairwise_dists(x1, x2,broadcast = True):
    """Helper function to calculate pairwise distances between tensors x1 and x2."""

    if broadcast:
        r = tf.expand_dims(x1, axis=1)
        D = tf.reduce_sum(tf.square(r-x2), axis=-1)
    else:
        r1 = tf.reduce_sum(x1 * x1, keepdims=True)
        r2 = tf.reduce_sum(x2 * x2, keepdims=True)
        D = r1 - 2 * tf.matmul(x1, tf.transpose(x2)) + tf.transpose(r2)
    return D

def _build_reconstruction_loss_mmd( reconstructed, y, batches):
        """
        Build the reconstruction loss part of the network if batch correction is being performed.

        :param reconstructed: the tensorflow op that was output by the decoder
        :param y: the tensorflow op for the target
        """
        loss_recon = tf.constant(0.)
        refrecon = tf.boolean_mask(reconstructed, tf.equal(batches, 0))
        refy = tf.boolean_mask(y, tf.equal(batches, 0))
        l = (refy - refrecon)**2
        loss_recon += tf.reduce_mean(l)


        nonrefrecon = tf.boolean_mask(reconstructed, tf.equal(batches, 1))
        nonrefy = tf.boolean_mask(y, tf.equal(batches, 1))

        mean1, var1 = tf.nn.moments(nonrefrecon, 0)
        mean2, var2 = tf.nn.moments(nonrefy, 0)
        l = ( ((nonrefrecon - mean1) / (tf.sqrt(var1+1e-6)+1e-6)) - ((nonrefy - mean2) / (tf.sqrt(var2+1e-6)+1e-6)) )**2

        loss_recon += .01*tf.reduce_mean(l)

        # self.loss_recon = nameop(self.loss_recon, 'loss_recon')
        # tf.add_to_collection('losses', self.loss_recon)

        return loss_recon

def _build_reg_b(embedded,batches,Y_labels = None):
     #"""Build the tensorflow graph for the MMD regularization."""
     #"MMD(X,Y)=||1n2???i,i???n??(xi)??(x???i)???2nm???i,jn??(xi)??(yj)+1m2???j,j???n??(yj)??(y???j)||2H"

    var_within = {}
    batch_sizes = {}
    loss_b = tf.constant(0.)

    e = embedded / tf.reduce_mean(embedded)
    K = _pairwise_dists(e, e)
    K = K / tf.reduce_max(K)
    K = _gaussian_kernel_matrix(K)

    if Y_labels is not None:  # reference batch with similar cell type
        sc_types_set = [0,1]
        #sc_types_set, sc_types_label = tf.unique(Y)
        for k in sc_types_set:
            _K = tf.boolean_mask(K,tf.equal(Y_labels, k))
            _batches = tf.boolean_mask(batches,tf.equal(Y_labels, k))
            print("K.shape2:",_K.shape)
            i = 0
            print("tf.equal:",tf.equal(_batches, i))
            batch1_rows = tf.boolean_mask(_K, tf.equal(_batches, i))
            print("batch1_rows:",batch1_rows)
            batch1_rowscols = tf.boolean_mask(tf.transpose(batch1_rows), tf.equal(_batches, i))

            K_b1 = batch1_rowscols
            n_rows_b1 = tf.cast(tf.reduce_sum(tf.boolean_mask(tf.ones_like(_batches), tf.equal(_batches, i))), tf.float32)
            K_b1 = tf.reduce_sum(K_b1) / (n_rows_b1**2)

            var_within[i] = K_b1
            batch_sizes[i] = n_rows_b1

            # nonreference batches
            j = 1
            batch2_rows = tf.boolean_mask(_K, tf.equal(_batches, j))
            batch2_rowscols = tf.boolean_mask(tf.transpose(batch2_rows), tf.equal(_batches, j))

            K_b2 = batch2_rowscols
            n_rows_b2 = tf.cast(tf.reduce_sum(tf.boolean_mask(tf.ones_like(_batches), tf.equal(_batches, j))), tf.float32)
            K_b2 = tf.reduce_sum(K_b2) / (n_rows_b2**2)

            var_within[j] = K_b2
            batch_sizes[j] = n_rows_b2

            K_12 = tf.boolean_mask(_K, tf.equal(_batches, i))
            K_12 = tf.boolean_mask(tf.transpose(K_12), tf.equal(_batches, j))
            K_12_ = tf.reduce_sum(tf.transpose(K_12))

            mmd_pair = var_within[i] + var_within[j] - 2 * K_12_ / (batch_sizes[i] * batch_sizes[j])
            loss_b += tf.abs(mmd_pair) * 1e6
    else:
        # reference batch
        i = 0
        batch1_rows = tf.boolean_mask(K, tf.equal(batches, i))
        print("batch1_rows:",batch1_rows)
        batch1_rowscols = tf.boolean_mask(tf.transpose(batch1_rows), tf.equal(batches, i))


        K_b1 = batch1_rowscols
        n_rows_b1 = tf.cast(tf.reduce_sum(tf.boolean_mask(tf.ones_like(batches), tf.equal(batches, i))), tf.float32)
        K_b1 = tf.reduce_sum(K_b1) / (n_rows_b1**2)

        var_within[i] = K_b1
        batch_sizes[i] = n_rows_b1

        # nonreference batches
        j = 1
        batch2_rows = tf.boolean_mask(K, tf.equal(batches, j))
        batch2_rowscols = tf.boolean_mask(tf.transpose(batch2_rows), tf.equal(batches, j))

        K_b2 = batch2_rowscols
        n_rows_b2 = tf.cast(tf.reduce_sum(tf.boolean_mask(tf.ones_like(batches), tf.equal(batches, j))), tf.float32)
        K_b2 = tf.reduce_sum(K_b2) / (n_rows_b2**2)

        var_within[j] = K_b2
        batch_sizes[j] = n_rows_b2

        K_12 = tf.boolean_mask(K, tf.equal(batches, i))
        K_12 = tf.boolean_mask(tf.transpose(K_12), tf.equal(batches, j))
        K_12_ = tf.reduce_sum(tf.transpose(K_12))

        mmd_pair = var_within[i] + var_within[j] - 2 * K_12_ / (batch_sizes[i] * batch_sizes[j])
        loss_b += tf.abs(mmd_pair) * 1e6

    return loss_b
##################################################

class Model(object):
    def __init__(self, dataname, dims, cluster_num,  learning_rate, batch_size, lambda_b = 0, lambda_c = 0, lambda_d = 0,lambda_e = 0, method="KMeans", n_cores=-1, noise_sd = 1.5,
                 init = "glorot_uniform", act = "relu"):
        self.dataname = dataname
        self.dims = dims
        self.cluster_num = cluster_num
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.noise_sd = noise_sd
        self.init = init
        self.method = method
        self.n_cores = n_cores
        # self.act = act

        # self.n_stacks = len(self.dims) - 1
        self.x = tf.compat.v1.placeholder(dtype=tf.float32, shape=(None, self.dims[0]))
        self.raw_h = tf.compat.v1.placeholder(dtype=tf.float32, shape=(None, self.dims[-1]))
        self.non_zero_mask= tf.compat.v1.placeholder(dtype=tf.float32, shape=(None, self.dims[0]))
        self.cluster_centers = tf.compat.v1.placeholder(dtype=tf.float32, shape=(None, self.dims[-1]))
        self.cluster_labels = tf.compat.v1.placeholder(dtype=tf.int32, shape=(None))
        self.batch_labels = tf.compat.v1.placeholder(dtype=tf.int32, shape=(None, ))
        self.Y_labels = tf.compat.v1.placeholder(dtype=tf.int32, shape=(None, ))
        self.encoder_sub = tf.compat.v1.placeholder(dtype=tf.float32, shape=(None, self.dims[-1]))
        self.landmk_tr   = tf.compat.v1.placeholder(dtype=tf.float32, shape=(None, self.dims[-1] ))
        self.encoder_test= tf.compat.v1.placeholder(dtype=tf.float32, shape=(None, self.dims[-1]))
        self.landmk_test   = tf.compat.v1.placeholder(dtype=tf.float32, shape=(None, self.dims[-1]))

        self.unscale_x = tf.compat.v1.placeholder(dtype=tf.float32, shape=(None, self.dims[0]))
        self.x_count = tf.compat.v1.placeholder(dtype=tf.float32, shape=(None, self.dims[0]))
        self.clusters = tf.compat.v1.get_variable(name=self.dataname + "/clusters_rep", shape=[self.cluster_num, self.dims[-1]],dtype=tf.float32, initializer=tf.compat.v1.glorot_uniform_initializer())
        self.gene_b = tf.nn.relu(tf.compat.v1.get_variable(name=self.dataname + "/gb", shape=[ self.dims[0]],dtype=tf.float32, initializer=tf.compat.v1.glorot_uniform_initializer()))
        self.GRN = tf.compat.v1.get_variable( name='grn',shape=[ self.dims[0],self.dims[0]],dtype=tf.float32, initializer=tf.compat.v1.glorot_uniform_initializer())


        self.select_m_h =tf.nn.relu(Dense(units=self.dims[-1], kernel_initializer=self.init, name='encoder_m_h')(self.x))
        self.select_m = tf.nn.sigmoid(Dense(units=self.dims[0], kernel_initializer=self.init, name='encoder_m' )(self.select_m_h))
        self.select_w_h =tf.nn.relu(Dense(units=self.dims[-1], kernel_initializer=self.init, name='encoder_w_h')(self.x))
        self.select_w = tf.nn.sigmoid(Dense(units=self.dims[0], kernel_initializer=self.init, name='encoder_w' )(self.select_w_h))
        self.h = tf.multiply(self.x,self.select_m)
        self.h = GaussianNoise(self.noise_sd, name='input_noise')(self.h)
        self.h = Dense(units=self.dims[-1], kernel_initializer=self.init, name='encoder_hidden')(self.h)


        self.auto_decode_X = Dense(units=self.dims[0],  activation=imAct, kernel_initializer=self.init, name='imX')(self.h)

        alp = tf.math.sigmoid(tf.compat.v1.get_variable(name=self.dataname + "/clusters_repsd", shape=[ self.dims[0]],dtype=tf.float32, initializer=tf.compat.v1.glorot_uniform_initializer()))
#        alp=tf.math.sigmoid(Dense(units=self.dims[0],  activation=imAct, kernel_initializer=self.init, name='confidence')(self.h))
        eye=1-tf.constant(np.eye(self.dims[0],dtype=np.float32))
#        t=tf.matmul(tf.transpose(self.select_m),self.select_m)/tf.cast(tf.shape(self.select_m)[0],tf.float32)
        self.imX=tf.multiply(self.unscale_x,self.non_zero_mask)+tf.multiply(self.auto_decode_X,1-self.non_zero_mask)
        self.imX = alp*tf.matmul(self.imX,tf.multiply(self.GRN,eye))+(1-alp)*self.gene_b
        
        self.imp_loss = tf.reduce_sum(input_tensor=tf.multiply((self.imX-self.unscale_x)**2/tf.reshape(tf.reduce_sum(input_tensor=self.non_zero_mask,axis=1),(-1,1)),self.non_zero_mask)) + \
                        tf.reduce_sum(input_tensor=tf.multiply((self.auto_decode_X-self.unscale_x)**2/tf.reshape(tf.reduce_sum(input_tensor=self.non_zero_mask,axis=1),(-1,1)),self.non_zero_mask ))
        
        self.cluster_loss=self.imp_loss+tf.reduce_sum(input_tensor=(self.h-tf.nn.embedding_lookup(params=self.cluster_centers,ids=self.cluster_labels))**2)
        self.optimizer = tf.compat.v1.train.AdamOptimizer(self.learning_rate)
        self.moptimizer = tf.compat.v1.train.AdamOptimizer(self.learning_rate)
        
        if lambda_d is not None: #batch-effect
            self.imp_mmd_loss = lambda_d *_build_reg_b(self.h,self.batch_labels) +  0.1*self.imp_loss
            self.imp_mmd_op = self.optimizer.minimize(self.imp_mmd_loss)
        
        if lambda_e is not None: #MARS
            self.tr_dist = tf.reduce_mean(tf.sqrt((tf.expand_dims(self.encoder_sub,axis = 1)- tf.expand_dims(self.landmk_tr,axis = 0))**2),axis = -1)
            Loss_i = tf.reduce_sum(self.tr_dist)

            nproto = 14
            self.test_dist = tf.reduce_min(tf.sqrt((tf.expand_dims(self.encoder_test, axis = 1) - tf.expand_dims(self.landmk_test,axis = 0))**2),axis = -1)
            Loss_u = tf.reduce_sum(self.test_dist) + tf.reduce_sum(tf.reduce_sum(input_tensor = tf.sqrt((tf.expand_dims(self.landmk_test,axis = 1) - tf.expand_dims(self.landmk_test,axis = 0))**2),axis = -1))/(nproto*nproto-nproto)
            self.mars_loss = self.imp_loss + Loss_i + lambda_e * Loss_u
            self.mars_op = self.optimizer.minimize(self.mars_loss)
        

        self.pretrain_op = self.optimizer.minimize(self.imp_loss)
        self.cluster_op = self.optimizer.minimize(self.cluster_loss)
        
    def train(self, adata, adata_unscaled, adata_cnt, split, epochs, random_seed, inherit_centroids, L2, linkage, gpu_option,type_to_mg=None):

        X = adata.X[:split].astype(np.float32)
        unscale_X = adata_unscaled[:split].X.astype(np.float32)
        count_X = adata_cnt.X[:split].astype(np.float32)
        Y = adata.obs["cell_groups"][:split]
        if 'batch' in adata.obs:  #batch labels
            Z = adata.obs["batch"][:split] 
        elif 'experiment' in adata.obs:  # experiment
            Z = adata.obs["experiment"][:split]
        else :
            Z = np.full(shape = Y.shape, fill_value = 1, dtype = np.float)

        test_X = adata.X[split:].astype(np.float32)
        unscale_test_X = adata_unscaled.X[split:].astype(np.float32)
        test_count_X = adata_cnt.X[split:].astype(np.float32)
        genes_name = adata.var["gene_name"][:split]
        cells_name = adata.obs['cell_type'][:split]
        nonzero_mask=get_mask(unscale_X)
        t1 = time.time()
        # np.random.seed(random_seed)
        # tf.compat.v1.set_random_seed(random_seed)
        batch_size = self.batch_size

        pretrain_epochs = epochs

        cell_type, Y_target = np.unique(Y, return_inverse=True)

        if X.shape[0] < batch_size:
            batch_size = X.shape[0]

        n_clusters = len(np.unique(Y_target))
        print("Mixed data has {} total clusters".format(n_clusters))

        eta = 0.
        print("end the data proprocess")
        init = tf.group(tf.compat.v1.global_variables_initializer(), tf.compat.v1.local_variables_initializer())
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_option

        config_ = tf.compat.v1.ConfigProto()
        config_.gpu_options.allow_growth = True
        config_.allow_soft_placement = True
        sess = tf.compat.v1.Session(config=config_)
        sess.run(init)

        print("Z",Z.shape)
        print("Z:",Z)

        fig = plt.figure(figsize=(8, 8))		# ????????????????????????
        plt.suptitle("Simulation_R", fontsize=14)		# ?????????????????????
        fig = draw_TSNE(fig,X,Y_target,Z,index=1,subtitle = "Raw Data")

        # print("begin model pretrain(imputation)")
        latent_repre = np.zeros((X.shape[0], self.dims[-1]))
        iteration_per_epoch = math.ceil(float(len(X)) / float(batch_size))
        iteration_per_epoch_train = math.ceil(float(len(X)) / float(batch_size))
        imp_loss=0
        h,sm  = sess.run([self.h,self.select_m],
                    feed_dict={
                        self.x: X,
                        self.unscale_x: unscale_X,
                        })
        h=np.squeeze(h)
        if L2:
            h = normalize(h)
        if inherit_centroids:
            kmeans = KMeans(n_clusters=n_clusters, init="k-means++")
            kmeans.fit(h)
            pre_clusters=kmeans.cluster_centers_

        # Stage: Imputation
        print("begin model pretrain(Imputation)")
        for i in range(pretrain_epochs):
            for j in range(iteration_per_epoch):
                batch_idx = random.sample(range(X.shape[0]), batch_size)
        
                _,latent, imp_loss = sess.run([self.pretrain_op,self.h ,self.imp_loss],
                feed_dict= {self.x: X[batch_idx],
                    self.unscale_x: unscale_X[batch_idx],
                    self.non_zero_mask: nonzero_mask[batch_idx],
                    self.x_count: count_X[batch_idx],
                    })
                latent_repre[batch_idx] = latent
    
            h,sm  = sess.run([self.h,self.select_m],
                        feed_dict={
                            self.x: X,
                            self.unscale_x: unscale_X,
                            })

            latent_repre = np.nan_to_num(latent_repre)
            if L2:
                latent_repre = normalize(latent_repre)
            if inherit_centroids:
                kmeans = KMeans(n_clusters=n_clusters, init=pre_clusters, n_init=1)
                kmeans.fit(latent_repre)
                pre_clusters=kmeans.cluster_centers_
            print(pretrain_epochs,i,'imputation loss',imp_loss)

        fig = draw_TSNE(fig, latent_repre,Y_target,Z,index=2,subtitle="Imputation",_inherit_centroids = inherit_centroids,_L2 = L2)
            
        # Stage: MMD + 0.01 * Imputation
        if 'batch' in adata.obs:  #batch labels
            print("begin model pretrain(MMD+ 0.01*Imputation)")
            for i in range(pretrain_epochs):
                for j in range(iteration_per_epoch):
                    batch_idx = random.sample(range(X.shape[0]), batch_size)
            
                    if 1:
                        _,latent, imp_loss = sess.run([self.imp_mmd_op,self.h ,self.imp_mmd_loss],
                        feed_dict= {self.x: X[batch_idx],
                            self.unscale_x: unscale_X[batch_idx],
                            self.non_zero_mask: nonzero_mask[batch_idx],
                            self.x_count: count_X[batch_idx],
                            self.batch_labels:Z[batch_idx],
                            #self.Y_labels:Y_target[batch_idx],
                            })
                        latent_repre[batch_idx] = latent
                    else:
                        _,latent = sess.run([self.imp_mmd_op,self.h,self.imp_mmd_loss],
                        feed_dict= {
                            self.x : X[batch_idx],
                            self.batch_labels : Z[batch_idx],
                            })
                        latent_repre[batch_idx] = latent
        
                latent_repre = np.nan_to_num(latent_repre)
                if L2:
                    latent_repre = normalize(latent_repre)
                if inherit_centroids:
                    kmeans = KMeans(n_clusters=n_clusters, init=pre_clusters, n_init=1)
                    kmeans.fit(latent_repre)
                    pre_clusters=kmeans.cluster_centers_
                print(pretrain_epochs,i,'mmd loss',imp_loss)
            fig = draw_TSNE(fig, latent_repre,Y_target,Z,index=3,subtitle="MMD",bSave= True,_inherit_centroids = inherit_centroids,_L2 = L2 )

        print(unscale_X.shape)
        print(unscale_test_X.shape)
        print(np.concatenate([unscale_X,unscale_test_X],axis=0).shape)
        h = np.array(sess.run([self.h],
                    feed_dict={
                        self.x: np.concatenate((X,test_X),axis=0),
                        self.unscale_x: np.concatenate([unscale_X,unscale_test_X],axis=0),
                        self.non_zero_mask: get_mask(np.concatenate([unscale_X,unscale_test_X],axis=0))
                    }))
        imX, select_m = sess.run([self.imX, self.select_m],
                                feed_dict={
                                    self.x: np.concatenate((X,test_X),axis=0),
                                    self.unscale_x: np.concatenate([unscale_X,unscale_test_X],axis=0),
                                    self.non_zero_mask: get_mask(np.concatenate([unscale_X,unscale_test_X],axis=0))
                                })
        select_w = sess.run([self.select_w],
                                feed_dict={
                                    self.x: np.concatenate((X,test_X),axis=0),
                                    self.unscale_x: np.concatenate([unscale_X,unscale_test_X],axis=0),
                                    self.non_zero_mask: get_mask(np.concatenate([unscale_X,unscale_test_X],axis=0))
                                })
        print(np.array(select_m).shape)
        print(np.array(select_w).shape)
        imX_df = pd.DataFrame(np.array(imX), index=cells_name, columns=genes_name)
        select_m_df = pd.DataFrame(np.array(select_m), columns=genes_name)
        select_w_df = pd.DataFrame(np.array(np.squeeze(select_w)), columns=genes_name)

        latent_repre = np.nan_to_num(latent_repre)
        if L2:
            latent_repre = normalize(latent_repre)
        
        if 'experiment' in adata.obs:  #MARS
            Y_pred = self.Build_MARS(X,unscale_X,nonzero_mask,count_X,sess,latent_repre,Z,Y_target)
        else:
            print("Tradition Way")
            if os.name == "nt":
                cache_dir = "F:\cache"
            else:
                cache_dir = None
            if inherit_centroids:
                kmeans = KMeans(n_clusters=n_clusters, init=pre_clusters)
                if self.method == "KMeans":
                    clustering = kmeans
                elif self.method == "SpectralClustering":
                    clustering = SpectralClustering(n_clusters=n_clusters)
                elif self.method == "AgglomerativeClustering":
                    clustering = AgglomerativeClustering(n_clusters=n_clusters, memory=cache_dir, linkage=linkage)
            else:
                kmeans = KMeans(n_clusters=n_clusters)
                if self.method == "KMeans":
                    clustering = kmeans
                elif self.method == "SpectralClustering":
                    clustering = SpectralClustering(n_clusters=n_clusters)
                elif self.method == "AgglomerativeClustering":
                    clustering = AgglomerativeClustering(n_clusters=n_clusters, memory=cache_dir, linkage=linkage)
            Y_pred = clustering.fit_predict(latent_repre)
            kmeans.fit(latent_repre)
            cluster_centers = kmeans.cluster_centers_
            # target_AMI = np.around(adjusted_mutual_info_score(Y_target, Y_pred), 4)
        
        fig = draw_TSNE(fig, latent_repre,Y_target,Z,index=3,subtitle="Clustering",_inherit_centroids = inherit_centroids,_L2 = L2,bSave= True)
        print("Y_pred",Y_pred)
        print("Y_target",Y_target)
        target_ARI = np.around(adjusted_rand_score(Y_target, Y_pred), 4)
        print('total ARI',target_ARI)

        imX = np.array(sess.run(
                    [self.imX],
                    feed_dict={
                        self.x: np.concatenate((X,test_X),axis=0) ,
                        self.unscale_x: np.concatenate((unscale_X,unscale_test_X),axis=0),
                        self.non_zero_mask:np.concatenate([nonzero_mask,unscale_test_X],axis=0) 
                        }))

        # [target_AMI, target_ARI, CHS, DBS, target_CPS, target_NMI]
        return imX, Y_pred, imX_df, select_m_df, select_w_df

    def Build_MARS(self,X,unscale_X,nonzero_mask,count_X,sess,latent_repre,Exp,Y_target):
        # latent_repre: output of hidder layerr for all experiments
        # Exp: label of experiments
        encoded_tr,landmk_tr,landmk_tr_labels,encoded_test,landmk_test,landmk_test_labels,cell_name_mappings = self.init_landmark(latent_repre,Exp,Y_target)
        encoded_tr,landmk_tr,landmk_tr_labels,encoded_test,landmk_test,landmk_test_labels,acc,Y_pred = self.Run_MARS(sess,X,unscale_X,nonzero_mask,count_X,Exp,encoded_tr,landmk_tr,landmk_tr_labels,encoded_test,landmk_test,landmk_test_labels)
        self.name_cell_types(encoded_tr,landmk_tr,landmk_tr_labels,encoded_test,landmk_test,landmk_test_labels,Y_pred,Exp,cell_name_mappings)
        return Y_pred

    def compute_kmean(self,latent_repre,n_clusters,pre_clusters=None,inherit_centroids = False,linkage = "ward"):
            if os.name == "nt":
                cache_dir = "F:\cache"
            else:
                cache_dir = None
            if inherit_centroids:
                kmeans = KMeans(n_clusters=n_clusters, init=pre_clusters)
                if self.method == "KMeans":
                    clustering = kmeans
                elif self.method == "SpectralClustering":
                    clustering = SpectralClustering(n_clusters=n_clusters)
                elif self.method == "AgglomerativeClustering":
                    clustering = AgglomerativeClustering(n_clusters=n_clusters, memory=cache_dir, linkage=linkage)
            else:
                kmeans = KMeans(n_clusters=n_clusters)
                if self.method == "KMeans":
                    clustering = kmeans
                elif self.method == "SpectralClustering":
                    clustering = SpectralClustering(n_clusters=n_clusters)
                elif self.method == "AgglomerativeClustering":
                    clustering = AgglomerativeClustering(n_clusters=n_clusters, memory=cache_dir, linkage=linkage)
            Y_pred = clustering.fit_predict(latent_repre)
            kmeans.fit(latent_repre)
            cluster_centers = kmeans.cluster_centers_

            return Y_pred,cluster_centers

    def Run_MARS(self,sess,X,unscale_X,nonzero_mask,count_X,Exp,encoded_tr,landmk_tr,landmk_tr_labels,encoded_test,landmk_test,landmk_test_labels):
        total_acc_tr = 0
        Y_pred = np.zeros(Exp.shape)
        index_unannotated =  np.where(Exp == unannotated_exp)

        for j in np.unique(landmk_tr_labels):
            index = np.where(landmk_tr_labels == j)
            sub_latent = encoded_tr[index]
            sub_landmk_tr = np.expand_dims(landmk_tr[j],axis = 0)
            sub_landmk_tr_labels = landmk_tr_labels[index]

            _,latent, mars_loss,tr_dist,test_dist = sess.run([self.mars_op,self.h ,self.mars_loss, self.tr_dist, self.test_dist],
            feed_dict= {self.x: X[index],
                    self.unscale_x: unscale_X[index],
                    self.non_zero_mask: nonzero_mask[index],
                    self.x_count: count_X[index],
                    self.encoder_sub : sub_latent,
                    self.landmk_tr   : landmk_tr,
                    self.encoder_test: encoded_test,
                    self.landmk_test : landmk_test,
                    })

            encoded_tr[index] = latent

            #annotated experiment
            _,cluster_centers = self.compute_kmean(latent,len(np.unique(sub_landmk_tr_labels)),sub_landmk_tr)
            Y_pred[index] = np.argmax(-tr_dist,axis = 1)
            landmk_tr[j] = cluster_centers

            #unannotated experiment
            _,cluster_centers = self.compute_kmean(encoded_test,len(np.unique(landmk_test_labels)),landmk_test)
            Y_pred[index_unannotated] = np.argmax(-test_dist,axis = 1)
            landmk_test = cluster_centers

        mean_acc_tr = total_acc_tr / len(np.unique(Exp))
        return encoded_tr,landmk_tr,landmk_tr_labels,encoded_test,landmk_test,landmk_test_labels,mean_acc_tr,Y_pred

    def init_landmark(self,latent_repre,Exp,Y_target):
        encoded_tr = []
        encoded_test = []
        landmk_tr  = []
        landmk_test  = []
        landmk_tr_labels = []
        landmk_test_labels = []

        unique_Exp  = np.unique(Exp)
        for exp in unique_Exp:
            index = np.where(Exp == exp)
            latent_uni = latent_repre[index]
            n_clusters = len(np.unique(Y_target[index]))
            Y_pred,cluster_centers = self.compute_kmean(latent_uni,n_clusters)
            if exp == unannotated_exp: # unannotated experiments
                encoded_test.append(latent_uni)
                landmk_test.append(cluster_centers)
                landmk_test_labels.append(Y_target[index])
            else:
                encoded_tr.append(latent_uni)
                landmk_tr.append(cluster_centers)
                landmk_tr_labels.append(Y_target[index])
        
        encoded_tr = np.squeeze(np.array(encoded_tr))
        encoded_test = np.squeeze(np.array(encoded_test))
        landmk_tr  = np.squeeze(np.array(landmk_tr))
        landmk_test  = np.squeeze(np.array(landmk_test))
        landmk_tr_labels = np.squeeze(np.array(landmk_tr_labels))
        landmk_test_labels = np.squeeze(np.array(landmk_test_labels))

        cell_name_mappings = np.unique(landmk_tr_labels)
        print("cell_name_mappings:",cell_name_mappings)
        return encoded_tr,landmk_tr,landmk_tr_labels,encoded_test,landmk_test,landmk_test_labels,cell_name_mappings

    def estimate_sigma(self,dataset):
        nex = dataset.shape[0]
        dst = []
        for i in range(nex):
            for j in range(i+1, nex):
                dst.append(distance.euclidean(dataset[i,:],dataset[j,:]))
        return np.std(dst)

    def name_cell_types(self,encoded_tr,landmk_tr,landmk_tr_labels,encoded_test,landmk_test,landmk_test_labels,Y_pred,Exp,cell_name_mappings):

        interp_names = defaultdict(list)
        ypred_test = Y_pred[np.where(Exp == unannotated_exp)]
        uniq_ytest = np.unique(ypred_test)
        for ytest in uniq_ytest:
            print('\nCluster label: {}'.format(str(ytest)))
            idx = np.where(ypred_test==ytest)
            subset_encoded = encoded_test[idx[0],:]
            mean = np.expand_dims(np.mean(subset_encoded, axis=0),0)
            
            sigma  = self.estimate_sigma(subset_encoded)
            prob = np.exp(-np.power(distance.cdist(mean, landmk_tr, metric='euclidean'),2)/(2*sigma*sigma))
            prob = np.squeeze(prob, 0)
            normalizat = np.sum(prob)
            print("mean.shape:",mean.shape)
            print("sigma.shape:",sigma)
            if normalizat==0:
                print('Unassigned')
                interp_names[ytest].append("unassigned")
                continue
            
            prob = np.divide(prob, normalizat)
            
            #Todo
            uniq_tr = np.unique(landmk_tr_labels)
            prob_unique = prob
            # prob_unique = []
            # for cell_type in uniq_tr: # sum probabilities of same landmarks
            #     prob_unique.append(np.sum(prob[np.where(landmk_tr_labels==cell_type)]))
            
            sorted = np.argsort(prob_unique, axis=0)
            top_match = 14
            best = uniq_tr[sorted[-top_match:]]
            sortedv = np.sort(prob_unique, axis=0)
            sortedv = sortedv[-top_match:]
            for idx, b in enumerate(best,start = -1):
                interp_names[ytest].append((cell_name_mappings[b], sortedv[idx]))
                print('{}: {}'.format(cell_name_mappings[b], sortedv[idx]))
                
        return interp_names

def draw_TSNE(fig,X,Y,Z,index = 1,subtitle = "",columns = 2, rows = 2,_inherit_centroids = False, _L2= False, bSave = False):
    _marker = ['x','s','D','*']
    _color = ['r','g','b','y','m','c','k']
    ts = sklearn.manifold.TSNE(n_components=2, init='pca', random_state=0)
    x = ts.fit_transform(X)
    Y = np.array(Y, dtype=int)

    sc_types_set, sc_types_labels = np.unique(Z, return_inverse=True)
    fig.add_subplot(columns, rows, index).title.set_text("Stage " + str(index) + ": " + subtitle)
    sample_size = len(Y)
    if len(np.unique(Y)) > len(_color):
        for i in range(sample_size):
            plt.scatter(x[i, 0], x[i, 1], c = Y[i],vmin=0, vmax=20,cmap=plt.cm.get_cmap('RdYlBu'),marker=_marker[sc_types_labels[i]],alpha=0.6)
    else:
        for i in range(sample_size):
            plt.scatter(x[i, 0], x[i, 1], c=_color[Y[i]],marker=_marker[sc_types_labels[i]],alpha=0.6)

    if bSave == True:
        save_dir = "D:/Result"
        if _inherit_centroids:
            save_dir+="_inherit_centroids"
        if _L2:
            save_dir+="_L2"
        save_dir+=".jpg"
        print("save_dir:",save_dir)
        plt.savefig(save_dir)
        #plt.show()
    
    return fig

import numpy as np
import sklearn
from sklearn import manifold
import matplotlib.pyplot as plt
# def draw_TSNE(x,label,batch,dir,index = 1):
#     # ?????????????????????
#     fig = plt.figure(figsize=(8, 8))		# ????????????????????????
#     plt.suptitle("Dimensionality Reduction and Visualization of Embedding ", fontsize=14)		# ?????????????????????
    
#     # t-SNE?????????????????????
#     ts = sklearn.manifold.TSNE(n_components=2, init='pca', random_state=0)
#     # ????????????
#     y = ts.fit_transform(x)
#     ax1 = fig.add_subplot(2, 2, index)
#     plt.scatter(y[:, 0], y[:, 1], c=label, cmap=plt.cm.Spectral)
#     ax1.set_title('t-SNE Curve', fontsize=14)
#     # ????????????
#     print("dir",dir)
#     plt.savefig(dir)
#     #plt.show()
