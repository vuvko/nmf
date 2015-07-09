#!/usr/bin/python2
# -*- coding: utf-8 -*-

from numpy import *
#from sklearn.cluster import KMeans
from yael import ynumpy

import config
from common import normalize_cols
from Q_matrix import generate_Q_matrix
from anchors import findAnchors
from fastRecover import do_recovery
from bhtsne import bh_tsne


def find_nearest(D, centroids, labels):
    num_labels = max(labels) + 1
    indecies = zeros((centroids.shape[0], ), dtype='int32')
    for label in xrange(num_labels):
        wh = where(labels == label)[0]
        sub_D = D[wh, :]
        # finding nearest point to centroid
        cur_c = tile(centroids[label, :], (sub_D.shape[0], 1))
        cur_dis = sum((sub_D - cur_c) ** 2)
        indecies[label] = wh[argmin(cur_dis)]
    return indecies


def reduce_tsne(D, to_dim=2):
    print('Reducing with t-SNE')
    return array([x for x in bh_tsne(D, verbose=True)])


def reduce_cluster(D, num_clusters, params=config.default_config()):
    print('Clustering:')
    D = ascontiguousarray(D.astype('float32'))
    centroids, qerr, dis, labels, nassign = ynumpy.kmeans(D, num_clusters, init='kmeans++', nt=params['num_threads'], output='all', redo=3, niter=params['kmeans_max_iter'], verbose=False)
    #kmeans = KMeans(n_init=1, n_clusters=params['num_clusters'], n_jobs=2, max_iter=params['kmeans_max_iter'])
    #kmeans.fit(D)
    print('Done.')
    #centroids = kmeans.cluster_centers_
    #labels = kmeans.labels_
    return centroids, labels


def reduce_multi_cluster(D, num_clusters, params=config.default_config()):
    print('Clustering:')
    D = ascontiguousarray(D.astype('float32'))
    #ncc = maximum(minimum(random.poisson(num_clusters, 15), 1000), 15)
    N = D.shape[0]
    #ncc = array([20, 50, 100, 250, 500, 1000, 2000, 4000, 6000])
    ncc = array([25 * (2 ** p) for p in xrange(int(log2(N / 75))  )])
    print(ncc)
    centroids = zeros((sum(ncc), D.shape[1]))
    labels = zeros((N, len(ncc)), dtype='int32')
    c = 0
    for it, nc in enumerate(ncc):
        new_centroids, _, _, new_labels, _ = ynumpy.kmeans(D.astype('float32'), nc, init='random', nt=params['num_threads'], output='all', redo=1, niter=params['kmeans_max_iter'], verbose=False)
        centroids[c:c+nc, :] = new_centroids
        labels[:, it] = new_labels.squeeze() + c
        c += nc
    print('Done.')
    return centroids, labels


def anchor_words(D, loss='L2', params=config.default_config()):
    Q = generate_Q_matrix(D * 100)
    anchors = findAnchors(Q, params['T'], params)
    W, topic_likelihoods = do_recovery(Q, anchors, loss, params)
    return W


def restore_cluster(W_reduced, labels, params):
    W = zeros((params['N'], params['T']))
    for word, label in enumerate(labels):
        W[word, :] = W_reduced[label, :]
    return normalize_cols(W)


def restore_multi_cluster(W_reduced, labels, params):
    W = zeros((params['N'], params['T']))
    for word in xrange(W.shape[0]):
        W[word, :] = mean(W_reduced[labels[word, :], :])
    return normalize_cols(W)
