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


def reduce_cluster(D, num_clusters, params=config.default_config()):
    print('Clustering:')
    centroids, qerr, dis, labels, nassign = ynumpy.kmeans(D.astype('float32'), num_clusters, init='kmeans++', nt=params['num_threads'], output='all', redo=10, niter=70, verbose=False)
    #kmeans = KMeans(n_init=1, n_clusters=params['num_clusters'], n_jobs=2, max_iter=params['kmeans_max_iter'])
    #kmeans.fit(D)
    print('Done.')
    #centroids = kmeans.cluster_centers_
    #labels = kmeans.labels_
    return centroids, labels


def anchor_words(D, loss='L2', params=config.default_config()):
    Q = generate_Q_matrix(D * 100)
    anchors = findAnchors(Q, params['T'], params)
    W, topic_likelihoods = do_recovery(Q, anchors, loss, params)
    return W


def unreduce_cluster(W_reduced, labels, params):
    W = zeros((params['N'], params['T']))
    for word, label in enumerate(labels):
        W[word, :] = W_reduced[label, :]
    return normalize_cols(W)
