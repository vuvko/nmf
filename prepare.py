#!/usr/bin/python2
# -*- coding: utf-8 -*-

from numpy import *
from sklearn.cluster import KMeans

import config
from Q_matrix import generate_Q_matrix
from anchors import findAnchors


def reduce_cluster(D, params=config.default_config()):
    print('Clustering:')
    kmeans = KMeans(n_init=1, n_clusters=params['num_clusters'], n_jobs=2, max_iter=params['k_means_max_iter'])
    kmeans.fit(D)
    print('Done.')
    centroids = kmeans.cluster_centers_
    return centroids


def anchor_words(D, loss='L2', params=config.default_config()):
    Q = generate_Q_matrix(D)
    anchors = findAnchors(Q, params['T'], params)
    W, topic_likelihoods = do_recovery(Q, anchors, loss, params)
    eps = params['eps']
    H = linalg.solve(dot(W.T, W) + eye(W.shape) * eps, dot(W.T, V))
    H[H < eps] = 0
    return W, H
