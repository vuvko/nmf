#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
from common import normalize_cols


def gen_matrix_uniform(params):
    return normalize_cols(np.random.uniform(size=(params['rows'], params['cols'])))


def gen_matrix_normal(params):
    return normalize_cols(abs(np.random.randn(params['rows'], params['cols'])))


def gen_matrix_sparse(params):
    rows = params['rows']
    cols = params['cols']
    sparsity = params['sparsity']
    M = np.zeros((rows, cols), dtype='float32')
    sz = int(rows * cols * (1 - sparsity))
    idx0_t = [i for i in range(rows) for j in range(cols)]
    np.random.shuffle(idx0_t)
    idx0 = idx0_t[:sz]
    idx1_t = [j for i in range(rows) for j in range(cols)]
    np.random.shuffle(idx1_t)
    idx1 = idx1_t[:sz]
    M[idx0, idx1] = np.random.sample(sz)
    if rows < cols:
        M[:, :rows] = M[:, :rows] + np.eye(rows) * params['eps']
        M[0, rows:] = params['eps']
    else:
        M[:cols, :] = M[:cols, :] + np.eye(cols) * params['eps']
    return normalize_cols(M)


def gen_documents(V, params):
    N, M = V.shape
    D = np.zeros(V.shape)
    doc_len = np.random.poisson(params['doc_len'], M)
    for d in range(M):
        D[:, d] = np.random.binomial(doc_len[d], V[:, d])
    return D


def gen_matrix_topic(params):
    N, T = params['rows'], params['cols']
    phi = np.zeros((N, T))
    sparse = params['sparsity'] # sparsness (the main parameter)
    if sparse < params['eps']:
        sparse = params['eps']
    elif sparse > 1:
        sparse = 1
    nkernel = params['nkernel'] # number of average kernel words in topic
    nnoise = params['nnoise'] # number of noise (smooth) topics
    ntopic = T - nnoise
    kernel = np.maximum(1, np.random.binomial(N, min(1, nkernel / (N*sparse)), ntopic))
    s = 0
    for i in range(ntopic):
        phi[s:s+kernel[i], i] = -np.sort(-np.random.exponential(0.5, kernel[i]))
        s = s + int(kernel[i] * sparse)
        if i < ntopic-1 and s + kernel[i+1] > N:
            kernel[i+1] = max(1, N - s)
            s = N - kernel[i+1]
    if N-s-kernel[-1]+1 > 0:
        if nnoise == 0:
            phi[s+kernel[-1]-1:, :] = np.random.random_sample((N-s-kernel[-1]+1, T))
        else:
            #phi[s+kernel[-1]-1:, ntopic:] = np.random.random_sample((N-s-kernel[-1]+1, nnoise))
            phi[:, ntopic:] = np.random.random_sample((N, nnoise))
    return normalize_cols(phi)
