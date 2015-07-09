#!/usr/bin/python2
# -*- coding: utf-8 -*-

from numpy import *
from measure import *


from sklearn.utils.linear_assignment_ import linear_assignment


def print_head(f):
    return
#    print('##################################################', file=f)


def normalize_cols(matrix):
    return matrix / maximum(sum(matrix, 0), 1e-7)


def normalize_rows(matrix):
    return (matrix.T / maximum(sum(matrix, 1), 1e-7)).T


def cost(V, W, H):
    #meas = cfg['measure'].split(',')
    return frobenius(V, dot(W, H)) / 2
    #return perplexity(V, W, H)
    #return kl(V, dot(W, H))


def get_permute(W_r, H_r, W, H, coeff):
    T_0 = W_r.shape[1]
    T = W.shape[1]
    comp = zeros((T_0, T))
    for t in xrange(T):
        p = sqrt(W_r) - sqrt(tile(W[:, t], (T_0, 1))).T
        comp[:, t] = sum(p ** 2, 0)
    if T < T_0:
        t = array(linear_assignment(comp.T))[:, [1, 0]]
        # http://stackoverflow.com/questions/2828059/sorting-arrays-in-numpy-by-column
        return t[t[:, 0].argsort()]
    else:
        return array(linear_assignment(comp))
