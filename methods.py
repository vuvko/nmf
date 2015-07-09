#!/usr/bin/python2
# -*- coding: utf-8 -*-

from numpy import *
import config
from common import normalize_cols, cost
from copy import deepcopy
#from artm/python_interface import *
#import artm/messages_pb2


def grad_desc(V, W, H, post='', cfg=config.default_config()):
    alpha = cfg[post + '_alpha']
    step = cfg[post + '_alpha_step']
    eps = cfg['eps']
    #print('Gradient Descent with alpha={alpha}.'.format(alpha=alpha))
    grad_W = dot((V - dot(W, H)), H.T)
    grad_H = dot(W.T, (V - dot(W, H)))
    #grad_W[grad_W < eps] = 0
    #grad_H[grad_H < eps] = 0
    W = W + alpha * grad_W
    W[(grad_W < eps) & (W < eps)] = 0
    W = normalize_cols(W)
    
    H = H + alpha * grad_H
    H[(grad_H < eps) & (H < eps)] = 0
    H = normalize_cols(H)
    
    alpha = alpha * step
    cfg[post + '_alpha'] = alpha
    return (W, H)


def als(V, W, H, post='', cfg=config.default_config()):
    #print('Alternating Least Squares.')
    eps = cfg['eps']
    H = linalg.solve(dot(W.T, W) + eye(W.shape[1]) * eps, dot(W.T, V))
    H[H < eps] = 0
    W = linalg.solve(dot(H, H.T) + eye(H.shape[0]) * eps, dot(H, V.T)).T
    W[W < eps] = 0
    return (W, H)


def hals(V, W, H, post='', cfg=config.default_config()):
    eps = cfg['eps']
    T = H.shape[0]
    W0 = W
    H0 = H
    for k in range(T):
        R = V - dot(W0, H0) + dot(W0[:, [k]], H0[[k], :])
        H0[k, :] = maximum(dot(R.T, W0[:, k]), 0).T / maximum(sum(W0[:, k] ** 2, 0), eps)
        W0[:, k] =  maximum(dot(R, H0[k, :].T), 0) / maximum(sum(H0[k, :] ** 2), eps)
    return W, H


def mult(V, W, H, post='', cfg=config.default_config()):
    #print('Gradient Descent with Multiplicative Update Rule.')
    eps = cfg['eps']
    H = H * dot(W.T, V) / maximum(dot(W.T, dot(W, H)), eps)
    W = W * dot(V, H.T) / maximum(dot(W, dot(H, H.T)), eps)
    return (W, H)


def mult_kl(V, W, H, post='', cfg=config.default_config()):
    eps = cfg['eps']
    tmp = V / (dot(W, H) + eps)
    W0 = W * dot(tmp, H.T)
    W = W0 / tile(maximum(sum(H.T, 0), eps), (W0.shape[0], 1))
    H0 = H * dot(W.T, tmp)
    H = H0 / tile(maximum(sum(W.T, 1), eps), (1, H0.shape[1]))
    return W, H


def plsa(V, W, H, post='', cfg=config.default_config()):
    eps = cfg['eps']
    tmp = V / maximum(dot(W, H), eps)
    H = normalize_cols(H * dot(W.T, tmp))
    W = normalize_cols(W * dot(tmp, H.T))
    return W, H


def plsa3D(V, W, H, post='', cfg=config.default_config()):
    #print('Probabilistic Latent Semantic Analysis.')
    eps = cfg['eps']
    (N, M) = V.shape
    T = H.shape[0]
    V3 = V.reshape(N, M, 1).repeat(T, 2).swapaxes(1, 2)
    W3 = W.reshape(N, T, 1).repeat(M, 2)
    H3 = H.T.reshape(M, T, 1).repeat(N, 2).swapaxes(0, 2)
    Q3 = dot(W, H).reshape(N, M, 1).repeat(T, 2).swapaxes(1, 2)
    Z = V3 * W3 * H3 / (Q3 + eps)
    W = normalize_cols(sum(Z, 2).reshape(N, T))
    H = normalize_cols(sum(Z, 0).reshape(T, M))
    return W, H


def cnmf(V, W, H, post='', cfg=config.default_config()):
    eps = cfg['eps']
    alpha = cfg[post + '_alpha']
    beta = cfg[post + '_beta']
    H = H * dot(W.T, V) / (dot(W.T, dot(W, H)) + beta * H + eps)
    W = W * dot(V, H.T) / (dot(W, dot(H, H.T)) + alpha * W + eps)
    return W, H
