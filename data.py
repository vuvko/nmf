#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function

import config
from os.path import join
from datetime import date
import numpy as np
from scipy.sparse import coo_matrix


def save_csv(V, W, H, name=str(date.today())):
    np.savetxt(join(cfg['data_dir'], name+'_V.csv'), V, delimiter=',')
    np.savetxt(join(cfg['data_dir'], name+'_W.csv'), W, delimiter=',')
    np.savetxt(join(cfg['data_dir'], name+'_H.csv'), H, delimiter=',')


def load_csv(name, cfg=config.default_config()):
    V = np.loadtxt(open(join(cfg['data_dir'], name+'_V.csv'), 'r'), delimiter=',')
    W = np.loadtxt(open(join(cfg['data_dir'], name+'_W.csv'), 'r'), delimiter=',')
    H = np.loadtxt(open(join(cfg['data_dir'], name+'_H.csv'), 'r'), delimiter=',')
    return (V, W, H)


def store_uci(D, name=str(date.today()), cfg=config.default_config()):
    print('Storing data in UCI format.')
    print('Destination:', cfg['data_dir'])
    print('Collection name:', name)
    N, M = D.shape
    nw = D.sum()
    with open(join(cfg['data_dir'], 'vocab.' + name + '.txt'), 'w') as f:
        print('Vocablurary...')
        for i in range(N):
            print(i, file=f)
    with open(join(cfg['data_dir'], 'docword.' + name + '.txt'), 'w') as f:
        print('DocWord matrix...')
        print(M, file=f)
        print(N, file=f)
        print(nw, file=f)
        cD = coo_matrix(D) # faster print
        for d, w, ndw in zip(cD.row, cD.col, cD.data):
            print(d + 1, w + 1, ndw, file=f)
    print('Done.')


def load_uci(name, cfg=config.default_config()):
    print('Loading data in UCI format.')
    print('From:', cfg['data_dir'])
    print('Collection name:', name)
    N = 0
    with open(join(cfg['data_dir'], 'docword.' + name + '.txt'), 'r') as f:
        M = int(f.readline())
        N = int(f.readline())
        D = np.zeros((N, M), dtype='float32')
        f.readline()
        for line in f:
            d, w, nwd = [int(x) for x in line.split(' ')]
            D[w-1, d-1] = D[w-1, d-1] + nwd
    vocab = np.arange(N).tolist()
    with open(join(cfg['data_dir'], 'vocab.' + name + '.txt'), 'r') as f:
        vocab = f.read().splitlines()
    return D, vocab
