#!/usr/bin/python2
# -*- coding: utf-8 -*-

from __future__ import print_function, division

import matplotlib as ml
import matplotlib.pyplot as plt
from common import get_permute
import numpy as np

import config
from bhtsne import bh_tsne
from annoter import AnnoteFinder


def plot_measure(measure, name=None, title=None, fig=None):
    plt.figure(fig)
    #plt.xlabel('Iteration', fontsize=32)
    plt.xticks(np.arange(0, measure.shape[0], 5))
    if name != None:
        plt.ylabel(name, fontsize=13)
    if title != None:
        plt.title(title)
    plt.plot(measure, linewidth=2)
    #plt.legend(['no preparation', 'anchor for W', 'kmeans for H', 'kmeans + anchor for W', 'tSNE4 for W', 'tSNE3 for W', 'tSNE2 for W'])
    #plt.legend(['ALS', 'Block', 'PLSA', 'ALS with preparation', 'Block with preparation', 'PLSA with preparation'])
    #plt.legend(['ALS', 'MU', 'Block', 'PLSA'])#, 'ALS with preparation', 'Block with preparation', 'PLSA with preparation'])
    plt.draw()


def show_matrices_recovered(W_r, H_r, W, H, cfg=config.default_config(), permute=True):
    if permute:
        idx = get_permute(W_r, H_r, W, H, cfg['munkres'])
    else:
        idx = np.array([range(W.shape[1]), range(W.shape[1])])
    #f, axarr = plt.subplots(nrows=1, ncols=1)
    #axarr[0, 0].imshow(1-W_r, cmap='gray')
    #axarr[0, 0].set_title('W real')
    #axarr[0, 1].imshow(1-H_r, cmap='gray')
    #axarr[0, 1].set_title('H real')
    plt.matshow(1-W[:, idx[:, 1]], cmap=plt.cm.gray)
    #plt.gray()
    #plt.title(u'Восстановленная W')
    #axarr[0].imshow(1-W[:, idx[:, 1]], cmap='gray')
    #axarr[0].set_title('W recovered')
    #axarr[1].imshow(1-H[idx[:, 1], :], cmap='gray')
    #axarr[1].set_title('H recovered')


def show_matrix(A, name=None, fig=None):
    if not fig:
        plt.figure()
    else:
        plt.figure(fig)
    plt.imshow(1-W)
    plt.gray()
    if name:
        plt.title(name)
    plt.draw()


def plot_matrix(A, title=None, labels=None, vocab=None, fig=None):
    #cmap = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0.25, 0.75, 0], [0.25, 0, 0.75], [0, 0.5, 0.5], [0.75, 0.25, 0], [0.75, 0, 0.25], [0, 0.75, 0.25], [0, 0.25, 0.75]]
    cmap = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [0, 1, 1]]#, [1, 0, 1], [1, 0.5, 0], [0.5, 0, 1], [0.5, 1, 0], [0.98, 0.39, 0]]
    cl = len(cmap)
    markers = ['o', 'd', '>', (5,1)]
    ml = len(markers)
    if not vocab:
        vocab = range(A.shape[0])
    res = np.array([x for x in bh_tsne(A, verbose=True)])
    if not fig:
        plt.figure()
    else:
        plt.figure(fig)
    if title:
        plt.title(title)
    #if all(labels) != None:
    #    plt.scatter(res[:, 0], res[:, 1], s=20, c=labels, alpha=0.5)
    #else:
    #    plt.scatter(res[:, 0], res[:, 1], s=20, alpha=0.5)
    for col in xrange(A.shape[1]):
        top_word = np.argmax(A[:, col])
        mk = (col // cl) % ml
        colors = np.zeros((A.shape[0], 4))
        colors[:, 0] = cmap[col % cl][0]
        colors[:, 1] = cmap[col % cl][1]
        colors[:, 2] = cmap[col % cl][2]
        colors[:, -1] = (A[:, col] / A[top_word, col])
        plt.scatter(res[:, 0], res[:, 1], c=colors, marker=markers[mk], s=30, edgecolor='none')
        plt.scatter(res[top_word, 0], res[top_word, 1], c=cmap[col % cl], marker=markers[mk], s=30, edgecolor='none', label=u'тема #'+str(col))
    if all(vocab) != None:
        af =  AnnoteFinder(res[:, 0], res[:, 1], vocab, xtol=0.1, ytol=0.1)
        plt.connect('button_press_event', af)
    plt.legend(scatterpoints=1,
           loc='best',
           ncol=3,
           fontsize=9)
    plt.draw()
    return res


def show_topics(W, words_number=5, vocab=None, topic_idxs=None):
    if not vocab:
        vocab = range(W.shape[0])
    if not topic_idxs:
        topic_idxs = range(W.shape[1])
    print('Top', words_number, 'words per topics:')
    for topic_idx in topic_idxs:
        words = np.argsort(-W[:, topic_idx])
        print('topic #', topic_idx, ':', [vocab[i] for i in words[:words_number]])


def save_topics(W, filename, vocab=None, topic_idxs=None):
    if not vocab:
        vocab = range(W.shape[0])
    if not topic_idxs:
        topic_idxs = range(W.shape[1])
    with open(filename, 'w') as f:
        for topic_idx in topic_idxs:
            words = np.argsort(-W[:, topic_idx])
            print('topic #', topic_idx, ':', file=f)
            str_words = ['  ' + str(vocab[i]) + ':' + str(W[i, topic_idx]) for i in words]
            print('\n'.join(str_words), file=f)

if __name__ != '__main__':
    ml.rcdefaults() # cбрасываем настройки на "по умолчанию"
    ml.rcParams['font.family'] = 'fantasy'
    ml.rcParams['font.fantasy'] = 'Times New Roman', 'Ubuntu','Arial','Tahoma','Calibri'
else:
    import os
    from os.path import join
    from data import load_uci
    ml.rcdefaults() # cбрасываем настройки на "по умолчанию"
    ml.rcParams['font.family'] = 'fantasy'
    ml.rcParams['font.fantasy'] = 'Times New Roman', 'Ubuntu','Arial','Tahoma','Calibri'
    cfg = config.load()
    _, vocab = load_uci(cfg['data_name'], cfg)
    W = np.loadtxt(join(cfg['result_dir'], cfg['experiment'] + '_W.csv'))
    res = plot_matrix(W, u'Распределение слов в темах', vocab=vocab)
    filename = join(cfg['result_dir'], cfg['experiment']+'_W.pdf')
    plt.savefig(filename, format='pdf')
    plt.show()
