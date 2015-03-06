#!/usr/bin/python2
# -*- coding: utf-8 -*-

import matplotlib as ml
import matplotlib.pyplot as plt
from common import get_permute
import numpy as np

import config
from bhtsne import bh_tsne


def plot_measure(measure, name=None, title=None, fig=None):
    plt.figure(fig)
    #plt.xlabel('Iteration', fontsize=32)
    plt.xticks(np.arange(0, measure.shape[0], 10))
    if name != None:
        plt.ylabel(name, fontsize=13)
    if title != None:
        plt.title(title)
    plt.plot(measure, linewidth=2)
    #plt.legend(['no preparation', 'anchor for W', 'kmeans for H', 'kmeans + anchor for W'])
    plt.legend(['ALS', 'MU', 'Block'])
    plt.draw()


def show_matrices_recovered(W_r, H_r, W, H, munkres, cfg=config.default_config(), permute=True):
    if permute:
        idx = get_permute(W_r, H_r, W, H, munkres, cfg['munkres'])
    else:
        idx = np.array([range(W.shape[1]), range(W.shape[1])])
    #f, axarr = plt.subplots(nrows=1, ncols=1)
    #axarr[0, 0].imshow(1-W_r, cmap='gray')
    #axarr[0, 0].set_title('W real')
    #axarr[0, 1].imshow(1-H_r, cmap='gray')
    #axarr[0, 1].set_title('H real')
    plt.figure()
    plt.imshow(1-W[:, idx[:, 1]])
    plt.gray()
    plt.title('W recovered')
    #axarr[0].imshow(1-W[:, idx[:, 1]], cmap='gray')
    #axarr[0].set_title('W recovered')
    #axarr[1].imshow(1-H[idx[:, 1], :], cmap='gray')
    #axarr[1].set_title('H recovered')
    plt.draw()


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


def plot_matrix(A, title=None, labels=None, fig=None):
    res = np.array([x for x in bh_tsne(A, verbose=True)])
    if not fig:
        plt.figure()
    else:
        plt.figure(fig)
    if title:
        plt.title(title)
    if labels == None:
        plt.scatter(res[:, 0], res[:, 1], s=20, alpha=0.5)
    else:
        plt.scatter(res[:, 0], res[:, 1], s=20, c=labels, alpha=0.5)
    plt.draw()


def show_topics(W, words_number=5, vocab=None, topic_idxs=None):
    if not vocab:
        vocab = range(W.shape[0])
    if not topic_idxs:
        topic_idxs = range(W.shape[1])
    print('Top', words_number, 'words per topics:')
    for topic_idx in topic_idxs:
        words = np.argsort(-W[:, topic_idx])
        print('topic #', topic_idx, ':', [vocab[i] for i in words[:words_number]])

if __name__ != '__main__':
    font = {'family' : 'normal',
            'weight' : 'normal',
            'size'   : 12}
    ml.rc('font', **font)
