#!/usr/bin/python2
# -*- coding: utf-8 -*-

from __future__ import print_function

import config
import generators
import methods
import measure
from common import cost, normalize_cols, hellinger, print_head, get_permute
from visualize import *
from data import *
from prepare import *

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as ml
from datetime import datetime, timedelta
from time import time
from copy import deepcopy


def gen_real(cfg=config.default_config()):
    N = cfg['N']
    T = cfg['T_0']
    M = cfg['M']
    gen_phi = getattr(generators, cfg['gen_phi'])
    cfg['rows'] = N
    cfg['cols'] = T
    cfg['sparsity'] = cfg['phi_sparsity']
    W_r = gen_phi(cfg)
    gen_theta = getattr(generators, cfg['gen_theta'])
    cfg['rows'] = T
    cfg['cols'] = M
    cfg['sparsity'] = cfg['theta_sparsity']
    H_r = gen_theta(cfg)
    #W_r = gen_matrix_sparse(N, T, 0.2)
    #W_r = gen_matrix_topic(cfg)
    #H_r = gen_matrix_sparse(T, M, 0.3)
    V = np.dot(W_r, H_r)
    return (V, W_r, H_r)


def gen_init(cfg=config.default_config()):
    N = cfg['N']
    T = cfg['T']
    M = cfg['M']
    gen_phi = getattr(generators, cfg['phi_init'])
    cfg['rows'] = N
    cfg['cols'] = T
    cfg['sparsity'] = cfg['phi_sparsity']
    W = gen_phi(cfg)
    gen_theta = getattr(generators, cfg['theta_init'])
    cfg['rows'] = T
    cfg['cols'] = M
    cfg['sparsity'] = cfg['theta_sparsity']
    H = gen_theta(cfg)
    return (W, H)


def run(V, W, H, W_r=None, H_r=None, cfg=config.default_config()):
    T = H.shape[0]
    eps = cfg['eps']
    schedule = cfg['schedule'].split(',')
    meas = cfg['measure'].split(',')
    val = np.zeros((cfg['max_iter']+2, len(meas)))
    hdist = np.zeros((cfg['max_iter']+2, 1))
    
    for i, fun_name in enumerate(meas):
        fun = getattr(measure, fun_name)
        val[0, i] = fun(V, np.dot(W, H))
    
    if cfg['compare_real']:
        #m = Munkres()
        idx = get_permute(W_r, H_r, W, H, cfg['munkres'])
        hdist[0] = hellinger(W[:, idx[:, 1]], W_r[:, idx[:, 0]]) / T
    if cfg['print_lvl'] > 1:
        print('Initial loss:', val[0])
    status = 0
    methods_num = len(schedule)
    it = -1
    for it in range(cfg['max_iter']):
        if cfg['print_lvl'] > 1:
            print('Iteration', it+1)
        W_old = deepcopy(W)
        H_old = deepcopy(H)
        method_name = schedule[it % methods_num]
        if cfg['print_lvl'] > 1:
            print('Method:', method_name)
        method = getattr(methods, method_name)
        (W, H) = method(V, W, H, method_name, cfg)
        if (it+1) % cfg['normalize_iter'] == 0:
            W = normalize_cols(W)
            H = normalize_cols(H)
        for j, fun_name in enumerate(meas):
            fun = getattr(measure, fun_name)
            val[it+1, j] = fun(V, np.dot(W, H))
        
        if cfg['compare_real']:
            idx = get_permute(W_r, H_r, W, H, cfg['munkres'])
            hdist[it+1] = hellinger(W[:, idx[:, 1]], W_r[:, idx[:, 0]]) / T
        
        if cfg['print_lvl'] > 1:
            print(val[it+1])
        if all(val[it, :] < eps):
            if cfg['print_lvl'] > 1:
                print('By cost.')
            status = 1
            break
        if abs(W_old - W).max() < eps and abs(H_old - H).max() < eps:
            if cfg['print_lvl'] > 1:
                print('By argument.')
            status = 2
            break
        #del W_old
        #del H_old
    if cfg['print_lvl'] > 1:
        print('Final:')
    W = normalize_cols(W)
    H = normalize_cols(H)
    for j, fun_name in enumerate(meas):
        fun = getattr(measure, fun_name)
        val[it+2:, j] = fun(V, np.dot(W, H))
    
    if cfg['compare_real']:
        idx = get_permute(W_r, H_r, W, H, cfg['munkres'])
        hdist[it+2:] = hellinger(W[:, idx[:, 1]], W_r[:, idx[:, 0]]) / T
    return (val, hdist, it, W, H, status)


def main(config_file='config.txt', results_file='results.txt', cfg=None):
    if cfg == None:
        cfg = config.load(config_file)
    if cfg['seed'] >= 0:
        np.random.seed(cfg['seed'])
    else:
        np.random.seed(None)
    eps = cfg['eps']
    N = cfg['N']
    T = cfg['T']
    M = cfg['M']
    vocab = None
    W_r = None
    H_r = None
    if cfg['run_info'] == 'results' or cfg['run_info'] == 1:
        cfg['print_lvl'] = 1
    elif cfg['run_info'] == 'run' or cfg['run_info'] == 2:
        cfg['print_lvl'] = 2
    else:
        cfg['print_lvl'] = 0
    if cfg['print_lvl'] > 0:
        print('Generating...')
    if cfg['load_data'] == 'uci' or cfg['load_data'] == 2:
        V, vocab = load_uci(cfg['data_name'], cfg)
        V = normalize_cols(V)
        N, M = V.shape
        cfg['N'], cfg['M'] = V.shape
        print('Size:', N, M)
    elif cfg['load_data'] == 'csv' or cfg['load_data'] == 1:
        _, W_r, H_r = load_csv(cfg['gen_name'], cfg)
        #plt.matshow(1-W_r, cmap=plt.cm.gray)
        #plt.title('real')
        V, vocab = load_uci(cfg['gen_name'], cfg)
        V = normalize_cols(V)
        N, M = V.shape
        cfg['N'], cfg['M'] = V.shape
        print('Size:', N, M)
        cfg['T_0'] = W_r.shape[1]
    else:
        V, W_r, H_r = gen_real(cfg)
    print('Checking assumption on V:', np.sum(V, axis=0).max())
    
    #tp = '0_5_100_16_500'
    #V_filename = 'datasets/V.' + tp + '.txt.csv'
    #W_filename = 'datasets/W.' + tp + '.txt.csv'
    #H_filename = 'datasets/H.' + tp + '.txt.csv'
    
    #V = np.loadtxt(V_filename, delimiter=',')
    #W_r = np.loadtxt(W_filename, delimiter=',')
    #H_r = np.loadtxt(H_filename, delimiter=',')
    #show_matrices(W_r, H_r)
    #plt.savefig('tm_tests/real' + tp + '.eps', format='eps')

    res = [0] * cfg['runs']
    finals = [0] * cfg['runs']
    hdist_runs = [0] * cfg['runs']
    exp_time = [0] * cfg['runs']
    meas = cfg['measure'].split(',')
    meas_name = [''] * len(meas)
    for i, f_name in enumerate(meas):
        f = getattr(measure, f_name + '_name')
        meas_name[i] = f()
    print('Measures:', meas_name)
    if cfg['compare_methods']:
        methods = cfg['schedule'].split(',')
        nmethods = len(methods)
    for r in range(cfg['runs']):
        if cfg['print_lvl'] > 0:
            print('Run', r+1)
        #(W, H) = gen_init(cfg)
        if cfg['print_lvl'] > 0:
            print('  Starting...')
        
        labels = None
        st = time()
        if r >= cfg['prepare'] and cfg['prepare'] >= 0 and cfg['prepare_method'] > 0:
            print('Preparing data...')
            if cfg['prepare_method'] == 1:
                W = anchor_words(V, 'L2', cfg)
                print('Solving for H')
                H = linalg.solve(np.dot(W.T, W) + np.eye(W.shape[1]) * eps, np.dot(W.T, V))
                H[H < eps] = 0
                H = normalize_cols(H)
            elif cfg['prepare_method'] == 2:
                centroids, labels = reduce_cluster(V.T, cfg['T'], cfg)
                W = centroids.T
                W[W < eps] = 0
                W = normalize_cols(W)
                print('Solving for H')
                H = linalg.solve(np.dot(W.T, W) + np.eye(W.shape[1]) * eps, np.dot(W.T, V))
                H[H < eps] = 0
                H = normalize_cols(H)
            elif cfg['prepare_method'] == 3:
                centroids, labels = reduce_cluster(V, cfg['num_clusters'], cfg)
                W = anchor_words(centroids, 'L2', cfg)
                print('Solving for H')
                H = linalg.solve(np.dot(W.T, W) + np.eye(W.shape[1]) * eps, np.dot(W.T, normalize_cols(centroids)))
                H[H < eps] = 0
                H = normalize_cols(H)
                W = restore_cluster(W, labels, cfg)
            elif cfg['prepare_method'] >= 4 and cfg['prepare_method'] <= 6:
                if cfg['prepare_method'] == 4:
                    red = reduce_tsne(V, to_dim=4)
                elif cfg['prepare_method'] == 5:
                    red = reduce_tsne(V, to_dim=3)
                elif cfg['prepare_method'] == 6:
                    red = reduce_tsne(V, to_dim=2)
                centroids, labels = reduce_cluster(red, cfg['num_clusters'], cfg)
                nearest_words = find_nearest(red, centroids, labels)
                V_reduced = normalize_cols(V[nearest_words, :])
                W = anchor_words(V_reduced, 'L2', cfg)
                print('Solving for H')
                H = linalg.solve(np.dot(W.T, W) + np.eye(W.shape[1]) * eps, np.dot(W.T, V_reduced))
                H[H < eps] = 0
                H = normalize_cols(H)
                W = restore_cluster(W, labels, cfg)
            elif cfg['prepare_method'] == 10:
                centroids, labels = reduce_multi_cluster(V, cfg['num_clusters'], cfg)
                W = anchor_words(centroids, 'L2', cfg)
                print('Solving for H')
                H = linalg.solve(np.dot(W.T, W) + np.eye(W.shape[1]) * eps, np.dot(W.T, normalize_cols(centroids)))
                H[H < eps] = 0
                H = normalize_cols(H)
                #W = restore_multi_cluster(W, labels, cfg)
                W = linalg.solve(dot(H, H.T) + eye(H.shape[0]) * eps, dot(H, V.T)).T
                W[W < eps] = 0
                W = normalize_cols(W)
        else:
            (W, H) = gen_init(cfg)
            #cur_frob = measure.frobenius(V, np.dot(W, H))
            #for init_it in xrange(200):
            #    W_new, H_new = gen_init(cfg)
            #    if measure.frobenius(V, np.dot(W_new, H_new)) #< cur_frob:
            #        W = deepcopy(W_new)
            #        H = deepcopy(H_new)
        se = time() - st
        print('Preparing took time:', timedelta(seconds=se))
        #labels=None
        #print('Preparing data...')
        #centroids, labels = reduce_cluster(V, cfg['T'], cfg)
        #H = centroids
        #H[H < eps] = 0
        #H = normalize_cols(H)
        #print('Solving for W')
        #W = linalg.solve(dot(H, H.T) + eye(H.shape[0]) * eps, dot(H, V.T)).T
        #W[W < eps] = 0
        #W = normalize_cols(W)
        
        #centroids, labels = reduce_cluster(V, cfg['num_clusters'], cfg)
        #W = anchor_words(centroids, 'L2', cfg)
        #print('Solving for H')
        #H = linalg.solve(np.dot(W.T, W) + np.eye(W.shape[1]) * eps, np.dot(W.T, normalize_cols(centroids)))
        #H[H < eps] = 0
        #H = normalize_cols(H)
        #W = restore_cluster(W, labels, cfg)
        
        #W = anchor_words(V, 'L2', cfg)
        #print('Solving for H')
        #H = linalg.solve(np.dot(W.T, W) + np.eye(W.shape[1]) * eps, np.dot(W.T, V))
        #H[H < eps] = 0
        #H = normalize_cols(H)
        
        #red = reduce_tsne(V, to_dim=3)
        #centroids, labels = reduce_cluster(red, cfg['num_clusters'], cfg)
        #print('c', centroids.shape, 'l', labels.shape)
        #nearest_words = find_nearest(red, centroids, labels)
        #print('nw:', nearest_words.shape)
        #V_reduced = V[nearest_words, :]
        #print('Vr', V_reduced.shape)
        #W = anchor_words(V_reduced, 'L2', cfg)
        #print('Solving for H')
        #H = linalg.solve(np.dot(W.T, W) + np.eye(W.shape[1]) * eps, np.dot(W.T, V_reduced))
        #H[H < eps] = 0
        #H = normalize_cols(H)
        #W = restore_cluster(W, labels, cfg)
        
        
        if cfg['compare_prepare'] > 0:
            if r > 0:
                print('Preparing data...')
                if r == 1:
                    W = anchor_words(V, 'L2', cfg)
                    print('Solving for H')
                    H = linalg.solve(np.dot(W.T, W) + np.eye(W.shape[1]) * eps, np.dot(W.T, V))
                    H[H < eps] = 0
                    H = normalize_cols(H)
                elif r == 2:
                    centroids, labels = reduce_cluster(V, cfg['T'], cfg)
                    H = centroids
                    H[H < eps] = 0
                    H = normalize_cols(H)
                    print('Solving for W')
                    W = linalg.solve(dot(H, H.T) + eye(H.shape[0]) * eps, dot(H, V.T)).T
                    W[W < eps] = 0
                    W = normalize_cols(W)
                elif r == 3:
                    centroids, labels = reduce_cluster(V, cfg['num_clusters'], cfg)
                    W = anchor_words(centroids, 'L2', cfg)
                    print('Solving for H')
                    H = linalg.solve(np.dot(W.T, W) + np.eye(W.shape[1]) * eps, np.dot(W.T, normalize_cols(centroids)))
                    H[H < eps] = 0
                    H = normalize_cols(H)
                    W = restore_cluster(W, labels, cfg)
                elif r >= 4 and r <= 6:
                    if r == 4:
                        red = reduce_tsne(V, to_dim=4)
                    elif r == 5:
                        red = reduce_tsne(V, to_dim=3)
                    elif r == 6:
                        red = reduce_tsne(V, to_dim=2)
                    centroids, labels = reduce_cluster(red, cfg['num_clusters'], cfg)
                    nearest_words = find_nearest(red, centroids, labels)
                    V_reduced = V[nearest_words, :]
                    W = anchor_words(V_reduced, 'L2', cfg)
                    print('Solving for H')
                    H = linalg.solve(np.dot(W.T, W) + np.eye(W.shape[1]) * eps, np.dot(W.T, V_reduced))
                    H[H < eps] = 0
                    H = normalize_cols(H)
                    W = restore_cluster(W, labels, cfg)
        if cfg['compare_methods'] > 0:
            cfg['schedule'] = methods[r % nmethods]
        start = time()
        (val, hdist, it, W, H, status) = run(V, W, H , W_r, H_r, cfg)
        stop = time()
        print('Run time:', timedelta(seconds=stop - start))
        exp_time[r] = stop - start
        res[r] = val
        hdist_runs[r] = hdist
        if cfg['print_lvl'] > 0:
            print('  Result:', val[-1, :])
        for i, fun_name in enumerate(cfg['finals'].split(',')):
            #val = np.array([r[:, i] for r in res])
            fun = getattr(measure, fun_name)
            name, val = fun(W, H)
            print(name, ':', val)
    print(cfg['experiment'])
    if cfg['experiment'] == '':
        exp_name = 'test'
    else:
        exp_name = cfg['experiment']
    #if cfg['save_results']:
    #    if cfg['save_file']:
    #        results_file = cfg['save_file']
    #    with open(results_file, 'w') as rf:
    #        print_head(rf)
    #        print('# Generated on {}'.format(datetime.today()), file=rf)
    #        print('# Experiments config:', file=rf)
    #        print('#   Number of experiments: {}'.format(cfg['runs']), file=rf)
    #        print('#   Methods schedule: {}'.format(cfg['schedule']), file=rf)
    #        print('#   Iterations number: {}'.format(cfg['max_iter']), file=rf)
    #        print('#   All experiments done in {}'.format(
    #            timedelta(seconds=sum(exp_time))), file=rf)
    #        print_head(rf)
    #        for r in range(cfg['runs']):
    #            print('# Run #{}. Done in {}'.format(r+1, 
    #                timedelta(seconds=exp_time[r])), file=rf)
    #            [print(val, file=rf) for val in res[:, r]]
    #            print_head(rf)
    if cfg['show_results']:
        if not os.path.exists(cfg['result_dir']):
            os.makedirs(cfg['result_dir'])
        np.savetxt(join(cfg['result_dir'], cfg['experiment'] + '_W.csv'), W)
        #show_topics(W, 25, vocab=vocab)
        save_topics(W, join(cfg['result_dir'], cfg['experiment'] + '_topics.txt'), vocab)
        #plot_matrix(V, 'Documents', labels=labels, vocab=vocab)
        #filename = os.path.join(cfg['result_dir'], cfg['experiment']+'_V.eps')
        #plt.savefig(filename, format='eps')
        #plot_matrix(W, u'Распределение слов в темах', labels, vocab)
        #filename = os.path.join(cfg['result_dir'], cfg['experiment']+'_W.pdf')
        #plt.savefig(filename, format='pdf')
        
        for i, fun_name in enumerate(cfg['measure'].split(',')):
            val = np.array([r[:, i] for r in res])
            fun = getattr(measure, fun_name + '_name')
            plot_measure(val.T, fun())
            filename = os.path.join(cfg['result_dir'], cfg['experiment']+'_'+fun_name+'.pdf')
            plt.savefig(filename, format='pdf')
        if cfg['compare_real']:
            print('Hellinger res:', hdist_runs[0][-1,0])
            plot_measure(np.array([r[:, 0] for r in hdist_runs]).T, measure.hellinger_name())
            show_matrices_recovered(W_r, H_r, W, H, cfg, permute=True)
            #plt.savefig('tm_tests/recovered_cnmf_' + tp + '.eps', format='eps')
        #plt.show()
    return res

if __name__ != '__main__':
    main()
    plt.show()

if __name__ == '__main__':
    cfg = config.load()
    if not os.path.exists(cfg['result_dir']):
        os.makedirs(cfg['result_dir'])
    if 1:
        cfg['compare_methods'] = 1
        cfg['save_results'] = 0
        cfg['show_results'] = 0
        cfg['prepare'] = 4
        cfg['runs'] = 8
        #cfg['T'] = 25
        res = main(cfg=cfg)
        colors = ['r', 'b', 'g', 'm']
        markers = ['o', '^', 'd', (5,1)]
        for i, fun_name in enumerate(cfg['measure'].split(',')):
            plt.figure()
            val = np.array([r[:, i] for r in res])
            fun = getattr(measure, fun_name + '_name')
            plt.ylabel(fun(), fontsize=13)
            if fun_name == 'perplexity':
                if cfg['data_name'] == 'nips':
                    plt.ylim(1250, 2750)
                    plt.yticks(np.arange(1250, 2751, 250))
                else:
                    plt.ylim(1000, 2500)
                    plt.yticks(np.arange(1000, 2501, 250))
            for j in xrange(4):
                plt.plot(val[j, :], linewidth=2, c=colors[j], marker='o', linestyle='--')
            for j in xrange(4):
                plt.plot(val[j+4, :], linewidth=2, c=colors[j], marker='^', linestyle='-')
            plt.legend(['ALS', 'HALS', 'MU', 'PLSA'])
            plt.draw()
            filename = os.path.join(cfg['result_dir'], cfg['experiment']+'_'+fun_name+'.pdf')
            plt.savefig(filename, format='pdf')
        plt.show()
    else:
        Ts = np.arange(cfg['T_begin'], cfg['T_end']+1, cfg['T_step'])
        #final_r = np.zeros()
        pr_methods = [0]#, 1, 10]
        colors = ['r', 'b', 'g', 'm']
        markers = ['o', '^', 'd', (5,1)]
        final_res = np.zeros((2, len(pr_methods), 4, len(Ts)))
        m = cfg['schedule']
        for p_it, p in enumerate(pr_methods):
            for it, T in enumerate(Ts):
                print('============')
                print('  T=',T)
                print('============')
                cfg['T'] = T
                cfg['schedule'] = m
                cfg['prepare_method'] = p
                cfg['num_clusters'] = int(T * 3.5)
                res = main(cfg=cfg)
                for i, fun_name in enumerate(cfg['measure'].split(',')):
                    val = np.array([r[-1, i] for r in res])
                    fun = getattr(measure, fun_name + '_name')
                    final_res[i, p_it, :, it] = val
        np.savetxt(os.path.join(cfg['result_dir'], cfg['experiment']+'_res.csv'), np.reshape(final_res, (2 * len(pr_methods) * 4, len(Ts))))
        plt.figure()
        plt.ylabel(u'Норма Фробениуса', fontsize=13)
        plt.xlabel(u'Число тем', fontsize=13)
        for p_it in xrange(len(pr_methods)):
            for met_i in xrange(4):
                plt.plot(Ts, final_res[0, p_it, met_i, :], c=colors[met_i], linewidth=2, marker=markers[p_it])
                plt.draw()
        plt.legend(['ALS', 'HALS', 'MU', 'PLSA'])
        filename = os.path.join(cfg['result_dir'], cfg['experiment']+'_'+'frobenius'+'.pdf')
        plt.savefig(filename, format='pdf')
        
        plt.figure()
        plt.ylabel(u'Перплексия', fontsize=13)
        plt.xlabel(u'Число тем', fontsize=13)
        for p_it in xrange(len(pr_methods)):
            for met_i in xrange(4):
                plt.plot(Ts, final_res[1, p_it, met_i, :], c=colors[met_i], linewidth=2, marker=markers[p_it])
                plt.draw()
        plt.legend(['ALS', 'HALS', 'MU', 'PLSA'])
        filename = os.path.join(cfg['result_dir'], cfg['experiment']+'_'+'perplexity'+'.pdf')
        plt.savefig(filename, format='pdf')
        plt.show()
