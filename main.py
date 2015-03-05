#!/usr/bin/python2
# -*- coding: utf-8 -*-

import config
import generators
import methods
import measure
from common import cost, normalize_cols, hellinger, print_head, get_permute
from visualize import *
from data import *
from prepare import *

#from munkres import Munkres
import numpy as np
import matplotlib.pyplot as pl
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
        m = Munkres()
        idx = get_permute(W_r, H_r, W, H, m, cfg['munkres'])
        hdist[0] = hellinger(W[:, idx[:, 1]], W_r[:, idx[:, 0]]) / T
    
    status = 0
    methods_num = len(schedule)
    for it in range(cfg['max_iter']):
        if cfg['print_lvl'] > 1:
            print('Iteration', it+1)
        W_old = deepcopy(W)
        H_old = deepcopy(H)
        method_name = schedule[it % methods_num]
        if cfg['print_lvl'] > 1:
            print('Method:', method_name)
        method = getattr(methods, method_name)
        (W, H) = method(V, W, H, methode_name, cfg)
        if (it+1) % cfg['normalize_iter'] == 0:
            W = normalize_cols(W)
            H = normalize_cols(H)
        for j, fun_name in enumerate(meas):
            fun = getattr(measure, fun_name)
            val[it+1, j] = fun(V, np.dot(W, H))
        
        if cfg['compare_real']:
            idx = get_permute(W_r, H_r, W, H, m, cfg['munkres'])
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
        idx = get_permute(W_r, H_r, W, H, m, cfg['munkres'])
        hdist[it+2:] = hellinger(W[:, idx[:, 1]], W_r[:, idx[:, 0]]) / T
    return (val, hdist, it, W, H, status)


def main(config_file='config.txt', results_file='results.txt', cfg=None):
    if cfg == None:
        cfg = config.load(config_file)
    eps = cfg['eps']
    N = cfg['N']
    T = cfg['T']
    M = cfg['M']
    #m = Munkres()
    m = None
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
        W_r = None
        H_r = None
        N, M = V.shape
        cfg['N'], cfg['M'] = V.shape
    elif cfg['load_data'] == 'csv' or cfg['load_data'] == 1:
        V, W_r, H_r = load(cfg['data_name'], cfg)
        N, M = V.shape
        cfg['N'], cfg['M'] = V.shape
        cfg['T_0'] = W_r.shape[1]
    else:
        V, W_r, H_r = gen_real(cfg)
    
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
    hdist_runs = [0] * cfg['runs']
    exp_time = [0] * cfg['runs']
    meas = cfg['measure'].split(',')
    meas_name = [''] * len(meas)
    for i, f_name in enumerate(meas):
        f = getattr(measure, f_name + '_name')
        meas_name[i] = f()
    print('Measures:', meas_name)
    for r in range(cfg['runs']):
        if cfg['print_lvl'] > 0:
            print('Run', r+1)
        (W, H) = gen_init(cfg)
        if cfg['print_lvl'] > 0:
            print('  Starting...')
        if cfg['prepare'] > 0:
            print('Preparing data...')
            W, H = anchor_words(reduce_cluster(V, cfg), 'L2', cfg)
        start = time()
        (val, hdist, it, W, H, status) = run(V, W, H , W_r, H_r, cfg)
        stop = time()
        exp_time[r] = stop - start
        res[r] = val
        hdist_runs[r] = hdist
        if cfg['print_lvl'] > 0:
            print('  Result:', val[-1, :])
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
        show_topics(W, 10, vocab=vocab)
        for i, fun_name in enumerate(cfg['measure'].split(',')):
            val = np.array([r[:, i] for r in res])
            fun = getattr(measure, fun_name + '_name')
            plot_measure(val.T, fun())
            plt.savefig('tests/'+cfg['experiment']+'.eps', format='eps')
        if cfg['compare_real']:
            plot_measure(np.array([r[:, 0] for r in hdist_runs]).T, measure.hellinger_name())
            show_matrices_recovered(W_r, H_r, W, H, m, cfg, permute=True)
            #plt.savefig('tm_tests/recovered_cnmf_' + tp + '.eps', format='eps')
        pl.show()
    return res

if __name__ == '__main__':
    main()
