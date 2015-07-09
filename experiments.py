#!/usr/bin/python2
# -*- coding: utf-8 -*-

from __future__ import print_function

import config
import main
import matplotlib.pyplot as plt


def experiment_old():
    cfg = config.load('config.txt')
    doubles = ['als', 'mult'];
    prim = 'plsa'
    once = ['als', 'mult']
    Ts = [15]
    Tn = ['Te']
    
    norms = [1, 2]
    suff = ''
    all_res = []
    #doubles = []
    # double algorithms
    for alg in doubles:
        for norm in norms:
            for (i, T) in enumerate(Ts):
                if alg == 'als' and T > 15:
                    continue
                cfg['experiment'] = alg + '_' + prim + '_' + Tn[i] + '_' + str(norm) + suff
                cfg['T'] = T
                cfg['schedule'] = alg + ',' + prim
                cfg['normalize_iter'] = norm
                print('\n', cfg['experiment'], '\n')
                res = main.main(cfg=cfg)
    
    # one algorithm
    for alg in once:
        for (i, T) in enumerate(Ts):
            if alg == 'als' and T > 15:
                continue
            cfg['experiment'] = alg + '_' + Tn[i] + '_1' + suff
            cfg['T'] = T
            cfg['schedule'] = alg
            cfg['normalize_iter'] = 1
            print('\n', cfg['experiment'], '\n')
            res = main.main(cfg=cfg)


def experiment():
    cfg = config.load('config.txt')
    data = ['kos']
    #data = ['nips']
    methods = ['als', 'mult', 'my_als', 'plsa']
    Ts = [5, 10, 15, 20]
    prepares = [0, 1, 2, 3]
    cfg['max_iter'] = 20
    for d in data:
        for alg in methods:
            for T in Ts:
                for pr in prepares:
                    cfg['experiment'] = d+'_'+alg+'_20_T'+str(T)+'_p'+str(pr)
                    cfg['data_name'] = d
                    cfg['T'] = T
                    cfg['schedule'] = alg
                    cfg['prepare_method'] = pr
                    print('============\n', cfg['experiment'], '\n============')
                    res = main.main(cfg=cfg)
                    plt.close('all')
    
if __name__ == '__main__':
    experiment()
