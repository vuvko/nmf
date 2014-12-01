#!/usr/bin/python2
# -*- coding: utf-8 -*-

import config
import main


def experiment():
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
    
if __name__ == '__main__':
    experiment()
