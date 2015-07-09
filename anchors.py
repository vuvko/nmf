from __future__ import division

import numpy as np
import sys
import os
import errno
from numpy.random import RandomState
import random_projection as rp
import gram_schmidt_stable as gs 

def findAnchors(Q, K, params, candidates=None):
    eps = params['eps']
    # Random number generator for generating dimension reduction
    if params['seed'] > 0:
        prng_W = RandomState(params['seed'])
    else:
        prng_W = RandomState(None)
    #checkpoint_prefix = params['checkpoint_prefix']
    new_dim = params['new_dim']
    
    if candidates == None:
        candidates = np.arange(Q.shape[0])

    # row normalize Q
    row_sums = Q.sum(1)
    row_sums[row_sums < eps] = eps
    for i in xrange(len(Q[:, 0])):
        Q[i, :] = Q[i, :]/float(row_sums[i])

    # Reduced dimension random projection method for recovering anchor words
    Q_red = rp.Random_Projection(Q.T, new_dim, prng_W)
    Q_red = Q_red.T
    (anchors, anchor_indices) = gs.Projection_Find(Q_red, K, candidates)

    # restore the original Q
    for i in xrange(len(Q[:, 0])):
        Q[i, :] = Q[i, :]*float(row_sums[i])

    return anchor_indices


