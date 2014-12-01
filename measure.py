#!/usr/bin/python2
# -*- coding: utf-8 -*-

import config

from numpy import *


def perplexity(A, B):
    return exp(-sum(A * log(B + 1e-7)) / (sum(A) + 1e-7))


def perplexity_name():
    return 'Perplexity'


def frobenius(A, B):
   return sum((A - B) ** 2)


def frobenius_name():
    return 'Frobenius norm'


def kl(A, B):
    return sum(A * log(A / (B + 1e-7) + 1e-7) - A + B)


def kl_name():
    return 'KL-divergence'


def hellinger(A, B):
    return sum((sqrt(A) - sqrt(B)) ** 2) / 2


def hellinger_name():
    return 'Hellinger distance'
