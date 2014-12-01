#!/usr/bin/python2
# -*- coding: utf-8 -*-

from collections import defaultdict 


def default_config():
    '''
    Setting default parameters
    '''
    params = defaultdict(str)
    params['data_dir'] = 'data'
    params['eps'] = 1e-7
    params['max_iter'] = 50
    params['runs'] = 1
    params['run_info'] = 'run'
    params['schedule'] = 'als'
    params['normalize_iter'] = 1
    params['save_results'] = 0
    params['show_results'] = 1
    params['N'] = 100
    params['M'] = 250
    params['T'] = 15
    return params


def load(config_filename='config.txt'):
    '''
    Loading configuration file for scripts.
    
    Syntax of file must be as follows:
      <key> = <value> # <comment>
    '''
    config = default_config()
    with open(config_filename, 'r') as config_file:
        for idx, line in enumerate(config_file):
            names = line.split('#')[0].split('=')
            name = names[0].strip()
            if name == '':
                continue
            #if name in config:
            #    raise KeyError('Config file has multiple definitions' + 
            #    ' of key \'{}\' on line {}'.format(name, idx))
            if len(names) != 2:
                raise SyntaxError('Config file has syntax error on' +
                ' line {}'.format(idx+1))
            value = names[1].strip()
            types = [int, float, str]
            for t in types:
                try:
                    config[name] = t(value)
                    break
                except ValueError:
                    continue
    return config


def main():
    print('This is config loader module.')

if __name__ == '__main__':
    main()
else:
    print('Imported', __name__)
