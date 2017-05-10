#!/usr/bin/env python2.7
from matplotlib import pyplot as plt
from matplotlib import rc
from get_tables_and_plots import METHODS

from pprint import pprint

import numpy as np
import pandas as pd

rc('text', usetex=True)


def plot(cpu_values, cpu_min_err, cpu_max_err, gpu_values, method_selection):
    plt.figure()

    data = cpu_values.loc[24, method_selection]
    data.index = pd.Index([METHODS[x] for x in data.index])
    data['Recursive (CUDA)'] = gpu_values[0]

    data /= 2 ** 24

    err_min = cpu_min_err.loc[24, method_selection]
    err_min['Recursive (CUDA)'] = gpu_values[1]

    err_min /= 2**24

    err_min = (data.values - err_min.values).reshape((1,1, err_min.shape[0]))

    err_max = cpu_max_err.loc[24, method_selection]
    err_max['Recursive (CUDA)'] = gpu_values[2]

    err_max /= 2 ** 24

    err_max = (err_max.values - data.values).reshape((1,1, err_max.shape[0]))

    err = np.concatenate((err_min, err_max), axis=1)
    data.plot.bar(rot=45, yerr=err, capsize=2)

    ax = plt.gca()
    ax.xaxis.set_tick_params(length=0)

    ax.set_ylabel('$\\frac{t}{n}$', size=16, rotation=0)
    ax.yaxis.set_label_coords(-0.08,0.5)


    plt.tight_layout()

    plt.savefig('gpu_runtime24.pdf')

if __name__ == '__main__':

    method_selection = ['COBRAShuffle', 'OutOfPlaceCOBRAShuffle',
                        'SemiRecursiveShuffle', 'RecursiveShuffle']

    values_cuda = [0.400539, 0.399458, 0.409658]

    run_times = pd.read_csv('g++_run_times.tsv', sep='\t', index_col='N_BITS')
    run_times_min_err = pd.read_csv('g++_run_times_min_err.tsv', sep='\t', index_col='N_BITS')
    run_times_max_err = pd.read_csv('g++_run_times_max_err.tsv', sep='\t', index_col='N_BITS')

    plot(run_times, run_times_min_err, run_times_max_err, values_cuda, method_selection)