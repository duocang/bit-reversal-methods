#!/usr/bin/env python2.7
"""
Helper script for automatic generation of result tables.
Simply call it without arguments, to do everything
"""
import os
import argparse
import math
import matplotlib.pyplot as plt
import subprocess

from matplotlib.cm import gist_rainbow as cm
from matplotlib import rc

import numpy as np
import platform

from collections import OrderedDict
from contextlib import contextmanager

try:
    import pandas as pd
except ImportError as e:
    print("this script needs a working installation of "
          "pandas in order to generate LaTex tables automatically")
    raise e

parser = argparse.ArgumentParser(description=__doc__)

parser.add_argument('--tex_tab', choices=['ctimes', 'rtimes', 'cpu'],
                    help='output a table in .tex format, for compile times, runtimes '
                         'or containing cpu specification')

parser.add_argument('--compiler', metavar="COMPILER", nargs=1,
                    required=False, default=['g++'],
                    choices=['g++', 'clang++'],
                    help='choose the compiler for which to generate the results (default=g++)'
                    )
parser.add_argument('--plot', choices=['ctimes', 'rtimes'], default=None,
                    help="generate plots for runtimes/compile times")

DIR = os.path.dirname(__file__)

# Methods dict is ordered; i.e. it determines the order of the methods appearing in the paper
#
# the keys denote the names of the methods in the implementation, the values appear as methodnames in the pdf
METHODS = OrderedDict([
    ('StockhamShuffle', 'Stockham'),
    ('NaiveShuffle', 'Bitwise'),
    ('TableShuffle', 'Bytewise'),
    ('LocalPairwiseShuffle', 'Pair bitwise'),
    ('OutOfPlaceCOBRAShuffle', 'COBRA (out-of-place)'),
    ('COBRAShuffle', 'COBRA (in-place)'),
    ('XORShuffle', 'XOR'),
    ('UnrolledShuffle', 'Unrolled'),
    ('RecursiveShuffle', 'Recursive'),
    ('SemiRecursiveShuffle', 'SemiRecursive'),
    ('OpenMPSemiRecursiveShuffle', 'SemiRecursive (OpenMP)')
])

cpu_cmds = {
    'L1i': '/sys/devices/system/cpu/cpu0/cache/index0/size',
    'L1d': '/sys/devices/system/cpu/cpu0/cache/index1/size',
    'L2': '/sys/devices/system/cpu/cpu0/cache/index2/size',
    'L3': '/sys/devices/system/cpu/cpu0/cache/index3/size',
    'MAX_SPEED': '/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq'
}

cpu_cmds_mac = {
    
    'L1i': "sysctl -a | grep l1i | perl -nle 'print $1 if /:.*?(\d+)/'",
    'L1d': "sysctl -a | grep l1d | perl -nle 'print $1 if /:.*?(\d+)/'",
    'L2': "sysctl -a | grep l2 | perl -nle 'print $1 if /:.*?(\d+)/'",
    'L3': "sysctl -a | grep l3 | perl -nle 'print $1 if /:.*?(\d+)/'",
    'MAX_SPEED': "system_profiler | grep Processor | perl -nle 'print $1 if /Processor Speed:.*?(\d+)/"
}

rc('text', usetex=True)


def fmt_val(val):

    if val in [None, 'None', '', 'nan', float('nan')] or math.isnan(val):
        return '-'
    else:
        val = float(val)
        exponent = int(math.floor(math.log10(val)))
        if exponent < -2:
            val *= 0.1 ** exponent
            return '{:.2f}e{:d}'.format(val, int(exponent))
        return '{:.2f}'.format(val)


def get_compile_times(compiler):
    return fmt_df(pd.read_csv(os.path.join(DIR, '{}_compile_times.tsv'.format(compiler)),
                              sep='\t', index_col=0))


def fmt_df(df):
    """
    bring dataframes to right format
    :return:
    """
    df.columns = pd.to_numeric(df.columns)
    df.columns.name = ''
    df.index = pd.Index([METHODS[x] for x in df.index])
    df.sort_index(level=METHODS.values(), inplace=True)
    return df.apply(pd.to_numeric, errors='coerce')


def get_run_times(compiler):

    values = fmt_df(pd.read_csv(os.path.join(DIR, '{}_run_times.tsv'.format(compiler)),
                                sep='\t', index_col=0).T)
    min_err = fmt_df(pd.read_csv(os.path.join(DIR, '{}_run_times_min_err.tsv'.format(compiler)),
                                 sep='\t', index_col=0).T)

    max_err = fmt_df(pd.read_csv(os.path.join(DIR, '{}_run_times_max_err.tsv'.format(compiler)),
                                 sep='\t', index_col=0).T)

    values = values.apply(adjust, axis=0)
    min_err = min_err.apply(adjust, axis=0)
    max_err = max_err.apply(adjust, axis=0)

    return values, min_err, max_err


@contextmanager
def tabular(f, width, index=True):
    if index:
        f.write('\\begin{tabular}{c|')
    else:
        f.write('\\begin{tabular}{')
    f.write('c' * width)
    f.write('}')
    yield
    f.write('\\end{tabular}')


def strip_data(f, latex_tab):
    """
    :param f: file
    :param str latex_tab:
    """
    for line in latex_tab.splitlines():
        if not any((line.startswith(x) for x in [
            '\\bottomrule', '\\toprule', '\\begin{tabular}', '\\end{tabular}'
        ])):
            f.write(line)
            f.write('\n')


def split(df, max_width):
    dfs = [df]
    while dfs[-1].shape[1] > max_width:
        df = dfs.pop()
        dfs.append(df.iloc[:, :max_width])
        dfs.append(df.iloc[:, max_width:])

    return dfs


def write_output(data, filename, max_width, fmt=False, index=True):
    if fmt:
        data = data.applymap(fmt_val)
    dfs = split(data, max_width)

    with open(os.path.join(DIR, filename), 'w') as f:
        with tabular(f, max_width, index):
            for part in dfs[:-1]:
                strip_data(
                    f, part.to_latex(longtable=False, index=index)
                )
                f.write('\\vspace{.5cm}\\\\')

            strip_data(
                f, dfs[-1].to_latex(longtable=False, index=index)
            )


def get_cpu_spec():
    sys_ = platform.system()
    if sys_ == 'Darwin':
        values = {}
        proc = subprocess.Popen(['sysctl', '-a'], stderr=subprocess.PIPE, stdout=subprocess.PIPE)
        stdout, stderr = proc.communicate()
        for line in stdout.splitlines():
            if 'l1d' in line:
                values['L1d'] = "{}K".format(int(line.split(":")[1]) / 1000)
            elif 'l1i' in line:
                values['L1i'] = "{}K".format(int(line.split(":")[1]) / 1000)
            elif 'l2' in line:
                values['L2'] = "{}K".format(int(line.split(":")[1]) / 1000)
            elif 'l3' in line:
                values['L3'] = "{}K".format(int(line.split(":")[1]) / 1000)
        
        proc = subprocess.Popen(['system_profiler'], stderr=subprocess.PIPE, stdout=subprocess.PIPE)
        stdout, stderr = proc.communicate()

        for line in stdout.splitlines():
            if 'Processor' in line:
                print line

        values = pd.Series(values)
        values['MAX_SPEED'] = '{}Ghz'.format(float(values['MAX_SPEED']) / 1000000)
        values = values.to_frame().transpose()
    else:
        values = pd.Series({name: open(file).read().strip() for name, file in cpu_cmds.iteritems()})
        values['MAX_SPEED'] = '{}Ghz'.format(float(values['MAX_SPEED']) / 1000000)

        with open('/proc/meminfo', 'r') as f:
            for line in f:
                if line.startswith('MemTotal'):
                    values['RAM'] = '{}GB'.format(int(line.strip().split()[-2]) / 1000000)

        values = values.to_frame().transpose()
    return values


def ls_iter(size):
    linestyles = ['-', '--', '-.', ':']
    l = len(linestyles)
    for i in xrange(size):
        yield linestyles[i % l]


def plot_compile_times(values, filename):
    fig = plt.figure()

    methods = METHODS.values()
    del methods[methods.index('SemiRecursive (OpenMP)')]

    l = len(methods)

    ls_it = ls_iter(l)

    for method in methods:

        color = cm(float(methods.index(method)) / l)
        ser = values.loc[method]
        y = ser.values
        x = ser.index.values
        plt.semilogy(x, y, label=method,
                     ls=next(ls_it),
                     color=color)

    plt.legend(ncol=3)

    ax = plt.gca()
    ax.set_ylabel('$t$', rotation=0)
    ax.set_xlabel('$b=\log(n)$')

    ax.yaxis.set_label_coords(-0.08,0.5)

    plt.tight_layout()

    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close(fig)


def plot_run_times2(values, err_min, err_max, range_, filename, zoom_methods):
    fig, ax = plt.subplots(1,1)

    fig.set_size_inches(5, 5)

    all_methods = METHODS.values()

    l = len(all_methods)
    ls_it = ls_iter(l)

    for method in all_methods:
        if method not in zoom_methods:
            next(ls_it)
            continue

        ser = values.loc[method].loc[range_[0]:range_[1]]
        err_min_ = err_min.loc[method].loc[range_[0]:range_[1]]
        err_max_ = err_max.loc[method].loc[range_[0]:range_[1]]

        color = cm(float(all_methods.index(method)) / l)

        y = ser.values
        x = ser.index.values
        plt.plot(x, y, label=method, ls=next(ls_it), color=color)
        plt.fill_between(x, err_min_, err_max_, alpha=0.1, color=color)


    # CUDA value
    plt.plot(24, 0.08738 / 2**24, 'o', label='SemiRecursive (CUDA)')

    plt.legend(ncol=2)

    ax.set_xlabel('$b=\log(n)$')
    ax.set_ylabel('$\\frac{t}{n}$', size=16, rotation=0)
    ax.yaxis.set_label_coords(-0.08,0.5)

    plt.tight_layout()

    plt.savefig(filename, dpi=300, bbox_inches='tight')

    plt.close(fig)


def plot_run_times(values, err_min, err_max, range_, zoom_methods, zoom_range, filename):
    fig, (ax1, ax2) = plt.subplots(1,2)

    fig.set_size_inches(10, 5)

    all_methods = METHODS.values()
    del all_methods[all_methods.index('SemiRecursive (OpenMP)')]

    l = len(all_methods)
    ls_it = ls_iter(l)

    plt.sca(ax1)

    for method in all_methods:

        ser = values.loc[method].loc[range_[0]:range_[1]]
        err_min_ = err_min.loc[method].loc[range_[0]:range_[1]]
        err_max_ = err_max.loc[method].loc[range_[0]:range_[1]]

        color = cm(float(all_methods.index(method)) / l)

        y = ser.values
        x = ser.index.values
        plt.plot(x, y, label=method, ls=next(ls_it), color=color)
        plt.fill_between(x, err_min_, err_max_, alpha=0.1, color=color)

    plt.legend(ncol=2)

    ax1.set_xlabel('$b=\log(n)$')
    ax1.set_ylabel('$\\frac{t}{n}$', size=16, rotation=0)
    ax1.yaxis.set_label_coords(-0.08,0.5)

    # zoom in to best scoring methods on higher input sizes

    plt.sca(ax2)
    ls_it = ls_iter(l)

    for method in all_methods:
        if method not in zoom_methods:
            next(ls_it)
            continue

        ser = values.loc[method].loc[zoom_range[0]:zoom_range[1]]
        err_min_ = err_min.loc[method].loc[zoom_range[0]:zoom_range[1]]
        err_max_ = err_max.loc[method].loc[zoom_range[0]:zoom_range[1]]

        color = cm(float(all_methods.index(method)) / l)

        y = ser.values
        x = ser.index.values
        plt.plot(x, y, label=method, ls=next(ls_it), color=color)
        plt.fill_between(x, err_min_, err_max_, alpha=0.1, color=color)


    plt.legend(ncol=2)

    ax = plt.gca()

    ax.set_xlabel('$b=\log(n)$')
    plt.tight_layout()

    plt.savefig(filename, dpi=300, bbox_inches='tight')

    plt.close(fig)


def get_asymmetric_error_array(err_min, err_max):
    err = []

    for key in err_min.index:
        err.append([err_min.loc[key], err_max.loc[key]])

    return np.array(err)


def adjust(x):
    return x / (2 ** x.name)


if __name__ == '__main__':
    args = parser.parse_args()

    done = False

    if args.tex_tab == 'ctimes':
        c_times = get_compile_times(args.compiler[0])
        write_output(
            c_times,
            '{}_compile_times.tex'.format(args.compiler[0]),
            13, True
            )
        done = True

    elif args.tex_tab == 'rtimes':
        values, min_err, max_err = get_run_times(args.compiler[0])
        write_output(
            values,
            '{}_run_times.tex'.format(args.compiler[0]),
            10, True
            )
        done = True

    elif args.tex_tab == 'cpu':
        write_output(
            get_cpu_spec(),
            'cpu_spec.tex',
            6, index=False
            )
        done = True

    if args.plot == 'ctimes':
        c_times = get_compile_times(args.compiler[0])
        plot_compile_times(c_times, '{}_compile_times.pdf'.format(args.compiler[0]))
        done = True

    elif args.plot == 'rtimes':
        values, min_err, max_err = get_run_times(args.compiler[0])
        plot_run_times(values,
                       min_err,
                       max_err,
                       range_=(8,30),
                       zoom_range=(20, 30),
                       zoom_methods=[
                           'Pair bitwise','COBRA (in-place)',
                           'COBRA (out-of-place)',
                           'Recursive',
                           'SemiRecursive'
                       ],
                       filename='{}_run_times.pdf'.format(args.compiler[0])
                       )

        if args.compiler[0] == 'g++':
            plot_run_times2(values,
                            min_err,
                            max_err,
                            range_=(20,30),
                            filename='open_mp_performance.pdf',
                            zoom_methods = [
                                'COBRA (in-place)',
                                'COBRA (out-of-place)',
                                'SemiRecursive',
                                'SemiRecursive (OpenMP)'
                            ]
                            )

        done = True

    if not done:
        parser.print_help()
