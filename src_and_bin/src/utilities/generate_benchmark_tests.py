#!/usr/bin/env python2.7
"""
Use this script to automatically generate source code of test files,
and compile them with time measurement.
"""
import warnings
import argparse
import os
import sys
import subprocess
import re
import math
from collections import defaultdict

import test_templates

# make sure working directory is correct
DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(DIR)

METHODS = ['NaiveShuffle', 'StockhamShuffle',
           'COBRAShuffle', 'LocalPairwiseShuffle',
           'RecursiveShuffle', 'SemiRecursiveShuffle',
           'TableShuffle', 'UnrolledShuffle', 'OutOfPlaceCOBRAShuffle', 'XORShuffle', 'OpenMPSemiRecursiveShuffle']

# set this prefix in front of output directories,
# to seperate optimization files from others
OPTIMIZATION_PREFIX = "opt"

TEST_CLASSES = dict(zip(METHODS, ['TestClass'] * len(METHODS)))
TEST_CLASSES['SemiRecursiveShuffle'] = 'TestClassSemiRec'
TEST_CLASSES['COBRAShuffle'] = 'TestClassCOBRA'
TEST_CLASSES['OutOfPlaceCOBRAShuffle'] = 'TestClassCOBRA'


def get_dir(args):
    if args.optimize is not None:
        dir_ = '_'.join((OPTIMIZATION_PREFIX, 'benchmarks'))
    else:
        dir_ = 'benchmarks'

    return os.path.join(dir_, args.METHOD)


def parse_output(stdout, stderr):
    print(stderr)
    print(stdout)

    sec = float('nan')
    minutes = float('nan')

    for line in stderr.split('\n'):
        try:
            minutes, sec = re.match('.*([0-9]{1,2}):([0-9]{2}\.[0-9]{2}).*',
                                    line.strip().split(' ')[2]).groups((1, 2))
        except:
            continue
    sec = float(sec)
    minutes = float(minutes)

    return sec + (minutes * 60.0)


def write_file(args, n_bits, directory, template, parameter=None, **kwargs):
    """
    :param args: arguments object as parsed from command line
    :param n_bits: number of bits
    :param directory: output directory for generated source code
    :param template: template use with given arguments
    :param parameter: if this is given the generated filename will have its value appended with an underscore
    :param kwargs: additional parameters needed for generating the test program source code
    :return:
    """
    if parameter is not None:
        file_name = args.METHOD + "_" + str(n_bits) + "_" + str(parameter) + ".cpp"
    else:
        file_name = args.METHOD + "_" + str(n_bits) + ".cpp"

    with open(os.path.join(directory, file_name), "w") as test_file:
        test_file.write(template.format(
            input_method=args.METHOD,
            input_datatype=args.type,
            input_logn=n_bits,
            n_runs=args.N_RUNS,
            quant=args.error_quantile,
            **kwargs
        ))


def write_and_compile_bit_range(args, directory, times, template, parameter, omit=None, **kwargs):
    """
    write and compile test code for a given range of bits.
    :param args: arguments as parsed from command line
    :param directory: output directory
    :param times: list, holding compilation times
    :param template: template string to specify for arguments
    :param parameter: parameter value
    :param str omit: omit expression
    :param kwargs: additional parameters to fill into template
    :return:
    """

    times_ = [None] * (args.MAX_BITS - args.MIN_BITS + 1)

    try:
        lbw = kwargs['log_block_width']
    except:
        lbw = None

    for i, n_bits in enumerate(xrange(args.MIN_BITS, args.MAX_BITS + 1)):
        if omit is not None:
            if eval(str(n_bits) + omit):
                continue

        if args.METHOD in ['COBRAShuffle', 'OutOfPlaceCOBRAShuffle']:
            if lbw > 0.5*n_bits:
                kwargs['log_block_width'] = int(math.floor(0.5*n_bits))
            else:
                kwargs['log_block_width'] = lbw

        write_file(args, n_bits, directory, template, parameter=parameter, **kwargs)

        cmd = ['make', 'benchmark', 'ALG={}'.format(args.METHOD),
               'N={}'.format(n_bits), 'TEST_CLASS={}'.format(TEST_CLASSES[args.METHOD]),
               'CC={}'.format(args.compiler)]

        if args.METHOD == 'OpenMPSemiRecursiveShuffle':
            cmd.append('FLAGS= -std=c++11 -Ofast -march=native -mtune=native -fopenmp')

        if args.optimize is not None:
            cmd.append('DIRNAME={}'.format(os.path.dirname(directory)))
            cmd.append('SUFFIX=_{}'.format(parameter))

        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            sec = parse_output(*proc.communicate())
            times_[i] = sec
        except AttributeError:
            warnings.warn('Compilation timing failed for method {}, with {} bits'.format(args.METHOD, n_bits))

    times.append(times_)


def main(args, times, omit=None):
    """
    :param args: arguments as parsed from command line
    :param times: list of timing results obtained so far
    :param omit: omit expression
    :return:
    """
    directory = get_dir(args)

    kwargs = {}

    if not os.path.exists(directory):
        os.makedirs(directory)

    if args.optimize is not None:
        assert args.METHOD in ['SemiRecursiveShuffle', 'COBRAShuffle',
                               'OutOfPlaceCOBRAShuffle'], 'No free parameters for this Algorithm'
        template = getattr(test_templates, args.METHOD)

        p_start, p_stop = args.optimize

        if args.METHOD == 'SemiRecursiveShuffle':
            p_name = 'recursions_remaining'
        else:
            p_name = 'log_block_width'

        for p in xrange(p_start, p_stop + 1):
            kwargs[p_name] = p
            write_and_compile_bit_range(args, directory, times, template, p, **kwargs)

    else:
        if args.METHOD == 'SemiRecursiveShuffle':
            assert args.recursions_remaining is not None, 'for this method you must specify the parameter recursions_remaining'
            kwargs['recursions_remaining'] = args.recursions_remaining
            template = getattr(test_templates, args.METHOD)
        elif args.METHOD == 'COBRAShuffle':
            assert args.log_block_width is not None, 'for this method you must specify the parameter log_block_width'
            kwargs['log_block_width'] = args.log_block_width
            template = getattr(test_templates, args.METHOD)
        elif args.METHOD == 'OutOfPlaceCOBRAShuffle':
            assert args.log_block_width is not None, 'for this method you must specify the parameter log_block_width'
            kwargs['log_block_width'] = args.log_block_width
            template = getattr(test_templates, args.METHOD)
        else:
            template = test_templates.default

        write_and_compile_bit_range(args, directory, times, template, None, omit, **kwargs)

    return times


def t_omit(str_):
    try:
        method, expr = str_.split(':')
        assert method in METHODS
        return method, expr
    except:
        raise argparse.ArgumentError('invalid argument {}'.format(str_))


def write_output(data, index, min_bits, max_bits, outfile):
    if outfile is None:
        f = sys.stdout
    else:
        f = open(outfile, 'w')

    f.write('\t'.join(['Method'] + [str(x) for x in xrange(min_bits, max_bits + 1)]) + '\n')

    for i, line in enumerate(data):
        f.write('\t'.join([index[i]] + [str(x) for x in line]))
        f.write('\n')

    f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('METHOD', choices=METHODS + ['All'],
                        help='name of the method, or All for building every method available.')
    parser.add_argument('MIN_BITS', type=int, help='minimum number of bits')
    parser.add_argument('MAX_BITS', type=int, help='maximum number of bits')
    parser.add_argument('N_RUNS', type=int, help='number of runs')
    parser.add_argument('--type', nargs="?", default="double", help='data type as argument for std::complex<TYPE>')
    parser.add_argument('--recursions-remaining', required=False, type=int, help='parameter for SemiRecursiveShuffle')
    parser.add_argument('--log-block-width', required=False, type=int, help='parameter for COBRAShuffle')
    parser.add_argument('-o', '--optimize', metavar=('PMin', 'PMax'), required=False, type=int, nargs=2, default=None,
                        help='Compile executables for performance comparison of parameters recursions_remaining/log_block_width')
    parser.add_argument('-f', '--file', required=False, default=None, type=str, help='output file, default is stdout')
    parser.add_argument('--omit', required=False, nargs='+', type=t_omit,
                        metavar='METHOD:>N',
                        help='add methods and threshold values here, to exclude them from compilation')
    parser.add_argument('--compiler', choices=['g++', 'clang++'], required=False, default='g++')

    parser.add_argument('-eq', '--error-quantile', required=False, default=0.025, help='Error quantile')

    args = parser.parse_args()

    times = []
    index = []

    omit_ = defaultdict(lambda: None)

    if args.omit is not None:
        for k, v in args.omit:
            omit_[k] = v

    if args.METHOD == 'All':
        assert args.recursions_remaining != None, 'Please specify --recursions-remaining in order to make SemiRecursiveShuffle work'
        assert args.log_block_width != None, 'Please specify --log_block_width in order to make COBRAShuffle work'

        for method in METHODS:
            if method == 'OpenMPSemiRecursiveShuffle' and args.compiler == 'clang++':
                continue
            args.METHOD = method
            times = main(args, times, omit_[method])
            index.append(method)

    else:
        times = main(args, times, omit_[args.METHOD])
        index.append(args.METHOD)

    if not args.optimize:
        write_output(times, index, args.MIN_BITS, args.MAX_BITS, args.file)

