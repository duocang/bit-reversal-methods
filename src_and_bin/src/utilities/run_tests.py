#!/usr/bin/env python2.7
import os
import argparse
import StringIO
import subprocess

import math

# this points to src_and_bin now, independent of the current working directory
SRC_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BIN_DIR = os.path.join(SRC_PATH, 'bin/benchmarks/{}')
O_DIR = os.path.join(SRC_PATH, 'bin/opt_benchmarks/{}')

METHODS = ['NaiveShuffle', 'StockhamShuffle',
           'COBRAShuffle', 'LocalPairwiseShuffle',
           'RecursiveShuffle', 'SemiRecursiveShuffle',
           'TableShuffle', 'UnrolledShuffle',
           'OutOfPlaceCOBRAShuffle', 'XORShuffle', 'OpenMPSemiRecursiveShuffle']


parser = argparse.ArgumentParser(description='run all compiled benchmark \
                                              tests for a given Method and write runtimes to a tsv-table')

parser.add_argument('METHOD', type=str,choices=METHODS + ['All'],
                    help='name of the method to run. Make sure that the benchmark' \
                          'tests have been compiled in advance. Use make all to do this,' \
                          'or have a look at generate_benchmark_tests.py.')
parser.add_argument('-o', '--optimization', action='store_true', default=False,
                    help='run executables in obenchmarks, \
                          to test performance depending on parameter')
parser.add_argument('FILE',
                    help='output file')

parser.add_argument('--compiler', choices=['g++', 'clang++'], default='g++')


def main(args, dir_, values, errors):

    for executable in os.listdir(dir_):
        if not executable.startswith(args.compiler):
            continue

        print(executable)

        cmd = [os.path.join(dir_, executable)]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = proc.communicate()

        method, n_runs, n_bits, mean_runtime, err_min, err_max = stdout.strip().split('\t')

        n_bits = int(n_bits)

        if args.optimization:
            parameter_value = executable.split('_')[3]
            try:
                errors[n_bits][parameter_value] = (err_min, err_max)
                values[n_bits][parameter_value] = mean_runtime
            except KeyError:
                errors[n_bits] = {parameter_value : (err_min, err_max)}
                values[n_bits] = {parameter_value: mean_runtime}
        else:
            try:
                errors[n_bits][method] = (err_min, err_max)
                values[n_bits][method] = mean_runtime
            except KeyError:
                errors[n_bits] = {method: (err_min, err_max)}
                values[n_bits] = {method: mean_runtime}

    return values, errors


def write_err_tables(err_values, outfile):
    """
    write error values to distinct tsv tables
    :param err_values: error values dict
    :param outfile: args.file -> gets modified with suffices for min and max error
    :return:
    """
    base_name, ext = os.path.splitext(outfile)
    with open(''.join((base_name + '_min_err', ext)), 'w') as min_err_f:
        with open(''.join((base_name + '_max_err', ext)), 'w') as max_err_f:
            header = ['N_BITS'] + METHODS
            min_err_f.write('\t'.join(header) + '\n')
            max_err_f.write('\t'.join(header) + '\n')

            for n_bits, vals in err_values.iteritems():
                min_err_line = [str(n_bits)] + ['nan']*(len(header)-1)
                max_err_line = [str(n_bits)] + ['nan']*(len(header)-1)
                for method, errs in vals.iteritems():
                    min_err, max_err = errs
                    min_err_line[header.index(method)] = str(min_err)
                    max_err_line[header.index(method)] = str(max_err)

                min_err_f.write('\t'.join(min_err_line) + '\n')
                max_err_f.write('\t'.join(max_err_line) + '\n')


def write_value_table(values, outfile, opt=False):
    if opt:
        with open(outfile, 'w') as f:
            header = ['N_BITS'] + [str(x) for x in values.values()[0].keys()]
            print(values)
            f.write('\t'.join(header) + '\n')

            for n_bits, vals in values.iteritems():
                line = [str(n_bits)] + ['nan']*(len(header)-1)
                for p_val, time in vals.iteritems():
                    line[header.index(str(p_val))] = str(time)

                f.write('\t'.join(line) + '\n')
    else:
        with open(outfile, 'w') as f:
            header = ['N_BITS'] + METHODS
            f.write('\t'.join(header) + '\n')

            for n_bits, vals in values.iteritems():
                line = [str(n_bits)] + ['nan']*(len(header)-1)
                for method, time in vals.iteritems():
                    line[header.index(method)] = str(time)
                f.write('\t'.join(line) + '\n')

if __name__ == '__main__':
    args = parser.parse_args()

    dir_ = BIN_DIR.format(args.METHOD)

    values = {}
    errors = {}

    if args.optimization:
        assert args.METHOD in ['COBRAShuffle', 'SemiRecursiveShuffle', 'OutOfPlaceCOBRAShuffle'], 'nothing to optimize'
        dir_ = O_DIR.format(args.METHOD)
        values, errors = main(args, dir_, values, errors)
    else:

        if args.METHOD == 'All':
            for method in METHODS:
                if args.compiler == 'clang++' and method == 'OpenMPSemiRecursiveShuffle':
                    continue
                args.METHOD = method
                dir_ = BIN_DIR.format(method)
                try:
                    values, errors = main(args, dir_, values, errors)
                except OSError:
                    continue

        else:
            values, errors = main(args, dir_, values, errors)

    write_value_table(values, args.FILE, args.optimization)
    if not args.optimization:
        write_err_tables(errors, args.FILE)



