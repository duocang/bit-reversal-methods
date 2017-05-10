"""
This file contains the templates to automatically instanciate the
different test classes according to a range of parameters provided via generate_benchmark_tests.py

To include a special test add a variable with the exact name here. If no special method is mentioned,
default will be used.

Includes are indicated in makefile
by -include path/to/dependency
"""
default = """
#include<complex>

int main() {{
    const unsigned char LOG_N = {input_logn};
    std::cout << "{input_method}" << "\t" << {n_runs} << "\t" << "{input_logn}" << "\t";
    TestRuntime < {input_method}, std::complex<{input_datatype}>, LOG_N>::run({n_runs}, {quant});
}}
"""
SemiRecursiveShuffle = """
#include<complex>

int main() {{
    const unsigned char RECURSIONS_REMAINING = {recursions_remaining};
    const unsigned char LOG_N = {input_logn};
    std::cout << "{input_method}" << "\t" << {n_runs} << "\t" << "{input_logn}" << "\t";
    TestRuntime < {input_method}, std::complex<{input_datatype}>, LOG_N, RECURSIONS_REMAINING>::run({n_runs}, {quant});
}}
"""
COBRAShuffle = """
#include<complex>

int main() {{
    const unsigned char LOG_BLOCK_WIDTH = {log_block_width};
    const unsigned char LOG_N = {input_logn};
    std::cout << "{input_method}" << "\t" << {n_runs} << "\t" << "{input_logn}" << "\t";
    TestRuntime < {input_method}, std::complex<{input_datatype}>, LOG_N, LOG_BLOCK_WIDTH>::run({n_runs}, {quant});
}}
"""
OutOfPlaceCOBRAShuffle = COBRAShuffle
