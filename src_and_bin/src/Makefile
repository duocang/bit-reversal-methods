FLAGS= -std=c++11 -Ofast -march=native -mtune=native
CC=g++
SRC_DIR= .
BIN_DIR= ../bin
DIRNAME=benchmarks
OUT_DIR= $(BIN_DIR)/$(DIRNAME)/$(ALG)

# specify time tool to make sure that this and not bashs builtin is used
TIME=/usr/bin/time

COMPILE_TIME_FILE=../../doc/paper/results/$(CC)_compile_times.tsv

TEST_CLASS=TestClass

# generates and compiles all test files, prints table of compile times in tsv format to stdout
all:
	$(SRC_DIR)/utilities/generate_benchmark_tests.py All 4 30 100 \
		--log-block-width 6 \
		--recursions-remaining 1 \
		--omit "UnrolledShuffle:>16" "NaiveShuffle:>26" "StockhamShuffle:>26" \
		"XORShuffle:>26" "TableShuffle:>26" \
		--file $(COMPILE_TIME_FILE) \
		--error-quantile 0 \
		--compiler $(CC)

test:
	$(SRC_DIR)/utilities/generate_benchmark_tests.py OutOfPlaceCOBRAShuffle 4 20 10 -o 2 8 --compiler clang++

opt_COBRA:
	$(SRC_DIR)/utilities/generate_benchmark_tests.py OutOfPlaceCOBRAShuffle 16 26 10 -o 5 8

opt_COBRAInPlace:
	$(SRC_DIR)/utilities/generate_benchmark_tests.py COBRAShuffle 16 26 10 -o 3 5

$(OUT_DIR):
	mkdir -p $(OUT_DIR)

clean:
	rm -rf $(OUT_DIR)*

# target for compiling single benchmark test
benchmark: $(SRC_DIR)/$(DIRNAME)/$(ALG)/$(ALG)_$(N)$(SUFFIX).cpp $(OUT_DIR)
	time $(CC) $(SRC_DIR)/$(DIRNAME)/$(ALG)/$(ALG)_$(N)$(SUFFIX).cpp $(FLAGS)\
	   	-o $(OUT_DIR)/$(CC)_$(ALG)_$(N)$(SUFFIX) \
		-include algorithms/$(ALG).hpp \
		-include utilities/$(TEST_CLASS).hpp
