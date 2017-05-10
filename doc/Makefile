.PHONY: all transcribe

all: ./paper/results/g++_compile_times.tsv\
   	./paper/results/g++_run_times.tsv\
	./paper/results/clang++_compile_times.tsv\
	./paper/results/clang++_run_times.tsv\
	transcribe 
	cd paper && $(MAKE)

transcribe:
	cd paper/results && $(MAKE)

./paper/results/%_compile_times.tsv:
	cd ../src_and_bin/src && $(MAKE) CC=$*

./paper/results/%_run_times.tsv:
	../src_and_bin/src/utilities/run_tests.py All --compiler $* ./paper/results/$*_run_times.tsv
