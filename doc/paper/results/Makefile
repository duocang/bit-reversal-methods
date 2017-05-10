
all: g++_compile_times.pdf g++_run_times.pdf \
	clang++_compile_times.pdf clang++_run_times.pdf cpu_spec.tex

%_run_times.pdf:
	./get_tables_and_plots.py --plot rtimes --compiler $*

%_compile_times.pdf:
	./get_tables_and_plots.py --plot ctimes --compiler $*

%_compile_times.tex:
	./get_tables_and_plots.py --tex_tab ctimes --compiler $*

%_run_times.tex:
	./get_tables_and_plots.py --tex_tab rtimes --compiler $*

cpu_spec.tex:
	./get_tables_and_plots.py --tex_tab cpu
