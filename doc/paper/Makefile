all: paper-english.pdf #paper-german.pdf

clean:
	rm -f *.aux *.bbl *.blg *.log *~ *.out results/*.bak

%.pdf: %.tex
	pdflatex $*.tex
	bibtex $*
	pdflatex $*.tex
	pdflatex $*.tex
