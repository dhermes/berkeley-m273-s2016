help:
	@echo 'Makefile for class presentation                                 '
	@echo '                                                                '
	@echo 'Usage:                                                          '
	@echo '   make slides               render slides from Jupyter Notebook'
	@echo '   make serve-slides         serve slides from Jupyter Notebook '
	@echo '   make serve-static-slides  serve slides from HTML             '

slides:
	jupyter nbconvert weno_computations.ipynb --to slides

serve-slides:
	jupyter nbconvert weno_computations.ipynb --to slides --post serve

serve-static-slides:
	cd slides-static/ && python -m SimpleHTTPServer

.PHONY: help slides serve-slides serve-static-slides
