help:
	@echo 'Makefile for a dg1 docs          '
	@echo '                                 '
	@echo 'Usage:                           '
	@echo '   make docs   Render Sphinx docs'

docs:
	sphinx-apidoc --separate --force -o docs/ .
	rm docs/modules.rst
	rm -fr docs/_build
	sphinx-build -W -b html -d docs/_build/doctrees docs docs/_build/html

.PHONY: help docs
