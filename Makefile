help:
	@echo 'Makefile for Berkeley M273 - Spring 2016'
	@echo '                                        '
	@echo 'Usage:                                  '
	@echo '   make docs   Render Sphinx docs       '

docs:
	sphinx-apidoc --separate --force -o docs/ assignment1/ assignment1/test_dg1.py
	rm -f docs/modules.rst
	rm -fr docs/_build
	sphinx-build -W -b html -d docs/_build/doctrees docs docs/_build/html

.PHONY: help docs
