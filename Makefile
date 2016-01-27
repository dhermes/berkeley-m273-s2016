help:
	@echo 'Makefile for Berkeley M273 - Spring 2016'
	@echo '                                        '
	@echo 'Usage:                                  '
	@echo '   make docs   Render Sphinx docs       '

docs:
	echo "PWD $(shell pwd)"  # Temporary hack to diagnose Travis failure
	sphinx-apidoc --separate --force -o docs/ .
	rm docs/modules.rst
	rm -fr docs/_build
	sphinx-build -W -b html -d docs/_build/doctrees docs docs/_build/html

.PHONY: help docs
