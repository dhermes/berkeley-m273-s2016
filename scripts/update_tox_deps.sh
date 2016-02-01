#!/bin/bash

set -ev

if [ -d .tox/py27 ]; then
  .tox/py27/bin/pip install --upgrade --requirement=requirements.txt
fi

if [ -d .tox/py34 ]; then
  .tox/py34/bin/pip install --upgrade --requirement=requirements.txt
fi

if [ -d .tox/lint ]; then
  .tox/lint/bin/pip install --upgrade --requirement=requirements.txt
fi

if [ -d .tox/docs ]; then
  .tox/docs/bin/pip install --upgrade --requirement=requirements.txt
fi

if [ -d .tox/coveralls ]; then
  .tox/coveralls/bin/pip install --upgrade --requirement=requirements.txt
fi
