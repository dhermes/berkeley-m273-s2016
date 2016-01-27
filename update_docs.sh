#!/bin/bash

set -ev

#########################################
# Only update docs if we are on Travis. #
#########################################
if [[ "${TRAVIS}" == "true" ]] && \
       [[ "${TRAVIS_PULL_REQUEST}" == "false" ]]; then
  echo "Building new docs on a merged commit."
else
  echo "Not on Travis, doing nothing."
  exit
fi

# Adding GitHub pages branch. `git submodule add` checks it
# out at HEAD.
GH_PAGES_DIR="ghpages"
git submodule add -b gh-pages \
    "https://${GH_OAUTH_TOKEN}@github.com/dhermes/berkeley-m273-s2016" \
    ${GH_PAGES_DIR}

# Sphinx will use the package version by default.
sphinx-build -W -b html -d docs/_build/doctrees docs docs/_build/html

# Update gh-pages with the created docs.
cd ${GH_PAGES_DIR}
git rm -fr .
# Add the new content.
cp -R ../docs/_build/html/* .

# Update the files push to gh-pages.
git add .
git status

if [[ -z "$(git status --porcelain)" ]]; then
    echo "Nothing to commit. Exiting without pushing changes."
    exit
fi

# Commit to gh-pages branch to apply changes.
git config --global user.email "travis@travis-ci.org"
git config --global user.name "travis-ci"
git commit -m "Update docs after merge to master."
# NOTE: This may fail if two docs updates (on merges to master)
#       happen in close proximity.
git push \
    "https://${GH_OAUTH_TOKEN}@github.com/dhermes/berkeley-m273-s2016" \
    HEAD:gh-pages
