r"""Custom script to run pep8 on the codebase.

This runs pep8 as a script via subprocess but only runs it on the
.py files that are checked in to the repository.

**Note**: This file `borrowed`_ from a project that one of the maintainers
          works on. Hence that license also applies here.

.. borrowed: https://github.com/GoogleCloudPlatform/gcloud-python/\
             blob/master/scripts/pep8_on_repo.py
"""


import os
import subprocess
import sys


def main():
    """Run pep8 on all Python files in the repository."""
    git_root = subprocess.check_output(
        ['git', 'rev-parse', '--show-toplevel']).strip()
    os.chdir(git_root)
    python_files = subprocess.check_output(['git', 'ls-files', '*py'])
    python_files = python_files.strip().split()

    pep8_command = ['pep8'] + python_files
    status_code = subprocess.call(pep8_command)
    sys.exit(status_code)


if __name__ == '__main__':
    main()
