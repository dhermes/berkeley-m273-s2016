import os

from setuptools import setup
from setuptools import find_packages


REQUIREMENTS = (
    'numpy',
    'six >= 1.6.1',
    'sympy',
)

setup(
    name='berkeley-m273-s2016',
    version='0.0.1',
    description='Berkeley Math 273 - Spring 2016',
    author='Daniel J. Hermes',
    author_email='daniel.j.hermes@gmail.com',
    scripts=[],
    url='https://github.com/dhermes/berkeley-m273-s2016',
    packages=find_packages(),
    license='Apache 2.0',
    platforms='Posix; MacOS X; Windows',
    include_package_data=True,
    zip_safe=False,
    install_requires=REQUIREMENTS,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
    ]
)
