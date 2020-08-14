#! /usr/bin/env python
# Copyright 2014-2017 Ryan Loomis <rloomis@cfa.harvard.edu>, J. Huang, and I. Czekala.
# Licensed under the MIT License.

# I don't use the ez_setup module because it causes us to automatically build
# and install a new setuptools module, which I'm not interested in doing.

from setuptools import setup

setup (
    name = 'VISIBLE',
    version = '0.2.4',

    # This package actually *is* zip-safe, but I've run into issues with
    # installing it as a Zip: in particular, the install sometimes fails with
    # "bad local file header", and reloading a module after a reinstall in
    # IPython gives an ImportError with the same message. These are annoying
    # enough and I don't really care so we just install it as flat files.
    zip_safe = False,

    packages = [
        'VISIBLE',
    ],

    # We want to go easy on the requires; some modules are going to require
    # more stuff, but others don't need much of anything. But, it's pretty
    # much impossible to do science without Numpy.
    install_requires = [
        'numpy >= 1.13.1',
        'scipy >= 0.19.1',
        'astropy >= 2.0.0',
        'matplotlib >= 2.1.0',
        'vis_sample >= 0.3.3'
    ],

    author = 'Ryan Loomis',
    author_email = 'rloomis@cfa.harvard.edu',
    description = 'Matched filtering for interferometric data',
    license = 'MIT',
    keywords = 'astronomy science',
    url = 'https://github.com/AstroChem/VISIBLE',

    long_description = \
    '''The matched_filter() method in VISIBLE allows you to apply an approximated matched filter to interferometric spectral line data and extract a signal. 
    ''',

    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering :: Astronomy',
    ],
)
