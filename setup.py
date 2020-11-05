#!/usr/bin/env python3

# Copyright (C) 2020 Jeremy Sanders <jeremy@jeremysanders.net>
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 3 of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program; if not, write to the Free Software Foundation,
# Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy as np

urls = {
    'Source': 'https://github.com/jeremysanders/mbproj2d',
    'Tracker': 'https://github.com/jeremysanders/mbproj2d/issues',
}

classifiers=[
    # How mature is this project? Common values are
    #   3 - Alpha
    #   4 - Beta
    #   5 - Production/Stable
    'Development Status :: 3 - Alpha',

    # Indicate who your project is intended for
    'Intended Audience :: Developers',

    # Pick your license as you wish (should match "license" above)
    'License :: OSI Approved :: GNU Lesser General Public License v3 or later (LGPLv3+)',

    # Specify the Python versions you support here. In particular, ensure
    # that you indicate whether you support Python 2, Python 3 or both.
    'Programming Language :: Python :: 3.3',
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
]

fastsourcefiles = [
    'mbproj2d/fast/fast.pyx',
    'mbproj2d/fast/project_cc.cc',
    'mbproj2d/fast/binimg_cc.cc',
]
extensions = [
    Extension(
        "fast",
        fastsourcefiles,
        extra_compile_args = [
            '-fno-math-errno',
            '-mavx2',
            '-std=c++11',
        ],
        include_dirs=[np.get_include()],
        # without this the __builtin_cpu_supports does not work on
        # some gcc versions
        libraries=['gcc'],
    ),
]

install_requires = [
    'numpy',
    'scipy',
    'cython',
    'pyfftw',
    'h5py',
    'emcee',
    'astropy',
]

setup(
    name = 'MBProj2D',
    version = '0.2',
    description = 'Forward-fitting cluster modelling',
    author = 'Jeremy Sanders',
    author_email = 'jeremy@jeremysanders.net',
    install_requires = install_requires,
    packages = ['mbproj2d'],
    ext_package = 'mbproj2d',
    ext_modules = cythonize(extensions),
    classifiers=classifiers,
    project_urls=urls,
)
