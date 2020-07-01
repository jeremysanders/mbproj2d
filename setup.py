#!/usr/bin/env python3

# Copyright (C) 2020 Jeremy Sanders <jeremy@jeremysanders.net>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy as np

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
        include_dirs=[np.get_include()]
    ),
]

install_requires = [
    'numpy',
    'scipy',
    'cython',
    'pyfftw',
    'h5py',
    'emcee',
]

setup(
    name = 'MBProj2D',
    version = '0.1',
    description = 'Forward fitting cluster modelling',
    author = 'Jeremy Sanders',
    author_email = 'jeremy@jeremysanders.net',
    url = 'https://github.com/jeremysanders/mbproj2d',
    install_requires = install_requires,
    packages = ['mbproj2d'],
    ext_package = 'mbproj2d',
    ext_modules = cythonize(extensions),
)
