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


sourcefiles = ['mbproj2d/fast/project.pyx', 'mbproj2d/fast/project_cc.cc']

extensions = [Extension(
    "mbproj2d.fast",
    sourcefiles,
    extra_compile_args=['-fno-math-errno', '-mavx2'],
)]

setup(
    ext_modules = cythonize(extensions),
)
