from setuptools import Extension, setup
from Cython.Build import cythonize


sourcefiles = ['mbproj2d/helpers/project.pyx', 'mbproj2d/helpers/project_cc.cc']

extensions = [Extension(
    "mbproj2d.helpers",
    sourcefiles,
    extra_compile_args=['-fno-math-errno', '-mavx2'],
)]

setup(
    ext_modules = cythonize(extensions),
)
