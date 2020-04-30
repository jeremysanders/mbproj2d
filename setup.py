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
