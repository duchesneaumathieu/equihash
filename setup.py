from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize("equihash/search/_inverted_index.pyx", language_level = "3")
)
