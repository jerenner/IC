from setuptools import setup, find_packages
from Cython.Build import cythonize
from glob import glob
from os import path
from subprocess import check_output
import numpy
numpy_include = path.join(path.dirname(numpy.__file__), 'core/include')


setup(name         = 'invisible_cities',
      version      = check_output('git describe --tags --always'.split()).decode(),
      description  = 'Data processing and reconstruction framework for the NEXT experiment',
      url          = 'https://github.com/nextic/IC',
      author       = 'NEXT collaboration',
      author_email = 'nextic@cern.ch',
      license      = 'MIT',
      packages     = find_packages(),
      ext_modules  = cythonize(glob('invisible_cities/**/*.pyx', recursive=True)),
      include_dirs = [numpy_include],
      zip_safe     = False,
)
