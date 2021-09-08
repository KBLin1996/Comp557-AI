# from setuptools import setup, Extension
# import numpy.distutils.misc_util

# define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

import numpy as np
from distutils.core import setup, Extension

setup(
    ext_modules=[Extension("_PyPacwar", ["PyPacwar.c", "PacWarGuts.c"])],
    include_dirs=[np.get_include()],
)
