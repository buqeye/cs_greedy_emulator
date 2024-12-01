from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

try:
      import subprocess
      result = subprocess.run(["gsl-config", "--prefix"], capture_output=True, text=True)
      print(f"GSL found in '{result.stdout.strip()}'")
except FileNotFoundError:
      print(f"Warning: GSL not found. Build may not work.")

import os
if 'MYLOCAL' not in os.environ:
     print("Warning: env variable 'MYLOCAL' not set. Assuming standard location.")
     path_mylocal_lib = os.environ.get('HOME') + "/src/lib"
else:
     path_mylocal_lib = os.environ.get('MYLOCAL') + "/lib"

ext_modules = [Extension('chiralPot', ['cpot.pyx'],
                         libraries=['localGt+', 'gsl', 'blas', 'cubature'],
                         library_dirs=['.', '/usr/local/lib', '/opt/homebrew/lib', path_mylocal_lib], 
                         language='c++')]

for e in ext_modules:
    e.cython_directives = {'language_level': "3"} 

setup(name = 'chiral potential extension module',
      cmdclass = {'build_ext': build_ext},
      ext_modules = ext_modules)
