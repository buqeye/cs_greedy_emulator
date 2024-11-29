from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import subprocess

try:
    result = subprocess.run(["gsl-config", "--prefix"], capture_output=True, text=True)
    print(f"GSL found in '{result.stdout}'")
except FileNotFoundError:
    print(f"Warning: GSL is not installed. Build will not work.")

ext_modules = [Extension('chiralPot', ['cpot.pyx'],
                         libraries=['localGt+', 'gsl', 'blas'],
                         library_dirs=['.', '/usr/local/lib', '/opt/homebrew/lib'], language='c++')]

setup(name = 'chiral potential extension module',
      cmdclass = {'build_ext': build_ext},
      ext_modules = ext_modules)
