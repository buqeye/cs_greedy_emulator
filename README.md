# Greedy Emulators for Nuclear Two-Body Scattering

## Python 3 code

* Numerov method in matrix form (FOM solver)
* Galerkin reduced order model (ROM) based on the Numerov method
* Proper Orthogonalization (POD)
* efficient offline-online decomposition
* error estimates and greedy algorithm
  

### Installing and testing the Python code

Install requirements by running:
```shell
python3 -m venv env
source env/bin/activate
python3 -m pip install -r requirements.txt
# deactivate ## when the job is done
```
Further, `Cython`, `gcc`, and `GSL` need to be installed for the chiral interactions.

Optional: set environment variable to plot the phase shift data obtained from the PWA '93, which are located in our Dropbox.
```shell
export NNPHASESHIFTS=<path to NN phase shifts>  # e.g., "~/Dropbox/uq-emulators/nn-online_phaseshifts"
```

Compile the local chiral interactions GT+ ([external C++ code](src/localGt+.cpp) provided by the developers):
```shell
make clean
make CXX=g++-14 # make sure to use the GNU c++ compiler, not clang
```

The chiral interactions can also be compiled manually. This is, however, not needed.
```shell
g++ -fPIC -O3 -shared -c src/localGt+.cpp -o liblocalGt+.so -I/usr/local/include -I./src/
python3 setup.py build_ext --inplace
```

The `data/` folder contains the values of the low-energy constants (LECs) of the GT+ family of local chiral potentials as `yaml` files.
These files can be generated via:
```shell
make lec_output
./lec_output
```
This will also run a unittest that checks whether the function returning the affine decomposition of the chiral interactions matches the output of the original function provided by the developers (i.e., not based on the affine decomposition).

Run a test calculation for the general KVP (can be skipped if only the new Galerkin emulator is of interest):

```shell
make test  # run predefined test calculation
python3 main.py -rm 25 -p chiral -er 0.01 40. 60 -lmax 4  # with some variables specified, for example
```

For more help run:
```shell
python3 main.py --help
```
Run the following pytest command to test important components of the code:

```python
python3 -m pytest tests.py
```

## License and acknowledgment

Private. Please contact Christian Drischler (<drischler@ohio.edu>) for license information.

[Dris21]:https://www.sciencedirect.com/science/article/pii/S0370269321007176?via%3Dihub
