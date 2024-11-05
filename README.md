# Eigenvector Continuation for two-body scattering

A fast emulator for nuclear scattering and transfer reactions using Eigenvector Continuation, which was developed in Drischler, Quinonez, Giuliani, Lovell, and Nunes, [Phys. Lett. B **823**, 136777][Dris21].

Contact: Christian Drischler (<drischler@frib.msu.edu>)


## Python 3 code

The original Python code (2021) features:
* general (complex) Kohn variational principle (e.g., for K-, S-, K-inverse-, and T-matrix)
* Kohn anomaly detection and removal
* modular object-oriented approach
* Gauss-Legendre quadrature
* parallel computing
* adaptive ODE solver for radial Schroedinger equation

Recently added features (2024) include:
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


## C++ code

The C++ code is a simple implementation of the K-matrix Kohn variational principle.
It does not provide all features of the Python 3 companion code. The libraries `boost` and `armadillo`, which can be installed via `homebrew` on MacOS, are required.

To build the test app `evc` and the library for external use follow these steps:
```shell
mkdir build && cd $_
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX:PATH=. ..
make -j 4 all install
./evc  # run build with default values [optional arguments: energy in MeV (first), angular momentum l (second)]
```
The include files are installed in `build/include` and the library in `build/lib`.


### Demystifying the library

```c++
// start the session, i.e., train the emulator
// the training and testing (chiral) Hamiltonians are described by the following struct, which is defined in "localGT+.h"
struct Lecs{double CS, CT, C1, C2, C3, C4, C5, C6, C7, CNN, CPP;};

int emulator_startSession (int numLecSets, Lecs *lecSets, double energy, int l);
// "lecSets" is a pointer to an array of type "Lecs" and length "numLecSets";
// the array contains the training Hamiltonians

// emulate and compute the phase shift using the KVP
int emulator_emulate(Lecs *lecSets, double *phaseShift);
// "lecSets" is a pointer to a single testing Hamiltonian (note the difference to training session);
// the function is thread-safe, so different testing Hamiltonians can be emulated in parallel;
// the KVP estimate of the phase shift is stored in "phaseShift"

// close the session, i.e., free memory
int emulator_closeSession();
```

## Backup to FRIB drive

To straightforwardly back up the current version of the repository follow these steps.
Add the FRIB server to your ssh configuration in `~/.ssh/config`:
```
Host frib
    HostName nsclgw1.nscl.msu.edu
    User <your netid>
```
Set the environment variable `FRIB_BACKUP_FOLDER` as follows:
```shell
export FRIB_BACKUP_FOLDER=frib:/projects/fewbody/<your last name>
```

Every time you want to make a backup simply run (in the project directory):
```shell
make backup
```

## License and acknowledgment

Private. Please contact Christian Drischler (<drischler@frib.msu.edu>) for license information.

Part of this work was published in
Drischler, Quinonez, Giuliani, Lovell, and Nunes, [Phys. Lett. B **823**, 136777][Dris21].

[Dris21]:https://www.sciencedirect.com/science/article/pii/S0370269321007176?via%3Dihub
