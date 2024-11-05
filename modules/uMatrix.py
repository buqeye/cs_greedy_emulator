import numpy as np
import re
from functools import lru_cache


class uMatrix:
    def __init__(self, lbl, uMat):
        self.lbl = lbl
        self.u = np.asarray(uMat)
        self.det = np.linalg.det(self.u)
        if np.isclose(np.abs(self.det), 0):
            raise ValueError(f"Asymptotic parametrization matrix for {self.lbl} is (near) singular.")
        self.uInv = np.linalg.inv(self.u)
        self.uInvT = np.linalg.inv(self.u).T
        self.uT = self.u.T

    def linFracTrafo(self, z):
        tmp = self.uT @ np.array([1, z])
        return tmp[1] / tmp[0]  # numerator / denominator


_uMatricesRaw = {"K": np.eye(2),
                 "S": np.array([[-1j, 1], [-1j, -1]]),
                 "T": np.array([[1, 0], [1j, 1]]),
                 "gK": lambda t: np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]]),
                 "gT": lambda t: np.array(
                       [[np.cos(t), np.sin(t)], [1j * np.cos(t) - np.sin(t), 1j * np.sin(t) + np.cos(t)]]),
                 "gS": lambda t: np.array(
                       [[-1j * np.cos(t) - np.sin(t), -1j * np.sin(t) + np.cos(t)],
                       [-1j * np.cos(t) + np.sin(t), -1j * np.sin(t) - np.cos(t)]]),
                 "hilbert": np.array([[1, 0.5], [0.5, 1./3.]]),
                 "hadamard": lambda t: np.array([[1, 1], [1, 1j*np.exp(t*1j)]]),
                 "hadamard*": lambda t: np.array([[1, np.exp(t*1j)], [np.exp(t*1j), 1j]]),  # modified Hadamard matrix
                 "hadamard**": lambda t: np.array([[1j, np.exp(t*1j)], [np.exp(t*1j), 1]])  # modified Hadamard matrix
                 }
# do not add here inverse matrix (will be discarded and added automatically);
# the T matrix parametrization does not need the factor "-np.pi" as
# suggested by Eq. (15) in Lucchese, Phys. Rev. A 40, 12;
# swap rows of 'u' to obtain parametrization for the inverse matrix,
# i.e., L vs Linv


@lru_cache(maxsize=64)
def getUmatrix(lbl, separator='-'):
    key, isInv, = lbl.replace('inv', ''), ('inv' in lbl)

    match = re.search(r"(.+)" + separator + r"([\d\.]+)", key)
    if match:
        key, angle = match.group(1), float(match.group(2))
    else:
        angle = None

    # print(key, angle)

    if key not in _uMatricesRaw.keys():
        raise ValueError(f"Asymptotic parametrization '{lbl}' is unknown.")
    else:
        umat = _uMatricesRaw[key]
        if callable(umat):
            if angle is None:
                raise ValueError(f"Generalized asymptotic parametrization '{lbl}' requested without angle specified.")
            umat = umat(np.radians(angle))

        if isInv:
            umat = np.flip(umat, axis=0)

        return uMatrix(lbl, umat)


def getInverseKvpLbl(lbl, sep="-"):
    if "inv" in lbl:
        return lbl.replace("inv", "")
    elif sep in lbl:
        return lbl.replace(sep, f"inv{sep}")
    else:
        return f"{lbl}inv"


if __name__ == "__main__":
    print( getUmatrix("gTinv-5").u)
    print(getUmatrix("gT-5").u)

