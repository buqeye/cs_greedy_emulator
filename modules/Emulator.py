import numpy as np
from functools import cached_property
from dataclasses import dataclass
from itertools import combinations
from collections import namedtuple

from constants import *
from RseSolver import RseSolver, LMatrix
from uMatrix import getUmatrix, getInverseKvpLbl


class Kvp:
    def __init__(self, targetKvp, baseKvp, LvecBase):
        self.targetIsBase = (baseKvp == targetKvp)
        self.targetUmat = getUmatrix(targetKvp)
        self.baseUmat = getUmatrix(baseKvp)
        self.LvecBase = LvecBase

    @property
    def baseLbl(self):
        return self.baseUmat.lbl

    @property
    def lbl(self):
        return self.targetUmat.lbl

    @cached_property
    def transformU(self):
        mat, isScalar = LMatrix.getWaveFuncU(self.targetUmat, self.baseUmat)
        if isScalar:
            return (mat[0, 0] / mat[0, 1])**2
        else:
            vec = self.linearFracTrafo(mat, self.LvecBase)
            return 1./np.outer(vec, vec)

    @staticmethod
    def linearFracTrafo(u, z):
        denom = u[0, 0] + z * u[1, 0]
        numer = u[0, 1] + z * u[1, 1]
        # note that 'z' will be an np.array most of the time, not just a scalar,
        # so we don't use 'LMatrix' here; also note that this uses u.T
        return numer / denom

    @cached_property
    def Lvec(self):
        if self.targetIsBase:
            return self.LvecBase
        else:
            return self.linearFracTrafo(self.baseUmat.u @ self.targetUmat.uInv, self.LvecBase)
            # in general {denominator, numerator} = det(u') (u'^(-1))^T u^T * {1, L}
            # but notice that det(u') is a global factor and thus doesn't matter;
            # also notice that u^T * {1, L} \propto {1, K}

    def getU(self, Upre):
        tmp = Upre if self.targetIsBase else Upre * self.transformU
        return tmp / self.targetUmat.det

    @staticmethod
    def calcInverse(U, *, usePinv=False, nugget=1e-12):
        if nugget < 0.:
            raise ValueError(f"Expected positive nugget, got {nugget}.")
        if usePinv:  # use Moore-Penrose inverse
            return np.linalg.pinv(U, rcond=nugget)
        else:  # use nugget regularization
            Ucond = np.copy(U)
            np.fill_diagonal(Ucond, Ucond.diagonal() + nugget)
            return np.linalg.inv(Ucond)

    def findStationaryApprox(self, Upre, *, takeBasisIndices=None, useLsSolver=True,
                             usePinv=False, nugget=1e-10):
        # obtain U matrix and L vector associated with current KVP
        U = self.getU(Upre)
        Lvec = self.Lvec

        # use only a subset of the basis wave functions to emulate
        if takeBasisIndices is not None:
            U = U[np.ix_(takeBasisIndices, takeBasisIndices)]
            Lvec = Lvec[np.ix_(takeBasisIndices)]

        # determine vector coefficients of trial wave function using a Lagrange multiplier
        if useLsSolver:
            # method 1) solve Lagrange equations as matrix equation, A @ x = b
            # step 1) add column to U
            basisSize, _ = U.shape  # 'U' is quadratic
            A = np.hstack((U, np.ones((basisSize, 1))))
            # step 2) add row to U
            A = np.vstack((A, np.append(np.ones(basisSize), 0)))
            # step 3) build b
            b = np.append(Lvec, 1)

            # solve matrix equation (approximately)
            x = np.linalg.lstsq(A, b, rcond=None)[0]
            cVec, lagrange = x[:-1], x[-1]
        else:
            # method 2) explicitly invert matrix U
            # compute the coefficients for the emulated wavefunction
            Uinv = self.calcInverse(U, usePinv=usePinv, nugget=nugget)
            lagrange = (np.sum(Uinv @ Lvec) - 1) / np.sum(Uinv)
            cVec = Uinv @ (Lvec - lagrange)

        # approximate L matrix
        Lapprox = (cVec @ Lvec) - (cVec @ U @ cVec) / 2.  # "ancScaling" is factored in "U"

        # return results and numerics report
        return StationaryApprox(cVec=cVec, Lmatrix=LMatrix(self.lbl, Lapprox),
                                numerics={  # "condNumber": np.linalg.cond(U),
                                          "detUInv": np.linalg.det(U), "lagrange": lagrange})


class Emulator:
    knownKvpSets = {"canonical":  {"K", "S", "T"} | {"Kinv", "Sinv", "Tinv"},
                   "small": {"K", "T"} ,  # Tinv not considered here because Tinv --> infty for large l
                   "small+": {"K", "T", "Tinv"} | {f"hadamard*-{angle}" for angle in (10, 30, 50, 60, 75, 90)},
                   "hadamard": {f"hadamard-{angle}" for angle in (0, 10, 30, 40, 60, 90, 120)},
                   "hadamard*": {f"hadamard*-{angle}" for angle in (0, 10, 30, 40, 60, 90, 120)},
                   "hadamard**": {f"hadamard**-{angle}" for angle in (0, 10, 30, 40, 60, 90, 120)},
                   "gS": {f"gS-{angle}" for angle in (0, 20, 45)},
                   "gT": {f"gT-{angle}" for angle in (0, 20, 45)}
                   }

    _ArrayEntry = namedtuple("ArrayEntry", ("index", "value"))

    def __init__(self, scattSols, *, requestedKvps="small+", partitionSize=2):
        """
        Constructor
        """
        self.scattSols = scattSols
        self.basisSize = len(self.scattSols)
        self.thetaIndepU = None
        self.weightedOverlapMat = None
        if requestedKvps in self.knownKvpSets.keys():
            self.kvpVersionsRequested = sorted(self.knownKvpSets[requestedKvps])  # sorted avoids random order of output
        else:
            self.kvpVersionsRequested = set(requestedKvps)

        self.scattExp = self.scattSols[0].scattExp
        self.grid = self.scattSols[0].grid
        self.potential = self.scattSols[0].potential
        self.kvpBaseVersion = self.scattSols[0].Lmatrix.lbl

        # self.detu = self.scattSols[0].detu
        self.ancScaling = 1./(self.scattSols[0].anc**2 * self.scattSols[0].scattExp.p)
        # we assume here all training vectors are sampled on the same grid;
        # could be added as a consistency check

        # set up variation of the basis size (aimed at removing Kohn anomalies)
        if partitionSize >= self.basisSize:
            raise ValueError(f"Requested partition size (={partitionSize}) > basis size (={self.basisSize}).")
        numPartitions = self.basisSize // partitionSize + int(self.basisSize % partitionSize > 0)
        partitions = [partitionSize * i for i in range(1, numPartitions)]

        basisIndices = np.arange(self.basisSize)
        removeBasisIndicesArray = np.split(basisIndices, partitions)
        self.takeBasisIndicesArray = [basisIndices]
        for arr in removeBasisIndicesArray:
            self.takeBasisIndicesArray.append(np.array(list(set(basisIndices)-set(arr))))

        # train the emulator
        self._train()

    @staticmethod
    def _computeSandwich(psi1, diagOperator, psi2):
        """
        For computing matrix elements of the potential in the EC basis.
        """
        return np.dot(diagOperator, psi1 * psi2)  # faster than 'np.sum(diagOperator * psi1 * psi2)'

    def _train(self):
        """
        trains the emulator
        """
        # slow implementation
        # self.thetaIndepU2 = np.empty((self.basisSize, self.basisSize), dtype=complex)
        # for indm, m in enumerate(self.scattSols):
        #     # pot = self.grid.weights * m.vr  # for use with '_computeSandwich'
        #     wmu = m.u * (self.grid.weights * m.vr)
        #     for indn, n in enumerate(self.scattSols):
        #         # self.thetaIndepU2[indm, indn] = self._computeSandwich(m.u, pot, n.u)  # not a symmetric matrix
        #         self.thetaIndepU2[indm, indn] = np.dot(wmu, n.u)  # '_computeSandwich' slightly slower

        # faster implementation
        B = np.array([n.u for n in self.scattSols])
        A = np.array([m.u * self.grid.weights * m.vr for m in self.scattSols])
        self.thetaIndepU = (A @ B.T).astype('cdouble')  # not a symmetric matrix

        # check that the two implementations return the same result (use "thetaIndepU2" for comparison)
        # if not np.allclose(self.thetaIndepU2, self.thetaIndepU, rtol=1e-12, atol=1e-12):
        #      raise ValueError("Not the same matrix!")

        # symmetrize thetaIndepU
        self.thetaIndepU += self.thetaIndepU.T  # now it's a symmetric matrix

        # prestore weighted overlap matrix to boot performance of 'emulate'
        self.weightedOverlapMat = np.array([[m.u * self.grid.weights * n.u for m in self.scattSols] for n in self.scattSols])
        # we could make use of the symmetry m <--> n; however, explicitly looping
        # over the basis wave functions actually seemed to be slightly slower.

        # initialize requested KVPs
        LvecBase = np.array([sol.Lmatrix.value for sol in self.scattSols])
        self.kvps = [Kvp(targetKvp, self.kvpBaseVersion, LvecBase) for targetKvp in self.kvpVersionsRequested]

    def emulate(self, lecs, *, exactSolutionOnly=False, useLsSolver=False, usePinv=False, nugget=1e-10,
                printResult=False, cacheExactSol=False, strictMode=False, anomalyAtol=1e-3):
        if exactSolutionOnly:  # return Smart Emulator with exact solution only (no emulation)
            return SmartKohnResult(scattSols=self.scattSols, lecs=lecs, cacheExactSol=True)

        if self.thetaIndepU is None:
            raise ValueError("Emulator needs training.")

        # slower implementation
        # thetaDepU2 = np.empty_like(self.thetaIndepU, dtype=complex)
        # pot = self.grid.weights * self.potential.eval(self.grid.points, lecs)
        # for indm, m in enumerate(self.scattSols):
        #     for indn, n in enumerate(self.scattSols):
        #         thetaDepU2[indm, indn] = self._computeSandwich(m.u, pot, n.u)

        # faster implementation
        pot = self.potential.eval(self.grid.points, lecs)
        thetaDepU = np.dot(self.weightedOverlapMat, pot).astype('cdouble')

        # # check that the two implementations return the same result (use "thetaDepU2" for comparison)
        # if not np.allclose(thetaDepU2, thetaDepU, rtol=1e-12, atol=1e-12):
        #      raise ValueError("Emulate: Not the same matrix!")

        # construct prelimary U matrix (does not contain det(u) factor)
        Upre = self.ancScaling * (2. * self.scattExp.mu) * (2. * thetaDepU - self.thetaIndepU)

        # iterate over KVP versions
        sols = []
        valuesUnnormalizedWeights = np.empty((0, 2))
        largestWeight = self._ArrayEntry(index=-1, value=0.)  # track the largest weight across different basis sets
        consistentSolutionFound = False
        for itakeBasisIndices, takeBasisIndices in enumerate(self.takeBasisIndicesArray):
            sol = SmartKohnResult(scattSols=self.scattSols, lecs=lecs, cacheExactSol=cacheExactSol)
            for kvp in self.kvps:
                # find stationary approximation of the functional associated with the KVP requested
                statSol = kvp.findStationaryApprox(Upre, takeBasisIndices=takeBasisIndices,
                                                   useLsSolver=useLsSolver, usePinv=usePinv, nugget=nugget)

                # add number of iterations
                statSol.numBasisIterReq = itakeBasisIndices+1

                # print out result (useful for debugging)
                if printResult:
                    statSol.print()

                # append current solution
                sol.add(statSol)

            # append Smart Emulator of current basis set
            sols.append(sol)

            # check for consistency among KVPs (to detect Kohn anomalies)
            consistentSolutionFound, valuesWeights = sol.consistencyCheck(atol=anomalyAtol)
            valuesUnnormalizedWeights = np.vstack((valuesUnnormalizedWeights, valuesWeights))

            # update 'largestWeight'
            index = np.argmax(valuesWeights[:, 1])
            if valuesWeights[index, 1] > largestWeight.value:
                largestWeight = self._ArrayEntry(index=itakeBasisIndices, value=valuesWeights[index, 1])

            if consistentSolutionFound:
                break  # no need to modify the basis size any further

        # handle the case where a consistent solution has been found
        if consistentSolutionFound:
            return sols[-1]
            # sols[-1] is the first basis set that passes the consistency check,
            # but one might think of more options

        # handle the case where Kohn anomalies could not be mitigated
        valuesUnnormalizedWeights = np.array(valuesUnnormalizedWeights)
        message = f"Couldn't mitigate Kohn anomaly for {lecs} in '{str(self.scattExp)}'."
        if strictMode:  # raise exception
            raise AssertionError(message)
        else:  # try to obtain a better estimate in terms of a weighted sum (no guarantee for success)
            print(message)

            # estimate L matrix
            weights = valuesUnnormalizedWeights[:, 1]
            weights /= np.sum(weights)  # normalize weights
            mean = np.dot(valuesUnnormalizedWeights[:, 0], weights)

            # add estimated L matrix to the SmartEmulator that has the largest weight and return
            sols[largestWeight.index].results["recommended"] = StationaryApprox(cVec=None, Lmatrix=LMatrix(self.kvpBaseVersion, mean), numerics={})
            return sols[largestWeight.index]


@dataclass
class StationaryApprox:
    cVec: np.array
    Lmatrix: LMatrix
    numerics: dict
    detUInvTol: float = 1e2
    lagrangeTol: float = 1e4

    @property
    def lbl(self):
        return self.Lmatrix.lbl

    @property
    def redFlagNumerics(self):
        cond1 = np.abs(self.numerics["detUInv"]) > self.detUInvTol
        cond2 = np.abs(self.numerics["lagrange"]) > self.lagrangeTol
        return cond1 or cond2
        # A problem is considered well-conditioned,
        # if its condition number is in the order of 10, 100, or 1000,
        # and ill-conditioned if its condition number is >1e6
        # [we'll be not so strict here]
        # Note that this function could be (re)defined in terms of 'self.badness',
        # but we want to disentangle here the two conditions

    @property
    def badness(self):
        tmp1 = self.numerics["detUInv"] / self.detUInvTol
        tmp2 = self.numerics["lagrange"] / self.lagrangeTol
        return np.sqrt(np.abs(tmp1)**2 + np.abs(tmp2)**2)

    def print(self):
        print(f'ci({self.Lmatrix.lbl}) =\t', np.real(self.cVec), '; sum(ci) =',
              np.sum(self.cVec), np.linalg.norm(self.cVec), "; delta =", self.Lmatrix.phaseShift,
              "; K matrix =", self.Lmatrix.valueAs("K"),
              " (anomalous)" if self.redFlagNumerics else "", self.numerics)


class SmartKohnResult:
    def __init__(self, scattSols, lecs, cacheExactSol=False):
        self.scattSols = scattSols
        self.lecs = lecs
        self.results = dict()
        self.exactSolution = self.calcExactSolution if cacheExactSol else None
        self.fullyConsistentResults = False  # true if all consistency checks pass
        self.numBasisIterReq = 0

    @property
    def baseKvp(self):
        return self.scattSols[0].Lmatrix.lbl

    def add(self, kohnRes):
        self.results[kohnRes.lbl] = kohnRes

    CheckResult = namedtuple("CheckResult", "mean residual")

    def getCheckResults(self, Llbl):
        # computes "L * L^(-1)" for the KVPs available, where L the matrix of the
        # exact scattering solutions used for training
        checks = dict()
        for elem in combinations(self.results.keys(), 2):
            A = self.results[elem[0]].Lmatrix.valueAs(Llbl)  # L matrix
            B = self.results[elem[1]].Lmatrix.valueAs(getInverseKvpLbl(Llbl))  # Linv matrix
            tmp = A*B
            # to avoid biasing the residuals (i.e., the checks), enforce the product to be >= 1
            residual = np.max([1e-14, np.abs(tmp - 1.), np.abs(1./tmp - 1.)])
            checks[(elem[0], elem[1])] = self.CheckResult(np.mean([A, 1./B]), residual)
        return checks

    def _calcValuesWeightsFromResiduals(self, resDict, equalWeights=False, normalizeWeights=False):
        values = np.array([elem.mean for elem in resDict.values()])

        if equalWeights:
            weights = np.ones(len(resDict))
        else:
            weights = 1./np.array([elem.residual for elem in resDict.values()])

        if normalizeWeights:
            weights /= np.sum(weights)

        return values, weights

    def consistencyCheck(self, atol=1e-3):
        checkResults = self.getCheckResults(self.baseKvp)
        checkResultsPassed = dict(filter(lambda elem: elem[1].residual < atol, checkResults.items()))
        numChecksPassed = len(checkResultsPassed)
        self.fullyConsistentResults = (numChecksPassed == len(checkResults))

        # Two different strategies if some of the checks pass: 1) apply equal weights to the ones that pass, or
        # 2) weight them by their residuals. The latter requires a special treatment of zero residuals
        # (the weights are the inverse of the residuals), but is more robust to variations in epsilon (and might
        # lead to better agreement with the exact solution.
        tmp = checkResultsPassed if numChecksPassed else checkResults
        values, unnormWeights = self._calcValuesWeightsFromResiduals(tmp)

        mean = np.dot(values, unnormWeights) / np.sum(unnormWeights)
        self.results["recommended"] = StationaryApprox(cVec=None, Lmatrix=LMatrix(self.baseKvp, mean), numerics={})

        return bool(numChecksPassed), np.transpose([values, unnormWeights])

    @property
    def kvpsAvail(self):
        return list(self.results.keys())

    @property
    def scattExp(self):
        return self.scattSols[0].scattExp

    def fl(self, kvpLbl):
        return self.results[kvpLbl].Lmatrix.fl(self.scattExp.p)

    def sigmaL(self, kvpLbl):
        return self.results[kvpLbl].Lmatrix.sigmaL(self.scattExp.potential.channel.L, self.scattExp.p)

    def dsigmaL(self, kvpLbl, atheta, deg=True):
        return self.results[kvpLbl].Lmatrix.dsigmaL(self.scattExp.potential.channel.L,
                                                    self.scattExp.p, atheta, deg=deg)
    def phaseShift(self, kvpLbl):
        return self.results[kvpLbl].Lmatrix.phaseShift

    def _getDefaultLbl(self, Llbl=None):
         return Llbl if Llbl is not None else self.kvpsAvail[0]

    def wfsMatrix(self, Llbl=None, prime=False):
        return np.array([sol.waveFunctionAs(self._getDefaultLbl(Llbl), prime=prime) for sol in self.scattSols]).T

    def u(self, Llbl=None, prime=False):
        lbl = self._getDefaultLbl(Llbl)
        return self.wfsMatrix(lbl, prime=prime) @ self.results[lbl].cVec

    def getLsFit(self, Llbl=None, phaseShiftOnly=False, prime=False):
        lbl = self._getDefaultLbl(Llbl)
        ciFit = np.dot(np.linalg.pinv(self.wfsMatrix(lbl, prime=prime)),
                       self.calcExactSolution.waveFunctionAs(lbl, prime=prime))
        Lmat = LMatrix(lbl, ciFit @ np.array([sol.Lmatrix.valueAs(lbl) for sol in self.scattSols]))
        if phaseShiftOnly:
            return Lmat.phaseShift
        else:
            return ciFit, self.wfsMatrix(lbl, prime=prime) @ ciFit, Lmat

    @cached_property
    def calcExactSolution(self):
        ind = 0
        scattSol = solve(self.scattSols[ind].scattExp, self.scattSols[ind].grid, [self.lecs], asympParam="T")
        return scattSol[0]