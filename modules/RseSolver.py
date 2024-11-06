from scipy.integrate import solve_ivp
from scipy.special import spherical_jn, spherical_yn, eval_legendre
import matplotlib.pyplot as plt
import numpy as np
from uMatrix import getUmatrix
from constants import *


def free_solutions_F_G(l, rho=None, p=None, r=None, derivative=False):
    """Riccati-Bessel functions"""
    if rho is None:
        if p is None or r is None:
            raise ValueError ("either `rho` or `p` and `r` need to be specified")
        rho = r*p   # here's no "np.pi/2 * l" because we use the j_l and eta_l below
    tmp = np.array([spherical_jn(l, rho), -spherical_yn(l, rho)])
    if derivative == False:
        ret = rho * tmp
    else:
        if p is None:
            raise ValueError (" `p` needs to be specified to compute derivative of F and G")
        tmp += rho * np.array([spherical_jn(l, rho, derivative=True), 
                              -spherical_yn(l, rho, derivative=True)])
        ret = p * tmp  # due to the derivative w.r.t. r (not pr)
    return np.asarray(ret).T

class LMatrix:
    def __init__(self, lbl, value=None):
        self.value = value
        self.uMat = getUmatrix(lbl)

    @property
    def lbl(self):
        return self.uMat.lbl

    def linearFracTrafo(self, u):
        tmp = u.T @ self._toFrac(self.value)
        return tmp[1] / tmp[0]  # {denominator, numerator}

    def _toFrac(self, value):
        return np.array([1, value])  # {denominator, numerator}

    def valueAs(self, LprimeLbl=None):
        if (LprimeLbl is None) or (self.lbl == LprimeLbl):
            return self.value
        else:
            up = getUmatrix(LprimeLbl)
            return self.linearFracTrafo(self.uMat.u @ up.uInv)
            # in general {denominator, numerator} = det(u') (u'^(-1))^T u^T * {1, L}
            # but notice that det(u') is a global factor and thus doesn't matter;
            # also notice that u^T * {1, L} \propto {1, K},

    @property
    def K(self):
        # just to illustrate the special case for the K matrix
        if self.lbl == "K":
            return self.value
        else:
            return self.linearFracTrafo(self.uMat.u)

    @property
    def S(self):
        # just to illustrate the special case for the S matrix
        if self.lbl == "S":
            return self.value
        else:
            return self.linearFracTrafo(self.uMat.u @ np.array([[1, 1], [-1j, 1j]]))

    @property
    def phaseShift(self):
        return np.arctan(self.K) * degrees  # K = tan delta
        # np.arctan() does not support complex types, so we can't use
        # the fraction representation of the K matrix here in general

        # return np.log(self.S)/2j * degrees  # S = exp[2i delta]
        # note that using np.angle() would be incorrect here, because it
        # returns only the real part of the (in general complex) phase shift

    def fl(self, p):  # p in fm**-1
        return (self.S - 1.) / (2j * p)  # fm

    def sigmaL(self, l, p):
        cs = (4.*np.pi) * (2*l + 1) * np.abs(self.fl(p)) ** 2  # fm**2
        return cs * 10.  # mb [100 fm**2 = 1 b]

    def dsigmaL(self, l, p, atheta, deg=True):  # p in fm**-1
        """
        Computes a given partial wave's contribution to the differential section
        See Eqs. (11.2) to (11.4) in Taylor’s book.
        """
        theta = atheta/degrees if deg else atheta
        diffCs = (2*l + 1) * self.fl(p) * eval_legendre(l, np.cos(theta))  # fm
        return diffCs * np.sqrt(10.)  # sqrt(mb)

    @staticmethod
    def getWaveFuncU(up, u):
        tmp = np.array([[+u.u[1, 1], +up.u[1, 1]],
                        [-u.u[1, 0], -up.u[1, 0]]])
        mat = u.u @ (tmp @ np.diag((up.det, u.det)))
        isSingular = np.isclose(np.abs(np.linalg.det(mat)), 0)
        return mat, isSingular

    def waveFunctionFactor(self, LprimeLbl):
        if self.lbl == LprimeLbl:
            return 1.
        up = getUmatrix(LprimeLbl)
        mat, isScalar = self.getWaveFuncU(up, self.uMat)
        if isScalar:
            return mat[0, 1] / mat[0, 0]
            # mat [1, 0] is always zero, so this is the correct limit value
        else:
            return self.linearFracTrafo(mat)

    def printWaveFuncDependencies(self, Lrequested=None):
        Larr = ("K", "S", "T", "Kinv", "Sinv", "Tinv") if (Lrequested is None) else Lrequested
        print(Larr)
        for mat0 in Larr:
            u = getUmatrix(mat0)
            # print(self.waveFunctionFactor(mat0), end="\t")
            for mat1 in Larr:
                up = getUmatrix(mat1)
                _, isScalar = self.getWaveFuncU(up, u)
                print(isScalar, end="\t")
            print("")


class ScatteringSolution:
    def __init__(self, scattExp, grid, f, f_lbl, 
                 vr, anc, Llbl, matching, 
                 fprime=None, matching_method="lsqfit"):
        self.scattExp = scattExp
        self.potential = scattExp.potential
        self.grid = grid
        self.matching = matching
        self.anc = anc
        
        assert f_lbl in ("u", "chi"), f"unknown label '{f_lbl}'"
        if fprime is None and matching_method != "lsqfit":  # lsqfit doesn't need u'
            self.fprime = np.gradient(f, self.grid.points, edge_order=2)
        self.u, self.uprime = self.convert_u_chi(f_lbl, f, fprime, to="u")
        
        self.vr = vr  # V(r) sampled on 'grid'
        self.Lmatrix = None
        if matching:
            self.match(anc, param=Llbl, method=matching_method)  # sets the scattering matrix 'self.Lmatrix' and 'self.anc'
        
        self.chi, self.chiprime = self.convert_u_chi("u", self.u, self.uprime, to="chi")

    def convert_u_chi(self, f_lbl, f, fprime, to="u"):
        if not self.matching:
            raise ValueError("matching required for wave function conversion")
        
        if f_lbl == to:
            return f, fprime
        else:
            if f_lbl == "u" and to == "chi":    
                sign = -1.
                prefac_f = 1./self.anc
                prefac_glob = 1.
            elif f_lbl == "chi" and to == "u":  
                sign = 1.
                prefac_f = 1.
                prefac_glob = 1./self.anc
            else:
                raise NotImplementedError
            p = self.scattExp.p
            l = self.scattExp.l
            ret_f = prefac_f * np.copy(f)
            ret_fprime = prefac_f * np.copy(fprime)
            ret_f       += sign * free_solutions_F_G(l, p=p, r=self.grid.points, derivative=False)[:, 0]
            ret_fprime  += sign * free_solutions_F_G(l, p=p, r=self.grid.points, derivative=True)[:, 0]
        return prefac_glob * ret_f, prefac_glob * ret_fprime
    
    def free_solutions_F_G(self, derivative=False):
        return free_solutions_F_G(self.scattExp.l, p=self.scattExp.p, r=self.grid.points, derivative=derivative)

    @property
    def r0(self):
        return self.grid.points[0]

    def waveFunctionAs(self, LprimeLbl=None, prime=False):
        tmp = self.uprime if prime else self.u
        if (LprimeLbl is None) or (self.Lmatrix.lbl == LprimeLbl):
            return tmp
        else:
            pre = 1./self.Lmatrix.waveFunctionFactor(LprimeLbl)
            return pre * tmp

    @property
    def fl(self):
        return self.Lmatrix.fl(self.scattExp.p)

    @property
    def sigmaL(self):
        return self.Lmatrix.sigmaL(self.scattExp.potential.channel.L, self.scattExp.p)

    def dsigmaL(self, atheta, deg=True):
        return self.Lmatrix.dsigmaL(self.scattExp.potential.channel.L,
                                    self.scattExp.p, atheta, deg=deg)

    @property
    def phaseShift(self):
        return self.Lmatrix.phaseShift

    @property
    def cond(self):
        assert self.linear_system is not None
        from numpy.linalg import cond
        return cond(self.linear_system[0].toarray()) 

    @property
    def singular_values(self):
        assert self.linear_system is not None
        from numpy.linalg import svd 
        return svd(self.linear_system[0].toarray(), full_matrices=False, compute_uv=False)
        # it would be better to use a routine for sparse matrices like the following, which does
        # not converge for some reason
        # from scipy.sparse.linalg import svds
        # s1= svds(self.linear_system[0], k=1, tol=1e-2, 
        #          maxiter=100, solver="lobpcg", 
        #          return_singular_vectors=False, which="SM")


    def _getPhiPhiPrime(self, r, uMat):
        l = self.scattExp.l
        phi = uMat.u @ free_solutions_F_G(l=l, r=r, p=self.scattExp.p, derivative=False).T  # (3.1.13)
        phiPrime = uMat.u @ free_solutions_F_G(l=l, r=r, p=self.scattExp.p, derivative=True).T  # (3.1.13)
        return phi.T, phiPrime.T

    def match(self, anc, *, method="lsqfit", param="K", num_pts=25):
        assert num_pts < int(0.1 * self.grid.getNumPointsTotal), "matching may use too many points"
        self.Lmatrix = LMatrix(param)
        self.rmatch = self.grid.points[-num_pts:]
        phi, phiPrime = self._getPhiPhiPrime(self.rmatch, self.Lmatrix.uMat)

        if method == "logderiv":
            logDeriv = self.u[-num_pts:]/self.uprime[-num_pts:]
            L_arr = -(phi[:,0]-logDeriv*phiPrime[:,0])/(phi[:,1]-logDeriv*phiPrime[:,1])     
            scale = np.mean(anc*(phi[:,0] + L_arr*phi[:,1]) / self.u[-num_pts:])
            L = np.mean(L_arr)
        elif method == "lsqfit":
            assert num_pts >= 2, "the number of points for matching has to be >= 2, got {num_pts}"
            a, b = np.linalg.lstsq(phi[-num_pts:,:2], self.u[-num_pts:], rcond=None)[0] 
            scale = 1 / a
            Kmat = b * scale
            L = Kmat if param == "K" else LMatrix("K", value=Kmat).valueAs(param)
        else:
            raise NotImplementedError(f"wave function matching method '{method}' unknown")

        self.Lmatrix.value = L
        self.scaling_factor = scale
        self.u *= scale
        self.uprime *= scale
        self.anc = anc

    def plot(self):
        plt.plot(self.grid.points, self.u)
        #plt.show()


class RseSolver:
    def __init__(self, scattExp, grid, inhomogeneous=True):
        self.scattExp = scattExp
        self.grid = grid
        self.inhomogeneous = inhomogeneous
        self.u_lbl = "chi" if inhomogeneous else "u"
        self.rseParams = {"grid": grid, "scattExp": scattExp, "potential": scattExp.potential, "inhomogeneous": inhomogeneous}
        
        # setup Numerov solver (for affine parameter dependences)
        from Numerov import AllAtOnceNumerov
        r0 = self.grid.points[0]
        r1 = self.grid.points[1]
        l = self.scattExp.l
        y1 = 0. if inhomogeneous else 1.  # can be scaled by an arbitrary constant
        y0 = y1 * (r0/r1)**(l+1)
        self.numerov_solver = AllAtOnceNumerov(xn=grid.points, y0=y0, y1=y1, 
                                               g=None, g_s=g_s_affine, params=self.rseParams)
    
    @property
    def potential(self):
        return self.scattExp.potential

    def _rse_coupled_ode(self, r, u, params):
        l = self.scattExp.l
        mu = self.scattExp.mu
        E = self.scattExp.en
        p = self.scattExp.p
        potential = self.potential
        lecs = params["lecs"]
        centrifugal = l * (l + 1) / r ** 2. if l > 0 else 0.
        second = (centrifugal + (2. * mu) * (potential.eval(r, lecs) - E)) * u[0]        
        if self.inhomogeneous:
            pr = p*r
            second += (2. * mu) * potential.eval(r, lecs) * spherical_jn(l, pr) * pr
        return [u[1], second]
    
    def _solve_ivp(self, lecs, params):       
        # solve the radial SE
        rseParams = {**params, "lecs": lecs}
        r0 = params["grid"].points[0]
        l = params["scattExp"].l
        assert r0 > 1e-25, "RK requires first grid point with nonzero value"
        yp0 = 0. if params["inhomogeneous"] else 1.  # can be scaled by an arbitrary constant
        y0 = r0 * yp0 / (l+1)

        sol = solve_ivp(self._rse_coupled_ode, 
                        (params["grid"].start, params["grid"].end), 
                        [complex(y0), complex(yp0)],
                        method=params["method"], 
                        args=(rseParams,), 
                        t_eval=params["grid"].points, 
                        rtol=1e-12, atol=1e-12)
        return sol.y[0], sol.y[1]

    def solve(self, lecList, matching=True, method='RK45', 
              asympParam="K", reduced_output=False):
        """
        Solves the single channel 2-body scattering problem by using a Runge-Kutta integrator
        from the scipy.integrate package, and then applying scattering boundary conditions.
        """
        if np.abs(self.grid.start) > 1e-6:
            raise ValueError(f"Grid doesn't start at zero; grid.points[0]={self.grid.points[0]}")

        rseParams = {**self.rseParams, "asympParam": asympParam, "method": method, "matching": matching}
        
        uprime_array = None
        if method == "Numerov_affine":
            if isinstance(lecList, np.ndarray):
                lecList_array = lecList
            else:
                lecList_array = self.potential.lec_array_from_dict(lecList)
            u_array = self.numerov_solver.solve(lecList_array)
        else:
            u_array = []
            lecList_dict = self.scattExp.potential.get_lec_dict(lecList)
            if method == "Numerov":
                from Numerov import numerov
                for lecs in lecList_dict:
                    rseParams["lecs"] = lecs
                    ab_sparse, rhs, u = numerov(xn=self.grid.points, 
                                                y0=self.numerov_solver.y0, 
                                                y1=self.numerov_solver.y1, 
                                                s=s if self.inhomogeneous else None, 
                                                g=g, params=rseParams)
                    u_array.append(u.astype(complex))
            else:
                uprime_array = []
                for lecs in lecList_dict:
                    rseParams["lecs"] = lecs
                    u, uprime = self._solve_ivp(lecs, rseParams)
                    u_array.append(u)
                    uprime_array.append(uprime)
                uprime_array = np.array(uprime_array).T
            u_array = np.array(u_array).T

        if uprime_array is None:
            uprime_array = np.gradient(u_array, self.grid.points, axis=0, edge_order=2)

        scattSols = []
        for ilecs, lecs in enumerate(lecList):
            scattSol = ScatteringSolution(scattExp=self.scattExp, 
                                          vr=None, # self.potential.eval(self.grid.points, lecs),
                                          grid=self.grid, 
                                          f=u_array[:,ilecs], 
                                          fprime=uprime_array[:,ilecs], 
                                          f_lbl=self.u_lbl, 
                                          anc=1., # /self.scattExp.p,
                                          Llbl=asympParam,
                                          matching=matching)
            scattSols.append(scattSol)

        if reduced_output:
            Lmatrix_values = np.array([scattSol.Lmatrix.value for scattSol in scattSols])
            if self.inhomogeneous:
                uarr = np.array([scattSol.chi for scattSol in scattSols]).T
            else:
                uarr = np.array([scattSol.u for scattSol in scattSols]).T
            return uarr, Lmatrix_values
        else:
            return scattSols

    
def g(r, params):

    l = params["scattExp"].l
    mu = params["scattExp"].mu
    E = params["scattExp"].en
    potential = params["potential"]
    lecs = params["lecs"]
    centrifugal = -l * (l + 1) / r ** 2. if l > 0 else 0.
    return centrifugal - (2. * mu) * (potential.eval(r, lecs) - E)

def s(r, params):
    from scipy.special import spherical_jn
    l = params["scattExp"].l
    mu = params["scattExp"].mu
    p = params["scattExp"].p
    potential = params["potential"]
    lecs = params["lecs"]
    pr = p*r
    return potential.eval(r, lecs) * spherical_jn(l, pr) * pr * (2. * mu) 

def g_s(r, params):
    l = params["scattExp"].l
    mu = params["scattExp"].mu
    E = params["scattExp"].en
    p = params["scattExp"].p
    potential = params["potential"]
    lecs = params["lecs"]
    pot = potential.eval(r, lecs)
    centrifugal = -l * (l + 1) / r ** 2. if l > 0 else 0.
    g_arr = centrifugal - (2. * mu) * (pot - E)
    s_arr = pot * spherical_jn(l, p*r) * p*r * (2. * mu) 
    return g_arr, s_arr

def g_s_affine(r, params):
    l = params["scattExp"].l
    mu = params["scattExp"].mu
    E = params["scattExp"].en
    p = params["scattExp"].p
    potential = params["potential"]
    V_arr = potential.evalAffine(r)
    g_arr = - (2. * mu) * V_arr
    centrifugal = -l * (l + 1) / r ** 2. if l > 0 else 0.
    g_arr[:, 0] += centrifugal + (2. * mu) * E
    if params["inhomogeneous"]:
        s_arr = np.einsum("i,ij->ij", spherical_jn(l, p*r) * p*r * (2. * mu), V_arr, optimize=True)
    else:
        s_arr = np.zeros_like(g_arr)
    return g_arr, s_arr