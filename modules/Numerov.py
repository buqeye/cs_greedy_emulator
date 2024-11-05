import numpy as np
from scipy.linalg import solve_banded
from scipy.sparse import spdiags


def diag_ord_form_to_mat(ab, ab_l_and_u, toarray=False):
    ab_B, N = ab.shape
    assert ab_B == sum(ab_l_and_u) + 1
    spmat = spdiags(ab, diags=np.arange(ab_l_and_u[1], -ab_l_and_u[0]-1, -1))
    return spmat.toarray() if toarray else spmat.tocsc()


class AllAtOnceNumerov:
    def __init__(self, xn, g, s=None, g_s=None, y0=0., y1=1., params=None, self_test=True) -> None:
        # build Numerov matrix
        ## preliminaries
        self.xn = xn
        self.N = len(xn)-1
        self.h = np.diff(xn)[0]  # assuming an equidistant grid
        self.step_fac = self.h**2 / 12
        self.params = params
        self.g = g
        self.s = s
        self.g_s = g_s
        if g_s is None:
            self.gn = self.g(xn, self.params)
            self.sn = np.zeros_like(self.gn) if s is None else s(xn, self.params)
        else:
            self.gn, self.sn = self.g_s(xn, params)
            self.g = self.s = None
        self.n_theta = self.gn.shape[1]
        self.y0 = y0
        self.y1 = y1

        # construct banded (!) matrix
        self.A_l_and_u= 2, 0
        self.A_bandwidth = sum(self.A_l_and_u)+1
        self.Abar_tensor = np.einsum("i,ja->aij", 
                                  np.array([1., 10., 1.]) * self.step_fac, 
                                  self.gn[2:,:], optimize="greedy")
        self.Abar_tensor[0, ...] += np.outer(np.array([1, -2, 1]), np.ones(self.N-1))
        self.Abar_tensor[:, 2, -2:] = 0
        self.Abar_tensor[:, -2, -1:] = 0
        # optional: we set here the ununsed elements to zero

        # build Numerov inhomogeneous term `s`
        mat = spdiags(np.outer([1., 10., 1.], np.ones(self.N+1)), 
                      diags=(0,1,2), m=self.N-1, n=self.N+1)
        l = params["scattExp"].l
        self.S_tensor = self.step_fac * mat @ self.sn 
        tmp = -self.step_fac * self.gn[1] * y1
        self.S_tensor[0, :] += 10*tmp
        self.S_tensor[1, :] += tmp
        self.S_tensor[0, 0] += ((l == 1)/6. + 2) * y1
        self.S_tensor[1, 0] += -y1
        self.S_tensor = self.S_tensor.T

        if self_test:
            self.test_prestore()

    @property
    def A_tensor(self):
        mat = spdiags(np.outer([1., 10., 1.], np.ones(self.N)), 
                      diags=(0,-1,-2), m=(self.N-1), n=self.N-1).toarray()
        tmp = self.step_fac * np.einsum("ij,ja->aij", mat, self.gn[2:,:], optimize=True)
        tmp[0, ...] += diag_ord_form_to_mat(np.outer(np.array([1, -2, 1]), np.ones(self.N-1)), 
                                            ab_l_and_u=self.A_l_and_u, toarray=True)
        return tmp

    def test_prestore(self, atol=1e-15):
        print("testing")
        def d(i,j):
            return int(i==j)
        l = self.params["scattExp"].l
        Sja = np.ones((self.n_theta, self.N-1)) * 1e9
        for j in range(1, self.N-1+1):
            for a in range(0, self.n_theta):
                Sja[a, j-1] = d(a,0) * ((2+1/6 * d(l,1)) * d(j,1) - d(j,2)) * self.y1 \
                    + self.step_fac * ( -self.gn[1, a] * (10 * d(j,1) + d(j,2) ) *self.y1  \
                                +  self.sn[j+1,a] + 10*self.sn[j,a] + self.sn[j-1,a])
                
        assert np.allclose(Sja, self.S_tensor, atol=atol, rtol=0.), "Sj inconsistent"

        def Dbar(i,j):
            return 1 - d(3,i) * d(j,self.N-2) - (d(2,i) + d(3,i)) * d(j,self.N-1)

        Aija = np.ones((self.n_theta, 3, self.N-1)) * 1e9
        for i in range(1,3+1):
            for j in range(1, self.N-1+1):
                for a in range(0, self.n_theta):
                    Aija[a, i-1,j-1] = d(a,0) * (1-3 * d(i,2)) * Dbar(i,j) \
                                + self.step_fac * self.gn[j+1,a] * (1 + 9 * d(i,2)) * Dbar(i,j)

        assert np.allclose(Aija, self.Abar_tensor, atol=atol, rtol=0.), "Aijk inconsistent"

    def get_linear_system(self, theta, ret_diag_form=True, file_dump=False):
        A = np.tensordot(theta, self.Abar_tensor, axes=1)
        if not ret_diag_form:
            A = diag_ord_form_to_mat(A, ab_l_and_u=self.A_l_and_u, toarray=True)
        s = np.tensordot(theta, self.S_tensor, axes=1) #  @ theta
        if file_dump:
            np.savetxt("class_A.csv", A)
            np.savetxt("class_s.csv", s)
        return A, s

    def solve(self, thetas):
        thetas = np.asarray(thetas)
        ret = []
        for theta in thetas:
            A_banded, s = self.get_linear_system(theta)
            sol = solve_banded(l_and_u=self.A_l_and_u, ab=A_banded, b=s)
            ret.append(np.concatenate([[self.y0, self.y1], sol]))
        return np.array(ret).T
    
    def residuals(self, xtilde, theta, squared=True, calc_error_bounds=False):
        A, s = self.get_linear_system(theta, ret_diag_form=False)
        # A = diag_ord_form_to_mat(A_banded, ab_l_and_u=self.A_l_and_u, toarray=True)
        residual = s - A @ xtilde
        norm_residual = np.linalg.norm(residual)
        lower_bound = None 
        upper_bound = None
        if calc_error_bounds:
            svals = np.linalg.svd(A, compute_uv=False)
            lower_bound = norm_residual / svals[0]  # sval_lm
            upper_bound = norm_residual / svals[-1]  # sval_sm
            assert lower_bound <= upper_bound, "lower bound has to be <= upper bound"
        return norm_residual**2 if squared else norm_residual, lower_bound, upper_bound



def numerov(xn, g, y0, y1, s=None, solve=True, unittest=False, params=None, file_dump=False):
    """
    implements the (N-2)x(N-2) Numerov matrix
    """
    # preliminaries
    if s is None:
        s = lambda x, args: 0.*x
    g_arr = g(xn, params)  # g and s could be sampled simultaneously
    s_arr = s(xn, params)
    N = len(xn) - 1
    h = np.diff(xn)[0]

    def K(gn, xi=1):
        return 1. + xi * h**2 / 12 * gn
    
    K1 = K(g_arr, 1)

    # build rhs vector s
    l = params["scattExp"].l
    rhs = h**2 /12 * np.array([s_arr[n] + 10*s_arr[n-1] + s_arr[n-2] for n in range(2, N+1)])
    rhs[0] += ((l == 1)/6.) * y1 + 2*K(g_arr[1], -5.) * y1
    rhs[1] -= y1 * K1[1] 

    # build matrix in diagonal ordered form
    ab = np.empty((3, N-1))
    ## first row
    ab[0, :] = K1[2:] 

    ## second row
    ab[1, :] = -2*K(g_arr, -5.)[2:]
    ab[1, -1] = 0 # not necessary but useful

    ## third row
    ab[2, :] = K1[2:]
    ab[2, -2:] = 0  # not necessary but useful
    
    # solve system   
    ab_sparse = diag_ord_form_to_mat(ab, ab_l_and_u=(2,0))
    
    if file_dump:
        np.savetxt("orig_ab.csv", ab)
        np.savetxt("orig_s.csv", rhs)

    if solve:
        # from scipy.sparse.linalg import spsolve
        # sol_sparse = spsolve(ab_sparse, rhs)

        # if unittest:
        from scipy.linalg import solve_banded
        sol = solve_banded(l_and_u=(2, 0), ab=ab, b=rhs)
        # assert np.allclose(sol, sol_sparse)

        return ab_sparse, rhs, np.concatenate(([y0, y1], sol))
    else:
        return ab_sparse, rhs, np.concatenate(([y0, y1]))


def numerov_iter(xn, g, y0=0, y1=0, s=None, params=None):
    if s is None:
        s = lambda x, args: 0.*x
    h = np.diff(xn)[0]

    def K(x, xi=1):
        return 1. + xi * h**2 / 12 * g(x, params)

    N = len(xn) - 1
    yn = np.empty(N+1)

    yn[0] = y0
    yn[1] = y1

    for n in range(1, N):
        yn[n+1] = 2 * yn[n] * K(xn[n], -5) - yn[n-1] * K(xn[n-1], 1)
        yn[n+1] += h**2/12 * (s(xn[n+1], params) + 10*s(xn[n], params) + s(xn[n-1], params))
        yn[n+1] /= K(xn[n+1], 1)
    return yn