from RseSolver import RseSolver
import numpy as np
from Grid import Grid
from Numerov import numerov, AllAtOnceNumerov
from scipy.linalg import orth


class AffineGROM:
    def __init__(self, scattExp, grid, free_lecs, 
                 num_snapshots_init=3, num_snapshots_max=15, 
                 approach="pod", pod_rcond=1e-12, 
                 init_snapshot_lecs=None,
                 greedy_max_iter=5, 
                 mode="linear",
                 seed=10203) -> None:
        # internal book keeping
        self.scattExp = scattExp
        self.grid = grid
        self.free_lecs = free_lecs
        assert num_snapshots_init < num_snapshots_max, "can't have more initial snapshots than maximally allowed"
        self.num_snapshots_init = num_snapshots_init
        self.num_snapshots_max =  num_snapshots_init if approach == "pod" else num_snapshots_max
        self.approach = approach
        self.pod_rcond = pod_rcond
        self.greedy_max_iter = greedy_max_iter
        self.mode = mode
        self.greedy_logging = None
        self.seed = seed

        # FOM solver (all-at-once Numerov)
        self.rse_solver = RseSolver(scattExp=self.scattExp, grid=self.grid, inhomogeneous=True)
        self.numerov_solver = self.rse_solver.numerov_solver
        
        # training and greedy algorithm if requested
        self.training(init_snapshot_lecs)
        if self.approach == "greedy":
            self.greedy_algorithm()
 
    @property
    def potential(self):
        return self.scattExp.potential
    
    def wavefct(self, cvec):
        return self.snapshot_matrix @ cvec
    
    def Lmatrix(self, cvec):
        return self.snapshot_Lvec @ cvec
                
    def training(self, snapshot_lecs):
        # if snapshot LECs are user-provided, use them; if not, run Latin Hypercube sampling
        if snapshot_lecs is None:
            self.lec_all_samples = self.potential.getLecsSample(self.free_lecs, 
                                                                as_dict=False,
                                                                n=self.num_snapshots_max, 
                                                                mode=self.mode,
                                                                seed=self.seed)
            # search-type runs could also be done using random samples via sorted(..., key=lambda dictx: dictx["V0"])
        else:
            self.lec_all_samples = snapshot_lecs
            self.num_snapshots_init = len(self.lec_all_samples)
            self.num_snapshots_max = self.num_snapshots_init

        # initial snapshot selection (random)
        rng = np.random.default_rng(seed=self.seed)
        self.lec_snapshots_idxs = set(rng.choice(range(self.num_snapshots_max), 
                                                 size=self.num_snapshots_init, replace=False))
        
        # building snapshot matrix for chi, not Psi
        init_lecs = np.take(a=self.lec_all_samples, indices=list(self.lec_snapshots_idxs), axis=0)
        self.snapshot_matrix, self.snapshot_Lvec = self.simulate(init_lecs)
        self.snapshot_matrix2 = self.rse_solver.solve(lecList=init_lecs, method="Numerov_affine", 
                                                      asympParam="K", matching=True, reduced_output=False)

        self.all_snapshot_idxs = set(range(len(self.lec_all_samples)))

        # POD snapshot matrix if requested
        if self.approach == "pod":
            self.apply_pod(update_offline_stage=False)

        # prestore arrays for efficient offline/online decomposition
        self.update_offline_stage()
    
    @staticmethod
    def truncated_svd(matrix, rcond=None):
        # modified from scipy's `orth()` function:
        # https://github.com/scipy/scipy/blob/v1.14.1/scipy/linalg/_decomp_svd.py#L302-L347
        from scipy.linalg import svd
        u, s, vh = svd(matrix, full_matrices=False)
        M, N = u.shape[0], vh.shape[1]
        if rcond is None:
            rcond = np.finfo(s.dtype).eps * max(M, N)
        tol = np.amax(s, initial=0.) * rcond
        num = np.sum(s > tol, dtype=int)  # = r
        Q = u[:, :num]
        return Q, s[:num], vh.conjugate().transpose()[:, :num]

    def apply_pod(self, update_offline_stage=False):
        prev_rank = self.snapshot_matrix.shape[1]
        Ur, S, Vr = self.truncated_svd(self.snapshot_matrix, rcond=self.pod_rcond)
        self.snapshot_Lvec = np.linalg.multi_dot((self.snapshot_Lvec, Vr, np.diag(1./S))) #TODO: could be written without constructing the diag matrix
        self.snapshot_matrix = Ur
        
        curr_rank = self.snapshot_matrix.shape[1]
        compression_rate = 1. - curr_rank / prev_rank
        print(f"using {curr_rank} out of {prev_rank} POD modes in total: compression rate is {compression_rate*100:.1f} %")
        
        if update_offline_stage:
            self.update_offline_stage()

    def update_offline_stage(self):
        self.X_red = self.snapshot_matrix[2:,2:]
        X_dagger = self.X_red.conjugate().T
        A_tensor = self.numerov_solver.A_tensor
        S_tensor = self.numerov_solver.S_tensor

        # emulator equations: reduction and projection
        self.A_tilde = X_dagger @ A_tensor @ self.X_red
        self.s_tilde = np.tensordot(X_dagger, S_tensor, axes=[1,1]).T

        # prestore tensors for error estimates
        einsum_args = dict(optimize=True, dtype=np.longdouble)
        self.error_term1 = np.einsum("ik,alk,blm,mj->ijab", 
                                      X_dagger, A_tensor.conjugate(), A_tensor, self.X_red, **einsum_args) 
        self.error_term2 = np.einsum("bk,aki,ij->baj", S_tensor.conjugate(), A_tensor, self.X_red, **einsum_args)
        self.error_term3 = S_tensor.conjugate().T @ S_tensor

    def simulate(self, lecList):
        return self.rse_solver.solve(lecList=lecList, method="Numerov_affine", 
                                     asympParam="K", matching=True, reduced_output=True)

    def emulate(self, lecList, estimate_error=False, errors_squared=True,
                calc_error_bounds=False, cond_number_threshold=None, self_test=False):
        ret = []
        errors = []
        error_bounds = []

        for lecs in lecList:
            # reconstruct linear system
            A_tilde = np.tensordot(lecs, self.A_tilde, axis=1)
            s_tilde = np.tensordot(lecs, self.s_tilde, axis=1)

            # solve linear system and emulate
            if cond_number_threshold is not None:
                cond_number = np.linalg.cond(A_tilde)
                if cond_number > cond_number_threshold:
                    print(f"Warning: condition number is large (aff)! {cond_number:.8e}")
            coeffs = np.linalg.solve(A_tilde, s_tilde)
            emulated_chi = self.X_red @ coeffs
            
            # estimate error (proportional to the real error at best)
            error, lower_bound, upper_bound = self.numerov_solver.residuals(emulated_chi, lecs, squared=errors_squared, calc_error_bounds=True)
            # error = self.estimate_error(lecs, coeffs, squared=errors_squared) if estimate_error else None
            # lower_bound = upper_bound = None
            errors.append(error)
            error_bounds.append((lower_bound, upper_bound))

            # check for internal consistency if requested
            if self_test and estimate_error is True:
                # reduced linear system
                tmp = self.numerov_solver.A_const_theta_dense
                A = tmp[0] + tmp[1] @ lecs
                AA_red = snapshot_matrix.transpose() @ A @ snapshot_matrix
                A_banded, s, y1_y2a = self.numerov_solver.get_linear_system(lecs)
                assert np.allclose(AA_red, A_red), "projected matrices don't match"
                assert np.allclose(snapshot_matrix.transpose() @ s, s_red), "projected rhs vectors don't match"

                # check that brute-force and affine error estimate match
                residual_ref = (A @ emulated_chi) - s
                error2a = residual_ref.T @ residual_ref
                error2a_comp = np.zeros(4, dtype=np.longdouble)
                error2a_comp[0] = (A @ emulated_chi).T.conjugate() @ (A @ emulated_chi) 
                error2a_comp[1] = (-s).T.conjugate() @ (A @ emulated_chi)
                error2a_comp[2] = (A @ emulated_chi).T.conjugate() @ (-s)
                error2a_comp[3] = s.T.conjugate() @ s
                assert np.allclose(error, error2a, rtol=0, atol=1e-14), "affine and brute-force error calculation don't match"
                error2n = self.numerov_solver.residuals(emulated_chi, lecs, norm=True)
                assert np.allclose(error2n, error, rtol=0, atol=1e-14), "affine and brute-force error calculation don't match (from AffineNumerovSolver)"

                # for elem in error2a_comp:
                #     print("\terror terms:", elem)
                # print(np.sum(error2a_comp), error2a, np.sum(error2a_comp) - error2a)
                # error2 = (A @ emulated_chi).T.conjugate() @ (A @ emulated_chi)
                # print("error (ref)", error2a)

            ret.append(np.concatenate(([self.numerov_solver.y0], y1_y2, emulated_chi)))

        fomChi = self.simulate(lecList)
        errors_fom = np.array([np.linalg.norm(fomChi[:,i]- ret[i]) for i in range(len(lecList))])
        return ret, np.array(errors), np.array(error_bounds), fomChi, errors_fom

    def estimate_error(self, lecs, coeffs, squared=False):
        lecs_H = lecs.conjugate().T   # we assume below that `lecs` are real
        coeffs_H = coeffs.conjugate().T
        S_const, y1_y2 = self.numerov_solver.get_S_const(lecs)
        S_const_H = S_const.conjugate().T
        einsum_args = dict(optimize=True, dtype=np.longdouble)

        # first term (x^\dagger A^\dagger A x)
        res = np.zeros(3, dtype=np.longdouble)
        res[0] = np.einsum("i,ijab,a,b,j->", coeffs_H, self.error2_1st_3, lecs_H, lecs, coeffs, **einsum_args)
        ## second term (s^dagger A x)
        res[1] = -2.*np.real(np.einsum("baj,a,b,j->", self.error2_2nd_4, lecs, lecs_H, coeffs, **einsum_args))
        ## third term (s^\dagger s)
        res[2] = np.linalg.multi_dot((lecs_H, self.error2_3rd_3, lecs))

        # sum the contributions and return
        total = np.max(((0., np.sum(res))))  # prevent eps^2 < 0 due to round off errors
        return total if squared else np.sqrt(total)

    def greedy_algorithm(self):
        self.greedy_logging = []
        current_est_mean_error = np.inf

        print(f"greedily improving the snapshot basis:")
        for niter in range(self.greedy_max_iter):
            print(f"\titeration #{niter+1} of max {self.greedy_max_iter}:")
            
            # determine candidate snapshots that the greedy algorithm can add
            candidate_snapshot_idxs = list(self.all_snapshot_idxs - self.lec_snapshots_idxs)
            # candidate_snapshots = np.take(a=self.lec_all_samples, indices=candidate_snapshot_idxs, axis=0)

            # determine snapshots at which to compute the errors
            emulate_snapshot_idxs = list(self.all_snapshot_idxs) if self.mode == "linear" else candidate_snapshot_idxs.copy()
            emulate_snapshots = np.take(a=self.lec_all_samples, indices=emulate_snapshot_idxs, axis=0)
            # in the case of "linear", we want to make error plots, so we emulate here all snapshots, 
            # including the ones we've already considered in the greedy iteration.
    
            # emulate candidate snapshots
            romChis, estErrors, estErrBounds, fomChis, realErrors = self.emulate(emulate_snapshots, estimate_error=True, 
                                                                                 errors_squared=False, calc_error_bounds=True)  
            self.greedy_logging.append([self.lec_snapshots_idxs.copy(), estErrors, estErrBounds, realErrors])
            # TODO: the final ROM would not compute the FOM results (just for checking/benchmarking for now)

            # check that the estimated mean error decreases
            mean_est_err = np.mean(estErrors)
            if mean_est_err > current_est_mean_error:
                print(f"\t\tWarning: estimated mean error has increased")
            current_est_mean_error = mean_est_err

            # select the candidate snapshot with maximum (estimated) error
            est_err_candidate_snapshots = np.take(a=estErrors, indices=candidate_snapshot_idxs, axis=0) if self.mode == "linear" else estErrors
            arg_max_err_est = np.argmax(est_err_candidate_snapshots)
            max_err_est = est_err_candidate_snapshots[arg_max_err_est]
            snapshot_idx_max_err_est = candidate_snapshot_idxs[arg_max_err_est]
            # another strategy could be to select more than one candidate snapshot to be added to the basis;
            # this could be done using, e.g., `np.argsort()`

            # check whether the error estimator found indeed the snapshot with maximum (estiamted) error # TODO: final ROM won't do that
            real_err_candidate_snapshots = np.take(a=realErrors, indices=candidate_snapshot_idxs, axis=0) if self.mode == "linear" else realErrors
            arg_max_err_real = np.argmax(real_err_candidate_snapshots)
            max_err_real = real_err_candidate_snapshots[arg_max_err_est]

            if arg_max_err_est != arg_max_err_real:
                print(f"\t\tWarning: estimated max error doesn't match real max error: arg {arg_max_err_est} vs {arg_max_err_real}")
            print(f"\t\testimated max error: {max_err_est:.3e} | real max error: {max_err_real:.3e}")
                        
            # update snapshot matrix by adding new FOM solution
            snapshot_matrix_prev_rank = self.snapshot_matrix.shape[1]
            self.snapshot_matrix = np.hstack((self.snapshot_matrix, np.atleast_2d(fomChis[:, snapshot_idx_max_err_est]).T))
            # TODO: using `newaxis` in the second argument might be more readable: fomChis[arg_max_err_real][:, np.newaxis]

            # update interal records of snapshots
            print(f"\t\tadding snapshot ID {snapshot_idx_max_err_est} to current basis {self.lec_snapshots_idxs}")
            self.lec_snapshots_idxs.add(snapshot_idx_max_err_est)

            # if POD enabled, re-orthogonalize the new snapshot matrix
            if self.pod:
                self.snapshot_matrix = orth(self.snapshot_matrix, rcond=self.rcond)
                snapshot_matrix_curr_rank = self.snapshot_matrix.shape[1]
                if snapshot_matrix_curr_rank != snapshot_matrix_prev_rank + 1:
                    print("\t\tadded snapshot got orthogonalized away")
                self.update_offline_stage()

            # TODO: break condition


class AffineNumerovEmulator:
    def __init__(self, potential, channel, scattExp, grid, free_lecs, range_factor=0.9,
                 num_snapshots_init=3, num_snapshots_max=1000, pod=False, rcond=1e-12, 
                 init_snapshot_lecs=None, greedy=False, greedy_max_iter=5, mode="linear") -> None:
        # internal book keeping
        self.potential = potential
        self.channel = channel
        self.scattExp = scattExp
        self.grid = grid
        self.free_lecs = free_lecs
        self.range_factor = range_factor
        assert num_snapshots_init < num_snapshots_max, "can't have more initial snapshots than maximally allowed"
        self.num_snapshots_init = num_snapshots_init
        self.num_snapshots_max = num_snapshots_max
        self.pod = pod
        self.rcond = rcond
        self.greedy = greedy
        self.greedy_max_iter = greedy_max_iter
        self.mode = mode
        self.greedy_logging = None

        # FOM solver (Numerov)
        params = {"scattExp": self.scattExp, "potential": self.scattExp.potential}
        self.numerov_solver = AffineNumerovSolver(grid.points, g=None, g_s=RseSolver.g_s_affine, 
                                                  y0=0., yp0=0., params=params) 
        
        # training and greedy algorithm if requested
        self.training(init_snapshot_lecs)
        if self.greedy:
            self.greedy_algorithm()
                
    def training(self, snapshot_lecs):
        # if snapshot LECs are user-provided, use them; if not, ran
        if snapshot_lecs is None:
            self.lec_all_samples = self.potential.getLecsSample(self.free_lecs, req_lecs=("V0", "V1"), as_dict=False,
                                                                n=self.num_snapshots_max, mode=self.mode,
                                                                range_factor=self.range_factor, seed=10203)
            # search-type runs could also be done using random samples via sorted(..., key=lambda dictx: dictx["V0"])
        else:
            self.lec_all_samples = snapshot_lecs
            self.num_snapshots_init = len(self.lec_all_samples)
            self.num_snapshots_max = self.num_snapshots_init

        # initial snapshot selection (random)
        rng = np.random.default_rng(seed=12345)
        self.lec_snapshots_idxs = set(rng.choice(range(self.num_snapshots_max), 
                                                 size=self.num_snapshots_init, replace=False))
        
        # building snapshot matrix for chi, not Psi
        init_lecs = np.take(a=self.lec_all_samples, indices=list(self.lec_snapshots_idxs), axis=0)
        self.snapshot_matrix = self.simulate(init_lecs)
        self.all_snapshot_idxs = set(range(len(self.lec_all_samples)))

        # POD snapshot matrix if requested
        if self.pod:
            self.apply_pod(update_offline_stage=False)

        # prestore arrays for efficient offline/online decomposition
        self.update_offline_stage()
    
    def apply_pod(self, update_offline_stage=False):
        prev_rank = self.snapshot_matrix.shape[1]
        self.snapshot_matrix = orth(self.snapshot_matrix, rcond=self.rcond)
        curr_rank = self.snapshot_matrix.shape[1]
        compression_rate = 1. - curr_rank / prev_rank
        print(f"using {curr_rank} POD modes out of {prev_rank} in total: compression rate is {compression_rate*100:.1f} %")
        
        if update_offline_stage:
            self.update_offline_stage()

    def update_offline_stage(self):
        import time 
        X = self.snapshot_matrix[3:,:]
        Xconj = X.conjugate()
        A_const, A_theta = self.numerov_solver.A_const_theta_dense
        einsum_args = dict(optimize=True, dtype=np.longdouble)

        # reduce and project Numerov matrix A
        self.A_const_red = np.einsum("ki,kl,lj->ij", Xconj, A_const, X, **einsum_args)
        self.A_theta_red = np.einsum("mi,mlk,lj", Xconj, A_theta, X, **einsum_args)

        # reduce and project Numerov right-hand side vector
        S_theta = self.numerov_solver.S_theta
        self.S_theta_red = np.einsum("ki,kj->ij", Xconj, S_theta, **einsum_args)
        # The reduced version of `S_const` cannot be prestored but that's ok since it's only a two-dimensional subspace

        # prestore tensors for error estimates
        snapshot_matrix = self.snapshot_matrix[3:,:]
        S_theta_H = S_theta.conjugate().T 

        # ## first term
        X_H = snapshot_matrix.conjugate().T
        A_const_H = A_const.conjugate().T
        self.error2_1st_1 = np.linalg.multi_dot((X_H, A_const_H, A_const, X))
        self.error2_1st_2 = 2.* np.real(np.einsum("ik,kl,lma,mj->ija", X_H, A_const_H, A_theta, X, **einsum_args))  # assumes `lecs` are real
        self.error2_1st_3 = np.einsum("ik,lka,lmb,mj->ijab", X_H, A_theta.conjugate(), A_theta, X, **einsum_args)

        # ## second term   
        self.error2_2nd_1 = np.linalg.multi_dot((A_const, snapshot_matrix))[:2, :]
        self.error2_2nd_2 = np.einsum("kia,ij->ajk", A_theta, snapshot_matrix, **einsum_args)[...,:2]
        self.error2_2nd_3 = np.einsum("bk,ki,ij->bj", S_theta_H, A_const, snapshot_matrix, **einsum_args)
        self.error2_2nd_4 = np.einsum("bk,kia,ij->baj", S_theta_H, A_theta, snapshot_matrix, **einsum_args)

        ## third term
        # the first term cannot be prestored because s_0 = s_0(theta)
        self.error2_3rd_2 = 2. * S_theta[:2,:]  # s_0 is just a length-2 vector
        self.error2_3rd_3 = S_theta_H @ S_theta

    def estimate_error(self, lecs, coeffs, squared=False):
        res = np.zeros(10, dtype=np.longdouble)
        lecs_H = lecs.conjugate().T   # we assume below that `lecs` are real
        coeffs_H = coeffs.conjugate().T
        S_const, y1_y2 = self.numerov_solver.get_S_const(lecs)
        S_const_H = S_const.conjugate().T
        einsum_args = dict(optimize=True, dtype=np.longdouble)

        # first term (x^\dagger A^\dagger A x)
        res[0] = np.linalg.multi_dot((coeffs_H, self.error2_1st_1, coeffs))
        res[1] = np.einsum("i,ija,a,j", coeffs_H, self.error2_1st_2, lecs, coeffs, **einsum_args)
        res[2] = np.einsum("i,ijab,a,b,j->", coeffs_H, self.error2_1st_3, lecs_H, lecs, coeffs, **einsum_args)
        # # could be one line: c^dagger @ (term1+term2+term3) @ c
        
        ## second term (s^dagger A x)
        res[3] = -2.*np.real(np.linalg.multi_dot((S_const_H, self.error2_2nd_1, coeffs)))
        res[4] = -2.*np.real(np.einsum("k,ajk,a,j->", S_const_H, self.error2_2nd_2, lecs, coeffs, **einsum_args))
        res[5] = -2.*np.real(np.einsum("bj,j,b->", self.error2_2nd_3, coeffs, lecs_H, **einsum_args))
        res[6] = -2.*np.real(np.einsum("baj,a,b,j->", self.error2_2nd_4, lecs, lecs_H, coeffs, **einsum_args))

        ## third term (s^\dagger s)
        res[7] = S_const_H @ S_const
        res[8] = np.real(np.linalg.multi_dot((S_const_H, self.error2_3rd_2, lecs)))
        res[9] = np.linalg.multi_dot((lecs_H, self.error2_3rd_3, lecs))

        # sum the contributions and return
        total = np.max(((0., np.sum(res))))  # prevent eps^2 < 0 due to round off errors
        return total if squared else np.sqrt(total)

    def greedy_algorithm(self):
        self.greedy_logging = []
        current_est_mean_error = np.inf

        print(f"greedily improving the snapshot basis:")
        for niter in range(self.greedy_max_iter):
            print(f"\titeration #{niter+1} of max {self.greedy_max_iter}:")
            
            # determine candidate snapshots that the greedy algorithm can add
            candidate_snapshot_idxs = list(self.all_snapshot_idxs - self.lec_snapshots_idxs)
            # candidate_snapshots = np.take(a=self.lec_all_samples, indices=candidate_snapshot_idxs, axis=0)

            # determine snapshots at which to compute the errors
            emulate_snapshot_idxs = list(self.all_snapshot_idxs) if self.mode == "linear" else candidate_snapshot_idxs.copy()
            emulate_snapshots = np.take(a=self.lec_all_samples, indices=emulate_snapshot_idxs, axis=0)
            # in the case of "linear", we want to make error plots, so we emulate here all snapshots, 
            # including the ones we've already considered in the greedy iteration.
    
            # emulate candidate snapshots
            romChis, estErrors, estErrBounds, fomChis, realErrors = self.emulate(emulate_snapshots, estimate_error=True, 
                                                                                 errors_squared=False, calc_error_bounds=True)  
            self.greedy_logging.append([self.lec_snapshots_idxs.copy(), estErrors, estErrBounds, realErrors])
            # TODO: the final ROM would not compute the FOM results (just for checking/benchmarking for now)

            # check that the estimated mean error decreases
            mean_est_err = np.mean(estErrors)
            if mean_est_err > current_est_mean_error:
                print(f"\t\tWarning: estimated mean error has increased")
            current_est_mean_error = mean_est_err

            # select the candidate snapshot with maximum (estimated) error
            est_err_candidate_snapshots = np.take(a=estErrors, indices=candidate_snapshot_idxs, axis=0) if self.mode == "linear" else estErrors
            arg_max_err_est = np.argmax(est_err_candidate_snapshots)
            max_err_est = est_err_candidate_snapshots[arg_max_err_est]
            snapshot_idx_max_err_est = candidate_snapshot_idxs[arg_max_err_est]
            # another strategy could be to select more than one candidate snapshot to be added to the basis;
            # this could be done using, e.g., `np.argsort()`

            # check whether the error estimator found indeed the snapshot with maximum (estiamted) error # TODO: final ROM won't do that
            real_err_candidate_snapshots = np.take(a=realErrors, indices=candidate_snapshot_idxs, axis=0) if self.mode == "linear" else realErrors
            arg_max_err_real = np.argmax(real_err_candidate_snapshots)
            max_err_real = real_err_candidate_snapshots[arg_max_err_est]

            if arg_max_err_est != arg_max_err_real:
                print(f"\t\tWarning: estimated max error doesn't match real max error: arg {arg_max_err_est} vs {arg_max_err_real}")
            print(f"\t\testimated max error: {max_err_est:.3e} | real max error: {max_err_real:.3e}")
                        
            # update snapshot matrix by adding new FOM solution
            snapshot_matrix_prev_rank = self.snapshot_matrix.shape[1]
            self.snapshot_matrix = np.hstack((self.snapshot_matrix, np.atleast_2d(fomChis[:, snapshot_idx_max_err_est]).T))
            # TODO: using `newaxis` in the second argument might be more readable: fomChis[arg_max_err_real][:, np.newaxis]

            # update interal records of snapshots
            print(f"\t\tadding snapshot ID {snapshot_idx_max_err_est} to current basis {self.lec_snapshots_idxs}")
            self.lec_snapshots_idxs.add(snapshot_idx_max_err_est)

            # if POD enabled, re-orthogonalize the new snapshot matrix
            if self.pod:
                self.snapshot_matrix = orth(self.snapshot_matrix, rcond=self.rcond)
                snapshot_matrix_curr_rank = self.snapshot_matrix.shape[1]
                if snapshot_matrix_curr_rank != snapshot_matrix_prev_rank + 1:
                    print("\t\tadded snapshot got orthogonalized away")
                self.update_offline_stage()

            # TODO: break condition

    def emulate(self, lecList, estimate_error=False, errors_squared=True,
                calc_error_bounds=False, cond_number_threshold=None, self_test=False):
        ret = []
        errors = []
        error_bounds = []

        for lecs in lecList:
            # reconstruct linear system
            A_red = self.A_const_red + self.A_theta_red @ lecs 
            S_const, y1_y2 = self.numerov_solver.get_S_const(lecs)
            snapshot_matrix = self.snapshot_matrix[3:,:]  # need to remove (y0, y1, y_2) 
            s_red = snapshot_matrix[:2,:].T.conjugate() @ S_const + self.S_theta_red @ lecs 

            # solve linear system and emulate
            if cond_number_threshold is not None:
                cond_number = np.linalg.cond(A_red)
                if cond_number > cond_number_threshold:
                    print(f"Warning: condition number is large (aff)! {cond_number:.8e}")
            coeffs = np.linalg.solve(A_red, s_red)
            emulated_chi = snapshot_matrix @ coeffs
            
            # estimate error (proportional to the real error at best)
            error, lower_bound, upper_bound = self.numerov_solver.residuals(emulated_chi, lecs, squared=errors_squared, calc_error_bounds=True)
            # error = self.estimate_error(lecs, coeffs, squared=errors_squared) if estimate_error else None
            # lower_bound = upper_bound = None
            errors.append(error)
            error_bounds.append((lower_bound, upper_bound))

            # check for internal consistency if requested
            if self_test and estimate_error is True:
                # reduced linear system
                tmp = self.numerov_solver.A_const_theta_dense
                A = tmp[0] + tmp[1] @ lecs
                AA_red = snapshot_matrix.transpose() @ A @ snapshot_matrix
                A_banded, s, y1_y2a = self.numerov_solver.get_linear_system(lecs)
                assert np.allclose(AA_red, A_red), "projected matrices don't match"
                assert np.allclose(snapshot_matrix.transpose() @ s, s_red), "projected rhs vectors don't match"

                # check that brute-force and affine error estimate match
                residual_ref = (A @ emulated_chi) - s
                error2a = residual_ref.T @ residual_ref
                error2a_comp = np.zeros(4, dtype=np.longdouble)
                error2a_comp[0] = (A @ emulated_chi).T.conjugate() @ (A @ emulated_chi) 
                error2a_comp[1] = (-s).T.conjugate() @ (A @ emulated_chi)
                error2a_comp[2] = (A @ emulated_chi).T.conjugate() @ (-s)
                error2a_comp[3] = s.T.conjugate() @ s
                assert np.allclose(error, error2a, rtol=0, atol=1e-14), "affine and brute-force error calculation don't match"
                error2n = self.numerov_solver.residuals(emulated_chi, lecs, norm=True)
                assert np.allclose(error2n, error, rtol=0, atol=1e-14), "affine and brute-force error calculation don't match (from AffineNumerovSolver)"

                # for elem in error2a_comp:
                #     print("\terror terms:", elem)
                # print(np.sum(error2a_comp), error2a, np.sum(error2a_comp) - error2a)
                # error2 = (A @ emulated_chi).T.conjugate() @ (A @ emulated_chi)
                # print("error (ref)", error2a)

            ret.append(np.concatenate(([self.numerov_solver.y0], y1_y2, emulated_chi)))

        fomChi = self.simulate(lecList)
        errors_fom = np.array([np.linalg.norm(fomChi[:,i]- ret[i]) for i in range(len(lecList))])
        return ret, np.array(errors), np.array(error_bounds), fomChi, errors_fom

    def simulate(self, lecList):
        A_banded, s, y1_y2, sol = self.numerov_solver.solve(lecList)
        return sol.T