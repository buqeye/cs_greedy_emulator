from RseSolver import RseSolver
import numpy as np
from Grid import Grid
from Numerov import numerov,  MatrixNumerov_ab, MatrixNumerov, AllAtOnceNumerov
from scipy.linalg import orth, qr, qr_insert, LinAlgError
from RseSolver import free_solutions_F_G


class AllAtOnceNumerovROM:
    """
    implements the GROM and LSPG ROM based on the matrix Numerov method ("all-at-once Numerov").
    It uses the `AllAtOnceNumerov` implementation, which has the `T` matrix as the last component
    in the solution vector. This emulator is complex-valued for real-valued potentials.

    The implementation mimicks the one in `AffineROM`. It is meant only for benchmarks.
    """
    def __init__(self, scattExp, grid, free_lecs, 
                 num_snapshots_init=3, num_snapshots_max=15, 
                 approach="pod", pod_rcond=1e-12, 
                 init_snapshot_lecs=None,
                 greedy_max_iter=5, 
                 mode="linear",
                 greedy_training_mode="grom",
                 inhomogeneous=True,
                 seed=10203) -> None:
        # internal book keeping
        self.scattExp = scattExp
        self.grid = grid
        self.free_lecs = free_lecs
        assert num_snapshots_init <= num_snapshots_max, "can't have more initial snapshots than maximally allowed"
        self.num_snapshots_init = num_snapshots_init
        self.num_snapshots_max =  num_snapshots_init if approach == "pod" else num_snapshots_max
        self.approach = approach
        self.pod_rcond = pod_rcond
        self.greedy_max_iter = greedy_max_iter
        self.mode = mode
        assert greedy_training_mode in ("grom", "lspg"), f"requested emulator training mode '{mode}' unknown"
        self.greedy_training_mode = greedy_training_mode
        self.seed = seed
        self.greedy_logging = []
        self.coercivity_constant = 1.

        # FOM solver (all-at-once Numerov)
        self.inhomogeneous = inhomogeneous
        rseParams = {"grid": grid, 
                     "scattExp": scattExp, 
                     "potential": scattExp.potential, 
                     "inhomogeneous": self.inhomogeneous
                     }
        from RseSolver import g_s_affine
        self.y0 = 0.
        self.numerov_solver = AllAtOnceNumerov(self.grid.points, 
                                               g=None, g_s=g_s_affine, 
                                               y0=self.y0, params=rseParams)
        self.n_theta = self.numerov_solver.n_theta
        
        # training
        self.training(init_snapshot_lecs)
 
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
            assert self.num_snapshots_init <= self.num_snapshots_max, "can't have more initial snapshots than maximally allowed"

        # initial snapshot selection (random)
        rng = np.random.default_rng(seed=self.seed)
        self.included_snapshots_idxs = set(rng.choice(range(self.num_snapshots_max), 
                                                 size=self.num_snapshots_init, replace=False))
        
        # building snapshot matrix for chi, not Psi
        init_lecs = np.take(a=self.lec_all_samples, indices=list(self.included_snapshots_idxs), axis=0)
        self.snapshot_matrix = self.simulate(init_lecs)

        self.all_snapshot_idxs = set(range(len(self.lec_all_samples)))

        # choose between snapshot approaches 
        # (requires updating the offline stage, see below)
        self.fom_solutions = np.copy(self.snapshot_matrix)
        if self.approach == "pod":
            self.apply_pod(update_offline_stage=True)
        elif self.approach == "greedy":
            self.apply_orthonormalization(update_offline_stage=True)
            self.greedy_algorithm()
        elif self.approach == "orth":
            self.apply_orthonormalization(update_offline_stage=True)
        elif self.approach is None: 
            self.update_offline_stage()
            print(f"Snapshot matrix not orthonormalized. Consider using `approach='orth'`")
        else:
            raise NotADirectoryError(f"Approach '{self.approach}' is unknown.")

    def apply_orthonormalization(self, update_offline_stage=True):
        q, r = qr(self.snapshot_matrix, mode='economic')
        self.snapshot_matrix = q
        self.snapshot_matrix_r = r

        if update_offline_stage:
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

    def apply_pod(self, update_offline_stage=True):
        prev_rank = self.snapshot_matrix.shape[1]
        # calling `sp.linalg.orth()`` would be enough here, but we might want to study different SVD truncations in the future
        Ur, S, Vr = self.truncated_svd(self.snapshot_matrix, rcond=self.pod_rcond)
        self.snapshot_matrix = Ur

        curr_rank = self.snapshot_matrix.shape[1]
        compression_rate = 1. - curr_rank / prev_rank
        print(f"using {curr_rank} out of {prev_rank} POD modes in total: compression rate is {compression_rate*100:.1f} %")
        
        if update_offline_stage:
            self.update_offline_stage()

    def update_offline_stage(self, verbose=True):
        # Note: when adding snapshots to the emulator basis, one does not need to recompute all tensors again,
        # rather one can update them for computational efficiency. Since this update only occurs in the offline 
        # stage of the emulator, we keep it in this proof-of-principle work simple and compute the tensors from scratch

        X_red = self.snapshot_matrix[1:,:]
        X_dagger = X_red.conjugate().T
        A_tensor = self.numerov_solver.A_tensor
        S_tensor = self.numerov_solver.S_tensor
        S_tensor_conj = S_tensor.conjugate()

        # GROM emulator equations: reduction and projection
        A_tensor_x_X_red = A_tensor @ X_red
        self.A_tilde_grom = X_dagger @ A_tensor_x_X_red
        self.s_tilde_grom = np.tensordot(X_dagger, S_tensor, axes=[1,1]).T

        # prestore tensors for error estimates
        einsum_args = dict(optimize="greedy", dtype=np.complex128)
        self.error_term1 = np.einsum("ki,alk,blm,mj->ijab", 
                                      X_red.conjugate(), A_tensor.conjugate(), A_tensor, X_red, **einsum_args) 
        self.error_term2 = np.einsum("bk,aki,ij->baj", S_tensor_conj, A_tensor, X_red, **einsum_args)
        self.error_term3 = S_tensor_conj @ S_tensor.T

        # construct Y matrix for alternative error estimation and LSPG-ROM
        tmp = np.column_stack(np.squeeze(np.split(A_tensor_x_X_red, self.n_theta, axis=0)))
        tmp = np.column_stack((tmp, S_tensor.T))
        self.Y_tensor = orth(tmp, rcond=None)
        prev_rank = min(tmp.shape)
        shape_Y_tensor = self.Y_tensor.shape
        curr_rank = min(shape_Y_tensor)
        compression_rate = 1. - curr_rank / prev_rank
        if verbose:
            print(f"POD[ Y ]: compression rate is {compression_rate*100:.1f} %; dim: {shape_Y_tensor}")
        assert shape_Y_tensor[1] < shape_Y_tensor[0] //4, "semi-reduced space for Y is large!"

        # prestore tensors for alternative error estimation  
        Y_conj = self.Y_tensor.conjugate()
        self.S_tensor_projY = np.einsum("iw,ai->aw", Y_conj, S_tensor, **einsum_args)
        self.A_tensor_projY = np.einsum("iw,aij,ju->awu", Y_conj, A_tensor, X_red, **einsum_args)

        # LSPG emulator equations
        Y_tensor_dagger = Y_conj.T
        self.A_tilde_lspg = Y_tensor_dagger @ A_tensor_x_X_red
        self.s_tilde_lspg = np.tensordot(Y_tensor_dagger, S_tensor, axes=[1,1]).T

    def get_scattering_matrix(self, sols_row, which="T"):
        if which == "T":
            return sols_row
        elif which == "S":
            return 1 + 2j * sols_row      
        elif which == "K":
            return sols_row / (1 + 1j * sols_row)     
        else:
            raise NotImplementedError(f"Scattering matrix '{which}' unknown.")

    def simulate(self, lecList, which="all"):
        full_sols = self.numerov_solver.solve(lecList)
        return full_sols if which == "all" else self.get_scattering_matrix(full_sols[-1,:], which=which)

    def emulate(self, lecList, which="all",
                estimate_norm_residual=False, mode="grom",
                calc_error_bounds=False, calibrate_norm_residual=False,
                cond_number_threshold=None, self_test=True, 
                lspg_rcond=None, lspg_solver="svd"):
        coeffs_all = []
        assert mode in ("grom", "lspg"), f"requested mode '{mode}' unknown"
        A_tilde_tensor = self.A_tilde_grom if mode == "grom" else self.A_tilde_lspg
        s_tilde_tensor = self.s_tilde_grom if mode == "grom" else self.s_tilde_lspg
        for lecs in lecList:
            # reconstruct linear system    
            A_tilde = np.tensordot(lecs, A_tilde_tensor, axes=1)
            s_tilde = np.tensordot(lecs, s_tilde_tensor, axes=1)

            # solve linear system and emulate
            if cond_number_threshold is not None:
                cond_number = np.linalg.cond(A_tilde)
                if cond_number > cond_number_threshold:
                    print(f"Warning: condition number is above threshold (aff)! {cond_number:.8e}")
            if mode == "grom":
                coeffs_curr = np.linalg.solve(A_tilde, s_tilde)
            else:
                if lspg_solver == "svd":
                    coeffs_curr, residuals, rank, svals = np.linalg.lstsq(A_tilde, s_tilde, rcond=lspg_rcond)
                else:
                    Q, R = np.linalg.qr(A_tilde)
                    coeffs_curr = np.linalg.solve(R, Q.conjugate().T @ s_tilde)  # Solve Rx = Q^dagger s_tilde
            coeffs_all.append(coeffs_curr)
            # print("sum of the emulator basis coeffs", np.sum(coeffs_curr))
        coeffs_all = np.array(coeffs_all).T
        if which != "all":
            emulated_Ts = self.snapshot_matrix[-1,:] @ coeffs_all
            return self.get_scattering_matrix(emulated_Ts, which=which)
        else:
            emulated_sols = self.snapshot_matrix @ coeffs_all

        if estimate_norm_residual:
            num_norm_residuals = len(lecList)
            norm_residuals = np.empty(num_norm_residuals)
            error_bounds = []
            for ilecs, lecs in enumerate(lecList):
                norm_residuals[ilecs] = self.reconstruct_norm_residual(lecs, coeffs_all[:,ilecs])

            if self_test or calc_error_bounds:
                norm_residuals_FOM = np.empty(num_norm_residuals)
                for ilecs, lecs in enumerate(lecList):
                    norm_residuals_FOM[ilecs], bounds = self.numerov_solver.residuals(emulated_sols[1:,ilecs], lecs, squared=False,
                                                                                      calc_error_bounds=calc_error_bounds)
                    error_bounds.append(bounds)  # may be None
                error_bounds = np.array(error_bounds)

                if self_test:
                    max_diff = np.max(np.abs(norm_residuals - norm_residuals_FOM))
                    # print("max_diff", max_diff)
                    assert np.allclose(max_diff, 0, atol=1e-10, rtol=0.1), f"something's wrong with the reconstructed residual; max diff: {max_diff}"

            if calibrate_norm_residual:
                norm_residuals *= self.coercivity_constant

            return emulated_sols, norm_residuals, error_bounds    
        else:
            return emulated_sols

    def reconstruct_norm_residual(self, lecs, coeffs):
        lecs_H = lecs.conjugate().T
        coeffs_H = coeffs.conjugate().T

        einsum_args = dict(optimize="greedy", dtype=np.complex128)

        # import time
        # start_time = time.time()
        
        # first term (x^\dagger A^\dagger A x)
        res = np.empty(3, dtype=np.complex128)
        res[0] = np.einsum("i,ijab,a,b,j->", coeffs_H, self.error_term1, lecs_H, lecs, coeffs, **einsum_args)
        ## second term (s^dagger A x)
        res[1] = -2.*np.real(np.einsum("baj,a,b,j->", self.error_term2, lecs, lecs_H, coeffs, **einsum_args))
        ## third term (s^\dagger s)
        res[2] = lecs_H @ self.error_term3 @ lecs  # dimensions probably too small for multidot to be more efficient

        # sum the contributions and return
        total = np.sqrt(np.max(((0., np.sum(res)))))  # prevent eps^2 < 0 due to round-off errors and ill-conditioning

        # end_time = time.time()
        # elapsed_time = end_time - start_time
        # print(f"Elapsed time (scalar): {elapsed_time*1e-6:.5e} micro seconds")

        # LSPG
        # start_time = time.time()

        component_S = np.tensordot(lecs, self.S_tensor_projY, axes=1)
        component_A = lecs @ (self.A_tensor_projY @ coeffs)
        total2 = np.linalg.norm(component_S - component_A)
        
        # end_time = time.time()
        # elapsed_time = end_time - start_time
        # print(f"Elapsed time (vector norm): {elapsed_time1e-6:.5e} seconds")
        # print(f"reconstructed norm residuals diff: {total - total2:.2e} | {total:.2e} {total2:.2e}")
        assert np.allclose(total, total2, atol=1e-8, rtol=0.), f"total and total2 inconsistent; diff: {total-total2}"
        
        return total2

    def greedy_algorithm(self, error_calibration_mode=False, mode=None,
                         calibrate_error_estimation=True, atol=1e-12,
                         lowest_mean_norm_residuals=1e-11,
                         logging=True, verbose=True):
        if mode is None:
            mode = self.greedy_training_mode
        if error_calibration_mode:
            calibrate_error_estimation = True
            max_iter = 1
        else: 
            max_iter = min(self.greedy_max_iter, max(0, self.num_snapshots_max-len(self.included_snapshots_idxs)))
        if max_iter > 0:
            print("snapshot idx already included in basis:", self.included_snapshots_idxs)
            print(f"now greedily improving the snapshot basis:")
        else:
            print("Nothing to be done. Maxed out available snapshots and/or number of iterations")

        current_mean_norm_residuals = np.inf
        for niter in range(max_iter):
            print(f"\titeration #{niter+1} of max {max_iter}:")
            
            # determine candidate snapshots that the greedy algorithm can add
            candidate_snapshot_idxs = list(self.all_snapshot_idxs - self.included_snapshots_idxs)
            if verbose: print("\tavailable candidate snapshot idx to be added:", candidate_snapshot_idxs)
            candidate_snapshots = np.take(a=self.lec_all_samples, indices=candidate_snapshot_idxs, axis=0)

            # determine snapshots at which to compute the errors
            emulate_snapshot_idxs = list(self.all_snapshot_idxs) if self.mode == "linear" else candidate_snapshot_idxs.copy()
            emulate_snapshots = np.take(a=self.lec_all_samples, indices=emulate_snapshot_idxs, axis=0)
            if verbose: print("\temulate snapshots:", emulate_snapshot_idxs)
            # in the case of the "linear" mode, we want to make error plots, so we emulate here all snapshots, 
            # including the ones we've already considered in the greedy iteration.
    
            # emulate candidate snapshots           
            emulated_sols, norm_residuals, error_bounds = self.emulate(emulate_snapshots, mode=mode,
                                                                       estimate_norm_residual=True, 
                                                                       calibrate_norm_residual=False,
                                                                       calc_error_bounds=logging, 
                                                                       cond_number_threshold=None, self_test=logging)

            if logging:
                # in practice, ROM will not compute the FOM results (just for checking/benchmarking)
                fom_sols = self.simulate(emulate_snapshots)
                norm_error_exact = np.linalg.norm(emulated_sols - fom_sols, axis=0)
                self.greedy_logging.append([self.included_snapshots_idxs.copy(), 
                                            fom_sols, emulated_sols, 
                                            norm_residuals, norm_error_exact, error_bounds])
                # both `norm_error_exact` and `norm_residuals` should be minimal at the snapshot locations 
                # already included in the basis; i.e., `self.included_snapshots_idxs`

            # check that the estimated mean error decreases
            mean_norm_residuals = np.mean(norm_residuals)
            if mean_norm_residuals > current_mean_norm_residuals:
                print(f"\t\tWarning: estimated mean error has increased. Terminating greedy iteration.")
                break
            if mean_norm_residuals < lowest_mean_norm_residuals:
                print(f"\t\tWarning: estimated mean error reached set bound < {lowest_mean_norm_residuals:.2e}. Terminating greedy iteration.")
                break
            current_mean_norm_residuals = mean_norm_residuals

            # select the candidate snapshot with maximum (estimated) error
            est_err_candidate_snapshots = np.take(a=norm_residuals, 
                                                  indices=candidate_snapshot_idxs, axis=0) if self.mode == "linear" else norm_residuals
            arg_max_err_est = np.argmax(est_err_candidate_snapshots)
            max_err_est = est_err_candidate_snapshots[arg_max_err_est]
            snapshot_idx_max_err_est = candidate_snapshot_idxs[arg_max_err_est]

            if logging:
                # check whether the error estimator found indeed the snapshot with maximum (estimated) error
                real_err_candidate_snapshots = np.take(a=norm_error_exact, 
                                                       indices=candidate_snapshot_idxs, axis=0) if self.mode == "linear" else norm_error_exact
                arg_max_err_real = np.argmax(real_err_candidate_snapshots)
                max_err_real = real_err_candidate_snapshots[arg_max_err_real]
                snapshot_idx_max_err_real = candidate_snapshot_idxs[arg_max_err_real]

                if arg_max_err_est != arg_max_err_real:
                    print(f"\t\tWarning: estimated max error doesn't match real max error: arg {arg_max_err_est} vs {arg_max_err_real}")
                print(f"\t\testimated max error: {max_err_est:.3e} | real max error: {max_err_real:.3e}")
                        
            # check whether accuracy goal is achieved
            scaled_max_err_est = self.coercivity_constant * max_err_est if calibrate_error_estimation else max_err_est
            if scaled_max_err_est < atol:
                print(f"accuracy goal 'atol = {atol}' achieved. Terminating greedy iteration.")
                break

            # perform FOM calculation at the location of max estimated error
            to_be_added_fom_sol = self.simulate([candidate_snapshots[arg_max_err_est]])

            # calibrate error estimator
            if calibrate_error_estimation:
                exact_error = np.linalg.norm(np.squeeze(to_be_added_fom_sol) - emulated_sols[:, snapshot_idx_max_err_est])
                self.coercivity_constant = exact_error / max_err_est
                assert self.coercivity_constant > 0., "coercivity constant is not positive"
                print(f"\t\tcoercivity constant: {self.coercivity_constant:.3e}")
                if logging:
                    self.greedy_logging[-1].append(self.coercivity_constant)

            if logging and (arg_max_err_est == arg_max_err_real):
                assert np.allclose(fom_sols[:, snapshot_idx_max_err_real], 
                                   np.squeeze(to_be_added_fom_sol), atol=1e-14, rtol=0.), "adding the wrong FOM solution to basis?"
                assert np.allclose(exact_error, max_err_real, atol=1e-12, rtol=0.), "calibrating the coercivity constant incorrectly?"
                
            # update snapshot matrix by adding new FOM solution and interal records
            if not error_calibration_mode and niter < max_iter-1:
                print(f"\t\tadding snapshot ID {snapshot_idx_max_err_est} to current basis {self.included_snapshots_idxs}")
                self.add_fom_solution_to_basis(to_be_added_fom_sol)
                self.included_snapshots_idxs.add(snapshot_idx_max_err_est)

    def add_fom_solution_to_basis(self, fom_sol):
        self.fom_solutions = np.column_stack((self.fom_solutions, fom_sol))
        if self.approach == "pod":
            self.snapshot_matrix = np.copy(self.fom_solutions)
            self.apply_pod(update_offline_stage=False)
            # one could do here an incremental SVD; since we do not explore
            # updating the snapshot matrix after POD, we perform here a
            # full truncated SVD again only for completeness
        elif self.approach in ("orth", "greedy"):
            try:
                self.snapshot_matrix, self.snapshot_matrix_r = qr_insert(Q=self.snapshot_matrix, 
                                                                         R=self.snapshot_matrix_r, 
                                                                         u=fom_sol, k=-1, 
                                                                         which='col', rcond=None)
                # `qr_insert` will raise a `LinAlgError` if one of the columns of u lies in the span of Q,
                # which is measured using `rcond`; if that is the case, we perform a QR decomposition
                # on the updated snapshot matrix, which results in an orthonormal basis  with the requested size
            except LinAlgError:
                print("Warning: need to perform full QR decomposition. Added snapshot is orthogonalzied away.")
                self.snapshot_matrix = np.copy(self.fom_solutions)
                self.apply_orthonormalization(update_offline_stage=False)
        elif self.approach is None: 
            self.snapshot_matrix = np.copy(self.fom_solutions)
        else:
            raise NotADirectoryError(f"Approach '{self.approach}' is unknown.")

        self.update_offline_stage()


class MatrixNumerovROM_ab:
    """
    implements the GROM and LSPG ROM based on the matrix Numerov method ("all-at-once Numerov").
    It uses the `AllAtOnceNumerov_ab` implementation, which has (a,b) explicitly 
    in the solution vector, not (a,b) or some scattering matrix.
    This emulator is real-valued for real-valued potentials.
    """
    def __init__(self, scattExp, grid, free_lecs, 
                 num_snapshots_init=3, num_snapshots_max=15, 
                 approach="pod", pod_rcond=1e-12, 
                 init_snapshot_lecs=None,
                 greedy_max_iter=5, 
                 mode="linear",
                 greedy_training_mode="grom",
                 inhomogeneous=True,
                 seed=10203) -> None:
        # internal book keeping
        self.scattExp = scattExp
        self.grid = grid
        self.free_lecs = free_lecs
        assert num_snapshots_init <= num_snapshots_max, "can't have more initial snapshots than maximally allowed"
        self.num_snapshots_init = num_snapshots_init
        self.num_snapshots_max =  num_snapshots_init if approach == "pod" else num_snapshots_max
        self.approach = approach
        self.pod_rcond = pod_rcond
        self.greedy_max_iter = greedy_max_iter
        self.mode = mode
        assert greedy_training_mode in ("grom", "lspg"), f"requested emulator training mode '{mode}' unknown"
        self.greedy_training_mode = greedy_training_mode
        self.seed = seed
        self.greedy_logging = []
        self.coercivity_constant = 1.

        # FOM solver (all-at-once Numerov)
        self.inhomogeneous = inhomogeneous
        rseParams = {"grid": grid, 
                     "scattExp": scattExp, 
                     "potential": scattExp.potential, 
                     "inhomogeneous": self.inhomogeneous
                     }
        from RseSolver import g_s_affine
        self.numerov_solver = MatrixNumerov_ab(self.grid.points, 
                                               g=None, g_s=g_s_affine, 
                                               y0=0., 
                                               y1=(0. if self.inhomogeneous else 1.), 
                                               params=rseParams)
        self.n_theta = self.numerov_solver.n_theta
        
        # training
        self.training(init_snapshot_lecs)
 
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
            assert self.num_snapshots_init <= self.num_snapshots_max, "can't have more initial snapshots than maximally allowed"

        # initial snapshot selection (random)
        rng = np.random.default_rng(seed=self.seed)
        self.included_snapshots_idxs = set(rng.choice(range(self.num_snapshots_max), 
                                                 size=self.num_snapshots_init, replace=False))
        
        # building snapshot matrix for chi, not Psi
        init_lecs = np.take(a=self.lec_all_samples, indices=list(self.included_snapshots_idxs), axis=0)
        self.snapshot_matrix = self.simulate(init_lecs)

        self.all_snapshot_idxs = set(range(len(self.lec_all_samples)))

        # choose between snapshot approaches 
        # (requires updating the offline stage, see below)
        self.fom_solutions = np.copy(self.snapshot_matrix)
        if self.approach == "pod":
            self.apply_pod(update_offline_stage=True)
        elif self.approach == "greedy":
            self.apply_orthonormalization(update_offline_stage=True)
            self.greedy_algorithm()
        elif self.approach == "orth":
            self.apply_orthonormalization(update_offline_stage=True)
        elif self.approach is None: 
            self.update_offline_stage()
            print(f"Snapshot matrix not orthonormalized. Consider using `approach='orth'`")
        else:
            raise NotADirectoryError(f"Approach '{self.approach}' is unknown.")

    def apply_orthonormalization(self, update_offline_stage=True):
        q, r = qr(self.snapshot_matrix, mode='economic')
        self.snapshot_matrix = q
        self.snapshot_matrix_r = r

        if update_offline_stage:
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

    def apply_pod(self, update_offline_stage=True):
        prev_rank = self.snapshot_matrix.shape[1]
        # calling `sp.linalg.orth()`` would be enough here, but we might want to study different SVD truncations in the future
        Ur, S, Vr = self.truncated_svd(self.snapshot_matrix, rcond=self.pod_rcond)
        self.snapshot_matrix = Ur

        curr_rank = self.snapshot_matrix.shape[1]
        compression_rate = 1. - curr_rank / prev_rank
        print(f"using {curr_rank} out of {prev_rank} POD modes in total: compression rate is {compression_rate*100:.1f} %")
        
        if update_offline_stage:
            self.update_offline_stage()

    def update_offline_stage(self, verbose=True):
        # Note: when adding snapshots to the emulator basis, one does not need to recompute all tensors again,
        # rather one can update them for computational efficiency. Since this update only occurs in the offline 
        # stage of the emulator, we keep it in this proof-of-principle work simple and compute the tensors from scratch

        X_red = self.snapshot_matrix[2:,:]
        X_dagger = X_red.conjugate().T
        A_tensor = self.numerov_solver.A_tensor
        S_tensor = self.numerov_solver.S_tensor
        S_tensor_conj = S_tensor.conjugate()

        # GROM emulator equations: reduction and projection
        A_tensor_x_X_red = A_tensor @ X_red
        self.A_tilde_grom = X_dagger @ A_tensor_x_X_red
        self.s_tilde_grom = np.tensordot(X_dagger, S_tensor, axes=[1,1]).T

        # prestore tensors for error estimates
        einsum_args = dict(optimize="greedy", dtype=np.longdouble)
        self.error_term1 = np.einsum("ki,alk,blm,mj->ijab", 
                                      X_red.conjugate(), A_tensor.conjugate(), A_tensor, X_red, **einsum_args) 
        self.error_term2 = np.einsum("bk,aki,ij->baj", S_tensor_conj, A_tensor, X_red, **einsum_args)
        self.error_term3 = S_tensor_conj @ S_tensor.T

        # construct Y matrix for alternative error estimation and LSPG-ROM
        tmp = np.column_stack(np.squeeze(np.split(A_tensor_x_X_red, self.n_theta, axis=0)))
        tmp = np.column_stack((tmp, S_tensor.T))
        self.Y_tensor = orth(tmp, rcond=None)
        prev_rank = min(tmp.shape)
        shape_Y_tensor = self.Y_tensor.shape
        curr_rank = min(shape_Y_tensor)
        compression_rate = 1. - curr_rank / prev_rank
        if verbose:
            print(f"POD[ Y ]: compression rate is {compression_rate*100:.1f} %; dim: {shape_Y_tensor}")
        assert shape_Y_tensor[1] < shape_Y_tensor[0] //4, "semi-reduced space for Y is large!"

        # prestore tensors for alternative error estimation  
        Y_conj = self.Y_tensor.conjugate()
        self.S_tensor_projY = np.einsum("iw,ai->aw", Y_conj, S_tensor, **einsum_args)
        self.A_tensor_projY = np.einsum("iw,aij,ju->awu", Y_conj, A_tensor, X_red, **einsum_args)

        # LSPG emulator equations
        Y_tensor_dagger = Y_conj.T
        self.A_tilde_lspg = Y_tensor_dagger @ A_tensor_x_X_red
        self.s_tilde_lspg = np.tensordot(Y_tensor_dagger, S_tensor, axes=[1,1]).T

    def get_scattering_matrix(self, sols_rows, which="T"):
        ab_arr = np.copy(sols_rows)
        if self.inhomogeneous:
            ab_arr[0, :] += 1
        if which == "S":
            num = ab_arr[0, :] + 1j*ab_arr[1, :]
            denom = ab_arr[0, :] - 1j*ab_arr[1, :]
            return num / denom
        elif which == "K":
            return ab_arr[1, :] / ab_arr[0, :]  # b / a 
        else:
            raise NotImplementedError(f"Scattering matrix '{which}' unknown.")

    def simulate(self, lecList, which="all"):
        full_sols = self.numerov_solver.solve(lecList)
        return full_sols if which == "all" else self.get_scattering_matrix(full_sols[-2:,:], which=which)

    def emulate(self, lecList, which="all", 
                estimate_norm_residual=False, mode="grom",
                calc_error_bounds=False, calibrate_norm_residual=False,
                cond_number_threshold=None, self_test=True, 
                lspg_rcond=None, lspg_solver="svd"):
        coeffs_all = []
        assert mode in ("grom", "lspg"), f"requested mode '{mode}' unknown"
        A_tilde_tensor = self.A_tilde_grom if mode == "grom" else self.A_tilde_lspg
        s_tilde_tensor = self.s_tilde_grom if mode == "grom" else self.s_tilde_lspg
        for lecs in lecList:
            # reconstruct linear system    
            A_tilde = np.tensordot(lecs, A_tilde_tensor, axes=1)
            s_tilde = np.tensordot(lecs, s_tilde_tensor, axes=1)

            # solve linear system and emulate
            if cond_number_threshold is not None:
                cond_number = np.linalg.cond(A_tilde)
                if cond_number > cond_number_threshold:
                    print(f"Warning: condition number is above threshold (aff)! {cond_number:.8e}")
            if mode == "grom":
                coeffs_curr = np.linalg.solve(A_tilde, s_tilde)
            else:
                if lspg_solver == "svd":
                    coeffs_curr, residuals, rank, svals = np.linalg.lstsq(A_tilde, s_tilde, rcond=lspg_rcond)
                else:
                    Q, R = np.linalg.qr(A_tilde)
                    coeffs_curr = np.linalg.solve(R, Q.conjugate().T @ s_tilde)  # Solve Rx = Q^dagger s_tilde
            coeffs_all.append(coeffs_curr)
            # print("sum of the emulator basis coeffs", np.sum(coeffs_curr))
        coeffs_all = np.array(coeffs_all).T
        if which != "all":
            emulated_Ts = self.snapshot_matrix[-2:,:] @ coeffs_all
            return self.get_scattering_matrix(emulated_Ts, which=which)
        else:
            emulated_sols = self.snapshot_matrix @ coeffs_all
            
        if estimate_norm_residual:
            num_norm_residuals = len(lecList)
            norm_residuals = np.empty(num_norm_residuals)
            error_bounds = []
            for ilecs, lecs in enumerate(lecList):
                norm_residuals[ilecs] = self.reconstruct_norm_residual(lecs, coeffs_all[:,ilecs])

            if self_test or calc_error_bounds:
                norm_residuals_FOM = np.empty(num_norm_residuals)
                for ilecs, lecs in enumerate(lecList):
                    norm_residuals_FOM[ilecs], bounds = self.numerov_solver.residuals(emulated_sols[2:,ilecs], lecs, squared=False,
                                                                                      calc_error_bounds=calc_error_bounds)
                    error_bounds.append(bounds)  # may be None
                error_bounds = np.array(error_bounds)

                if self_test:
                    max_diff = np.max(np.abs(norm_residuals - norm_residuals_FOM))
                    # print("max_diff", max_diff)
                    assert np.allclose(max_diff, 0, atol=1e-10, rtol=0.1), f"something's wrong with the reconstructed residual; max diff: {max_diff}"

            if calibrate_norm_residual:
                norm_residuals *= self.coercivity_constant

            return emulated_sols, norm_residuals, error_bounds    
        else:
            return emulated_sols       

    def reconstruct_norm_residual(self, lecs, coeffs):
        lecs_H = lecs.conjugate().T
        coeffs_H = coeffs.conjugate().T

        einsum_args = dict(optimize="greedy", dtype=np.longdouble)

        # import time
        # start_time = time.time()
        
        # first term (x^\dagger A^\dagger A x)
        res = np.empty(3, dtype=np.longdouble)
        res[0] = np.einsum("i,ijab,a,b,j->", coeffs_H, self.error_term1, lecs_H, lecs, coeffs, **einsum_args)
        ## second term (s^dagger A x)
        res[1] = -2.*np.real(np.einsum("baj,a,b,j->", self.error_term2, lecs, lecs_H, coeffs, **einsum_args))
        ## third term (s^\dagger s)
        res[2] = lecs_H @ self.error_term3 @ lecs  # dimensions probably too small for multidot to be more efficient

        # sum the contributions and return
        total = np.sqrt(np.max(((0., np.sum(res)))))  # prevent eps^2 < 0 due to round-off errors and ill-conditioning

        # end_time = time.time()
        # elapsed_time = end_time - start_time
        # print(f"Elapsed time (scalar): {elapsed_time*1e-6:.5e} micro seconds")

        # LSPG
        # start_time = time.time()

        component_S = np.tensordot(lecs, self.S_tensor_projY, axes=1)
        component_A = lecs @ (self.A_tensor_projY @ coeffs)
        total2 = np.linalg.norm(component_S - component_A)
        
        # end_time = time.time()
        # elapsed_time = end_time - start_time
        # print(f"Elapsed time (vector norm): {elapsed_time1e-6:.5e} seconds")
        # print(f"reconstructed norm residuals diff: {total - total2:.2e} | {total:.2e} {total2:.2e}")
        assert np.allclose(total, total2, atol=1e-8, rtol=0.), f"total and total2 inconsistent; diff: {total-total2}"
        
        return total2

    def greedy_algorithm(self, error_calibration_mode=False, mode=None,
                         calibrate_error_estimation=True, atol=1e-12,
                         lowest_mean_norm_residuals=1e-11,
                         logging=True, verbose=True):
        if mode is None:
            mode = self.greedy_training_mode
        if error_calibration_mode:
            calibrate_error_estimation = True
            max_iter = 1
        else: 
            max_iter = min(self.greedy_max_iter, max(0, self.num_snapshots_max-len(self.included_snapshots_idxs)))
        if max_iter > 0:
            print("snapshot idx already included in basis:", self.included_snapshots_idxs)
            print(f"now greedily improving the snapshot basis:")
        else:
            print("Nothing to be done. Maxed out available snapshots and/or number of iterations")

        current_mean_norm_residuals = np.inf
        for niter in range(max_iter):
            print(f"\titeration #{niter+1} of max {max_iter}:")
            
            # determine candidate snapshots that the greedy algorithm can add
            candidate_snapshot_idxs = list(self.all_snapshot_idxs - self.included_snapshots_idxs)
            if verbose: print("\tavailable candidate snapshot idx to be added:", candidate_snapshot_idxs)
            candidate_snapshots = np.take(a=self.lec_all_samples, indices=candidate_snapshot_idxs, axis=0)

            # determine snapshots at which to compute the errors
            emulate_snapshot_idxs = list(self.all_snapshot_idxs) if self.mode == "linear" else candidate_snapshot_idxs.copy()
            emulate_snapshots = np.take(a=self.lec_all_samples, indices=emulate_snapshot_idxs, axis=0)
            if verbose: print("\temulate snapshots:", emulate_snapshot_idxs)
            # in the case of the "linear" mode, we want to make error plots, so we emulate here all snapshots, 
            # including the ones we've already considered in the greedy iteration.
    
            # emulate candidate snapshots           
            emulated_sols, norm_residuals, error_bounds = self.emulate(emulate_snapshots, mode=mode,
                                                                       estimate_norm_residual=True, 
                                                                       calibrate_norm_residual=False,
                                                                       calc_error_bounds=logging, 
                                                                       cond_number_threshold=None, self_test=logging)

            if logging:
                # in practice, ROM will not compute the FOM results (just for checking/benchmarking)
                fom_sols = self.simulate(emulate_snapshots)
                norm_error_exact = np.linalg.norm(emulated_sols - fom_sols, axis=0)
                self.greedy_logging.append([self.included_snapshots_idxs.copy(), 
                                            fom_sols, emulated_sols, 
                                            norm_residuals, norm_error_exact, error_bounds])
                # both `norm_error_exact` and `norm_residuals` should be minimal at the snapshot locations 
                # already included in the basis; i.e., `self.included_snapshots_idxs`

            # check that the estimated mean error decreases
            mean_norm_residuals = np.mean(norm_residuals)
            if mean_norm_residuals > current_mean_norm_residuals:
                print(f"\t\tWarning: estimated mean error has increased. Terminating greedy iteration.")
                break
            if mean_norm_residuals < lowest_mean_norm_residuals:
                print(f"\t\tWarning: estimated mean error reached set bound < {lowest_mean_norm_residuals:.2e}. Terminating greedy iteration.")
                break
            current_mean_norm_residuals = mean_norm_residuals

            # select the candidate snapshot with maximum (estimated) error
            est_err_candidate_snapshots = np.take(a=norm_residuals, 
                                                  indices=candidate_snapshot_idxs, axis=0) if self.mode == "linear" else norm_residuals
            arg_max_err_est = np.argmax(est_err_candidate_snapshots)
            max_err_est = est_err_candidate_snapshots[arg_max_err_est]
            snapshot_idx_max_err_est = candidate_snapshot_idxs[arg_max_err_est]

            if logging:
                # check whether the error estimator found indeed the snapshot with maximum (estimated) error
                real_err_candidate_snapshots = np.take(a=norm_error_exact, 
                                                       indices=candidate_snapshot_idxs, axis=0) if self.mode == "linear" else norm_error_exact
                arg_max_err_real = np.argmax(real_err_candidate_snapshots)
                max_err_real = real_err_candidate_snapshots[arg_max_err_real]
                snapshot_idx_max_err_real = candidate_snapshot_idxs[arg_max_err_real]

                if arg_max_err_est != arg_max_err_real:
                    print(f"\t\tWarning: estimated max error doesn't match real max error: arg {arg_max_err_est} vs {arg_max_err_real}")
                print(f"\t\testimated max error: {max_err_est:.3e} | real max error: {max_err_real:.3e}")
                        
            # check whether accuracy goal is achieved
            scaled_max_err_est = self.coercivity_constant * max_err_est if calibrate_error_estimation else max_err_est
            if scaled_max_err_est < atol and niter > 0:  # need to have completed at least one iteration for calibration at this point
                print(f"accuracy goal 'atol = {atol}' achieved. Terminating greedy iteration.")
                break

            # perform FOM calculation at the location of max estimated error
            to_be_added_fom_sol = self.simulate([candidate_snapshots[arg_max_err_est]])

            # calibrate error estimator
            if calibrate_error_estimation:
                exact_error = np.linalg.norm(np.squeeze(to_be_added_fom_sol) - emulated_sols[:, snapshot_idx_max_err_est])
                self.coercivity_constant = exact_error / max_err_est
                assert self.coercivity_constant > 0., "coercivity constant is not positive"
                print(f"\t\tcoercivity constant: {self.coercivity_constant:.3e}")
                if logging:
                    self.greedy_logging[-1].append(self.coercivity_constant)

            if logging and (arg_max_err_est == arg_max_err_real):
                assert np.allclose(fom_sols[:, snapshot_idx_max_err_real], 
                                   np.squeeze(to_be_added_fom_sol), atol=1e-14, rtol=0.), "adding the wrong FOM solution to basis?"
                assert np.allclose(exact_error, max_err_real, atol=1e-12, rtol=0.), "calibrating the coercivity constant incorrectly?"
                
            # update snapshot matrix by adding new FOM solution and interal records
            if not error_calibration_mode and niter < max_iter-1:
                print(f"\t\tadding snapshot ID {snapshot_idx_max_err_est} to current basis {self.included_snapshots_idxs}")
                self.add_fom_solution_to_basis(to_be_added_fom_sol)
                self.included_snapshots_idxs.add(snapshot_idx_max_err_est)

    def add_fom_solution_to_basis(self, fom_sol):
        self.fom_solutions = np.column_stack((self.fom_solutions, fom_sol))
        if self.approach == "pod":
            self.snapshot_matrix = np.copy(self.fom_solutions)
            self.apply_pod(update_offline_stage=False)
            # one could do here an incremental SVD; since we do not explore
            # updating the snapshot matrix after POD, we perform here a
            # full truncated SVD again only for completeness
        elif self.approach in ("orth", "greedy"):
            try:
                self.snapshot_matrix, self.snapshot_matrix_r = qr_insert(Q=self.snapshot_matrix, 
                                                                         R=self.snapshot_matrix_r, 
                                                                         u=fom_sol, k=-1, 
                                                                         which='col', rcond=None)
                # `qr_insert` will raise a `LinAlgError` if one of the columns of u lies in the span of Q,
                # which is measured using `rcond`; if that is the case, we perform a QR decomposition
                # on the updated snapshot matrix, which results in an orthonormal basis  with the requested size
            except LinAlgError:
                print("Warning: need to perform full QR decomposition. Added snapshot is orthogonalzied away.")
                self.snapshot_matrix = np.copy(self.fom_solutions)
                self.apply_orthonormalization(update_offline_stage=False)
        elif self.approach is None: 
            self.snapshot_matrix = np.copy(self.fom_solutions)
        else:
            raise NotADirectoryError(f"Approach '{self.approach}' is unknown.")

        self.update_offline_stage()


class MatrixNumerovROM:
    """
    implements the GROM and LSPG ROM based on the matrix Numerov method ("all-at-once Numerov").
    It uses the `MatrixNumerov` implementation, which has only the sampled
    wave function in the solution vector, not (a,b) or some scattering matrix.
    This emulator is real-valued for real-valued potentials.
    """
    def __init__(self, scattExp, grid, free_lecs=None, 
                 num_snapshots_init=3, num_snapshots_max=15, 
                 approach="pod", pod_rcond=1e-12, pod_num_modes=None,
                 init_snapshot_lecs=None,
                 greedy_max_iter=5, 
                 mode="linear",
                 greedy_training_mode="grom",
                 num_pts_fit_asympt_limit=25,
                 inhomogeneous=True,
                 verbose=True,
                 seed=10203) -> None:
        # internal book keeping
        self.scattExp = scattExp
        self.grid = grid
        self.free_lecs = free_lecs
        assert bool(free_lecs) != bool(init_snapshot_lecs), "either `free_lecs` or `init_snapshot_lecs` has to be specified" 
        self.num_snapshots_init = num_snapshots_init
        self.num_snapshots_max = num_snapshots_max
        self.approach = approach
        self.pod_rcond = pod_rcond
        self.pod_num_modes = pod_num_modes
        self.greedy_max_iter = greedy_max_iter
        self.mode = mode
        assert greedy_training_mode in ("grom", "lspg"), f"requested emulator training mode '{mode}' unknown"
        self.greedy_training_mode = greedy_training_mode
        self.num_pts_fit_asympt_limit = num_pts_fit_asympt_limit
        self.verbose = verbose
        self.seed = seed
        self.greedy_logging = []
        self.coercivity_constant = 1.
        self.inhomogeneous = inhomogeneous
        self.mask_fit_asympt_limit = np.concatenate((np.zeros(grid.getNumPointsTotal - self.num_pts_fit_asympt_limit), 
                                                    np.ones(self.num_pts_fit_asympt_limit))).astype(bool)
        self.design_matrix_FG = free_solutions_F_G(l=self.scattExp.l, r=grid.points[self.mask_fit_asympt_limit], 
                                                   p=self.scattExp.p, derivative=False) / self.scattExp.p
        self.F_grid = free_solutions_F_G(l=self.scattExp.l, r=grid.points, 
                                         p=self.scattExp.p, derivative=False)[:, 0]  # could be improved together with the lines above
        
        self.S = np.zeros((grid.getNumPointsTotal, self.num_pts_fit_asympt_limit))
        self.S[self.mask_fit_asympt_limit, :] = np.eye(self.num_pts_fit_asympt_limit)
        self.fit_matrix = np.linalg.pinv(self.design_matrix_FG) @ self.S.T
        self.norm_Minv_Sdagger = np.linalg.norm(self.fit_matrix, ord=2)  # 2-norm (i.e., largest singular vector)
        assert self.norm_Minv_Sdagger < 10., f"Warning: larger 2-norm for propagating errors in the phase shifts: {self.Minv_Sdagger}"

        # FOM solver (all-at-once Numerov)
        rseParams = {"grid": grid, 
                     "scattExp": scattExp, 
                     "potential": scattExp.potential, 
                     "inhomogeneous": self.inhomogeneous
                     }
        from RseSolver import g_s_affine
        self.numerov_solver = MatrixNumerov(self.grid.points, 
                                            g=None, g_s=g_s_affine, 
                                            y0=0., 
                                            y1=(0. if self.inhomogeneous else 1.), 
                                            params=rseParams)
        self.n_theta = self.numerov_solver.n_theta
        
        # training
        self.training(init_snapshot_lecs)
 
    @property
    def potential(self):
        return self.scattExp.potential
    
    def wavefct(self, cvec):
        return self.snapshot_matrix @ cvec
    
    
    def Lmatrix(self, cvec):
        return self.snapshot_Lvec @ cvec
                
                
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
            self.lec_all_samples = self.potential.lec_array_from_dict(snapshot_lecs)
            self.num_snapshots_max = len(self.lec_all_samples)

        # initial snapshot selection (random)
        rng = np.random.default_rng(seed=self.seed)
        if self.num_snapshots_init is None:
            self.num_snapshots_init = self.num_snapshots_max
        assert self.num_snapshots_init <= self.num_snapshots_max, "can't request more initial snapshots than maximally allowed"
        self.included_snapshots_idxs = set(rng.choice(range(self.num_snapshots_max), 
                                                      size=self.num_snapshots_init, replace=False))

        # building snapshot matrix
        init_lecs = np.take(a=self.lec_all_samples, indices=list(self.included_snapshots_idxs), axis=0)
        self.snapshot_matrix = self.simulate(init_lecs)

        self.all_snapshot_idxs = set(range(len(self.lec_all_samples)))

        # choose between snapshot approaches 
        # (requires updating the offline stage, see below)
        self.fom_solutions = np.copy(self.snapshot_matrix)
        if self.approach == "pod":
            self.apply_pod(update_offline_stage=True)
        elif self.approach == "greedy":
            self.apply_orthonormalization(update_offline_stage=True)
            self.greedy_algorithm()
        elif self.approach == "orth":
            self.apply_orthonormalization(update_offline_stage=True)
        elif self.approach is None: 
            self.update_offline_stage(update_matrix_asympt_limit=True)
            if self.verbose: 
                print(f"Warning: snapshot matrix not orthonormalized. Consider using `approach='orth'`")
        else:
            raise NotImplementedError(f"Approach '{self.approach}' is unknown.")

    def apply_orthonormalization(self, update_offline_stage=True):
        prev_shape = self.snapshot_matrix.shape
        q, r = qr(self.snapshot_matrix, mode='economic')
        self.snapshot_matrix = q
        self.snapshot_matrix_r = r
        curr_shape = self.snapshot_matrix.shape
        assert curr_shape == prev_shape, "number of basis elements decreased due to orthonormalization "

        if update_offline_stage:
            self.update_offline_stage(update_matrix_asympt_limit=True)

    @staticmethod
    def truncated_svd(matrix, rcond=None, num_modes=None):
        # modified from scipy's `orth()` function:
        # https://github.com/scipy/scipy/blob/v1.14.1/scipy/linalg/_decomp_svd.py#L302-L347
        from scipy.linalg import svd
        u, s, vh = svd(matrix, full_matrices=False)
        M, N = u.shape[0], vh.shape[1]
        if rcond is None:
            rcond = np.finfo(s.dtype).eps * max(M, N)
        tol = np.amax(s, initial=0.) * rcond
        num = np.sum(s > tol, dtype=int) if num_modes is None else num_modes # = r
        Q = u[:, :num]
        return Q, s[:num], vh.conjugate().transpose()[:, :num]

    def apply_pod(self, update_offline_stage=True):
        prev_rank = self.snapshot_matrix.shape[1]
        # calling `sp.linalg.orth()`` would be enough here, but we might want to study different SVD truncations in the future
        Ur, S, Vr = self.truncated_svd(self.snapshot_matrix, rcond=self.pod_rcond, 
                                       num_modes=self.pod_num_modes)
        self.snapshot_matrix = Ur

        curr_rank = self.snapshot_matrix.shape[1]
        compression_rate = 1. - curr_rank / prev_rank
        if self.verbose:
            print(f"using {curr_rank} out of {prev_rank} POD modes in total: compression rate is {compression_rate*100:.1f} %")
        
        if update_offline_stage:
            self.update_offline_stage(update_matrix_asympt_limit=True)

    def update_offline_stage(self, update_matrix_asympt_limit=True):
        # Note: when adding snapshots to the emulator basis, one does not need to recompute all tensors again,
        # rather one can update them for computational efficiency. Since this update only occurs in the offline 
        # stage of the emulator, we keep it in this proof-of-principle work simple and compute the tensors from scratch

        X_red = self.snapshot_matrix[2:,:]
        X_dagger = X_red.conjugate().T
        A_tensor = self.numerov_solver.A_tensor
        S_tensor = self.numerov_solver.S_tensor
        S_tensor_conj = S_tensor.conjugate()

        # GROM emulator equations: reduction and projection
        A_tensor_x_X_red = A_tensor @ X_red
        self.A_tilde_grom = X_dagger @ A_tensor_x_X_red
        self.s_tilde_grom = np.tensordot(X_dagger, S_tensor, axes=[1,1]).T

        # prestore tensors for error estimates
        einsum_args = dict(optimize="greedy", dtype=np.longdouble)
        self.error_term1 = np.einsum("ki,alk,blm,mj->ijab", 
                                      X_red.conjugate(), A_tensor.conjugate(), A_tensor, X_red, **einsum_args) 
        self.error_term2 = np.einsum("bk,aki,ij->baj", S_tensor_conj, A_tensor, X_red, **einsum_args)
        self.error_term3 = S_tensor_conj @ S_tensor.T

        # construct Y matrix for alternative error estimation and LSPG-ROM
        tmp = np.column_stack(np.squeeze(np.split(A_tensor_x_X_red, self.n_theta, axis=0)))
        tmp = np.column_stack((tmp, S_tensor.T))
        self.Y_tensor = orth(tmp, rcond=None)
        prev_rank = min(tmp.shape)
        shape_Y_tensor = self.Y_tensor.shape
        curr_rank = min(shape_Y_tensor)
        compression_rate = 1. - curr_rank / prev_rank
        if self.verbose:
            print(f"POD[ Y ]: compression rate is {compression_rate*100:.1f} %; dim: {shape_Y_tensor}")
        assert shape_Y_tensor[1] < shape_Y_tensor[0] //4, "semi-reduced space for Y is large!"

        # prestore tensors for alternative error estimation  
        Y_conj = self.Y_tensor.conjugate()
        self.S_tensor_projY = np.einsum("iw,ai->aw", Y_conj, S_tensor, **einsum_args)
        self.A_tensor_projY = np.einsum("iw,aij,ju->awu", Y_conj, A_tensor, X_red, **einsum_args)

        # LSPG emulator equations
        Y_tensor_dagger = Y_conj.T
        self.A_tilde_lspg = Y_tensor_dagger @ A_tensor_x_X_red
        self.s_tilde_lspg = np.tensordot(Y_tensor_dagger, S_tensor, axes=[1,1]).T

        # emulator equation
        if update_matrix_asympt_limit:
            self.matrix_asympt_limit = np.linalg.lstsq(self.design_matrix_FG, 
                                                       X_red[self.mask_fit_asympt_limit[2:],:], rcond=None)[0] 
        
    def get_scattering_matrix(self, ab_arr, full_sols=None, which="K"):
        ab_arr = np.copy(ab_arr)
        if self.inhomogeneous:
            ab_arr[0, :] += self.scattExp.p
        if which == "ab":
            return ab_arr
        elif which == "delta":
            return np.rad2deg(np.arctan2(ab_arr[1, :], ab_arr[0, :]))  # arctan2(b, a) in degrees
        elif which == "S":
            num = ab_arr[0, :] + 1j*ab_arr[1, :]
            denom = ab_arr[0, :] - 1j*ab_arr[1, :]
            return num / denom
        elif which == "K":
            return ab_arr[1, :] / ab_arr[0, :]  # b / a 
        elif which == "u": 
            tmp = full_sols
            if self.inhomogeneous:
                tmp += self.F_grid[:, np.newaxis]
            return tmp / ab_arr[0, :]
        else:
            raise NotImplementedError(f"Scattering matrix '{which}' unknown.")

    def simulate(self, lecList, which="all"):
        full_sols = self.numerov_solver.solve(lecList)
        if which == "all":
            return full_sols
        else:
            ab_arr = np.linalg.lstsq(self.design_matrix_FG, 
                                     full_sols[self.mask_fit_asympt_limit,:], rcond=None)[0] 
            return self.get_scattering_matrix(ab_arr, full_sols=full_sols, which=which)

    def emulate(self, lecList, which="all", 
                estimate_norm_residual=False, mode=None,
                calc_error_bounds=False, calibrate_norm_residual=False,
                cond_number_threshold=None, self_test=True, 
                lspg_rcond=None, lspg_solver="svd"):
        if mode is None:
            mode = self.greedy_training_mode
        assert mode in ("grom", "lspg"), f"requested mode '{mode}' unknown"
        if self.verbose:
            print(f"emulating '{which}' using '{mode}'")
        coeffs_all = []
        A_tilde_tensor = self.A_tilde_grom if mode == "grom" else self.A_tilde_lspg
        s_tilde_tensor = self.s_tilde_grom if mode == "grom" else self.s_tilde_lspg
        for lecs in lecList:
            # reconstruct linear system    
            A_tilde = np.tensordot(lecs, A_tilde_tensor, axes=1)
            s_tilde = np.tensordot(lecs, s_tilde_tensor, axes=1)

            # solve linear system and emulate
            if cond_number_threshold is not None:
                cond_number = np.linalg.cond(A_tilde)
                if cond_number > cond_number_threshold:
                    print(f"Warning: condition number is above threshold (aff)! {cond_number:.8e}")
            if mode == "grom":
                coeffs_curr = np.linalg.solve(A_tilde, s_tilde)
            else:
                if lspg_solver == "svd":
                    coeffs_curr, residuals, rank, svals = np.linalg.lstsq(A_tilde, s_tilde, rcond=lspg_rcond)
                else:
                    Q, R = np.linalg.qr(A_tilde)
                    coeffs_curr = np.linalg.solve(R, Q.conjugate().T @ s_tilde)  # Solve Rx = Q^dagger s_tilde
            coeffs_all.append(coeffs_curr)
            # print("sum of the emulator basis coeffs", np.sum(coeffs_curr))
        coeffs_all = np.array(coeffs_all).T
        
        if which in ("ab", "delta", "u", "K", "S"):
            ab_arr = self.matrix_asympt_limit @ coeffs_all
            emulated_sols = self.snapshot_matrix @ coeffs_all if which == "u" else None
            return self.get_scattering_matrix(ab_arr, full_sols=emulated_sols, which=which)
            # alternatively, one could use the following options:
            # relevant_sols = self.snapshot_matrix[self.mask_fit_asympt_limit,:] @ coeffs_all
            # ab_arr = np.linalg.lstsq(self.design_matrix_FG, relevant_sols, rcond=None)[0]
            # ab_arr = np.linalg.pinv(self.design_matrix_FG) @ relevant_sols
        else:
            emulated_sols = self.snapshot_matrix @ coeffs_all
        
        # enforce initial conditions
        # emulated_sols[0, :] = self.numerov_solver.y0
        # emulated_sols[1, :] = self.numerov_solver.y1

        if estimate_norm_residual:
            num_norm_residuals = len(lecList)
            norm_residuals = np.empty(num_norm_residuals)
            error_bounds = []
            for ilecs, lecs in enumerate(lecList):
                norm_residuals[ilecs] = self.reconstruct_norm_residual(lecs, coeffs_all[:,ilecs])

            if self_test or calc_error_bounds:
                norm_residuals_FOM = np.empty(num_norm_residuals)
                for ilecs, lecs in enumerate(lecList):
                    norm_residuals_FOM[ilecs], bounds = self.numerov_solver.residuals(emulated_sols[2:,ilecs], lecs, squared=False,
                                                                                      calc_error_bounds=calc_error_bounds)
                    error_bounds.append(bounds)  # may be None
                error_bounds = np.array(error_bounds)

                if self_test:
                    max_diff = np.max(np.abs(norm_residuals - norm_residuals_FOM))
                    # print("max_diff", max_diff)
                    assert np.allclose(max_diff, 0, atol=1e-10, rtol=0.1), f"something's wrong with the reconstructed residual; max diff: {max_diff}"

            if calibrate_norm_residual:
                norm_residuals *= self.coercivity_constant

            return emulated_sols, norm_residuals, error_bounds    
        else:
            return emulated_sols

    def reconstruct_norm_residual(self, lecs, coeffs):
        lecs_H = lecs.conjugate().T
        coeffs_H = coeffs.conjugate().T

        einsum_args = dict(optimize="greedy", dtype=np.longdouble)

        # import time
        # start_time = time.time()
        
        # first term (x^\dagger A^\dagger A x)
        res = np.empty(3, dtype=np.longdouble)
        res[0] = np.einsum("i,ijab,a,b,j->", coeffs_H, self.error_term1, lecs_H, lecs, coeffs, **einsum_args)
        ## second term (s^dagger A x)
        res[1] = -2.*np.real(np.einsum("baj,a,b,j->", self.error_term2, lecs, lecs_H, coeffs, **einsum_args))
        ## third term (s^\dagger s)
        res[2] = lecs_H @ self.error_term3 @ lecs  # dimensions probably too small for multidot to be more efficient

        # sum the contributions and return
        total = np.sqrt(np.max(((0., np.sum(res)))))  # prevent eps^2 < 0 due to round-off errors and ill-conditioning

        # end_time = time.time()
        # elapsed_time = end_time - start_time
        # print(f"Elapsed time (scalar): {elapsed_time*1e-6:.5e} micro seconds")

        # LSPG
        # start_time = time.time()

        component_S = np.tensordot(lecs, self.S_tensor_projY, axes=1)
        component_A = lecs @ (self.A_tensor_projY @ coeffs)
        total2 = np.linalg.norm(component_S - component_A)
        
        # end_time = time.time()
        # elapsed_time = end_time - start_time
        # print(f"Elapsed time (vector norm): {elapsed_time1e-6:.5e} seconds")
        # print(f"reconstructed norm residuals diff: {total - total2:.2e} | {total:.2e} {total2:.2e}")
        assert np.allclose(total, total2, atol=1e-8, rtol=0.), f"total and total2 inconsistent; diff: {total-total2}"
        
        return total2

    def greedy_algorithm(self, req_num_iter=None, mode=None,
                         calibrate_error_estimation=True, atol=1e-12,
                         lowest_mean_norm_residuals=1e-11,
                         logging=True):
        if mode is None:
            mode = self.greedy_training_mode

        max_num_iter = self.num_snapshots_max-len(self.included_snapshots_idxs)
        if req_num_iter is None:
            req_num_iter = min(self.greedy_max_iter, max_num_iter)
        elif calibrate_error_estimation:
            req_num_iter += 1

        if max_num_iter > 0 and req_num_iter >0 and req_num_iter <= max_num_iter:
            if self.verbose:
                print("snapshot idx already included in basis:", self.included_snapshots_idxs)
                print(f"now greedily improving the snapshot basis:")
        else:
            if self.verbose:
                print("Nothing to be done. Maxed out available snapshots and/or number of iterations")
            return

        current_mean_norm_residuals = np.inf
        for niter in range(req_num_iter):
            if self.verbose:
                print(f"\titeration #{niter+1} of max {req_num_iter}:")
            
            # determine candidate snapshots that the greedy algorithm can add
            candidate_snapshot_idxs = list(self.all_snapshot_idxs - self.included_snapshots_idxs)
            if self.verbose: print("\tavailable candidate snapshot idx to be added:", candidate_snapshot_idxs)
            candidate_snapshots = np.take(a=self.lec_all_samples, indices=candidate_snapshot_idxs, axis=0)

            # determine snapshots at which to compute the errors
            emulate_snapshot_idxs = list(self.all_snapshot_idxs) if self.mode == "linear" else candidate_snapshot_idxs.copy()
            emulate_snapshots = np.take(a=self.lec_all_samples, indices=emulate_snapshot_idxs, axis=0)
            if self.verbose: print("\temulate snapshots:", emulate_snapshot_idxs)
            # in the case of the "linear" mode, we want to make error plots, so we emulate here all snapshots, 
            # including the ones we've already considered in the greedy iteration.
    
            # emulate candidate snapshots           
            emulated_sols, norm_residuals, error_bounds = self.emulate(emulate_snapshots, mode=mode,
                                                                       estimate_norm_residual=True, 
                                                                       calibrate_norm_residual=False,
                                                                       calc_error_bounds=logging, 
                                                                       cond_number_threshold=None, self_test=logging)

            if logging:
                # in practice, ROM will not compute the FOM results (just for checking/benchmarking)
                fom_sols = self.simulate(emulate_snapshots)
                norm_error_exact = np.linalg.norm(emulated_sols - fom_sols, axis=0)
                self.greedy_logging.append([self.included_snapshots_idxs.copy(), 
                                            fom_sols, emulated_sols, 
                                            norm_residuals, norm_error_exact, error_bounds])
                # both `norm_error_exact` and `norm_residuals` should be minimal at the snapshot locations 
                # already included in the basis; i.e., `self.included_snapshots_idxs`

            # check that the estimated mean error decreases
            mean_norm_residuals = np.mean(norm_residuals)
            if mean_norm_residuals > current_mean_norm_residuals:
                if self.verbose: print(f"\t\tWarning: estimated mean error has increased ({mean_norm_residuals:.5e} vs {current_mean_norm_residuals:.5e}).")
            if mean_norm_residuals < lowest_mean_norm_residuals:  # safeguard against numerical instabilities
                if self.verbose: print(f"\t\tWarning: estimated mean error reached requested bound < {lowest_mean_norm_residuals:.2e}.")
            current_mean_norm_residuals = mean_norm_residuals

            # select the candidate snapshot with maximum (estimated) error
            est_err_candidate_snapshots = np.take(a=norm_residuals, 
                                                  indices=candidate_snapshot_idxs, axis=0) if self.mode == "linear" else norm_residuals
            arg_max_err_est = np.argmax(est_err_candidate_snapshots)
            max_err_est = est_err_candidate_snapshots[arg_max_err_est]
            snapshot_idx_max_err_est = candidate_snapshot_idxs[arg_max_err_est]
            snapshot_max_err_est = candidate_snapshots[arg_max_err_est]

            if logging:
                # check whether the error estimator found indeed the snapshot with maximum (estimated) error
                real_err_candidate_snapshots = np.take(a=norm_error_exact, 
                                                       indices=candidate_snapshot_idxs, axis=0) if self.mode == "linear" else norm_error_exact
                arg_max_err_real = np.argmax(real_err_candidate_snapshots)
                max_err_real = real_err_candidate_snapshots[arg_max_err_real]
                snapshot_idx_max_err_real = candidate_snapshot_idxs[arg_max_err_real]

                if arg_max_err_est != arg_max_err_real and self.verbose:
                    print(f"\t\tWarning: estimated max error doesn't match real max error: arg {arg_max_err_est} vs {arg_max_err_real}")
                if self.verbose: print(f"\t\testimated max error: {max_err_est:.3e} | real max error: {max_err_real:.3e}")
                        
            # check whether accuracy goal is achieved
            scaled_max_err_est = self.coercivity_constant * max_err_est if calibrate_error_estimation else max_err_est
            if scaled_max_err_est < atol and niter > 0:  # need to have completed at least one iteration for calibration at this point
                if self.verbose: print(f"accuracy goal 'atol = {atol}' achieved. Terminating greedy iteration.")
                break

            # perform FOM calculation at the location of max estimated error
            to_be_added_fom_sol = self.simulate([snapshot_max_err_est])
            assert np.allclose(snapshot_max_err_est, 
                               self.lec_all_samples[snapshot_idx_max_err_est]), "performed FOM calculation at wrong location?"

            # calibrate error estimator
            if calibrate_error_estimation:
                loc = emulate_snapshot_idxs.index(snapshot_idx_max_err_est) if self.mode == "linear" else arg_max_err_est
                assert emulate_snapshot_idxs[loc] == snapshot_idx_max_err_est, "wrong ID"
                exact_error = np.linalg.norm(np.squeeze(to_be_added_fom_sol) - emulated_sols[:, loc])
                self.coercivity_constant = exact_error / max_err_est
                assert self.coercivity_constant > 0., "coercivity constant is not positive"
                if self.verbose: print(f"\t\tcoercivity constant: {self.coercivity_constant:.3e}")
                if logging:
                    self.greedy_logging[-1].append(self.coercivity_constant)
                    ab_emulated = self.emulate(emulate_snapshots, which="ab", 
                                               mode=mode, estimate_norm_residual=False, 
                                               calibrate_norm_residual=False, 
                                               calc_error_bounds=False, 
                                               cond_number_threshold=None, 
                                               self_test=False)
                    delta_error_est = np.rad2deg(1) * self.norm_Minv_Sdagger / np.linalg.norm(ab_emulated, axis=0) 
                    delta_error_est *= self.coercivity_constant*norm_residuals
                    delta_simulated = self.simulate(emulate_snapshots, which="delta")
                    delta_emulated = self.emulate(emulate_snapshots, which="delta", 
                                               mode=mode, estimate_norm_residual=False, 
                                               calibrate_norm_residual=False, 
                                               calc_error_bounds=False, 
                                               cond_number_threshold=None, 
                                               self_test=False)
                    self.greedy_logging[-1].extend([delta_emulated, delta_simulated, delta_error_est])

            if logging and (arg_max_err_est == arg_max_err_real):
                assert np.allclose(fom_sols[:, loc], 
                                   np.squeeze(to_be_added_fom_sol), atol=1e-14, rtol=0.), "adding the wrong FOM solution to basis?"
                assert np.allclose(exact_error, max_err_real, atol=1e-12, rtol=0.), "calibrating the coercivity constant incorrectly?"
                
            # update snapshot matrix by adding new FOM solution and interal records
            if niter < req_num_iter-1:
                if self.verbose: print(f"\t\tadding snapshot ID {snapshot_idx_max_err_est} to current basis {self.included_snapshots_idxs}")
                self.add_fom_solution_to_basis(to_be_added_fom_sol)
                self.included_snapshots_idxs.add(snapshot_idx_max_err_est)

    def add_fom_solution_to_basis(self, fom_sol):
        self.fom_solutions = np.column_stack((self.fom_solutions, fom_sol))
        if self.approach == "pod":
            self.snapshot_matrix = np.copy(self.fom_solutions)
            self.apply_pod(update_offline_stage=True)
            # one could do here an incremental SVD; since we do not explore
            # updating the snapshot matrix after POD, we perform here a
            # full truncated SVD again only for completeness
        elif self.approach in ("orth", "greedy"):
            try:
                self.snapshot_matrix, self.snapshot_matrix_r = qr_insert(Q=self.snapshot_matrix, 
                                                                         R=self.snapshot_matrix_r, 
                                                                         u=fom_sol, 
                                                                         k=self.snapshot_matrix.shape[1],  # -1 here would **not** make the last column (append)
                                                                         which='col', rcond=None)

                # update offline stage given the new snapshot matrix
                self.update_offline_stage(update_matrix_asympt_limit=False)
                # for performance optimization, we add one column to `matrix_asympt_limit` manually,
                # instead of letting `update_offline_stage` recompute it completely.
                to_be_added_a_b = np.linalg.lstsq(self.design_matrix_FG, self.snapshot_matrix[self.mask_fit_asympt_limit, -1], rcond=None)[0] 
                self.matrix_asympt_limit = np.column_stack((self.matrix_asympt_limit, to_be_added_a_b))

                # useful for debugging: check that updated a,b matrix matches recomputed a,b matrix
                # ab_arr = np.linalg.lstsq(self.design_matrix_FG, 
                #                         self.snapshot_matrix[self.mask_fit_asympt_limit, :], rcond=None)[0] 
                # assert np.allclose(self.matrix_asympt_limit, ab_arr), "something's wrong with the updated (a,b) matrix (containing the asympt. limit parameters)"

                # `qr_insert` will raise a `LinAlgError` if one of the columns of u lies in the span of Q,
                # which is measured using `rcond`; if that is the case, we perform a QR decomposition
                # on the updated snapshot matrix, which results in an orthonormal basis  with the requested size
            except LinAlgError:
                if self.verbose: print("Warning: need to perform full QR decomposition. Added snapshot is 'orthogonalzied away'.")
                self.snapshot_matrix = np.copy(self.fom_solutions)
                self.apply_orthonormalization(update_offline_stage=True)
                raise NotImplementedError("This part may be correct, but should be further checked. Consider implementing re-orthogonalization.")
        elif self.approach is None: 
            self.snapshot_matrix = np.copy(self.fom_solutions)
            self.update_offline_stage(update_matrix_asympt_limit=True)  # here could be an incremental update
        else:
            raise NotADirectoryError(f"Approach '{self.approach}' is unknown.")
