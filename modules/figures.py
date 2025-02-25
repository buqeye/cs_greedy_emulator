import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from NumerovEmulator import MatrixNumerovROM
from Potential import Potential
from Channel import Channel
from ScatteringExp import ScatteringExp
from Grid import Grid
from plots.rc_params import *
from itertools import combinations


class convergenceAnalysis:
    def __init__(self, 
                 param_samples,
                 l=0, E_MeV=50,
                 potential_lbl="minnesota",
                 snapshot_range=(3, 8+1),
                 num_sample=400,
                 inhomogeneous=True, 
                 emulator_type="grom", 
                 which="K"):
        self.snapshot_range = snapshot_range
        self.num_sample = num_sample
        self.inhomogeneous = inhomogeneous
        self.emulator_type = emulator_type
        self.which = which
        self.rng = np.random.default_rng(seed=None)

        potentialArgs = {"label": potential_lbl, "kwargs": {"potId": 213}}
        channel = Channel(S=0, L=l, LL=l, J=l, channel=0)
        potential = Potential(channel, **potentialArgs)

        self.E_MeV = E_MeV
        self.inhomogenenous = True
        self.scattExp = ScatteringExp(E_MeV=E_MeV, potential=potential)

        # generate training data
        rmatch = 12
        rmin = 1e-12 
        self.grid = Grid(rmin, rmatch, numIntervals=1, numPointsPerInterval=1000,
                         type="linear", test=False) 

        # lecs arrays for training and validation
        self.param_samples = param_samples
        self.lecs_list_validation = potential.lec_array_from_dict(param_samples["validation"])
        self.lecs_list_training = potential.lec_array_from_dict(param_samples["training"])

        # common arguments of the emulators
        self.args = dict(scattExp=self.scattExp, 
                        grid=self.grid, 
                        # free_lecs={"V0": VR_range, "V1": Vs_range}, 
                        # num_snapshots_max=25,  # irrelevant for POD
                        # init_snapshot_lecs=param_samples["training"],
                        pod_rcond=None, 
                        greedy_max_iter=0, 
                        mode="random",
                        inhomogeneous=inhomogeneous,
                        verbose=False,
                        logging=False,
                        seed=None
                        )

        # simulator results (doesn't matter which `approach` is chosen)
        tmp = MatrixNumerovROM(num_snapshots_init=None, 
                            init_snapshot_lecs=param_samples["training"], 
                            approach="pod", **self.args)
        self.output_simulator = tmp.simulate(self.lecs_list_validation, which=which)

        self.greedy_emulators = self.get_worst_best_init_greedy_emulator()

    def y_axis_quantity(self, output_emulator):
        # specify what to plot on y axis
        x_emu = self.scattExp.p / output_emulator
        x_sim = self.scattExp.p / self.output_simulator
        return np.log10(np.abs((x_emu - x_sim) / x_sim))
        # return np.log10(np.abs(x_emu - x_sim) / (np.abs(x_emu) + np.abs(x_emu)) * 2.)

    def get_worst_best_init_greedy_emulator(self, take_all=True):
        # find best and worst initial training basis (as measured by the mean error) to start off the greedy algorithm
        use_set = self.param_samples["training"]
        combinatorics = list(combinations(range(len(use_set)), self.snapshot_range[0]))
        size = len(combinatorics) if take_all else np.min((self.num_sample, len(combinatorics)))
        
        print(f"scanning {size} (out of {len(combinatorics)}) for best/worst initial emulator basis")
        lec_idxs_array = self.rng.choice(combinatorics, size=size, replace=False)
        tmp_arr = []
        for idxs in lec_idxs_array:
            lecs = np.take(use_set, indices=idxs)
            emul = MatrixNumerovROM(init_snapshot_lecs=lecs, 
                                    num_snapshots_init=None, 
                                    approach="orth", **self.args)
            tmp = emul.emulate(self.lecs_list_validation, mode=self.emulator_type, which=self.which)
            tmp_arr.append(self.y_axis_quantity(tmp))
        tmp_arr = np.array(tmp_arr)
        means = np.mean(tmp_arr, axis=1)
        initial_lecs_idxs_worst = lec_idxs_array[np.argmax(means)]
        initial_lecs_idxs_best = lec_idxs_array[np.argmin(means)]
        print("worst idxs:", initial_lecs_idxs_worst)
        print("best idxs:", initial_lecs_idxs_best)

        greedy_emul_arr = []
        for lbl, idxs in (("worst", initial_lecs_idxs_worst), 
                          ("best", initial_lecs_idxs_best)):
            # lecs = np.take(use_set, indices=idxs)
            tmp = MatrixNumerovROM(num_snapshots_init=None, 
                                init_snapshot_lecs=use_set,
                                included_snapshots_idxs=set(idxs),
                                approach="greedy", 
                                label=f"Greedy ({lbl})", **self.args)
            greedy_emul_arr.append(tmp)
        return greedy_emul_arr

    def track_POD_emulator(self, df_out=None):
        print("tracking POD emulator")
        df = pd.DataFrame()
        for num_snapshots in range(*self.snapshot_range):
            print(f"\trunning with {num_snapshots} snapshots")
            pod_emul = MatrixNumerovROM(pod_num_modes=num_snapshots, 
                                        init_snapshot_lecs=self.param_samples["training"],
                                        num_snapshots_init=None, 
                                        approach="pod", label="POD", **self.args)
            tmp = pod_emul.emulate(self.lecs_list_validation, 
                                   mode=self.emulator_type, which=self.which)
            df_tmp = pd.DataFrame(data={"approach": "POD", "num_snapshots": num_snapshots, 
                                        "error": self.y_axis_quantity(tmp)})
            df = pd.concat((df, df_tmp))
        return df if df_out is None else pd.concat((df_out, df))

    def track_LHS_emulator(self, df_out=None):
        ## randomly selects training points from the training set
        print("tracking LHS emulator")
        df = pd.DataFrame()
        for num_snapshots in range(*self.snapshot_range):
            combinatorics = list(combinations(self.param_samples["training"], num_snapshots))
            size = np.min((self.num_sample, len(combinatorics)))
            lhs_lecs_array = self.rng.choice(combinatorics, size=size, replace=False)
            print(f"\trunning with {num_snapshots} snapshots")
            for lecs in lhs_lecs_array:
                lhs_emul = MatrixNumerovROM(init_snapshot_lecs=lecs, 
                                            num_snapshots_init=None, 
                                            approach="orth", **self.args)
                tmp = lhs_emul.emulate(self.lecs_list_validation, 
                                    mode=self.emulator_type, which=self.which)
                df_tmp = pd.DataFrame(data={"approach": "LHS", "num_snapshots": num_snapshots, 
                                            "error": self.y_axis_quantity(tmp)})
                df = pd.concat((df, df_tmp))
        return df if df_out is None else pd.concat((df_out, df))

    def track_greedy_emulators(self, df_out=None):
        print("tracking greedy emulators")
        df = pd.DataFrame()
        for num_snapshots in range(*self.snapshot_range):
            print(f"\trunning with {num_snapshots} snapshots")
            for emu in self.greedy_emulators:
                assert num_snapshots == len(emu.included_snapshots_idxs), "mismatching number of included snapshots"
                tmp = emu.emulate(self.lecs_list_validation, mode=self.emulator_type, which=self.which)
                df_tmp = pd.DataFrame(data={"approach": emu.label, "num_snapshots": num_snapshots, 
                                            "error": self.y_axis_quantity(tmp)})
                df = pd.concat((df, df_tmp))
                emu.greedy_algorithm(req_num_iter=1)
        return df if df_out is None else pd.concat((df_out, df))