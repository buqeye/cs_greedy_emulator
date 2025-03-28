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
from math import comb
from string import ascii_lowercase as alphabet


class convergenceAnalysis:
    def __init__(self, 
                 param_samples,
                 channel_lbl="1S0np", 
                 E_MeV=50,
                 potential_lbl="minnesota",
                 snapshot_range=(3, 8+1),
                 num_sample=400,
                 inhomogeneous=True, 
                 emulator_type="lspg", 
                 which="K",
                 seed=None,
                 take_all=True):
        self.snapshot_range = snapshot_range
        self.E_MeV = E_MeV
        self.num_sample = num_sample
        self.inhomogeneous = inhomogeneous
        self.emulator_type = emulator_type
        self.which = which
        self.rng = np.random.default_rng(seed=seed)

        potentialArgs = {"label": potential_lbl, "potId": 213}
        channel = Channel(as_str=channel_lbl)
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
                        greedy_training_mode=self.emulator_type,
                        verbose=False,
                        logging=False,
                        seed=None
                        )

        # simulator results (doesn't matter which `approach` is chosen)
        tmp = MatrixNumerovROM(num_snapshots_init=None, 
                            init_snapshot_lecs=param_samples["training"], 
                            approach="pod", **self.args)
        self.output_simulator = tmp.simulate(self.lecs_list_validation, which=which)

        self.greedy_emulators = self.get_worst_best_init_greedy_emulator(take_all=take_all)

    def y_axis_quantity(self, output_emulator):
        # specify what to plot on y axis
        x_emu = self.scattExp.p / output_emulator
        x_sim = self.scattExp.p / self.output_simulator
        # return np.abs((x_emu - x_sim) / x_sim)
        # return np.log10(np.abs((x_emu - x_sim) / x_sim))
        return np.abs(x_emu - x_sim) / (np.abs(x_emu) + np.abs(x_sim)) * 2.

    def get_worst_best_init_greedy_emulator(self, take_all=True):
        # find best and worst initial training basis (as measured by the mean error) to start off the greedy algorithm
        use_set = self.param_samples["training"]
        combinatorics = list(combinations(range(len(use_set)), self.snapshot_range[0]))
        size = len(combinatorics) if take_all else np.min((self.num_sample, len(combinatorics)))
        
        print(f"scanning {size} (out of {len(combinatorics)}) for best/worst initial emulator basis")
        lec_idxs_array = self.rng.choice(combinatorics, size=size, replace=False)
        tmp_arr = []
        if take_all:
            assert len(lec_idxs_array) == comb(len(use_set), self.snapshot_range[0]), "something's wrong with the combinatorics"
        for idxs in lec_idxs_array:
            lecs = np.take(use_set, indices=idxs)
            emul = MatrixNumerovROM(init_snapshot_lecs=lecs, 
                                    num_snapshots_init=None, 
                                    approach="orth", **self.args)
            tmp = emul.emulate(self.lecs_list_validation, mode=self.emulator_type, which=self.which)
            tmp_arr.append(self.y_axis_quantity(tmp))
        tmp_arr = np.array(tmp_arr)
        means = np.median(tmp_arr, axis=1)
        self.initial_lecs_idxs_worst = lec_idxs_array[np.argmax(means)]
        self.initial_lecs_idxs_best = lec_idxs_array[np.argmin(means)]
        print("worst idxs:", self.initial_lecs_idxs_worst)
        print("best idxs:", self.initial_lecs_idxs_best)

        greedy_emul_arr = []
        for lbl, idxs in (("worst", self.initial_lecs_idxs_worst), 
                          ("best", self.initial_lecs_idxs_best)):
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
            assert pod_emul.snapshot_matrix.shape[1] == num_snapshots, "something's wrong in the POD emulator"
            df_tmp = pd.DataFrame(data={"approach": "POD", 
                                        "num_snapshots": num_snapshots, 
                                        "E_MeV": self.E_MeV,
                                        "error": self.y_axis_quantity(tmp), 
                                        "emulator_type": self.emulator_type}
                                        )
            df = pd.concat((df, df_tmp))
        return df if df_out is None else pd.concat((df_out, df))

    def track_LHS_emulator(self, df_out=None, max_samples=100000):
        ## randomly selects training points from the training set
        print("tracking LHS emulator")
        df = pd.DataFrame()
        for num_snapshots in range(*self.snapshot_range):
            if comb(len(self.param_samples["training"]), num_snapshots) > max_samples:
                use_samples = self.rng.choice(self.param_samples["training"], 
                                              size=max_samples, replace=False)
            else:
                use_samples = self.param_samples["training"]

            combinatorics = list(combinations(use_samples, num_snapshots))
            size = np.min((self.num_sample, len(combinatorics)))
            lhs_lecs_array = self.rng.choice(combinatorics, size=size, replace=False)
            print(f"\trunning with {num_snapshots} snapshots ({len(lhs_lecs_array)} samples)")
            for lecs in lhs_lecs_array:
                lhs_emul = MatrixNumerovROM(init_snapshot_lecs=lecs, 
                                            num_snapshots_init=None, 
                                            approach="orth", **self.args)
                tmp = lhs_emul.emulate(self.lecs_list_validation, 
                                    mode=self.emulator_type, which=self.which)
                assert lhs_emul.snapshot_matrix.shape[1] == num_snapshots, "something's wrong in the LHS emulator"
                df_tmp = pd.DataFrame(data={"approach": "LHS", 
                                            "num_snapshots": num_snapshots, 
                                            "E_MeV": self.E_MeV,
                                            "error": self.y_axis_quantity(tmp), 
                                            "emulator_type": self.emulator_type}
                                            )
                df = pd.concat((df, df_tmp))
        return df if df_out is None else pd.concat((df_out, df))

    def track_greedy_emulators(self, df_out=None):
        print("tracking greedy emulators")
        df = pd.DataFrame()

        for num_snapshots in range(*self.snapshot_range):
            print(f"\trunning with {num_snapshots} snapshots")
            # for debugging: re-build greedy emulators
            # greedy_emul_arr_new = []
            # for lbl, idxs in (("worst", self.initial_lecs_idxs_worst), 
            #                 ("best", self.initial_lecs_idxs_best)):
            #     print(f"init greedy run no: {num_snapshots-self.snapshot_range[0]+1}")
            #     tmp = MatrixNumerovROM(num_snapshots_init=None, 
            #                         init_snapshot_lecs=self.param_samples["training"],
            #                         included_snapshots_idxs=set(idxs),
            #                         approach="greedy", 
            #                         label=f"Greedy ({lbl}) [new]", 
            #                         **{**self.args, 
            #                            **{"greedy_max_iter": num_snapshots-self.snapshot_range[0]+1}})
            #     print(num_snapshots, tmp.included_snapshots_idxs)
            #     greedy_emul_arr_new.append(tmp)

            # for emu in greedy_emul_arr_new:
            #     print(num_snapshots, len(emu.included_snapshots_idxs))
            #     assert num_snapshots == len(emu.included_snapshots_idxs), "mismatching number of included snapshots [new]"
            #     tmp = emu.emulate(self.lecs_list_validation, mode=self.emulator_type, which=self.which)
            #     df_tmp = pd.DataFrame(data={"approach": emu.label, "num_snapshots": num_snapshots, "E_MeV": self.E_MeV,
            #                                 "error": self.y_axis_quantity(tmp)})
            #     df = pd.concat((df, df_tmp))
            
            for emu in self.greedy_emulators:
                assert num_snapshots == len(emu.included_snapshots_idxs), "mismatching number of included snapshots"
                tmp = emu.emulate(self.lecs_list_validation, mode=self.emulator_type, which=self.which)
                df_tmp = pd.DataFrame(data={"approach": emu.label, 
                                            "num_snapshots": num_snapshots, 
                                            "E_MeV": self.E_MeV,
                                            "error": self.y_axis_quantity(tmp),
                                            "emulator_type": self.emulator_type
                                            })
                df = pd.concat((df, df_tmp))
                emu.greedy_algorithm(req_num_iter=1)
        return df if df_out is None else pd.concat((df_out, df))
    

def convergenceFig(df_res, E_MeV_arr, emulator_type):
    import matplotlib.ticker as ticker 
    import seaborn as sns
    from constants import cm
    fig, axs = plt.subplots(2, 1, figsize=(8.6*cm,12.6*cm), 
                            sharex=True, sharey=True, constrained_layout=True)

    for iE_MeV, E_MeV in enumerate(E_MeV_arr):
        df = df_res[(df_res["emulator_type"] == emulator_type) & (df_res["E_MeV"] == E_MeV)]
        ax = axs[iE_MeV]
        # sns.violinplot(data=df, x="num_snapshots", y="error", hue="approach",
        #             gap=.15, split=True, inner="quart", ax=ax, fill=True,
        #             legend=("auto" if iE_MeV==0 else False))
        sns.boxplot(data=df, x="num_snapshots", y="error", hue="approach",
                    gap=.25, ax=ax, fill=True, width=.8, log_scale=True,
                    showfliers = False,
                    whis=(1, 99), legend="auto")
        if iE_MeV == 0:
            ax.set_title("relative error in $p/K$")
        ax.xaxis.set_minor_locator(plt.NullLocator())
    #     ax.set_xlim(1.5,12.5)
    #     ax.xaxis.set_major_locator(ticker.MultipleLocator(1)) 
        ax.set_xlabel("Number of snapshots/dominant POD modes")
        # ax.set_ylim(top=6, bottom=-11)
        # ax.set_yscale('log')
        # ax.set_ylabel(r"base-10 logarithmic relative error in $|p/K|$")
        ax.set_ylabel("")
        ax.text(0.7, 0.78, f"$E = {E_MeV_arr[iE_MeV]}" + "\; \mathrm{MeV}$ ", 
                transform=ax.transAxes)
        props = dict(boxstyle='round', facecolor='lightgray', alpha=0.5)
        emulator_type_lbl = {"GROM": "G-ROM", "LSPG": "LSPG-ROM"}
        ax.text(0.8, 0.89, f"{emulator_type_lbl[emulator_type.upper()]}", 
                transform=ax.transAxes, bbox=props)
        ax.text(0.75, 0.1, f"{alphabet[iE_MeV]})", transform=ax.transAxes)
        ax.text(0.75, 0.1, f"{alphabet[iE_MeV]})", transform=ax.transAxes)
        # ax.text(0.05, 0.05, "base-10 logarithmic relative error in $|p/K|$", 
        #         transform=ax.transAxes)
        # if iE_MeV == 0:
        ax.legend(ncol=2, loc="lower left", fontsize=7, handlelength=2)
        # plt.ylim(bottom=1e-9)
        fig.savefig(f"convergence_minnesota_{emulator_type}_logaxis_symerror.pdf")