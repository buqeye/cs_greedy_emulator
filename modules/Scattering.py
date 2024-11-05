from Potential import Potential
import os
import multiprocessing as mp
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import RseSolver
from ScatteringExp import ScatteringExp
from Grid import Grid
from Emulator import Emulator


class Scattering:
    _colors = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71", '#f39c12']*25
    _lineStyles = ["-", "--", ":", "-."]*25

    def __init__(self, rmatch, energies, channels, trainingLecList, testingLecList, potentialArgs, cacheExactSol=False):
        self.rmatch = rmatch
        self.energies = energies
        self.channels = channels
        self.trainingLecList = trainingLecList
        self.testingLecList = testingLecList
        self.potentialArgs = potentialArgs
        self.cacheExactSol = cacheExactSol

        # check that "rmatch" is large enough for the potential to vanish
        self.isRmatchLargeEnough()

        print(f"Running the emulator with {len(trainingLecList)} training points and {len(testingLecList)} testing points")
        pool = mp.Pool(mp.cpu_count())
        self.res = np.array(pool.starmap(iteration,
                                         [(energy, chan, trainingLecList, testingLecList,
                                         rmatch, potentialArgs, cacheExactSol) for chan in channels for energy in energies]))
        pool.close()

        # reshape "res" in-place for convenient access to data;
        # indices: res[channels][energies][testingPoint]
        self.res.shape = (self.numChannels, self.numEnergies, self.numTestingLecs)

    def isRmatchLargeEnough(self, rtol=0, atol=1e-6):
        for chan in self.channels:
            potential = Potential(chan, **self.potentialArgs)
            Vrmatch = [potential.eval(self.rmatch, lecs) for lecs in self.trainingLecList]
            if not np.allclose(np.abs(Vrmatch), 0, rtol=rtol, atol=atol):
                raise ValueError(f"rmatch = {self.rmatch} fm is not large enough for potential to vanish in {chan.spectNotation}. Try increasing 'rmatch'.")

    @property
    def numChannels(self):
        return len(self.channels)

    @property
    def numEnergies(self):
        return len(self.energies)

    @property
    def numTestingLecs(self):
        return len(self.testingLecList)

    @property
    def numTrainingLecs(self):
        return len(self.trainingLecList)

    def getPhaseShifts(self, type, ichan, itestPoint, method, *, realPartOnly=True):
        if type == "emulated":
            arr = np.array([elem[itestPoint].phaseShift(method) for elem in self.res[ichan]])
        elif type == "exact":
            arr = np.array([elem[itestPoint].exactSolution.phaseShift for elem in self.res[ichan]])
        elif type == "lsfit":
            arr = np.array([elem[itestPoint].getLsFit(method, phaseShiftOnly=True) for elem in self.res[ichan]])
        elif type == "emulatedError":
            arr = np.abs(self.getPhaseShifts("emulated", ichan, itestPoint, method, realPartOnly=False)\
                  - self.getPhaseShifts("exact", ichan, itestPoint, method, realPartOnly=False))
        elif type == "lsfitError":
            arr = self.getPhaseShifts("lsfit", ichan, itestPoint, method, realPartOnly=False) \
                  - self.getPhaseShifts("exact", ichan, itestPoint, method, realPartOnly=False)
        else:
            raise ValueError(f"Keyword '{type}' unknown to Phase Shifts.")

        return np.real(arr) if realPartOnly else arr

    def plotPhaseShifts(self, plotErrors=False, showExactSol=False, showLsFit=False):
        # create new pdf for figures
        pdf = PdfPages('phaseShifts.pdf')

        # create new figure for each channel
        for ichan, chan in enumerate(self.channels):
            # create new figure
            fig, ax = plt.subplots(1, 1, constrained_layout=True)
            ax.set_title(f"{chan.spectNotationTeX} with {self.numTrainingLecs} training Hamiltonians "
                         + f"and {self.numTestingLecs} testing Hamiltonians")
            plotTypeSuffix = "Error" if plotErrors else ""

            # iterate over testing Hamiltonians and KVP methods
            for itestPoint, testPoint in enumerate(self.testingLecList):
                for imethod, method in enumerate(self.res[ichan][0][itestPoint].kvpsAvail):
                    legendLbl = f"${method}$ KVP" if itestPoint == 0 else ""
                    y = self.getPhaseShifts("emulated"+plotTypeSuffix, ichan, itestPoint, method, realPartOnly=True)
                    ax.plot(self.energies, y, c=Scattering._colors[imethod], ls=Scattering._lineStyles[itestPoint],
                            label=legendLbl)

                # if requested, compute exact solution (without emulating)
                if showExactSol and not plotErrors:
                    legendLbl = "exact" if itestPoint == 0 else ""
                    exactSols = self.getPhaseShifts("exact", ichan, itestPoint, None, realPartOnly=True)
                    ax.plot(self.energies, exactSols, c="k", ls="--", lw=2, label=legendLbl)

                # plot result from least squares fit
                if showLsFit:
                    useMethod = "T"
                    legendLbl = f"ls fit ({useMethod})" if itestPoint == 0 else ""
                    lsVals = self.getPhaseShifts("lsfit"+plotTypeSuffix, ichan, itestPoint, useMethod, realPartOnly=True)
                    ax.plot(self.energies, lsVals, c="Gold", ls=Scattering._lineStyles[itestPoint], lw=1.2, label=legendLbl)

            # plot data from partial wave analysis
            path = os.environ.get('NNPHASESHIFTS')
            if path is not None and not plotErrors:
                try:
                    filepath = f"{path}/PWA93-{chan.spectNotation}-np.txt"
                    dataElab, dataDelta = np.loadtxt(filepath, unpack=True)
                    # pandas has problems recognizing the columns
                    # note that Elab = 2 Ecm (we plot over Ecm)
                    ax.plot(dataElab/2., dataDelta, marker="o", c="k", ls="", label="PWA '93", markevery=50)
                except FileNotFoundError:
                    print(f"Couldn't find PWA-93 data in '{path}'.")

            ax.set_xlim(self.energies[0], self.energies[-1])
            ax.set_ylim(0, 1)

            ax.set_xlabel(r"$E_\mathrm{cm}$ [MeV]")
            ax.set_ylabel((r"Abs. Error in " if plotErrors else "") + r"$\delta$ [deg]")

            ax.legend(loc="best", handlelength=5)

            self.setAxRcParams(ax)

            pdf.savefig(fig)
            plt.close()

        self.setPdfMetaData(pdf)
        pdf.close()

    def getDiffCrossSections(self, type, ien, itestPoint, method, theta):
        if type == "emulated":
            arr = np.sum([elem[ien][itestPoint].dsigmaL(method, theta) for elem in self.res], axis=0)
        elif type == "exact":
            arr = np.sum([elem[ien][itestPoint].exactSolution.dsigmaL(theta) for elem in self.res], axis=0)
        elif type == "emulatedError":
            return np.abs(self.getDiffCrossSections("emulated", ien, itestPoint, method, theta)\
                           - self.getDiffCrossSections("exact", ien, itestPoint, method, theta))
        else:
            raise ValueError(f"Keyword '{type}' unknown to Cross Sections.")

        return np.abs(arr) ** 2

    def plotDiffCrossSections(self, *, numTheta=180, maxNumEnergies=100, plotErrors=False, showExactSol=True):
        # create new pdf for figures
        pdf = PdfPages('diffCrossSections.pdf')
        plotTypeSuffix = "Error" if plotErrors else ""

        # grid in theta
        theta = np.linspace(0., 180., numTheta)

        # create new figure at each energy
        for ien, en in enumerate(self.energies):
            # create new figure
            fig, ax = plt.subplots(1, 1, constrained_layout=True)
            ax.set_title(f"$Ecm = $ {en:.2f} MeV with {self.numTrainingLecs} training "
                          + f"and {self.numTestingLecs} testing Hamiltonians")

            # iterate over testing Hamiltonians and KVP methods
            for itestPoint, testPoint in enumerate(self.testingLecList):
                for imethod, method in enumerate(self.res[0][0][itestPoint].kvpsAvail):
                    legendLbl = f"${method}$ KVP" if itestPoint == 0 else ""
                    y = self.getDiffCrossSections("emulated"+plotTypeSuffix, ien, itestPoint, method, theta)
                    ax.semilogy(theta, y, c=Scattering._colors[imethod],
                                ls=Scattering._lineStyles[itestPoint], label=legendLbl)

                # if requested, compute exact solution (without emulating)
                if showExactSol and not plotErrors:
                    legendLbl = "exact" if itestPoint == 0 else ""
                    yexact = self.getDiffCrossSections("exact", ien, itestPoint, None, theta)
                    ax.plot(theta, yexact, c="k", ls="--", lw=2, label=legendLbl)

            # set params
            ax.set_xlim(theta[0], theta[-1])
            # ax.set_ylim(-30, 120)

            ax.set_xlabel(r"$\theta$ [deg]")
            ax.set_ylabel((r"Abs. Error in " if plotErrors else "")
                          + r"$\mathrm{d}\sigma/\mathrm{d}\Omega$ [mb/rad]")

            ax.legend(loc="best", handlelength=5)

            self.setAxRcParams(ax)

            pdf.savefig(fig)
            plt.close()

            # prevent unnecessarily large pdf files due to large number of pages
            if ien >= maxNumEnergies:
                print(f"Warning: maximum number of pdf pages reached (safe guard). Stopping at Ecm = {en} MeV.")
                break

        self.setPdfMetaData(pdf)
        pdf.close()

    def getCrossSections(self, type, iens, itestPoint, method):
        if type == "emulated":
            arr = np.sum([[elem[ien][itestPoint].sigmaL(method) for ien in iens] for elem in self.res], axis=0)
        elif type == "exact":
            arr = np.sum([[elem[ien][itestPoint].exactSolution.sigmaL for ien in iens] for elem in self.res], axis=0)
        elif type == "emulatedError":
            arr = np.abs(self.getCrossSections("emulated", iens, itestPoint, method)\
                       - self.getCrossSections("exact", iens, itestPoint, method))
        else:
            raise ValueError(f"Keyword '{type}' unknown to Cross Sections.")

        return arr

    def plotCrossSections(self, plotErrors=False, showExactSol=False):
        # create new pdf for figures
        pdf = PdfPages('crossSections.pdf')

        # create new figure
        fig, ax = plt.subplots(1, 1, constrained_layout=True)
        ax.set_title(f"{self.numTrainingLecs} training "
                     + f"and {self.numTestingLecs} testing Hamiltonians")
        plotTypeSuffix = "Error" if plotErrors else ""

        # iterate over testing Hamiltonians and KVP methods
        iens = range(len(self.energies))
        for itestPoint, testPoint in enumerate(self.testingLecList):
            for imethod, method in enumerate(self.res[0][0][itestPoint].kvpsAvail):
                legendLbl = f"${method}$ KVP" if itestPoint == 0 else ""
                y = self.getCrossSections("emulated"+plotTypeSuffix, iens, itestPoint, method)
                ax.semilogy(self.energies, y, c=Scattering._colors[imethod],
                            ls=Scattering._lineStyles[itestPoint], label=legendLbl)

            # if requested, compute exact solution (without emulating)
            if showExactSol and not plotErrors:
                legendLbl = "exact" if itestPoint == 0 else ""
                yexact = self.getCrossSections("exact"+plotTypeSuffix, iens, itestPoint, None)
                ax.plot(self.energies, yexact, c="k", ls="--", lw=2, label=legendLbl)

        ax.set_xlim(self.energies[0], self.energies[-1])
        # ax.set_ylim(-30, 120)

        ax.set_xlabel(r"$E_\mathrm{cm}$ [MeV]")
        ax.set_ylabel((r"Abs. Error in " if plotErrors else "") + r"$\sigma$ [mb]")

        ax.legend(loc="best", handlelength=5)

        self.setAxRcParams(ax)
        self.setPdfMetaData(pdf)

        pdf.savefig(fig)
        pdf.close()
        plt.close()


    @staticmethod
    def setAxRcParams(ax):
        ax.xaxis.set_ticks_position('both')
        ax.tick_params(axis="x", direction="in")
        ax.yaxis.set_ticks_position('both')
        ax.tick_params(axis="y", direction="in")

    @staticmethod
    def setPdfMetaData(pdf):
        d = pdf.infodict()
        d['Title'] = 'Nuclear phaseshifts'
        d['Author'] = 'UQ Group'
        d['Subject'] = 'nuclear scattering'
        d['Keywords'] = 'scattering emulator eigenvector continuation'

def iteration(E_MeV, channel, trainingLecList, testingLecList, rmatch, potentialArgs, cacheExactSol=True):
    # some preparations
    potential = Potential(channel, **potentialArgs)
    scattExp = ScatteringExp(E_MeV=E_MeV, potential=potential)

    # generate training data
    grid = Grid(1e-6, rmatch, numIntervals=4, numPointsPerInterval=32,
                type="Gauss", test=False)

    # generate the training points (with default method)
    scattSols = RseSolver.solve(scattExp, grid, trainingLecList, asympParam="S")

    # print dependencies of wave function for different asymptotic parametrization;
    # useful for debugging (has to be independent of Hamiltonian)
    # for sol in scattSols:
    #     sol.Lmatrix.printWaveFuncDependencies()
    # return

    # train the emulator
    evc = Emulator(scattSols)

    # use trained emulator to emulate scattering problems
    results = []
    for testLecs in testingLecList:
       results.append(evc.emulate(testLecs, useLsSolver=True, usePinv=False, printResult=False,
                                  cacheExactSol=cacheExactSol))

    return results