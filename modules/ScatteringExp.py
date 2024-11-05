import numpy as np
from constants import *


class ScatteringExp:
    def __init__(self, E_MeV, potential, *, Z=0, A=1, Aproj=1):
        """
        represents a scattering experiment

        Parameters:
        ----------_
        E_MeV: energy in MeV
        potential: 
        Z: proton number
        A: mass number of the target
        Aproj: mass number of the projectile
        """
        self.en_MeV = E_MeV
        self.potential = potential
        self.Z = Z
        self.A = A
        self.Aproj = Aproj

        self.en = self.en_MeV/hbarc
        self.mu = self._getMu()
        self.p = self._getP()

    @property
    def l(self):
        """returns the angular momentum quantum number"""
        return self.potential.channel.L

    def _getMu(self):
        """returns the reduced mass \mu"""
        M1 = self.Aproj * mNeutron
        M2 = self.A * mNeutron
        return (M1 * M2) / (M1 + M2)

    def _getP(self):
        """returns the momentum corresponding to the given energy"""
        return np.sqrt(2.*self.mu*self.en)

    def __str__(self):
        """returns the string representation of the class"""
        return f"ScatteringExp A={self.Aproj}-->(A={self.A}, Z={self.Z}) at {self.en_MeV} MeV"

