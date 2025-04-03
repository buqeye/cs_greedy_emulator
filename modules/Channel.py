class Channel:
    def __init__(self, *, S=None, L=None, LL=None, J=None, channel=None, as_str=None):
        if as_str is not None:
            S, L, LL, J, channel = self.parse_spectNotation(as_str)
        self.S = S # spin quantum number
        self.L = L # angular momentum quantum number (ket)
        self.LL = LL # angular momentum quantum number (bra)
        self.J = J # total # angular momentum quantum number
        self.channel = channel # isospin projection times 2
        self.check()
        if as_str is not None:
            assert self.spectNotation == as_str, f"check parser of channel string: {self.spectNotation} vs {as_str}"
    
    def parse_spectNotation(self, as_str):
        """
        parses channel label (string) and converts it to numbers; 
        e.g., 1S0np --> S=0, L=LL=0, J=0, T=1, channel=0
        """
        assert len(as_str) in (5, 6), "channel string has unexpected length"
        S = (int(as_str[0]) -  1) // 2
        if len(as_str) == 5:
            L = LL = int(self._SPECTNOTATIONL_INV[as_str[1]])
        else:
            L = int(self._SPECTNOTATIONL_INV[as_str[1]])
            LL = int(self._SPECTNOTATIONL_INV[as_str[2]])
        J = int(as_str[-3])
        channel = {"np": 0, "pp": 1, "nn": -1}[as_str[-2:]]
        return S, L, LL, J, channel 
    
    @staticmethod
    def isOdd(val):
        """checks whether `val` is odd"""
        return bool(val % 2)

    def check(self):
        """
        checks that the channel is physical, obeying the Pauli principle,
        parity conservation, angular momentum coupling, isospin coupling algebra
        """
        tmp = self.S + self.T
        if not (self.isOdd(tmp + self.L) and self.isOdd(tmp + self.LL)):
            raise ValueError(f"Channel {self} is Pauli forbidden.")

        if abs(self.L - self.LL) not in (0, 2):
            raise ValueError(f"Channel {self} doesn't conserve parity.")

        if not(self.L+self.S >= self.J >= abs(self.L - self.S)):
            raise ValueError(f"Channel {self} doesn't obey angular momentum algebra.")
        
        if not(self.LL+self.S >= self.J >= abs(self.LL - self.S)):
            raise ValueError(f"Channel {self} doesn't obey angular momentum algebra.")
        
        if abs(self.channel) > self.T:
            raise ValueError(f"Channel {self} doesn't obey isospin coupling algebra.")

    @property
    def LdotS(self):
        """returns the expectation value of L cdot S"""
        return 0.5 * (self.J**2 - self.L*(self.L+1) - self.S*(self.S+1)) if self.L == self.LL else 0

    @property
    def T(self):
        """returns the isospin quantum number T"""
        return 0 if (self.L+self.S) % 2 else 1

    @property
    def g(self):
        """returns the spin degeneracy factor g"""
        return 2*self.S+1

    _SPECTNOTATIONL = {0: "S", 1: "P", 2: "D", 3: "F", 4: "G", 5: "H", 6: "I"}
    _SPECTNOTATIONL_INV = {v: k for k, v in _SPECTNOTATIONL.items()}
    
    _ISOSPIN_LBLS = {0: "np", 1: "pp", -1: "nn"}
    _ISOSPIN_LBLS_INV = {v: k for k, v in _ISOSPIN_LBLS.items()}

    @property
    def Lstr(self):
        """returns the character associated with the angular momentum (S, P, D, ...)"""
        if self.L in Channel._SPECTNOTATIONL.keys() and self.LL in Channel._SPECTNOTATIONL.keys():
            if self.L == self.LL:
                return Channel._SPECTNOTATIONL[self.L]
            else:
                return Channel._SPECTNOTATIONL[self.L] + Channel._SPECTNOTATIONL[self.LL]
        else:
            return f"$(L={self.L})$" if self.L == self.LL else f"$(L={self.L}, LL={self.LL})$"

    @property
    def spectNotation(self):
        """returns the spectral notation of the channel (string)"""
        return f"{self.g}{self.Lstr}{self.J}{self._ISOSPIN_LBLS[self.channel]}"

    @property
    def spectNotationTeX(self):
        """returns the spectral notation of the channel (TeX)"""
        return f"$^{self.g}\\mathrm{{{self.Lstr}}}_{self.J}^{{({self._ISOSPIN_LBLS[self.channel]})}}$"

    def __str__(self):
        """returns the string representation of the class"""
        return f"{self.spectNotation} [ S={self.S}; L={self.L}; LL={self.LL}; J={self.J}; T={self.T}; chan={self.channel} ]"


