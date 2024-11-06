import numpy as np

import chiralPot
from constants import *


class Potential:
    def __init__(self, channel, **kwargs):
        self.channel = channel
        self.name = kwargs["label"]
        self.kwargs = kwargs["kwargs"]
        if self.name == 'chiral':
            self.potentialFunction = chiral
        elif self.name == 'woodssaxon':
                self.potentialFunction = woodssaxon
        elif self.name == 'optical':
                self.potentialFunction = optical
        elif self.name == 'minnesota':
                self.potentialFunction = minnesota
                self.lecLabels = ("V0", "V1", "K0", "K1")
                self.lecAffineLabels = ("V0", "V1")
                self.lecBaseValues = {"V0": 200, "V1": -91.85, "K0": 1.487, "K1": 0.465}
        else:
            raise ValueError(f"Potential '{self.name}' unknown.")
        _, testingLecs = self.sampleLecs
        self.parameter_names = list(testingLecs[0].keys())

    def eval(self, r, lecs):
        if isinstance(r, np.ndarray):
            # return np.array([ self.potentialFunction(rval, **lecs) / hbarc for rval in r])
            return np.array(list(map(lambda rval: self.potentialFunction(rval, self.channel, **lecs, **self.kwargs), r)),
                            dtype=np.double) / hbarc
        elif isinstance(r, float):
            return self.potentialFunction(r, self.channel, **lecs, **self.kwargs) / hbarc
        else:
            raise ValueError("Check input parameters of 'Potential.eval()'.")

    def evalAffine(self, r):
        if self.name != "minnesota":
            raise NotImplementedError
        Vcomp = [np.zeros_like(r)]
        r2 = r ** 2
        K_arr = [1.487, 0.465]
        for K in K_arr:
            Vcomp.append(np.exp(-K * r2))
        return np.array(Vcomp).T  / hbarc

    def lec_array_from_dict(self, lecList_dict):
        if self.name == "minnesota":
            return np.array([[1.] +[ lecs_dict[lbl] for lbl in self.lecAffineLabels] for lecs_dict in lecList_dict])
        else:
            raise NotImplementedError

    def get_lec_dict(self, lecList_arr):
        if self.name == "minnesota":
            ret = []
            for lecs in lecList_arr:
                if isinstance(lecs, dict):
                    ret.append(lecs)
                else:
                    ret.append({lec_lbl: lec for lec_lbl, lec in zip(self.lecAllLabels, lecs)})
            return ret
        else:
            raise NotImplementedError

    @property
    def sampleLecs(self):
        return Potential.getSampleLecs(self.name)

    def getLecsSample(self, lecs_variation, n=100, seed=123, 
                      mode="random", as_dict=True):
        lecs_lbl_to_be_varied = list(lecs_variation.keys())
        d = len(lecs_lbl_to_be_varied)
        bounds = np.array(list(lecs_variation.values()))     
        if mode == "random":
            from scipy.stats import qmc
            sampler = qmc.LatinHypercube(d=d, seed=seed)
            samples = qmc.scale(sampler.random(n), bounds[:,0], bounds[:,1])
        elif mode == "linear":
            lin_spaces = (np.linspace(bounds[i,0], bounds[i,1], n) for i in range(len(lecs_lbl_to_be_varied)))
            samples = np.array(np.meshgrid(*lin_spaces, indexing='ij')).T.reshape(-1,len(lecs_lbl_to_be_varied))
        else:
            raise ValueError(f"mode {mode} unknown to `getLecsSample()`")
        
        ret = [{**self.lecBaseValues, 
                **{key: samples[j, ikey] for ikey, key in enumerate(lecs_lbl_to_be_varied)}} for j in range(len(samples))]

        return ret if as_dict else self.lec_array_from_dict(ret)
    

    @staticmethod
    def getSampleLecs(potLbl, A=1):
        """
        Parameters
        ----------
        potential

        Returns
        -------
        LECs for some training and testing
        """

        trainingLecs = []
        testingLecs = []

        if potLbl == 'chiral':
            trainingLecs.append({"CS": 5., "CT": 0.2, "C1": -0.14084, "C2": 0.04243,
            "C3": -0.12338, "C4": 0.11018, "C5": -2.11254,
            "C6": 0.15898, "C7": -0.26994, "CNN": 0.04344, "CPP": 0.062963})

            trainingLecs.append({"CS": 6., "CT": 0.2, "C1": -0.14084, "C2": 0.04243,
            "C3": -0.12338, "C4": 0.11018, "C5": -2.11254,
            "C6": 0.15898, "C7": -0.26994, "CNN": 0.04344, "CPP": 0.062963})

            trainingLecs.append({"CS": 5., "CT": 0.3, "C1": -0.14084, "C2": 0.04243,
            "C3": -0.12338, "C4": 0.11018, "C5": -2.11254,
            "C6": 0.15898, "C7": -0.26994, "CNN": 0.04344, "CPP": 0.062963})

            # for i in range(1000):
            trainingLecs.append({"CS": 6., "CT": 0.3, "C1": -0.14084, "C2": 0.04243,
            "C3": -0.12338, "C4": 0.11018, "C5": -2.11254,
            "C6": 0.15898, "C7": -0.26994, "CNN": 0.04344, "CPP": 0.062963})

            testingLecs.append({"CS": 5.43850, "CT": 0.27672, "C1": -0.14084, "C2": 0.04243,
            "C3": -0.12338, "C4": 0.11018, "C5": -2.11254,
            "C6": 0.15898, "C7": -0.26994, "CNN": 0.04344, "CPP": 0.062963})

            testingLecs.append({"CS": 5.53850, "CT": 0.37672, "C1": -0.14084, "C2": 0.04243,
            "C3": -0.12338, "C4": 0.11018, "C5": -2.11254,
            "C6": 0.15898, "C7": -0.26994, "CNN": 0.04344, "CPP": 0.062963})

        elif potLbl == 'woodssaxon':
            # trainingLecs.append({"depth": 40, "radius": 3, "diffuseness": 0.5})
            # trainingLecs.append({"depth": 60, "radius": 3, "diffuseness": 0.5})
            # trainingLecs.append({"depth": 40, "radius": 4, "diffuseness": 0.5})
            # trainingLecs.append({"depth": 60, "radius": 4, "diffuseness": 0.5})
            # # trainingLecs.append({"depth": 55, "radius": 3, "diffuseness": 0.5})
            #
            # testingLecs.append({"depth": 50, "radius": 3, "diffuseness": 0.5})

            tmp = [{'diffuseness': 0.303, 'depth': 66.69, 'radius': 4.018},
             {'diffuseness': 0.549, 'depth': 58.072, 'radius': 3.611},
             {'diffuseness': 0.667, 'depth': 48.636, 'radius': 3.995},
             {'diffuseness': 0.4, 'depth': 66.28, 'radius': 3.981},
             {'diffuseness': 0.35, 'depth': 60.526, 'radius': 3.928},
             {'diffuseness': 0.372, 'depth': 34.963, 'radius': 3.542},
             {'diffuseness': 0.413, 'depth': 35.917, 'radius': 3.584},
             {'diffuseness': 0.604, 'depth': 43.708, 'radius': 3.795}]
            for elem in tmp:
                trainingLecs.append(elem)

            tmp = [{'diffuseness': 0.628, 'depth': 39.358, 'radius': 3.253}]
            for elem in tmp:
                testingLecs.append(elem)

        elif potLbl == 'optical':
            tmp = 1.2 * np.cbrt(A)
            trainingLecs.append({"V": 45, "R": tmp, "a": 0.65, "Vw": 5, "Rw": tmp, "aw": 0.65})
            trainingLecs.append({"V": 55, "R": tmp, "a": 0.65, "Vw": 5, "Rw": tmp, "aw": 0.65})
            trainingLecs.append({"V": 45, "R": tmp, "a": 0.65, "Vw": 15, "Rw": tmp, "aw": 0.65})
            trainingLecs.append({"V": 55, "R": tmp, "a": 0.65, "Vw": 15, "Rw": tmp, "aw": 0.65})

            testingLecs.append({"V": 50, "R": tmp, "a": 0.65, "Vw": 10, "Rw": tmp, "aw": 0.65})

        elif potLbl == "minnesota":
            #trainingLecs.append({"V0": 0, "V1": -291.85, "K0": 1.487, "K1": 0.465})
            #trainingLecs.append({"V0": 100, "V1": 8.15, "K0": 1.487, "K1": 0.465})
            #trainingLecs.append({"V0": 300, "V1": -191.85, "K0": 1.487, "K1": 0.465})
            #trainingLecs.append({"V0": 300, "V1": 8.15, "K0": 1.487, "K1": 0.465})

            #testingLecs.append({"V0": 200, "K0": -91.85, "V1": 1.487, "K1": 0.465})

            #trainingLecs.append({"V0": 300, "V1": 8.15, "K0": 1.487, "K1": 0.465})

            trainingLecs.append({"V0": 0,   "V1": -291.85, "K0": 1.487, "K1": 0.465})
            trainingLecs.append({"V0": 100, "V1": 8.15, "K0": 1.487, "K1": 0.465})
            trainingLecs.append({"V0": 300, "V1": -191.85, "K0": 1.487, "K1": 0.465})
            trainingLecs.append({"V0": 300, "V1": 8.15, "K0": 1.487, "K1": 0.465})

            testingLecs.append({"V0": 200, "V1": -91.85, "K0": 1.487, "K1": 0.465})


        else:
            raise ValueError(f"Potential '{potLbl}' unknown.")

        return trainingLecs, testingLecs


def woodssaxon(x, chan, **kwargs):
    return -kwargs["depth"] / (1. + np.exp((x - kwargs["radius"]) / kwargs["diffuseness"]))


def woodssurface(x, chan, **kwargs):
    expon = np.exp(-(x - kwargs["radius"]) / kwargs["diffuseness"])

    return (-kwargs["depth"] * 4. * expon) / (1. + expon) ** 2.


def minnesota(x, chan, **kwargs):
    x2 = x ** 2
    exp0 = -kwargs["K0"] * x2
    exp1 = -kwargs["K1"] * x2

    # prevent overflow due to the exp(x**2) terms
    #if (np.abs(np.array([exp0, exp1])) > 8.).any():
    #    return 0.
    #else:
    return kwargs["V0"] * np.exp(exp0) + kwargs["V1"] * np.exp(exp1)


def minnesota_full(x, chan, **kwargs):
    u = 1.
    PijSig = 2 * chan.S * (chan.S+1) - 3
    PijR = (-1) ** chan.L
    exp_terms = (chan.L == chan.LL) * np.exp(-x ** 2 * np.array([kwargs[lbl] for lbl in ("K0", "K1", "K2")]))  
    return np.array([1, 0.5 * (1.+PijSig), 0.5 * (1+PijSig)]) @ exp_terms * (u/2. + (2. - u)/2. * PijR)
    # page 133 https://tuprints.ulb.tu-darmstadt.de/5649/7/thesis_krueger_printed.pdf  
    # https://pdf.sciencedirectassets.com/271584/1-s2.0-S0375947400X06835/1-s2.0-0375947477900070/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjED8aCXVzLWVhc3QtMSJHMEUCIQCPd3uptolLp2%2FmbqxiEnUljW2izYgwBw4Xi73z5hXAMQIgdGMhEbE8h4XdlS5UXiKeqno9DaetU7UvQDCBJl6qWZwqvAUIl%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FARAFGgwwNTkwMDM1NDY4NjUiDJQMy1AhT5dO79mY4iqQBeZy%2BpmG2iw9Lkc7aBU%2FPRF1bVhzzlaBREBQeHR17rFTFsZ1Ntubk4pcubCCdpeOXv4EgQpVFsRqDc11pHcUsRytzegy4fDv%2BaAN%2BWk6Mqzzs8%2F0ekIYpiNPZ8nHSqAF2fQv7N5skPfkYpArMTWqMGajCpTptECDq%2Fzlz1Vn7ivAMjeVxMPgNW1k4YcjhpSc32THN%2FjZftp9sEoZnkGoy7IqnBpUp2FtMP5NnWtEQ5FqBxczvkmnVUrkae2t4QkDpiAy26C5kVpvurQ0VLe0SFpWBeVFyyU2HbR1sMZIyyHCJ7pmvuMBXxGv7%2F03E%2FxS9YqTgErrvyQuxQ4Y0VJvSaiRWD6IVjh5mMooP11OQeqBMWOXja%2BbsbTPFT2338r7e9T%2FRulTH38Dg5quGPeAQsW4eCWBYe8onnylnvP3CzQzRjm0WL8Xt6FBAvuCkCgryq4M8BfBzDkVbFuwWpcfTBoQ06daU0BrIOL0gPRRPGs2uZVqGLFbEe%2F4%2BgVzYtTDse4moGznUCOkIjZMnTj0xE0asNKqf%2BAdUEyA3%2FsyUdv2JOc1K1TaomzSnk565OmT27qs1VEpqXEFdqEzQYESIRUk%2BwzO2tr2KGrSTME2HbK2W5xBQW8cOldo8HbRrfKFlp06qIv2eiuXzhQmU8KKJr102d8rKxNcCJalt%2BXe7HnVjeozSOaKv1zgmCo84TQT%2BmqZt1uaQCY0nVFLGsGj3uhJivVMXfLmV6tbERi1KUkJpH0g0%2BFA27RCsDcU99qt9G2Uo0x%2BK05zotJVGxicX%2FMabheI3hPxZSbo7Ad6wcTs%2B1u7E1%2BYgVwwFoEEg25rGCt3vjcm2Y8sElv63wOhaPw6XVlGU2qziwkGJMvsd2V9MJ%2FIprgGOrEBwFGlm1C7m4WEhKdcNd89uNSRFzOQdZVI0QHuw%2Bv4qcl8snOFh2qTWXpu1YaLrCEQNVETu%2Fw%2FbQPsm9e7%2BMPYV%2FR7g3JIaZpxFn5LL26kk1TfSWj76FiE7G4lIprBX3wAUSpuMdIY9D8lhsV2RCwCqHG7Ad4VHZyuN5UkQ2j%2FrZxnnCNvhvZxA%2FhAcyF%2FuorI4qVgKrmGj%2BrnjuuwQNTw8sv%2BWNduOAew56aYOWOH%2FZqq&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20241011T230826Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTYVJWICFV2%2F20241011%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=6dcad81573da3774fa7f90ef1f98b842514cfed489d979c190892bf88162761b&hash=cd07d672cd5b4d5a050ab7fcd9e8694331864c8c1115644ef12f7981dd885261&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=0375947477900070&tid=spdf-c1d987b1-1e71-440a-927d-26c8b6819bbc&sid=e2340e23821ca14ab57937527006cd7c652egxrqa&type=client&tsoh=d3d3LXNjaWVuY2VkaXJlY3QtY29tLnByb3h5LmxpYnJhcnkub2hpby5lZHU%3D&ua=13125904575256515053&rr=8d128316dfc61233&cc=us
    # https://arxiv.org/pdf/1106.3557


def chiral(x, chan, **kwargs):
    channel = chiralPot.Channel(S=chan.S, L=chan.L, LL=chan.LL, J=chan.J, channel=chan.channel)
    potId = kwargs["potId"]  # 213  # [order][cutoff][sfr cutoff]
    lecs = chiralPot.Lecs(kwargs["CS"], kwargs["CT"], kwargs["C1"], kwargs["C2"], kwargs["C3"], kwargs["C4"],
                          kwargs["C5"], kwargs["C6"], kwargs["C7"], kwargs["CNN"], kwargs["CPP"])
    return chiralPot.Vrlocal(x, potId, channel, lecs)

def chiral_affine(x, chan, **kwargs):
    channel = chiralPot.Channel(S=chan.S, L=chan.L, LL=chan.LL, J=chan.J, channel=chan.channel)
    potId = kwargs["potId"]  # 213  # [order][cutoff][sfr cutoff]
    ret = np.zeros(12, dtype=np.double)
    chiralPot.Vrlocal_affine(x, potId, channel, ret)
    return ret

def chiral_lec_trafo_matrix():
    """
    basis (CS, CT, C1, C2, ..., C7, CNN, CPP)
    """
    mat = np.zeros((11,11))
    mat[0, :2] = [1,1]
    mat[1, :2] = [1,-3]
    mat[2, 2:6] = [1,-3, 1, -3]
    mat[3, 7:9] = [1, -3]
    mat[4, 2:6] = [1,1, -3, -3]
    mat[5, 2:6] = 4*[1]
    mat[6, 2:6] = [1,-3, -3, 9]
    mat[7, 6] =  1/2
    mat[8, 7:9] = [1, 1]
    mat[9, 9] = 1
    mat[10, 10] = 1
    return mat

def optical(x, chan, **kwargs):
    kwargsV = {"depth": kwargs["V"], "radius": kwargs["R"], "diffuseness": kwargs["a"]}
    kwargsW = {"depth": kwargs["Vw"], "radius": kwargs["Rw"], "diffuseness": kwargs["aw"]}
    return complex(woodssaxon(x, chan, **kwargsV),
                   woodssaxon(x, chan, **kwargsW))
