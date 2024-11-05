import sys
sys.path.append("./modules")

import argparse
import numpy as np

from Potential import Potential
from Channel import Channel
from Scattering import Scattering


def main():
    """
    Test runs for Emulator
    """
    # init parser
    parser = argparse.ArgumentParser(description='An efficient emulator for nuclear scattering and transfer reactions',
                                     epilog="Please report bugs and other issues.")

    # add arguments to parser
    parser.add_argument('-rm', '--rmatch', metavar="r", action='store', default=15., type=float,
                        help="sets value for `rmatch` in fm for matching to the free solution (e.g., asymptotic limit)")


    group = parser.add_mutually_exclusive_group()
    group.add_argument('-en', '--energy', metavar="Ecm", action='store', type=float,
                        help="sets the center-of-mass energy `Ecm` in MeV")
    group.add_argument('-er', '--energyRange', metavar=("START", "END", "N"), nargs=3, action='store',
                       default=[0.01, 40, 10],
                       help="sets the linear space from [start, end] with N points \
                       for the center-of-mass energies `Ecm` in MeV")

    parser.add_argument("-p", "--potential", action='store', choices=['chiral', 'woodssaxon', 'optical', 'minnesota'], # dest="potential",
                        default='chiral', type=str,
                        help="specifies the nuclear potential")

    group = parser.add_mutually_exclusive_group()
    group.add_argument('-l', action='store', type=int,
                        help="sets a specific value for the angular momentum requested (in single channel mode)")
    group.add_argument('-lmax', action='store', default=2, type=int,
                        help="sets the range [0, lmax] for the angular momenta requested (in single channel mode)")

    parser.add_argument("-cs", "--crossSections", action="store_true", help="plot total cross sections")
    parser.add_argument("-dcs", "--diffCrossSections", action="store_true", help="plot differential cross sections")
    parser.add_argument("-err", "--errorsOnly", action="store_true", help="plot absolute errors only")

    parser.add_argument("-v", "--verbose", action="store_true", help="prints brief status report")

    # process arguments from cli [we could add here validation of the user input provided]
    args = parser.parse_args()

    potentialArgs = {"label": args.potential, "kwargs": {"potId": 213}}
    if args.energy:
        energies = [args.energy]
    else:
        energies = np.linspace(float(args.energyRange[0]), float(args.energyRange[1]), int(args.energyRange[2]))

    if args.l is not None:
        lvalues = [args.l]
    else:
        lvalues = list(range(0, args.lmax+1))

    trainingLecList, testingLecList = Potential.getSampleLecs(potentialArgs["label"])
    channels = [Channel(S=0, L=l, LL=l, J=l, channel=0) for l in lvalues]

    if args.verbose:
        print("status report:")
        print("l: ", lvalues)
        print("energies: ", energies)
        print("channels: ", channels)

    scatt = Scattering(args.rmatch, energies, channels, trainingLecList, testingLecList, potentialArgs, cacheExactSol=True)

    scatt.plotPhaseShifts(plotErrors=args.errorsOnly, showExactSol=True, showLsFit=False)
    if args.diffCrossSections:
        scatt.plotDiffCrossSections(plotErrors=args.errorsOnly, showExactSol=True)
    if args.crossSections:
        scatt.plotCrossSections(plotErrors=args.errorsOnly, showExactSol=True)


if __name__ == "__main__":
    main()
