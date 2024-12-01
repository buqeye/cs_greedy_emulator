import numpy as np
import pytest  
import sys
sys.path.append("./modules")


def test_chiral_affine():
    from Potential import chiral, chiral_affine, chiralms, chiralms_affine
    from Channel import Channel
    from glob import glob
    from itertools import product 
    import yaml
    
    Jmax = 6
    S_arr = range(2)
    L_arr = range(Jmax+2)
    LL_arr = range(Jmax+2)
    J_arr = range(Jmax+1)
    mT_arr = range(-1,2)

    lec_files = glob('data/localGT+_*.yaml')
    lec_lbls = ("CS", "CT", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "CNN", "CPP")

    for lec_file in lec_files:
        with open(lec_file, 'r') as stream: 
            lecs = yaml.safe_load(stream)
            lec_vec = np.array([1.] + [lecs[elem] for elem in lec_lbls])
        
        for (S, L, LL, J, mT) in product(S_arr, L_arr, LL_arr, J_arr, mT_arr):
            try:
                channel = Channel(S=S, L=L, LL=LL, J=J, channel=mT)
            except ValueError:
                continue  # ignore unphysical pw channel
                
            for r in np.random.uniform(low=1e-3, high=4., size=100):
                aff_decomp = chiral_affine(r, channel, potId=lecs["potId"])
                potval_via_aff_decomp = aff_decomp @ lec_vec
                potval = chiral(r, channel, use_default_lec_values=False, **lecs)
                assert np.allclose(potval_via_aff_decomp, potval, atol=1e-11, rtol=0.), \
                    f"Coordinate Space: difference (aff. vs. nonaffine evaluation): {potval_via_aff_decomp - potval:e} MeV  |  " + \
                    f"LEC file: {lec_file}"
                potval_default = chiral(r, channel, use_default_lec_values=True, potId=lecs["potId"])
                assert np.allclose(potval, potval_default, atol=1e-11, rtol=0.), \
                    f"Coordinate Space: difference (default LEC values vs externally given LECs values: {potval_via_aff_decomp - potval_default:e} MeV  |  " + \
                    f"LEC file: {lec_file} | Channel: {(S, L, LL, J, mT)}"

            for k, kk in np.random.uniform(low=1e-3, high=8., size=(10,2)):
                aff_decomp = chiralms_affine(k, kk, channel, potId=lecs["potId"])
                potval_via_aff_decomp = aff_decomp @ lec_vec
                potval = chiralms(k, kk, channel,use_default_lec_values=False, **lecs)
                assert np.allclose(potval_via_aff_decomp, potval, atol=1e-6, rtol=0.), \
                    f"Momentum Space: difference (aff. vs. nonaffine evaluation): {potval_via_aff_decomp - potval:e} MeV  |  " + \
                    f"LEC file: {lec_file} | Channel: {(S, L, LL, J, mT)}"
                potval_default = chiralms(k, kk, channel, use_default_lec_values=True, potId=lecs["potId"])
                assert np.allclose(potval, potval_default, atol=1e-11, rtol=0.), \
                    f"Momentum Space: difference (default LEC values vs externally given LECs values: {potval_via_aff_decomp - potval_default:e} MeV  |  " + \
                    f"LEC file: {lec_file} | Channel: {(S, L, LL, J, mT)}"

# @pytest.fixture(scope='class', autouse=True)  
# def params(l=100, E_MeV=50, rmatch=12, rmin=1e-2, inhomogeneous=True):
#     potentialArgs = {"label": "minnesota", "kwargs": {"potId": 213}}
#     from modules.Channel import Channel
#     from modules.Potential import Potential
#     channel = Channel(S=0, L=l, LL=l, J=l, channel=0)
#     potential = Potential(channel, **potentialArgs)
#     trainingLecList, testingLecList = Potential.getSampleLecs(potentialArgs["label"])

#     from modules.ScatteringExp import ScatteringExp
#     scattExp = ScatteringExp(E_MeV=E_MeV, potential=potential)

#     from modules.Grid import Grid
#     grid = Grid(rmin, rmatch, numIntervals=1, numPointsPerInterval=10000,
#                 type="linear", test=False) 
    
#     params = {"grid": grid, "scattExp": scattExp, "potential": scattExp.potential,
#               "lecs": testingLecList[0], "inhomogeneous": True}
#     return params


# class Test_numerov_gs:
#     def test_g_s(self, params):
#         """
#         tests whether `g` and `s` give the same results as `g_s` (combined evaluation)
#         """
#         from modules.RseSolver import g, s, g_s
#         grid = params["grid"]
#         g_ret = g(grid.points, params)
#         s_ret = s(grid.points, params)
#         gs_ret = g_s(grid.points, params)

#         allclose_args = {"rtol": 0., "atol": 1e-14}
#         assert np.allclose(g_ret, gs_ret[0], **allclose_args), "g incorrect"
#         assert np.allclose(s_ret, gs_ret[1], **allclose_args), "s incorrect"
        
#     def test_g_s_affine(self, params):
#         grid = params["grid"]
#         from modules.RseSolver import  g_s, g_s_affine
#         g_ret, s_ret = g_s(grid.points, params)
#         g_arr, s_arr = g_s_affine(grid.points, params)

#         allclose_args = {"rtol": 0., "atol": 1e-8}
#         lecs = params["potential"].lec_array_from_dict(params["lecs"])
#         assert np.allclose(g_ret, g_arr @ lecs, **allclose_args), "g incorrect"
#         assert np.allclose(s_ret, s_arr @ lecs, **allclose_args), "s incorrect"

# # test potential affiness


# def test_numerov_fom():
#     from Potential import Potential
#     from Channel import Channel
#     from ScatteringExp import ScatteringExp
#     from Grid import Grid
#     import RseSolver
#     potentialArgs = {"label": "minnesota", "kwargs": {"potId": 213}}
#     inhomogeneous=False
#     E_MeV = 50
#     for l in range(4):
#         channel = Channel(S=0, L=l, LL=l, J=l, channel=0)
#         potential = Potential(channel, **potentialArgs)
#         trainingLecList, testingLecList = Potential.getSampleLecs(potentialArgs["label"])

#         scattExp = ScatteringExp(E_MeV=E_MeV, potential=potential)

#         # generate training data
#         grid = Grid(1e-12, 12, numIntervals=1, numPointsPerInterval=1000,
#                     type="linear", test=False) 
        
#         num = RseSolver.solve(scattExp, grid, testingLecList, 
#                                 method="Numerov_class", inhomogeneous=inhomogeneous,
#                                 asympParam="K", matching=True)
#         num2 = RseSolver.solve(scattExp, grid, testingLecList, 
#                                 method="Numerov", inhomogeneous=inhomogeneous,
#                                 asympParam="K", matching=True)
#         numk = RseSolver.solve(scattExp, grid, testingLecList, 
#                                 method="RK45", inhomogeneous=inhomogeneous,
#                                 asympParam="K", matching=True)
        
#         allclose_args = {"rtol": 0., "atol": 1e-8}
#         assert np.allclose(np.abs(num[0].u-num2[0].u), 0, **allclose_args)
#         assert np.allclose(np.abs(numk[0].u-num2[0].u), 0, **allclose_args)


if __name__ == "__main__":
    test_chiral_affine()