#include <cmath>
#include <iomanip>
#include "Emulator.h"
#include "solveRSE.h"
#include "Grid.h"


Emulator::Emulator(std::vector<Lecs> &lecs, double energy, int l, double rEnd, unsigned numPoints,
                   double rStart) : energy(energy), p(getMomentumP(energy)),
                                    channel{0, l, l, l, 0},
                                    grid(rStart, rEnd, numPoints, LINEAR),
                                    trainingPoints(lecs),
                                    nTrainPoints(lecs.size()),
                                    deltaUtileThetaIndep(nTrainPoints,nTrainPoints, arma::fill::zeros),
                                    wfs(nTrainPoints), wfsMat(numPoints,nTrainPoints),
                                    kMats(nTrainPoints), pots(nTrainPoints){
    train();
}

void Emulator::train(){
    printf("training emulator with %lu points\n", nTrainPoints);
    // preparation
    RseParams params{energy, mu, p, 213, &channel, nullptr};

    // solve RSE exactly for all training points
    auto numPoints = grid.getNp();
    for(long unsigned i=0; i<nTrainPoints; ++i) {
        wfs[i].resize(numPoints,2);
        params.lecs = &trainingPoints[i];
        solveRSE(grid.getPoints(), wfs[i], kMats[i], params);

        arma::vec & currentPot = pots[i];
        currentPot.resize(grid.getNp());
        for(unsigned j=0; j<numPoints; ++j)
            currentPot.at(j) = Vrlocal(grid.at((int)j), 213, params.channel, params.lecs);
    }

    // for convenience build a matrix that contains the training wavefunctions as rows
    //for (long unsigned j = 0; j < nTrainPoints; ++j) wfsMat.col(j) = wfs[j].col(0);

    // compute the parts of U that only depend on the basis param sets
    // build first part of matrix Delta U tilde
    for(long unsigned m=0; m<nTrainPoints; ++m) {
        arma::mat pot = arma::diagmat(pots[m]) * grid.getWeightMat();
        for(long unsigned n=0; n<nTrainPoints; ++n) {
            deltaUtileThetaIndep.at(m,n) += computeOverlap(wfs[m].col(0), pot, wfs[n].col(0));
            //deltaUtileThetaIndep.at(m,n) += computeOverlap(wfs[m],pots[n],wfs[n]);
            // note that Eq. (9) in Furnstahl et al. is symmetric, so no need to
            // compute the overlaps for the second term (transposed matrix added below)
        }
    }

    // symmetrize matrix Delta U tilde
    deltaUtileThetaIndep += deltaUtileThetaIndep.t();
    //deltaUtileThetaIndep.print();
}

void Emulator::emulate(double *phaseShift, Lecs & lecs, bool usePinv) {
    // Computes the coefficients for a linear combination of basis wfs which emulates the desired wf.
    // Constructs the emulated wf using said linear combination.

    // sample (weighted) interaction
    auto numPoints = grid.getNp();
    arma::vec weightedVr(numPoints);
    for(unsigned j=0; j<numPoints; ++j)
        weightedVr[j] = Vrlocal(grid.at((int) j), 213, &channel, &lecs) * grid.getWeightsAt((int) j);
    arma::mat pot = arma::diagmat(weightedVr);

    // build theta-independent part of matrix Delta U tilde
    arma::cx_mat deltaUtileThetaDep(nTrainPoints, nTrainPoints, arma::fill::zeros);
    for(long unsigned m=0; m<nTrainPoints; ++m) {
        for(long unsigned n=0; n<nTrainPoints; ++n) {
            deltaUtileThetaDep.at(m,n) += computeOverlap(wfs[m].col(0), pot, wfs[n].col(0));
            //deltaUtileThetaIndep.at(m,n) += computeOverlap(wfs[m],pots[n],wfs[n]);
            // note that Eq. (9) in Furnstahl et al. is symmetric, so no need to
            // compute the overlaps for the second term (transposed matrix added below)
        }
    }

    // construct the matrix Delta U tile and its inverse
    arma::cx_mat deltaUtilde = 2. * mu * (2. * deltaUtileThetaDep - deltaUtileThetaIndep) / consts::hbarc;
    arma::cx_mat deltaUtildeInv;

    // add a nugget to regularize or use pseudo inverse
    if(usePinv)
        deltaUtildeInv = arma::pinv(deltaUtilde, nugget);
    else {
        deltaUtilde.diag() += nugget;
        deltaUtildeInv = deltaUtilde.i();
    }

    // compute the lagrange multiplier lambda
    std::complex<double> lambda = (-1. + arma::accu(deltaUtildeInv * (kMats/p))) / arma::accu(deltaUtildeInv);
    arma::cx_vec coeffs = deltaUtildeInv * (kMats/p - lambda);

    // build emulated wf
    // arma::cx_vec wfsEmu = wfsMat * coeffs;

    // compute emulated Kmat (for phase shifts)
    std::complex<double> kmatEmul = arma::as_scalar(coeffs.t() * kMats - (p/2.) * (coeffs.t() * deltaUtilde * coeffs));
    auto delta = atan(kmatEmul) * 180./M_PI;

    // print out coefficients of trial wavefunction
    std::cout << "phaseshift [deg]: " << delta << std::endl;
    printf("ci (emu) = ");
    for(auto & elem : coeffs) printf("((%lf +I %lf) ", elem.real(), elem.imag());
    printf("; sum(ci) %lf\n", (arma::accu(coeffs)).real());

    // output real part of phase shift
    *phaseShift = delta.real();
}