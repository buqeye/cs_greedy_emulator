#ifndef EVC_EMULATOR
#define EVC_EMULATOR

#include <armadillo>
#include "localGt+.h"
#include "phys_constants.h"
#include "Grid.h"


class Emulator {
private:
    double mu = consts::mNeutron/2.;
    double nugget = 1e-10;
    double energy;
    double p;
    Channel channel;
    Grid grid;
    std::vector<Lecs> & trainingPoints;
    long unsigned nTrainPoints;

    arma::mat deltaUtileThetaIndep;
    std::vector<arma::mat> wfs; // stores value and derivative
    arma::mat wfsMat;
    arma::vec kMats;
    std::vector<arma::vec> pots;

    // For computing matrix elements of the potential in the EVC basis.
    static inline double computeOverlap(const arma::vec & psi1, const arma::mat & op,
                                 const arma::vec & psi2) {
        return arma::as_scalar( psi1.t() * op * psi2 ) ;
    }

    inline double getMomentumP(double aenergy){
        return sqrt(2.*mu*aenergy/consts::hbarc);
    }

    void train();
public:
    void emulate(double *phaseShift, Lecs & lecs, bool usePinv=false);

    explicit Emulator(std::vector<Lecs> & lecs, double energy=20., int l=0,
             double rEnd=20., unsigned numPoints=500, double rStart=1e-6);
};


#endif //EVC_EMULATOR
