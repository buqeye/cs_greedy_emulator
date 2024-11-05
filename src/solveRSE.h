#ifndef EVC_SOLVERSE_H
#define EVC_SOLVERSE_H

#include "localGt+.h"
#include <armadillo>
#include "phys_constants.h"

struct RseParams{
    double energy;
    double mu;
    double p;
    unsigned pot;
    Channel *channel;
    Lecs *lecs;

    inline double getk(){
        return sqrt(2.*mu*energy/consts::hbarc);
    }
};


void solveRSE (const arma::vec & grid, arma::mat & uMat, double & kMat, RseParams &params);

#endif //EVC_SOLVERSE_H
