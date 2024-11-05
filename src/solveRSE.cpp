#include <stdio.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_odeiv2.h>

#include "solveRSE.h"
#include "phys_constants.h"
#include <boost/math/special_functions/bessel_prime.hpp>


// Solve second-order radial Schroedinger equation (second-order ODE)
//    -u"(r) + (l*(l+1)/r**2. + (2*mu)*(V(r) - E)) * u(r) = 0
// which can be converted into a first order system  by introducing y = u'(r),
//     u' = y
//     y' = ( l*(l+1)/r**2. + (2*mu)*(V(r) - E) ) * u(r)


int func (double r, const double uin[], double uout[], void *params) {
    auto rseParams = (RseParams *)params;
    const auto l = rseParams->channel->L;
    const auto en = rseParams->energy;
    const auto mu = rseParams->mu;

    const auto pot = rseParams->pot;
    const auto channel = rseParams->channel;
    const auto lecs = rseParams->lecs;

    const double vPot = Vrlocal(r, pot, channel, lecs) ;

    uout[0] = uin[1];
    uout[1] = ( l*(l+1)/(r*r) +
                (2.*mu) * (vPot - en) * consts::MeV
              ) * uin[0];
    return GSL_SUCCESS;
}

int jac (double r, const double u[], double *dfdu, double dfdr[], void *params) {
    auto rseParams = (RseParams *)params;
    const auto l = rseParams->channel->L;
    const auto en = rseParams->energy;
    const auto mu = rseParams->mu;

    const auto pot = rseParams->pot;
    const auto channel = rseParams->channel;
    const auto lecs = rseParams->lecs;

    const double vPot = Vrlocal(r, pot, channel, lecs);

    gsl_matrix_view dfdy_mat = gsl_matrix_view_array (dfdu, 2, 2);
    gsl_matrix *m = &dfdy_mat.matrix;
    gsl_matrix_set(m, 0, 0, 0.0);
    gsl_matrix_set(m, 0, 1, 1.0);
    gsl_matrix_set(m, 1, 0, l*(l+1)/(r*r) + (2.*mu) * (vPot - en) * consts::MeV);
    gsl_matrix_set(m, 1, 1, 0.0);
    dfdr[0] = 0.0;
    dfdr[1] = 0.0;
    return GSL_SUCCESS;
}


void solveRSE (const arma::vec & grid, arma::mat & uMat, double & kMat, RseParams &params) {
    const gsl_odeiv2_step_type *T = gsl_odeiv2_step_rkf45;

    gsl_odeiv2_step *s = gsl_odeiv2_step_alloc(T, 2);
    gsl_odeiv2_system sys = {func, jac, 2, &params};

    double r = grid[0];
    double u[2] = {0.0, 1.0}, u_err[2];
    double dudt_in[2], dudt_out[2];

    uMat.reshape(grid.size(), 2);
    uMat.at(0, 0) = u[0];
    uMat.at(0, 1) = u[1];

    GSL_ODEIV_FN_EVAL(&sys, r, u, dudt_in);
    //printf("%.5lf | %.5lf %.5lf\n", r, u[0], u[1]);

    // TODO: it's much better to use an adaptive solver instead [especially for large rmatch]
    for (arma::uword i = 0; i < grid.size() - 1; ++i) {
        double stepSize = grid[i + 1] - grid[i];
        int status = gsl_odeiv2_step_apply(s, r, stepSize, u, u_err, dudt_in, dudt_out, &sys);
        if (status != GSL_SUCCESS) break;

        dudt_in[0] = dudt_out[0];
        dudt_in[1] = dudt_out[1];

        r = grid[i + 1];

        //printf("%.5lf | %.5lf %.5lf\n", r, u[0], u[1]);
        uMat.at(i + 1, 0) = u[0];
        uMat.at(i + 1, 1) = u[1];
    }

    gsl_odeiv2_step_free(s);

    // matching to free solution (e.g., asymptotic limit)
    unsigned lastGridP = grid.size()-1;
    double rmatch = grid.at(lastGridP);
    auto p = params.p;
    double rho = p * rmatch;
    auto l = params.channel->L;

    double F = rho * boost::math::sph_bessel(l, rho);
    double G = -rho * boost::math::sph_neumann(l, rho);

    double Fprime = (F / rho + rho * boost::math::sph_bessel_prime(l, rho)) * p;
    double Gprime = (G / rho - rho * boost::math::sph_neumann_prime(l, rho)) * p;

    // dertermine R and K matrix
    double Rmat = (uMat.at(lastGridP,0) / uMat.at(lastGridP,1)) / rmatch ;
    double Kmat = -(F - rmatch * Rmat * Fprime) / (G - rmatch * Rmat * Gprime);

    // rescale the wave function and its derivative
    double psi_an = (F + Kmat * G) / p;
    double scale = psi_an / uMat.at(lastGridP, 0);
    uMat *= scale;
    kMat = Kmat;
}