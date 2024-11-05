#include "Grid.h"

void Grid::computePointsWeights(){
    if(gridtype == LINEAR){
        // points
        points = arma::linspace(start, end, numPoints);

        // weights
        // Riemann sum (decide whether upper or lower sum is used)
        // assume that grid can be arbitrary (support not only linspaces)
        weights = arma::diff(points);
        weights.resize(numPoints); weights[numPoints-1]=0.; // arma::reshape() would work too but is less flexible
    } else{
        puts("not supported");
        exit(EXIT_FAILURE);
    }

    weightMat = arma::diagmat(weights);
}