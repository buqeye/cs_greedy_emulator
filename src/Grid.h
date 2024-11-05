#ifndef EVC_GRID_H
#define EVC_GRID_H

#include <armadillo>

enum GRIDTYPE {LINEAR, GAUSSLEG};

class Grid{
    GRIDTYPE gridtype;
    double start;
    double end;
    unsigned numPoints;
    arma::vec points;
    arma::vec weights;
    arma::mat weightMat;

public:

    Grid(double start, double end, unsigned numPoints, GRIDTYPE gridtype){
        this->gridtype = gridtype;
        this->start = start;
        this->end = end;
        this->numPoints = numPoints;

        computePointsWeights();
    }

    inline auto getNp() {
        return numPoints;
    }

    inline auto & getPoints(){
        return points;
    }

    inline auto & getWeights(){
        return weights;
    }

    inline double getPointsAt(int i){
        return points[i];
    }

    inline double getWeightsAt(int i){
        return weights[i];
    }

    const inline arma::mat & getWeightMat(){
        return weightMat; // arma::randu(500, 500);
    }

    inline double at(int i){
        return points[i];
    }

    void computePointsWeights();
};

#endif //EVC_GRID_H
