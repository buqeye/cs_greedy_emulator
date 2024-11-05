#include <cstdio>
#include <cstdlib>

#include "Emulator.h"
#include "localGt+.h"
#include "phaseShift.h"


int main (int argc, char **argv){
    double energy = 50., l = 0;
    if(argc != 3){
        puts("usage: first argument 'Energy in MeV'; second argument 'l'. Using default values.");
    } else {
        energy = std::atof(argv[1]);
        l = std::atoi(argv[2]);
    }

    std::vector<Lecs> trainingPoints; trainingPoints.reserve(8);
    std::vector<Lecs> testingPoints; testingPoints.reserve(8);

    trainingPoints.push_back({5.,
            0.2,
            -0.14084,
            0.04243,
            -0.12338,
            0.11018,
            -2.11254,
            0.15898,
            -0.26994,
            0.04344,
            0.062963});

    trainingPoints.push_back({6.,
            0.2,
            -0.14084,
            0.04243,
            -0.12338,
            0.11018,
            -2.11254,
            0.15898,
            -0.26994,
            0.04344,
            0.062963});

    trainingPoints.push_back({5.,
            0.3,
            -0.14084,
            0.04243,
            -0.12338,
            0.11018,
            -2.11254,
            0.15898,
            -0.26994,
            0.04344,
            0.062963});

    trainingPoints.push_back({6.,
            0.3,
            -0.14084,
            0.04243,
            -0.12338,
            0.11018,
            -2.11254,
            0.15898,
            -0.26994,
            0.04344,
            0.062963});

    testingPoints.push_back({5.43850,
          0.27672,
          -0.14084,
          0.04243,
          -0.12338,
          0.11018,
          -2.11254,
          0.15898,
          -0.26994,
          0.04344,
          0.062963});

    // trainingPoints.push_back(testingPoints[0]);

    Emulator emu(trainingPoints, energy, l);

    double phaseShift;
    emu.emulate(&phaseShift, testingPoints[0]);

    puts("\nre-calculate using external functions");
    emulator_startSession((int)trainingPoints.size(), &trainingPoints[0], energy, l);
    emulator_emulate(&testingPoints[0], &phaseShift);
    emulator_closeSession();
    return EXIT_SUCCESS;
}
