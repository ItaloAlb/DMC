//////////////////////////////////////////////////////////////
//                                                          //
// g++ -std=c++17 -fopenmp -g -o dmc main.cpp dmc.cpp       //
// ./dmc [to run with parallelization]                      //
// OMP_NUM_THREADS=1 ./dmc [to run without parallelization] //
//                                                          //
//////////////////////////////////////////////////////////////

#include <vector>
#define _USE_MATH_DEFINES
#include <cmath>
#include <random>
#include <array>
#include <deque>
#include <omp.h>

#include <iostream>
#include <fstream>
#include <iomanip>

namespace Constants {
    const int MAX_N_WALKERS = 15000;
    const int N_WALKERS_TARGET = 5000;
    const int MAX_BRANCH_FACTOR = 3;
    const int DEFAULT_N_PARTICLE = 2;
    const int DEFAULT_N_DIM = 2;

    const double REFERENCE_ENERGY = -0.5;
    const double ALPHA = 0.5;

    const double MIN_POPULATION_RATIO = 1e-8;
    const double MIN_DISTANCE = 1e-8;
    const double FINITE_DIFFERENCE_STEP = 1e-8;
    const double FINITE_DIFFERENCE_STEP_2 = FINITE_DIFFERENCE_STEP * FINITE_DIFFERENCE_STEP;
}

struct Walker {
        std::vector<double> position;
        std::vector<double> drift;
        double localEnergy;

        Walker() = default;

        Walker(int nParticles, int dim);
};

class DMC {
    private:
        int nWalkers, nParticles, dim;
        double deltaTau, referenceEnergy, instEnergy, meanEnergy;

        std::vector<std::mt19937> gens;
        // std::normal_distribution<double> dist;
        // std::uniform_real_distribution<double> uniform;

        std::array<Walker, Constants::MAX_N_WALKERS> walkers;

        void initializeWalkers();

        std::vector<double> getDrift(const std::vector<double>& position) const;

        double getLocalEnergy(const std::vector<double>& position);

        void updateReferenceEnergy(double blockEnergy);

        // double potentialEnergy(int i) const;

        double potentialEnergy(const std::vector<double>& position) const;

        double driftGreenFunction(const std::vector<double>& newPosition, const std::vector<double>& oldPosition, const std::vector<double>& oldDrift) const;

        double branchGreenFunction(double newLocalEnergy, double oldLocalEnergy) const;
        
        double trialWaveFunction(const std::vector<double>& position) const;

        void timeStep();

        void blockStep(int nSteps);

    public: 
        DMC(double deltaTau, 
            int nWalkers = Constants::N_WALKERS_TARGET, 
            int nParticles = Constants::DEFAULT_N_PARTICLE, 
            int dim = Constants::DEFAULT_N_DIM);

        void run();
        
};