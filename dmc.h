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

const double TAU = 0.01;

const int MAX_N_WALKERS = 15000;
const int N_WALKERS_TARGET = 5000;

const double ELECTRON_CHARGE = -1.0;
const double ELECTRON_MASS = 1.0;

const double HOLE_CHARGE = +1.0;
const double HOLE_MASS = 1.0;

const double REFERENCE_ENERGY = -0.5;
const double ALPHA = 0.5;
const int MAX_BRANCH = 3.0;

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
        double referenceEnergy, instEnergy, meanEnergy;

        std::vector<std::mt19937> gens;
        // std::normal_distribution<double> dist;
        // std::uniform_real_distribution<double> uniform;

        std::array<Walker, MAX_N_WALKERS> walkers;

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
        DMC(int nWalkers = 5000, int nParticles = 2, int dim = 2);

        void run();
        
};