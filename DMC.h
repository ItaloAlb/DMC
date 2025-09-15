#ifndef DMC_H
#define DMC_H

#include <vector>

const double TAU = 0.001;
const double SQRT_TAU = sqrt(TAU);

const int MAX_N_WALKERS = 10000;
const int N_WALKERS_TARGET = 5000;
const int MAX_N_PARTICLES = 2;

const double ELECTRON_CHARGE = -1.0;
const double ELECTRON_MASS = 1.0;

const double HOLE_CHARGE = +1.0;
const double HOLE_MASS = 1.0;

const double REFERENCE_ENERGY = -1.0;

void gauss(double& g1, double& g2);

struct Walker {
        bool isAlive;
        std::vector<double> position;
        std::vector<double> drift;
        double localEnergy;
        double oldLocalEnergy;

        Walker() = default;

        Walker(int nParticles, int dim);
};

struct Particle {
    double mass;
    double charge;
};


class DMC {
    private:
        int nWalkers, nParticles, dim;

        double maxLocalEnergy, minLocalEnergy, lastStepEnergy, refEnergy, a, b;

        Walker walkers[MAX_N_WALKERS];

        std::vector<Particle> particles;

        void initializeWalkers();

        void walk();

        void branch();

        void updateDrift(int i);

        void updateLocalEnergy(int i);

        void updateReferenceEnergy();

        double potentialEnergy(int i);

        double jastrowLaplacian(int i);

        double jastrowGradSquared(int i);

        double trialWaveFunction(int i);

    public: 
        DMC(int nWalkers = 1000, int nParticles = 2, int dim = 2);

        int getNumberWalkers();

        void timeStep();

        void blockStep(int nSteps);

        double meanLocalEnergy() const;

        
};

#endif // DMC_H