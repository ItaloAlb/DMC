#include "dmc.h"

Walker::Walker(int nParticles, int dim)
    : localEnergy(0.0)
{
    position.resize(nParticles * dim, 0.0);
	drift.resize(nParticles * dim, 0.0);
}

DMC::DMC(double deltaTau_, int nWalkers_, int nParticles_, int dim_)
    : deltaTau(deltaTau_),
      nWalkers(nWalkers_) ,
      nParticles(nParticles_),
      dim(dim_),
      referenceEnergy(REFERENCE_ENERGY),
      instEnergy(0.0),
      meanEnergy(0.0)
{
    int nThreads = omp_get_max_threads();
    gens.resize(nThreads);

    std::random_device rd;
    for (int i = 0; i < nThreads; i++)
    {
        gens[i].seed(rd() + i);
    }
    
    initializeWalkers();
}

void DMC::timeStep(){
    // Create a temporary array to store the new generation of walkers
    std::array<Walker, MAX_N_WALKERS> newWalkers;
    // Counter for the number of walkers in the new generation
    int newNWalkers = 0;
    // Accumulator for the total local energy of the new ensemble
    double ensembleEnergy = 0.0;
    #pragma omp parallel reduction(+:ensembleEnergy)
    {
        int threadId = omp_get_thread_num();
        auto& gen = gens[threadId];
        std::normal_distribution<double> dist(0.0, std::sqrt(deltaTau));
        std::uniform_real_distribution<double> uniform(0.0, 1.0);

        // Iterate over each walker in the current ensemble
        #pragma omp for
        for(int i = 0; i < nWalkers; i++) {
            // Create a new position vector, initialized with the current walker's position
            std::vector<double> newPosition = walkers[i].position;

            // Propose a new position for the walker using a random walk step and drift term
            for(int j = 0; j < nParticles * dim; j++){
                double chi = dist(gen); // Random number from a normal distribution (diffusion term)
                // Update the position component: newPosition = oldPosition + diffusion_term + drift_term * time_step
                newPosition[j] += chi + deltaTau * walkers[i].drift[j];
            }

            // Calculate the trial wave function at the old and new positions
            double oldPsi = trialWaveFunction(walkers[i].position);
            double newPsi = trialWaveFunction(newPosition);

            // Calculate the local energy at the old and new positions
            double oldLocalEnergy = getLocalEnergy(walkers[i].position);
            double newLocalEnergy = getLocalEnergy(newPosition);
            
            // Check if the proposed move crosses a nodal surface (where Psi changes sign)
            // Moves that cross nodal surfaces are typically rejected in fixed-node DMC
            bool crossedNodalSurface = (oldPsi > 0 && newPsi < 0) || (oldPsi < 0 && newPsi > 0);

            // If the nodal surface is not crossed, proceed with the Metropolis-Hastings acceptance step
            if (!crossedNodalSurface) {
                // Calculate the drift at the new proposed position
                std::vector<double> newDrift = getDrift(newPosition);
                // Calculate the forward Green's function for the drift term
                double forwardDriftGreenFunction = driftGreenFunction(newPosition, walkers[i].position, walkers[i].drift);
                // Calculate the backward Green's function for the drift term
                double backwardDriftGreenFunction = driftGreenFunction(walkers[i].position, newPosition, newDrift);

                // Calculate the acceptance probability for the Metropolis-Hastings step
                // This ensures that the walkers sample the distribution proportional to Psi^2
                double acceptanceProbability = 
                std::min(1.0, (backwardDriftGreenFunction * newPsi * newPsi) / (forwardDriftGreenFunction * oldPsi * oldPsi));
                
                // Accept or reject the proposed move based on the acceptance probability
                if (uniform(gen) < acceptanceProbability) {
                    // If accepted, update the walker's position, drift, and local energy
                    walkers[i].position = newPosition;
                    walkers[i].drift = newDrift;
                    walkers[i].localEnergy = newLocalEnergy;
                }
                // If rejected, the walker remains at its old position with its old drift and local energy
            }
            
            // Determine the branching factor (number of copies of the walker)
            // This is based on the local energy and the reference energy (implicitly via branchGreenFunction)
            double eta = uniform(gen); // Random number for stochastic branching
            // The branch factor determines how many copies of the walker are made
            // It's typically an integer, calculated from the Green's function and a random number
            double branchFactor = static_cast<int>(eta + branchGreenFunction(newLocalEnergy, oldLocalEnergy));
            // If the branch factor is positive, create copies of the walker
            if (branchFactor > 0) {
                #pragma omp critical
                for(int n = 0; n < branchFactor; n++) {
                    if (newNWalkers >= MAX_N_WALKERS) {
                        break;
                    }
                    // Add the local energy of the copied walker to the ensemble energy
                    ensembleEnergy += walkers[i].localEnergy;
                    // Add the copied walker to the new generation of walkers
                    newWalkers[newNWalkers] = walkers[i];
                    newNWalkers++;
                }
            }
        }
    }
    
    // Update the instantaneous energy of the ensemble
    if (newNWalkers == 0) {std::cerr << "[WARNING] Population = " << newNWalkers << std::endl;}
    instEnergy = newNWalkers > 0 ? ensembleEnergy / newNWalkers: 0.0;
    // Replace the old generation of walkers with the new generation
    walkers = newWalkers;
    // Update the total number of walkers
    nWalkers = newNWalkers;
}

void DMC::blockStep(int nSteps) {
    
}

void DMC::updateReferenceEnergy(double blockEnergy) {
    double ratio = static_cast<double>(nWalkers) / static_cast<double>(N_WALKERS_TARGET);
    if (ratio < 1e-12) ratio = 1e-12;
    referenceEnergy = blockEnergy - ALPHA * std::log(ratio);
}

double DMC::driftGreenFunction(const std::vector<double>& newPosition,
                               const std::vector<double>& oldPosition,
                               const std::vector<double>& oldDrift) const {
    int d = nParticles * dim;
    // Δ = (R - R' - τ v_D(R'))
    double norm2 = 0.0;
    for (int j = 0; j < d; j++) {
        double diff = newPosition[j] - oldPosition[j] - deltaTau * oldDrift[j];
        norm2 += diff * diff;
    }
    // 1 / (2πτ)^(N/2)
    double factor = 1.0 / std::pow(2.0 * M_PI * deltaTau, 0.5 * d);

    double exponent = - norm2 / (2.0 * deltaTau);
    // 1 / (2πτ)^(N/2) * exp(-Δ / (2 * τ))
    return factor * std::exp(exponent);
}

double DMC::branchGreenFunction(double newLocalEnergy,
                                double oldLocalEnergy) const {
    // exp(- τ/2 [E_L(R) + E_L(R') - 2E_T])
    return std::exp(- 0.5 * deltaTau * (newLocalEnergy + oldLocalEnergy - 2.0 * referenceEnergy));
}

std::vector<double> DMC::getDrift(const std::vector<double>& position) const {
    int d = nParticles * dim;
    std::vector<double> drift(d, 0.0);
    double h = 1e-6;
    for (int i = 0; i < d; i++) {
        std::vector<double> Rp = position, Rm = position;
        Rp[i] += h;
        Rm[i] -= h;
        double forwardPsi = std::log(std::abs(trialWaveFunction(Rp)));
        double backwardPsi = std::log(std::abs(trialWaveFunction(Rm)));

        double lnDiff = forwardPsi - backwardPsi;
        drift[i] = lnDiff / (2.0 * h);
    }
    return drift;
}

double DMC::getLocalEnergy(const std::vector<double>& position) {
    int d = nParticles * dim;
    double h = 1e-8;
    double lap = - 2 * d * std::log(std::abs(trialWaveFunction(position)));
    double grad = 0.0;
    for (int i = 0; i < d; i++) {
        std::vector<double> Rp = position, Rm = position;
        Rp[i] += h;
        Rm[i] -= h;
        double forwardPsi = std::log(std::abs(trialWaveFunction(Rp)));
        double backwardPsi = std::log(std::abs(trialWaveFunction(Rm)));
        double diff = std::abs((forwardPsi - backwardPsi) / (2.0 * h));
        grad += diff * diff;
        lap += forwardPsi + backwardPsi;
    }
    lap = lap / (h * h);
    return - 0.5 * (lap + grad) + potentialEnergy(position);
}

double DMC::potentialEnergy(const std::vector<double>& position) const {
    double dx = position[0] - position[2];
    double dy = position[1] - position[3];
    double dx2 = dx * dx;
    double dy2 = dy * dy;
    double r = std::sqrt(dx2 + dy2);
    if (r < 1e-2) r = 1e-2;
    return -1.0 / r;
}

double DMC::trialWaveFunction(const std::vector<double>& position) const {
    double dx = position[0] - position[2];
    double dy = position[1] - position[3];
    double dx2 = dx * dx;
    double dy2 = dy * dy;
    double r = std::sqrt(dx2 + dy2);
    if (r < 1e-8) r = 1e-2;
    double r2 = r * r;
    double c1 = 1.0;
    double c2 = 1.0;
    double c3 = 1.0;
    // return  - c1 * r / (1 - c2 * r);
    return c1 * r2 * std::log(r) * std::exp(- c2 * r2) - c3 * r * (1 - std::exp(- c2 * r2));
}

void DMC::initializeWalkers() {
    // For simplicity and to directly address sampling from |Psi|^2,
    // we'll use a basic Metropolis-Hastings-like approach for initialization.

    std::mt19937 gen = gens[0];
    std::normal_distribution<double> dist_(0.0, 1.0);
    std::uniform_real_distribution<double> uniform(0.0, 1.0);

    const int nEquilibrationSteps = 100; // Number of Metropolis steps for equilibration
    const double stepSize = 0.5; // Step size for Metropolis proposal
    const double L = 10.0;

    
    for (int i = 0; i < nWalkers; ++i) {
        walkers[i].position.resize(nParticles * dim, 0.0);
        walkers[i].drift.resize(nParticles * dim, 0.0);

        // Start with random positions for all walkers
        for (int j = 0; j < nParticles * dim; ++j) {
            walkers[i].position[j] = 2 * uniform(gen) * L - L;
        }

        std::vector<double> currentPosition = walkers[i].position;
        double currentPsiSquared = trialWaveFunction(currentPosition);
        currentPsiSquared *= currentPsiSquared;

        // Now, equilibrate these walkers to sample from |Psi|^2 using Metropolis-Hastings
        for (int step = 0; step < nEquilibrationSteps; ++step) {
            // Propose a new position
            std::vector<double> proposedPosition = currentPosition;
            for (int j = 0; j < nParticles * dim; ++j) {
                proposedPosition[j] += dist_(gen) * stepSize; // Random walk step
            }

            double proposedPsiSquared = trialWaveFunction(proposedPosition);
            proposedPsiSquared *= proposedPsiSquared;

            // Acceptance probability
            double acceptanceRatio = proposedPsiSquared / currentPsiSquared;

            if (uniform(gen) < std::min(1.0, acceptanceRatio)) {
                currentPosition = proposedPosition;
                currentPsiSquared = proposedPsiSquared;
            }
            // If rejected, walker stays at currentPosition
        }

        walkers[i].position = currentPosition;
        // After equilibration, initialize drift and localEnergy for each walker
        walkers[i].drift = getDrift(walkers[i].position);
        walkers[i].localEnergy = getLocalEnergy(walkers[i].position);
        instEnergy += walkers[i].localEnergy;
    }
    instEnergy = instEnergy / nWalkers;
}

void DMC::run() {
    int nBlockSteps = 300;
    int nStepsPerBlock = 50;

    std::ofstream fout("dmc.dat");

    std::deque<double> energyQueue;

    for(int j = 0; j < nBlockSteps; j++) {
        double blockEnergy = 0.0;
        for(int i = 0; i < nStepsPerBlock; i++) {
            timeStep();
            blockEnergy += instEnergy;
        }
        blockEnergy = blockEnergy / nStepsPerBlock;

        energyQueue.push_back(blockEnergy);
        if (energyQueue.size() > 100) {
            energyQueue.pop_front();
        }

        meanEnergy = 0.0;
        for (double e : energyQueue) {
            meanEnergy += e;
        }
        meanEnergy /= energyQueue.size();

        double ratio = static_cast<double>(nWalkers) / static_cast<double>(N_WALKERS_TARGET);
        updateReferenceEnergy(blockEnergy);

        fout << j << " "
             << blockEnergy << " "
             << referenceEnergy << " "
             << meanEnergy << " "
             << nWalkers << "\n";
        
        
        std::cout << "Block " << j
                  << " | Energy = " << blockEnergy
                  << " | Ref Energy = " << referenceEnergy
                  << " | Mean Energy = " << meanEnergy
                  << " | Population = " << nWalkers
                  << std::endl;
    }

    fout.close();
}