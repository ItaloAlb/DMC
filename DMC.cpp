#include "DMC.h"
#include <fstream>
#include <iostream>
#include <random>

// Gerador global de números aleatórios
std::mt19937 rng(std::random_device{}());

void gauss(double& g1, double& g2) {
    static std::normal_distribution<double> norm_dist(0.0, 1.0);
    g1 = norm_dist(rng);
    g2 = norm_dist(rng);
}

Walker::Walker(int nParticles, int dim)
    : isAlive(true), localEnergy(0.0)
{
    position.resize(nParticles * dim, 0.0);
	drift.resize(nParticles * dim, 0.0);
}

DMC::DMC(int nWalkers_, int nParticles_, int dim_)
	: nWalkers(nWalkers_), nParticles(nParticles_), dim(dim_)
{
	particles.resize(nParticles);
	particles[0].mass = ELECTRON_MASS;
	particles[0].charge = ELECTRON_CHARGE;

	particles[1].mass = HOLE_MASS;
	particles[1].charge = HOLE_CHARGE;

	a = -1.0;
	b = 0.5;

	initializeWalkers();
}

void DMC::initializeWalkers() {
	double xmin = -10.0;
	double xmax = 10.0;
	double minDist2 = 1e-6;
	std::uniform_real_distribution<double> uniform_dist(0.0, 1.0);

	for (int i = 0; i < nWalkers; ++i) {
		walkers[i] = Walker(nParticles, dim);

		bool accepted = false;
		while (!accepted) {
			for (int j = 0; j < nParticles * dim; ++j) {
				double u = uniform_dist(rng);
				walkers[i].position[j] = xmin + (xmax - xmin) * u;
			}

			accepted = true;

			for (int a = 0; a < nParticles; ++a) {
				for (int b = a + 1; b < nParticles; ++b) {
					double dx = walkers[i].position[2 * a] - walkers[i].position[2 * b];
					double dy = walkers[i].position[2 * a + 1] - walkers[i].position[2 * b + 1];
					double r2 = dx * dx + dy * dy;

					if (r2 < minDist2) {
						accepted = false;
						break;
					}
				}
				if (!accepted) break;
			}
		}

		updateDrift(i);
		updateLocalEnergy(i);
	}
}

void DMC::walk() {
	int idx = 0;
	for (int i = 0; i < nWalkers; ++i) {
		if (walkers[i].isAlive) {
			for (int j = 0; j < nParticles; ++j) {
				double g1, g2;
				gauss(g1, g2);
				int k = dim * j;

				double m = particles[j].mass;

				walkers[i].position[k + 0] += walkers[i].drift[k + 0] * TAU + g1 * SQRT_TAU;
				walkers[i].position[k + 1] += walkers[i].drift[k + 1] * TAU + g2 * SQRT_TAU;
			}

			updateDrift(i);
			updateLocalEnergy(i);

			if (i != idx)
				walkers[idx] = walkers[i];
			++idx;
		}
	}
	nWalkers = idx;
}

//void DMC::walk() {
//	int idx = 0;
//	std::uniform_real_distribution<double> uniform_dist(0.0, 1.0);
//
//	for (int i = 0; i < nWalkers; ++i) {
//		if (!walkers[i].isAlive) continue;
//
//		walkers[i].oldPosition = walkers[i].position;
//		walkers[i].oldLocalEnergy = walkers[i].localEnergy;
//		std::vector<double> oldDrift = walkers[i].drift;
//
//		double oldPsi = trialWaveFunction(i);
//
//		for (int j = 0; j < nParticles; ++j) {
//			double g1, g2;
//			gauss(g1, g2);
//			int k = dim * j;
//
//			walkers[i].position[k + 0] += walkers[i].drift[k + 0] * TAU + g1 * SQRT_TAU;
//			walkers[i].position[k + 1] += walkers[i].drift[k + 1] * TAU + g2 * SQRT_TAU;
//		}
//
//		updateDrift(i);
//		double newPsi = trialWaveFunction(i);
//
//		double greenRatio = 1.0;
//		{
//			double forward = 0.0;
//			double backward = 0.0;
//			for (int j = 0; j < nParticles * dim; ++j) {
//				double deltaF = walkers[i].position[j] - walkers[i].oldPosition[j] - TAU * oldDrift[j];
//				forward += deltaF * deltaF;
//
//				double deltaB = walkers[i].oldPosition[j] - walkers[i].position[j] - TAU * walkers[i].drift[j];
//				backward += deltaB * deltaB;
//			}
//
//			forward = std::exp(-forward / (2.0 * TAU));
//			backward = std::exp(-backward / (2.0 * TAU));
//			greenRatio = backward / forward;
//		}
//
//		double W = (newPsi * newPsi) / (oldPsi * oldPsi) * greenRatio;
//
//		double u = uniform_dist(rng);
//		if (u < std::min(1.0, W)) {
//			updateLocalEnergy(i);
//		}
//		else {
//			walkers[i].position = walkers[i].oldPosition;
//			walkers[i].drift = oldDrift;
//		}
//		if (i != idx) walkers[idx] = walkers[i];
//		++idx;
//	}
//
//	nWalkers = idx;
//}

void DMC::branch() {
	int nBirth = 0;
	std::uniform_real_distribution<double> uniform_dist(0.0, 1.0);
	double refEnergy = referenceEnergy();

	for (int i = 0; i < nWalkers; i++) {
		if (walkers[i].localEnergy > maxLocalEnergy) { maxLocalEnergy = walkers[i].localEnergy;}

		if (walkers[i].localEnergy < minLocalEnergy) { minLocalEnergy = walkers[i].localEnergy;}

		double rBirth = uniform_dist(rng);
		double rDeath = uniform_dist(rng);
		if (walkers[i].localEnergy < refEnergy && rBirth < (refEnergy - walkers[i].localEnergy) * TAU) {
			if (nWalkers + nBirth < MAX_N_WALKERS) {
				walkers[nWalkers + nBirth] = walkers[i];
				nBirth++;
			}
		}

		else if (walkers[i].localEnergy > refEnergy && rDeath < (walkers[i].localEnergy - refEnergy) * TAU) {
			walkers[i].isAlive = false;
		}
	}
	nWalkers += nBirth;
}

//void DMC::branch() {
//	int nBirth = 0;
//	std::uniform_real_distribution<double> uniform_dist(0.0, 1.0);
//	double refEnergy = referenceEnergy();
//
//	for (int i = 0; i < nWalkers; ++i) {
//		if (!walkers[i].isAlive) continue;
//
//		double E_old = walkers[i].oldLocalEnergy;
//		double E_new = walkers[i].localEnergy;
//
//		double PB = std::exp(-TAU * (0.5 * (E_old + E_new) - refEnergy));
//		int nCopies = static_cast<int>(PB + uniform_dist(rng));
//
//		if (nCopies == 0) {
//			walkers[i].isAlive = false;
//		}
//		else {
//			for (int c = 1; c < nCopies && nWalkers + nBirth < MAX_N_WALKERS; ++c) {
//				walkers[nWalkers + nBirth] = walkers[i];
//				++nBirth;
//			}
//		}
//	}
//
//	nWalkers += nBirth;
//}

double DMC::jastrowLaplacian(int i) {
	double x1 = walkers[i].position[0];
	double y1 = walkers[i].position[1];
	double x2 = walkers[i].position[2];
	double y2 = walkers[i].position[3];

	double dx = x1 - x2;
	double dy = y1 - y2;
	double r2 = dx * dx + dy * dy;
	double r = sqrt(r2);
	if (r < 1e-8) r = 1e-8;

	double br = b * r;
	double denom2 = (1.0 + br) * (1.0 + br);
	double denom3 = denom2 * (1.0 + br);

	double lap = 2.0 * a * (1.0 / (r * denom2) - 2.0 * b / denom3);
	return lap;
}

double DMC::potentialEnergy(int i) {
	double dx = walkers[i].position[0] - walkers[i].position[2];
	double dy = walkers[i].position[1] - walkers[i].position[3];
	double r = sqrt(dx * dx + dy * dy);

	if (r < 1e-8) r = 1e-8;

	return - 1 / r;
}


void DMC::updateLocalEnergy(int i) {
	walkers[i].localEnergy = - 0.5 * jastrowLaplacian(i) + potentialEnergy(i);
}

void DMC::updateDrift(int i) {
	double dx = walkers[i].position[0] - walkers[i].position[2];
	double dy = walkers[i].position[1] - walkers[i].position[3];

	double r2 = dx * dx + dy * dy;
	double r = sqrt(r2);
	if (r < 1e-8) r = 1e-8;

	double factor = a / (r * (1.0 + b * r) * (1.0 + b * r));

	walkers[i].drift[0] = factor * dx;
	walkers[i].drift[1] = factor * dy;

	walkers[i].drift[2] = - factor * dx;
	walkers[i].drift[3] = - factor * dy;
}

void DMC::timeStep() {
	walk();
	branch();
}

double DMC::meanLocalEnergy() const {
	double sum = 0.0;
	int count = 0;

	for (int i = 0; i < nWalkers; ++i) {
		if (walkers[i].isAlive) {
			sum += walkers[i].localEnergy;
			++count;
		}
	}

	return (count > 0) ? sum / count : 0.0;
}

double DMC::referenceEnergy() const {
	return meanLocalEnergy() - ((nWalkers - N_WALKERS_TARGET) / (N_WALKERS_TARGET * TAU));
}


double DMC::trialWaveFunction(int i) {
	double dx = walkers[i].position[0] - walkers[i].position[2];
	double dy = walkers[i].position[1] - walkers[i].position[3];
	double r = std::sqrt(dx * dx + dy * dy);
	if (r < 1e-8) r = 1e-8;

	double exponent = a * r / (1.0 + b * r);
	return std::exp(exponent);
}