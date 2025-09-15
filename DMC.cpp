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
    : isAlive(true), localEnergy(0.0), oldLocalEnergy(0.0)
{
    position.resize(nParticles * dim, 0.0);
	drift.resize(nParticles * dim, 0.0);
}

DMC::DMC(int nWalkers_, int nParticles_, int dim_)
	: nWalkers(nWalkers_), nParticles(nParticles_), dim(dim_)
{
	a = -0.5;
	b = 0.5;
	c1 = 0.25;
	c2 = 0.2;
	c3 = 1.00078125;

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
		updateReferenceEnergy();
	}
}

//void DMC::walk() {
//	int idx = 0;
//	for (int i = 0; i < nWalkers; ++i) {
//		if (walkers[i].isAlive) {
//			double g1, g2;
//			gauss(g1, g2);
//
//			walkers[i].position[0] += walkers[i].drift[0] * TAU + g1 * SQRT_TAU;
//			walkers[i].position[1] += walkers[i].drift[1] * TAU + g2 * SQRT_TAU;
//
//			updateDrift(i);
//			updateLocalEnergy(i);
//
//			if (i != idx)
//				walkers[idx] = walkers[i];
//			++idx;
//		}
//	}
//	nWalkers = idx;
//}

void DMC::walk() {
	int idx = 0;
	for (int i = 0; i < nWalkers; ++i) {
		if (!walkers[i].isAlive) continue;

		for (int k = 0; k < nParticles; ++k) {
			double g1, g2;
			gauss(g1, g2);

			int ix = 2 * k;
			int iy = ix + 1;

			walkers[i].position[ix] += walkers[i].drift[ix] * TAU + g1 * SQRT_TAU;
			walkers[i].position[iy] += walkers[i].drift[iy] * TAU + g2 * SQRT_TAU;
		}

		updateDrift(i);
		updateLocalEnergy(i);

		if (i != idx) walkers[idx] = walkers[i];
		++idx;
	}
	nWalkers = idx;
}

void DMC::branch() {
	int nBirth = 0;
	std::uniform_real_distribution<double> uniform_dist(0.0, 1.0);

	double stepEnergy = 0.0;
	double weight = 0.0;
	
	for (int i = 0; i < nWalkers; i++) {
		const double PB = std::exp(-TAU * (0.5 * (walkers[i].oldLocalEnergy + walkers[i].localEnergy) - refEnergy));

		stepEnergy += PB * walkers[i].localEnergy;
		weight += PB;

		const double u = uniform_dist(rng);

		const int nCopies = static_cast<int>(std::floor(PB + u));

		if (nCopies <= 0) {
			walkers[i].isAlive = false;
			continue;
		}

		for (int c = 1; c < nCopies; c++) {
			if (nWalkers + nBirth >= MAX_N_WALKERS) break;
			walkers[nWalkers + nBirth] = walkers[i];
			nBirth++;
		}

		if (walkers[i].localEnergy > maxLocalEnergy) { maxLocalEnergy = walkers[i].localEnergy;}

		if (walkers[i].localEnergy < minLocalEnergy) { minLocalEnergy = walkers[i].localEnergy;}

		//double rBirth = uniform_dist(rng);
		//double rDeath = uniform_dist(rng);
		//if (walkers[i].localEnergy < refEnergy && rBirth < (refEnergy - walkers[i].localEnergy) * TAU) {
		//	if (nWalkers + nBirth < MAX_N_WALKERS) {
		//		walkers[nWalkers + nBirth] = walkers[i];
		//		nBirth++;
		//	}
		//}

		//else if (walkers[i].localEnergy > refEnergy && rDeath < (walkers[i].localEnergy - refEnergy) * TAU) {
		//	walkers[i].isAlive = false;
		//}
	}
	nWalkers += nBirth;
	lastStepEnergy = stepEnergy / weight;
}

//double DMC::jastrowLaplacian(int i) {
//	double x1 = walkers[i].position[0];
//	double y1 = walkers[i].position[1];
//
//	double r = std::sqrt(x1 * x1 + y1 * y1);
//	if (r < 1e-8) r = 1e-8;
//
//	double lap = (a - a * b * r) / (r * std::pow(1.0 + b * r, 3));
//	return lap;
//}

double DMC::jastrowLaplacian(int i) {
	Walker& currentWalker = walkers[i];

	int nParticles = currentWalker.position.size() / 2;

	double laplacian = 0.0;

	for (int k = 0; k < nParticles; ++k) {
		int k_idx_x = k * 2;
		int k_idx_y = k * 2 + 1;

		for (int m = 0; m < nParticles; ++m) {
			if (m == k) {
				continue;
			}

			int m_idx_x = m * 2;
			int m_idx_y = m * 2 + 1;

			double dx_mk = currentWalker.position[k_idx_x] - currentWalker.position[m_idx_x];
			double dy_mk = currentWalker.position[k_idx_y] - currentWalker.position[m_idx_y];

			double r2_mk = dx_mk * dx_mk + dy_mk * dy_mk;
			double r_mk = std::sqrt(r2_mk);

			if (r_mk < 1e-8) {
				r_mk = 1e-8;
			}

			double term = (this->a * (1.0 - this->b * r_mk)) / (r_mk * std::pow(1.0 + this->b * r_mk, 3));
			laplacian += term;
		}
	}
	return laplacian;
}

double DMC::jastrowGradSquared(int i) {
	double gradSquared = 0.0;

	for (int k = 0; k < nParticles; ++k) {
		double grad_x_k = 0.0;
		double grad_y_k = 0.0;

		int k_idx_x = k * 2;
		int k_idx_y = k * 2 + 1;

		for (int m = 0; m < nParticles; ++m) {

			if (m == k) {
				continue;
			}

			int m_idx_x = m * 2;
			int m_idx_y = m * 2 + 1;

			double dx_mk = walkers[i].position[k_idx_x] - walkers[i].position[m_idx_x];
			double dy_mk = walkers[i].position[k_idx_y] - walkers[i].position[m_idx_y];

			double r2_mk = dx_mk * dx_mk + dy_mk * dy_mk;
			double r_mk = std::sqrt(r2_mk);

			if (r_mk < 1e-8) {
				r_mk = 1e-8;
			}

			double denominator = r_mk * (1.0 + this->b * r_mk) * (1.0 + this->b * r_mk);
			double factor = this->a / denominator;

			grad_x_k += factor * dx_mk;
			grad_y_k += factor * dy_mk;
		}
		gradSquared += (grad_x_k * grad_x_k) + (grad_y_k * grad_y_k);
	}
	return gradSquared;
}

//double DMC::potentialEnergy(int i) {
//	double dx = walkers[i].position[0] - 0;
//	double dy = walkers[i].position[1] - 0;
//	double r = sqrt(dx * dx + dy * dy);
//
//	if (r < 1e-8) r = 1e-8;
//
//	return - 1 / r;
//}

//double DMC::potentialEnergy(int i) {
//	const double eps = 1e-8;
//	auto& position = walkers[i].position;
//
//	double V = 0.0;
//
//	for (int p = 0; p < nParticles; ++p) {
//		double xp = position[2 * p];
//		double yp = position[2 * p + 1];
//
//		for (int q = p + 1; q < nParticles; ++q) {
//			double dx = xp - position[2 * q];
//			double dy = yp - position[2 * q + 1];
//			double r2 = dx * dx + dy * dy;
//			double r = std::sqrt(r2);
//			if (r < eps) r = eps;
//
//			V += -1.0 / r;
//		}
//	}
//
//	return V;
//}

double DMC::potentialEnergy(int i) {
	constexpr double eps2 = 1e-16;
	const auto& position = walkers[i].position;

	const double dx = position[0] - position[2];
	const double dy = position[1] - position[3];
	const double r = std::sqrt(dx * dx + dy * dy + eps2);

	double gamma = 0.577;
	double r0 = 27.116;

	//return - 1.0 / r;
	return -1 / r0 * (std::log(r / (r + r0)) + (gamma - std::log(2)) * std::exp(- r / r0));
}

//void DMC::updateLocalEnergy(int i) {
//	walkers[i].oldLocalEnergy = walkers[i].localEnergy;
//	walkers[i].localEnergy = - 0.5 * (jastrowLaplacian(i) - jastrowGradSquared(i)) + potentialEnergy(i);
//}

void DMC::updateLocalEnergy(int i) {
	walkers[i].oldLocalEnergy = walkers[i].localEnergy;

	double dx = walkers[i].position[0] - walkers[i].position[2];
	double dy = walkers[i].position[1] - walkers[i].position[3];
	double r2 = dx * dx + dy * dy;
	double r = std::sqrt(r2);
	double logr = std::log(r);

	double expmc2 = std::exp(-c2 * r2);
	double exppc2 = std::exp(c2 * r2);

	double u = c1 * r2 * logr * expmc2 - c3 * r * (1 - expmc2);

	double u1 = expmc2 * (c1 * r - c3 * (exppc2 + 2 * c2 * r2 - 1) + 2 * c1 * r * (1 - c2 * r2) * logr);

	double u2 = expmc2 * (c1 * (3 - 4 * c2 * r2) + 2 * c2 * c3 * r * (2 * c2 * r2 - 3) + 2 * c1 * (1 - 5 * c2 * r2 + 2 * c2 * c2 * r2 * r2) * logr);

	walkers[i].localEnergy = u2 + u1 / r + u1 * u1 + potentialEnergy(i);
}


//void DMC::updateDrift(int i) {
//	double dx = walkers[i].position[0] - 0;
//	double dy = walkers[i].position[1] - 0;
//
//	double r2 = dx * dx + dy * dy;
//	double r = sqrt(r2);
//	if (r < 1e-8) r = 1e-8;
//
//	double factor = a / (r * (1.0 + b * r) * (1.0 + b * r));
//
//	walkers[i].drift[0] = factor * dx;
//	walkers[i].drift[1] = factor * dy;
//}

//void DMC::updateDrift(int i) {
//	constexpr double eps = 1e-12;
//	auto& position = walkers[i].position;
//	auto& drift = walkers[i].drift;
//	std::fill(walkers[i].drift.begin(), walkers[i].drift.end(), 0.0);
//
//	for (int k = 0; k < nParticles; ++k) {
//		for (int m = k + 1; m < nParticles; ++m) {
//			double dx = position[2 * k] - position[2 * m];
//			double dy = position[2 * k + 1] - position[2 * m + 1];
//			double r2 = dx * dx + dy * dy;
//			double r = std::sqrt(r2 < eps ? eps : r2);
//			double t = 1.0 + b * r;
//			double fac = a / (r * t * t);
//
//			double fx = fac * dx;
//			double fy = fac * dy;
//
//			drift[2 * k] += fx;
//			drift[2 * k + 1] += fy;
//			drift[2 * m] -= fx;
//			drift[2 * m + 1] -= fy;
//		}
//	}
//}

void DMC::updateDrift(int i) {
	constexpr double eps2 = 1e-16;
	auto& position = walkers[i].position;
	auto& drift = walkers[i].drift;

	const double dx = position[0] - position[2];
	const double dy = position[1] - position[3];
	const double r2 = dx * dx + dy * dy + eps2;
	const double r = std::sqrt(r2);
	double logr = std::log(r);

	double expmc2 = std::exp(-c2 * r2);
	double exppc2 = std::exp(c2 * r2);

	double u1 = expmc2 * (c1 * r - c3 * (exppc2 + 2 * c2 * r2 - 1) + 2 * c1 * r * (1 - c2 * r2) * logr);

	const double fx = u1 / r * dx;
	const double fy = u1 / r * dy;

	drift[0] = fx;
	drift[1] = fy;

	drift[2] = -fx;
	drift[3] = -fy;
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

void DMC::updateReferenceEnergy() {
	refEnergy = meanLocalEnergy();
}

void DMC::blockStep(int nSteps) {
	double blockEnergy = 0.0;

	for (int k = 0; k < nSteps; ++k) {
		timeStep();              
		blockEnergy += lastStepEnergy;
	}

	refEnergy = blockEnergy / nSteps;
}

int DMC::getNumberWalkers() {
	return nWalkers;
}