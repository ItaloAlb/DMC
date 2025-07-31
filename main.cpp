#include <fstream>
#include <iostream>
#include "DMC.h"

int main()
{
    DMC dmc(5000, 2, 2); // 2000 walkers, 2 partículas, 2D

    int nSteps = 100000;
    int equilSteps = 10000;

    double sumAvg = 0.0;
    int navg = 0;

    std::ofstream fout("dmc_output.dat");

    for (int step = 0; step < nSteps; ++step) {
        dmc.timeStep();

        double vbar = dmc.meanLocalEnergy();  // energia média local no passo atual
        double avg = 0.0;

        if (step >= equilSteps) {
            sumAvg += vbar;
            ++navg;
            avg = sumAvg / navg;
        }

        if (step % 20 == 0) {
            double time = step * TAU;

            fout << time << " " << vbar << " " << avg * 27.2114 << "\n";
            fout.flush();
        }

        if (step % 100 == 0) {
            std::cout << "[Step " << step << "]  E_local = " << vbar << "\n";
        }
    }

    fout.close();
    return 0;
}
