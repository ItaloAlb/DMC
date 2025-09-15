#include <fstream>
#include <iostream>
#include "DMC.h"

int main()
{
    DMC dmc(1000, 2);

    const int nSteps = 1'000'000;   // passos totais desejados
    const int equilSteps = 10'000;      // passos para equilibrar (em steps)
    const int BLOCK = 1'000;       // tamanho do bloco (em steps)
    const int WINDOW = 1'000;       // janela da média móvel (em blocos)
    const double HARTREE_TO_EV = 27.211386245988;

    const int equilBlocks = (equilSteps + BLOCK - 1) / BLOCK;

    std::vector<double> win(WINDOW, 0.0);
    int head = 0;
    int filled = 0;
    double sumWin = 0.0;

    std::ofstream fout("dmc.dat");
    fout << "# time  E(Ha)  <E>(Ha)  <E>(eV)\n";

    int stepDone = 0;
    int blockCount = 0;

    while (stepDone < nSteps) {
        const int nBlockStep = std::min(BLOCK, nSteps - stepDone);

        dmc.blockStep(nBlockStep);

        stepDone += nBlockStep;
        ++blockCount;

        const double E_inst = dmc.meanLocalEnergy();

        if (blockCount > equilBlocks) {
            if (filled == WINDOW) {
                sumWin -= win[head]; 
            }
            else {
                ++filled;
            }
            win[head] = E_inst;
            sumWin += E_inst;
            head = (head + 1) % WINDOW;
        }

        const double E_mean = (filled > 0 ? sumWin / filled : 0.0);

        const double time = stepDone * TAU;

        if (blockCount % 1 == 0) {
            fout << time << " "
                << E_inst << " "
                << E_mean << " "
                << (E_mean * HARTREE_TO_EV) << "\n";
            fout.flush();
        }

        if (blockCount % 10 == 0) {
            std::cout << "[Block " << blockCount << " | steps " << stepDone << "]  "
                << " E_inst (Hartree) = " <<       E_inst
                << " | <E> (Hartree) = " << E_mean
                << " | <E> (eV) = " <<    (E_mean * HARTREE_TO_EV) 
                << " | n Walkers = " << dmc.getNumberWalkers() << "\n";
        }
    }

    fout.close();
    return 0;
}