#include <JuceHeader.h>
#include "Eigen/Dense"
#define private public
#include "DiracGNNEngine.h"
#undef private
#include <cassert>
#include <cmath>
#include <iostream>

int main()
{
    DiracGNNEngine e;
    e.prepare(16000.0, 128);
    e.gnnForwardPerBinEwmaMs.store(0.10);
    e.gnnForwardPerBinPeakMs.store(0.10);
    e.gnnTimingReferenceMicCount.store(5);

    const double p5 = e.predictGNNForwardMs(64, 128, true, 5);
    const double p10 = e.predictGNNForwardMs(64, 128, true, 10);
    const double expectedRatio = (10.0 * 11.0) / (5.0 * 6.0);
    const double adjustedRatio = (p10 - 1.0) / (p5 - 1.0);
    assert(p10 > p5);
    assert(std::abs(adjustedRatio - expectedRatio) < 1.0e-6);

    e.gnnTimingReferenceMicCount.store(10);
    const double p10ref = e.predictGNNForwardMs(64, 128, true, 10);
    assert(std::abs(p10ref - p5) < 1.0e-9);

    std::cout << "node-aware GNN admission N=5->10 ratio=" << adjustedRatio << ": PASS\n";
    return 0;
}
