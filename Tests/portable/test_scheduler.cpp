#include <JuceHeader.h>
#include "Eigen/Dense"
#define private public
#include "DiracGNNEngine.h"
#undef private

#include <chrono>
#include <iostream>
#include <tuple>
#include <vector>

int main()
{
    DiracGNNEngine engine;
    engine.prepare(16000, 128);

    const std::vector<std::tuple<double, float, int>> cases {
        {0.0, 1.00f, 4}, {4.0, 1.00f, 2}, {7.0, 1.00f, 4},
        {15.0, 1.00f, 8}, {25.0, 1.00f, 16}, {50.0, 1.00f, 32},
        {90.0, 1.00f, 64}, {4.0, 0.70f, 4}, {4.0, 0.60f, 8},
        {4.0, 0.45f, 16}, {4.0, 0.30f, 32}, {4.0, 0.15f, 64}
    };

    bool ok = true;
    for (const auto& [forwardMs, reservoirFraction, expectedStride] : cases)
    {
        const int stride = engine.chooseGNNInferenceStride(forwardMs, reservoirFraction);
        std::cout << "ms=" << forwardMs
                  << " q=" << reservoirFraction
                  << " stride=" << stride
                  << " expected=" << expectedStride << "\n";
        ok = ok && stride == expectedStride;
    }

    volatile int sink = 0;
    constexpr int iterations = 10000000;
    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < iterations; ++i)
        sink += engine.chooseGNNInferenceStride(7.0 + (i & 1), 0.9f);

    const double nsPerDecision =
        std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - start).count()
        / iterations;
    std::cout << "decision_ns=" << nsPerDecision << " sink=" << sink << "\n";

    return ok ? 0 : 1;
}
