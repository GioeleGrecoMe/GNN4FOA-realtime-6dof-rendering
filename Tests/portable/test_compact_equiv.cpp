#include <JuceHeader.h>
#include "Eigen/Dense"
#define private public
#include "DiracGNNEngine.h"
#undef private

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>

int main()
{
    DiracGNNEngine engine;
    engine.prepare(16000, 128);

    for (int mic = 0; mic < DiracGNNEngine::kTrainingMicCount; ++mic)
        for (int bin = 0; bin < engine.numBins; ++bin)
        {
            engine.E_obs(bin, mic) = 0.01f + 0.001f * mic + 0.00001f * bin;
            engine.psi_obs(bin, mic) = 0.15f + 0.1f * mic;
            const float angle = 0.013f * (bin + 1) * (mic + 1);
            engine.u_obs_X(bin, mic) = std::cos(angle);
            engine.u_obs_Y(bin, mic) = std::sin(angle);
            engine.u_obs_Z(bin, mic) = 0.1f;
        }

    const std::array<float, DiracGNNEngine::kNumMics> x {-5.0f, 5.0f, -5.0f, 5.0f, 0.0f};
    const std::array<float, DiracGNNEngine::kNumMics> y { 5.0f, 5.0f, -5.0f, -5.0f, 7.0f};
    const std::array<float, DiracGNNEngine::kNumMics> z {};

    std::array<int, 257> allBins {};
    for (int i = 0; i < 257; ++i)
        allBins[(size_t) i] = i;

    Eigen::MatrixXf full = Eigen::MatrixXf::Zero(257, 5);
    Eigen::MatrixXf compact = Eigen::MatrixXf::Zero(257, 5);
    engine.forwardGNN(0.0f, 0.0f, 0.0f, x, y, z, full, 5, allBins.data(), 257);

    std::array<int, 64> selectedBins {};
    for (int i = 0; i < 64; ++i)
        selectedBins[(size_t) i] = (i * 4) % 257;
    engine.forwardGNN(0.0f, 0.0f, 0.0f, x, y, z, compact, 5, selectedBins.data(), 64);

    double maxAbs = 0.0;
    double squaredError = 0.0;
    int samples = 0;
    for (int row = 0; row < 64; ++row)
        for (int mic = 0; mic < DiracGNNEngine::kTrainingMicCount; ++mic)
        {
            const double delta = compact(selectedBins[(size_t) row], mic)
                               - full(selectedBins[(size_t) row], mic);
            maxAbs = std::max(maxAbs, std::abs(delta));
            squaredError += delta * delta;
            ++samples;
        }

    std::cout << "max_abs=" << maxAbs
              << " rms=" << std::sqrt(squaredError / samples) << "\n";
    return maxAbs == 0.0 ? 0 : 1;
}
