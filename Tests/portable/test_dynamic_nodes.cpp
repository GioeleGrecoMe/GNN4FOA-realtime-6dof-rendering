#include <JuceHeader.h>
#include "Eigen/Dense"
#define private public
#include "DiracGNNEngine.h"
#undef private

#include <algorithm>
#include <array>
#include <iostream>

int main()
{
    DiracGNNEngine engine;
    engine.prepare(16000, 128);

    for (int mic = 0; mic < DiracGNNEngine::kNumMics; ++mic)
        for (int bin = 0; bin < 257; ++bin)
        {
            engine.E_obs(bin, mic) = 0.01f + 0.001f * mic + 0.00001f * bin;
            engine.psi_obs(bin, mic) = 0.2f + 0.05f * mic;
            engine.u_obs_X(bin, mic) = 0.7f;
            engine.u_obs_Y(bin, mic) = 0.5f;
            engine.u_obs_Z(bin, mic) = 0.1f;
        }

    const std::array<float, DiracGNNEngine::kNumMics> x {-5.0f, 5.0f, -5.0f, 5.0f, 0.0f, 7.0f, -7.0f, 0.0f, 7.0f, -7.0f};
    const std::array<float, DiracGNNEngine::kNumMics> y { 5.0f, 5.0f, -5.0f, -5.0f, 7.0f, 0.0f, 0.0f, -7.0f, 7.0f, -7.0f};
    const std::array<float, DiracGNNEngine::kNumMics> z {};
    std::array<int, 257> bins {};
    for (int i = 0; i < 257; ++i)
        bins[(size_t) i] = i;

    for (int nodeCount = 2; nodeCount <= DiracGNNEngine::kNumMics; ++nodeCount)
    {
        engine.weightsRaw.setZero();
        bool ok = engine.forwardGNN(0.0f, 0.0f, 0.0f,
                                    x, y, z, engine.weightsRaw,
                                    nodeCount, bins.data(), 257);

        engine.weightsCandidate = engine.weightsRaw;
        ok = ok && engine.constrainGNNWeightSumsForBins(engine.weightsCandidate,
                                                        nodeCount, bins.data(), 257);

        float minSum = 1.0e9f;
        float maxSum = 0.0f;
        bool bounded = true;
        for (int bin = 0; bin < 257; ++bin)
        {
            float sum = 0.0f;
            for (int mic = 0; mic < nodeCount; ++mic)
                sum += engine.weightsCandidate(bin, mic);

            minSum = std::min(minSum, sum);
            maxSum = std::max(maxSum, sum);
            bounded = bounded && sum >= 0.5f - 2.0e-6f && sum <= 2.0f + 2.0e-6f;
        }

        ok = ok && bounded;
        std::cout << "N=" << nodeCount << " ok=" << ok
                  << " sum_range=[" << minSum << "," << maxSum << "]\n";
        if (!ok)
            return 2;
    }

    return 0;
}
