#include <JuceHeader.h>
#include "Eigen/Dense"
#define private public
#include "DiracGNNEngine.h"
#undef private
#include <array>
#include <cassert>
#include <cmath>
#include <iostream>

int main()
{
    DiracGNNEngine e;
    e.prepare(16000.0, 128);

    // 1) A re-entering bin must be able to discard every old temporal sample.
    constexpr int k = 40;
    for (int slot = 0; slot < DiracGNNEngine::kMaxTemporalWindow; ++slot)
        e.weightHistoryValid[(size_t) slot][k] = 1u;
    e.invalidateTemporalHistoryForBin(k);
    for (int slot = 0; slot < DiracGNNEngine::kMaxTemporalWindow; ++slot)
        assert(e.weightHistoryValid[(size_t) slot][k] == 0u);

    // 2) Acoustic novelty must react to learned inputs even with static geometry.
    std::array<int, 1> bins{{k}};
    e.lastInferenceFeatureValid[k] = 1u;
    for (int m = 0; m < DiracGNNEngine::kNumMics; ++m)
    {
        e.lastInferenceE(k,m) = 1.0f;
        e.lastInferencePsi(k,m) = 0.2f;
        e.lastInferenceUX(k,m) = 1.0f;
        e.lastInferenceUY(k,m) = 0.0f;
        e.lastInferenceUZ(k,m) = 0.0f;
        e.E_obs(k,m) = 1.0f;
        e.psi_obs(k,m) = 0.2f;
        e.u_obs_X(k,m) = 1.0f;
        e.u_obs_Y(k,m) = 0.0f;
        e.u_obs_Z(k,m) = 0.0f;
    }
    float novelty = -1.0f;
    assert(!e.acousticFeaturesChanged(bins.data(), 1, DiracGNNEngine::kTrainingMicCount, novelty));
    assert(novelty < 1.0e-6f);

    // Strong energy/direction change, same positions: should request faster GNN.
    for (int m = 0; m < DiracGNNEngine::kNumMics; ++m)
    {
        e.E_obs(k,m) = (m & 1) ? 0.15f : 3.0f;
        e.u_obs_X(k,m) = 0.0f;
        e.u_obs_Y(k,m) = 1.0f;
    }
    assert(e.acousticFeaturesChanged(bins.data(), 1, DiracGNNEngine::kTrainingMicCount, novelty));
    assert(novelty > 0.1f);

    // 3) The learned transition is deliberately only two 8-ms hops.
    assert(DiracGNNEngine::kGNNBlendFrames == 2);

    // 4) Very fast backends may recover the original 125-Hz behaviour.
    assert(e.chooseGNNInferenceStride(2.0, 1.0f) == 1);
    assert(e.chooseGNNInferenceStride(4.0, 1.0f) == 2);

    // 5) Worst-case startup timing seeds reservoir/cadence, but the first real
    // compact forward must replace it rather than decay toward it for seconds.
    DiracGNNEngine timing;
    timing.prepare(16000.0, 128);
    timing.seedGNNForwardTimingEstimate(123.0);
    std::array<float,DiracGNNEngine::kNumMics> x{-5,5,-5,5,0}, y{5,5,-5,-5,7}, z{}, az{};
    timing.processSTFTFrame(0,0,0,x,y,z,az,.2f,-120.f,1,1,2,1.0f,512.0f);
    const double realMs = timing.getLastGNNForwardMs();
    const double ewmaMs = timing.getGNNForwardEwmaMs();
    assert(realMs > 0.0);
    assert(std::abs(realMs - ewmaMs) < 1.0e-6);

    std::cout << "temporal safety: re-entry, acoustic novelty, 2-hop blend, stride1 PASS\n";
    return 0;
}
