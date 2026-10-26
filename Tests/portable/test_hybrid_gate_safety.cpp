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
    // Enter at configured threshold; once inside, stay learned until 3 dB lower.
    const float t = 1.0f;
    assert(!DiracGNNEngine::gateWantsGNN(false, 0.99f, t));
    assert( DiracGNNEngine::gateWantsGNN(false, 1.00f, t));
    assert( DiracGNNEngine::gateWantsGNN(true,  0.75f, t));
    assert( DiracGNNEngine::gateWantsGNN(true,  0.51f, t));
    assert(!DiracGNNEngine::gateWantsGNN(true,  0.49f, t));

    // Both directions take exactly two 8-ms model hops.
    float mix = 0.0f;
    mix = DiracGNNEngine::advanceGateMix(mix, true);
    assert(std::abs(mix - 0.5f) < 1.0e-7f);
    mix = DiracGNNEngine::advanceGateMix(mix, true);
    assert(std::abs(mix - 1.0f) < 1.0e-7f);
    mix = DiracGNNEngine::advanceGateMix(mix, false);
    assert(std::abs(mix - 0.5f) < 1.0e-7f);
    mix = DiracGNNEngine::advanceGateMix(mix, false);
    assert(std::abs(mix) < 1.0e-7f);

    // A coverage hole during a normal hold requests ONLY the missing bins;
    // cadence-due frames remain full-batch updates.
    assert( DiracGNNEngine::shouldUseCoverageOnly(true,  3, 1));
    assert( DiracGNNEngine::shouldUseCoverageOnly(true, 15, 7));
    assert(!DiracGNNEngine::shouldUseCoverageOnly(true,  0, 1));
    assert(!DiracGNNEngine::shouldUseCoverageOnly(false, 4, 1));
    assert(!DiracGNNEngine::shouldUseCoverageOnly(true,  4, 0));

    // Admission prediction scales with the batch actually about to run.
    DiracGNNEngine e;
    e.prepare(16000.0, 128);
    e.seedGNNForwardTimingEstimate(100.0); // full 257-bin conservative benchmark
    const double p1 = e.predictGNNForwardMs(1, 200, true, 5);
    const double p16 = e.predictGNNForwardMs(16, 200, true, 5);
    const double p128 = e.predictGNNForwardMs(128, 200, true, 5);
    assert(p1 > 0.0 && p1 < p16 && p16 < p128);
    assert(p128 < e.predictGNNForwardMs(200, 200, true, 5));

    // Parameter crossfade is continuous even though Delaunay and GNN use
    // different interpolation operators. Opposing direct directions become
    // diffuse at the midpoint instead of producing a hard spatial switch.
    DiracGNNEngine::SpatialParams d, g;
    d.energy = 1.0f; d.diffuseness = 0.0f; d.ux = 1.0f;
    g.energy = 0.5f; g.diffuseness = 0.0f; g.ux = -1.0f;
    auto p0 = DiracGNNEngine::blendSpatialParams(d, g, 0.0f);
    auto ph = DiracGNNEngine::blendSpatialParams(d, g, 0.5f);
    auto p1e = DiracGNNEngine::blendSpatialParams(d, g, 1.0f);
    assert(std::abs(p0.energy - 1.0f) < 1.0e-7f);
    assert(std::abs(ph.energy - 0.75f) < 1.0e-7f);
    assert(std::abs(p1e.energy - 0.5f) < 1.0e-7f);
    assert(ph.diffuseness > 0.999f);
    assert(std::abs(p0.ux - 1.0f) < 1.0e-7f);
    assert(std::abs(p1e.ux + 1.0f) < 1.0e-7f);

    // Pure learned endpoint must preserve the v1.0.1 learned renderer
    // semantics exactly: weighted energy, weight-normalized psi, and
    // energy/directness-weighted direction.
    constexpr int k = 12;
    for (int m = 0; m < 3; ++m)
    {
        e.weightsGNNConstrained(k,m) = 0.1f * (float) (m + 1);
        e.E_obs(k,m) = 1.0f + (float) m;
        e.psi_obs(k,m) = 0.1f * (float) (m + 1);
        e.u_obs_X(k,m) = (m == 0) ? 1.0f : 0.0f;
        e.u_obs_Y(k,m) = (m == 1) ? 1.0f : 0.0f;
        e.u_obs_Z(k,m) = (m == 2) ? 1.0f : 0.0f;
    }
    const auto lp = e.interpolateLearnedParams(k, 3);
    float manualE=0, manualPsi=0, sumW=0, vx=0,vy=0,vz=0;
    for (int m=0;m<3;++m)
    {
        const float w=e.weightsGNNConstrained(k,m), em=e.E_obs(k,m), ps=e.psi_obs(k,m);
        manualE += w*em; sumW += w; manualPsi += w*ps;
        const float direct=w*em*(1.0f-ps);
        vx += direct*e.u_obs_X(k,m); vy += direct*e.u_obs_Y(k,m); vz += direct*e.u_obs_Z(k,m);
    }
    manualPsi /= sumW;
    const float vn=std::sqrt(vx*vx+vy*vy+vz*vz);
    assert(std::abs(lp.energy-manualE) < 1.0e-6f);
    assert(std::abs(lp.diffuseness-manualPsi) < 1.0e-6f);
    assert(std::abs(lp.ux-vx/vn) < 1.0e-6f);
    assert(std::abs(lp.uy-vy/vn) < 1.0e-6f);
    assert(std::abs(lp.uz-vz/vn) < 1.0e-6f);

    std::cout << "hybrid gate safety: hysteresis, bidirectional fade, partial coverage, bin-aware admission, learned endpoint PASS\n";
    return 0;
}
