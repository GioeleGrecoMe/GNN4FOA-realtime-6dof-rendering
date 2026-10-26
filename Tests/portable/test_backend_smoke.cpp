#include <JuceHeader.h>
#include "Eigen/Dense"
#define private public
#include "DiracGNNEngine.h"
#undef private

#include <array>
#include <cmath>
#include <iostream>

int main()
{
    DiracGNNEngine engine;
    engine.prepare(16000, 128);

    for (int mic = 0; mic < DiracGNNEngine::kNumMics; ++mic)
        for (int bin = 0; bin < engine.numBins; ++bin)
        {
            engine.E_obs(bin, mic) = 0.02f + 0.001f * mic + 0.00001f * bin;
            engine.psi_obs(bin, mic) = 0.2f + 0.03f * mic;
            const float a = 0.017f * (bin + 1) * (mic + 1);
            engine.u_obs_X(bin, mic) = std::cos(a);
            engine.u_obs_Y(bin, mic) = std::sin(a);
            engine.u_obs_Z(bin, mic) = 0.1f;
        }

    const std::array<float, DiracGNNEngine::kNumMics> x
        {-5.f, 5.f, -5.f, 5.f, 0.f, 7.f, -7.f, 0.f, 7.f, -7.f};
    const std::array<float, DiracGNNEngine::kNumMics> y
        { 5.f, 5.f, -5.f, -5.f, 7.f, 0.f, 0.f, -7.f, 7.f, -7.f};
    const std::array<float, DiracGNNEngine::kNumMics> z {};
    const std::array<int, 2> bins {16, 64};

    for (int n : {5})
    {
        Eigen::MatrixXf out = Eigen::MatrixXf::Zero(257, DiracGNNEngine::kNumMics);
        bool ok = engine.forwardGNN(0.3f, -0.2f, 0.1f, x, y, z, out, n,
                                    bins.data(), (int) bins.size());
        ok = ok && engine.constrainGNNWeightSumsForBins(out, n, bins.data(), (int) bins.size());
        for (int b : bins)
        {
            float sum = 0.0f;
            for (int m = 0; m < n; ++m)
                sum += out(b, m);
            ok = ok && std::isfinite(sum) && sum >= 0.5f - 2.0e-6f && sum <= 2.0f + 2.0e-6f;
        }
        std::cout << "backend=" << DiracGNNEngine::getMatrixBackendName()
                  << " N=" << n << " ok=" << ok << "\n";
        if (!ok)
            return 2;
    }
    return 0;
}
