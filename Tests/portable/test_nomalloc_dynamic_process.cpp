#define EIGEN_RUNTIME_NO_MALLOC
#include <JuceHeader.h>
#include "Eigen/Dense"
#define private public
#include "DiracGNNEngine.h"
#undef private

#include <array>
#include <iostream>

int main()
{
    DiracGNNEngine engine;
    engine.prepare(16000, 128);

    const std::array<float, DiracGNNEngine::kNumMics> x {-5.0f, 5.0f, -5.0f, 5.0f, 0.0f, 7.0f, -7.0f, 0.0f, 7.0f, -7.0f};
    const std::array<float, DiracGNNEngine::kNumMics> y { 5.0f, 5.0f, -5.0f, -5.0f, 7.0f, 0.0f, 0.0f, -7.0f, 7.0f, -7.0f};
    const std::array<float, DiracGNNEngine::kNumMics> z {};
    const std::array<float, DiracGNNEngine::kNumMics> azimuth {};

    Eigen::internal::set_is_malloc_allowed(false);
    for (int nodeCount = 1; nodeCount <= DiracGNNEngine::kNumMics; ++nodeCount)
    {
        for (int frame = 0; frame < 6; ++frame)
            engine.processSTFTFrame(0.0f, 0.0f, 0.0f,
                                    x, y, z, azimuth,
                                    0.2f, -120.0f, 1, 1,
                                    nodeCount, 1.0f, 512.0f);

        std::cout << "N=" << nodeCount
                  << " stage=" << engine.getLastGNNStage()
                  << " inf=" << engine.getGNNInferenceCount()
                  << " stride=" << engine.getGNNInferenceStride() << "\n";
    }
    Eigen::internal::set_is_malloc_allowed(true);

    return 0;
}
