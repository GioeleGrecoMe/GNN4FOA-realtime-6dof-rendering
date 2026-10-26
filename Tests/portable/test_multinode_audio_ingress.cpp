#include <JuceHeader.h>
#include "Eigen/Dense"
#define private public
#include "DiracGNNEngine.h"
#undef private
#include <array>
#include <cassert>
#include <cmath>
#include <iostream>

static void runCase(int activeMics)
{
    DiracGNNEngine e;
    e.prepare(16000.0, 128);

    juce::AudioBuffer<float> in, out;
    in.setSize(DiracGNNEngine::kNumInputChannels, 1);
    out.setSize(4, 1);
    for (int ch = 0; ch < DiracGNNEngine::kNumInputChannels; ++ch)
        in.setSample(ch, 0, 0.01f * (float) (ch + 1));
    out.clear();

    std::array<float, DiracGNNEngine::kNumMics> x {-5.0f, 5.0f, -5.0f, 5.0f, 0.0f, 7.0f, -7.0f, 0.0f, 7.0f, -7.0f};
    std::array<float, DiracGNNEngine::kNumMics> y { 5.0f, 5.0f, -5.0f,-5.0f, 7.0f, 0.0f, 0.0f, -7.0f, 7.0f, -7.0f};
    std::array<float, DiracGNNEngine::kNumMics> z {};
    std::array<float, DiracGNNEngine::kNumMics> yaw {};

    e.process(in, out, 0.0f, 0.0f, 0.0f, x, y, z, yaw,
              0.2f, -80.0f, 0, 1, activeMics, 1.0f, 500.0f);

    for (int ch = 0; ch < DiracGNNEngine::kNumInputChannels; ++ch)
    {
        const float got = e.inputRingBuffer.getSample(ch, 0);
        const float expected = ch < activeMics * 4 ? 0.01f * (float) (ch + 1) : 0.0f;
        assert(std::abs(got - expected) < 1.0e-7f);
    }
}

int main()
{
    runCase(1);
    runCase(2);
    runCase(3);
    runCase(4);
    runCase(5);
    runCase(6);
    runCase(7);
    runCase(8);
    runCase(9);
    runCase(10);
    std::cout << "multi-node audio ingress 4*N: PASS\n";
    return 0;
}
