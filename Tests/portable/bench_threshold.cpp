#include <array>
#include <chrono>
#include <cmath>
#include <iostream>

int main()
{
    std::array<float, 257> energy{};
    for (int k = 0; k < 257; ++k)
        energy[(size_t) k] = 1.0e-6f + 0.001f * (float) (k + 1);

    const float peak = energy.back();
    const float thresholdDb = -47.5f;
    volatile int sink = 0;
    constexpr int frames = 200000;

    const auto t0 = std::chrono::steady_clock::now();
    for (int frame = 0; frame < frames; ++frame)
        for (int k = 0; k < 257; ++k)
        {
            const float db = 10.0f * std::log10((energy[(size_t) k] + 1.0e-8f) / (peak + 1.0e-8f));
            sink += db >= thresholdDb;
        }
    const auto t1 = std::chrono::steady_clock::now();

    const auto t2 = t1;
    for (int frame = 0; frame < frames; ++frame)
    {
        const float linearThreshold = (peak + 1.0e-8f) * std::pow(10.0f, 0.1f * thresholdDb);
        for (int k = 0; k < 257; ++k)
            sink += (energy[(size_t) k] + 1.0e-8f) >= linearThreshold;
    }
    const auto t3 = std::chrono::steady_clock::now();

    const double logarithmicMs = std::chrono::duration<double, std::milli>(t1 - t0).count() / frames;
    const double linearMs = std::chrono::duration<double, std::milli>(t3 - t2).count() / frames;
    std::cout << "logarithmic_ms_per_frame=" << logarithmicMs
              << " linear_ms_per_frame=" << linearMs
              << " speedup=" << logarithmicMs / linearMs
              << " sink=" << sink << "\n";
}
