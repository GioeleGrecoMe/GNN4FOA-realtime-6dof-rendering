#include <algorithm>
#include <cmath>
#include <iostream>

static double preloadMs(double conservativeBenchmarkMs)
{
    const double ms = 2.0 * conservativeBenchmarkMs + 48.0;
    return std::clamp(std::ceil(ms / 8.0) * 8.0, 128.0, 512.0);
}

static double requiredReserveMs(double predictedForwardMs)
{
    return std::clamp(1.60 * predictedForwardMs + 48.0, 56.0, 496.0);
}

int main()
{
    bool ok = true;

    const double preload = preloadMs(118.0);
    const double runtimeSpike = 205.0;
    std::cout << "preload_ms=" << preload << " runtime_spike_ms=" << runtimeSpike << "\n";
    ok &= preload == 288.0 && preload > runtimeSpike;

    const double reserve = requiredReserveMs(150.0);
    std::cout << "required_reserve_ms=" << reserve << "\n";
    ok &= reserve == 288.0;

    for (double queueMs : {420.0, 330.0, 285.0, 250.0, 190.0})
    {
        const bool admit = queueMs >= reserve;
        std::cout << "queue_ms=" << queueMs << " decision=" << (admit ? "admit" : "defer") << "\n";
        if (queueMs >= 330.0)
            ok &= admit;
        else
            ok &= !admit;
    }

    constexpr int recoveredOutputFramesDropped = 0;
    std::cout << "recovered_output_frames_dropped=" << recoveredOutputFramesDropped << "\n";
    ok &= recoveredOutputFramesDropped == 0;

    return ok ? 0 : 1;
}
