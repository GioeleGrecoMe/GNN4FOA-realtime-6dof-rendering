#include <algorithm>
#include <cmath>
#include <iostream>

static int preloadHops(double benchMs, int baseHops, int maxHops)
{
    const double calibrated = benchMs > 0.0 ? 2.0 * benchMs + 48.0 : baseHops * 8.0;
    const int hops = (int) std::ceil(calibrated / 8.0);
    return std::max(baseHops, std::min(maxHops, hops));
}

static double admissionReserveMs(double predictedForwardMs)
{
    return std::clamp(1.60 * predictedForwardMs + 48.0, 56.0, 496.0);
}

int main()
{
    struct C { double ms; int x86; int arm; };
    const C cases[] = {
        {0,16,8}, {5,16,8}, {40,16,16}, {80,26,26},
        {120,36,36}, {200,56,40}, {250,64,40}
    };
    bool ok = true;
    for (auto c : cases)
    {
        const int x = preloadHops(c.ms,16,64);
        const int a = preloadHops(c.ms,8,40);
        std::cout << "bench=" << c.ms << " x86_hops=" << x << " arm_hops=" << a << "\n";
        ok &= x == c.x86 && a == c.arm;
    }
    for (double predicted : {5.0,20.0,40.0,80.0,120.0,200.0,280.0})
    {
        const double reserve = admissionReserveMs(predicted);
        std::cout << "predicted=" << predicted << " reserve=" << reserve << "\n";
        ok &= reserve > predicted;
        ok &= reserve >= 56.0 && reserve <= 496.0;
    }
    return ok ? 0 : 1;
}
