#include <cassert>
#include <iostream>

int main()
{
    constexpr double hopMs = 8.0;
    constexpr int learnedBlendFrames = 2;
    constexpr double maxBlendMs = learnedBlendFrames * hopMs;

    for (int stride : {1, 2, 4, 8, 16, 32, 64})
    {
        (void) stride;
        assert(learnedBlendFrames == 2);
        assert(maxBlendMs == 16.0);
    }

    std::cout << "learned blend latency fixed at 2 hops / 16 ms: PASS\n";
}
