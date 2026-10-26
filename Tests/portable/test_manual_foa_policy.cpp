#include "ManualFoaPolicy.h"
#include <cassert>
#include <iostream>

int main()
{
    using gnn4foa::ManualFoaPolicy;
    assert(ManualFoaPolicy::kPhysicalInputChannels == 40);
    assert(ManualFoaPolicy::kOutputChannels == 4);
    for (int n = 1; n <= 10; ++n)
    {
        assert(ManualFoaPolicy::sanitiseNodeCount(n) == n);
        assert(ManualFoaPolicy::channelsForNodes(n) == 4 * n);
    }
    assert(ManualFoaPolicy::sanitiseNodeCount(0) == 1);
    assert(ManualFoaPolicy::sanitiseNodeCount(99) == 10);
    std::cout << "manual FOA policy 40-in/4-out, N=1..10: PASS\n";
}
