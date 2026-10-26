#include <cassert>
#include <iostream>

static int forwardsForAudioFrames(int audioFrames, int stride)
{
    int cd = 0, count = 0;
    for (int f = 0; f < audioFrames; ++f)
    {
        if (cd <= 0) { ++count; cd = stride - 1; }
        else --cd;
    }
    return count;
}

int main()
{
    // Model-time cadence depends only on audio frames, not on how quickly the host
    // renders those frames in wall-clock time (realtime, anticipative, offline).
    assert(forwardsForAudioFrames(125, 4) == 32);
    assert(forwardsForAudioFrames(125, 8) == 16);
    assert(forwardsForAudioFrames(125, 16) == 8);
    std::cout << "audio-time scheduler cadence independent of host render speed: PASS\n";
}
