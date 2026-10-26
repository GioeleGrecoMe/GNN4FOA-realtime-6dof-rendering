// GNN4FOA v2
// Authors: Gioele Greco, Filippo Marri, Mirco Pezzoli, Fabio Antonacci
// Image and Sound Processing Lab (ISPL), Politecnico di Milano
// Copyright (c) 2026 Gioele Greco, Filippo Marri, Mirco Pezzoli, Fabio Antonacci

#pragma once

namespace gnn4foa
{
struct ManualFoaPolicy
{
    static constexpr int kChannelsPerNode = 4;
    static constexpr int kMaxNodes = 10;
    static constexpr int kPhysicalInputChannels = kChannelsPerNode * kMaxNodes;
    static constexpr int kOutputChannels = 4;

    static constexpr int sanitiseNodeCount(int nodes) noexcept
    {
        return nodes < 1 ? 1 : (nodes > kMaxNodes ? kMaxNodes : nodes);
    }

    static constexpr int channelsForNodes(int nodes) noexcept
    {
        return kChannelsPerNode * sanitiseNodeCount(nodes);
    }
};
}
