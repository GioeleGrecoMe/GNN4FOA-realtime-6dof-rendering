#pragma once
#include <cstddef>
typedef unsigned long vDSP_Length;
static inline void vDSP_mmul(const float* A, long IA,
                             const float* B, long IB,
                             float* C, long IC,
                             vDSP_Length M, vDSP_Length N, vDSP_Length P)
{
    for (vDSP_Length r = 0; r < M; ++r)
        for (vDSP_Length c = 0; c < N; ++c)
        {
            float sum = 0.0f;
            for (vDSP_Length k = 0; k < P; ++k)
                sum += A[(r * P + k) * (vDSP_Length) IA]
                     * B[(k * N + c) * (vDSP_Length) IB];
            C[(r * N + c) * (vDSP_Length) IC] = sum;
        }
}
