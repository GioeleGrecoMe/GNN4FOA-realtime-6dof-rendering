#include <JuceHeader.h>
#include "Eigen/Dense"
#define private public
#include "DiracGNNEngine.h"
#undef private
#include <cmath>
#include <iostream>
#include <array>

static bool close(float a, float b, float eps=2.0e-6f){ return std::abs(a-b) <= eps; }
static float sumRow(const Eigen::MatrixXf& w, int r, int n){ float s=0; for(int m=0;m<n;++m)s+=w(r,m); return s; }

int main()
{
    DiracGNNEngine e; e.prepare(16000,128);
    Eigen::MatrixXf w = Eigen::MatrixXf::Zero(257,DiracGNNEngine::kNumMics);
    std::array<int,3> bins{{3,17,91}};

    // Low sum: ratios 1:2:2, raw sum 0.05 -> target sum 0.5.
    w(3,0)=0.01f; w(3,1)=0.02f; w(3,2)=0.02f;
    // In-range: must remain bitwise/near unchanged, sum 1.5.
    w(17,0)=0.2f; w(17,1)=0.5f; w(17,2)=0.8f;
    // High sum: ratios 1:2:2, raw sum 5 -> target sum 2.
    w(91,0)=1.0f; w(91,1)=2.0f; w(91,2)=2.0f;

    const float lowRatio01 = w(3,0)/w(3,1), lowRatio21 = w(3,2)/w(3,1);
    const float highRatio01 = w(91,0)/w(91,1), highRatio21 = w(91,2)/w(91,1);
    const float mid0=w(17,0), mid1=w(17,1), mid2=w(17,2);

    bool ok = e.constrainGNNWeightSumsForBins(w,3,bins.data(),(int)bins.size());
    ok = ok && close(sumRow(w,3,3),0.5f) && close(sumRow(w,17,3),1.5f) && close(sumRow(w,91,3),2.0f);
    ok = ok && close(w(3,0)/w(3,1),lowRatio01) && close(w(3,2)/w(3,1),lowRatio21);
    ok = ok && close(w(91,0)/w(91,1),highRatio01) && close(w(91,2)/w(91,1),highRatio21);
    ok = ok && close(w(17,0),mid0) && close(w(17,1),mid1) && close(w(17,2),mid2);
    for(int r:bins) for(int m=3;m<DiracGNNEngine::kNumMics;++m) ok = ok && close(w(r,m),0.0f);

    // Tiny-but-positive checkpoint-like row must not be mistaken for zero.
    Eigen::MatrixXf tiny = Eigen::MatrixXf::Zero(257,DiracGNNEngine::kNumMics);
    tiny(8,0)=1e-9f; tiny(8,1)=2e-9f; tiny(8,2)=3e-9f;
    std::array<int,1> tinyBin{{8}};
    ok = ok && e.constrainGNNWeightSumsForBins(tiny,3,tinyBin.data(),1);
    ok = ok && close(sumRow(tiny,8,3),0.5f,5e-6f);
    ok = ok && close(tiny(8,1)/tiny(8,0),2.0f,1e-5f) && close(tiny(8,2)/tiny(8,0),3.0f,1e-5f);

    // Invalid/zero row must still fail closed.
    Eigen::MatrixXf zero = Eigen::MatrixXf::Zero(257,DiracGNNEngine::kNumMics);
    std::array<int,1> zeroBin{{12}};
    const bool zeroAccepted = e.constrainGNNWeightSumsForBins(zero,3,zeroBin.data(),1);
    ok = ok && !zeroAccepted && close(sumRow(zero,12,3),0.0f);

    std::cout << "low_sum=" << sumRow(w,3,3)
              << " mid_sum=" << sumRow(w,17,3)
              << " high_sum=" << sumRow(w,91,3)
              << " tiny_sum=" << sumRow(tiny,8,3)
              << " ratios_preserved=" << (ok ? "yes" : "NO") << "\n";
    return ok ? 0 : 2;
}
