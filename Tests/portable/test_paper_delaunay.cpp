#include <JuceHeader.h>
#include "Eigen/Dense"
#define private public
#include "DiracGNNEngine.h"
#undef private
#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>

namespace
{
void require(bool condition, const char* message)
{
    if (!condition)
    {
        std::cerr << "FAIL: " << message << "\n";
        std::exit(2);
    }
}

int nonzero(const Eigen::MatrixXf& w, int row, int n)
{
    int c = 0;
    for (int m = 0; m < n; ++m)
        if (w(row,m) > 1.0e-6f) ++c;
    return c;
}

float sumrow(const Eigen::MatrixXf& w, int row, int n)
{
    float s = 0.0f;
    for (int m = 0; m < n; ++m) s += w(row,m);
    return s;
}
}

int main()
{
    DiracGNNEngine e;
    e.prepare(16000.0, 128);
    std::array<float,DiracGNNEngine::kNumMics> x {0.0f, 4.0f, 0.0f, 4.0f, 2.0f};
    std::array<float,DiracGNNEngine::kNumMics> y {0.0f, 0.0f, 4.0f, 4.0f, 1.2f};
    Eigen::MatrixXf w = Eigen::MatrixXf::Zero(257,DiracGNNEngine::kNumMics);

    // Interior: one Delaunay triplet only; all other microphones exactly zero.
    e.computeDelaunayWeights(0.8f, 0.7f, x, y, w, 5);
    require(std::abs(sumrow(w,0,5) - 1.0f) < 1.0e-6f, "interior weights must sum to one");
    require(nonzero(w,0,5) <= 3, "interior must use at most one triplet");

    // At a known microphone the geometric interpolation must be cardinal.
    e.computeDelaunayWeights(x[0], y[0], x, y, w, 5);
    require(w(0,0) > 0.99999f, "known microphone must have unit barycentric weight");
    for (int m=1;m<5;++m)
        require(w(0,m) < 1.0e-6f, "non-coincident microphones must be zero at a known node");

    // Exterior: closest boundary point of the triangulated region, never
    // inverse-distance interpolation over all active microphones.
    e.computeDelaunayWeights(-10.0f, 2.0f, x, y, w, 5);
    require(std::abs(sumrow(w,0,5) - 1.0f) < 1.0e-6f, "exterior clamped weights must sum to one");
    require(nonzero(w,0,5) <= 2, "exterior projection onto an edge must use at most two vertices");

    // Broad sparse-support regression: the geometric baseline must never turn
    // back into an all-node mixer anywhere around the recording region.
    for (int yi=-4; yi<=12; ++yi)
        for (int xi=-4; xi<=12; ++xi)
        {
            e.computeDelaunayWeights(0.5f*xi, 0.5f*yi, x, y, w, 5);
            require(nonzero(w,0,5) <= 3, "Delaunay baseline must never activate more than one local triplet");
            require(std::abs(sumrow(w,0,5)-1.0f) < 1.0e-5f, "Delaunay weights must remain unit-sum");
        }

    // Cocircular geometry must form ONE deterministic triangulation, not the
    // overlapping collection of all empty-circumcircle triplets.
    std::array<float,DiracGNNEngine::kNumMics> sx {0.0f, 1.0f, 1.0f, 0.0f, 0.0f};
    std::array<float,DiracGNNEngine::kNumMics> sy {0.0f, 0.0f, 1.0f, 1.0f, 0.0f};
    e.computeDelaunayWeights(0.25f, 0.25f, sx, sy, w, 4);
    require(nonzero(w,0,4) == 2, "square diagonal point must lie on one deterministic diagonal");
    require(w(0,1) < 1.0e-6f && w(0,3) < 1.0e-6f, "deterministic square triangulation must not mix both diagonals");
    e.computeDelaunayWeights(-1.0f, 0.5f, sx, sy, w, 4);
    require(std::abs(w(0,0)-0.5f) < 1.0e-5f && std::abs(w(0,3)-0.5f) < 1.0e-5f,
            "outside square must project to closest hull edge");
    require(w(0,1) < 1.0e-6f && w(0,2) < 1.0e-6f, "outside square must not use remote microphones");

    // The geometrically chosen triangulation must not depend on the order in
    // which the same microphone positions are presented.
    e.computeDelaunayWeights(0.2f, 0.7f, sx, sy, w, 4);
    const std::array<float,4> value { 1.0f, 4.0f, 9.0f, 16.0f };
    float referenceValue = 0.0f;
    for (int m=0; m<4; ++m) referenceValue += w(0,m) * value[(size_t)m];
    std::array<float,DiracGNNEngine::kNumMics> px {1.0f, 0.0f, 0.0f, 1.0f, 0.0f};
    std::array<float,DiracGNNEngine::kNumMics> py {1.0f, 1.0f, 0.0f, 0.0f, 0.0f};
    const std::array<float,4> pvalue {9.0f, 16.0f, 1.0f, 4.0f};
    e.computeDelaunayWeights(0.2f, 0.7f, px, py, w, 4);
    float permutedValue = 0.0f;
    for (int m=0; m<4; ++m) permutedValue += w(0,m) * pvalue[(size_t)m];
    require(std::abs(referenceValue - permutedValue) < 1.0e-5f,
            "Delaunay result must be invariant to microphone numbering");

    for (int yi=1; yi<10; ++yi)
        for (int xi=1; xi<10; ++xi)
        {
            e.computeDelaunayWeights(0.1f*xi, 0.1f*yi, sx, sy, w, 4);
            require(nonzero(w,0,4) <= 3, "cocircular square grid must never activate overlapping triplets");
            require(std::abs(sumrow(w,0,4)-1.0f) < 1.0e-5f, "square grid barycentric sum must remain one");
        }

    // Two-node supported extension: clamped 1-D barycentric segment.
    e.computeDelaunayWeights(2.0f, 0.0f, x, y, w, 2);
    require(std::abs(w(0,0)-0.5f) < 1.0e-6f && std::abs(w(0,1)-0.5f) < 1.0e-6f,
            "two-node midpoint must interpolate linearly");
    e.computeDelaunayWeights(-3.0f, 0.0f, x, y, w, 2);
    require(w(0,0) > 0.99999f && w(0,1) < 1.0e-6f, "two-node exterior must clamp to closest endpoint");

    // Eq. (10)-(11): opposite normalized intensities cancel and naturally make
    // the target fully diffuse. This is NOT direct interpolation of psi.
    w.setZero();
    w(0,0)=0.5f; w(0,1)=0.5f;
    e.E_obs(0,0)=1.0f; e.E_obs(0,1)=1.0f;
    e.psi_obs(0,0)=0.0f; e.psi_obs(0,1)=0.0f;
    e.u_obs_X(0,0)=1.0f; e.u_obs_Y(0,0)=0.0f; e.u_obs_Z(0,0)=0.0f;
    e.u_obs_X(0,1)=-1.0f; e.u_obs_Y(0,1)=0.0f; e.u_obs_Z(0,1)=0.0f;
    auto p = e.interpolatePaperDelaunayParams(0,w,2);
    require(std::abs(p.energy-1.0f) < 1.0e-6f, "Eq9 energy interpolation mismatch");
    require(p.diffuseness > 0.99999f, "Eq10 vector cancellation must produce Eq11 full diffuseness");

    // Eq. (9): energy is barycentrically interpolated independently of the
    // normalized-intensity interpolation.
    w.setZero(); w(0,0)=0.25f; w(0,1)=0.75f;
    e.E_obs(0,0)=2.0f; e.E_obs(0,1)=6.0f;
    e.psi_obs(0,0)=1.0f; e.psi_obs(0,1)=1.0f;
    p = e.interpolatePaperDelaunayParams(0,w,2);
    require(std::abs(p.energy-5.0f) < 1.0e-6f, "Eq9 weighted energy must be exact");
    require(p.diffuseness > 0.99999f, "fully diffuse inputs must remain fully diffuse");

    // Prototype selection is the exact closest microphone (no 15% spatial
    // hysteresis). A 2-frame phase-aligned de-click transition is allowed.
    std::array<float,DiracGNNEngine::kNumMics> z {};
    e.currentAnchor = -1;
    e.selectAnchor(1.9f,0.0f,0.0f,x,y,z,2,true);
    require(e.currentAnchor==0, "prototype must initially be closest microphone 0");
    e.selectAnchor(2.1f,0.0f,0.0f,x,y,z,2,true);
    require(e.currentAnchor==1, "prototype must switch immediately at nearest-microphone boundary");
    require(e.anchorCrossfadeRemaining==DiracGNNEngine::kPaperAnchorCrossfadeFrames,
            "nearest prototype switch must use only the short de-click transition");

    // Learned mode deliberately retains its anchor hysteresis/crossfade
    // so the baseline correction does not modify GNN audio during A/B tests.
    e.currentAnchor = -1; e.previousAnchor = -1; e.anchorCrossfadeRemaining = 0;
    e.selectAnchor(1.9f,0.0f,0.0f,x,y,z,2,false);
    require(e.currentAnchor==0, "learned anchor starts at nearest microphone");
    e.selectAnchor(2.1f,0.0f,0.0f,x,y,z,2,false);
    require(e.currentAnchor==0, "learned mode must retain anchor hysteresis");
    e.selectAnchor(2.3f,0.0f,0.0f,x,y,z,2,false);
    require(e.currentAnchor==1, "learned anchor must switch once the hysteresis condition is met");
    require(e.anchorCrossfadeRemaining==DiracGNNEngine::kLearnedAnchorCrossfadeFrames,
            "learned anchor transition length must remain unchanged");


    // Ten-node geometry: Delaunay must remain local (<=3 non-zero inside,
    // <=2 on exterior hull projection) and unit-sum.
    std::array<DiracGNNEngine::Vec2, DiracGNNEngine::kNumMics> p10 {{{-5,5},{5,5},{-5,-5},{5,-5},{0,7},{7,0},{-7,0},{0,-7},{7,7},{-7,-7}}};
    std::array<float, DiracGNNEngine::kNumMics> x10 {}, y10 {};
    for (int m=0; m<DiracGNNEngine::kNumMics; ++m) { x10[m]=p10[m].x; y10[m]=p10[m].y; }
    Eigen::MatrixXf w10 = Eigen::MatrixXf::Zero(257, DiracGNNEngine::kNumMics);
    e.computeDelaunayWeights(0.4f, -0.3f, x10, y10, w10, DiracGNNEngine::kNumMics);
    float s10=0.0f; int nz10=0;
    for (int m=0; m<DiracGNNEngine::kNumMics; ++m) { s10 += w10(0,m); if (w10(0,m)>1e-6f) ++nz10; }
    require(std::abs(s10-1.0f)<1e-5f && nz10<=3, "N=10 interior Delaunay must be local and unit-sum");
    e.computeDelaunayWeights(20.0f, 20.0f, x10, y10, w10, DiracGNNEngine::kNumMics);
    s10=0.0f; nz10=0;
    for (int m=0; m<DiracGNNEngine::kNumMics; ++m) { s10 += w10(0,m); if (w10(0,m)>1e-6f) ++nz10; }
    require(std::abs(s10-1.0f)<1e-5f && nz10<=2, "N=10 exterior Delaunay must project to a hull edge");
    std::cout << "paper Delaunay: deterministic local triangulation, boundary clamp, Eq9-11, nearest prototype PASS\n";
    return 0;
}
