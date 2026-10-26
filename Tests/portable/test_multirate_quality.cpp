#include <JuceHeader.h>
#include "Eigen/Dense"
#define private public
#include "DiracGNNEngine.h"
#undef private
#include <array>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>
#include <iomanip>

static constexpr float PI=3.14159265358979323846f;
static std::array<int,8> bins{{8,16,32,48,64,96,128,192}};

static void fillFrame(DiracGNNEngine& e, int frame, float modHz, bool step)
{
    const float t = frame / 125.0f;
    const float s = std::sin(2*PI*modHz*t);
    const float c = std::cos(2*PI*modHz*t);
    const float st = (step && frame >= 62) ? 1.0f : 0.0f;
    for(int m=0;m<5;++m) for(int f=0; f<e.numBins; ++f) {
        float phase = .031f*(f+1)*(m+1);
        float ebase = 0.02f + .004f*m + .00002f*f;
        e.E_obs(f,m)= ebase * std::exp(0.45f*s + 0.35f*st*((m%2)?1.f:-1.f));
        e.psi_obs(f,m)= std::clamp(0.35f + .12f*std::sin(phase + .7f*s + .8f*st), 0.02f, .98f);
        float a = phase + .35f*c + st*(.45f + .08f*m);
        e.u_obs_X(f,m)=std::cos(a);
        e.u_obs_Y(f,m)=std::sin(a);
        e.u_obs_Z(f,m)=0.15f*std::sin(.5f*a);
    }
}

static Eigen::MatrixXf infer(DiracGNNEngine& e,
                             const std::array<float,DiracGNNEngine::kNumMics>& x,
                             const std::array<float,DiracGNNEngine::kNumMics>& y,
                             const std::array<float,DiracGNNEngine::kNumMics>& z)
{
    Eigen::MatrixXf out=Eigen::MatrixXf::Zero(257,DiracGNNEngine::kNumMics);
    e.forwardGNN(0.3f,-0.2f,0.1f,x,y,z,out,5,bins.data(),(int)bins.size());
    e.constrainGNNWeightSumsForBins(out,5,bins.data(),(int)bins.size());
    return out;
}

static void run(float modHz, bool step, int stride)
{
    DiracGNNEngine ref, test; ref.prepare(16000,128); test.prepare(16000,128);
    std::array<float,DiracGNNEngine::kNumMics>x{-5,5,-5,5,0}, y{5,5,-5,-5,7}, z{};
    Eigen::MatrixXf current=Eigen::MatrixXf::Zero(257,DiracGNNEngine::kNumMics), from=current, to=current;
    bool have=false; int blend=0, cd=0; const int BLEND=2;
    double se=0, sae=0, maxe=0; long n=0;
    for(int frame=0;frame<125;++frame){
        fillFrame(ref,frame,modHz,step); fillFrame(test,frame,modHz,step);
        Eigen::MatrixXf ideal=infer(ref,x,y,z);
        if(!have || cd<=0){
            Eigen::MatrixXf fresh=infer(test,x,y,z);
            if(!have){ current=fresh; have=true; blend=0; }
            else { from=current; to=fresh; blend=BLEND; }
            cd=stride-1;
        } else --cd;
        if(blend>0){ int completed=BLEND-blend+1; float a=(float)completed/BLEND; current=(1-a)*from+a*to; --blend; if(blend==0) current=to; }
        for(int bi:bins) for(int m=0;m<5;++m){ double d=current(bi,m)-ideal(bi,m); se+=d*d; sae+=std::abs(d); maxe=std::max(maxe,std::abs(d)); ++n; }
    }
    std::cout<<std::fixed<<std::setprecision(8)
      <<"modHz="<<modHz<<" step="<<step<<" stride="<<stride
      <<" updateHz="<<(125.0/stride)<<" blendFrames="<<BLEND
      <<" rmse="<<std::sqrt(se/n)<<" mae="<<(sae/n)<<" max="<<maxe<<"\n";
}

int main(int argc, char** argv)
{
    const bool smoke = argc > 1 && std::string(argv[1]) == "--smoke";
    if (smoke)
    {
        // Representative low/high modulation and discontinuity cases for the
        // pre-Xcode release gate. The complete sweep remains available by
        // running this executable without --smoke.
        run(0.5f, false, 4);
        run(2.0f, false, 16);
        run(5.0f, false, 64);
        run(0.5f, true, 4);
        run(0.5f, true, 32);
        return 0;
    }

    for (float hz : {0.5f, 2.0f, 5.0f})
        for (int k : {2, 4, 8, 16, 32, 64})
            run(hz, false, k);
    for (int k : {2, 4, 8, 16, 32, 64})
        run(0.5f, true, k);
}
