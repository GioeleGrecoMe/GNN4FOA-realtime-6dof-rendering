#include <JuceHeader.h>
#include "Eigen/Dense"
#define private public
#include "DiracGNNEngine.h"
#undef private
#include <chrono>
#include <iostream>
#include <array>
#include <cmath>

static void fillFeatures(DiracGNNEngine& e){
    for(int m=0;m<DiracGNNEngine::kNumMics;++m) for(int f=0;f<e.numBins;++f){
        e.E_obs(f,m)=0.01f+0.001f*m+0.00001f*f;
        e.psi_obs(f,m)=0.15f+0.1f*m;
        const float a=0.013f*(float)(f+1)*(float)(m+1);
        e.u_obs_X(f,m)=std::cos(a); e.u_obs_Y(f,m)=std::sin(a); e.u_obs_Z(f,m)=0.1f;
    }
}
int main(){
    DiracGNNEngine e; e.prepare(16000,128); fillFeatures(e);
    std::array<float,DiracGNNEngine::kNumMics>x{-5,5,-5,5,0,7,-7,0,7,-7},y{5,5,-5,-5,7,0,0,-7,7,-7},z{};
    std::array<int,257> bins{}; for(int i=0;i<257;++i)bins[i]=i;
    std::cout.setf(std::ios::fixed); std::cout.precision(6);
    for(int n=2;n<=DiracGNNEngine::kNumMics;++n){
      for(int bc: {32,64,128,257}){
        int it = (n >= 8 ? 1 : (bc >= 257 ? 2 : 3));
        auto t=std::chrono::steady_clock::now();
        for(int i=0;i<it;++i) e.forwardGNN(0,0,0,x,y,z,e.weightsRaw,n,bins.data(),bc);
        auto ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t).count()/it;
        std::cout<<"N="<<n<<" bins="<<bc<<" avg_ms="<<ms<<"\n";
      }
    }
    const int loops=10000000; volatile int sink=0; auto t=std::chrono::steady_clock::now();
    for(int i=0;i<loops;++i) sink += e.chooseGNNInferenceStride((double)(i&31), (float)(i&1023)/1023.0f);
    auto ns=std::chrono::duration<double,std::nano>(std::chrono::steady_clock::now()-t).count()/loops;
    std::cout<<"scheduler_ns="<<ns<<" sink="<<sink<<"\n";
}
