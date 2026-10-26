#pragma once
#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>
#define jassert(x) assert(x)
namespace juce {
template <class T> T jlimit(T lo,T hi,T v){ return std::max(lo,std::min(hi,v)); }
template <class T> T jmin(T a,T b){ return std::min(a,b); }
template <class T> T jmax(T a,T b){ return std::max(a,b); }
struct Time { static double getMillisecondCounterHiRes(){ using C=std::chrono::steady_clock; static auto s=C::now(); return std::chrono::duration<double,std::milli>(C::now()-s).count(); }};
template<class T> class AudioBuffer {
 public:
  void setSize(int ch,int n,bool=false,bool=false,bool=false){ c=ch; s=n; d.assign((size_t)c*(size_t)s,T{}); }
  void clear(){ std::fill(d.begin(),d.end(),T{}); }
  int getNumChannels() const{return c;} int getNumSamples() const{return s;}
  T getSample(int ch,int i) const{return d[(size_t)ch*s+i];}
  void setSample(int ch,int i,T v){d[(size_t)ch*s+i]=v;} void addSample(int ch,int i,T v){d[(size_t)ch*s+i]+=v;}
 private:int c=0,s=0;std::vector<T>d;
};
namespace dsp {
class FFT { public: explicit FFT(int){} void performRealOnlyForwardTransform(float*, bool=false) const{} void performRealOnlyInverseTransform(float*) const{} };
template<class T> class WindowingFunction { public: enum WindowingMethod{hann}; WindowingFunction(size_t n,WindowingMethod,bool,T):N(n){} void multiplyWithWindowingTable(T* p,size_t n) const { for(size_t i=0;i<n;++i) p[i]=(T)(0.5-0.5*std::cos(2.0*3.14159265358979323846*(double)i/(double)(n-1))); } private:size_t N;};
}
}
