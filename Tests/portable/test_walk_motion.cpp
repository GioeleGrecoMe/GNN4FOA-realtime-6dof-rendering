#include <algorithm>
#include <cmath>
#include <iostream>

struct WalkState
{
    float x=0, y=0, vx=0, vy=0;
    void tick(float dt, float ix, float iy, bool precision)
    {
        const float norm=std::sqrt(ix*ix+iy*iy);
        if(norm>1.f){ ix/=norm; iy/=norm; }
        const float speed=precision?0.35f:1.35f;
        const bool input=norm>0.f;
        const float tau=input?0.115f:0.075f;
        const float alpha=dt>0.f?1.f-std::exp(-dt/tau):0.f;
        vx += alpha*(ix*speed-vx);
        vy += alpha*(iy*speed-vy);
        if(!input && std::hypot(vx,vy)<0.004f) vx=vy=0.f;
        x += vx*dt; y += vy*dt;
    }
};

int main()
{
    bool ok=true;
    constexpr float dt=0.02f;
    WalkState normal;
    for(int i=0;i<100;++i) normal.tick(dt,1,0,false);
    const float normalDist=std::hypot(normal.x,normal.y);
    const float normalSpeed=std::hypot(normal.vx,normal.vy);
    std::cout << "normal_2s_distance=" << normalDist << " speed=" << normalSpeed << "\n";
    ok &= normalDist>2.45f && normalDist<2.65f && std::abs(normalSpeed-1.35f)<0.01f;

    WalkState diagonal;
    for(int i=0;i<100;++i) diagonal.tick(dt,1,1,false);
    const float diagonalDist=std::hypot(diagonal.x,diagonal.y);
    std::cout << "diagonal_2s_distance=" << diagonalDist << "\n";
    ok &= std::abs(diagonalDist-normalDist)<0.002f;

    WalkState precision;
    for(int i=0;i<150;++i) precision.tick(dt,1,0,true);
    const float precisionSpeed=std::hypot(precision.vx,precision.vy);
    const float precisionDist=std::hypot(precision.x,precision.y);
    std::cout << "precision_speed=" << precisionSpeed << " distance=" << precisionDist << "\n";
    ok &= std::abs(precisionSpeed-0.35f)<0.005f && precisionDist>0.95f && precisionDist<1.05f;

    const float xBefore=normal.x;
    for(int i=0;i<25;++i) normal.tick(dt,0,0,false);
    const float coast=normal.x-xBefore;
    const float finalSpeed=std::hypot(normal.vx,normal.vy);
    std::cout << "release_coast=" << coast << " final_speed=" << finalSpeed << "\n";
    ok &= coast<0.12f && finalSpeed<0.01f;
    return ok?0:1;
}
