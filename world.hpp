#pragma once
#ifdef __METAL_VERSION__
#define WP device
#define WF inline
#define WT thread
#else
#include <cmath>
#include <cstdint>
#include <algorithm>
using uint = uint32_t;
#define WP
#define WF inline
#define WT
using std::sqrt; using std::fabs; using std::fmin; using std::fmax;
#endif

// Shared scalar layout is identical in C++ and MSL. Depth is ray range in metres.
struct WVec { float x,y,z; };
WF WVec wv(float x,float y,float z) { return {x,y,z}; }
WF WVec wa(WVec a,WVec b) { return wv(a.x+b.x,a.y+b.y,a.z+b.z); }
WF WVec ws(WVec a,WVec b) { return wv(a.x-b.x,a.y-b.y,a.z-b.z); }
WF WVec wm(WVec a,float s) { return wv(a.x*s,a.y*s,a.z*s); }
WF float wd(WVec a,WVec b) { return a.x*b.x+a.y*b.y+a.z*b.z; }
WF float wl(WVec a) { return sqrt(wd(a,a)); }
WF WVec wn(WVec a) { return wm(a,1.0f/fmax(wl(a),1e-8f)); }
struct WObstacle { uint kind; float center[3], size[3], velocity[3]; };
struct WWorld { WObstacle obstacles[16]; uint count,seed,family; float goal[3],wind[3]; };
WF uint wrng(WT uint& s) { s^=s<<13; s^=s>>17; s^=s<<5; return s; }
WF float wurand(WT uint& s) { return float(wrng(s)>>8)*(1.0f/16777216.0f); }
WF WVec wc(WP const WObstacle& o,float t) { return wv(o.center[0]+t*o.velocity[0],o.center[1]+t*o.velocity[1],o.center[2]+t*o.velocity[2]); }
WF float wray_sphere(WVec p,WVec d,WVec c,float r) {
    WVec q=ws(p,c); float b=wd(q,d), k=wd(q,q)-r*r, disc=b*b-k;
    if(disc<0) return 1000;
    float lo=-b-sqrt(disc),hi=-b+sqrt(disc);
    return lo>=0?lo:(hi>=0?hi:1000);
}
WF float wray_box(WVec p,WVec d,WVec c,WVec extent) {
    float lo=-1e20f,hi=1e20f;
    float pp[3]={p.x-c.x,p.y-c.y,p.z-c.z},dd[3]={d.x,d.y,d.z},ee[3]={extent.x,extent.y,extent.z};
    for(uint j=0;j<3;j++) {
        if(fabs(dd[j])<1e-8f) { if(fabs(pp[j])>ee[j]) return 1000; }
        else { float a=(-ee[j]-pp[j])/dd[j],b=(ee[j]-pp[j])/dd[j]; lo=fmax(lo,fmin(a,b)); hi=fmin(hi,fmax(a,b)); }
    }
    if(hi<fmax(lo,0.0f)) return 1000;
    return lo>=0?lo:hi;
}
WF float wray_cylinder(WVec p,WVec d,WVec c,float r,float h) {
    WVec q=ws(p,c); float a=d.x*d.x+d.y*d.y,b=q.x*d.x+q.y*d.y,k=q.x*q.x+q.y*q.y-r*r;
    float best=1000,disc=b*b-a*k;
    if(a>1e-10f && disc>=0) {
        float roots[2]={(-b-sqrt(disc))/a,(-b+sqrt(disc))/a};
        for(uint j=0;j<2;j++) { float t=roots[j]; if(t>=0 && fabs(q.z+t*d.z)<=h) best=fmin(best,t); }
    }
    if(fabs(d.z)>1e-8f) for(int sign=-1;sign<=1;sign+=2) {
        float t=(float(sign)*h-q.z)/d.z; float x=q.x+t*d.x,y=q.y+t*d.y;
        if(t>=0 && x*x+y*y<=r*r) best=fmin(best,t);
    }
    return best;
}
WF float wray(WP const WWorld& w,WVec p,WVec d,float time) {
    float best=12;
    // Room bounds: x[-2,14], y[-5,5], z[0,5]. The inside exit is the nearest wall.
    best=fmin(best,wray_box(p,d,wv(6,0,2.5f),wv(8,5,2.5f)));
    for(uint i=0;i<w.count;i++) {
        WP const WObstacle& o=w.obstacles[i]; WVec c=wc(o,time); float t=1000;
        if(o.kind==0) t=wray_box(p,d,c,wv(o.size[0],o.size[1],o.size[2]));
        if(o.kind==1) t=wray_sphere(p,d,c,o.size[0]);
        if(o.kind==2) t=wray_cylinder(p,d,c,o.size[0],o.size[2]);
        best=fmin(best,t);
    }
    return fmax(0.0f,best);
}
WF float wclearance(WP const WWorld& w,WVec p,float time) {
    float best=fmin(fmin(p.x+2,14-p.x),fmin(fmin(p.y+5,5-p.y),fmin(p.z,5-p.z)))-0.18f;
    for(uint i=0;i<w.count;i++) {
        WP const WObstacle& o=w.obstacles[i]; WVec q=ws(p,wc(o,time)); float distance;
        if(o.kind==1) distance=wl(q)-o.size[0];
        else if(o.kind==2) {
            float a=sqrt(q.x*q.x+q.y*q.y)-o.size[0],b=fabs(q.z)-o.size[2];
            distance=sqrt(fmax(a,0.0f)*fmax(a,0.0f)+fmax(b,0.0f)*fmax(b,0.0f))+fmin(fmax(a,b),0.0f);
        } else {
            WVec k=wv(fabs(q.x)-o.size[0],fabs(q.y)-o.size[1],fabs(q.z)-o.size[2]);
            distance=wl(wv(fmax(k.x,0.0f),fmax(k.y,0.0f),fmax(k.z,0.0f)))+fmin(fmax(k.x,fmax(k.y,k.z)),0.0f);
        }
        best=fmin(best,distance-0.18f);
    }
    return best;
}
WF WVec wcamera(uint pixel) {
    float y=(float(pixel%20)+0.5f)/20*2-1,z=(float(pixel/20)+0.5f)/16*2-1;
    return wn(wv(1,-y*1.0f,-z*0.75f)); // FLU pinhole; 90-degree horizontal FOV.
}
WF void wadd(WP WWorld& w,uint kind,WVec center,WVec half_extent,WVec velocity) {
    if(w.count>=16) return;
    WP WObstacle& o=w.obstacles[w.count++];
    o.kind=kind;
    o.center[0]=center.x;o.center[1]=center.y;o.center[2]=center.z;
    o.size[0]=half_extent.x;o.size[1]=half_extent.y;o.size[2]=half_extent.z;
    o.velocity[0]=velocity.x;o.velocity[1]=velocity.y;o.velocity[2]=velocity.z;
}
WF void wadd_box(WP WWorld& w,WVec center,WVec half_extent) {
    wadd(w,0,center,half_extent,wv(0,0,0));
}
WF void wadd_table(WP WWorld& w,float x,float y,float half_x,float half_y,float top_z) {
    const float slab_half_z=0.08f;
    const float underside=top_z-slab_half_z;
    wadd_box(w,wv(x,y,top_z),wv(half_x,half_y,slab_half_z));
    const float leg_x=half_x-0.12f,leg_y=half_y-0.12f,leg_half=0.08f;
    const float leg_center_z=underside*0.5f,leg_half_z=underside*0.5f;
    wadd_box(w,wv(x-leg_x,y-leg_y,leg_center_z),wv(leg_half,leg_half,leg_half_z));
    wadd_box(w,wv(x-leg_x,y+leg_y,leg_center_z),wv(leg_half,leg_half,leg_half_z));
    wadd_box(w,wv(x+leg_x,y-leg_y,leg_center_z),wv(leg_half,leg_half,leg_half_z));
    wadd_box(w,wv(x+leg_x,y+leg_y,leg_center_z),wv(leg_half,leg_half,leg_half_z));
}
WF float wroute_y(WP const WWorld& w,float x,float distance) {
    return w.goal[1]*(x/distance);
}
WF void wadd_doorway(WP WWorld& w,float wall_x,float gap_y,float gap_z,float half_gap,float half_gap_z) {
    const float low_y=gap_y-half_gap,high_y=gap_y+half_gap;
    const float low_z=gap_z-half_gap_z,high_z=gap_z+half_gap_z;
    const float wall_half_x=0.12f;
    wadd_box(w,wv(wall_x,(-5.0f+low_y)*0.5f,2.5f),wv(wall_half_x,(low_y+5.0f)*0.5f,2.5f));
    wadd_box(w,wv(wall_x,(high_y+5.0f)*0.5f,2.5f),wv(wall_half_x,(5.0f-high_y)*0.5f,2.5f));
    wadd_box(w,wv(wall_x,0,low_z*0.5f),wv(wall_half_x,5.0f,low_z*0.5f));
    wadd_box(w,wv(wall_x,0,(high_z+5.0f)*0.5f),wv(wall_half_x,5.0f,(5.0f-high_z)*0.5f));
}
WF void wgenerate_doorway(WP WWorld& w,WT uint& rng,float distance) {
    const float wall_x=distance*(0.42f+0.16f*wurand(rng));
    const float route_fraction=wall_x/distance;
    const float gap_y=w.goal[1]*route_fraction+(wurand(rng)-0.5f)*1.20f;
    const float half_gap=0.40f+0.30f*wurand(rng); // clear doorway width 0.8 to 1.4 m
    const float route_z=1.5f+(w.goal[2]-1.5f)*route_fraction;
    const float gap_z=route_z+(wurand(rng)-0.5f)*0.20f;
    const float half_gap_z=0.65f+0.25f*wurand(rng); // opening height 1.3 to 1.8 m
    wadd_doorway(w,wall_x,gap_y,gap_z,half_gap,half_gap_z);
}
WF void wgenerate_two_doorways(WP WWorld& w,WT uint& rng,float distance) {
    for(uint door=0;door<2;door++) {
        const float fraction=door==0?0.33f+0.02f*(wurand(rng)-0.5f):0.67f+0.02f*(wurand(rng)-0.5f);
        const float wall_x=distance*fraction;
        const float route_fraction=wall_x/distance;
        const float gap_y=wroute_y(w,wall_x,distance)+(wurand(rng)-0.5f)*1.00f; // independent offsets within +/-0.5 m
        const float route_z=1.5f+(w.goal[2]-1.5f)*route_fraction;
        const float gap_z=route_z+(wurand(rng)-0.5f)*0.24f;
        const float half_gap=0.40f+0.30f*wurand(rng);
        const float half_gap_z=0.65f+0.25f*wurand(rng);
        wadd_doorway(w,wall_x,gap_y,gap_z,half_gap,half_gap_z);
    }
}
WF void wgenerate_table_or_counter(WP WWorld& w,WT uint& rng,float distance) {
    const float x=distance*(0.45f+0.10f*wurand(rng));
    const float y=wroute_y(w,x,distance)+(wurand(rng)-0.5f)*0.20f;
    if(wurand(rng)<0.5f) {
        const float half_x=0.55f+0.20f*wurand(rng);
        const float half_y=0.75f+0.20f*wurand(rng);
        const float top_z=1.55f+0.50f*wurand(rng);
        wadd_table(w,x,y,half_x,half_y,top_z);
    }
    else {
        const float half_x=0.60f+0.20f*wurand(rng);
        const float half_y=0.80f+0.20f*wurand(rng);
        const float height=1.35f+0.80f*wurand(rng);
        wadd_box(w,wv(x,y,height*0.5f),wv(half_x,half_y,height*0.5f));
    }
}
WF void wgenerate_mixed(WP WWorld& w,WT uint& rng,float distance) {
    const float table_x=distance*(0.46f+0.08f*wurand(rng));
    const float table_y=wroute_y(w,table_x,distance)+(wurand(rng)-0.5f)*0.16f;
    wadd_table(w,table_x,table_y,0.58f+0.10f*wurand(rng),0.76f+0.10f*wurand(rng),1.62f+0.42f*wurand(rng));

    const float counter_x=distance*0.22f;
    const float counter_y=wroute_y(w,counter_x,distance)-1.30f;
    const float counter_h=1.55f+0.45f*wurand(rng);
    wadd_box(w,wv(counter_x,counter_y,counter_h*0.5f),wv(0.30f,0.50f,counter_h*0.5f));

    const float sphere_x=distance*0.72f;
    const float sphere_y=wroute_y(w,sphere_x,distance)+1.35f;
    const float sphere_r=0.34f+0.10f*wurand(rng);
    const float sphere_z=1.05f+1.10f*wurand(rng);
    const float sphere_vy=(wurand(rng)<0.5f?-1.0f:1.0f)*(0.18f+0.16f*wurand(rng));
    wadd(w,1,wv(sphere_x,sphere_y,sphere_z),wv(sphere_r,sphere_r,sphere_r),wv(0,sphere_vy,0));

    const float pole_r0=0.20f+0.08f*wurand(rng),pole_r1=0.20f+0.08f*wurand(rng);
    wadd(w,2,wv(distance*0.34f,wroute_y(w,distance*0.34f,distance)+2.05f,2.5f),wv(pole_r0,pole_r0,2.5f),wv(0,0,0));
    wadd(w,2,wv(distance*0.82f,wroute_y(w,distance*0.82f,distance)-2.05f,2.5f),wv(pole_r1,pole_r1,2.5f),wv(0,0,0));

    const float low_box_x=distance*0.74f;
    const float low_box_y=wroute_y(w,low_box_x,distance)-0.88f;
    wadd_box(w,wv(low_box_x,low_box_y,0.48f),wv(0.34f,0.30f,0.48f));
}
WF void wgenerate(WP WWorld& w,uint seed,uint family,float distance) {
    uint rng=seed?seed:1; w.seed=seed;w.family=family;w.count=family==0?0:8;
    w.goal[0]=distance; w.goal[1]=(wurand(rng)-0.5f)*2; w.goal[2]=1.5f+(wurand(rng)-0.5f)*0.8f;
    for(uint j=0;j<3;j++) w.wind[j]=0;
    if(family==4) { w.count=0; wgenerate_doorway(w,rng,distance); return; }
    if(family==5) { w.count=0; wgenerate_table_or_counter(w,rng,distance); return; }
    if(family==6) { w.count=0; wgenerate_mixed(w,rng,distance); return; }
    if(family==8) { w.count=0; wgenerate_two_doorways(w,rng,distance); return; }
    for(uint i=0;i<16;i++) {
        WP WObstacle& o=w.obstacles[i]; o.kind=family==2?2:(family==3?1:0);
        o.center[0]=1.2f+wurand(rng)*fmax(0.1f,distance-2.2f);
        o.center[1]=(wurand(rng)-0.5f)*7; o.center[2]=family==2?2.5f:0.6f+wurand(rng)*3;
        o.size[0]=0.2f+wurand(rng)*0.35f;o.size[1]=0.3f+wurand(rng)*0.55f;o.size[2]=family==2?2.5f:0.2f+wurand(rng)*0.75f;
        o.velocity[0]=0;o.velocity[1]=family==3?(wurand(rng)-0.5f)*0.8f:0;o.velocity[2]=0;
    }
}
#undef WP
#undef WF
#undef WT
