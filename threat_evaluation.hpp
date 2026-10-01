#pragma once

#include "world.hpp"

// Host-side setup for controlled threat evaluations. The generated TTC is the
// nominal straight-line encounter time, not the policy's measured TTC.
enum class ThreatKind : uint32_t { Approach = 0, LateralCrossing = 1 };

inline bool threat_inside_room(WVec p,float margin) {
    return p.x-margin>=-2.0f && p.x+margin<=14.0f &&
           p.y-margin>=-5.0f && p.y+margin<=5.0f &&
           p.z-margin>=0.0f && p.z+margin<=5.0f;
}

// Mutate a generated world after initial GPU reset, before the first rollout. The drone
// starts at (0,0,1.5), targets (4,0,1.5), and has effective radius .18 m.
// The host decides whether initial velocity is zero or drone_speed; TTC remains
// a nominal construction parameter, not a measurement of the closed-loop path.
// Return false without changing `world` if the setup starts overlapped, the
// sphere path intersects existing geometry, or the sphere leaves the room.
inline bool add_threat_sphere(WWorld& world,uint seed,ThreatKind kind,
                              float drone_speed,float threat_speed,float nominal_ttc,
                              uint& obstacle_index) {
    constexpr float drone_radius=0.18f;
    constexpr float sphere_radius=0.35f;
    constexpr float combined_radius=drone_radius+sphere_radius;
    if((kind!=ThreatKind::Approach && kind!=ThreatKind::LateralCrossing) ||
       drone_speed<=0.0f || threat_speed<=0.0f || nominal_ttc<=0.0f || world.count>=16) return false;

    WWorld candidate=world;
    candidate.goal[0]=4.0f;candidate.goal[1]=0.0f;candidate.goal[2]=1.5f;
    const WVec start=wv(0.0f,0.0f,1.5f);
    if(wclearance(candidate,start,0.0f)<=0.0f) return false;

    // Keep deterministic offsets small enough to preserve the intended encounter.
    uint rng=(seed^0x9e3779b9u);if(rng==0)rng=1;
    const float lateral=(wurand(rng)-0.5f)*0.20f; // +/- 0.10 m
    const float vertical=(wurand(rng)-0.5f)*0.16f; // +/- 0.08 m
    WVec center,velocity=wv(0.0f,0.0f,0.0f);
    if(kind==ThreatKind::Approach) {
        center=wv((drone_speed+threat_speed)*nominal_ttc+combined_radius,
                  lateral,1.5f+vertical);
        velocity.x=-threat_speed;
    }
    else {
        center=wv(drone_speed*nominal_ttc,
                  -threat_speed*nominal_ttc+lateral,1.5f+vertical);
        velocity.y=threat_speed;
    }
    if(!threat_inside_room(center,combined_radius)) return false;
    const float start_dx=center.x-start.x,start_dy=center.y-start.y,start_dz=center.z-start.z;
    if(sqrt(start_dx*start_dx+start_dy*start_dy+start_dz*start_dz)<=combined_radius) return false;

    // Check the moving sphere against existing geometry at several points on
    // its nominal path. wclearance already includes the drone radius.
    for(uint sample=0;sample<=8;sample++) {
        const float t=nominal_ttc*(float(sample)/8.0f);
        const WVec p=wa(center,wm(velocity,t));
        if(!threat_inside_room(p,combined_radius) ||
           wclearance(candidate,p,t)<=sphere_radius) return false;
    }

    obstacle_index=candidate.count;
    WObstacle& obstacle=candidate.obstacles[candidate.count++];
    obstacle.kind=1;
    obstacle.center[0]=center.x;obstacle.center[1]=center.y;obstacle.center[2]=center.z;
    obstacle.size[0]=sphere_radius;obstacle.size[1]=sphere_radius;obstacle.size[2]=sphere_radius;
    obstacle.velocity[0]=velocity.x;obstacle.velocity[1]=velocity.y;obstacle.velocity[2]=velocity.z;
    world=candidate;
    return true;
}
