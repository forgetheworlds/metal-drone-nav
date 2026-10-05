#pragma once

// perception_support.hpp — portable geometry-prior support module (the one
// mechanism of the perception-support experiment).
//
// The core guidance (guidance.hpp) answers "how far clear is this direction?"
// with nav_memory_clearance(), which initialises3.0 and only ever REDUCES it
// for retained range hits. DirectionsWithout any retained hit — rear/side
// space outside the swept camera frustum — therefore read as "clear to3 m"
// (Root's probe: results/root-contract-review/unknown-clearance.*). In
// nav_guidance_memory those unknown directions keep the full free-space bonus
// and no deficit penalty, so an unobserved lateral/back direction can out-score
// a measured-but-close goal direction, and mode17/22 execute
// prior + scale*(actor - prior) — a direct path to blind lateral/backward
// commands.
//
// This module keeps the core semantics where measurements exist and changes
// ONLY the unmeasured case:
//   - measured obstacle hit      -> same as core (nearest, capped at3 m)
//   - usable measured depth supports direction -> a bounded free-range estimate
//   - outside the frustum with no retained hit -> UNKNOWN (<0)
// Unknown is NOT occupied (movement stays allowed: inspect/turn/backtrack and
// narrow corridors are preserved) but it earns no free-space bonus, no deficit
// credit and only a capped prior speed — cautious, not blocked.
// No simulator/world/collisiontruth enters this code: inputs are measured
// range rings, recorded ego poses, current frustum/mount profile, timestamps
// (frame/valid counts) and the observation context, exactly as the core prior.
//
// The header is dual-use like guidance.hpp: host tests/probe and the Metal
// source string (appended after base_source()) share ONE definition. The
// NAV/frontend can include this file directly for deployment (calibrated
// owner reviews the interface; no edits in their worktree).

#ifdef __METAL_VERSION__
#define PS_INLINE inline
#define PS_THREAD thread
#define PS_DEVICE device
#else
#include <cmath>
#include <algorithm>
#include <cstdint>
#define PS_INLINE inline
#define PS_THREAD
#define PS_DEVICE
using std::sqrt; using std::fabs; using std::fmin; using std::fmax;
using uint=uint32_t;
#endif

// Sentinel for "no measurement for this direction" (never a clearance value).
#ifdef __METAL_VERSION__
constant float PS_UNKNOWN=-1.0f;
#else
static constexpr float PS_UNKNOWN=-1.0f;
#endif
// In-frustum measured free bound when no obstacle was hit (core-compatible
//3.0 cap; also the score cap via min(free,3)).
#ifdef __METAL_VERSION__
constant float PS_MEASURED_FREE=3.0f;
#else
static constexpr float PS_MEASURED_FREE=3.0f;
#endif
// Prior command speed cap when the selected direction is UNKNOWN
// (fraction = speed*0.5 <=0.25). Motion allowed, deliberately slow.
#ifdef __METAL_VERSION__
constant float PS_UNKNOWN_SPEED_CAP=0.5f;
#else
static constexpr float PS_UNKNOWN_SPEED_CAP=0.5f;
#endif

// Body-frame frustum test for the ACTIVE legacy profile (tan_h=1.0 both
// profiles; tan_v passed from sensor_profile). Mount offset is along +x and
// does not change the direction cone.
PS_INLINE bool ps_in_current_frustum(PS_THREAD const float* dir,float tan_v) {
    return dir[0]>0.0f&&fabs(dir[1])<=dir[0]&&fabs(dir[2])<=tan_v*dir[0];
}

// Same swept-sphere hit query as guidance.hpp::nav_memory_clearance, but the
// no-hit result is UNKNOWN instead of an implicit "clear3 m". Returns PS_UNKNOWN
// or the nearest measured distance (<=3 m).
PS_INLINE float ps_memory_clearance(PS_THREAD const float* direction,
                                    PS_DEVICE const float* range_ring,
                                    PS_DEVICE const float* pose_ring,
                                    PS_THREAD const float* current_pose,
                                    uint latest_frame,uint valid_frames,float sensor_dt,float tan_v,float mount_x=0.0f) {
    float nearest=3.0f;
    bool measured=false;
    for(uint back=0;back<valid_frames&&back<8;back++) {
        const uint frame=(latest_frame+8u-back)%8u;
        PS_DEVICE const float* old_pose=pose_ring+frame*12;
        PS_DEVICE const float* ranges=range_ring+frame*320;
        const float age=float(back)*fmax(sensor_dt,0.0f);
        for(uint row=0;row<8;row++)for(uint col=0;col<10;col++) {
            const uint px=row*2*20+col*2;
            uint hit=px;float range=ranges[px];uint pixels[3]={px+1,px+20,px+21};
            for(uint j=0;j<3;j++)if(ranges[pixels[j]]<range){range=ranges[pixels[j]];hit=pixels[j];}
            if(range<=0.01f||range>=11.9f)continue;
            float ry=1.0f-(float(hit%20)+.5f)/10.0f,rz=tan_v*(1.0f-(float(hit/20)+.5f)/8.0f);
            float inv=1.0f/sqrt(1+ry*ry+rz*rz);float ray[3]={inv,ry*inv,rz*inv};
            const float wx=old_pose[3]*ray[0]+old_pose[4]*ray[1]+old_pose[5]*ray[2];
            const float wy=old_pose[6]*ray[0]+old_pose[7]*ray[1]+old_pose[8]*ray[2];
            const float wz=old_pose[9]*ray[0]+old_pose[10]*ray[1]+old_pose[11]*ray[2];
            const float dx=old_pose[0]+mount_x*old_pose[3]+range*wx-current_pose[0];
            const float dy=old_pose[1]+mount_x*old_pose[6]+range*wy-current_pose[1];
            const float dz=old_pose[2]+mount_x*old_pose[9]+range*wz-current_pose[2];
            const float x=current_pose[3]*dx+current_pose[6]*dy+current_pose[9]*dz;
            const float y=current_pose[4]*dx+current_pose[7]*dy+current_pose[10]*dz;
            const float z=current_pose[5]*dx+current_pose[8]*dy+current_pose[11]*dz;
            const float along=x*direction[0]+y*direction[1]+z*direction[2];
            const float radius=0.20f+0.015f*range+0.15f*age;
            if(along<=0.0f||along>3.0f+radius)continue;
            const float lateral2=fmax(x*x+y*y+z*z-along*along,0.0f);
            const float radius2=radius*radius;
            if(lateral2<radius2){nearest=fmin(nearest,fmax(along-sqrt(radius2-lateral2),0.0f));measured=true;}
        }
    }
    return measured?nearest:PS_UNKNOWN;
}

// Free-space evidence from the newest usable depth frame. Reproject the
// candidate corridor into the camera pose that captured that frame; current
// body frustum membership alone is not evidence when the sensor is delayed.
// This is a sampled range bound, not a certificate for an entire swept body.
PS_INLINE float ps_measured_free_bound(PS_THREAD const float* direction,
                                      PS_DEVICE const float* range_ring,
                                      PS_DEVICE const float* pose_ring,
                                      PS_THREAD const float* current_pose,
                                      uint latest_frame,uint valid_frames,
                                      float tan_v,float mount_x=0.0f) {
    if(valid_frames==0)return PS_UNKNOWN;
    const uint frame=latest_frame%8u;
    PS_DEVICE const float* capture=pose_ring+frame*12;
    PS_DEVICE const float* ranges=range_ring+frame*320;
    const float mount=mount_x;
    float world_direction[3];
    for(uint i=0;i<3;i++)world_direction[i]=current_pose[3+i*3]*direction[0]
        +current_pose[4+i*3]*direction[1]+current_pose[5+i*3]*direction[2];
    float bound=3.0f;
    const float distances[2]={0.4f,3.0f};
    for(uint sample=0;sample<2;sample++) {
        float offset[3];
        for(uint i=0;i<3;i++)offset[i]=current_pose[i]+distances[sample]*world_direction[i]
            -capture[i]-mount*capture[3+i*3];
        float point[3];
        for(uint i=0;i<3;i++)point[i]=capture[3+i]*offset[0]
            +capture[6+i]*offset[1]+capture[9+i]*offset[2];
        if(!ps_in_current_frustum(point,tan_v))return PS_UNKNOWN;
        const float image_x=(1.0f-point[1]/point[0])*10.0f;
        const float image_y=(1.0f-point[2]/(point[0]*tan_v))*8.0f;
        const int col=int(fmin(19.0f,fmax(0.0f,image_x)));
        const int row=int(fmin(15.0f,fmax(0.0f,image_y)));
        for(int dy=-1;dy<=1;dy++)for(int dx=-1;dx<=1;dx++) {
            const int y=row+dy,x=col+dx;
            if(y<0||y>=16||x<0||x>=20)continue;
            const float range=ranges[uint(y*20+x)];
            if(!(range>0.03f&&range<=12.0001f))return PS_UNKNOWN;
            bound=fmin(bound,range);
        }
    }
    float travel2=0.0f;
    for(uint i=0;i<3;i++) {
        const float delta=current_pose[i]-capture[i]-mount*capture[3+i*3];
        travel2+=delta*delta;
    }
    return fmax(0.0f,bound-sqrt(travel2));
}

// Corrected history-aware prior. Structurally identical to
// guidance.hpp::nav_guidance_memory with THREE marked deltas (search PS:):
//   PS1 free-space classification (measured / in-frustum / UNKNOWN)
//   PS2 scoring: UNKNOWN is neutral (no bonus, no deficit) — not "clear"
//   PS3 speed: an UNKNOWN best direction is capped at PS_UNKNOWN_SPEED_CAP
// Candidate set, velocity sweep, goal speed cap and atanh output are unchanged.
// This experimental profile changes how sampled measurement support is classified.
PS_INLINE void ps_nav_guidance_memory(PS_THREAD const float* current_range,
                                   PS_THREAD const float* previous_range,
                                   PS_THREAD const float* goal_body_unit,
                                   float goal_distance,
                                   PS_THREAD const float* body_velocity,
                                   float sensor_dt,
                                   float tan_v,
                                   PS_DEVICE const float* range_ring,
                                   PS_DEVICE const float* pose_ring,
                                   PS_THREAD const float* current_pose,
                                   uint latest_frame,uint valid_frames,
                                   PS_THREAD float* hint,float mount_x=0.0f,
                                   PS_DEVICE const float* evidence=nullptr) {
    float gnorm=sqrt(goal_body_unit[0]*goal_body_unit[0]+goal_body_unit[1]*goal_body_unit[1]+goal_body_unit[2]*goal_body_unit[2]);
    if(gnorm<1e-6f||goal_distance<0.05f){hint[0]=hint[1]=hint[2]=0;return;}
    float goal[3]={goal_body_unit[0]/gnorm,goal_body_unit[1]/gnorm,goal_body_unit[2]/gnorm};
    const float speed=sqrt(body_velocity[0]*body_velocity[0]+body_velocity[1]*body_velocity[1]+body_velocity[2]*body_velocity[2]);
    const float stop_margin=0.28f+speed*fmax(sensor_dt,0.0f)+0.25f*speed*speed;
    float goal_cell_dot=-1e20f;uint goal_row=0,goal_col=0;
    for(uint row=0;row<8;row++)for(uint col=0;col<10;col++){float d[3];nav_ray(row,col,tan_v,d);float a=d[0]*goal[0]+d[1]*goal[1]+d[2]*goal[2];if(a>goal_cell_dot){goal_cell_dot=a;goal_row=row;goal_col=col;}}
    const float desired_speed=fmin(1.6f,sqrt(4.0f*fmax(goal_distance-0.28f,0.0f)));
    float best_score=-1e20f,best_free=0.0f,best_dir[3]={1,0,0};bool best_unknown=false;
    for(uint candidate=0;candidate<85;candidate++) {
        float d[3];uint row=goal_row,col=goal_col;
        if(candidate<80){row=candidate/10;col=candidate%10;nav_ray(row,col,tan_v,d);}
        else if(candidate==80){d[0]=goal[0];d[1]=goal[1];d[2]=goal[2];}else{d[0]=0;d[1]=candidate>=83?(candidate==83?1.0f:-1.0f):0;d[2]=candidate<83?(candidate==81?1.0f:-1.0f):0;}
        float sweep[3]={1.2f*d[0]+0.35f*body_velocity[0],1.2f*d[1]+0.35f*body_velocity[1],1.2f*d[2]+0.35f*body_velocity[2]};float sweep_norm=sqrt(sweep[0]*sweep[0]+sweep[1]*sweep[1]+sweep[2]*sweep[2]);for(uint j=0;j<3;j++)sweep[j]/=fmax(sweep_norm,1e-6f);
        float clearance=evidence?evidence[candidate*2]:
            ps_memory_clearance(sweep,range_ring,pose_ring,current_pose,latest_frame,valid_frames,sensor_dt,tan_v,mount_x);
        const float observed=nav_min3x3(current_range,row,col);
        // Occupancy evidence and sampled free-space support are distinct.
        // Never borrow a forward image cell for a rear/side goal candidate.
        const float support=evidence?evidence[candidate*2+1]:
            ps_measured_free_bound(sweep,range_ring,pose_ring,current_pose,
                                   latest_frame,valid_frames,tan_v,mount_x);
        float free=clearance;
        if(support>=0.0f)free=free>=0.0f?fmin(free,support):support;
        if(free>=0.0f&&candidate<81&&ps_in_current_frustum(d,tan_v))
            free=fmin(free,observed);
        const bool unknown=free<0.0f;
        float temporal_risk=0.0f;
        if(!unknown&&ps_in_current_frustum(d,tan_v)&&sensor_dt>1e-4f)for(int dr=-1;dr<=1;dr++)for(int dc=-1;dc<=1;dc++){
            int r=int(row)+dr,c=int(col)+dc;if(r<0||r>=8||c<0||c>=10)continue;uint k=uint(r*10+c);
            float old=previous_range[k],now=current_range[k];if(old<11.9f&&now<11.9f){float ray[3];nav_ray(uint(r),uint(c),tan_v,ray);float closing=(old-now)/fmax(sensor_dt,1e-3f)-(body_velocity[0]*ray[0]+body_velocity[1]*ray[1]+body_velocity[2]*ray[2]);if(closing>0.5f)temporal_risk=fmax(temporal_risk,fmin(0.20f*closing,1.0f));}
        }
        if(!unknown)free=fmax(free-temporal_risk,0.0f);
        const float alignment=d[0]*goal[0]+d[1]*goal[1]+d[2]*goal[2];
        const float required=stop_margin+0.15f+desired_speed*desired_speed*.25f;
        // PS2: UNKNOWN is neutral — no free bonus, no deficit penalty.
        const float score=unknown
            ?1.4f*alignment
            :1.4f*alignment+0.20f*fmin(free,3.0f)-(free<required?3.0f*(required-free)/required:0.0f);
        if(score>best_score){best_score=score;best_free=free;best_dir[0]=d[0];best_dir[1]=d[1];best_dir[2]=d[2];best_unknown=unknown;}
    }
    const bool goal_in_fov=goal[0]>0.0f&&fabs(goal[1])<=goal[0]&&fabs(goal[2])<=tan_v*goal[0];
    float command_speed;
    if(best_unknown) {
        // PS3: cautious translation into unmeasured space (never blocked).
        command_speed=PS_UNKNOWN_SPEED_CAP;
    } else {
        const float after_stop=fmax(best_free-stop_margin-0.15f,0.0f);
        command_speed=fmin(desired_speed,sqrt(4.0f*after_stop));
    }
    if(!goal_in_fov)command_speed=fmin(command_speed,0.25f);
    const float fraction=fmin(0.8f,command_speed*0.5f);
    // atanh-space hint, identical output contract to guidance.hpp.
    for(uint i=0;i<3;i++)hint[i]=nav_atanh(fraction*best_dir[i]);
}
