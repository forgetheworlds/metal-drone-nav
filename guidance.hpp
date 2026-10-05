#pragma once

#ifdef __METAL_VERSION__
#define NAV_INLINE inline
#define NAV_THREAD thread
#define NAV_DEVICE device
#else
#include <cmath>
#include <algorithm>
#include <cstdint>
#define NAV_INLINE inline
#define NAV_THREAD
#define NAV_DEVICE
using uint=uint32_t;
using std::sqrt; using std::log; using std::fmin; using std::fmax;
#endif

// 8x10 ranges are 2x2 min-pooled from the 16x20 FLU pinhole image.
// Range values are metres. Columns run left-to-right; rows run up-to-down.
// `tan_v` is the camera profile's vertical half-tangent (sensor_profile.hpp);
// 0.75 is the legacy source model, 0.8 is the measured native RangeFinder.
NAV_INLINE void nav_ray(uint row,uint col,float tan_v,NAV_THREAD float* d) {
    const float y=0.9f-0.2f*float(col),z=(0.875f-0.25f*float(row))*tan_v;
    const float inv=1.0f/sqrt(1.0f+y*y+z*z);
    d[0]=inv;d[1]=y*inv;d[2]=z*inv;
}
NAV_INLINE float nav_min3x3(NAV_THREAD const float* range,uint row,uint col) {
    float nearest=12.0f;
    for(int dr=-1;dr<=1;dr++)for(int dc=-1;dc<=1;dc++) {
        int r=int(row)+dr,c=int(col)+dc;
        if(r>=0&&r<8&&c>=0&&c<10)nearest=fmin(nearest,range[r*10+c]);
    }
    return nearest;
}
NAV_INLINE float nav_atanh(float x) { return 0.5f*log((1.0f+x)/(1.0f-x)); }

// Return an atanh-space xyz mean hint. The actor's tanh maps this to a
// body-frame velocity fraction. The total command speed is capped at 0.8.
NAV_INLINE void nav_guidance(NAV_THREAD const float* current_range,
                         NAV_THREAD const float* previous_range,
                         NAV_THREAD const float* goal_body_unit,
                         float goal_distance,
                         NAV_THREAD const float* body_velocity,
                         float sensor_dt,
                         float tan_v,
                         NAV_THREAD float* hint) {
    float gnorm=sqrt(goal_body_unit[0]*goal_body_unit[0]+goal_body_unit[1]*goal_body_unit[1]+goal_body_unit[2]*goal_body_unit[2]);
    if(gnorm<1e-6f||goal_distance<0.05f){hint[0]=hint[1]=hint[2]=0;return;}
    float goal[3]={goal_body_unit[0]/gnorm,goal_body_unit[1]/gnorm,goal_body_unit[2]/gnorm};
    const float speed=sqrt(body_velocity[0]*body_velocity[0]+body_velocity[1]*body_velocity[1]+body_velocity[2]*body_velocity[2]);
    const float stop_margin=0.28f+speed*fmax(sensor_dt,0.0f)+0.25f*speed*speed;
    float best_score=-1e20f,best_dir[3]={1,0,0},best_range=0;
    float goal_cell_dot=-1e20f;uint goal_row=0,goal_col=0;
    for(uint row=0;row<8;row++)for(uint col=0;col<10;col++) {
        float d[3];nav_ray(row,col,tan_v,d);
        const float alignment=d[0]*goal[0]+d[1]*goal[1]+d[2]*goal[2];
        const float nearby=nav_min3x3(current_range,row,col);
        float risk=0.0f;
        if(sensor_dt>1e-4f)for(int dr=-1;dr<=1;dr++)for(int dc=-1;dc<=1;dc++) {
            int r=int(row)+dr,c=int(col)+dc;if(r<0||r>=8||c<0||c>=10)continue;
            const uint k=uint(r*10+c);float old=previous_range[k],now=current_range[k];
            if(old<11.9f&&now<11.9f) {
                float ray[3];nav_ray(uint(r),uint(c),tan_v,ray);
                float unexpected=(old-now)/fmax(sensor_dt,1e-3f)-(body_velocity[0]*ray[0]+body_velocity[1]*ray[1]+body_velocity[2]*ray[2]);
                if(unexpected>0.5f)risk=fmax(risk,fmin(0.20f*unexpected,1.0f));
            }
        }
        const float safe_range=fmax(nearby-risk,0.0f);
        const float score=1.4f*alignment+0.12f*fmin(safe_range,6.0f);
        if(score>best_score){best_score=score;best_dir[0]=d[0];best_dir[1]=d[1];best_dir[2]=d[2];best_range=safe_range;}
        if(alignment>goal_cell_dot){goal_cell_dot=alignment;goal_row=row;goal_col=col;}
    }
    const float max_speed=1.6f; // 0.8 of the simulator's 2 m/s command scale.
    const float desired_speed=fmin(max_speed,sqrt(4.0f*fmax(goal_distance-0.28f,0.0f)));
    const float direct_range=fmax(nav_min3x3(current_range,goal_row,goal_col),0.0f);
    float direction[3]={best_dir[0],best_dir[1],best_dir[2]},available=best_range;
    if(goal[0]>0.0f && direct_range>stop_margin+0.25f*desired_speed*desired_speed) {
        direction[0]=goal[0];direction[1]=goal[1];direction[2]=goal[2];available=direct_range;
    }
    const float after_stop=fmax(available-stop_margin-0.15f,0.0f);
    const float command_speed=fmin(desired_speed,sqrt(4.0f*after_stop));
    const float fraction=fmin(0.8f,command_speed*0.5f);
    // Give sideways/upward clearance time before advancing into a close surface.
    if(direct_range<1.2f && available>direct_range+0.25f){
        const float advance=fmax(0.05f,fmin(1.0f,(direct_range-0.25f)/0.95f));direction[0]*=advance;
        const float norm=sqrt(direction[0]*direction[0]+direction[1]*direction[1]+direction[2]*direction[2]);
        for(uint i=0;i<3;i++)direction[i]/=fmax(norm,1e-6f);
    }
    for(uint i=0;i<3;i++)hint[i]=nav_atanh(fraction*direction[i]);
}

// Preserve frozen policy behavior unless the diagnostic profile opts in.
#ifndef NAV_MEMORY_USE_CAPTURE_AGE
#define NAV_MEMORY_USE_CAPTURE_AGE 0
#endif
NAV_INLINE float nav_memory_point_radius(float range,float history_age,float latest_capture_age) {
    const float age=fmax(history_age,0.0f)+
        (NAV_MEMORY_USE_CAPTURE_AGE?fmax(latest_capture_age,0.0f):0.0f);
    return 0.20f+0.015f*range+0.15f*age;
}

// Return the first swept-sphere contact along a body-frame direction using only
// range hits and recorded ego poses. Pose layout: world xyz, then body-to-world
// row-major rotation. Pooled rays stand for their 2x2 source pixels; the radius
// term covers body size, pooling uncertainty, and age uncertainty.
NAV_INLINE float nav_memory_clearance(NAV_THREAD const float* direction,
                                      NAV_DEVICE const float* range_ring,
                                      NAV_DEVICE const float* pose_ring,
                                      NAV_THREAD const float* current_pose,
                                      uint latest_frame,uint valid_frames,float sensor_dt,float tan_v,float latest_capture_age=0.0f) {
    float nearest=3.0f;
    for(uint back=0;back<valid_frames&&back<8;back++) {
        const uint frame=(latest_frame+8u-back)%8u;
        NAV_DEVICE const float* old_pose=pose_ring+frame*12;
        NAV_DEVICE const float* ranges=range_ring+frame*320;
        const float age=float(back)*fmax(sensor_dt,0.0f);
        for(uint row=0;row<8;row++)for(uint col=0;col<10;col++) {
            const uint px=row*2*20+col*2;
            uint hit=px;float range=ranges[px];uint pixels[3]={px+1,px+20,px+21};for(uint j=0;j<3;j++)if(ranges[pixels[j]]<range){range=ranges[pixels[j]];hit=pixels[j];}
            if(range<=0.01f||range>=11.9f)continue;
            float ry=1.0f-(float(hit%20)+.5f)/10.0f,rz=tan_v*(1.0f-(float(hit/20)+.5f)/8.0f);float inv=1.0f/sqrt(1+ry*ry+rz*rz);float ray[3]={inv,ry*inv,rz*inv};
            const float wx=old_pose[3]*ray[0]+old_pose[4]*ray[1]+old_pose[5]*ray[2];
            const float wy=old_pose[6]*ray[0]+old_pose[7]*ray[1]+old_pose[8]*ray[2];
            const float wz=old_pose[9]*ray[0]+old_pose[10]*ray[1]+old_pose[11]*ray[2];
            const float dx=old_pose[0]+range*wx-current_pose[0];
            const float dy=old_pose[1]+range*wy-current_pose[1];
            const float dz=old_pose[2]+range*wz-current_pose[2];
            const float x=current_pose[3]*dx+current_pose[6]*dy+current_pose[9]*dz;
            const float y=current_pose[4]*dx+current_pose[7]*dy+current_pose[10]*dz;
            const float z=current_pose[5]*dx+current_pose[8]*dy+current_pose[11]*dz;
            const float along=x*direction[0]+y*direction[1]+z*direction[2];
            const float radius=nav_memory_point_radius(range,age,latest_capture_age);
            if(along<=0.0f||along>3.0f+radius)continue;
            const float lateral2=fmax(x*x+y*y+z*z-along*along,0.0f);
            const float radius2=radius*radius;
            if(lateral2<radius2)nearest=fmin(nearest,fmax(along-sqrt(radius2-lateral2),0.0f));
        }
    }
    return nearest;
}

// History-aware prior. latest_frame is the absolute ring frame that contains
// the newest usable observation; valid_frames counts captured frames since
// reset, capped at 8. Range and pose rings use the matching frame modulo 8.
NAV_INLINE void nav_guidance_memory(NAV_THREAD const float* current_range,
                                   NAV_THREAD const float* previous_range,
                                   NAV_THREAD const float* goal_body_unit,
                                   float goal_distance,
                                   NAV_THREAD const float* body_velocity,
                                   float sensor_dt,
                                   float tan_v,
                                   NAV_DEVICE const float* range_ring,
                                   NAV_DEVICE const float* pose_ring,
                                   NAV_THREAD const float* current_pose,
                                   uint latest_frame,uint valid_frames,
                                   NAV_THREAD float* hint,NAV_DEVICE const float* cached_clearance=nullptr,float latest_capture_age=0.0f) {
    float gnorm=sqrt(goal_body_unit[0]*goal_body_unit[0]+goal_body_unit[1]*goal_body_unit[1]+goal_body_unit[2]*goal_body_unit[2]);
    if(gnorm<1e-6f||goal_distance<0.05f){hint[0]=hint[1]=hint[2]=0;return;}
    float goal[3]={goal_body_unit[0]/gnorm,goal_body_unit[1]/gnorm,goal_body_unit[2]/gnorm};
    const float speed=sqrt(body_velocity[0]*body_velocity[0]+body_velocity[1]*body_velocity[1]+body_velocity[2]*body_velocity[2]);
    const float stop_margin=0.28f+speed*fmax(sensor_dt,0.0f)+0.25f*speed*speed;
    float goal_cell_dot=-1e20f;uint goal_row=0,goal_col=0;
    for(uint row=0;row<8;row++)for(uint col=0;col<10;col++){float d[3];nav_ray(row,col,tan_v,d);float a=d[0]*goal[0]+d[1]*goal[1]+d[2]*goal[2];if(a>goal_cell_dot){goal_cell_dot=a;goal_row=row;goal_col=col;}}
    const float desired_speed=fmin(1.6f,sqrt(4.0f*fmax(goal_distance-0.28f,0.0f)));
    float best_score=-1e20f,best_free=0.0f,best_dir[3]={1,0,0};
    for(uint candidate=0;candidate<85;candidate++) {
        float d[3];uint row=goal_row,col=goal_col;
        if(candidate<80){row=candidate/10;col=candidate%10;nav_ray(row,col,tan_v,d);}
        else if(candidate==80){d[0]=goal[0];d[1]=goal[1];d[2]=goal[2];}else{d[0]=0;d[1]=candidate>=83?(candidate==83?1.0f:-1.0f):0;d[2]=candidate<83?(candidate==81?1.0f:-1.0f):0;}
        float sweep[3]={1.2f*d[0]+0.35f*body_velocity[0],1.2f*d[1]+0.35f*body_velocity[1],1.2f*d[2]+0.35f*body_velocity[2]};float sweep_norm=sqrt(sweep[0]*sweep[0]+sweep[1]*sweep[1]+sweep[2]*sweep[2]);for(uint j=0;j<3;j++)sweep[j]/=fmax(sweep_norm,1e-6f);
        float free=cached_clearance?cached_clearance[candidate]:nav_memory_clearance(sweep,range_ring,pose_ring,current_pose,latest_frame,valid_frames,sensor_dt,tan_v,latest_capture_age);
        const float observed=nav_min3x3(current_range,row,col);
        if(candidate<81)free=fmin(free,observed);
        // The exact goal may fall between rays; the 3x3 min is a conservative cone check.
        float temporal_risk=0.0f;
        if(sensor_dt>1e-4f)for(int dr=-1;dr<=1;dr++)for(int dc=-1;dc<=1;dc++){
            int r=int(row)+dr,c=int(col)+dc;if(r<0||r>=8||c<0||c>=10)continue;uint k=uint(r*10+c);
            float old=previous_range[k],now=current_range[k];if(old<11.9f&&now<11.9f){float ray[3];nav_ray(uint(r),uint(c),tan_v,ray);float closing=(old-now)/fmax(sensor_dt,1e-3f)-(body_velocity[0]*ray[0]+body_velocity[1]*ray[1]+body_velocity[2]*ray[2]);if(closing>0.5f)temporal_risk=fmax(temporal_risk,fmin(0.20f*closing,1.0f));}
        }
        free=fmax(free-temporal_risk,0.0f);
        const float alignment=d[0]*goal[0]+d[1]*goal[1]+d[2]*goal[2];
        const float required=stop_margin+0.15f+desired_speed*desired_speed*.25f;
        const float score=1.4f*alignment+0.20f*fmin(free,3.0f)-(free<required?3.0f*(required-free)/required:0.0f);
        if(score>best_score){best_score=score;best_free=free;best_dir[0]=d[0];best_dir[1]=d[1];best_dir[2]=d[2];}
    }
    // Horizontal half-tangent is 1.0 for both profiles; the vertical half-FOV
    // follows the selected camera profile.
    const bool goal_in_fov=goal[0]>0.0f&&fabs(goal[1])<=goal[0]&&fabs(goal[2])<=tan_v*goal[0];
    const float after_stop=fmax(best_free-stop_margin-0.15f,0.0f);
    float command_speed=fmin(desired_speed,sqrt(4.0f*after_stop));
    if(!goal_in_fov)command_speed=fmin(command_speed,0.25f);
    const float fraction=fmin(0.8f,command_speed*0.5f);
    for(uint i=0;i<3;i++)hint[i]=nav_atanh(fraction*best_dir[i]);
}

#undef NAV_INLINE
#undef NAV_THREAD
#undef NAV_DEVICE
