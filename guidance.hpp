#pragma once

#ifdef __METAL_VERSION__
#define NAV_INLINE inline
#define NAV_THREAD thread
#else
#include <cmath>
#include <algorithm>
#include <cstdint>
#define NAV_INLINE inline
#define NAV_THREAD
using uint=uint32_t;
using std::sqrt; using std::log; using std::fmin; using std::fmax;
#endif

// 8x10 ranges are 2x2 min-pooled from the 16x20 FLU pinhole image.
// Range values are metres. Columns run left-to-right; rows run up-to-down.
NAV_INLINE void nav_ray(uint row,uint col,NAV_THREAD float* d) {
    const float y=0.9f-0.2f*float(col),z=0.65625f-0.1875f*float(row);
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
                         NAV_THREAD float* hint) {
    float gnorm=sqrt(goal_body_unit[0]*goal_body_unit[0]+goal_body_unit[1]*goal_body_unit[1]+goal_body_unit[2]*goal_body_unit[2]);
    if(gnorm<1e-6f||goal_distance<0.05f){hint[0]=hint[1]=hint[2]=0;return;}
    float goal[3]={goal_body_unit[0]/gnorm,goal_body_unit[1]/gnorm,goal_body_unit[2]/gnorm};
    const float speed=sqrt(body_velocity[0]*body_velocity[0]+body_velocity[1]*body_velocity[1]+body_velocity[2]*body_velocity[2]);
    const float stop_margin=0.28f+speed*fmax(sensor_dt,0.0f)+0.25f*speed*speed;
    float best_score=-1e20f,best_dir[3]={1,0,0},best_range=0;
    float goal_cell_dot=-1e20f;uint goal_row=0,goal_col=0;
    for(uint row=0;row<8;row++)for(uint col=0;col<10;col++) {
        float d[3];nav_ray(row,col,d);
        const float alignment=d[0]*goal[0]+d[1]*goal[1]+d[2]*goal[2];
        const float nearby=nav_min3x3(current_range,row,col);
        float risk=0.0f;
        if(sensor_dt>1e-4f)for(int dr=-1;dr<=1;dr++)for(int dc=-1;dc<=1;dc++) {
            int r=int(row)+dr,c=int(col)+dc;if(r<0||r>=8||c<0||c>=10)continue;
            const uint k=uint(r*10+c);float old=previous_range[k],now=current_range[k];
            if(old<11.9f&&now<11.9f) {
                float ray[3];nav_ray(uint(r),uint(c),ray);
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
    for(uint i=0;i<3;i++)hint[i]=nav_atanh(fraction*direction[i]);
}

#undef NAV_INLINE
#undef NAV_THREAD
