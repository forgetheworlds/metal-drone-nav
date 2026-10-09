#pragma once
#ifndef __METAL_VERSION__
#include <cmath>
#endif
// Fixed experimental variance profile, not an inference command guard.
// 0.25 near the 0.35m hold region; smoothly return to1 by three goal radii.
inline float nav_goal_noise_scale(float distance_m) {
    if(distance_m<=0.35f)return 0.25f;
    if(distance_m>=1.05f)return 1.0f;
    const float t=(distance_m-0.35f)/0.70f;
    return 0.25f+0.75f*t*t*(3.0f-2.0f*t);
}
inline float nav_goal_log_std_offset(float distance_m) {
#ifdef __METAL_VERSION__
    return log(nav_goal_noise_scale(distance_m));
#else
    return std::log(nav_goal_noise_scale(distance_m));
#endif
}
