#pragma once

// Seeded stress variation for the frozen L2F/Crazyflie plant. These ranges are
// experiment settings; they are not estimates of hardware uncertainty.
#ifdef __METAL_VERSION__
#define RL_DOMAIN_THREAD thread
typedef uint RlDomainIndex;
#else
#include "physics.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#define RL_DOMAIN_THREAD
typedef uint32_t RlDomainIndex;
#endif

struct RLPhysicsDomainRange {
    float mass_scale_min, mass_scale_max;
    float inertia_axis_scale_min, inertia_axis_scale_max;
    float thrust_gain_min, thrust_gain_max;
    float rising_lag_scale_min, rising_lag_scale_max;
    float falling_lag_scale_min, falling_lag_scale_max;
    float minimum_thrust_to_weight;
};

struct RLPhysicsDomainSample {
    float mass_scale;
    float inertia_axis_scale[3];
    float thrust_gain[4];
    float rising_lag_scale[4];
    float falling_lag_scale[4];
    float thrust_to_weight;
};

// Stress bounds for the current Crazyflie base plant. The base model remains
// unchanged when amplitude is zero. These bounds do not add unmeasured drag.
inline RLPhysicsDomainRange rl_physics_domain_stress_range() {
    return {0.90f,1.10f, 0.90f,1.10f, 0.90f,1.10f,
            0.75f,1.35f, 0.75f,1.35f, 1.15f};
}

inline RlDomainIndex rl_physics_domain_next(RL_DOMAIN_THREAD RlDomainIndex& state) {
    RlDomainIndex value=state?state:1;
    value^=value<<13;value^=value>>17;value^=value<<5;
    state=value?value:1;
    return state;
}

inline float rl_physics_domain_uniform(RL_DOMAIN_THREAD RlDomainIndex& state) {
    return float(rl_physics_domain_next(state)>>8)*(1.0f/16777216.0f);
}

inline float rl_physics_domain_scale(RL_DOMAIN_THREAD RlDomainIndex& state,
                                     float low,float high,float amplitude) {
    const float draw=low+(high-low)*rl_physics_domain_uniform(state);
    return 1.0f+amplitude*(draw-1.0f);
}

inline bool rl_physics_domain_inverse3(RL_DOMAIN_THREAD const float* matrix,
                                       RL_DOMAIN_THREAD float* inverse) {
    const float a=matrix[0],b=matrix[1],c=matrix[2];
    const float d=matrix[3],e=matrix[4],f=matrix[5];
    const float g=matrix[6],h=matrix[7],i=matrix[8];
    for(RlDomainIndex j=0;j<9;j++)if(!isfinite(matrix[j]))return false;
    if(fabs(b-d)>1.0e-6f||fabs(c-g)>1.0e-6f||fabs(f-h)>1.0e-6f)return false;
    const float minor2=a*e-b*d;
    const float determinant=a*(e*i-f*h)-b*(d*i-f*g)+c*(d*h-e*g);
    if(!(a>0.0f) || !(minor2>0.0f) || !(determinant>0.0f) || !isfinite(determinant))return false;
    const float reciprocal=1.0f/determinant;
    inverse[0]=(e*i-f*h)*reciprocal;
    inverse[1]=(c*h-b*i)*reciprocal;
    inverse[2]=(b*f-c*e)*reciprocal;
    inverse[3]=(f*g-d*i)*reciprocal;
    inverse[4]=(a*i-c*g)*reciprocal;
    inverse[5]=(c*d-a*f)*reciprocal;
    inverse[6]=(d*h-e*g)*reciprocal;
    inverse[7]=(b*g-a*h)*reciprocal;
    inverse[8]=(a*e-b*d)*reciprocal;
    for(RlDomainIndex j=0;j<9;j++)if(!isfinite(inverse[j]))return false;
    return true;
}

inline float rl_physics_domain_thrust_to_weight(RL_DOMAIN_THREAD const RLPhysicsParams& params) {
    float max_vertical_thrust=0.0f;
    const float action=params.action_max;
    for(RlDomainIndex rotor=0;rotor<4;rotor++){
        const RlDomainIndex i=3*rotor;
        RL_DOMAIN_THREAD const float* coefficient=params.rotor_thrust_coefficients+i;
        const float magnitude=coefficient[0]+coefficient[1]*action+coefficient[2]*action*action;
        max_vertical_thrust+=params.rotor_thrust_directions[i+2]*magnitude;
    }
    const float weight=params.mass*fabs(params.gravity_world[2]);
    return weight>0.0f?max_vertical_thrust/weight:0.0f;
}

// Shared host/MSL episode sampler. The caller supplies a separate per-episode
// seed, so no CPU work or hidden policy input is needed inside the flight loop.
// `amplitude==0` copies all 88 legacy floats exactly and consumes no RNG draws.
inline bool rl_physics_sample_domain(RL_DOMAIN_THREAD const RLPhysicsParams& nominal,
                                     RL_DOMAIN_THREAD RlDomainIndex& episode_rng,
                                     float amplitude,
                                     RL_DOMAIN_THREAD const RLPhysicsDomainRange& range,
                                     RL_DOMAIN_THREAD RLPhysicsParams& varied,
                                     RL_DOMAIN_THREAD RLPhysicsDomainSample& sample) {
    if(!isfinite(amplitude)||amplitude<0.0f||amplitude>1.0f||
       !isfinite(range.minimum_thrust_to_weight)||range.minimum_thrust_to_weight<1.0f||
       !isfinite(range.mass_scale_min)||!isfinite(range.mass_scale_max)||
       !isfinite(range.inertia_axis_scale_min)||!isfinite(range.inertia_axis_scale_max)||
       !isfinite(range.thrust_gain_min)||!isfinite(range.thrust_gain_max)||
       !isfinite(range.rising_lag_scale_min)||!isfinite(range.rising_lag_scale_max)||
       !isfinite(range.falling_lag_scale_min)||!isfinite(range.falling_lag_scale_max)||
       !(range.mass_scale_min>0.0f&&range.mass_scale_min<=range.mass_scale_max)||
       !(range.inertia_axis_scale_min>0.0f&&range.inertia_axis_scale_min<=range.inertia_axis_scale_max)||
       !(range.thrust_gain_min>0.0f&&range.thrust_gain_min<=range.thrust_gain_max)||
       !(range.rising_lag_scale_min>0.0f&&range.rising_lag_scale_min<=range.rising_lag_scale_max)||
       !(range.falling_lag_scale_min>0.0f&&range.falling_lag_scale_min<=range.falling_lag_scale_max))return false;
    varied=nominal;
    sample.mass_scale=1.0f;
    sample.thrust_to_weight=0.0f;
    for(RlDomainIndex axis=0;axis<3;axis++)sample.inertia_axis_scale[axis]=1.0f;
    for(RlDomainIndex rotor=0;rotor<4;rotor++){
        sample.thrust_gain[rotor]=1.0f;
        sample.rising_lag_scale[rotor]=1.0f;
        sample.falling_lag_scale[rotor]=1.0f;
    }
    if(amplitude>0.0f){
        sample.mass_scale=rl_physics_domain_scale(episode_rng,range.mass_scale_min,range.mass_scale_max,amplitude);
        for(RlDomainIndex axis=0;axis<3;axis++)
            sample.inertia_axis_scale[axis]=rl_physics_domain_scale(episode_rng,range.inertia_axis_scale_min,range.inertia_axis_scale_max,amplitude);
        for(RlDomainIndex rotor=0;rotor<4;rotor++){
            sample.thrust_gain[rotor]=rl_physics_domain_scale(episode_rng,range.thrust_gain_min,range.thrust_gain_max,amplitude);
            sample.rising_lag_scale[rotor]=rl_physics_domain_scale(episode_rng,range.rising_lag_scale_min,range.rising_lag_scale_max,amplitude);
            sample.falling_lag_scale[rotor]=rl_physics_domain_scale(episode_rng,range.falling_lag_scale_min,range.falling_lag_scale_max,amplitude);
        }

        varied.mass=nominal.mass*sample.mass_scale;
        float axis_root[3];
        for(RlDomainIndex axis=0;axis<3;axis++)axis_root[axis]=sqrt(sample.inertia_axis_scale[axis]);
        for(RlDomainIndex row=0;row<3;row++)for(RlDomainIndex col=0;col<3;col++)
            varied.inertia[3*row+col]=nominal.inertia[3*row+col]*sample.mass_scale*axis_root[row]*axis_root[col];
        if(!rl_physics_domain_inverse3(varied.inertia,varied.inertia_inverse))return false;

        for(RlDomainIndex rotor=0;rotor<4;rotor++){
            for(RlDomainIndex coefficient=0;coefficient<3;coefficient++)
                varied.rotor_thrust_coefficients[3*rotor+coefficient]=
                    nominal.rotor_thrust_coefficients[3*rotor+coefficient]*sample.thrust_gain[rotor];
            varied.rotor_time_constants_rising[rotor]=
                nominal.rotor_time_constants_rising[rotor]*sample.rising_lag_scale[rotor];
            varied.rotor_time_constants_falling[rotor]=
                nominal.rotor_time_constants_falling[rotor]*sample.falling_lag_scale[rotor];
        }
    }
    sample.thrust_to_weight=rl_physics_domain_thrust_to_weight(varied);
    return sample.thrust_to_weight>=range.minimum_thrust_to_weight;
}

#ifndef __METAL_VERSION__
inline bool rl_physics_domain_validate(const RLPhysicsParams& params,
                                      const RLPhysicsDomainSample& sample,
                                      const RLPhysicsDomainRange& range,
                                      float inverse_tolerance=2.0e-4f) {
    if(!std::isfinite(params.mass)||params.mass<=0.0f||
       !std::isfinite(sample.thrust_to_weight)||sample.thrust_to_weight<range.minimum_thrust_to_weight)return false;
    for(int row=0;row<3;row++)for(int col=0;col<3;col++){
        float product=0.0f;
        for(int k=0;k<3;k++)product+=params.inertia[3*row+k]*params.inertia_inverse[3*k+col];
        const float expected=row==col?1.0f:0.0f;
        if(std::fabs(product-expected)>inverse_tolerance)return false;
        if(std::fabs(params.inertia[3*row+col]-params.inertia[3*col+row])>1.0e-6f)return false;
    }
    return true;
}
#endif

#undef RL_DOMAIN_THREAD
