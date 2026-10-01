#pragma once

// CPU reference for the fixed Metal navigation workload. Include this header
// after SimRun and SimConfig are declared. It uses the same packed world,
// RAPTOR and L2F types as the Metal path, and Accelerate CBLAS for batched MLPs.
#ifndef ACCELERATE_NEW_LAPACK
#define ACCELERATE_NEW_LAPACK 1
#endif
#include <Accelerate/Accelerate.h>
#include <dispatch/dispatch.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>
#include "world.hpp"
#include "raptor.hpp"
#include "physics.hpp"
#include "ppo.hpp"
#include "guidance.hpp"

namespace cpu_reference {

constexpr uint32_t sensor_pixels = 320;
constexpr uint32_t sensor_frames = 8;
constexpr uint32_t actor_dim = uint32_t(fixed_ppo::actor_obs_dim);
constexpr uint32_t depth_features = (actor_dim - (actor_dim==184?24:21)) / 2;
constexpr uint32_t critic_dim = 32;
constexpr uint32_t hidden_dim = 64;
constexpr uint32_t action_dim = 4;
constexpr uint32_t minibatch_size = 256;
constexpr float gamma = 0.99f;
constexpr float gae_lambda = 0.95f;
constexpr float clip_epsilon = 0.2f;
constexpr float value_coefficient = 0.5f;
constexpr float grad_clip = 0.5f;
constexpr float actor_learning_rate = 3.0e-4f;
constexpr float critic_learning_rate = 3.0e-4f;
constexpr float adam_beta1 = 0.9f;
constexpr float adam_beta2 = 0.999f;
constexpr float adam_epsilon = 1.0e-8f;
static_assert(actor_dim==181 || actor_dim==184 || actor_dim==661,"CPU actor supports only pooled-181 or raw-661 input");

struct Metrics {
    double wall_seconds = 0.0;
    uint64_t updates = 0;
    uint64_t episodes = 0;
    uint64_t successes = 0;
    uint64_t collisions = 0;
    uint64_t timeouts = 0;
    double final_progress = 0.0;
    double total_path = 0.0;
    double total_elapsed = 0.0;
    double success_time = 0.0;
    float policy_loss = 0.0f;
    float value_loss = 0.0f;
    float entropy_loss = 0.0f;
    float ratio = 0.0f;
};

inline float clampf(float x, float lo, float hi) {
    return std::max(lo, std::min(hi, x));
}

inline float sim_normal(uint32_t& rng) {
    const float u = std::max(wurand(rng), 1.0e-7f);
    const float v = wurand(rng);
    return std::sqrt(-2.0f * std::log(u)) * std::cos(6.28318530718f * v);
}

inline void rotation(const float q[4], float r[9]) {
    const float w=q[0],x=q[1],y=q[2],z=q[3];
    r[0]=1-2*(y*y+z*z);r[1]=2*(x*y-w*z);r[2]=2*(x*z+w*y);
    r[3]=2*(x*y+w*z);r[4]=1-2*(x*x+z*z);r[5]=2*(y*z-w*x);
    r[6]=2*(x*z-w*y);r[7]=2*(y*z+w*x);r[8]=1-2*(x*x+y*y);
}

inline void initialize_policy(fixed_ppo::ActorParams& actor,
                              fixed_ppo::CriticParams& critic) {
    actor.values.fill(0.0f);
    critic.values.fill(0.0f);
    uint32_t seed = 1789;
    auto normal = [&]() {
        const float u=std::max(wurand(seed),1.0e-7f);
        return std::sqrt(-2.0f*std::log(u))*std::cos(6.2831853f*wurand(seed));
    };
    for(size_t i=0;i<fixed_ppo::actor_b1_offset;i++) actor.values[i]=normal()*0.04f;
    for(size_t i=fixed_ppo::actor_w2_offset;i<fixed_ppo::actor_b2_offset;i++) actor.values[i]=normal()*0.01f;
    for(size_t i=0;i<action_dim;i++) actor.values[fixed_ppo::actor_log_std_offset+i]=-1.0f;
    for(size_t i=0;i<fixed_ppo::critic_b1_offset;i++) critic.values[i]=normal()*0.15f;
    for(size_t i=fixed_ppo::critic_w2_offset;i<fixed_ppo::critic_b2_offset;i++) critic.values[i]=normal()*0.1f;
}

// GCD reuses its process-wide workers; it does not create threads in the loop.
template<class F>
inline void parallel_for(size_t count, F fn) {
    dispatch_apply_f(count, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED,0), &fn,
        [](void* context, size_t index) { (*static_cast<F*>(context))(index); });
}

class Trainer {
public:
    SimConfig cfg;
    uint32_t horizon;
    RaptorWeights raptor;
    RLPhysicsParams physics;
    fixed_ppo::ActorParams actor;
    fixed_ppo::CriticParams critic;
    uint64_t optimizer_step=0;
    std::vector<RLPhysicsState> states;
    std::vector<SimRun> runs;
    std::vector<WWorld> worlds;
    std::vector<WVec> camera_rays;
    std::vector<float> sensors, poses, commands;
    std::vector<float> observations, critic_observations, actions, old_logp, values;
    std::vector<float> rewards, next_values, advantages, returns;
    std::vector<uint8_t> terminated, truncated, reset_after_bootstrap;
    std::array<float,fixed_ppo::actor_param_count> actor_m{},actor_v{};
    std::array<float,fixed_ppo::critic_param_count> critic_m{},critic_v{};

    Trainer(const SimConfig& config, uint32_t rollout_horizon,
            const RaptorWeights& motor_policy, const RLPhysicsParams& dynamics)
        : cfg(config), horizon(rollout_horizon), raptor(motor_policy), physics(dynamics) {
        initialize_policy(actor,critic);
        const size_t envs=cfg.n, rows=envs*size_t(horizon);
        states.resize(envs);runs.resize(envs);worlds.resize(envs);
        camera_rays.resize(sensor_pixels);
        for(uint32_t k=0;k<sensor_pixels;k++)camera_rays[k]=wcamera(k);
        sensors.resize(envs*sensor_frames*sensor_pixels);
        poses.resize(envs*sensor_frames*12);
        commands.resize(envs*sensor_frames*action_dim);
        observations.resize(rows*actor_dim);critic_observations.resize(rows*critic_dim);
        actions.resize(rows*action_dim);old_logp.resize(rows);values.resize(rows);
        rewards.resize(rows);next_values.resize(rows);advantages.resize(rows);returns.resize(rows);
        terminated.resize(rows);truncated.resize(rows);reset_after_bootstrap.resize(envs);
        current_actor_hidden.resize(envs*hidden_dim);current_actor_means.resize(envs*action_dim);
        current_critic_hidden.resize(envs*hidden_dim);current_critic_values.resize(envs);
        next_critic_observations.resize(envs*critic_dim);next_critic_hidden.resize(envs*hidden_dim);
        next_critic_values.resize(envs);
        mb_actor_hidden.resize(minibatch_size*hidden_dim);mb_actor_means.resize(minibatch_size*action_dim);
        mb_critic_hidden.resize(minibatch_size*hidden_dim);mb_critic_values.resize(minibatch_size);
        d_means.resize(minibatch_size*action_dim);d_log_stds.resize(minibatch_size*action_dim);
        d_values.resize(minibatch_size);loss_rows.resize(minibatch_size*4);
        actor_hidden_delta.resize(minibatch_size*hidden_dim);critic_hidden_delta.resize(minibatch_size*hidden_dim);
        actor_grad.resize(fixed_ppo::actor_param_count);critic_grad.resize(fixed_ppo::critic_param_count);
        actor_pre.resize(std::max(envs,size_t(minibatch_size))*hidden_dim);critic_pre.resize(std::max(envs,size_t(minibatch_size))*hidden_dim);
        minibatch_starts.reserve((rows+minibatch_size-1)/minibatch_size);
        for(uint32_t n=0;n<cfg.n;n++)reset_env(n,true);
    }

    Metrics run(uint32_t rollouts) {
        Metrics metrics{};
        const auto start=std::chrono::steady_clock::now();
        float metric_policy=0,metric_value=0,metric_entropy=0,metric_ratio=0;
        for(uint32_t r=0;r<rollouts;r++) {
            collect_rollout();
            fixed_ppo::compute_gae(horizon,cfg.n,rewards.data(),values.data(),next_values.data(),
                terminated.data(),truncated.data(),gamma,gae_lambda,advantages.data(),returns.data());
            fixed_ppo::normalize_advantages(advantages.data(),advantages.size());
            train_rollout(r,metric_policy,metric_value,metric_entropy,metric_ratio);
        }
        metrics.wall_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
        metrics.updates=optimizer_step;
        for(const auto& run:runs) {
            metrics.episodes+=run.episodes;metrics.successes+=run.successes;
            metrics.collisions+=run.collisions;metrics.timeouts+=run.timeouts;
            metrics.final_progress+=run.final_progress;metrics.total_path+=run.total_path;
            metrics.total_elapsed+=run.total_elapsed;metrics.success_time+=run.success_time;
        }
        metrics.policy_loss=metric_policy;metrics.value_loss=metric_value;
        metrics.entropy_loss=metric_entropy;metrics.ratio=metric_ratio;
        return metrics;
    }

    bool validate_batched_forward(float tolerance=2.0e-5f) {
        constexpr uint32_t b=3;
        std::vector<float> obs(size_t(b)*actor_dim),co(size_t(b)*critic_dim);
        std::vector<float> ah(size_t(b)*hidden_dim),am(size_t(b)*action_dim);
        std::vector<float> ch(size_t(b)*hidden_dim),cv(b);
        for(size_t i=0;i<obs.size();i++)obs[i]=.2f*std::sin(float(i)*.0037f);
        for(size_t i=0;i<co.size();i++)co[i]=.3f*std::cos(float(i)*.17f);
        actor_forward_batch(obs.data(),b,ah.data(),am.data());
        critic_forward_batch(co.data(),b,ch.data(),cv.data());
        float error=0.0f;
        std::array<float,hidden_dim> h{};
        std::array<float,action_dim> mean{};
        for(uint32_t n=0;n<b;n++) {
            fixed_ppo::actor_forward(obs.data()+size_t(n)*actor_dim,actor,h.data(),mean.data());
            for(uint32_t j=0;j<hidden_dim;j++)error=std::max(error,std::fabs(h[j]-ah[size_t(n)*hidden_dim+j]));
            for(uint32_t j=0;j<action_dim;j++)error=std::max(error,std::fabs(mean[j]-am[size_t(n)*action_dim+j]));
            const float value=fixed_ppo::critic_forward(co.data()+size_t(n)*critic_dim,critic,h.data());
            for(uint32_t j=0;j<hidden_dim;j++)error=std::max(error,std::fabs(h[j]-ch[size_t(n)*hidden_dim+j]));
            error=std::max(error,std::fabs(value-cv[n]));
        }
        return error<=tolerance;
    }

private:
    std::vector<float> current_actor_hidden, current_actor_means, current_critic_hidden, current_critic_values;
    std::vector<float> actor_pre,critic_pre;
    std::vector<float> next_critic_observations, next_critic_hidden, next_critic_values;
    std::vector<float> mb_actor_hidden, mb_actor_means, mb_critic_hidden, mb_critic_values;
    std::vector<float> d_means,d_log_stds,d_values,loss_rows;
    std::vector<float> actor_hidden_delta,actor_grad,critic_hidden_delta,critic_grad;
    std::vector<uint32_t> minibatch_starts;

    bool clean_training_env(uint32_t n) const {
        return cfg.eval==0 && (cfg.family==12 || cfg.family==13) && (n%2==0);
    }
    uint32_t effective_sensor_delay(uint32_t n) const {
        return clean_training_env(n)?0:cfg.sensor_delay;
    }
    uint32_t effective_command_delay(uint32_t n) const {
        return clean_training_env(n)?0:cfg.command_delay;
    }
    float effective_wind(uint32_t n) const {
        return clean_training_env(n)?0.0f:cfg.wind;
    }
    float effective_depth_noise(uint32_t n) const {
        return clean_training_env(n)?0.0f:cfg.depth_noise;
    }
    float effective_dropout(uint32_t n) const {
        return clean_training_env(n)?0.0f:cfg.dropout;
    }

    void reset_env(uint32_t n,bool first) {
        auto& s=states[n];auto& run=runs[n];
        uint32_t rng=first ? cfg.seed+n*747796405u+2891336453u : run.rng;
        uint32_t episode_family=wtraining_family(cfg.family,rng);
        wgenerate(worlds[n],wrng(rng),episode_family,cfg.distance);
        worlds[n].wind[0]=effective_wind(n);worlds[n].wind[1]=0;worlds[n].wind[2]=0;
        for(int j=0;j<3;j++) {
            s.position[j]=j==2?1.5f:0;s.linear_velocity[j]=0;s.angular_velocity_body[j]=0;
            run.desired_velocity[j]=0;run.reference_position[j]=s.position[j];
        }
        s.orientation_wxyz[0]=1;for(int j=1;j<4;j++)s.orientation_wxyz[j]=0;
        const float target=physics.mass*9.81f/4;
        const float c0=physics.rotor_thrust_coefficients[0],c1=physics.rotor_thrust_coefficients[1],c2=physics.rotor_thrust_coefficients[2];
        const float hover=clampf((-c1+std::sqrt(c1*c1-4*c2*(c0-target)))/(2*c2),physics.action_min,physics.action_max);
        raptor_reset(raptor,run.hidden);
        for(int j=0;j<4;j++){s.rpm[j]=hover;run.motors[j]=0;run.previous_nav[j]=0;}
        if(episode_family==10 || episode_family==11) {
            s.linear_velocity[0]=cfg.speed;
            run.desired_velocity[0]=cfg.speed;
            run.previous_nav[0]=1.0f;
        }
        std::fill(sensors.begin()+size_t(n)*sensor_frames*sensor_pixels,
                  sensors.begin()+size_t(n+1)*sensor_frames*sensor_pixels,12.0f);
        std::fill(commands.begin()+size_t(n)*sensor_frames*action_dim,
                  commands.begin()+size_t(n+1)*sensor_frames*action_dim,0.0f);
        run.yaw=0;run.rng=rng;run.steps=0;run.path=run.elapsed=run.peak_speed=0;
        run.min_clearance=12;run.initial_distance=wl(wv(worlds[n].goal[0],worlds[n].goal[1],worlds[n].goal[2]-1.5f));
        if(first) {
            run.successes=run.collisions=run.timeouts=run.episodes=0;
            run.success_time=run.total_path=run.total_elapsed=run.final_progress=0;
            std::fill(poses.begin()+size_t(n)*sensor_frames*12,
                      poses.begin()+size_t(n+1)*sensor_frames*12,0.0f);
        }
    }

    void collect_depth_and_obs(uint32_t t,uint32_t n) {
        auto& run=runs[n];auto& s=states[n];const auto& world=worlds[n];
        if(cfg.eval && run.episodes)return;
        if(run.steps%cfg.sensor_period==0) {
            float r[9];rotation(s.orientation_wxyz,r);
            const size_t frame=(run.steps/cfg.sensor_period)%sensor_frames;
            for(uint32_t j=0;j<3;j++)poses[(size_t(n)*sensor_frames+frame)*12+j]=s.position[j];
            for(uint32_t j=0;j<9;j++)poses[(size_t(n)*sensor_frames+frame)*12+3+j]=r[j];
            for(uint32_t k=0;k<sensor_pixels;k++) {
                const WVec ray=camera_rays[k];
                const WVec d=wv(r[0]*ray.x+r[1]*ray.y+r[2]*ray.z,
                                r[3]*ray.x+r[4]*ray.y+r[5]*ray.z,
                                r[6]*ray.x+r[7]*ray.y+r[8]*ray.z);
                float depth=wray(world,wv(s.position[0],s.position[1],s.position[2]),d,run.elapsed);
                const float noise=effective_depth_noise(n),dropout=effective_dropout(n);
                if(noise!=0 || dropout!=0) {
                    uint32_t rng=run.rng+k*1664525u+run.steps*1013904223u;
                    depth=clampf(depth+noise*sim_normal(rng),0.0f,12.0f);
                    if(wurand(rng)<dropout)depth=12.0f;
                }
                sensors[(size_t(n)*sensor_frames+frame)*sensor_pixels+k]=depth;
            }
        }
        const uint32_t available=run.steps/cfg.sensor_period;
        const uint32_t sensor_delay=effective_sensor_delay(n);
        const uint32_t frame=available>sensor_delay?available-sensor_delay:0;
        const uint32_t prev=frame>0?frame-1:0;
        const size_t row=(size_t(t)*cfg.n+n)*actor_dim;
        const size_t crow=(size_t(t)*cfg.n+n)*critic_dim;
        for(uint32_t k=0;k<depth_features;k++) {
            float current=12.0f,previous=12.0f;
            if(depth_features==sensor_pixels) {
                current=sensors[(size_t(n)*sensor_frames+frame%sensor_frames)*sensor_pixels+k];
                previous=sensors[(size_t(n)*sensor_frames+prev%sensor_frames)*sensor_pixels+k];
            } else {
                const uint32_t y=(k/10)*2,x=(k%10)*2;
                for(uint32_t dy=0;dy<2;dy++)for(uint32_t dx=0;dx<2;dx++) {
                    const uint32_t pixel=(y+dy)*20+x+dx;
                    current=std::min(current,sensors[(size_t(n)*sensor_frames+frame%sensor_frames)*sensor_pixels+pixel]);
                    previous=std::min(previous,sensors[(size_t(n)*sensor_frames+prev%sensor_frames)*sensor_pixels+pixel]);
                }
            }
            observations[row+k]=available>=sensor_delay?current/12.0f:1.0f;
            observations[row+depth_features+k]=available>=sensor_delay?previous/12.0f:1.0f;
        }
        float r[9];rotation(s.orientation_wxyz,r);
        const WVec delta=wv(world.goal[0]-s.position[0],world.goal[1]-s.position[1],world.goal[2]-s.position[2]);
        const float distance=std::max(wl(delta),1.0e-6f);
        const size_t context=row+2*depth_features;
        for(uint32_t j=0;j<3;j++) {
            observations[context+j]=(r[j]*delta.x+r[3+j]*delta.y+r[6+j]*delta.z)/distance;
            observations[context+4+j]=(r[j]*s.linear_velocity[0]+r[3+j]*s.linear_velocity[1]+r[6+j]*s.linear_velocity[2])/4.0f;
            observations[context+7+j]=s.angular_velocity_body[j]/4.0f;
            observations[context+10+j]=r[6+j];
        }
        observations[context+3]=std::min(distance/10.0f,1.5f);
        for(uint32_t j=0;j<4;j++)observations[context+13+j]=run.previous_nav[j];
        observations[context+17]=float(run.steps-frame*cfg.sensor_period)*physics.dt*cfg.substeps;
        const float ref[3]={run.reference_position[0]-s.position[0],run.reference_position[1]-s.position[1],run.reference_position[2]-s.position[2]};
        for(uint32_t j=0;j<3;j++)observations[context+18+j]=clampf((r[j]*ref[0]+r[3+j]*ref[1]+r[6+j]*ref[2])*2.0f,-1.0f,1.0f);
        if constexpr(actor_dim==184) {
            float cur[80],prev_range[80],goal[3],vel[3],hint[3],current_pose[12];
            for(uint32_t j=0;j<80;j++) {
                cur[j]=observations[row+j]*12.0f;
                prev_range[j]=observations[row+80+j]*12.0f;
            }
            for(uint32_t j=0;j<3;j++) {
                goal[j]=observations[context+j];
                vel[j]=observations[context+4+j]*4.0f;
                current_pose[j]=s.position[j];
            }
            for(uint32_t j=0;j<9;j++)current_pose[j+3]=r[j];
            const float sensor_dt=float(cfg.sensor_period)*physics.dt*cfg.substeps;
            if(cfg.geometry_memory) {
                const uint32_t valid_frames=available>=sensor_delay
                    ?std::min(frame+1,sensor_frames-sensor_delay):0;
                nav_guidance_memory(cur,prev_range,goal,distance,vel,sensor_dt,
                    sensors.data()+size_t(n)*sensor_frames*sensor_pixels,
                    poses.data()+size_t(n)*sensor_frames*12,current_pose,
                    frame,valid_frames,hint);
            } else nav_guidance(cur,prev_range,goal,distance,vel,sensor_dt,hint);
            for(uint32_t j=0;j<3;j++)observations[row+actor_dim-3+j]=hint[j];
        }
        critic_observation(s,world,run,critic_observations.data()+crow);
    }

    static void critic_observation(const RLPhysicsState& s,const WWorld& w,const SimRun& run,float* out) {
        std::fill(out,out+critic_dim,0.0f);
        for(uint32_t j=0;j<3;j++) {
            out[j]=(w.goal[j]-s.position[j])/10.0f;out[j+3]=s.linear_velocity[j]/4.0f;
            out[j+6]=s.angular_velocity_body[j]/4.0f;out[j+13]=s.position[j]/10.0f;
            out[j+16]=(run.reference_position[j]-s.position[j])*2.0f;
        }
        for(uint32_t j=0;j<4;j++)out[9+j]=s.orientation_wxyz[j];
        out[30]=w.wind[0];out[31]=w.wind[1];
        out[19]=wclearance(w,wv(s.position[0],s.position[1],s.position[2]),run.elapsed)/5.0f;
        out[20]=float(run.steps)/200.0f;
        for(uint32_t j=0;j<std::min(w.count,3u);j++) {
            const WVec c=wc(w.obstacles[j],run.elapsed);
            out[21+j*3]=(c.x-s.position[0])/10.0f;out[22+j*3]=(c.y-s.position[1])/10.0f;out[23+j*3]=(c.z-s.position[2])/10.0f;
        }
    }

    void collect_rollout() {
        const size_t envs=cfg.n;
        for(uint32_t t=0;t<horizon;t++) {
            parallel_for(envs,[&](size_t n){collect_depth_and_obs(t,uint32_t(n));});
            const float* obs=observations.data()+size_t(t)*envs*actor_dim;
            const float* co=critic_observations.data()+size_t(t)*envs*critic_dim;
            actor_forward_batch(obs,uint32_t(envs),current_actor_hidden.data(),current_actor_means.data());
            critic_forward_batch(co,uint32_t(envs),current_critic_hidden.data(),current_critic_values.data());
            for(uint32_t n=0;n<cfg.n;n++) {
                const size_t row=size_t(t)*envs+n;
                values[row]=current_critic_values[n];
                uint32_t rng=runs[n].rng;
                for(uint32_t a=0;a<action_dim;a++) {
                    const float action=current_actor_means[n*action_dim+a]
                        +std::exp(actor.values[fixed_ppo::actor_log_std_offset+a])*sim_normal(rng);
                    actions[row*action_dim+a]=action;
                    commands[(size_t(n)*sensor_frames+runs[n].steps%sensor_frames)*action_dim+a]=std::tanh(action);
                }
                old_logp[row]=fixed_ppo::gaussian_log_prob(actions.data()+row*action_dim,
                    current_actor_means.data()+n*action_dim,actor.values.data()+fixed_ppo::actor_log_std_offset);
                runs[n].rng=rng;
                const uint32_t command_delay=effective_command_delay(n);
                const uint32_t applied=runs[n].steps>command_delay?runs[n].steps-command_delay:0;
                for(uint32_t a=0;a<action_dim;a++)
                    runs[n].previous_nav[a]=runs[n].steps<command_delay?0.0f:
                        commands[(size_t(n)*sensor_frames+applied%sensor_frames)*action_dim+a];
                float r[9];rotation(states[n].orientation_wxyz,r);
                float v[3]={runs[n].previous_nav[0]*cfg.speed,
                                  runs[n].previous_nav[1]*cfg.speed,
                                  runs[n].previous_nav[2]*cfg.speed*(cfg.velocity_contract==0?0.5f:1.0f)};
                if(cfg.velocity_contract==1){float scale=std::min(1.0f,cfg.speed/std::max(wl(wv(v[0],v[1],v[2])),1e-8f));for(float& component:v)component*=scale;}
                for(int j=0;j<3;j++) runs[n].desired_velocity[j]=r[j*3]*v[0]+r[j*3+1]*v[1]+r[j*3+2]*v[2];
                runs[n].yaw+=runs[n].previous_nav[3]*physics.dt*cfg.substeps*0.5f;
            }
            parallel_for(envs,[&](size_t n){advance_env(t,uint32_t(n));});
            critic_forward_batch(next_critic_observations.data(),uint32_t(envs),next_critic_hidden.data(),next_critic_values.data());
            for(uint32_t n=0;n<cfg.n;n++)next_values[size_t(t)*envs+n]=next_critic_values[n];
            parallel_for(envs,[&](size_t n){if(reset_after_bootstrap[n])reset_env(uint32_t(n),false);});
        }
    }

    void actor_forward_batch(const float* obs,uint32_t batch,float* hidden,float* means) {
        constexpr int I=int(actor_dim),H=int(hidden_dim),A=int(action_dim);
        cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasTrans,int(batch),H,I,1.0f,
                    obs,I,actor.values.data()+fixed_ppo::actor_w1_offset,I,0.0f,actor_pre.data(),H);
        for(uint32_t n=0;n<batch;n++)for(uint32_t h=0;h<hidden_dim;h++)
            hidden[size_t(n)*hidden_dim+h]=std::tanh(actor_pre[size_t(n)*hidden_dim+h]+actor.values[fixed_ppo::actor_b1_offset+h]);
        cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasTrans,int(batch),A,H,1.0f,
                    hidden,H,actor.values.data()+fixed_ppo::actor_w2_offset,H,0.0f,means,A);
        for(uint32_t n=0;n<batch;n++)for(uint32_t a=0;a<action_dim;a++)
            {means[size_t(n)*action_dim+a]+=actor.values[fixed_ppo::actor_b2_offset+a];if constexpr(actor_dim==184){if(a<3)means[size_t(n)*action_dim+a]+=obs[size_t(n)*actor_dim+actor_dim-3+a];}}
    }

    void critic_forward_batch(const float* obs,uint32_t batch,float* hidden,float* values_out) {
        constexpr int I=int(critic_dim),H=int(hidden_dim);
        cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasTrans,int(batch),H,I,1.0f,
                    obs,I,critic.values.data()+fixed_ppo::critic_w1_offset,I,0.0f,critic_pre.data(),H);
        for(uint32_t n=0;n<batch;n++)for(uint32_t h=0;h<hidden_dim;h++)
            hidden[size_t(n)*hidden_dim+h]=std::tanh(critic_pre[size_t(n)*hidden_dim+h]+critic.values[fixed_ppo::critic_b1_offset+h]);
        cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,int(batch),1,H,1.0f,
                    hidden,H,critic.values.data()+fixed_ppo::critic_w2_offset,1,0.0f,values_out,1);
        for(uint32_t n=0;n<batch;n++)values_out[n]+=critic.values[fixed_ppo::critic_b2_offset];
    }

    void advance_env(uint32_t t,uint32_t n) {
        auto& run=runs[n];auto s=states[n];const auto& world=worlds[n];
        float h[16],motors[4];std::copy(run.hidden,run.hidden+16,h);std::copy(run.motors,run.motors+4,motors);
        const WVec goal=wv(world.goal[0],world.goal[1],world.goal[2]);
        const float before=wl(ws(goal,wv(s.position[0],s.position[1],s.position[2])));
        const float yaw=run.yaw,c=std::cos(yaw),si=std::sin(yaw);bool collision=false;
        float step_clearance=12.0f;
        const float wind[3]={world.wind[0]*physics.mass,world.wind[1]*physics.mass,world.wind[2]*physics.mass};
        for(uint32_t step=0;step<cfg.substeps;step++) {
            float motor_obs[22],v[3]={run.desired_velocity[0],run.desired_velocity[1],run.desired_velocity[2]};
            for(int j=0;j<3;j++)run.reference_position[j]+=v[j]*physics.dt;
            const float dp[3]={s.position[0]-run.reference_position[0],s.position[1]-run.reference_position[1],s.position[2]-run.reference_position[2]};
            const float dv[3]={s.linear_velocity[0]-v[0],s.linear_velocity[1]-v[1],s.linear_velocity[2]-v[2]};
            motor_obs[0]=clampf(c*dp[0]+si*dp[1],-.5f,.5f);motor_obs[1]=clampf(-si*dp[0]+c*dp[1],-.5f,.5f);motor_obs[2]=clampf(dp[2],-.5f,.5f);
            const float ch=std::cos(yaw*.5f),sh=std::sin(yaw*.5f);
            const float q[4]={ch*s.orientation_wxyz[0]+sh*s.orientation_wxyz[3],
                              ch*s.orientation_wxyz[1]+sh*s.orientation_wxyz[2],
                              ch*s.orientation_wxyz[2]-sh*s.orientation_wxyz[1],
                              ch*s.orientation_wxyz[3]-sh*s.orientation_wxyz[0]};
            float r[9];rotation(q,r);for(int j=0;j<9;j++)motor_obs[3+j]=r[j];
            motor_obs[12]=clampf(c*dv[0]+si*dv[1],-1.0f,1.0f);
            motor_obs[13]=clampf(-si*dv[0]+c*dv[1],-1.0f,1.0f);motor_obs[14]=clampf(dv[2],-1.0f,1.0f);
            for(int j=0;j<3;j++)motor_obs[15+j]=s.angular_velocity_body[j];
            for(int j=0;j<4;j++)motor_obs[18+j]=motors[j];
            raptor_forward(raptor,motor_obs,h,motors);raptor_clip_action(motors);
            RLPhysicsState next{};rl_physics_step(s,motors,wind,physics,next);
            bool valid=true;
            for(int j=0;j<3;j++)valid=valid&&std::isfinite(next.position[j])&&std::isfinite(next.linear_velocity[j])&&std::isfinite(next.angular_velocity_body[j]);
            for(int j=0;j<4;j++)valid=valid&&std::isfinite(next.orientation_wxyz[j])&&std::isfinite(next.rpm[j]);
            if(!valid){collision=true;break;}
            run.path+=wl(ws(wv(s.position[0],s.position[1],s.position[2]),wv(next.position[0],next.position[1],next.position[2])));
            s=next;run.elapsed+=physics.dt;
            const float clearance=wclearance(world,wv(s.position[0],s.position[1],s.position[2]),run.elapsed);
            step_clearance=std::min(step_clearance,clearance);
            run.min_clearance=std::min(run.min_clearance,clearance);
            run.peak_speed=std::max(run.peak_speed,wl(wv(s.linear_velocity[0],s.linear_velocity[1],s.linear_velocity[2])));
            if(clearance<=0 || !std::isfinite(s.position[0]) || !std::isfinite(s.position[1]) || !std::isfinite(s.position[2])){collision=true;break;}
        }
        run.steps++;states[n]=s;std::copy(h,h+16,run.hidden);std::copy(motors,motors+4,run.motors);
        const float after=wl(ws(goal,wv(s.position[0],s.position[1],s.position[2])));
        const bool success=after<.35f&&!collision,timeout=run.steps>=cfg.max_steps;
        const size_t row=size_t(t)*cfg.n+n;
        rewards[row]=(before-after)*2.0f-.01f
            -cfg.risk_coef*clampf((0.6f-step_clearance)/0.6f,0.0f,1.0f)
            +(success?10.0f:0.0f)-(collision?10.0f:0.0f);
        terminated[row]=uint8_t(collision||success);truncated[row]=uint8_t(timeout&&!terminated[row]);
        reset_after_bootstrap[n]=uint8_t(success||collision||timeout);
        if(reset_after_bootstrap[n]) {
            run.successes+=success;run.collisions+=collision;run.timeouts+=timeout&&!terminated[row];run.episodes++;
            run.success_time+=success?run.elapsed:0;run.total_path+=run.path;run.total_elapsed+=run.elapsed;
            run.final_progress+=1.0f-after/std::max(run.initial_distance,1.0e-4f);
        }
        critic_observation(s,world,run,next_critic_observations.data()+size_t(n)*critic_dim);
    }

    void actor_gradients(const float* obs,const float* hidden,uint32_t batch,
                         const float* dm,const float* dlogstd,std::vector<float>& delta) {
        constexpr int H=int(hidden_dim),A=int(action_dim),I=int(actor_dim);
        cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,int(batch),H,A,1.0f,
                    dm,A,actor.values.data()+fixed_ppo::actor_w2_offset,H,0.0f,delta.data(),H);
        for(uint32_t n=0;n<batch;n++)for(uint32_t h=0;h<hidden_dim;h++)
            delta[size_t(n)*H+h]*=1.0f-hidden[size_t(n)*H+h]*hidden[size_t(n)*H+h];
        std::fill(actor_grad.begin(),actor_grad.end(),0.0f);
        cblas_sgemm(CblasRowMajor,CblasTrans,CblasNoTrans,H,I,int(batch),1.0f,
                    delta.data(),H,obs,I,0.0f,actor_grad.data()+fixed_ppo::actor_w1_offset,I);
        cblas_sgemm(CblasRowMajor,CblasTrans,CblasNoTrans,A,H,int(batch),1.0f,
                    dm,A,hidden,H,0.0f,actor_grad.data()+fixed_ppo::actor_w2_offset,H);
        for(uint32_t n=0;n<batch;n++)for(uint32_t h=0;h<hidden_dim;h++)
            actor_grad[fixed_ppo::actor_b1_offset+h]+=delta[size_t(n)*H+h];
        for(uint32_t n=0;n<batch;n++)for(uint32_t a=0;a<action_dim;a++) {
            actor_grad[fixed_ppo::actor_b2_offset+a]+=dm[size_t(n)*A+a];
            actor_grad[fixed_ppo::actor_log_std_offset+a]+=dlogstd[size_t(n)*A+a];
        }
    }

    void critic_gradients(const float* obs,const float* hidden,uint32_t batch,
                          const float* dv,std::vector<float>& delta) {
        constexpr int H=int(hidden_dim),I=int(critic_dim);
        cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasTrans,int(batch),H,1,1.0f,
                    dv,1,critic.values.data()+fixed_ppo::critic_w2_offset,1,0.0f,delta.data(),H);
        for(uint32_t n=0;n<batch;n++)for(uint32_t h=0;h<hidden_dim;h++)
            delta[size_t(n)*H+h]*=1.0f-hidden[size_t(n)*H+h]*hidden[size_t(n)*H+h];
        std::fill(critic_grad.begin(),critic_grad.end(),0.0f);
        cblas_sgemm(CblasRowMajor,CblasTrans,CblasNoTrans,H,I,int(batch),1.0f,
                    delta.data(),H,obs,I,0.0f,critic_grad.data()+fixed_ppo::critic_w1_offset,I);
        cblas_sgemm(CblasRowMajor,CblasTrans,CblasNoTrans,1,H,int(batch),1.0f,
                    dv,1,hidden,H,0.0f,critic_grad.data()+fixed_ppo::critic_w2_offset,H);
        for(uint32_t n=0;n<batch;n++) {
            critic_grad[fixed_ppo::critic_b2_offset]+=dv[n];
            for(uint32_t h=0;h<hidden_dim;h++)critic_grad[fixed_ppo::critic_b1_offset+h]+=delta[size_t(n)*H+h];
        }
    }

    void clip_and_adam(std::vector<float>& grad,float* params,float* m,float* v,size_t count) {
        float sum=0.0f;for(float g:grad)sum+=g*g;
        const float norm=std::sqrt(sum+1.0e-20f);
        const float scale=std::min(1.0f,grad_clip/norm);
        for(float& g:grad)g*=scale;
        const float b1corr=1.0f-std::pow(adam_beta1,float(optimizer_step));
        const float b2corr=1.0f-std::pow(adam_beta2,float(optimizer_step));
        const float lr=cfg.learning_rate;
        for(size_t i=0;i<count;i++) {
            m[i]=adam_beta1*m[i]+(1.0f-adam_beta1)*grad[i];
            v[i]=adam_beta2*v[i]+(1.0f-adam_beta2)*grad[i]*grad[i];
            const float mh=m[i]/b1corr,vh=v[i]/b2corr;
            params[i]-=lr*mh/(std::sqrt(vh)+adam_epsilon);
        }
    }

    void train_rollout(uint32_t rollout,float& out_policy,float& out_value,float& out_entropy,float& out_ratio) {
        const uint32_t rows=cfg.n*horizon;
        minibatch_starts.clear();for(uint32_t i=0;i<rows;i+=minibatch_size)minibatch_starts.push_back(i);
        std::mt19937 rng(0x9e3779b9u+rollout*0x85ebca6bu);
        std::shuffle(minibatch_starts.begin(),minibatch_starts.end(),rng);
        const uint32_t updates=uint32_t(minibatch_starts.size())*2;
        float policy_sum=0,value_sum=0,entropy_sum=0,ratio_sum=0;
        const float entropy_coef=cfg.entropy_coef;
        for(uint32_t epoch=0;epoch<2;epoch++) {
            if(epoch==1)std::shuffle(minibatch_starts.begin(),minibatch_starts.end(),rng);
            for(uint32_t begin:minibatch_starts) {
                const uint32_t b=std::min<uint32_t>(minibatch_size,rows-begin);
                const float* ob=observations.data()+size_t(begin)*actor_dim;
                const float* co=critic_observations.data()+size_t(begin)*critic_dim;
                const float* act=actions.data()+size_t(begin)*action_dim;
                const float* oldlp=old_logp.data()+begin;const float* oldv=values.data()+begin;
                const float* adv=advantages.data()+begin;const float* ret=returns.data()+begin;
                actor_forward_batch(ob,b,mb_actor_hidden.data(),mb_actor_means.data());
                critic_forward_batch(co,b,mb_critic_hidden.data(),mb_critic_values.data());
                for(uint32_t n=0;n<b;n++) {
                    const auto loss=fixed_ppo::sample_loss_and_grad(act+size_t(n)*action_dim,
                        mb_actor_means.data()+size_t(n)*action_dim,actor.values.data()+fixed_ppo::actor_log_std_offset,
                        oldlp[n],oldv[n],mb_critic_values[n],adv[n],ret[n],float(b),clip_epsilon,
                        value_coefficient,entropy_coef,d_means.data()+size_t(n)*action_dim,
                        d_log_stds.data()+size_t(n)*action_dim,&d_values[n]);
                    loss_rows[n*4]=loss.policy;loss_rows[n*4+1]=loss.value;
                    loss_rows[n*4+2]=loss.entropy;loss_rows[n*4+3]=loss.ratio;
                    policy_sum+=loss.policy;value_sum+=loss.value;entropy_sum+=loss.entropy;ratio_sum+=loss.ratio/float(b);
                }
                actor_gradients(ob,mb_actor_hidden.data(),b,d_means.data(),d_log_stds.data(),actor_hidden_delta);
                critic_gradients(co,mb_critic_hidden.data(),b,d_values.data(),critic_hidden_delta);
                ++optimizer_step;
                clip_and_adam(actor_grad,actor.values.data(),actor_m.data(),actor_v.data(),fixed_ppo::actor_param_count);
                clip_and_adam(critic_grad,critic.values.data(),critic_m.data(),critic_v.data(),fixed_ppo::critic_param_count);
                for(uint32_t a=0;a<action_dim;a++)actor.values[fixed_ppo::actor_log_std_offset+a]=clampf(actor.values[fixed_ppo::actor_log_std_offset+a],-2.0f,.5f);
            }
        }
        const float scale=1.0f/float(updates);
        out_policy=policy_sum*scale;out_value=value_sum*scale;out_entropy=entropy_sum*scale;out_ratio=ratio_sum*scale;
    }

    void reset_all() {
        for(uint32_t n=0;n<cfg.n;n++)reset_env(n,true);
        std::fill(sensors.begin(),sensors.end(),12.0f);std::fill(commands.begin(),commands.end(),0.0f);
        actor_m.fill(0);actor_v.fill(0);critic_m.fill(0);critic_v.fill(0);optimizer_step=0;
    }
};

} // namespace cpu_reference
