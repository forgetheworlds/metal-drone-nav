#pragma once

// Small C++/MSL shared seam for goal-conditioned task setup and terminal
// accounting. Include after world.hpp and physics.metal on MSL; no world or
// simulator ABI changes.
#ifndef __METAL_VERSION__
#include "world.hpp"
#include "physics.hpp"
#endif

// world.hpp undefines its address-space macros after its own definitions.
#ifdef __METAL_VERSION__
#define NT_WP device
#define NT_WT thread
#define NT_WF inline
#else
#define NT_WP
#define NT_WT
#define NT_WF inline
#endif

#ifdef __METAL_VERSION__
constant uint NAV_TASK_STAGE_OPEN_GOAL = 0u;
constant uint NAV_TASK_STAGE_NEAR_GOAL_HOLD = 1u;
constant uint NAV_TASK_STAGE_CLUTTER_GOAL = 2u;
constant uint NAV_TASK_OBJECTIVE_WAYPOINT = 0u;
constant uint NAV_TASK_OBJECTIVE_FINAL_HOLD = 1u;
constant uint NAV_TASK_GENERATION_INVALID_CONFIG = 1u;
constant uint NAV_TASK_GENERATION_UNSUPPORTED_DYNAMIC = 2u;
constant uint NAV_TASK_GENERATION_EXHAUSTED = 3u;
constant uint NAV_TASK_GENERATION_READY = 4u;
constant uint NAV_TASK_GENERATION_UNSUPPORTED_FAMILY = 5u;
constant uint NAV_TASK_MAX_WITNESS_POINTS = 4u;
constant uint NAV_TASK_MAX_GENERATION_ATTEMPTS = 64u;
#else
constexpr uint NAV_TASK_STAGE_OPEN_GOAL = 0u;
constexpr uint NAV_TASK_STAGE_NEAR_GOAL_HOLD = 1u;
constexpr uint NAV_TASK_STAGE_CLUTTER_GOAL = 2u;
constexpr uint NAV_TASK_OBJECTIVE_WAYPOINT = 0u;
constexpr uint NAV_TASK_OBJECTIVE_FINAL_HOLD = 1u;
constexpr uint NAV_TASK_GENERATION_INVALID_CONFIG = 1u;
constexpr uint NAV_TASK_GENERATION_UNSUPPORTED_DYNAMIC = 2u;
constexpr uint NAV_TASK_GENERATION_EXHAUSTED = 3u;
constexpr uint NAV_TASK_GENERATION_READY = 4u;
constexpr uint NAV_TASK_GENERATION_UNSUPPORTED_FAMILY = 5u;
constexpr uint NAV_TASK_MAX_WITNESS_POINTS = 4u;
constexpr uint NAV_TASK_MAX_GENERATION_ATTEMPTS = 64u;
using std::cos;
using std::sin;
using std::ceil;
#endif

#ifdef __METAL_VERSION__
#define NAV_TASK_CONST constant
#else
#define NAV_TASK_CONST const
#endif

NT_WF bool navigation_task_finite(float value);

// Curriculum parameters are per rollout/setup, not legacy SimConfig fields.
// Difficulty selects the requested start-goal distance; it does not claim a
// calibrated difficulty score for the generated geometry.
struct NavigationTaskConfig {
    uint family;
    uint stage;
    uint objective;
    uint require_detour;
    uint max_generation_attempts;
    uint max_nav_steps;
    float scene_distance_m;
    float difficulty;
    float goal_distance_min_m;
    float goal_distance_max_m;
    float minimum_endpoint_clearance_m;
    float minimum_witness_clearance_m;
    float goal_radius_m;
    float stable_speed_mps;
    float stable_hold_s;
    float nav_period_s;
    float initial_speed_min_mps;
    float initial_speed_max_mps;
    float progress_reward_per_m;
    float time_cost_per_s;
    float contact_penalty;
    float stable_arrival_bonus;
    float command_smoothness_weight;
    float start_min_xyz[3];
    float start_max_xyz[3];
    float goal_min_xyz[3];
    float goal_max_xyz[3];
};

struct NavigationTaskControl {
    uint enabled;
    NavigationTaskConfig config;
};

// Ready-to-use baseline parameters. Callers can adjust difficulty and ranges
// after this helper; static clutter is restricted to supported static worlds.
#ifndef __METAL_VERSION__
NT_WF bool navigation_task_default_config(NT_WT NavigationTaskConfig& cfg,
                                          uint stage,uint family,float difficulty) {
    if(stage>NAV_TASK_STAGE_CLUTTER_GOAL||!navigation_task_finite(difficulty)||
       difficulty<0.0f||difficulty>1.0f)return false;
    if(stage==NAV_TASK_STAGE_OPEN_GOAL||stage==NAV_TASK_STAGE_NEAR_GOAL_HOLD) {
        if(family!=0u)return false;
    } else if(!(family==1u||family==2u||family==4u||family==5u))return false;
    cfg=NavigationTaskConfig{};
    cfg.family=family;cfg.stage=stage;cfg.objective=NAV_TASK_OBJECTIVE_FINAL_HOLD;
    cfg.require_detour=0u;cfg.max_generation_attempts=NAV_TASK_MAX_GENERATION_ATTEMPTS;
    cfg.max_nav_steps=400u;cfg.scene_distance_m=8.0f;cfg.difficulty=difficulty;
    cfg.minimum_endpoint_clearance_m=0.30f;cfg.minimum_witness_clearance_m=0.02f;
    cfg.goal_radius_m=0.35f;cfg.stable_speed_mps=0.50f;cfg.stable_hold_s=0.20f;
    cfg.nav_period_s=0.05f;cfg.initial_speed_min_mps=0.0f;cfg.initial_speed_max_mps=0.0f;
    cfg.progress_reward_per_m=2.0f;cfg.time_cost_per_s=0.20f;
    cfg.contact_penalty=10.0f;cfg.stable_arrival_bonus=10.0f;
    cfg.command_smoothness_weight=0.05f;
    const float start_low[3]={-1.0f,-2.0f,0.8f};
    const float start_high[3]={1.0f,2.0f,2.4f};
    const float goal_low[3]={-1.2f,-4.0f,0.5f};
    const float goal_high[3]={13.2f,4.0f,4.5f};
    for(uint axis=0;axis<3;axis++) {
        cfg.start_min_xyz[axis]=start_low[axis];cfg.start_max_xyz[axis]=start_high[axis];
        cfg.goal_min_xyz[axis]=goal_low[axis];cfg.goal_max_xyz[axis]=goal_high[axis];
    }
    if(stage==NAV_TASK_STAGE_NEAR_GOAL_HOLD) {
        cfg.goal_distance_min_m=0.8f;cfg.goal_distance_max_m=2.0f;
        for(uint axis=0;axis<3;axis++) {
            cfg.start_min_xyz[axis]=goal_low[axis];
            cfg.start_max_xyz[axis]=goal_high[axis];
        }
    } else if(stage==NAV_TASK_STAGE_CLUTTER_GOAL) {
        cfg.goal_distance_min_m=2.0f;cfg.goal_distance_max_m=7.0f;
    } else {
        cfg.goal_distance_min_m=2.0f;cfg.goal_distance_max_m=6.0f;
    }
    return true;
}
#endif

// This per-environment sidecar is separate from SimRun/checkpoint ABI. Its
// witness points are generator diagnostics only. The actor receives only the
// ordinary depth, ego-state and goal observation assembled by sim_observe.
struct NavigationTaskState {
    uint generation_status;
    uint valid;
    uint generation_attempts;
    uint family;
    uint stage;
    uint objective;
    uint max_nav_steps;
    uint step_count;
    uint stable_ticks;
    uint was_inside_goal;
    uint waypoint_event_latched;
    uint terminal;
    float start_position[3];
    float start_yaw_rad;
    float start_velocity_world[3];
    float goal_position[3];
    float initial_distance_m;
    float previous_distance_m;
    float difficulty;
    float start_clearance_m;
    float goal_clearance_m;
    float direct_segment_clearance_m;
    float witness_min_clearance_m;
    float witness_length_m;
    float stable_time_s;
    float last_executed_world_command[4]; // xyz velocity and yaw rate
    uint witness_point_count;
    float witness_points[NAV_TASK_MAX_WITNESS_POINTS][3];
};

struct NavigationTaskStep {
    float reward;
    float goal_distance_m;
    float actual_speed_mps;
    uint waypoint_passed;
    uint stable_success;
    uint collision;
    uint timeout;
    uint episode_done;
};

struct NavigationTaskWitness {
    WVec points[NAV_TASK_MAX_WITNESS_POINTS];
    uint count;
    float min_clearance_m;
    float length_m;
};

#ifndef __METAL_VERSION__
static_assert(sizeof(NavigationTaskConfig) == 140, "navigation task config ABI changed");
static_assert(sizeof(NavigationTaskControl) == 144, "navigation task control ABI changed");
#endif

NT_WF bool navigation_task_finite(float value) {
#ifdef __METAL_VERSION__
    return isfinite(value);
#else
    return std::isfinite(value);
#endif
}

NT_WF float navigation_task_clamp(float value,float lo,float hi) {
    return fmin(fmax(value,lo),hi);
}

NT_WF WVec navigation_task_random_direction(NT_WT uint& rng) {
    const float z=2.0f*wurand(rng)-1.0f;
    const float angle=6.28318530718f*wurand(rng);
    const float radial=sqrt(fmax(0.0f,1.0f-z*z));
    return wv(radial*cos(angle),radial*sin(angle),z);
}

NT_WF WVec navigation_task_random_point(NT_WT uint& rng,NAV_TASK_CONST float* low,NAV_TASK_CONST float* high) {
    return wv(low[0]+(high[0]-low[0])*wurand(rng),
              low[1]+(high[1]-low[1])*wurand(rng),
              low[2]+(high[2]-low[2])*wurand(rng));
}

NT_WF float navigation_task_distance(NT_WT const float* a,NT_WT const float* b) {
    const float x=b[0]-a[0],y=b[1]-a[1],z=b[2]-a[2];
    return sqrt(x*x+y*y+z*z);
}

NT_WF bool navigation_task_inside_bounds(WVec point,NAV_TASK_CONST float* low,NAV_TASK_CONST float* high) {
    return point.x>=low[0]&&point.x<=high[0]&&point.y>=low[1]&&point.y<=high[1]&&
           point.z>=low[2]&&point.z<=high[2];
}

NT_WF float navigation_task_sample_goal_distance(NT_WT uint& rng,NAV_TASK_CONST NavigationTaskConfig& cfg) {
    const float span=cfg.goal_distance_max_m-cfg.goal_distance_min_m;
    const float center=cfg.goal_distance_min_m+cfg.difficulty*span;
    const float jitter=(wurand(rng)-0.5f)*0.20f*span;
    return navigation_task_clamp(center+jitter,cfg.goal_distance_min_m,cfg.goal_distance_max_m);
}

NT_WF float navigation_task_segment_clearance(NT_WP const WWorld& world,WVec a,WVec b) {
    const float distance=wl(ws(b,a));
    uint count=uint(ceil(distance/0.05f));
    if(count<1u)count=1u;
    if(count>512u)return -1e6f;
    float minimum=1e20f;
    for(uint sample=0;sample<=count;sample++) {
        const float t=float(sample)/float(count);
        const WVec point=wa(a,wm(ws(b,a),t));
        minimum=fmin(minimum,wclearance(world,point,0.0f));
    }
    return minimum;
}

NT_WF float navigation_task_test_path(NT_WP const WWorld& world,NT_WT const WVec* points,uint count) {
    if(count<2u||count>NAV_TASK_MAX_WITNESS_POINTS)return -1e6f;
    float minimum=1e20f;
    for(uint segment=0;segment+1u<count;segment++)
        minimum=fmin(minimum,navigation_task_segment_clearance(world,points[segment],points[segment+1u]));
    return minimum;
}

NT_WF float navigation_task_path_length(NT_WT const WVec* points,uint count) {
    float total=0;
    for(uint segment=0;segment+1u<count;segment++)total+=wl(ws(points[segment+1u],points[segment]));
    return total;
}

NT_WF void navigation_task_save_witness(NT_WT NavigationTaskWitness& out,NT_WT const WVec* points,
                                     uint count,float clearance) {
    out.count=count;
    out.min_clearance_m=clearance;
    out.length_m=navigation_task_path_length(points,count);
    for(uint i=0;i<NAV_TASK_MAX_WITNESS_POINTS;i++)out.points[i]=i<count?points[i]:wv(0,0,0);
}

// Return only a validated direct line or one of a bounded set of geometric
// doglegs. A blocked straight line never silently becomes an accepted task.
NT_WF bool navigation_task_find_witness(NT_WP const WWorld& world,WVec start,WVec goal,
                                     NAV_TASK_CONST NavigationTaskConfig& cfg,
                                     NT_WT NavigationTaskWitness& witness,
                                     NT_WT float& direct_clearance) {
    const WVec direct[2]={start,goal};
    direct_clearance=navigation_task_test_path(world,direct,2u);
    if(direct_clearance>cfg.minimum_witness_clearance_m && cfg.require_detour==0u) {
        navigation_task_save_witness(witness,direct,2u,direct_clearance);
        return true;
    }
    if(direct_clearance>cfg.minimum_witness_clearance_m)return false;

    const WVec delta=ws(goal,start);
    const float x_fraction[5]={0.10f,0.25f,0.50f,0.75f,0.90f};
    const float side_y[2]={-4.60f,4.60f};
    for(uint side=0;side<2u;side++)for(uint i=0;i<5u;i++) {
        const float t=x_fraction[i];
        const WVec waypoint=wv(start.x+t*delta.x,side_y[side],start.z+t*delta.z);
        const WVec path[3]={start,waypoint,goal};
        const float clearance=navigation_task_test_path(world,path,3u);
        if(clearance>cfg.minimum_witness_clearance_m) {
            navigation_task_save_witness(witness,path,3u,clearance);
            return true;
        }
    }
    const float side_offset[4]={-2.4f,-1.2f,1.2f,2.4f};
    for(uint i=0;i<4u;i++) {
        const float t=0.5f;
        const WVec waypoint=wv(start.x+t*delta.x,
                               navigation_task_clamp((start.y+goal.y)*0.5f+side_offset[i],-4.60f,4.60f),
                               start.z+t*delta.z);
        const WVec path[3]={start,waypoint,goal};
        const float clearance=navigation_task_test_path(world,path,3u);
        if(clearance>cfg.minimum_witness_clearance_m) {
            navigation_task_save_witness(witness,path,3u,clearance);
            return true;
        }
    }
    const float detour_z[2]={0.55f,4.45f};
    for(uint i=0;i<2u;i++) {
        const WVec waypoint=wv(start.x+0.5f*delta.x,start.y+0.5f*delta.y,detour_z[i]);
        const WVec path[3]={start,waypoint,goal};
        const float clearance=navigation_task_test_path(world,path,3u);
        if(clearance>cfg.minimum_witness_clearance_m) {
            navigation_task_save_witness(witness,path,3u,clearance);
            return true;
        }
    }
    for(uint side=0;side<2u;side++) {
        const float y=side_y[side];
        const WVec first=wv(start.x+0.20f*delta.x,y,start.z+0.20f*delta.z);
        const WVec second=wv(start.x+0.80f*delta.x,y,start.z+0.80f*delta.z);
        const WVec path[4]={start,first,second,goal};
        const float clearance=navigation_task_test_path(world,path,4u);
        if(clearance>cfg.minimum_witness_clearance_m) {
            navigation_task_save_witness(witness,path,4u,clearance);
            return true;
        }
    }
    return false;
}

NT_WF bool navigation_task_static_world(NT_WP const WWorld& world) {
    for(uint obstacle=0;obstacle<world.count;obstacle++)for(uint axis=0;axis<3;axis++)
        if(fabs(world.obstacles[obstacle].velocity[axis])>1e-8f)return false;
    return true;
}

// Generate a safe start/world/goal triple. A failed bounded search is explicit:
// caller must reschedule or stop, never keep an empty or unsafe task.
NT_WF uint navigation_generate_task(NT_WP WWorld& world,NT_WP NavigationTaskState& task,
                                 NT_WT uint& rng,NAV_TASK_CONST NavigationTaskConfig& cfg) {
    task.generation_status=NAV_TASK_GENERATION_INVALID_CONFIG;
    task.generation_attempts=0;
    task.valid=0;
    const bool valid_stage=cfg.stage<=NAV_TASK_STAGE_CLUTTER_GOAL;
    const bool valid_objective=cfg.objective<=NAV_TASK_OBJECTIVE_FINAL_HOLD;
    if(!valid_stage||!valid_objective||cfg.family>16u||
       cfg.require_detour>1u||
       !navigation_task_finite(cfg.scene_distance_m)||cfg.scene_distance_m<3.0f||cfg.scene_distance_m>10.0f||
       !navigation_task_finite(cfg.difficulty)||cfg.difficulty<0.0f||cfg.difficulty>1.0f||
       !navigation_task_finite(cfg.goal_distance_min_m)||!navigation_task_finite(cfg.goal_distance_max_m)||
       cfg.goal_distance_min_m<=cfg.goal_radius_m||cfg.goal_distance_max_m<cfg.goal_distance_min_m||
       !navigation_task_finite(cfg.minimum_endpoint_clearance_m)||cfg.minimum_endpoint_clearance_m<0.30f||
       !navigation_task_finite(cfg.minimum_witness_clearance_m)||cfg.minimum_witness_clearance_m<0.02f||
       !navigation_task_finite(cfg.goal_radius_m)||cfg.goal_radius_m<=0.0f||
       !navigation_task_finite(cfg.stable_speed_mps)||cfg.stable_speed_mps<0.0f||
       !navigation_task_finite(cfg.stable_hold_s)||cfg.stable_hold_s<0.0f||
       !navigation_task_finite(cfg.nav_period_s)||cfg.nav_period_s<=0.0f||
       !navigation_task_finite(cfg.initial_speed_min_mps)||!navigation_task_finite(cfg.initial_speed_max_mps)||
       cfg.initial_speed_min_mps<0.0f||cfg.initial_speed_max_mps<cfg.initial_speed_min_mps||
       !navigation_task_finite(cfg.progress_reward_per_m)||cfg.progress_reward_per_m<0.0f||
       !navigation_task_finite(cfg.time_cost_per_s)||cfg.time_cost_per_s<0.0f||
       !navigation_task_finite(cfg.contact_penalty)||cfg.contact_penalty<0.0f||
       !navigation_task_finite(cfg.stable_arrival_bonus)||cfg.stable_arrival_bonus<0.0f||
       !navigation_task_finite(cfg.command_smoothness_weight)||cfg.command_smoothness_weight<0.0f||
       cfg.max_nav_steps==0u||cfg.max_generation_attempts==0u)return NAV_TASK_GENERATION_INVALID_CONFIG;
    if((cfg.stage==NAV_TASK_STAGE_OPEN_GOAL||cfg.stage==NAV_TASK_STAGE_NEAR_GOAL_HOLD)&&cfg.family!=0u)
        return NAV_TASK_GENERATION_INVALID_CONFIG;
    if(cfg.stage==NAV_TASK_STAGE_CLUTTER_GOAL&&
       !(cfg.family==1u||cfg.family==2u||cfg.family==4u||cfg.family==5u))
        return NAV_TASK_GENERATION_UNSUPPORTED_FAMILY;
    for(uint axis=0;axis<3;axis++) {
        const float room_min=axis==0?-2.0f:(axis==1?-5.0f:0.0f);
        const float room_max=axis==0?14.0f:5.0f;
        if(!navigation_task_finite(cfg.start_min_xyz[axis])||!navigation_task_finite(cfg.start_max_xyz[axis])||
           !navigation_task_finite(cfg.goal_min_xyz[axis])||!navigation_task_finite(cfg.goal_max_xyz[axis])||
           cfg.start_min_xyz[axis]>=cfg.start_max_xyz[axis]||cfg.goal_min_xyz[axis]>=cfg.goal_max_xyz[axis]||
           cfg.start_min_xyz[axis]<room_min+cfg.minimum_endpoint_clearance_m||
           cfg.start_max_xyz[axis]>room_max-cfg.minimum_endpoint_clearance_m||
           cfg.goal_min_xyz[axis]<room_min+cfg.minimum_endpoint_clearance_m||
           cfg.goal_max_xyz[axis]>room_max-cfg.minimum_endpoint_clearance_m)
            return NAV_TASK_GENERATION_INVALID_CONFIG;
    }

    uint attempts=cfg.max_generation_attempts;
    if(attempts>NAV_TASK_MAX_GENERATION_ATTEMPTS)attempts=NAV_TASK_MAX_GENERATION_ATTEMPTS;
    for(uint attempt=0;attempt<attempts;attempt++) {
        task.generation_attempts=attempt+1u;
        const uint scene_seed=wrng(rng);
        wgenerate(world,scene_seed,cfg.family,cfg.scene_distance_m);
        if(cfg.stage==NAV_TASK_STAGE_CLUTTER_GOAL&&!navigation_task_static_world(world)) {
            task.generation_status=NAV_TASK_GENERATION_UNSUPPORTED_DYNAMIC;
            return task.generation_status;
        }

        const WVec direction=navigation_task_random_direction(rng);
        const float distance=navigation_task_sample_goal_distance(rng,cfg);
        WVec start,goal;
        if(cfg.stage==NAV_TASK_STAGE_NEAR_GOAL_HOLD) {
            goal=navigation_task_random_point(rng,cfg.goal_min_xyz,cfg.goal_max_xyz);
            start=ws(goal,wm(direction,distance));
        } else {
            start=navigation_task_random_point(rng,cfg.start_min_xyz,cfg.start_max_xyz);
            goal=wa(start,wm(direction,distance));
        }
        if(!navigation_task_finite(goal.x)||!navigation_task_finite(goal.y)||!navigation_task_finite(goal.z))continue;
        if(!navigation_task_inside_bounds(start,cfg.start_min_xyz,cfg.start_max_xyz)||
           !navigation_task_inside_bounds(goal,cfg.goal_min_xyz,cfg.goal_max_xyz))continue;
        const float start_clear=wclearance(world,start,0.0f);
        const float goal_clear=wclearance(world,goal,0.0f);
        if(start_clear<=cfg.minimum_endpoint_clearance_m||goal_clear<=cfg.minimum_endpoint_clearance_m)continue;
        const float actual_distance=wl(ws(goal,start));
        if(actual_distance<cfg.goal_distance_min_m||actual_distance>cfg.goal_distance_max_m)continue;

        NavigationTaskWitness witness{};
        float direct_clearance=0.0f;
        if(!navigation_task_find_witness(world,start,goal,cfg,witness,direct_clearance))continue;
        world.goal[0]=goal.x;world.goal[1]=goal.y;world.goal[2]=goal.z;

        task.family=cfg.family;task.stage=cfg.stage;task.objective=cfg.objective;
        task.max_nav_steps=cfg.max_nav_steps;task.step_count=0;task.stable_ticks=0;
        task.was_inside_goal=0;task.waypoint_event_latched=0;task.terminal=0;
        task.start_position[0]=start.x;task.start_position[1]=start.y;task.start_position[2]=start.z;
        task.start_yaw_rad=(wurand(rng)*2.0f-1.0f)*3.14159265359f;
        const float initial_speed=cfg.initial_speed_min_mps+
            (cfg.initial_speed_max_mps-cfg.initial_speed_min_mps)*wurand(rng);
        task.start_velocity_world[0]=direction.x*initial_speed;
        task.start_velocity_world[1]=direction.y*initial_speed;
        task.start_velocity_world[2]=direction.z*initial_speed;
        task.goal_position[0]=goal.x;task.goal_position[1]=goal.y;task.goal_position[2]=goal.z;
        task.initial_distance_m=actual_distance;task.previous_distance_m=actual_distance;
        task.difficulty=cfg.difficulty;
        task.start_clearance_m=start_clear;task.goal_clearance_m=goal_clear;
        task.direct_segment_clearance_m=direct_clearance;
        task.witness_min_clearance_m=witness.min_clearance_m;task.witness_length_m=witness.length_m;
        task.stable_time_s=0.0f;
        for(uint axis=0;axis<3;axis++)task.last_executed_world_command[axis]=task.start_velocity_world[axis];
        task.last_executed_world_command[3]=0.0f;
        task.witness_point_count=witness.count;
        for(uint point=0;point<NAV_TASK_MAX_WITNESS_POINTS;point++)
            for(uint axis=0;axis<3;axis++)task.witness_points[point][axis]=point<witness.count?
                (axis==0?witness.points[point].x:(axis==1?witness.points[point].y:witness.points[point].z)):0.0f;
        task.valid=1;
        task.generation_status=NAV_TASK_GENERATION_READY;
        return task.generation_status;
    }
    task.generation_status=NAV_TASK_GENERATION_EXHAUSTED;
    return task.generation_status;
}

// Copy only physical start state. The caller sets RAPTOR recurrence/motors and
// integrates the persistent reference from start_position separately.
NT_WF void navigation_task_apply_start(NT_WP const NavigationTaskState& task,NT_WP RLPhysicsState& state) {
    for(uint axis=0;axis<3;axis++) {
        state.position[axis]=task.start_position[axis];
        state.linear_velocity[axis]=task.start_velocity_world[axis];
        state.angular_velocity_body[axis]=0.0f;
    }
    const float half_yaw=task.start_yaw_rad*0.5f;
    state.orientation_wxyz[0]=cos(half_yaw);state.orientation_wxyz[1]=0.0f;
    state.orientation_wxyz[2]=0.0f;state.orientation_wxyz[3]=sin(half_yaw);
}

// A waypoint event is nonterminal. Final-goal success requires a stable hold,
// so crossing the radius at speed cannot end an episode successfully.
NT_WF NavigationTaskStep navigation_task_step(NT_WP NavigationTaskState& task,
                                           NAV_TASK_CONST NavigationTaskConfig& cfg,
                                           NT_WP const float* position_world,
                                           NT_WP const float* velocity_world,
                                           NT_WT const float* executed_world_command,
                                           uint actual_contact,uint forced_timeout) {
    NavigationTaskStep result{};
    if(task.generation_status!=NAV_TASK_GENERATION_READY||task.terminal!=0u) {
        result.episode_done=1u;return result;
    }
    const float dx=task.goal_position[0]-position_world[0];
    const float dy=task.goal_position[1]-position_world[1];
    const float dz=task.goal_position[2]-position_world[2];
    const float distance=sqrt(dx*dx+dy*dy+dz*dz);
    const float speed=sqrt(velocity_world[0]*velocity_world[0]+velocity_world[1]*velocity_world[1]+velocity_world[2]*velocity_world[2]);
    if(!navigation_task_finite(distance)||!navigation_task_finite(speed)) {
        result.collision=1u;result.episode_done=1u;task.terminal=1u;return result;
    }
    task.step_count++;
    result.goal_distance_m=distance;result.actual_speed_mps=speed;
    const float progress=task.previous_distance_m-distance;
    result.reward=cfg.progress_reward_per_m*progress-cfg.time_cost_per_s*cfg.nav_period_s;
    task.previous_distance_m=distance;
    const uint inside=(distance<=cfg.goal_radius_m)?1u:0u;
    result.waypoint_passed=(task.objective==NAV_TASK_OBJECTIVE_WAYPOINT&&inside!=0u&&task.waypoint_event_latched==0u)?1u:0u;
    if(result.waypoint_passed)task.waypoint_event_latched=1u;
    const uint stable_now=(inside!=0u&&speed<=cfg.stable_speed_mps)?1u:0u;
    if(task.objective==NAV_TASK_OBJECTIVE_FINAL_HOLD&&stable_now!=0u)
        task.stable_time_s+=cfg.nav_period_s;
    else task.stable_time_s=0.0f;
    task.stable_ticks=uint(ceil(task.stable_time_s/fmax(cfg.nav_period_s,1e-6f)-1e-5f));
    result.stable_success=(task.objective==NAV_TASK_OBJECTIVE_FINAL_HOLD&&stable_now!=0u&&
                           task.stable_time_s+1e-6f>=cfg.stable_hold_s)?1u:0u;
    result.collision=actual_contact?1u:0u;
    result.timeout=(forced_timeout||task.step_count>=task.max_nav_steps)?1u:0u;
    if(cfg.command_smoothness_weight>0.0f) {
        float difference=0.0f;
        for(uint axis=0;axis<4;axis++) {
            const float delta=executed_world_command[axis]-task.last_executed_world_command[axis];
            difference+=delta*delta;
            task.last_executed_world_command[axis]=executed_world_command[axis];
        }
        result.reward-=cfg.command_smoothness_weight*difference*cfg.nav_period_s;
    }
    if(result.collision)result.reward-=cfg.contact_penalty;
    if(result.stable_success&&!result.collision)result.reward+=cfg.stable_arrival_bonus;
    result.episode_done=(result.collision||result.stable_success||result.timeout)?1u:0u;
    task.terminal=result.episode_done;
    return result;
}

#undef NAV_TASK_CONST

#undef NT_WP
#undef NT_WT
#undef NT_WF
