#pragma once

// Include this Objective-C++ host helper after main.mm declares Metal, Sim,
// SimConfig, SimRun, and after challenge_evaluation.hpp is available.
// It imports explicit Webots challenge metadata into the existing WWorld ABI.

#import <Foundation/Foundation.h>
#include "../world.hpp"
#include "../deployment.hpp"
#include "../ppo.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace simulator_comparison {

struct Scene {
    WWorld world{};
    std::string family_name;
    uint32_t seed=0;
    float distance=0;
    std::vector<std::array<float,3>> witness;
};

struct Episode {
    std::string family_name;
    uint32_t seed=0,mode=17,steps=0;
    bool success=false,collision=false,timeout=false;
    float time_s=0,path_m=0,final_error_m=0,peak_speed_mps=0,min_clearance_m=0;
};

[[noreturn]] inline void fail(const std::string& text) {
    throw std::runtime_error("simulator comparison: "+text);
}

inline NSDictionary* read_json_dict(const std::string& path) {
    NSString* file_path=[NSString stringWithUTF8String:path.c_str()];
    NSData* data=[NSData dataWithContentsOfFile:file_path];
    if(!data)fail("cannot read JSON: "+path);
    NSError* error=nil;
    id root=[NSJSONSerialization JSONObjectWithData:data options:NSJSONReadingFragmentsAllowed error:&error];
    if(error || ![root isKindOfClass:[NSDictionary class]])
        fail("invalid JSON object: "+path);
    return (NSDictionary*)root;
}

inline id field(NSDictionary* object,const char* name,const std::string& where) {
    return challenge_evaluation::required_field(object,name,where);
}

inline Scene load_bounded_webots_scene(const std::string& path) {
    NSDictionary* record=read_json_dict(path);
    const std::string where=path;
    const std::string schema=challenge_evaluation::required_string(field(record,"schema",where),where+".schema");
    if(schema!="webots-challenge-v1")fail(where+" has an unsupported schema");
    id bounded_value=field(record,"bounded_room",where);
    if(![bounded_value isKindOfClass:[NSNumber class]] || ![(NSNumber*)bounded_value boolValue])
        fail(where+" is an open-room scene; paired comparison requires bounded_room=true");
    NSDictionary* bounds=challenge_evaluation::required_dict(field(record,"room_bounds_m",where),where+".room_bounds_m");
    const std::array<std::array<float,2>,3> expected{{{{-2,14}},{{-5,5}},{{0,5}}}};
    const char* axis_names[3]={"x","y","z"};
    for(uint32_t axis=0;axis<3;axis++) {
        NSArray* pair=challenge_evaluation::required_array(field(bounds,axis_names[axis],where+".room_bounds_m"),where+".room_bounds_m");
        if([pair count]!=2)fail(where+" room bound must have two coordinates");
        const float lo=challenge_evaluation::required_float(pair[0],where+".room_bounds_m.lower");
        const float hi=challenge_evaluation::required_float(pair[1],where+".room_bounds_m.upper");
        if(std::fabs(lo-expected[axis][0])>1e-5f||std::fabs(hi-expected[axis][1])>1e-5f)
            fail(where+" room bounds do not match world.hpp");
    }

    Scene scene;
    scene.seed=challenge_evaluation::required_uint(field(record,"seed",where),where+".seed");
    const std::string family=challenge_evaluation::required_string(field(record,"family",where),where+".family");
    // Reuse the equivalent procedural family ids for labels/diagnostics. The
    // obstacles below still come only from the saved Webots scene record.
    if(family=="doorway"){scene.family_name=family;scene.world.family=4;}
    else if(family=="table_overhang"){scene.family_name=family;scene.world.family=5;}
    else if(family=="mixed_clutter"){scene.family_name=family;scene.world.family=6;}
    else fail(where+" has an unknown Webots family");
    scene.world.seed=scene.seed;
    const auto start=challenge_evaluation::required_vec3(field(record,"start_xyz_m",where),where+".start_xyz_m");
    const auto goal=challenge_evaluation::required_vec3(field(record,"goal_xyz_m",where),where+".goal_xyz_m");
    const auto expected_start=std::array<float,3>{{0,0,1.5f}};
    for(uint32_t j=0;j<3;j++) {
        if(std::fabs(start[j]-expected_start[j])>1e-5f)fail(where+" start state does not match the Metal reset");
        scene.world.goal[j]=goal[j];
        scene.world.wind[j]=0.0f;
    }
    const float radius=challenge_evaluation::required_float(field(record,"body_collision_radius_m",where),where+".body_collision_radius_m");
    if(std::fabs(radius-0.18f)>1e-5f)fail(where+" collision radius does not match Webots vehicle PROTO");
    scene.distance=std::sqrt((goal[0]-start[0])*(goal[0]-start[0])+
                             (goal[1]-start[1])*(goal[1]-start[1])+
                             (goal[2]-start[2])*(goal[2]-start[2]));
    NSArray* obstacles=challenge_evaluation::required_array(field(record,"obstacles",where),where+".obstacles");
    if([obstacles count]>16)fail(where+" has more than 16 Webots obstacles");
    scene.world.count=uint32_t([obstacles count]);
    for(uint32_t i=0;i<scene.world.count;i++) {
        NSDictionary* input=challenge_evaluation::required_dict(obstacles[i],where+".obstacles[]");
        const std::string kind=challenge_evaluation::required_string(field(input,"kind",where),where+".obstacles.kind");
        const auto center=challenge_evaluation::required_vec3(field(input,"center",where),where+".obstacles.center");
        WObstacle& output=scene.world.obstacles[i];
        for(uint32_t j=0;j<3;j++){output.center[j]=center[j];output.velocity[j]=0;}
        if(kind=="box") {
            const auto size=challenge_evaluation::required_vec3(field(input,"size",where),where+".obstacles.size");
            output.kind=0;for(uint32_t j=0;j<3;j++)output.size[j]=0.5f*size[j];
        } else if(kind=="cylinder") {
            output.kind=2;
            output.size[0]=challenge_evaluation::required_float(field(input,"radius",where),where+".obstacles.radius");
            output.size[1]=output.size[0];
            output.size[2]=0.5f*challenge_evaluation::required_float(field(input,"height",where),where+".obstacles.height");
        } else if(kind=="sphere") {
            output.kind=1;
            const float r=challenge_evaluation::required_float(field(input,"radius",where),where+".obstacles.radius");
            output.size[0]=output.size[1]=output.size[2]=r;
        } else fail(where+" has unsupported obstacle kind "+kind);
    }
    NSDictionary* witness=challenge_evaluation::required_dict(field(record,"route_witness",where),where+".route_witness");
    NSArray* points=challenge_evaluation::required_array(field(witness,"points_xyz_m",where+".route_witness"),where+".route_witness.points_xyz_m");
    if([points count]<2)fail(where+" route witness must have at least two points");
    for(id point:points)scene.witness.push_back(challenge_evaluation::required_vec3(point,where+".route_witness.point"));
    for(const auto& point:scene.witness)
        if(wclearance(scene.world,wv(point[0],point[1],point[2]),0.0f)<0.0f)
            fail(where+" witness intersects the mapped Metal world");
    return scene;
}

inline fixed_ppo::ActorParams load_deployed_actor(const std::string& path) {
    static_assert(fixed_ppo::actor_obs_dim==184,"Webots deployed policies require 184 inputs");
    nav_deployment::NavigationPolicy policy;
    std::string error;
    if(!policy.load(path,&error))fail(error);
    if(policy.metadata().weight_count!=fixed_ppo::actor_param_count)
        fail("deployment weights do not match fixed PPO actor layout");
    fixed_ppo::ActorParams actor{};
    std::copy(policy.weights().begin(),policy.weights().end(),actor.values.begin());
    return actor;
}

// The caller must have compiled base_source()+PPO_TRAINER_MSL. This runner
// replaces only the sampled WWorld with explicit Webots geometry; it does not
// pass Webots contact points, waypoints, or obstacle metadata to the actor.
inline Episode run_metal_actor(Metal& metal,const Scene& scene,
                               const fixed_ppo::ActorParams& actor,
                               uint32_t mode=17,float speed=1.5f,
                               uint32_t max_steps=160) {
    if(mode!=17&&mode!=13)fail("comparison mode must be learned-guidance 17 or geometric-only 13");
    if(!std::isfinite(speed)||speed<=0||speed>1.5f||!max_steps)fail("invalid comparison speed or horizon");
    SimConfig config;config.n=1;config.family=0;config.mode=mode;config.eval=1;
    config.seed=scene.seed;config.distance=scene.distance;config.speed=speed;
    config.max_steps=max_steps;config.sensor_period=1;config.substeps=5;
    config.velocity_contract=1;config.geometry_memory=1;config.wind=0;
    config.depth_noise=0;config.dropout=0;config.command_delay=0;config.sensor_delay=0;
    Sim sim(metal,config,32);
    std::memcpy(sim.actor.contents,actor.values.data(),sizeof(actor.values));
    auto* worlds=static_cast<WWorld*>(sim.worlds.contents);
    worlds[0]=scene.world;
    auto* runs=static_cast<SimRun*>(sim.runs.contents);
    runs[0].initial_distance=scene.distance;
    auto cb=[metal.queue commandBuffer];
    sim.collect(cb,max_steps);
    metal.finish(cb);
    const auto* states=static_cast<const RLPhysicsState*>(sim.states.contents);
    const float dx=states[0].position[0]-scene.world.goal[0];
    const float dy=states[0].position[1]-scene.world.goal[1];
    const float dz=states[0].position[2]-scene.world.goal[2];
    Episode result;
    result.family_name=scene.family_name;result.seed=scene.seed;result.mode=mode;result.steps=runs[0].steps;
    result.success=runs[0].successes>0;result.collision=runs[0].collisions>0;result.timeout=runs[0].timeouts>0;
    result.time_s=runs[0].elapsed;result.path_m=runs[0].path;result.final_error_m=std::sqrt(dx*dx+dy*dy+dz*dz);
    result.peak_speed_mps=runs[0].peak_speed;result.min_clearance_m=runs[0].min_clearance;
    return result;
}

inline std::array<Episode,2> compare_modes(Metal& metal,const Scene& scene,
                                          const fixed_ppo::ActorParams& actor,
                                          float speed=1.5f,uint32_t max_steps=160) {
    return {run_metal_actor(metal,scene,actor,17,speed,max_steps),
            run_metal_actor(metal,scene,actor,13,speed,max_steps)};
}

} // namespace simulator_comparison
