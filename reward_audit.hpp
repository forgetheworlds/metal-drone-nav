#pragma once

// Read-only reward audit for a small training-bank subset. Include after
// Sim, checkpoint helpers, and challenge_evaluation.hpp are declared.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace reward_audit {

constexpr uint32_t kFamilyCount = 3;
constexpr std::array<uint32_t, kFamilyCount> kFamilies{{14, 15, 16}};
constexpr std::array<float, 3> kGammas{{0.99f, 0.995f, 0.997f}};
constexpr std::array<float, 2> kRiskCoefficients{{0.0f, 0.1f}};

struct RewardAuditFrame {
    float position[3];
    float local_goal[3];
    float raw_reward;
    float elapsed_s;
    uint32_t steps;
    uint32_t successes;
    uint32_t collisions;
    uint32_t timeouts;
};
static_assert(sizeof(RewardAuditFrame) == 48, "reward audit MSL layout changed");

struct Point {
    std::array<float, 3> position{};
    std::array<float, 3> local_goal{};
    float raw_reward = 0;
    float elapsed_s = 0;
    uint32_t steps = 0;
    uint32_t successes = 0;
    uint32_t collisions = 0;
    uint32_t timeouts = 0;
    bool final_waypoint = false;
};

struct Trace {
    std::string controller;
    uint32_t mode = 0;
    uint32_t waypoint_count = 0;
    bool collision = false;
    bool timeout = false;
    bool final_success = false;
    float start_clearance_m = 0;
    float elapsed_s = 0;
    float path_m = 0;
    std::vector<Point> points;
};

static const char* kCaptureMSL = R"MSL(
struct RewardAuditFrame {
    float position[3];
    float local_goal[3];
    float raw_reward;
    float elapsed_s;
    uint steps;
    uint successes;
    uint collisions;
    uint timeouts;
};
kernel void reward_audit_capture(device const RLPhysicsState* states [[buffer(0)]],
                                 device const SimRun* runs [[buffer(1)]],
                                 device const WWorld* worlds [[buffer(2)]],
                                 device const float* rewards [[buffer(3)]],
                                 constant SimConfig& cfg [[buffer(4)]],
                                 device RewardAuditFrame* frames [[buffer(5)]],
                                 constant uint& frame_index [[buffer(6)]],
                                 uint n [[thread_position_in_grid]]) {
    if(n>=cfg.n)return;
    const uint row=cfg.tick*cfg.n+n;
    RewardAuditFrame frame;
    for(uint axis=0;axis<3;axis++) {
        frame.position[axis]=states[n].position[axis];
        frame.local_goal[axis]=worlds[n].goal[axis];
    }
    frame.raw_reward=rewards[row];
    frame.elapsed_s=runs[n].elapsed;
    frame.steps=runs[n].steps;
    frame.successes=runs[n].successes;
    frame.collisions=runs[n].collisions;
    frame.timeouts=runs[n].timeouts;
    frames[frame_index]=frame;
}
)MSL";

inline float distance(const float* a, const float* b) {
    const float x=a[0]-b[0], y=a[1]-b[1], z=a[2]-b[2];
    return std::sqrt(x*x+y*y+z*z);
}

inline float distance(const std::array<float,3>& a, const std::array<float,3>& b) {
    return distance(a.data(),b.data());
}

struct ActorPolicy {
    PpoCheckpointHeader header{};
    std::vector<float> actor;
};

inline ActorPolicy load_actor(const std::string& checkpoint) {
    std::ifstream input(checkpoint,std::ios::binary);
    require(bool(input),"reward audit cannot read checkpoint: "+checkpoint);
    const PpoCheckpointHeader header=read_checkpoint_header(input);
    require(bool(input)&&header.actor_count==fixed_ppo::actor_param_count&&header.version>=3&&header.version<=9,
            "reward audit checkpoint dimensions/version are unsupported");
    require(header.config.velocity_contract==1,"reward audit requires velocity contract1");
    ActorPolicy policy;policy.header=header;policy.actor.resize(header.actor_count);
    input.read(reinterpret_cast<char*>(policy.actor.data()),std::streamsize(policy.actor.size()*sizeof(float)));
    require(bool(input)&&std::all_of(policy.actor.begin(),policy.actor.end(),[](float value){return std::isfinite(value);}),
            "reward audit checkpoint actor is truncated or non-finite");
    return policy;
}

inline void enqueue_step(Sim& sim,id<MTLCommandBuffer> command_buffer,
                         id<MTLComputePipelineState> capture_pipeline,
                         id<MTLBuffer> capture_buffer,uint tick,uint capture_index) {
    Metal& metal=sim.m;
    const id<MTLBuffer> config=sim.configs[tick];
    const size_t environments=sim.cfg.n;
    metal.dispatch(command_buffer,sim.depth_p,environments*320,
                   {sim.states,sim.runs,sim.worlds,sim.sensors,sim.physics,config,sim.poses});
    if(sim.cfg.geometry_memory) {
        metal.dispatch(command_buffer,sim.memory_points_p,environments*640,
                       {sim.states,sim.runs,sim.sensors,sim.poses,sim.memory_points,sim.physics,config});
        metal.dispatch(command_buffer,sim.memory_candidates_p,environments*85,
                       {sim.states,sim.worlds,sim.memory_points,sim.memory_clearances,config});
    }
    metal.dispatch(command_buffer,sim.observe_p,environments,
                   {sim.states,sim.runs,sim.worlds,sim.sensors,sim.obs,sim.co,sim.physics,config,
                    sim.poses,sim.memory_clearances},64);
    encode(metal,command_buffer,sim.simd_actor?"ppo_actor_forward_simd_fused":"ppo_actor_forward",
           sim.simd_actor?((environments+7)/8)*256:environments,
           {{sim.obs,size_t(tick)*environments*fixed_ppo::actor_obs_dim*sizeof(float)},
            {sim.actor,0},{sim.actor_workspace,0},
            {sim.actions,size_t(tick)*environments*4*sizeof(float)},{sim.env_count,0}},
           sim.simd_actor?256:64);
    metal.dispatch(command_buffer,sim.act_p,environments,
                   {sim.states,sim.runs,sim.worlds,sim.obs,sim.co,sim.actor,sim.critic,
                    sim.actions,sim.logp,sim.values,sim.commands,sim.physics,config},64);
    metal.dispatch(command_buffer,sim.advance_p,environments,
                   {sim.states,sim.runs,sim.worlds,sim.sensors,sim.commands,sim.raptor,sim.critic,
                    sim.rewards,sim.next_values,sim.terminated,sim.truncated,sim.physics,config,
                    sim.bank_worlds,sim.bank_schedule,sim.bank_control,sim.bank_active_ids,
                    sim.bank_transition_ids,sim.environment_physics,sim.runtime_control,
                    sim.task_states,sim.task_control,sim.potential_fields,sim.potential_spec,
                    sim.potential_control},64);

    id<MTLComputeCommandEncoder> encoder=[command_buffer computeCommandEncoder];
    [encoder setComputePipelineState:capture_pipeline];
    [encoder setBuffer:sim.states offset:0 atIndex:0];
    [encoder setBuffer:sim.runs offset:0 atIndex:1];
    [encoder setBuffer:sim.worlds offset:0 atIndex:2];
    [encoder setBuffer:sim.rewards offset:0 atIndex:3];
    [encoder setBuffer:config offset:0 atIndex:4];
    [encoder setBuffer:capture_buffer offset:0 atIndex:5];
    [encoder setBytes:&capture_index length:sizeof(capture_index) atIndex:6];
    [encoder dispatchThreads:MTLSizeMake(environments,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)];
    [encoder endEncoding];
}

inline Point decode_frame(const RewardAuditFrame& frame,bool final_waypoint) {
    Point point;
    for(uint32_t axis=0;axis<3;axis++) {
        point.position[axis]=frame.position[axis];
        point.local_goal[axis]=frame.local_goal[axis];
    }
    point.raw_reward=frame.raw_reward;
    point.elapsed_s=frame.elapsed_s;
    point.steps=frame.steps;
    point.successes=frame.successes;
    point.collisions=frame.collisions;
    point.timeouts=frame.timeouts;
    point.final_waypoint=final_waypoint;
    return point;
}

inline void collect_segment(Sim& sim,id<MTLComputePipelineState> capture_pipeline,
                            id<MTLBuffer> capture_buffer,uint base_step,uint count) {
    require(count<=sim.horizon&&base_step+count<=sim.horizon,
            "reward audit capture exceeds allocated rollout horizon");
    auto command_buffer=[sim.m.queue commandBuffer];
    for(uint tick=0;tick<count;tick++)
        enqueue_step(sim,command_buffer,capture_pipeline,capture_buffer,tick,base_step+tick);
    sim.m.finish(command_buffer);
}

inline void prepare_sim(Sim& sim,const challenge_evaluation::Level& level,
                        const ActorPolicy& policy) {
    std::memcpy(sim.actor.contents,policy.actor.data(),policy.actor.size()*sizeof(float));
    static_cast<WWorld*>(sim.worlds.contents)[0]=level.world;
    auto& run=static_cast<SimRun*>(sim.runs.contents)[0];
    run.rng=level.seed;run.steps=0;run.initial_distance=std::sqrt(
        level.world.goal[0]*level.world.goal[0]+level.world.goal[1]*level.world.goal[1]+
        (level.world.goal[2]-1.5f)*(level.world.goal[2]-1.5f));
    run.reference_position[0]=0;run.reference_position[1]=0;run.reference_position[2]=1.5f;
}

inline Trace run_global_policy(Metal& metal,const challenge_evaluation::Level& level,
                               const ActorPolicy& policy,uint32_t max_steps,float risk_coef,
                               id<MTLComputePipelineState> capture_pipeline) {
    SimConfig config;config.n=1;config.eval=1;config.mode=17;config.family=0;
    config.seed=level.seed;config.distance=level.distance;config.speed=1.5f;
    config.max_steps=max_steps;config.geometry_memory=policy.header.config.geometry_memory;
    config.velocity_contract=policy.header.config.velocity_contract;config.risk_coef=risk_coef;
    Sim sim(metal,config,max_steps);
    prepare_sim(sim,level,policy);
    const auto start=static_cast<const RLPhysicsState*>(sim.states.contents)[0];
    Trace trace;trace.controller="global_goal_mode17";trace.mode=17;
    trace.start_clearance_m=wclearance(level.world,wv(start.position[0],start.position[1],start.position[2]),0.0f);
    auto capture=metal.buffer(size_t(max_steps)*sizeof(RewardAuditFrame));
    collect_segment(sim,capture_pipeline,capture,0,max_steps);
    const auto* frames=static_cast<const RewardAuditFrame*>(capture.contents);
    uint32_t previous_steps=0;
    for(uint32_t i=0;i<max_steps;i++) {
        if(frames[i].steps<=previous_steps)break;
        trace.points.push_back(decode_frame(frames[i],true));
        previous_steps=frames[i].steps;
    }
    const auto& run=static_cast<const SimRun*>(sim.runs.contents)[0];
    trace.collision=run.collisions!=0;
    trace.timeout=run.timeouts!=0;
    trace.final_success=run.successes!=0;
    trace.elapsed_s=run.elapsed;trace.path_m=run.path;
    return trace;
}

inline Trace run_witness_policy(Metal& metal,const challenge_evaluation::Level& level,
                                const ActorPolicy& policy,uint32_t control_mode,uint32_t max_steps,
                                float risk_coef,id<MTLComputePipelineState> capture_pipeline) {
    require(level.witness_route.size()>=2,"reward audit level has no geometric witness");
    require(control_mode==20||control_mode==21,"reward audit witness controller must be mode20 or mode21");
    SimConfig config;config.n=1;config.eval=1;config.mode=control_mode;config.family=0;
    config.seed=level.seed;config.distance=level.distance;config.speed=1.5f;
    config.max_steps=max_steps;config.geometry_memory=policy.header.config.geometry_memory;
    config.velocity_contract=policy.header.config.velocity_contract;config.risk_coef=risk_coef;
    Sim sim(metal,config,max_steps);
    prepare_sim(sim,level,policy);
    auto* world=static_cast<WWorld*>(sim.worlds.contents);
    auto* run=static_cast<SimRun*>(sim.runs.contents);
    const auto start=static_cast<const RLPhysicsState*>(sim.states.contents)[0];
    Trace trace;trace.controller=control_mode==20?"goal_script_witness_mode20":"actor_witness_mode21";trace.mode=control_mode;
    trace.start_clearance_m=wclearance(level.world,wv(start.position[0],start.position[1],start.position[2]),0.0f);
    auto capture=metal.buffer(size_t(max_steps)*sizeof(RewardAuditFrame));
    uint32_t waypoint=1,steps_completed=0;
    while(waypoint<level.witness_route.size()&&steps_completed<max_steps) {
        for(uint32_t axis=0;axis<3;axis++)world[0].goal[axis]=level.witness_route[waypoint][axis];
        const uint32_t before_steps=run[0].steps;
        collect_segment(sim,capture_pipeline,capture,steps_completed,max_steps-steps_completed);
        const uint32_t after_steps=run[0].steps;
        const uint32_t active=after_steps-before_steps;
        require(active<=max_steps-steps_completed,"witness audit step counter moved backwards");
        const auto* frames=static_cast<const RewardAuditFrame*>(capture.contents);
        const bool final_waypoint=waypoint+1==level.witness_route.size();
        uint32_t last_recorded=before_steps;
        for(uint32_t local=0;local<active;local++) {
            const auto& frame=frames[steps_completed+local];
            if(frame.steps<=last_recorded)continue;
            trace.points.push_back(decode_frame(frame,final_waypoint));
            last_recorded=frame.steps;
        }
        steps_completed=after_steps;
        if(run[0].collisions||run[0].timeouts)break;
        if(run[0].successes) {
            trace.waypoint_count++;
            if(final_waypoint)break;
            // Keep the same physics and RAPTOR state, but clear the previous
            // waypoint's terminal counters before the next route segment.
            run[0].successes=0;run[0].episodes=0;run[0].success_time=0;run[0].final_progress=0;
            waypoint++;
        } else break;
    }
    trace.collision=run[0].collisions!=0;
    trace.timeout=run[0].timeouts!=0;
    trace.final_success=run[0].successes!=0&&waypoint+1==level.witness_route.size();
    trace.elapsed_s=run[0].elapsed;trace.path_m=run[0].path;
    return trace;
}

inline std::array<float,3> as_array(const float* value) {
    return {value[0],value[1],value[2]};
}

inline float discounted_constant_cost(float gamma,uint32_t steps) {
    double total=0,discount=1;
    for(uint32_t i=0;i<steps;i++){total+=discount;discount*=gamma;}
    return float(-0.01*total);
}

struct StepAccounting {
    float global_progress_m=0;
    float base_cost=-0.01f;
    float risk_fraction=0;
    bool collision=false;
    bool final_success=false;
    bool timeout=false;
};

inline std::vector<StepAccounting> compare_risk_traces(const challenge_evaluation::Level& level,
                                                        const Trace& zero_risk,const Trace& risk,
                                                        float delta_risk=0.1f) {
    require(zero_risk.mode==risk.mode&&zero_risk.points.size()==risk.points.size()&&!zero_risk.points.empty(),
            "risk-coefficient runs do not have the same active trajectory length");
    const std::array<float,3> global_goal{{level.world.goal[0],level.world.goal[1],level.world.goal[2]}};
    const auto start=std::array<float,3>{{0,0,1.5f}};
    std::array<float,3> previous=start;
    std::vector<StepAccounting> result;result.reserve(zero_risk.points.size());
    for(size_t i=0;i<zero_risk.points.size();i++) {
        const Point& a=zero_risk.points[i];const Point& b=risk.points[i];
        require(a.steps==b.steps&&a.successes==b.successes&&a.collisions==b.collisions&&a.timeouts==b.timeouts,
                "risk coefficient changed deterministic episode events");
        for(uint32_t axis=0;axis<3;axis++) {
            require(std::fabs(a.position[axis]-b.position[axis])<2e-5f,
                    "risk coefficient changed deterministic trajectory");
            require(std::fabs(a.local_goal[axis]-b.local_goal[axis])<2e-5f,
                    "risk coefficient changed waypoint schedule");
        }
        const float progress_local=distance(previous,a.local_goal)-distance(a.position,a.local_goal);
        const float local_success=a.successes?10.0f:0.0f;
        const float collision=a.collisions?1.0f:0.0f;
        const float common=2.0f*progress_local+local_success-10.0f*collision;
        const float no_risk=a.raw_reward-common;
        const float with_risk=b.raw_reward-common;
        require(std::fabs(no_risk+0.01f)<4e-4f,
                "base PPO reward changed; expected time plus isolated risk/contact terms");
        const float risk_fraction=(no_risk-with_risk)/delta_risk;
        require(risk_fraction>=-2e-3f&&risk_fraction<=1.002f&&std::isfinite(risk_fraction),
                "risk term could not be recovered from paired reward runs");
        StepAccounting step;step.global_progress_m=distance(previous,global_goal)-distance(a.position,global_goal);
        step.base_cost=no_risk;step.risk_fraction=std::max(0.0f,std::min(1.0f,risk_fraction));
        step.collision=a.collisions!=0;
        step.timeout=a.timeouts!=0;
        step.final_success=!step.collision&&distance(a.position,global_goal)<=0.35f;
        result.push_back(step);previous=a.position;
        // Score the global mission only through first arrival, collision or
        // timeout. Mode20 may continue to the stricter local waypoint radius.
        if(step.final_success||step.collision||step.timeout)break;
    }
    require(zero_risk.collision==risk.collision&&zero_risk.timeout==risk.timeout&&
            zero_risk.final_success==risk.final_success,"risk coefficient changed the terminal outcome");
    return result;
}

inline void run(Metal& metal,const std::string& checkpoint,const std::string& bank_path,
                const std::string& output_path,uint32_t per_family=6,
                uint32_t global_steps=400,uint32_t witness_steps=400) {
    require(fixed_ppo::actor_obs_dim==184,"reward audit requires the guided 184-input actor");
    require(per_family>0&&per_family<=30&&global_steps>0&&global_steps<=1200&&
            witness_steps>=global_steps&&witness_steps<=1600,"reward audit subset/step budget is invalid");
    metal.compile(base_source()+kCaptureMSL);
    const id<MTLComputePipelineState> capture_pipeline=metal.pipeline("reward_audit_capture");
    const ActorPolicy policy=load_actor(checkpoint);
    std::string bank_hash;
    const std::string world_hash=challenge_evaluation::sha256_file(std::string(SOURCE_DIR)+"/world.hpp");
    const auto all_levels=challenge_evaluation::load_split(bank_path,"train",world_hash,bank_hash);
    std::array<uint32_t,kFamilyCount> family_count{};
    std::vector<challenge_evaluation::Level> levels;levels.reserve(per_family*kFamilyCount);
    for(const auto& level:all_levels) {
        for(uint32_t i=0;i<kFamilyCount;i++)if(level.family==kFamilies[i]&&family_count[i]<per_family) {
            levels.push_back(level);family_count[i]++;break;
        }
    }
    for(uint32_t i=0;i<kFamilyCount;i++)require(family_count[i]==per_family,
            "training split has too few levels for family "+std::to_string(kFamilies[i]));

    const std::filesystem::path output(output_path);
    if(output.has_parent_path())std::filesystem::create_directories(output.parent_path());
    std::ofstream csv(output_path);require(bool(csv),"reward audit cannot write output CSV");
    csv<<"failure_id,family,controller,mode,risk_coef,gamma,global_mission_steps,sim_collision,sim_timeout,global_success,"
          "start_clearance_m,global_mission_path_m,global_mission_elapsed_s,sim_path_m,sim_elapsed_s,"
          "global_progress_m,progress_return,progress_return_discounted,"
          "time_loss,time_loss_discounted,contact_loss,contact_loss_discounted,final_goal_bonus,"
          "final_goal_bonus_discounted,risk_loss_undiscounted,risk_loss_discounted,undiscounted_global_return,"
          "discounted_global_return,analytical_hover_available,analytical_hover_undiscounted,"
          "analytical_hover_discounted,bank_sha256,actor_sha256\n";
    const std::string actor_hash=challenge_evaluation::sha256_file(checkpoint);
    uint64_t total_steps=0;double risk_error=0;uint64_t risk_samples=0;
    for(const auto& level:levels) {
        const struct Controller { uint32_t mode; const char* label; bool witness; } controllers[] = {
            {17,"global_goal_mode17",false},
            {20,"goal_script_witness_mode20",true},
            {21,"actor_witness_mode21",true},
        };
        for(const auto& controller:controllers) {
            const bool witness_controller=controller.witness;
            const uint32_t max_steps=witness_controller?witness_steps:global_steps;
            std::array<Trace,2> traces;
            for(uint32_t risk_index=0;risk_index<2;risk_index++) {
                traces[risk_index]=witness_controller?
                    run_witness_policy(metal,level,policy,controller.mode,max_steps,kRiskCoefficients[risk_index],capture_pipeline):
                    run_global_policy(metal,level,policy,max_steps,kRiskCoefficients[risk_index],capture_pipeline);
            }
            const auto accounting=compare_risk_traces(level,traces[0],traces[1]);
            total_steps+=accounting.size();
            for(const auto& item:accounting){risk_error+=std::fabs(item.base_cost+0.01f);risk_samples++;}
            const bool static_world=navigation_task_static_world(level.world);
            const bool hover_available=static_world&&traces[0].start_clearance_m>0.6f;
            for(uint32_t risk_index=0;risk_index<2;risk_index++)for(float gamma:kGammas) {
                const float risk_coef=kRiskCoefficients[risk_index];
                require(!accounting.empty()&&traces[risk_index].points.size()>=accounting.size(),
                        "reward audit has no scored global mission frames");
                double progress_return=0,time_loss=0,contact_loss=0,goal_bonus=0;
                double undiscounted=0,discounted=0,risk_loss=0,risk_loss_discounted=0;
                double progress_discounted=0,time_loss_discounted=0,contact_loss_discounted=0,goal_bonus_discounted=0;
                double discount=1,progress=0;
                for(const auto& step:accounting) {
                    const double progress_reward=2.0*step.global_progress_m;
                    const double contact=step.collision?-10.0:0.0;
                    const double bonus=step.final_success?10.0:0.0;
                    const double risk=risk_coef*step.risk_fraction;
                    const double global_reward=progress_reward+step.base_cost-risk+contact+bonus;
                    progress_return+=progress_reward;time_loss+=step.base_cost;contact_loss+=contact;goal_bonus+=bonus;
                    undiscounted+=global_reward;discounted+=discount*global_reward;
                    risk_loss+=risk;risk_loss_discounted+=discount*risk;
                    progress_discounted+=discount*progress_reward;time_loss_discounted+=discount*step.base_cost;
                    contact_loss_discounted+=discount*contact;goal_bonus_discounted+=discount*bonus;
                    progress+=step.global_progress_m;discount*=gamma;
                }
                const float hover_undiscounted=hover_available?float(-0.01*accounting.size()):0.0f;
                const float hover_discounted=hover_available?discounted_constant_cost(gamma,uint32_t(accounting.size())):0.0f;
                double mission_path=0;
                std::array<float,3> previous_position{{0,0,1.5f}};
                for(size_t i=0;i<accounting.size();i++) {
                    mission_path+=distance(previous_position,traces[risk_index].points[i].position);
                    previous_position=traces[risk_index].points[i].position;
                }
                const bool global_success=accounting.back().final_success;
                const float mission_elapsed=traces[risk_index].points[accounting.size()-1].elapsed_s;
                csv<<challenge_evaluation::csv_cell(level.failure_id)<<','<<level.family<<','
                   <<controller.label<<','<<controller.mode<<','<<challenge_evaluation::number_cell(risk_coef)<<','
                   <<challenge_evaluation::number_cell(gamma)<<','<<accounting.size()<<','
                   <<traces[risk_index].collision<<','<<traces[risk_index].timeout<<','
                   <<global_success<<','
                   <<challenge_evaluation::number_cell(traces[0].start_clearance_m)<<','
                   <<challenge_evaluation::number_cell(mission_path)<<','
                   <<challenge_evaluation::number_cell(mission_elapsed)<<','
                   <<challenge_evaluation::number_cell(traces[risk_index].path_m)<<','
                   <<challenge_evaluation::number_cell(traces[risk_index].elapsed_s)<<','
                   <<challenge_evaluation::number_cell(progress)<<','
                   <<challenge_evaluation::number_cell(progress_return)<<','
                   <<challenge_evaluation::number_cell(progress_discounted)<<','
                   <<challenge_evaluation::number_cell(time_loss)<<','
                   <<challenge_evaluation::number_cell(time_loss_discounted)<<','
                   <<challenge_evaluation::number_cell(contact_loss)<<','
                   <<challenge_evaluation::number_cell(contact_loss_discounted)<<','
                   <<challenge_evaluation::number_cell(goal_bonus)<<','
                   <<challenge_evaluation::number_cell(goal_bonus_discounted)<<','
                   <<challenge_evaluation::number_cell(risk_loss)<<','
                   <<challenge_evaluation::number_cell(risk_loss_discounted)<<','
                   <<challenge_evaluation::number_cell(undiscounted)<<','
                   <<challenge_evaluation::number_cell(discounted)<<','
                   <<(hover_available?"true":"false")<<','
                   <<challenge_evaluation::number_cell(hover_undiscounted)<<','
                   <<challenge_evaluation::number_cell(hover_discounted)<<','<<bank_hash<<','<<actor_hash<<'\n';
            }
        }
        csv.flush();require(bool(csv),"reward audit CSV write failed");
    }
    std::cout<<"reward_audit train_levels="<<levels.size()<<" families=14,15,16 risk=0,.1 gamma=.99,.995,.997"
             <<" transitions="<<total_steps<<" mean_time_cost_residual="<<(risk_samples?risk_error/risk_samples:0)
             <<" actor_sha256="<<actor_hash<<" bank_sha256="<<bank_hash<<" output="<<output_path<<"\n";
}

} // namespace reward_audit
