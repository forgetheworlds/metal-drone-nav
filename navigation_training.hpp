#pragma once

// Goal-arrival experiments use the existing actor and PPO trainer. The new
// task contract is explicit and does not replace historical benchmarks.
namespace navigation_training {

inline NavigationTaskControl task_settings(uint32_t stage,uint32_t family) {
    NavigationTaskControl control{};control.enabled=1;
    require(navigation_task_default_config(control.config,stage,family,.5f),"unsupported task stage/family");
    control.config.time_cost_per_s=.2f;
    control.config.command_smoothness_weight=.05f;
    control.config.initial_speed_max_mps=stage==1?.5f:1.0f;
    return control;
}

inline void configure(Sim& sim,const NavigationTaskControl& tasks,float amplitude) {
    require(std::isfinite(amplitude)&&amplitude>=0&&amplitude<=1,"domain amplitude must be0..1");
    require(std::fabs(tasks.config.nav_period_s-sim.cfg.substeps*.01f)<1e-6f,"task/simulator clock mismatch");
    require(((ChallengeBankControl*)sim.bank_control.contents)->enabled==0,"task generator and frozen bank cannot both own the world");
    NavigationRuntimeConfig runtime{};runtime.enabled=1;
    runtime.domain_amplitude=amplitude;runtime.domain_seed=0x3f84d5b5u;
    runtime.domain_range=rl_physics_domain_stress_range();
    std::memcpy(sim.runtime_control.contents,&runtime,sizeof(runtime));
    std::memcpy(sim.task_control.contents,&tasks,sizeof(tasks));sim.reset();
}

inline void validate_rollout(const Sim& sim) {
    const auto* tasks=(const NavigationTaskState*)sim.task_states.contents;
    const auto* physics=(const RLPhysicsParams*)sim.environment_physics.contents;
    for(uint32_t env=0;env<sim.cfg.n;env++) {
        require(tasks[env].valid && tasks[env].generation_status==NAV_TASK_GENERATION_READY,
                "task generation rejected env="+std::to_string(env)+" status="+std::to_string(tasks[env].generation_status));
        require(std::isfinite(physics[env].mass)&&physics[env].mass>0,"episode dynamics sample rejected");
    }
}

inline void load_actor(Sim& sim,const std::string& path,bool critic=false) {
    std::ifstream file(path,std::ios::binary);const auto header=read_checkpoint_header(file);
    require(header.actor_count==fixed_ppo::actor_param_count,"task actor dimensions");
    file.read((char*)sim.actor.contents,sim.actor.length);
    if(critic) {
        require(header.critic_count==fixed_ppo::critic_param_count,"task critic dimensions");
        file.read((char*)sim.critic.contents,sim.critic.length);
    }
    require(bool(file),"task parameter load failed");
}

struct Score {
    double success=0,collision=0,timeout=0,successful_time_s=0;
};

inline Score evaluate(Metal& metal,const float* actor,uint32_t stage,uint32_t family,
                      float amplitude,uint32_t seed,const std::string& output="",uint32_t mode=17) {
    SimConfig config;config.n=128;config.eval=1;config.mode=mode;config.seed=seed;
    config.family=family;config.speed=1.5f;config.distance=8;config.max_steps=400;config.geometry_memory=1;
    Sim sim(metal,config,32);const auto settings=task_settings(stage,family);
    configure(sim,settings,amplitude);
    std::memcpy(sim.actor.contents,actor,sim.actor.length);validate_rollout(sim);
    const auto* initial=(const NavigationTaskState*)sim.task_states.contents;
    std::vector<NavigationTaskState> task_snapshot(initial,initial+config.n);
    auto commands=[metal.queue commandBuffer];sim.collect(commands,400);metal.finish(commands);validate_rollout(sim);
    std::ofstream file;
    if(!output.empty()) {
        const auto destination=std::filesystem::path(output);
        if(destination.has_parent_path())std::filesystem::create_directories(destination.parent_path());
        file.open(output);require(bool(file),"cannot write task evaluation");
        file<<"seed,env,stage,family,mode,domain_amplitude,start_x,start_y,start_z,start_yaw,initial_vx,initial_vy,initial_vz,goal_x,goal_y,goal_z,initial_distance_m,direct_clearance_m,witness_length_m,mass_kg,success,collision,timeout,time_s,path_m,goal_error_m,final_speed_mps,hold_s,peak_speed_mps,min_sphere_clearance_m,success_rule\n";
    }
    Score result;
    const auto* runs=(const SimRun*)sim.runs.contents;
    const auto* states=(const RLPhysicsState*)sim.states.contents;
    const auto* final_tasks=(const NavigationTaskState*)sim.task_states.contents;
    const auto* physics=(const RLPhysicsParams*)sim.environment_physics.contents;
    for(uint32_t env=0;env<config.n;env++) {
        const auto& run=runs[env];const auto& task=task_snapshot[env];const auto& state=states[env];
        require(run.episodes==1,"evaluation must complete exactly one task per environment");
        result.success+=run.successes;result.collision+=run.collisions;result.timeout+=run.timeouts;
        result.successful_time_s+=run.success_time;
        const float dx=task.goal_position[0]-state.position[0],dy=task.goal_position[1]-state.position[1],dz=task.goal_position[2]-state.position[2];
        const float error=std::sqrt(dx*dx+dy*dy+dz*dz);
        const float speed=std::sqrt(state.linear_velocity[0]*state.linear_velocity[0]+state.linear_velocity[1]*state.linear_velocity[1]+state.linear_velocity[2]*state.linear_velocity[2]);
        if(run.successes)require(error<=settings.config.goal_radius_m+1e-5f && speed<=settings.config.stable_speed_mps+1e-5f && final_tasks[env].stable_time_s>=.2f-1e-6f,"stable-arrival invariant failed");
        if(file.is_open()) {
            file<<seed<<','<<env<<','<<stage<<','<<task.family<<','<<mode<<','<<amplitude;
            for(float value:task.start_position)file<<','<<value;
            file<<','<<task.start_yaw_rad;
            for(float value:task.start_velocity_world)file<<','<<value;
            for(float value:task.goal_position)file<<','<<value;
            file<<','<<task.initial_distance_m<<','<<task.direct_segment_clearance_m<<','<<task.witness_length_m<<','<<physics[env].mass
                <<','<<run.successes<<','<<run.collisions<<','<<run.timeouts<<','<<run.elapsed<<','<<run.path<<','<<error<<','<<speed
                <<','<<final_tasks[env].stable_time_s<<','<<run.peak_speed<<','<<run.min_clearance<<",radius0.35_speed0.5_hold0.2\n";
        }
    }
    result.success/=config.n;result.collision/=config.n;result.timeout/=config.n;
    if(result.success>0)result.successful_time_s/=result.success*config.n;
    std::cout<<"task_eval stage="<<stage<<" family="<<family<<" seed="<<seed<<" amplitude="<<amplitude<<" mode="<<mode
             <<" success="<<result.success<<" collision="<<result.collision<<" timeout="<<result.timeout<<" mean_arrival_s="<<result.successful_time_s<<"\n";
    return result;
}

inline void train(Metal& metal,uint32_t rollouts,const std::string& checkpoint,
                  const std::string& warmstart,uint32_t stage,uint32_t family,float amplitude,uint32_t seed) {
    require(rollouts>0&&fixed_ppo::actor_obs_dim==184,"task training requires positive rollouts and guided build");
    metal.compile(base_source()+PPO_TRAINER_MSL);
    SimConfig config;config.n=128;config.mode=22;config.family=family;config.seed=seed;
    config.speed=1.5f;config.distance=8;config.max_steps=400;config.geometry_memory=1;
    config.entropy_coef=.001f;config.learning_rate=.0001f;
    Sim sim(metal,config,32);const auto settings=task_settings(stage,family);configure(sim,settings,amplitude);
    PPOTrainer trainer(sim);const bool resume=std::filesystem::exists(checkpoint);
    if(resume)trainer.load_checkpoint(checkpoint,family,32,config.n,seed);
    else {
        load_actor(sim,warmstart,true);
        for(uint32_t axis=0;axis<4;axis++)((float*)sim.actor.contents)[fixed_ppo::actor_log_std_offset+axis]=-1;
    }
    validate_rollout(sim);
    const std::filesystem::path path(checkpoint);
    if(path.has_parent_path())std::filesystem::create_directories(path.parent_path());
    std::ofstream history(checkpoint+".history.csv",std::ios::app);require(bool(history),"task history write");
    if(!resume)history<<"rollout,total_transitions,wall_s,collect_gpu_s,update_gpu_s,policy_loss,value_loss,ratio,success,collision,timeout,mean_arrival_s\n";
    const double start=seconds();const uint32_t finish=trainer.completed_rollouts+rollouts;
    const auto evaluate_current=[&](const std::string& output){return evaluate(metal,(const float*)sim.actor.contents,stage,family,amplitude,800001,output);};
    Score best;
    if(resume && std::filesystem::exists(checkpoint+".best")) {
        Sim selected(metal,config,32);load_actor(selected,checkpoint+".best");
        best=evaluate(metal,(const float*)selected.actor.contents,stage,family,amplitude,800001);
    }
    if(!resume) {
        const auto baseline=evaluate_current(checkpoint+".initial-dev.csv");
        best=baseline;trainer.save_checkpoint(checkpoint+".best",family,seed,trainer.completed_rollouts);
        history<<0<<",0,0,0,0,0,0,1,"<<baseline.success<<','<<baseline.collision<<','<<baseline.timeout<<','<<baseline.successful_time_s<<'\n';history.flush();
    }
    for(uint32_t rollout=trainer.completed_rollouts;rollout<finish;rollout++) {@autoreleasepool {
        // Reject generation/dynamics errors before their transitions affect PPO.
        auto collect=[metal.queue commandBuffer];sim.collect(collect);
        const double collection_gpu=metal.finish(collect);validate_rollout(sim);
        auto update=[metal.queue commandBuffer];trainer.rollout_update(update,rollout);
        const double update_gpu=metal.finish(update);trainer.completed_rollouts=rollout+1;
        if((rollout+1)%50==0 || rollout+1==finish) {
            trainer.save_checkpoint(checkpoint,family,seed,rollout+1);
            const auto score=evaluate_current(checkpoint+".dev.csv");
            if(score.success>best.success || (score.success==best.success && score.collision<best.collision) ||
               (score.success==best.success && score.collision==best.collision && score.successful_time_s<best.successful_time_s)) {
                best=score;trainer.save_checkpoint(checkpoint+".best",family,seed,rollout+1);
                std::filesystem::copy_file(checkpoint+".dev.csv",checkpoint+".best-dev.csv",std::filesystem::copy_options::overwrite_existing);
            }
            const auto* metrics=(const float*)trainer.metric_mean.contents;
            history<<rollout+1<<','<<uint64_t(rollout+1)*config.n*32<<','<<seconds()-start<<','<<collection_gpu<<','<<update_gpu
                   <<','<<metrics[0]<<','<<metrics[1]<<','<<metrics[3]<<','<<score.success<<','<<score.collision<<','<<score.timeout<<','<<score.successful_time_s<<'\n';history.flush();
            std::cout<<"task_train rollout="<<rollout+1<<" policy_loss="<<metrics[0]<<" value_loss="<<metrics[1]<<" ratio="<<metrics[3]<<" wall_s="<<seconds()-start<<"\n";
            sim.report("train-arrival",0);
        }
    }}
    // Checkpoints remain experimental; promotion requires protected old-task
    // retention and independent simulator results outside this training loop.
}

} // namespace navigation_training
