#define NAVIGATION_IMITATION_EMBEDDED
#include "navigation_imitation.mm"

// Diagnose the unchanged source objective with real RAPTOR flights. The route
// teacher is privileged; its return is an objective check, not policy evidence.
int main(int argc,char** argv) { @autoreleasepool { try {
    require(argc==4||argc==6,"reward_audit TRAIN_BANK CHECKPOINT OUTPUT_DIRECTORY [BEGIN COUNT]");
    const size_t begin=argc==6?std::stoul(argv[4]):0;
    const size_t count=argc==6?std::stoul(argv[5]):30;
    require(count>0&&begin+count<=30,"audit range must be within the 30 TRAIN corner cases");
    const std::string output=argv[3];
    std::filesystem::create_directories(output);
    std::string bank_hash;
    const auto levels=challenge_evaluation::load_split(argv[1],"train",
        challenge_evaluation::sha256_file(std::string(SOURCE_DIR)+"/world.hpp"),bank_hash);
    require(bank_hash=="e2170fbe8e8c2d074ffb08de4269175a6de0bdcd410c3b60bd699ee7b5fc491e","source bank mismatch");
    Metal metal; metal.compile(base_source()+PPO_TRAINER_MSL);
    std::ofstream trace(output+"/transitions.csv"), summary(output+"/episodes.csv");
    require(bool(trace)&&bool(summary),"cannot write audit output");
    trace.precision(10);summary.precision(10);
    trace<<"failure_id,control,tick,time_s,x,y,z,distance_before_m,distance_after_m,reward,progress_reward,time_cost,risk_cost,terminal_reward,value,next_value,terminated,truncated\n";
    summary<<"failure_id,control,success,collision,timeout,time_s,path_m,min_clearance_m,return_gamma99,progress_return,time_return,risk_return,terminal_return,undiscounted_return,negative_progress_ticks,next_value_at_end,bootstrap_return,initial_value,bank_sha256\n";
    size_t cases=0;
    for(const auto& level:levels) {
        if(level.family!=14)continue;
        const size_t case_index=cases++;
        if(case_index<begin||case_index>=begin+count)continue;
        for(const std::string control:{"actor","route_teacher","scan_then_teacher","hover"}) {
            SimConfig config;config.n=1;config.eval=1;config.mode=17;
            config.family=14;config.seed=level.seed;config.speed=1.5f;
            config.max_steps=400;config.geometry_memory=1;config.risk_coef=.1f;
            Sim sim(metal,config,1);navigation_training::load_actor(sim,argv[2],true);
            *static_cast<WWorld*>(sim.worlds.contents)=level.world;
            auto& run=*static_cast<SimRun*>(sim.runs.contents);
            auto& state=*static_cast<RLPhysicsState*>(sim.states.contents);
            const navigation_imitation::Point goal={level.world.goal[0],level.world.goal[1],level.world.goal[2]};
            const auto distance=[&](){return navigation_imitation::length(navigation_imitation::difference(goal,{state.position[0],state.position[1],state.position[2]}));};
            run.initial_distance=distance();
            std::vector<float> rewards,values,next_values;
            std::vector<uint8_t> terminated,truncated;
            double discount=1,discounted=0,progress_sum=0,time_sum=0,risk_sum=0,terminal_sum=0,total=0;
            uint negative_progress_ticks=0;
            while(!run.episodes&&run.steps<400) {
                const float before=distance();float value=0;
                if(control=="actor") {
                    auto command=[metal.queue commandBuffer];sim.collect(command,1);metal.finish(command);
                    value=*static_cast<float*>(sim.values.contents);
                } else {
                    navigation_imitation::observe_student(sim);
                    float hidden[64];
                    value=fixed_ppo::critic_forward(static_cast<float*>(sim.co.contents),
                        *static_cast<fixed_ppo::CriticParams*>(sim.critic.contents),hidden);
                    navigation_imitation::Point velocity{};float yaw=0;
                    if(control=="scan_then_teacher"&&run.steps<40) {velocity={0,.15f,0};yaw=.9f;}
                    else if(control!="hover") {
                        const navigation_imitation::Point position={state.position[0],state.position[1],state.position[2]};
                        const auto target=navigation_imitation::route_target(level.world,level.witness_route,position,run.elapsed);
                        velocity=navigation_imitation::difference(target,position);
                        const float speed=navigation_imitation::length(velocity);
                        const float scale=std::min(1.f,1.49f/std::max(speed,1e-8f));
                        for(float& component:velocity)component*=scale;
                    }
                    navigation_imitation::advance_teacher(sim,velocity,yaw);
                }
                const float after=distance(), reward=*static_cast<float*>(sim.rewards.contents);
                const auto term=*static_cast<uint8_t*>(sim.terminated.contents);
                const auto trunc=*static_cast<uint8_t*>(sim.truncated.contents);
                const float next_value=*static_cast<float*>(sim.next_values.contents);
                const double progress=2.0*(double(before)-after),terminal=run.successes?10.0:(run.collisions?-10.0:0.0);
                const double risk=progress-.01+terminal-reward;
                require(std::isfinite(reward)&&risk>=-1e-5&&risk<=.10001,"reward decomposition mismatch");
                negative_progress_ticks+=progress<0;
                discounted+=discount*reward;progress_sum+=discount*progress;
                time_sum-=discount*.01;risk_sum-=discount*risk;terminal_sum+=discount*terminal;total+=reward;
                trace<<level.failure_id<<','<<control<<','<<run.steps<<','<<run.elapsed<<','
                     <<state.position[0]<<','<<state.position[1]<<','<<state.position[2]<<','
                     <<before<<','<<after<<','<<reward<<','<<progress<<",-0.01,"<<-risk<<','
                     <<terminal<<','<<value<<','<<next_value<<','<<uint(term)<<','<<uint(trunc)<<'\n';
                rewards.push_back(reward);values.push_back(value);next_values.push_back(next_value);
                terminated.push_back(term);truncated.push_back(trunc);discount*=.99;
            }
            const double bootstrap=terminated.back()?0:discount*next_values.back();
            summary<<level.failure_id<<','<<control<<','<<run.successes<<','<<run.collisions<<','<<run.timeouts<<','
                   <<run.elapsed<<','<<run.path<<','<<run.min_clearance<<','<<discounted<<','<<progress_sum<<','
                   <<time_sum<<','<<risk_sum<<','<<terminal_sum<<','<<total<<','<<negative_progress_ticks<<','
                   <<next_values.back()<<','<<bootstrap<<','<<values.front()<<','<<bank_hash<<'\n';summary.flush();trace.flush();
            std::cout<<level.failure_id<<' '<<control<<" success="<<run.successes<<" contact="<<run.collisions
                     <<" return="<<discounted<<" terminal_return="<<terminal_sum<<std::endl;
        }
    }
    return 0;
} catch(const std::exception& error) {std::cerr<<"ERROR: "<<error.what()<<'\n';return 1;} } }
