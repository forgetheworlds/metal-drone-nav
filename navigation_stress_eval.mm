// Frozen-policy stress probes on exact bank geometry. No training or retiming.
#define WAYPOINT_EMBEDDED
#include "navigation_critic_training.mm"

int main(int argc,char** argv) {@autoreleasepool {try {
    require(argc==5,"stress_eval CHECKPOINT BANK OUT_CSV PROFILE");
    const std::string checkpoint=argv[1],bank_path=argv[2],output=argv[3],profile=argv[4];
    waypoint::BankControl control{};std::string bank_hash;const auto bank=waypoint::read_bank(bank_path,control,bank_hash);
    require(control.period==1,"stress evaluation requires one task per environment");
    SimConfig cfg;cfg.n=control.count;cfg.eval=1;cfg.mode=17;cfg.seed=700001;cfg.speed=1.5f;cfg.max_steps=400;cfg.geometry_memory=1;
    float amplitude=0;
    if(profile=="nominal"){}
    else if(profile=="depth-noise")cfg.depth_noise=.03f;
    else if(profile=="dropout")cfg.dropout=.05f;
    else if(profile=="sensor-delay")cfg.sensor_delay=2;
    else if(profile=="command-delay")cfg.command_delay=2;
    else if(profile=="both-delay"){cfg.sensor_delay=2;cfg.command_delay=2;}
    else if(profile=="dynamics")amplitude=1;
    else if(profile=="combined"){
        cfg.depth_noise=.03f;cfg.dropout=.05f;cfg.sensor_delay=2;cfg.command_delay=2;amplitude=1;
    }else throw std::runtime_error("unknown stress profile");
    Metal metal;metal.compile(base_source()+PPO_TRAINER_MSL+waypoint::kWaypointKernels);
    Sim sim(metal,cfg,32);
    if(amplitude>0) {
        NavigationRuntimeConfig runtime{};runtime.enabled=1;runtime.domain_amplitude=amplitude;
        runtime.domain_seed=0x3f84d5b5u;runtime.domain_range=rl_physics_domain_stress_range();
        std::memcpy(sim.runtime_control.contents,&runtime,sizeof(runtime));sim.reset();
    }
    navigation_training::load_actor(sim,checkpoint,false);
    auto run=waypoint::make_local_run(sim,bank,control,metal.pipeline("waypoint_task_apply"));
    auto probe=[metal.queue commandBuffer];waypoint::local_probe(run,probe);metal.finish(probe);
    waypoint::verify_reset(run,bank,control,cfg.sensor_delay==0&&cfg.depth_noise==0&&cfg.dropout==0);
    const auto* params=static_cast<const RLPhysicsParams*>(sim.environment_physics.contents);
    std::ofstream raw_physics(output+".physics.bin",std::ios::binary);
    require(bool(raw_physics),"cannot write exact sampled plant parameters");
    raw_physics.write(reinterpret_cast<const char*>(params),sim.environment_physics.length);
    require(bool(raw_physics),"sampled plant write failed");
    std::ofstream dynamics(output+".physics.csv");require(bool(dynamics),"cannot write sampled plant parameters");
    dynamics<<"env,mass,inertia_x,inertia_y,inertia_z\n";
    for(uint env=0;env<cfg.n;env++) {
        require(std::isfinite(params[env].mass)&&params[env].mass>0,"invalid sampled mass");
        dynamics<<env<<','<<params[env].mass<<','<<params[env].inertia[0]<<','<<params[env].inertia[4]<<','<<params[env].inertia[8]<<'\n';
    }
    auto commands=[metal.queue commandBuffer];waypoint::local_collect(run,commands,400);metal.finish(commands);
    const auto* results=static_cast<const SimRun*>(sim.runs.contents);
    uint success=0,collision=0,timeout=0;
    for(uint env=0;env<cfg.n;env++){require(results[env].episodes==1,"stress eval must finish exactly one episode");success+=results[env].successes;collision+=results[env].collisions;timeout+=results[env].timeouts;}
    waypoint::write_eval_csv(output,profile,nullptr,bank,control,sim,700001,17);
    std::cout<<"stress="<<profile<<" tasks="<<cfg.n<<" success="<<success<<" contact="<<collision<<" timeout="<<timeout
             <<" noise_m="<<cfg.depth_noise<<" dropout="<<cfg.dropout<<" sensor_delay_ms="<<cfg.sensor_delay*50
             <<" command_delay_ms="<<cfg.command_delay*50<<" domain_amplitude="<<amplitude
             <<" ego_state=ideal wind=none bank_sha256="<<bank_hash<<'\n';
    return 0;
}catch(const std::exception& e){std::cerr<<"ERROR: "<<e.what()<<'\n';return 1;}}}
