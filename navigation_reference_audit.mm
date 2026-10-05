// Measure the actual controller reference hidden behind the actor's clipped input.
#define WAYPOINT_EMBEDDED
#include "navigation_critic_training.mm"

int main(int argc,char** argv) {@autoreleasepool {try {
    require(argc==4,"reference_audit CHECKPOINT BANK OUT_CSV");
    waypoint::BankControl control{};std::string hash;const auto bank=waypoint::read_bank(argv[2],control,hash);
    require(control.period==1,"reference audit requires one evaluation task per environment");
    Metal metal;metal.compile(base_source()+PPO_TRAINER_MSL+waypoint::kWaypointKernels);
    SimConfig cfg;cfg.n=control.count;cfg.eval=1;cfg.mode=17;cfg.seed=700001;cfg.speed=1.5f;cfg.max_steps=400;cfg.geometry_memory=1;
    Sim sim(metal,cfg,400);navigation_training::load_actor(sim,argv[1],true);
    std::memset(sim.obs.contents,0,sim.obs.length);std::memset(sim.co.contents,0,sim.co.length);
    auto run=waypoint::make_local_run(sim,bank,control,metal.pipeline("waypoint_task_apply"));
    auto commands=[metal.queue commandBuffer];waypoint::local_collect(run,commands,400);metal.finish(commands);
    const auto* episodes=static_cast<const SimRun*>(sim.runs.contents);
    const auto* obs=static_cast<const float*>(sim.obs.contents);const auto* critic=static_cast<const float*>(sim.co.contents);
    std::ofstream out(argv[3]);require(bool(out),"cannot write reference audit");
    out<<"env,success,collision,timeout,step,goal_distance_m,true_ref_norm_m,actor_ref_max_abs,true_body_ref_max_m\n";
    for(uint env=0;env<cfg.n;env++) {
        const uint steps=std::min(episodes[env].steps,400u);
        for(uint step=steps>100?steps-100:0;step<steps;step++) {
            const size_t row=size_t(step)*cfg.n+env;
            const float* c=critic+row*fixed_ppo::critic_obs_dim;const float* a=obs+row*fixed_ppo::actor_obs_dim;
            float rotation[9];const float w=c[9],x=c[10],y=c[11],z=c[12];
            rotation[0]=1-2*(y*y+z*z);rotation[1]=2*(x*y-w*z);rotation[2]=2*(x*z+w*y);
            rotation[3]=2*(x*y+w*z);rotation[4]=1-2*(x*x+z*z);rotation[5]=2*(y*z-w*x);
            rotation[6]=2*(x*z-w*y);rotation[7]=2*(y*z+w*x);rotation[8]=1-2*(x*x+y*y);
            float norm=0,largest=0,actor_largest=0;
            for(uint axis=0;axis<3;axis++) {
                norm+=c[16+axis]*c[16+axis]*.25f;
                const float body=(rotation[axis]*c[16]+rotation[3+axis]*c[17]+rotation[6+axis]*c[18])*.5f;
                largest=std::max(largest,std::fabs(body));
                actor_largest=std::max(actor_largest,std::fabs(a[fixed_ppo::context_offset+18+axis]));
                require(std::fabs(std::clamp(body*2,-1.0f,1.0f)-a[fixed_ppo::context_offset+18+axis])<2e-5f,"actor/critic reference reconstruction differs");
            }
            out<<env<<','<<episodes[env].successes<<','<<episodes[env].collisions<<','<<episodes[env].timeouts
               <<','<<step<<','<<a[fixed_ppo::context_offset+3]*10<<','<<std::sqrt(norm)<<','<<actor_largest<<','<<largest<<'\n';
        }
    }
    std::cout<<"PASS exact reference reconstruction and400tick flight capture\n";return 0;
}catch(const std::exception& e){std::cerr<<"ERROR: "<<e.what()<<'\n';return 1;}}}
