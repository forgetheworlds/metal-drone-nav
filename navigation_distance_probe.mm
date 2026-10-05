// Inference-only distance representation experiment. The physical goal is untouched.
#define WAYPOINT_EMBEDDED
#include "navigation_critic_training.mm"

static const char* distance_cap_kernel=R"METAL(
kernel void navigation_distance_cap(device float* observations [[buffer(0)]],
                                    constant SimConfig& cfg [[buffer(1)]],
                                    uint env [[thread_position_in_grid]]) {
    if(env>=cfg.n)return;
    const uint row=(cfg.tick*cfg.n+env)*PPO_ACTOR_OBS;
    observations[row+SIM_CONTEXT_OFFSET+3]=min(observations[row+SIM_CONTEXT_OFFSET+3],0.3f);
}
)METAL";

int main(int argc,char** argv) {@autoreleasepool {try {
    require(argc==5||(argc==2&&std::string(argv[1])=="--test"),
            "navigation_distance_probe CHECKPOINT BANK CSV CAP_0_OR_1 | --test");
    Metal metal;
    metal.compile(base_source()+PPO_TRAINER_MSL+waypoint::kWaypointKernels+distance_cap_kernel);
    if(argc==2) {
        SimConfig config;config.n=4;config.tick=2;
        std::vector<float> original(4*config.n*fixed_ppo::actor_obs_dim,0.17f),expected;
        for(uint32_t env=0;env<config.n;env++) {
            const size_t index=(config.tick*config.n+env)*fixed_ppo::actor_obs_dim+163;
            original[index]=0.1f+0.4f*env;
        }
        expected=original;
        for(uint32_t env=0;env<config.n;env++) {
            const size_t index=(config.tick*config.n+env)*fixed_ppo::actor_obs_dim+163;
            expected[index]=std::min(expected[index],0.3f);
        }
        auto observations=metal.buffer(original.size()*sizeof(float),original.data());
        auto settings=metal.buffer(sizeof(config),&config);
        auto commands=[metal.queue commandBuffer];
        metal.dispatch(commands,metal.pipeline("navigation_distance_cap"),config.n,{observations,settings},64);
        metal.finish(commands);
        require(std::memcmp(observations.contents,expected.data(),expected.size()*sizeof(float))==0,
                "distance transform changed another feature, tick or environment");
        std::cout<<"PASS: exact distance-only transform; other features and ticks byte-identical\n";
        return 0;
    }
    const std::string checkpoint=argv[1],bank_path=argv[2],output=argv[3];
    const std::string cap_argument=argv[4];
    require(cap_argument=="0"||cap_argument=="1","cap must be 0 or 1");
    waypoint::BankControl control{};std::string bank_hash;
    const auto bank=waypoint::read_bank(bank_path,control,bank_hash);
    std::ifstream file(checkpoint,std::ios::binary);
    const auto header=read_checkpoint_header(file);
    require(header.actor_count==fixed_ppo::actor_param_count,"distance probe actor dimensions");
    std::vector<float> actor(header.actor_count);
    file.read(reinterpret_cast<char*>(actor.data()),actor.size()*sizeof(float));
    require(bool(file),"distance probe actor read failed");
    std::cout<<"distance_cap="<<cap_argument<<" normalized_cap=0.3 full_goal_preserved=1\n";
    waypoint::evaluate_bank(metal,actor.data(),nullptr,bank,control,17,700001,1.5f,output,
                           "distance-probe",control.count/control.period,400,0,0,10,10,
                           cap_argument=="1"?metal.pipeline("navigation_distance_cap"):nullptr);
    return 0;
}catch(const std::exception& error){std::cerr<<"ERROR: "<<error.what()<<"\n";return 1;}}}
