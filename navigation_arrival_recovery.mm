// Open-room recovery diagnostic, not a deployable policy or a capability claim.
#define WAYPOINT_EMBEDDED
#include "navigation_critic_training.mm"

static const char* recovery_kernel=R"MSL(
kernel void arrival_recovery(device const float* obs [[buffer(0)]],
                              device float* means [[buffer(1)]],
                              device const SimRun* runs [[buffer(2)]],
                              constant SimConfig& cfg [[buffer(3)]],
                              uint env [[thread_position_in_grid]]) {
    if(env>=cfg.n||runs[env].episodes||runs[env].steps<300)return;
    const uint row=cfg.tick*cfg.n+env,base=row*PPO_ACTOR_OBS,ctx=base+SIM_CONTEXT_OFFSET;
    const float distance=obs[ctx+3]*10;
    if(distance>2)return;
    // Move the adapter's reference toward the supplied goal with a .5s time
    // constant. Both quantities are already present in deployable actor inputs.
    float3 goal=float3(obs[ctx],obs[ctx+1],obs[ctx+2])*distance;
    float3 reference=float3(obs[ctx+18],obs[ctx+19],obs[ctx+20])*.5f;
    float3 velocity=(goal-reference)/.5f;
    velocity*=min(1.0f,1.5f/max(length(velocity),1e-8f));
    float open=0;for(uint i=0;i<20;i++)open+=obs[base+i]*12>2.5f;
    const float gate=.25f+.75f*open/20;
    for(uint axis=0;axis<3;axis++) {
        const float prior=obs[base+PPO_ACTOR_OBS-3+axis];
        const float target=nav_atanh(clamp(velocity[axis]/1.5f,-.98f,.98f));
        means[row*4+axis]=prior+(target-prior)/gate;
    }
    means[row*4+3]=0;
}
)MSL";

int main(int argc,char** argv) {@autoreleasepool {try {
    require(argc==5,"arrival_recovery CHECKPOINT OPEN_BANK OUT_CSV ENABLE_0_OR_1");
    const std::string checkpoint=argv[1],path=argv[2],output=argv[3],enable=argv[4];
    require(enable=="0"||enable=="1","recovery enable must be0/1");
    waypoint::BankControl control{};std::string hash;const auto bank=waypoint::read_bank(path,control,hash);
    for(const auto& row:bank)require(row.world.count==0,"arrival diagnostic is restricted to obstacle-free rooms");
    Metal metal;metal.compile(base_source()+PPO_TRAINER_MSL+waypoint::kWaypointKernels+recovery_kernel);
    std::ifstream file(checkpoint,std::ios::binary);const auto header=read_checkpoint_header(file);
    require(header.actor_count==fixed_ppo::actor_param_count,"arrival diagnostic actor dimensions");
    std::vector<float> actor(header.actor_count);file.read(reinterpret_cast<char*>(actor.data()),actor.size()*4);
    require(bool(file),"arrival diagnostic actor read failed");
    waypoint::evaluate_bank(metal,actor.data(),nullptr,bank,control,17,700001,1.5f,output,
                           "arrival-recovery",control.count/control.period,400,0,0,10,10,nullptr,
                           enable=="1"?metal.pipeline("arrival_recovery"):nullptr);
    return 0;
}catch(const std::exception& e){std::cerr<<"ERROR: "<<e.what()<<'\n';return 1;}}}
