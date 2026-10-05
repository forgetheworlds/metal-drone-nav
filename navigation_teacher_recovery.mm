// Frozen BC recovery from learner states. Diagnostic only: no deployment switch.
#define WAYPOINT_EMBEDDED
#include "navigation_critic_training.mm"

static const char* recovery_kernel=R"MSL(
kernel void teacher_recovery(device const float* observations [[buffer(0)]],
                              device float* means [[buffer(1)]],
                              device const SimRun* runs [[buffer(2)]],
                              constant SimConfig& cfg [[buffer(3)]],
                              device const float* teacher [[buffer(4)]],
                              uint env [[thread_position_in_grid]]) {
    if(env>=cfg.n||runs[env].episodes||runs[env].steps<300)return;
    const uint row=cfg.tick*cfg.n+env,base=row*PPO_ACTOR_OBS;
    if(observations[base+SIM_CONTEXT_OFFSET+3]*10>2)return;
    float obs[PPO_ACTOR_OBS],hidden[64],mean[4];
    for(uint i=0;i<PPO_ACTOR_OBS;i++)obs[i]=observations[base+i];
    ppo_actor_mean(teacher,obs,hidden,mean);
    for(uint i=0;i<4;i++)means[row*4+i]=mean[i];
}
)MSL";

static std::vector<float> actor(const std::string& path) {
    std::ifstream file(path,std::ios::binary);const auto header=read_checkpoint_header(file);
    require(header.actor_count==fixed_ppo::actor_param_count,"teacher recovery actor dimensions");
    std::vector<float> weights(header.actor_count);file.read(reinterpret_cast<char*>(weights.data()),weights.size()*4);
    require(bool(file),"teacher recovery actor read failed");return weights;
}

int main(int argc,char** argv) {@autoreleasepool {try {
    require(argc==5,"teacher_recovery LEARNER TEACHER OPEN_BANK OUT_CSV");
    waypoint::BankControl control{};std::string hash;const auto bank=waypoint::read_bank(argv[3],control,hash);
    for(const auto& entry:bank)require(entry.world.count==0,"teacher recovery is restricted to empty rooms");
    auto student=actor(argv[1]),teacher=actor(argv[2]);
    Metal metal;metal.compile(base_source()+PPO_TRAINER_MSL+waypoint::kWaypointKernels+recovery_kernel);
    auto reference=metal.buffer(teacher.size()*4,teacher.data());
    waypoint::evaluate_bank(metal,student.data(),nullptr,bank,control,17,700001,1.5f,argv[4],
                           "teacher-recovery",control.count/control.period,400,0,0,10,10,nullptr,
                           metal.pipeline("teacher_recovery"),reference);
    return 0;
}catch(const std::exception& e){std::cerr<<"ERROR: "<<e.what()<<'\n';return 1;}}}
