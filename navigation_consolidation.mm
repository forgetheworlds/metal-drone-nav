// Successful source flights from frozen teachers train one deployable student.
// Teacher/group IDs exist only in the offline sampler, never in actor inputs.
#define WAYPOINT_EMBEDDED
#include "navigation_critic_training.mm"
#include <map>

namespace consolidation {
using Point=std::array<float,3>;
constexpr uint observation_count=fixed_ppo::actor_obs_dim;
constexpr uint hint_offset=observation_count-3;
static_assert(observation_count==184,"consolidation uses the preserved guided actor");
static float length(Point p){return std::sqrt(p[0]*p[0]+p[1]*p[1]+p[2]*p[2]);}

struct Demonstration {
    std::array<float,observation_count> observation;
    std::array<float,4> command; // body velocity /1.5 after sphere cap; yaw rate /.5
    uint32_t group,record;
};
static_assert(sizeof(Demonstration)==760,"consolidation row ABI changed");
struct DatasetHeader {
    char magic[8];uint32_t version,observation_count;uint64_t rows;
    uint32_t row_bytes,groups;
};
static_assert(sizeof(DatasetHeader)==32,"consolidation header ABI changed");

// Exact bounded-command gradient and finite-difference fixture from navigation_imitation.mm.
static const char* imitation_kernel=R"MSL(
kernel void imitation_gradient(device const float* obs [[buffer(0)]],device const float* means [[buffer(1)]],
    device const float* target [[buffer(2)]],device float* gradient [[buffer(3)]],device float* losses [[buffer(4)]],
    constant uint& count [[buffer(5)]],uint n [[thread_position_in_grid]]){
    if(n>=count)return;float open=0;for(uint k=0;k<20;k++)open+=obs[n*PPO_ACTOR_OBS+k]*12>2.5f;
    float gate=.25f+.75f*open/20;float3 prior=float3(obs[n*PPO_ACTOR_OBS+PPO_ACTOR_OBS-3],obs[n*PPO_ACTOR_OBS+PPO_ACTOR_OBS-2],obs[n*PPO_ACTOR_OBS+PPO_ACTOR_OBS-1]);
    float3 mean=float3(means[n*4],means[n*4+1],means[n*4+2]);float3 q=tanh(prior+gate*(mean-prior));
    float magnitude=length(q);float3 command=q/max(1.0f,magnitude);
    float3 desired=float3(target[n*4],target[n*4+1],target[n*4+2]);float3 delta=command-desired;
    float3 derivative=delta;
    if(magnitude>1)derivative=(delta-command*dot(command,delta))/magnitude;
    derivative*=gate*(1-q*q)/float(count);
    float yaw=tanh(gate*means[n*4+3]),dyaw=yaw-target[n*4+3];
    gradient[n*4]=derivative.x;gradient[n*4+1]=derivative.y;gradient[n*4+2]=derivative.z;
    gradient[n*4+3]=dyaw*gate*(1-yaw*yaw)/float(count);
    losses[n]=.5f*(dot(delta,delta)+dyaw*dyaw);
}
)MSL";

static void verify_gradient(Metal& metal,const std::string& output){
    constexpr uint batch=3;std::array<float,batch*observation_count> observations{};
    std::array<float,batch*4> means{{.5f,-.25f,.1f,.2f,3.f,2.f,-2.f,.5f,-.7f,.3f,.2f,-.2f}};
    std::array<float,batch*4> targets{{.2f,.4f,0,0,-.2f,.5f,.1f,0,.4f,-.3f,.1f,0}};
    for(uint n=0;n<batch;n++){
        for(uint k=0;k<20;k++)observations[n*observation_count+k]=n==0?.1f:(n==1?1.f:(k%2?1.f:.1f));
        observations[n*observation_count+hint_offset]=.2f;observations[n*observation_count+hint_offset+1]=-.1f;
    }
    auto obs=metal.buffer(sizeof(observations),observations.data()),mu=metal.buffer(sizeof(means),means.data());
    auto target=metal.buffer(sizeof(targets),targets.data()),gradient=metal.buffer(sizeof(means)),losses=metal.buffer(batch*4);
    auto count=PPOTrainer::scalar(metal,batch);auto cb=[metal.queue commandBuffer];
    metal.dispatch(cb,metal.pipeline("imitation_gradient"),batch,{obs,mu,target,gradient,losses,count},64);metal.finish(cb);
    const auto loss=[&](uint n){
        float gate=n==0?.25f:(n==1?1.f:.625f);Point q{};
        for(uint k=0;k<3;k++){float prior=observations[n*observation_count+hint_offset+k];q[k]=std::tanh(prior+gate*(means[n*4+k]-prior));}
        float scale=1/std::max(1.f,length(q)),value=0;
        for(uint k=0;k<3;k++){float error=q[k]*scale-targets[n*4+k];value+=error*error;}
        float yaw=std::tanh(gate*means[n*4+3])-targets[n*4+3];return .5f*(value+yaw*yaw)/batch;
    };
    float worst=0;constexpr float epsilon=.001f;
    for(uint n=0;n<batch;n++)for(uint k=0;k<4;k++){
        uint i=n*4+k;float value=means[i];means[i]=value+epsilon;float plus=loss(n);
        means[i]=value-epsilon;float minus=loss(n);means[i]=value;
        worst=std::max(worst,std::fabs((plus-minus)/(2*epsilon)-static_cast<float*>(gradient.contents)[i]));
    }
    require(worst<1e-4f,"imitation deployed-command gradient differs from finite differences");
    const std::filesystem::path path(output);
    if(path.has_parent_path())std::filesystem::create_directories(path.parent_path());
    std::ofstream report(output);require(bool(report),"cannot write gradient verification");report<<"{\"passed\":true,\"cases\":3,\"checked_components\":12,\"includes_spherical_cap\":true,\"finite_difference_max_error\":"<<worst<<"}\n";
    std::cout<<"imitation_gradient PASS max_error="<<worst<<std::endl;
}



static std::array<float,4> teacher_command(const fixed_ppo::ActorParams& actor,const float* obs) {
    float hidden[64],means[4];fixed_ppo::actor_forward(obs,actor,hidden,means);
    float open=0;for(uint i=0;i<20;i++)open+=obs[i]*12>2.5f;
    const float gate=.25f+.75f*open/20;
    std::array<float,4> result;
    for(uint i=0;i<3;i++) {
        const float prior=obs[hint_offset+i];
        result[i]=std::tanh(prior+gate*(means[i]-prior));
    }
    const float magnitude=length({result[0],result[1],result[2]});
    for(uint i=0;i<3;i++)result[i]/=std::max(1.0f,magnitude);
    result[3]=std::tanh(gate*means[3]);return result;
}

static int collect(const std::string& checkpoint,const std::string& bank_path,
                   const std::string& output,uint32_t group) {
    require(group<16,"offline group budget exceeded");
    Metal metal;metal.compile(base_source()+PPO_TRAINER_MSL+waypoint::kWaypointKernels+imitation_kernel);
    waypoint::BankControl original_control{};std::string bank_sha;
    const auto original=waypoint::read_bank(bank_path,original_control,bank_sha);
    const uint32_t envs=original_control.count/original_control.period;
    require(envs>0&&envs<=128,"teacher environment count must be 1..128");
    std::vector<Demonstration> retained;uint successes=0,contacts=0,timeouts=0;
    float worst_label_error=0;
    std::ofstream episodes(output+".episodes.csv");require(bool(episodes),"cannot write teacher episode records");
    episodes<<"record,group,success,collision,timeout,time_s,path_m,final_distance_m,rows_retained\n";
    for(uint32_t slot=0;slot<original_control.period;slot++) {
        std::vector<waypoint::BankEntry> bank;
        for(uint env=0;env<envs;env++)bank.push_back(original[env*original_control.period+slot]);
        waypoint::BankControl control{1,envs};SimConfig cfg;
        cfg.n=envs;cfg.eval=1;cfg.mode=17;cfg.seed=700001;cfg.speed=1.5f;
        cfg.max_steps=400;cfg.geometry_memory=1;
        Sim sim(metal,cfg,32);navigation_training::load_actor(sim,checkpoint,false);
        fixed_ppo::ActorParams actor;std::memcpy(actor.values.data(),sim.actor.contents,sim.actor.length);
        auto run=waypoint::make_local_run(sim,bank,control,metal.pipeline("waypoint_task_apply"));
        auto probe=[metal.queue commandBuffer];waypoint::local_probe(run,probe);metal.finish(probe);
        waypoint::verify_reset(run,bank,control,true);
        std::vector<std::vector<Demonstration>> trajectories(envs);
        std::vector<uint32_t> before_steps(envs);std::vector<bool> active(envs);
        for(uint tick=0;tick<400;tick++) {@autoreleasepool {
            const SimRun* before=static_cast<const SimRun*>(sim.runs.contents);
            bool any=false;
            for(uint env=0;env<envs;env++){active[env]=before[env].episodes==0;before_steps[env]=before[env].steps;any|=active[env];}
            if(!any)break;
            auto commands=[metal.queue commandBuffer];waypoint::local_tick(run,commands,tick,tick==0);metal.finish(commands);
            const float* obs=static_cast<const float*>(sim.obs.contents);
            const float* applied=static_cast<const float*>(sim.commands.contents);
            for(uint env=0;env<envs;env++)if(active[env]) {
                Demonstration row{};row.group=group;row.record=env*original_control.period+slot;
                std::copy_n(obs+((tick%32)*envs+env)*observation_count,observation_count,row.observation.data());
                row.command=teacher_command(actor,row.observation.data());
                std::array<float,4> actual;
                std::copy_n(applied+(env*8+before_steps[env]%8)*4,4,actual.data());
                const float norm=length({actual[0],actual[1],actual[2]});
                for(uint axis=0;axis<3;axis++)actual[axis]/=std::max(1.0f,norm);
                for(uint axis=0;axis<4;axis++)worst_label_error=std::max(worst_label_error,std::fabs(actual[axis]-row.command[axis]));
                require(worst_label_error<1e-4f,"teacher label differs from actual deployed body command");
                for(float x:row.observation)require(std::isfinite(x),"non-finite teacher observation");
                trajectories[env].push_back(row);
            }
        }}
        const auto* completed=static_cast<const SimRun*>(sim.runs.contents);
        const auto* states=static_cast<const RLPhysicsState*>(sim.states.contents);
        for(uint env=0;env<envs;env++) {
            const SimRun& episode=completed[env];require(episode.episodes==1,"teacher must complete exactly one flight");
            successes+=episode.successes;contacts+=episode.collisions;timeouts+=episode.timeouts;
            const auto& goal=bank[env].goal_position;double squared=0;
            for(uint axis=0;axis<3;axis++){const double d=states[env].position[axis]-goal[axis];squared+=d*d;}
            const size_t kept=episode.successes?trajectories[env].size():0;
            if(episode.successes)retained.insert(retained.end(),trajectories[env].begin(),trajectories[env].end());
            episodes<<(env*original_control.period+slot)<<','<<group<<','<<episode.successes<<','<<episode.collisions
                    <<','<<episode.timeouts<<','<<episode.elapsed<<','<<episode.path<<','<<std::sqrt(squared)<<','<<kept<<'\n';
        }
        episodes.flush();std::cout<<"teacher_slot="<<slot<<" successes="<<successes<<" rows="<<retained.size()<<std::endl;
    }
    require(!retained.empty(),"teacher produced no successful source trajectories");
    DatasetHeader header{};std::memcpy(header.magic,"NAVBC1\0",8);header.version=1;
    header.observation_count=observation_count;header.rows=retained.size();header.row_bytes=sizeof(Demonstration);header.groups=group+1;
    std::ofstream dataset(output,std::ios::binary);require(bool(dataset),"cannot open teacher dataset");
    dataset.write(reinterpret_cast<const char*>(&header),sizeof(header));
    dataset.write(reinterpret_cast<const char*>(retained.data()),retained.size()*sizeof(Demonstration));
    dataset.flush();require(bool(dataset),"teacher dataset write failed");dataset.close();
    std::ofstream summary(output+".json");summary<<"{\"group\":"<<group<<",\"episodes\":"<<original.size()
        <<",\"successes\":"<<successes<<",\"contacts\":"<<contacts<<",\"timeouts\":"<<timeouts
        <<",\"rows\":"<<retained.size()<<",\"label_max_error\":"<<worst_label_error
        <<",\"teacher_sha256\":\""<<challenge_evaluation::sha256_file(checkpoint)
        <<"\",\"bank_entry_sha256\":\""<<bank_sha<<"\",\"dataset_sha256\":\""
        <<challenge_evaluation::sha256_file(output)<<"\",\"actor_inputs\":\"184 deployable observations; no group or teacher identity\"}\n";
    return 0;
}

static std::vector<Demonstration> read_dataset(const std::string& path) {
    std::ifstream file(path,std::ios::binary);DatasetHeader header{};
    file.read(reinterpret_cast<char*>(&header),sizeof(header));
    require(bool(file)&&std::memcmp(header.magic,"NAVBC1\0",8)==0&&header.version==1&&
            header.observation_count==observation_count&&header.row_bytes==sizeof(Demonstration)&&
            header.rows>0&&header.rows<=1000000,"invalid consolidation dataset header");
    require(std::filesystem::file_size(path)==sizeof(header)+header.rows*sizeof(Demonstration),"consolidation dataset size mismatch");
    std::vector<Demonstration> data(header.rows);
    file.read(reinterpret_cast<char*>(data.data()),data.size()*sizeof(Demonstration));require(bool(file),"dataset read failed");
    for(const auto& row:data) {
        require(row.group<16,"dataset group out of range");
        for(float x:row.observation)require(std::isfinite(x),"dataset observation is non-finite");
        for(float x:row.command)require(std::isfinite(x)&&std::fabs(x)<=1.00001f,"dataset body label outside action contract");
        require(length({row.command[0],row.command[1],row.command[2]})<=1.00001f,"dataset velocity exceeds sphere cap");
    }
    return data;
}

static int train(const std::string& warm,const std::string& output,uint32_t updates,uint32_t seed,
                 const std::vector<std::string>& paths) {
    require(updates>0&&updates<=20000,"consolidation update budget 1..20000");
    std::vector<Demonstration> data;
    std::array<std::map<uint32_t,std::vector<size_t>>,16> by_record;
    for(const auto& path:paths){auto next=read_dataset(path);data.insert(data.end(),next.begin(),next.end());}
    for(size_t i=0;i<data.size();i++)by_record[data[i].group][data[i].record].push_back(i);
    std::array<std::vector<std::vector<size_t>>,16> episodes;
    std::vector<uint> groups;
    for(uint group=0;group<16;group++)if(!by_record[group].empty()) {
        groups.push_back(group);
        for(auto& entry:by_record[group])episodes[group].push_back(std::move(entry.second));
    }
    require(!groups.empty(),"no student data");
    Metal metal;metal.compile(base_source()+PPO_TRAINER_MSL+waypoint::kWaypointKernels+imitation_kernel);
    verify_gradient(metal,output+".gradient.json");
    SimConfig cfg;cfg.n=128;cfg.mode=22;cfg.seed=seed;cfg.geometry_memory=1;cfg.speed=1.5f;
    Sim policy(metal,cfg,32);navigation_training::load_actor(policy,warm,true);PPOTrainer checkpoint(policy);
    constexpr uint batch=512;
    auto obs=metal.buffer(batch*observation_count*4),target=metal.buffer(batch*4*4),hidden=metal.buffer(batch*64*4);
    auto means=metal.buffer(batch*4*4),dmean=metal.buffer(batch*4*4),dstd=metal.buffer(batch*4*4),delta=metal.buffer(batch*64*4);
    auto gradient=metal.buffer(policy.actor.length),first=metal.buffer(policy.actor.length),second=metal.buffer(policy.actor.length);
    auto losses=metal.buffer(batch*4),count=PPOTrainer::scalar(metal,batch),parameter_count=PPOTrainer::scalar(metal,uint(fixed_ppo::actor_param_count));
    auto norm_limit=PPOTrainer::scalar(metal,.5f),factor=metal.buffer(4),optimizer=metal.buffer(sizeof(PpoAdamHostConfig));
    std::memset(first.contents,0,first.length);std::memset(second.contents,0,second.length);std::memset(dstd.contents,0,dstd.length);
    std::vector<float> original_critic(fixed_ppo::critic_param_count);std::memcpy(original_critic.data(),policy.critic.contents,policy.critic.length);
    std::array<float,4> original_std;std::copy_n(static_cast<float*>(policy.actor.contents)+fixed_ppo::actor_log_std_offset,4,original_std.data());
    std::mt19937 random(seed);std::ofstream history(output+".history.csv");require(bool(history),"cannot write student history");
    history<<"update,balanced_command_loss,wall_s\n";const double started=seconds();
    for(uint step=1;step<=updates;step++) {@autoreleasepool {
        for(uint row=0;row<batch;row++) {
            // Equal groups, then equal successful episodes, then a frame within the episode.
            const auto& records=episodes[groups[(row+step)%groups.size()]];
            const auto& indices=records[std::uniform_int_distribution<size_t>(0,records.size()-1)(random)];
            const auto& sample=data[indices[std::uniform_int_distribution<size_t>(0,indices.size()-1)(random)]];
            std::memcpy(static_cast<float*>(obs.contents)+row*observation_count,sample.observation.data(),observation_count*4);
            std::memcpy(static_cast<float*>(target.contents)+row*4,sample.command.data(),4*4);
        }
        PpoAdamHostConfig settings{.0003f,.9f,.999f,1e-8f,0,step};std::memcpy(optimizer.contents,&settings,sizeof(settings));
        auto commands=[metal.queue commandBuffer];
        metal.dispatch(commands,metal.pipeline("ppo_actor_forward"),batch,{obs,policy.actor,hidden,means,count},64);
        metal.dispatch(commands,metal.pipeline("imitation_gradient"),batch,{obs,means,target,dmean,losses,count},64);
        metal.dispatch(commands,metal.pipeline("ppo_actor_hidden_delta"),batch*64,{policy.actor,dmean,hidden,delta,count},128);
        metal.dispatch(commands,metal.pipeline("ppo_actor_grad_direct"),fixed_ppo::actor_param_count,{obs,hidden,dmean,dstd,delta,gradient,count},128);
        metal.dispatch(commands,metal.pipeline("ppo_grad_scale_factor"),256,{gradient,factor,parameter_count,norm_limit},256);
        metal.dispatch(commands,metal.pipeline("ppo_apply_grad_scale"),fixed_ppo::actor_param_count,{gradient,factor,parameter_count},128);
        metal.dispatch(commands,metal.pipeline("ppo_adam_update"),fixed_ppo::actor_param_count,{policy.actor,gradient,first,second,parameter_count,optimizer},128);
        metal.finish(commands);
        if(step==1||step%100==0||step==updates) {
            double loss=0;for(uint row=0;row<batch;row++)loss+=static_cast<float*>(losses.contents)[row]/batch;
            history<<step<<','<<loss<<','<<seconds()-started<<'\n';history.flush();
            // A parameter warmstart: PPO optimizer remains fresh; this is not RL resume.
            checkpoint.save_checkpoint(output,0,seed,0);
            std::cout<<"student_update="<<step<<" loss="<<loss<<std::endl;
        }
    }}
    require(std::memcmp(original_critic.data(),policy.critic.contents,policy.critic.length)==0,"student fitting changed critic");
    require(std::memcmp(original_std.data(),static_cast<float*>(policy.actor.contents)+fixed_ppo::actor_log_std_offset,16)==0,"student fitting changed exploration parameters");
    std::ofstream summary(output+".summary.json");summary<<"{\"updates\":"<<updates<<",\"seed\":"<<seed<<",\"rows\":"<<data.size()
        <<",\"groups\":"<<groups.size()<<",\"actor_observations\":184,\"critic_unchanged\":true,\"exploration_unchanged\":true,\"teacher_identity_input\":false}\n";
    return 0;
}
}

int main(int argc,char** argv) {@autoreleasepool {try {
    require(argc>=2,"collect CHECKPOINT BANK DATA GROUP | train WARM OUT UPDATES SEED DATA...");
    const std::string command=argv[1];
    if(command=="collect"){require(argc==6,"collect CHECKPOINT BANK DATA GROUP");return consolidation::collect(argv[2],argv[3],argv[4],std::stoul(argv[5]));}
    if(command=="train"){require(argc>=7,"train WARM OUT UPDATES SEED DATA...");std::vector<std::string> paths(argv+6,argv+argc);return consolidation::train(argv[2],argv[3],std::stoul(argv[4]),std::stoul(argv[5]),paths);}
    throw std::runtime_error("unknown consolidation command");
}catch(const std::exception& e){std::cerr<<"ERROR: "<<e.what()<<'\n';return 1;}}}
