// Source-simulator demonstrations only. Privileged routes control the teacher;
// the student always observes the final goal and real sensor/history buffers.
#define main embedded_navigation_main
#include "main.mm"
#undef main
#include <random>

namespace navigation_imitation {
using Point = std::array<float,3>;
constexpr uint observation_count = uint(fixed_ppo::actor_obs_dim);
constexpr uint hint_offset = observation_count - 3;
static_assert(observation_count==184 || observation_count==824, "imitation requires a versioned guided actor");
struct Demonstration {
    std::array<float,observation_count> observation;
    std::array<float,4> command; // body velocity / speed cap, yaw / .5
};
static Point difference(Point a,Point b){return {a[0]-b[0],a[1]-b[1],a[2]-b[2]};}
static float length(Point p){return std::sqrt(p[0]*p[0]+p[1]*p[1]+p[2]*p[2]);}
static Point interpolate(Point a,Point b,float u){return {a[0]+u*(b[0]-a[0]),a[1]+u*(b[1]-a[1]),a[2]+u*(b[2]-a[2])};}
static bool visible(const WWorld& world,Point from,Point to,float time){
    const int count=std::max(1,int(std::ceil(length(difference(to,from))/.02f)));
    for(int i=0;i<=count;i++){
        const auto p=interpolate(from,to,float(i)/count);
        if(wclearance(world,wv(p[0],p[1],p[2]),time)<.04f)return false;
    }
    return true;
}
static Point route_target(const WWorld& world,const std::vector<Point>& route,Point position,float time){
    size_t segment=0;float fraction=0,best=1e9f;
    for(size_t i=0;i+1<route.size();i++){
        auto d=difference(route[i+1],route[i]),r=difference(position,route[i]);
        float square=d[0]*d[0]+d[1]*d[1]+d[2]*d[2];
        float u=std::clamp((r[0]*d[0]+r[1]*d[1]+r[2]*d[2])/std::max(square,1e-8f),0.f,1.f);
        float error=length(difference(interpolate(route[i],route[i+1],u),position));
        if(error<best){best=error;segment=i;fraction=u;}
    }
    Point selected=interpolate(route[segment],route[segment+1],fraction);
    for(size_t i=segment;i+1<route.size();i++){
        float begin=i==segment?fraction:0;
        const int count=std::max(1,int(std::ceil(length(difference(route[i+1],route[i]))/.05f)));
        for(int j=0;j<=count;j++){
            auto candidate=interpolate(route[i],route[i+1],begin+(1-begin)*float(j)/count);
            if(length(difference(candidate,position))>1.5f || !visible(world,position,candidate,time))return selected;
            selected=candidate;
        }
    }
    return selected;
}

static void observe_student(Sim& sim){
    auto& m=sim.m;auto cb=[m.queue commandBuffer];auto c=sim.configs[0];
    m.dispatch(cb,sim.depth_p,320,{sim.states,sim.runs,sim.worlds,sim.sensors,sim.physics,c,sim.poses});
    m.dispatch(cb,sim.memory_points_p,640,{sim.states,sim.runs,sim.sensors,sim.poses,sim.memory_points,sim.physics,c});
    m.dispatch(cb,sim.memory_candidates_p,85,{sim.states,sim.worlds,sim.memory_points,sim.memory_clearances,c});
    m.dispatch(cb,sim.observe_p,1,{sim.states,sim.runs,sim.worlds,sim.sensors,sim.obs,sim.co,sim.physics,c,sim.poses,sim.memory_clearances},64);
    m.finish(cb);
}
static void advance_teacher(Sim& sim,Point velocity,float normalized_yaw=0){
    auto& run=*static_cast<SimRun*>(sim.runs.contents);
    const auto& state=*static_cast<RLPhysicsState*>(sim.states.contents);
    float rotation[9];cpu_reference::rotation(state.orientation_wxyz,rotation);
    auto* commands=static_cast<float*>(sim.commands.contents);
    for(int axis=0;axis<3;axis++){
        float body=rotation[axis]*velocity[0]+rotation[3+axis]*velocity[1]+rotation[6+axis]*velocity[2];
        run.previous_nav[axis]=body/sim.cfg.speed;
        run.desired_velocity[axis]=velocity[axis];
        commands[(run.steps%8)*4+axis]=run.previous_nav[axis];
    }
    require(std::isfinite(normalized_yaw)&&std::fabs(normalized_yaw)<=1,"teacher yaw command outside deployment contract");
    run.previous_nav[3]=normalized_yaw;commands[(run.steps%8)*4+3]=normalized_yaw;
    const auto& physics=*static_cast<RLPhysicsParams*>(sim.physics.contents);
    run.yaw+=normalized_yaw*physics.dt*sim.cfg.substeps*.5f;
    auto& m=sim.m;auto cb=[m.queue commandBuffer];
    m.dispatch(cb,sim.advance_p,1,{sim.states,sim.runs,sim.worlds,sim.sensors,sim.commands,sim.raptor,sim.critic,
        sim.rewards,sim.next_values,sim.terminated,sim.truncated,sim.physics,sim.configs[0],
        sim.bank_worlds,sim.bank_schedule,sim.bank_control,sim.bank_active_ids,sim.bank_transition_ids,
        sim.environment_physics,sim.runtime_control,sim.task_states,sim.task_control,
        sim.potential_fields,sim.potential_spec,sim.potential_control},64);
    m.finish(cb);
}
static Point student_velocity(const Demonstration& row,const fixed_ppo::ActorParams& actor,const RLPhysicsState& state){
    float hidden[64],mean[4],rotation[9];fixed_ppo::actor_forward(row.observation.data(),actor,hidden,mean);
    float open=0;for(uint k=0;k<20;k++)open+=row.observation[k]*12>2.5f;
    float gate=.25f+.75f*open/20;Point body{};
    for(uint k=0;k<3;k++){float prior=row.observation[hint_offset+k];body[k]=std::tanh(prior+gate*(mean[k]-prior));}
    float scale=1.5f/std::max(1.f,length(body));for(float& value:body)value*=scale;
    cpu_reference::rotation(state.orientation_wxyz,rotation);
    return {rotation[0]*body[0]+rotation[1]*body[1]+rotation[2]*body[2],
            rotation[3]*body[0]+rotation[4]*body[1]+rotation[5]*body[2],
            rotation[6]*body[0]+rotation[7]*body[1]+rotation[8]*body[2]};
}
static std::vector<Demonstration> collect(Metal& metal,const std::string& bank,const std::string& output,const std::string& student=""){
    std::string hash;
    const auto levels=challenge_evaluation::load_split(bank,"train",challenge_evaluation::sha256_file(std::string(SOURCE_DIR)+"/world.hpp"),hash);
    require(hash=="e2170fbe8e8c2d074ffb08de4269175a6de0bdcd410c3b60bd699ee7b5fc491e","demonstration bank differs from approved mirrored source bank");
    std::ofstream csv(output+"/teacher.csv");
    csv<<"failure_id,success,collision,timeout,elapsed_s,path_m,minimum_clearance_m,rows_retained,bank_sha256\n";
    fixed_ppo::ActorParams actor;
    if(!student.empty()){
        std::ifstream file(student,std::ios::binary);auto header=read_checkpoint_header(file);
        require(header.actor_count==fixed_ppo::actor_param_count,"aggregation actor dimensions");
        file.read(reinterpret_cast<char*>(actor.values.data()),sizeof(actor));require(bool(file),"aggregation actor read");
    }
    std::vector<Demonstration> retained;uint successful=0,count=0;
    for(const auto& level:levels){if(level.family!=14)continue;
        SimConfig cfg;cfg.n=1;cfg.eval=1;cfg.mode=17;cfg.speed=1.5f;cfg.max_steps=400;cfg.geometry_memory=1;cfg.seed=level.seed;
        Sim sim(metal,cfg,1);*static_cast<WWorld*>(sim.worlds.contents)=level.world;
        auto& run=*static_cast<SimRun*>(sim.runs.contents);
        auto& state=*static_cast<RLPhysicsState*>(sim.states.contents);
        Point goal={level.world.goal[0],level.world.goal[1],level.world.goal[2]};
        run.initial_distance=length(difference(goal,{state.position[0],state.position[1],state.position[2]}));
        std::vector<Demonstration> episode;
        while(!run.episodes && run.steps<400){
            observe_student(sim); // world.goal never changes: teacher route cannot leak here.
            Point position={state.position[0],state.position[1],state.position[2]};
            Point target=route_target(level.world,level.witness_route,position,run.elapsed);
            Point velocity=difference(target,position);float magnitude=length(velocity);
            float scale=std::min(1.f,1.49f/std::max(magnitude,1e-8f));
            for(float& component:velocity)component*=scale;
            Demonstration row;std::memcpy(row.observation.data(),sim.obs.contents,sizeof(row.observation));
            require(std::memcmp(static_cast<WWorld*>(sim.worlds.contents)->goal,level.world.goal,sizeof(level.world.goal))==0,"student goal changed");
            float rotation[9];cpu_reference::rotation(state.orientation_wxyz,rotation);
            for(int axis=0;axis<3;axis++)row.command[axis]=(rotation[axis]*velocity[0]+rotation[3+axis]*velocity[1]+rotation[6+axis]*velocity[2])/cfg.speed;
            row.command[3]=0;
            // Dataset aggregation labels states visited by a 50/50 mixture.
            // Failed episodes are retained as corrective labels, not successful
            // expert flights. Labels must have a clear short teacher segment.
            if(student.empty()||visible(level.world,position,interpolate(position,target,.1f),run.elapsed))episode.push_back(row);
            if(!student.empty()){
                auto predicted=student_velocity(row,actor,state);
                for(uint k=0;k<3;k++)velocity[k]=.5f*(velocity[k]+predicted[k]);
            }
            advance_teacher(sim,velocity);
        }
        bool success=run.successes>0 && !run.collisions;
        if(success||!student.empty())retained.insert(retained.end(),episode.begin(),episode.end());
        if(success)successful++;
        csv<<level.failure_id<<','<<success<<','<<bool(run.collisions)<<','<<bool(run.timeouts)<<','<<run.elapsed<<','<<run.path<<','<<run.min_clearance<<','<<((success||!student.empty())?episode.size():0)<<','<<hash<<'\n';csv.flush();count++;
        std::cout<<"teacher "<<count<<"/30 success="<<success<<" time="<<run.elapsed<<" rows="<<retained.size()<<std::endl;
    }
    require(!retained.empty()&&retained.size()<=20000,"demonstration budget exceeded or no successful flights");
    std::ofstream dataset(output+"/demonstrations.bin",std::ios::binary);
    uint64_t rows=retained.size();dataset.write(reinterpret_cast<char*>(&rows),sizeof(rows));
    dataset.write(reinterpret_cast<const char*>(retained.data()),retained.size()*sizeof(Demonstration));
    require(bool(dataset),"demonstration write failed");
    std::cout<<"teacher_success="<<successful<<"/"<<count<<" retained="<<rows<<std::endl;
    return retained;
}

// MSE in the deployed bounded command. Its Jacobian includes the guidance gate
// and spherical velocity cap, so unreachable raw-mean labels do not dominate.
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

static void evaluate_student(Metal& metal,const std::string& checkpoint,const std::string& bank,const std::string& output){
    if constexpr(observation_count==824)raw_depth_bank_evaluate(metal,checkpoint,bank,"dev",output,17,1.5f,400);
    else challenge_evaluation::run(metal,checkpoint,bank,"dev",output,17,1.5f,400);
}

static void train(Metal& metal,const std::vector<Demonstration>& dataset,const std::string& warm,const std::string& bank,const std::string& out,uint updates){
    constexpr uint batch=256;
    SimConfig cfg;cfg.mode=22;cfg.geometry_memory=1;cfg.speed=1.5;cfg.max_steps=400;cfg.seed=42;
    Sim policy(metal,cfg,32);navigation_training::load_actor(policy,warm,true);PPOTrainer checkpoint(policy);
    auto observations=metal.buffer(batch*observation_count*4),targets=metal.buffer(batch*4*4),hidden=metal.buffer(batch*64*4);
    auto means=metal.buffer(batch*4*4),dmean=metal.buffer(batch*4*4),dstd=metal.buffer(batch*4*4),delta=metal.buffer(batch*64*4);
    auto gradient=metal.buffer(fixed_ppo::actor_param_count*4),first=metal.buffer(fixed_ppo::actor_param_count*4),second=metal.buffer(fixed_ppo::actor_param_count*4);
    auto losses=metal.buffer(batch*4),count=PPOTrainer::scalar(metal,batch),parameter_count=PPOTrainer::scalar(metal,uint(fixed_ppo::actor_param_count));
    auto norm_limit=PPOTrainer::scalar(metal,.5f),factor=metal.buffer(4),optimizer=metal.buffer(sizeof(PpoAdamHostConfig));
    std::memset(first.contents,0,first.length);std::memset(second.contents,0,second.length);std::memset(dstd.contents,0,dstd.length);
    std::mt19937 random(42);std::uniform_int_distribution<size_t> sample(0,dataset.size()-1);
    std::ofstream history(out+"/history.csv");history<<"update,batch_command_mse,wall_s\n";
    const double start=seconds();
    checkpoint.save_checkpoint(out+"/warm.bin",0,42,0);
    evaluate_student(metal,out+"/warm.bin",bank,out+"/warm-dev.csv");
    for(uint step=1;step<=updates;step++){@autoreleasepool{
        for(uint n=0;n<batch;n++){
            const auto& row=dataset[sample(random)];
            std::memcpy(static_cast<float*>(observations.contents)+n*observation_count,row.observation.data(),observation_count*4);
            std::memcpy(static_cast<float*>(targets.contents)+n*4,row.command.data(),4*4);
        }
        PpoAdamHostConfig settings{.0003f,.9f,.999f,1e-8f,0,step};std::memcpy(optimizer.contents,&settings,sizeof(settings));
        auto cb=[metal.queue commandBuffer];
        metal.dispatch(cb,metal.pipeline("ppo_actor_forward"),batch,{observations,policy.actor,hidden,means,count},64);
        metal.dispatch(cb,metal.pipeline("imitation_gradient"),batch,{observations,means,targets,dmean,losses,count},64);
        metal.dispatch(cb,metal.pipeline("ppo_actor_hidden_delta"),batch*64,{policy.actor,dmean,hidden,delta,count},128);
        metal.dispatch(cb,metal.pipeline("ppo_actor_grad_direct"),fixed_ppo::actor_param_count,{observations,hidden,dmean,dstd,delta,gradient,count},128);
        metal.dispatch(cb,metal.pipeline("ppo_grad_scale_factor"),256,{gradient,factor,parameter_count,norm_limit},256);
        metal.dispatch(cb,metal.pipeline("ppo_apply_grad_scale"),fixed_ppo::actor_param_count,{gradient,factor,parameter_count},128);
        metal.dispatch(cb,metal.pipeline("ppo_adam_update"),fixed_ppo::actor_param_count,{policy.actor,gradient,first,second,parameter_count,optimizer},128);
        metal.finish(cb);
        if(step==1||step%100==0||step==updates){
            double loss=0;for(uint n=0;n<batch;n++)loss+=static_cast<float*>(losses.contents)[n];loss/=batch;
            history<<step<<','<<loss<<','<<seconds()-start<<'\n';history.flush();
            const std::string path=out+"/candidate-"+std::to_string(step)+".bin";
            // Actor-only imitation: checkpoint optimizer buffers intentionally zero.
            // This is a compatible parameter warmstart, not a PPO resume receipt.
            checkpoint.save_checkpoint(path,0,42,0);
            evaluate_student(metal,path,bank,out+"/dev-"+std::to_string(step)+".csv");
            std::cout<<"imitation_update="<<step<<" command_mse="<<loss<<" wall_s="<<seconds()-start<<std::endl;
        }
    }}
}
} // namespace navigation_imitation

#ifndef NAVIGATION_IMITATION_EMBEDDED
int main(int argc,char** argv){@autoreleasepool{try{
    if(argc==3&&std::string(argv[1])=="--check"){
        Metal metal;metal.compile(base_source()+PPO_TRAINER_MSL+navigation_imitation::imitation_kernel);
        navigation_imitation::verify_gradient(metal,argv[2]);return 0;
    }
    require(argc>=4,"navigation_imitation BANK WARM_CHECKPOINT OUTPUT_DIR [UPDATES=1000]");
    const uint updates=argc>4?std::stoul(argv[4]):1000;require(updates>0&&updates<=2000,"imitation update budget1..2000");
    std::filesystem::create_directories(argv[3]);Metal metal;
    metal.compile(base_source()+PPO_TRAINER_MSL+navigation_imitation::imitation_kernel);
    const bool aggregate=argc>5;
    auto dataset=navigation_imitation::collect(metal,argv[1],argv[3],aggregate?argv[2]:"");
    if(aggregate){
        std::ifstream file(argv[5],std::ios::binary);uint64_t count=0;file.read(reinterpret_cast<char*>(&count),sizeof(count));
        require(bool(file)&&count>0&&count+dataset.size()<=20000,"invalid or oversized previous demonstration dataset");
        require(std::filesystem::file_size(argv[5])==8+count*sizeof(navigation_imitation::Demonstration),"previous demonstration observation format mismatch");
        size_t begin=dataset.size();dataset.resize(begin+count);
        file.read(reinterpret_cast<char*>(dataset.data()+begin),count*sizeof(navigation_imitation::Demonstration));
        require(bool(file),"truncated previous demonstration dataset");
        std::cout<<"aggregated_rows="<<dataset.size()<<std::endl;
    }
    navigation_imitation::train(metal,dataset,argv[2],argv[1],argv[3],updates);
    return 0;
}catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}}}
#endif
