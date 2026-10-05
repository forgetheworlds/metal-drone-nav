#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <unordered_map>
#include <cstring>
#include "world.hpp"
#include "sensor_profile.hpp"
#include "threat_evaluation.hpp"
#include "raptor.hpp"
#include "physics.hpp"
#include "navigation_runtime.hpp"
#include "navigation_tasks.hpp"
#include "ppo.hpp"
#include "profile.hpp"

static std::string read_text(const std::string& path) {
    std::ifstream f(path); if(!f) throw std::runtime_error("Cannot read "+path);
    return std::string(std::istreambuf_iterator<char>(f),{});
}
static void require(bool b,const std::string& msg) { if(!b) throw std::runtime_error(msg); }
static double seconds() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
static const char* world_kernels=R"MSL(
kernel void world_depth(device const WWorld* worlds [[buffer(0)]], device const float* poses [[buffer(1)]], device const float* times [[buffer(2)]], device float* out [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    uint n=i/320,k=i%320; const device float* p=poses+n*12; WVec ray=wcamera(k);
    WVec d=wv(p[3]*ray.x+p[4]*ray.y+p[5]*ray.z,p[6]*ray.x+p[7]*ray.y+p[8]*ray.z,p[9]*ray.x+p[10]*ray.y+p[11]*ray.z);
    out[i]=wray(worlds[n],wv(p[0],p[1],p[2]),d,times[n]);
}
kernel void world_depth_serial(device const WWorld* worlds [[buffer(0)]], device const float* poses [[buffer(1)]], device const float* times [[buffer(2)]], device float* out [[buffer(3)]], uint n [[thread_position_in_grid]]) {
    const device float* p=poses+n*12;
    for(uint k=0;k<320;k++) { WVec ray=wcamera(k); WVec d=wv(p[3]*ray.x+p[4]*ray.y+p[5]*ray.z,p[6]*ray.x+p[7]*ray.y+p[8]*ray.z,p[9]*ray.x+p[10]*ray.y+p[11]*ray.z);out[n*320+k]=wray(worlds[n],wv(p[0],p[1],p[2]),d,times[n]); }
}
kernel void world_clearance(device const WWorld* worlds [[buffer(0)]],device const float* poses [[buffer(1)]],device const float* times [[buffer(2)]],device float* out [[buffer(3)]],uint n [[thread_position_in_grid]]) {
    const device float* p=poses+n*12;out[n]=wclearance(worlds[n],wv(p[0],p[1],p[2]),times[n]);
}
)MSL";

struct Metal {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLLibrary> library;
    std::unordered_map<std::string,id<MTLComputePipelineState>> pipelines;
    M3TimestampProfile profiler;
    Metal(bool profile=false) {
        device=MTLCreateSystemDefaultDevice();require(device!=nil,"No Metal device"); queue=[device newCommandQueue];if(profile)profiler.initialize(device);
    }
    void compile(std::string source) {
        MTLCompileOptions* options=[MTLCompileOptions new];options.mathMode=MTLMathModeSafe;options.mathFloatingPointFunctions=MTLMathFloatingPointFunctionsPrecise;
        NSError* e=nil;library=[device newLibraryWithSource:[NSString stringWithUTF8String:source.c_str()] options:options error:&e];
        require(library!=nil,e?e.localizedDescription.UTF8String:"Metal compile failed");
    }
    id<MTLComputePipelineState> pipeline(const char* name) {
        auto found=pipelines.find(name);if(found!=pipelines.end())return found->second;
        NSError* e=nil;id<MTLFunction> fn=[library newFunctionWithName:[NSString stringWithUTF8String:name]];require(fn!=nil,std::string("Missing kernel ")+name);
        id<MTLComputePipelineState> p=[device newComputePipelineStateWithFunction:fn error:&e];require(p!=nil,e?e.localizedDescription.UTF8String:"Pipeline failed");pipelines[name]=p;return p;
    }
    id<MTLBuffer> buffer(size_t size,const void* data=nullptr) {
        id<MTLBuffer> b=[device newBufferWithLength:std::max(size,size_t(4)) options:MTLResourceStorageModeShared];require(b!=nil,"Buffer allocation failed");if(data)std::memcpy(b.contents,data,size);return b;
    }
    const char* pipelineName(id<MTLComputePipelineState> p){for(const auto& item:pipelines)if(item.second==p)return item.first.c_str();return "unknown";}
    void dispatch(id<MTLCommandBuffer> cb,id<MTLComputePipelineState> p,size_t count,std::initializer_list<id<MTLBuffer>> buffers,size_t group=128) {
        id<MTLComputeCommandEncoder> e=profiler.encoder(cb,profiler.armed?pipelineName(p):nullptr);[e setComputePipelineState:p];uint j=0;for(auto b:buffers)[e setBuffer:b offset:0 atIndex:j++];
        [e dispatchThreads:MTLSizeMake(count,1,1) threadsPerThreadgroup:MTLSizeMake(std::min(group,size_t(p.maxTotalThreadsPerThreadgroup)),1,1)];[e endEncoding];
    }
    double finish(id<MTLCommandBuffer> cb) { MTLTimestamp cpu0=0,gpu0=0;if(profiler.armed)[device sampleTimestamps:&cpu0 gpuTimestamp:&gpu0];[cb commit];[cb waitUntilCompleted];require(cb.status!=MTLCommandBufferStatusError,cb.error?cb.error.localizedDescription.UTF8String:"GPU failure");profiler.report(cb,cpu0,gpu0);return cb.GPUEndTime-cb.GPUStartTime; }
};

static const char* control_test_kernels=R"MSL(
kernel void raptor_fixture(device const float* weights [[buffer(0)]],device const float* obs [[buffer(1)]],device float* actions [[buffer(2)]],device float* hidden_out [[buffer(3)]],uint n [[thread_position_in_grid]]) {
    float h[16],o[22],a[4];raptor_reset(weights,h);
    for(uint t=0;t<16;t++){for(uint j=0;j<22;j++)o[j]=obs[t*22+j];raptor_forward(weights,o,h,a);for(uint j=0;j<4;j++)actions[(n*16+t)*4+j]=a[j];}
    for(uint j=0;j<16;j++)hidden_out[n*16+j]=h[j];
}
kernel void physics_fixture(device const RLPhysicsState* states [[buffer(0)]],device const float* actions [[buffer(1)]],device const float* winds [[buffer(2)]],device RLPhysicsState* out [[buffer(3)]],constant RLPhysicsParams& params [[buffer(4)]],uint n [[thread_position_in_grid]]) {
    RLPhysicsState s=states[n],next;float a[4],wind[3];for(uint j=0;j<4;j++)a[j]=actions[n*4+j];for(uint j=0;j<3;j++)wind[j]=winds[n*3+j];RLPhysicsParams local_params=params;rl_physics_step(s,a,wind,local_params,next);out[n]=next;
}
)MSL";
static std::string base_source() {
    std::string root=SOURCE_DIR;
    const std::string actor_obs_define="#define FIXED_PPO_ACTOR_OBS_DIM "+std::to_string(fixed_ppo::actor_obs_dim)+"\n";
    const std::string critic_obs_define="#define FIXED_PPO_CRITIC_OBS_DIM "+std::to_string(fixed_ppo::critic_obs_dim)+"\n";
#ifdef NAV_CRITIC_CONTROL_STATE
    const std::string critic_state_define="#define NAV_CRITIC_CONTROL_STATE "+std::to_string(NAV_CRITIC_CONTROL_STATE)+"\n";
#else
    const std::string critic_state_define="#define NAV_CRITIC_CONTROL_STATE 0\n";
#endif
    const std::string sensor_profile_define="#define NAV_SENSOR_PROFILE "+std::to_string(NAV_SENSOR_PROFILE)+"\n";
    return "#include <metal_stdlib>\nusing namespace metal;\n"+sensor_profile_define+read_text(root+"/world.hpp")+read_text(root+"/sensor_profile.hpp")+world_kernels+read_text(root+"/raptor.metal")+read_text(root+"/physics.metal")+read_text(root+"/physics_domain.hpp")+read_text(root+"/navigation_runtime.hpp")+read_text(root+"/navigation_tasks.hpp")+read_text(root+"/training_potential.hpp")+actor_obs_define+critic_obs_define+critic_state_define+read_text(root+"/ppo.metal")+read_text(root+"/guidance.hpp")+read_text(root+"/sim.metal")+read_text(root+"/memory.metal")+control_test_kernels;
}
static std::vector<float> poses(size_t n) { std::vector<float> p(n*12,0);for(size_t i=0;i<n;i++){p[i*12+2]=1.5f;p[i*12+3]=p[i*12+7]=p[i*12+11]=1;}return p; }
static void world_tests(Metal& m) {
    require(fabs(wray_sphere(wv(0,0,0),wv(1,0,0),wv(3,0,0),1)-2)<1e-6,"sphere hit");
    require(fabs(wray_sphere(wv(3,0,0),wv(1,0,0),wv(3,0,0),1)-1)<1e-6,"sphere inside exit");
    require(wray_sphere(wv(0,2,0),wv(1,0,0),wv(3,0,0),1)==1000,"sphere miss");
    require(fabs(wray_sphere(wv(0,1,0),wv(1,0,0),wv(3,0,0),1)-3)<1e-6,"sphere tangent");
    require(fabs(wray_box(wv(0,0,0),wv(1,0,0),wv(3,0,0),wv(1,1,1))-2)<1e-6,"box hit");
    require(wray_box(wv(0,2,0),wv(1,0,0),wv(3,0,0),wv(1,1,1))==1000,"parallel box miss");
    require(fabs(wray_cylinder(wv(0,0,0),wv(1,0,0),wv(3,0,0),1,1)-2)<1e-6,"cylinder side");
    require(fabs(wray_cylinder(wv(3,0,3),wv(0,0,-1),wv(3,0,0),1,1)-2)<1e-6,"cylinder cap");
    constexpr size_t n=128;std::vector<WWorld> w(n);auto p=poses(n);std::vector<float> t(n);
    for(size_t i=0;i<n;i++){wgenerate(w[i],uint(i+197),uint(i%4),8);t[i]=float(i%7)*0.1f;p[i*12+1]=(float(i%5)-2)*0.3f;}
    auto wb=m.buffer(sizeof(WWorld)*n,w.data()),pb=m.buffer(p.size()*4,p.data()),tb=m.buffer(n*4,t.data()),out=m.buffer(n*320*4),clear=m.buffer(n*4);
    auto cb=[m.queue commandBuffer];m.dispatch(cb,m.pipeline("world_depth"),n*320,{wb,pb,tb,out});m.dispatch(cb,m.pipeline("world_clearance"),n,{wb,pb,tb,clear});m.finish(cb);
    float error=0,ce=0;for(size_t i=0;i<n;i++) { for(uint k=0;k<320;k++)error=fmax(error,fabs(wray(w[i],wv(p[i*12],p[i*12+1],p[i*12+2]),wcamera(k),t[i])-((float*)out.contents)[i*320+k]));ce=fmax(ce,fabs(wclearance(w[i],wv(p[i*12],p[i*12+1],p[i*12+2]),t[i])-((float*)clear.contents)[i])); }
    require(error<3e-5f && ce<3e-5f,"world CPU/Metal mismatch");
    WWorld special{};special.count=1;special.obstacles[0].kind=1;special.obstacles[0].center[0]=3;special.obstacles[0].center[2]=1.5f;special.obstacles[0].size[0]=0.5f;special.obstacles[0].velocity[0]=-1;
    require(fabs(wray(special,wv(0,0,1.5f),wv(1,0,0),1)-1.5f)<1e-6f,"moving sphere time");require(wclearance(special,wv(2,0,1.5f),1)<0,"independent collision");special.count=0;require(wray(special,wv(0,0,2.5f),wv(1,0,0),0)==12,"max range clamp");
    std::cout<<"world PASS analytic rays/collision/motion + "<<n*320<<" CPU/Metal rays; max_depth_error="<<error<<" clearance_error="<<ce<<"\n";
}
static RaptorWeights load_raptor() { RaptorWeights w;require(raptor_load_weights((std::string(SOURCE_DIR)+"/assets/raptor.bin").c_str(),&w),"RAPTOR load failed");return w; }
static void raptor_tests(Metal& m) {
    auto weights=load_raptor();std::ifstream f(std::string(SOURCE_DIR)+"/assets/raptor.bin",std::ios::binary);f.seekg(16+sizeof(RaptorWeights));char head[20];f.read(head,20);require(f && std::memcmp(head,"RPFIX1\0\0",8)==0,"RAPTOR official fixture header");
    std::vector<float> obs(16*22),expected(16*4);f.read((char*)obs.data(),obs.size()*4);f.read((char*)expected.data(),expected.size()*4);require(bool(f),"RAPTOR fixture read");
    float h[16],a[4];raptor_reset(weights,h);float official_error=0;
    for(int t=0;t<16;t++){raptor_forward(weights,obs.data()+t*22,h,a);for(int j=0;j<4;j++)official_error=fmax(official_error,fabs(a[j]-expected[t*4+j]));}
    constexpr size_t n=128;auto wb=m.buffer(sizeof(weights),&weights),ob=m.buffer(obs.size()*4,obs.data()),ab=m.buffer(n*16*4*4),hb=m.buffer(n*16*4);
    auto cb=[m.queue commandBuffer];m.dispatch(cb,m.pipeline("raptor_fixture"),n,{wb,ob,ab,hb});m.finish(cb);float gpuerror=0,herror=0;
    for(size_t i=0;i<n;i++){for(int t=0;t<16;t++)for(int j=0;j<4;j++)gpuerror=fmax(gpuerror,fabs(((float*)ab.contents)[(i*16+t)*4+j]-expected[t*4+j]));for(int j=0;j<16;j++)herror=fmax(herror,fabs(((float*)hb.contents)[i*16+j]-h[j]));}
    std::cout<<"RAPTOR debug CPU="<<official_error<<" GPU="<<gpuerror<<" h="<<herror<<"\n";require(official_error<2e-5 && gpuerror<2e-5 && herror<2e-5,"RAPTOR official parity failed");
    std::cout<<"RAPTOR PASS official sequence CPU_error="<<official_error<<" Metal_error="<<gpuerror<<" recurrent_CPU_Metal_error="<<herror<<" N="<<n<<" steps=16\n";
}
static void physics_tests(Metal& m) {
    struct Header { char magic[8]; uint32_t version,cases,horizon; } h{};
    static_assert(sizeof(Header)==20);
    std::ifstream f(std::string(SOURCE_DIR)+"/assets/physics.bin",std::ios::binary);
    require(bool(f),"L2F physics fixtures missing; build reference.cpp with RAPTOR's pinned rl-tools submodule");
    f.read(reinterpret_cast<char*>(&h),sizeof(h));
    require(f && std::memcmp(h.magic,"RLPHYS1\0",8)==0 && h.version==1,"L2F physics fixture header mismatch");
    require(h.cases>0 && h.horizon>0,"L2F physics fixture has no samples");
    RLPhysicsParams params{};f.read(reinterpret_cast<char*>(&params),sizeof(params));require(bool(f),"L2F physics params read failed");
    const size_t n=size_t(h.cases)*h.horizon;
    std::vector<RLPhysicsState> states(n),expected(n),cpu(n);
    std::vector<float> actions(n*4),winds(n*3);
    for(size_t i=0;i<n;i++){
        f.read(reinterpret_cast<char*>(&states[i]),sizeof(RLPhysicsState));
        f.read(reinterpret_cast<char*>(actions.data()+i*4),4*sizeof(float));
        f.read(reinterpret_cast<char*>(winds.data()+i*3),3*sizeof(float));
        f.read(reinterpret_cast<char*>(&expected[i]),sizeof(RLPhysicsState));
        require(bool(f),"L2F physics fixture record truncated");
        rl_physics_step(states[i],actions.data()+i*4,winds.data()+i*3,params,cpu[i]);
    }
    float cpu_error=0.0f;
    for(size_t i=0;i<n;i++){
        const float* actual=reinterpret_cast<const float*>(&cpu[i]);
        const float* reference=reinterpret_cast<const float*>(&expected[i]);
        for(size_t j=0;j<17;j++)cpu_error=fmax(cpu_error,fabs(actual[j]-reference[j]));
    }
    auto sb=m.buffer(n*sizeof(RLPhysicsState),states.data()),ab=m.buffer(actions.size()*sizeof(float),actions.data());
    auto wb=m.buffer(winds.size()*sizeof(float),winds.data()),pb=m.buffer(sizeof(params),&params),ob=m.buffer(n*sizeof(RLPhysicsState));
    auto cb=[m.queue commandBuffer];m.dispatch(cb,m.pipeline("physics_fixture"),n,{sb,ab,wb,ob,pb},128);m.finish(cb);
    const auto* gpu=reinterpret_cast<const RLPhysicsState*>(ob.contents);float metal_error=0.0f;
    for(size_t i=0;i<n;i++){
        const float* actual=reinterpret_cast<const float*>(&gpu[i]);
        const float* reference=reinterpret_cast<const float*>(&expected[i]);
        for(size_t j=0;j<17;j++)metal_error=fmax(metal_error,fabs(actual[j]-reference[j]));
    }
    require(cpu_error<2.0e-5f && metal_error<2.0e-5f,"L2F CPU/Metal physics parity failed");
    std::cout<<"physics PASS RAPTOR-pinned rl-tools official L2F fixtures cases="<<h.cases<<" steps="<<h.horizon
             <<" samples="<<n<<" CPU_error="<<cpu_error<<" Metal_error="<<metal_error<<"\n";
}

static void raptor_px4_adapter_tests() {
    constexpr float yaw = 0.7f;
    const float c = std::cos(yaw), s = std::sin(yaw);
    const float position[3] = {1.3f, -0.4f, 2.1f};
    const float desired_velocity[3] = {0.8f, -1.1f, 0.3f};
    const float target_position[3] = {
        position[0] + desired_velocity[0] * 0.5f,
        position[1] + desired_velocity[1] * 0.5f,
        position[2] + desired_velocity[2] * 0.5f
    };
    const float world_velocity[3] = {-1.4f, 0.3f, 0.8f};
    const float body_rates[3] = {0.9f, -0.2f, 0.4f};
    const float previous_action[4] = {-0.6f, 0.3f, 0.7f, 0.15f};
    const float target_orientation[4] = {std::cos(yaw * 0.5f), 0, 0, std::sin(yaw * 0.5f)};
    const float current_orientation[4] = {0.92736185f, 0.1735133f, -0.12628517f, 0.30754274f};
    float got[22], expected[22]{};
    raptor_pack_observation(position, current_orientation, world_velocity, body_rates,
                            target_position, target_orientation, desired_velocity,
                            previous_action, got);

    const float dp[3] = {position[0]-target_position[0], position[1]-target_position[1], position[2]-target_position[2]};
    const float dv[3] = {world_velocity[0]-desired_velocity[0], world_velocity[1]-desired_velocity[1], world_velocity[2]-desired_velocity[2]};
    expected[0] = raptor_clip_error(c*dp[0]+s*dp[1], 0.5f);
    expected[1] = raptor_clip_error(-s*dp[0]+c*dp[1], 0.5f);
    expected[2] = raptor_clip_error(dp[2], 0.5f);
    expected[12] = raptor_clip_error(c*dv[0]+s*dv[1], 1.0f);
    expected[13] = raptor_clip_error(-s*dv[0]+c*dv[1], 1.0f);
    expected[14] = raptor_clip_error(dv[2], 1.0f);
    const float qinv[4] = {target_orientation[0], -target_orientation[1], -target_orientation[2], -target_orientation[3]};
    const float qw=qinv[0], qx=qinv[1], qy=qinv[2], qz=qinv[3];
    const float aw=current_orientation[0], ax=current_orientation[1], ay=current_orientation[2], az=current_orientation[3];
    const float q[4] = {qw*aw-qx*ax-qy*ay-qz*az,
                        qw*ax+qx*aw+qy*az-qz*ay,
                        qw*ay-qx*az+qy*aw+qz*ax,
                        qw*az+qx*ay-qy*ax+qz*aw};
    expected[3] = 1-2*q[2]*q[2]-2*q[3]*q[3]; expected[4] = 2*q[1]*q[2]-2*q[0]*q[3]; expected[5] = 2*q[1]*q[3]+2*q[0]*q[2];
    expected[6] = 2*q[1]*q[2]+2*q[0]*q[3]; expected[7] = 1-2*q[1]*q[1]-2*q[3]*q[3]; expected[8] = 2*q[2]*q[3]-2*q[0]*q[1];
    expected[9] = 2*q[1]*q[3]-2*q[0]*q[2]; expected[10] = 2*q[2]*q[3]+2*q[0]*q[1]; expected[11] = 1-2*q[1]*q[1]-2*q[2]*q[2];
    for(int i=0;i<3;i++) expected[15+i]=body_rates[i];
    for(int i=0;i<4;i++) expected[18+i]=previous_action[i];
    float max_error=0;
    for(int i=0;i<22;i++) max_error=std::max(max_error,std::fabs(got[i]-expected[i]));
    require(max_error < 2e-6f, "RAPTOR/PX4 FLU trajectory-adapter parity");
    std::cout << "RAPTOR adapter PASS PX4 target-frame transform (0.5 s preview fixture), max_error=" << max_error << "\n";
}

static void depth_benchmark(Metal& m) {
    auto parallel=m.pipeline("world_depth"),serial=m.pipeline("world_depth_serial");
    std::cout<<"N,depth_parallel_ms,depth_serial_ms,cpu_ms,parallel_rays_per_sec\n";
    for(size_t n:{1,32,128,512,2048,8192,32768}) {
        std::vector<WWorld>w(n);for(size_t i=0;i<n;i++)wgenerate(w[i],uint(i+197),1,8);auto p=poses(n);std::vector<float>times(n,0);
        auto wb=m.buffer(n*sizeof(WWorld),w.data()),pb=m.buffer(p.size()*4,p.data()),tb=m.buffer(n*4,times.data()),out=m.buffer(n*320*4);
        double par=1e9,ser=1e9;for(int j=0;j<4;j++){auto cb=[m.queue commandBuffer];m.dispatch(cb,parallel,n*320,{wb,pb,tb,out});par=std::min(par,m.finish(cb));cb=[m.queue commandBuffer];m.dispatch(cb,serial,n,{wb,pb,tb,out});ser=std::min(ser,m.finish(cb));}
        double start=seconds();volatile float sink=0;for(size_t i=0;i<n;i++)for(uint k=0;k<320;k++)sink=wray(w[i],wv(0,0,1.5f),wcamera(k),0);double cpu=seconds()-start;(void)sink;
        std::cout<<n<<","<<par*1000<<","<<ser*1000<<","<<cpu*1000<<","<<n*320/par<<"\n";
    }
}
static void ppo_tests(Metal& m) {
    using namespace fixed_ppo;
    struct PpoAdamHostConfig { float learning_rate, beta1, beta2, epsilon, weight_decay; uint32_t step; };
    require(run_cpu_self_tests(), "PPO CPU finite-difference/Adam self-test failed");
    auto check = [](const float* actual, const float* expected, size_t count, float tol, const char* label) {
        float e = 0.0f;
        for (size_t i = 0; i < count; ++i) e = std::max(e, std::fabs(actual[i] - expected[i]));
        require(e <= tol, std::string("PPO CPU/Metal mismatch: ") + label + " err=" + std::to_string(e));
        return e;
    };
    auto u32 = [&](uint32_t x) { return m.buffer(sizeof(x), &x); };
    auto f32 = [&](float x) { return m.buffer(sizeof(x), &x); };
    constexpr uint32_t B = 3;
    ActorParams ap;
    CriticParams cp;
    for (size_t i = 0; i < actor_param_count; ++i) ap.values[i] = 0.0015f * std::sin(float(i) * 0.017f);
    for (size_t i = 0; i < critic_param_count; ++i) cp.values[i] = 0.003f * std::cos(float(i) * 0.071f);
    for (size_t a = 0; a < action_dim; ++a) ap.values[actor_log_std_offset + a] = -0.2f + 0.03f * float(a);
    std::vector<float> actor_obs(B * actor_obs_dim), critic_obs(B * critic_obs_dim);
    for (size_t i = 0; i < actor_obs.size(); ++i) actor_obs[i] = 0.2f * std::sin(float(i) * 0.013f);
    for (size_t i = 0; i < critic_obs.size(); ++i) critic_obs[i] = 0.3f * std::cos(float(i) * 0.11f);
    std::vector<float> cpu_hidden(B * hidden_dim), cpu_means(B * action_dim), cpu_chidden(B * hidden_dim), cpu_values(B);
    for (uint32_t n = 0; n < B; ++n) {
        actor_forward(actor_obs.data() + n * actor_obs_dim, ap, cpu_hidden.data() + n * hidden_dim, cpu_means.data() + n * action_dim);
        cpu_values[n] = critic_forward(critic_obs.data() + n * critic_obs_dim, cp, cpu_chidden.data() + n * hidden_dim);
    }
    auto actor_obs_b = m.buffer(actor_obs.size() * 4, actor_obs.data());
    auto critic_obs_b = m.buffer(critic_obs.size() * 4, critic_obs.data());
    auto ap_b = m.buffer(sizeof(ap.values), ap.values.data());
    auto log_std_b = m.buffer(action_dim * 4, ap.values.data() + actor_log_std_offset);
    auto cp_b = m.buffer(sizeof(cp.values), cp.values.data());
    auto ah_b = m.buffer(cpu_hidden.size() * 4), means_b = m.buffer(cpu_means.size() * 4);
    auto ch_b = m.buffer(cpu_chidden.size() * 4), values_b = m.buffer(cpu_values.size() * 4);
    auto batch_b = u32(B);
    auto cb = [m.queue commandBuffer];
    m.dispatch(cb, m.pipeline("ppo_actor_forward"), B, {actor_obs_b, ap_b, ah_b, means_b, batch_b});
    m.dispatch(cb, m.pipeline("ppo_critic_forward"), B, {critic_obs_b, cp_b, ch_b, values_b, batch_b});
    m.finish(cb);
    check((float*)means_b.contents, cpu_means.data(), cpu_means.size(), 2.0e-5f, "actor forward");
    check((float*)ah_b.contents, cpu_hidden.data(), cpu_hidden.size(), 2.0e-5f, "actor hidden");
    constexpr uint32_t FW=256;
    std::vector<float> wide_obs(FW*actor_obs_dim),wide_h(FW*hidden_dim),wide_mean(FW*action_dim);
    for(size_t i=0;i<wide_obs.size();++i)wide_obs[i]=.2f*std::sin(float(i)*.0031f);
    for(uint32_t n=0;n<FW;++n)actor_forward(wide_obs.data()+n*actor_obs_dim,ap,wide_h.data()+n*hidden_dim,wide_mean.data()+n*action_dim);
    auto wide_obs_b=m.buffer(wide_obs.size()*4,wide_obs.data()),wide_h_b=m.buffer(wide_h.size()*4),wide_mean_b=m.buffer(wide_mean.size()*4),wide_batch_b=u32(FW);
    cb=[m.queue commandBuffer];m.dispatch(cb,m.pipeline("ppo_actor_forward_simd_fused"),((FW+7)/8)*256,{wide_obs_b,ap_b,wide_h_b,wide_mean_b,wide_batch_b},256);m.finish(cb);
    check((float*)wide_h_b.contents,wide_h.data(),wide_h.size(),2.0e-5f,"SIMD fused actor hidden 256");
    check((float*)wide_mean_b.contents,wide_mean.data(),wide_mean.size(),2.0e-5f,"SIMD fused actor means 256");
    check((float*)values_b.contents, cpu_values.data(), cpu_values.size(), 2.0e-5f, "critic forward");
    check((float*)ch_b.contents, cpu_chidden.data(), cpu_chidden.size(), 2.0e-5f, "critic hidden");

    constexpr uint32_t T = 3, N = 2;
    std::vector<float> rewards{1, 2, 3, 4, 5, 6}, vals{.1f, .2f, .3f, .4f, .5f, .6f};
    std::vector<float> next{.2f, .3f, .4f, .5f, .6f, .7f}, adv(6), ret(6);
    std::vector<uint8_t> term{0, 0, 1, 0, 0, 0}, trunc{0, 0, 0, 1, 0, 0};
    compute_gae(T, N, rewards.data(), vals.data(), next.data(), term.data(), trunc.data(), .99f, .95f, adv.data(), ret.data());
    auto rb=m.buffer(24,rewards.data()), vb=m.buffer(24,vals.data()), nb=m.buffer(24,next.data());
    auto tb=m.buffer(6,term.data()), xb=m.buffer(6,trunc.data()), ab=m.buffer(24), retb=m.buffer(24);
    auto t_b=u32(T), n_b=u32(N); auto gamma_b=f32(.99f), lambda_b=f32(.95f);
    cb=[m.queue commandBuffer];
    m.dispatch(cb,m.pipeline("ppo_gae"),N,{rb,vb,nb,tb,xb,ab,retb,t_b,n_b,gamma_b,lambda_b});m.finish(cb);
    check((float*)ab.contents,adv.data(),adv.size(),2e-5f,"GAE terminal/truncation");
    check((float*)retb.contents,ret.data(),ret.size(),2e-5f,"GAE returns");

    // The no-op radius must preserve the historical standardized advantages;
    // a finite radius clips only after normalization.
    const std::vector<float> norm_input{-40.0f,-2.0f,-1.0f,0.0f,1.0f,2.0f,10.0f,40.0f};
    auto check_normalization=[&](float radius,const char* label){
        std::vector<float> expected_norm=norm_input;float mean=0.0f;
        for(float value:norm_input)mean+=value;mean/=float(norm_input.size());
        float variance=0.0f;for(float value:norm_input){const float d=value-mean;variance+=d*d;}
        variance/=float(norm_input.size());const float inv=1.0f/std::sqrt(variance+1.0e-8f);
        for(float& value:expected_norm){const float normalized=(value-mean)*inv;value=std::max(-radius,std::min(radius,normalized));}
        auto values=m.buffer(norm_input.size()*sizeof(float),norm_input.data());auto count=u32(uint32_t(norm_input.size()));
        auto epsilon=f32(1.0e-8f),clip_radius=f32(radius);auto command=[m.queue commandBuffer];
        m.dispatch(command,m.pipeline("ppo_normalize_advantages"),1,{values,count,epsilon,clip_radius},1);m.finish(command);
        check((float*)values.contents,expected_norm.data(),expected_norm.size(),2.0e-6f,label);
    };
    check_normalization(std::numeric_limits<float>::max(),"default advantage-normalization compatibility");
    check_normalization(2.0f,"bounded standardized-advantage clip");

    // Verify the training-only anchor kernel applies a restoring gradient to
    // the frozen launch parameters. Zero lambda remains skipped by PPOTrainer.
    std::vector<float> anchor_grad{.1f,-.2f,.3f,-.4f},anchor_params{.6f,.4f,-.1f,.2f},anchor_ref{.2f,.1f,.1f,-.1f};
    std::vector<float> anchor_expected(anchor_grad.size());for(size_t i=0;i<anchor_grad.size();i++)anchor_expected[i]=anchor_grad[i]+.25f*(anchor_params[i]-anchor_ref[i]);
    auto anchor_grad_b=m.buffer(anchor_grad.size()*4,anchor_grad.data()),anchor_params_b=m.buffer(anchor_params.size()*4,anchor_params.data()),anchor_ref_b=m.buffer(anchor_ref.size()*4,anchor_ref.data());
    auto anchor_lambda_b=f32(.25f),anchor_count_b=u32(uint32_t(anchor_grad.size()));
    cb=[m.queue commandBuffer];m.dispatch(cb,m.pipeline("ppo_anchor_grad"),anchor_grad.size(),{anchor_grad_b,anchor_params_b,anchor_ref_b,anchor_lambda_b,anchor_count_b},4);m.finish(cb);
    check((float*)anchor_grad_b.contents,anchor_expected.data(),anchor_expected.size(),1.0e-7f,"actor-anchor gradient");

    std::vector<float> actions(B*action_dim), oldlp(B), oldv(B), advantages{.5f,-.6f,1.0f}, targets{.3f,-.2f,.7f};
    for (size_t i=0;i<actions.size();++i) actions[i]=.15f*std::cos(float(i)*.37f);
    for (uint32_t n=0;n<B;++n) { oldlp[n]=gaussian_log_prob(actions.data()+n*action_dim,cpu_means.data()+n*action_dim,ap.values.data()+actor_log_std_offset)-.04f*(float(n)-1); oldv[n]=cpu_values[n]-.1f; }
    std::vector<float> dmu(B*action_dim), dstd(B*action_dim), dv(B), loss(B*4), cpu_dmu(B*action_dim),cpu_dstd(B*action_dim),cpu_dv(B),cpu_loss(B*4);
    for(uint32_t n=0;n<B;++n){auto l=sample_loss_and_grad(actions.data()+n*action_dim,cpu_means.data()+n*action_dim,ap.values.data()+actor_log_std_offset,oldlp[n],oldv[n],cpu_values[n],advantages[n],targets[n],float(B),.2f,.5f,.01f,cpu_dmu.data()+n*action_dim,cpu_dstd.data()+n*action_dim,&cpu_dv[n]);cpu_loss[n*4]=l.policy;cpu_loss[n*4+1]=l.value;cpu_loss[n*4+2]=l.entropy;cpu_loss[n*4+3]=l.ratio;}
    auto actions_b=m.buffer(actions.size()*4,actions.data()), oldlp_b=m.buffer(oldlp.size()*4,oldlp.data()), oldv_b=m.buffer(oldv.size()*4,oldv.data());
    auto adv_b=m.buffer(advantages.size()*4,advantages.data()), targets_b=m.buffer(targets.size()*4,targets.data());
    auto dmu_b=m.buffer(dmu.size()*4),dstd_b=m.buffer(dstd.size()*4),dv_b=m.buffer(dv.size()*4),loss_b=m.buffer(loss.size()*4);
    auto clip_b=f32(.2f),vc_b=f32(.5f),ec_b=f32(.01f);
    cb=[m.queue commandBuffer];
    m.dispatch(cb,m.pipeline("ppo_sample_loss_grad"),B,{actions_b,means_b,log_std_b,oldlp_b,values_b,oldv_b,adv_b,targets_b,dmu_b,dstd_b,dv_b,loss_b,batch_b,clip_b,vc_b,ec_b});m.finish(cb);
    check((float*)dmu_b.contents,cpu_dmu.data(),cpu_dmu.size(),3e-5f,"policy mean gradient");
    check((float*)dstd_b.contents,cpu_dstd.data(),cpu_dstd.size(),3e-5f,"policy log-std gradient");
    check((float*)dv_b.contents,cpu_dv.data(),cpu_dv.size(),3e-5f,"value gradient");
    check((float*)loss_b.contents,cpu_loss.data(),cpu_loss.size(),3e-5f,"loss metrics");

    std::vector<float> apart(actor_param_count*B), ag(actor_param_count), capart(critic_param_count*B), cg(critic_param_count);
    std::vector<float> cpuapart(actor_param_count*B),cpuag(actor_param_count),cpucapart(critic_param_count*B),cpucg(critic_param_count);
    for(uint32_t n=0;n<B;++n){actor_backward_sample(actor_obs.data()+n*actor_obs_dim,cpu_hidden.data()+n*hidden_dim,cpu_dmu.data()+n*action_dim,cpu_dstd.data()+n*action_dim,ap,cpuapart.data()+n*actor_param_count);critic_backward_sample(critic_obs.data()+n*critic_obs_dim,cpu_chidden.data()+n*hidden_dim,cpu_dv[n],cp,cpucapart.data()+n*critic_param_count);}
    // Convert the CPU reference to parameter-major layout used by Metal.
    for(uint32_t n=0;n<B;++n)for(size_t p=0;p<actor_param_count;++p)apart[p*B+n]=cpuapart[n*actor_param_count+p];
    for(uint32_t n=0;n<B;++n)for(size_t p=0;p<critic_param_count;++p)capart[p*B+n]=cpucapart[n*critic_param_count+p];
    reduce_sample_grads<actor_param_count>(cpuapart.data(),B,cpuag.data());
    reduce_sample_grads<critic_param_count>(cpucapart.data(),B,cpucg.data());
    auto apart_b=m.buffer(apart.size()*4,apart.data()), ag_b=m.buffer(ag.size()*4), capart_b=m.buffer(capart.size()*4,capart.data()),cg_b=m.buffer(cg.size()*4);
    auto apcount_b=u32(actor_param_count), cpcount_b=u32(critic_param_count);
    cb=[m.queue commandBuffer];
    m.dispatch(cb,m.pipeline("ppo_actor_backward"),B,{actor_obs_b,ah_b,ap_b,dmu_b,dstd_b,apart_b,batch_b});
    m.dispatch(cb,m.pipeline("ppo_critic_backward"),B,{critic_obs_b,ch_b,cp_b,dv_b,capart_b,batch_b});
    m.dispatch(cb,m.pipeline("ppo_reduce_grads"),actor_param_count,{apart_b,ag_b,apcount_b,batch_b});
    m.dispatch(cb,m.pipeline("ppo_reduce_grads"),critic_param_count,{capart_b,cg_b,cpcount_b,batch_b});m.finish(cb);
    check((float*)ag_b.contents,cpuag.data(),cpuag.size(),3e-5f,"actor backprop/reduction");
    check((float*)cg_b.contents,cpucg.data(),cpucg.size(),3e-5f,"critic backprop/reduction");
    auto adelta_b=m.buffer(B*hidden_dim*4),cdelta_b=m.buffer(B*hidden_dim*4),direct_ag=m.buffer(actor_param_count*4),direct_cg=m.buffer(critic_param_count*4);
    cb=[m.queue commandBuffer];
    m.dispatch(cb,m.pipeline("ppo_actor_hidden_delta"),B*hidden_dim,{ap_b,dmu_b,ah_b,adelta_b,batch_b});
    m.dispatch(cb,m.pipeline("ppo_actor_grad_direct"),actor_param_count,{actor_obs_b,ah_b,dmu_b,dstd_b,adelta_b,direct_ag,batch_b});
    m.dispatch(cb,m.pipeline("ppo_critic_hidden_delta"),B*hidden_dim,{cp_b,dv_b,ch_b,cdelta_b,batch_b});
    m.dispatch(cb,m.pipeline("ppo_critic_grad_direct"),critic_param_count,{critic_obs_b,ch_b,dv_b,cdelta_b,direct_cg,batch_b});m.finish(cb);
    check((float*)direct_ag.contents,cpuag.data(),cpuag.size(),3e-5f,"direct actor gradient");
    check((float*)direct_cg.contents,cpucg.data(),cpucg.size(),3e-5f,"direct critic gradient");

    std::vector<float> m1(actor_param_count,0),v1(actor_param_count,0), cpu_m1(actor_param_count,0),cpu_v1(actor_param_count,0);
    ActorParams cpu_ap=ap;
    PpoAdamHostConfig cfg{.003f,.9f,.999f,1e-8f,0.0f,1};
    adam_update(cpu_ap.values.data(),cpuag.data(),cpu_m1.data(),cpu_v1.data(),actor_param_count,cfg.step,cfg.learning_rate,cfg.beta1,cfg.beta2,cfg.epsilon,cfg.weight_decay);
    auto m1_b=m.buffer(m1.size()*4,m1.data()),v1_b=m.buffer(v1.size()*4,v1.data()),cfg_b=m.buffer(sizeof(cfg),&cfg);
    cb=[m.queue commandBuffer];m.dispatch(cb,m.pipeline("ppo_adam_update"),actor_param_count,{ap_b,ag_b,m1_b,v1_b,apcount_b,cfg_b});m.finish(cb);
    check((float*)ap_b.contents,cpu_ap.values.data(),actor_param_count,3e-5f,"Adam parameters");
    check((float*)m1_b.contents,cpu_m1.data(),actor_param_count,2e-6f,"Adam first moment");
    check((float*)v1_b.contents,cpu_v1.data(),actor_param_count,2e-6f,"Adam second moment");

    // Repeated GPU PPO updates on a zero observation and fixed target action.
    ActorParams toy{};for(size_t a=0;a<action_dim;++a)toy.values[actor_log_std_offset+a]=-.5f;
    std::vector<float> zero_obs(B*actor_obs_dim,0), target_action(action_dim);
    target_action[0]=.35f;target_action[1]=-.2f;target_action[2]=.15f;target_action[3]=.4f;
    std::vector<float> toy_actions(B*action_dim), toy_adv(B,1), toy_returns(B,0), toy_values(B,0), toy_oldv(B,0);
    for(uint32_t n=0;n<B;++n)std::copy(target_action.begin(),target_action.end(),toy_actions.begin()+n*action_dim);
    auto zobs_b=m.buffer(zero_obs.size()*4,zero_obs.data()),toy_ap_b=m.buffer(sizeof(toy.values),toy.values.data());
    auto toy_hidden_b=m.buffer(B*hidden_dim*4),toy_means_b=m.buffer(B*action_dim*4);
    auto toy_actions_b=m.buffer(toy_actions.size()*4,toy_actions.data()), toy_values_b=m.buffer(B*4,toy_values.data());
    auto toy_log_std_b=m.buffer(action_dim*4,toy.values.data()+actor_log_std_offset);
    auto toy_oldv_b=m.buffer(B*4,toy_oldv.data()),toy_adv_b=m.buffer(B*4,toy_adv.data()),toy_ret_b=m.buffer(B*4,toy_returns.data());
    auto toy_dmu_b=m.buffer(B*action_dim*4),toy_dstd_b=m.buffer(B*action_dim*4),toy_dv_b=m.buffer(B*4),toy_loss_b=m.buffer(B*4*4);
    std::vector<float> toy_oldlp(B),toy_part(actor_param_count*B),toy_grad(actor_param_count),toy_m(actor_param_count,0),toy_v(actor_param_count,0);
    auto toy_oldlp_b=m.buffer(B*4),toy_part_b=m.buffer(toy_part.size()*4),toy_grad_b=m.buffer(toy_grad.size()*4),toy_m_b=m.buffer(toy_m.size()*4,toy_m.data()),toy_v_b=m.buffer(toy_v.size()*4,toy_v.data());
    auto toy_params_count_b=u32(actor_param_count);
    auto toy_optimizer=PpoAdamHostConfig{.025f,.9f,.999f,1e-8f,0.0f,1};
    auto toy_opt_b=m.buffer(sizeof(toy_optimizer),&toy_optimizer);
    auto toy_error=[&]() { float e=0;for(size_t a=0;a<action_dim;++a){float d=((float*)toy_means_b.contents)[a]-target_action[a];e+=d*d;}return std::sqrt(e); };
    cb=[m.queue commandBuffer];m.dispatch(cb,m.pipeline("ppo_actor_forward"),B,{zobs_b,toy_ap_b,toy_hidden_b,toy_means_b,batch_b});m.finish(cb);
    const float start_error=toy_error();
    for(uint32_t step=1;step<=36;++step){
        const float* means=(float*)toy_means_b.contents;
        std::memcpy(toy_log_std_b.contents,(const float*)toy_ap_b.contents+actor_log_std_offset,action_dim*4);
        for(uint32_t n=0;n<B;++n)toy_oldlp[n]=gaussian_log_prob(toy_actions.data()+n*action_dim,means+n*action_dim,(const float*)toy_ap_b.contents+actor_log_std_offset);
        std::memcpy(toy_oldlp_b.contents,toy_oldlp.data(),B*4);
        toy_optimizer.step=step;std::memcpy(toy_opt_b.contents,&toy_optimizer,sizeof(toy_optimizer));
        cb=[m.queue commandBuffer];
        m.dispatch(cb,m.pipeline("ppo_sample_loss_grad"),B,{toy_actions_b,toy_means_b,toy_log_std_b,toy_oldlp_b,toy_values_b,toy_oldv_b,toy_adv_b,toy_ret_b,toy_dmu_b,toy_dstd_b,toy_dv_b,toy_loss_b,batch_b,clip_b,vc_b,ec_b});
        m.dispatch(cb,m.pipeline("ppo_actor_backward"),B,{zobs_b,toy_hidden_b,toy_ap_b,toy_dmu_b,toy_dstd_b,toy_part_b,batch_b});
        m.dispatch(cb,m.pipeline("ppo_reduce_grads"),actor_param_count,{toy_part_b,toy_grad_b,toy_params_count_b,batch_b});
        m.dispatch(cb,m.pipeline("ppo_adam_update"),actor_param_count,{toy_ap_b,toy_grad_b,toy_m_b,toy_v_b,toy_params_count_b,toy_opt_b});
        m.dispatch(cb,m.pipeline("ppo_actor_forward"),B,{zobs_b,toy_ap_b,toy_hidden_b,toy_means_b,batch_b});m.finish(cb);
        std::memcpy(toy_log_std_b.contents,(float*)toy_ap_b.contents+actor_log_std_offset,action_dim*4);
    }
    const float final_error=toy_error();
    require(final_error < start_error * .45f,"Metal PPO toy task did not learn target action: start="+std::to_string(start_error)+" final="+std::to_string(final_error));
    std::cout<<"ppo PASS CPU/Metal forward, terminal+truncation GAE, default/clipped advantage normalization, anchor gradient, PPO gradients, reduction, Adam; toy target error "<<start_error<<" -> "<<final_error<<"\n";
}

static void geodesic_direction_gradient_test(Metal& metal) {
    if(fixed_ppo::actor_obs_dim!=184)return;
    std::array<float,184> obs{};
    for(uint32_t ray=0;ray<20;ray++)obs[ray]=0.1f; // gate floor=.25
    obs[181]=0.24f;obs[182]=-0.16f;obs[183]=0.08f;
    std::array<float,4> means{{0.7f,-0.35f,0.2f,-0.1f}},gradient{};
    std::array<float,16> target{};target[0]=-0.35f;target[1]=0.93f;target[2]=0.11f;
    const float target_norm=std::sqrt(target[0]*target[0]+target[1]*target[1]+target[2]*target[2]);
    for(uint32_t axis=0;axis<3;axis++)target[axis]/=target_norm;
    target[3]=1.0f;
    constexpr float coefficient=0.2f,epsilon=1.0e-3f;
    auto ob=metal.buffer(sizeof(obs),obs.data()),mb=metal.buffer(sizeof(means),means.data());
    auto tb=metal.buffer(sizeof(target),target.data()),gb=metal.buffer(sizeof(gradient),gradient.data());
    const uint32_t batch=1;auto bb=metal.buffer(sizeof(batch),&batch);auto cbcoef=metal.buffer(sizeof(coefficient),&coefficient);
    auto commands=[metal.queue commandBuffer];
    metal.dispatch(commands,metal.pipeline("ppo_actor_geodesic_direction_grad"),1,
                   {ob,mb,tb,gb,bb,cbcoef},1);metal.finish(commands);
    std::memcpy(gradient.data(),gb.contents,sizeof(gradient));
    const auto loss=[&](const std::array<float,4>& candidate){
        constexpr float gate=.25f;
        float q[3],norm2=0,dot=0;
        for(uint32_t axis=0;axis<3;axis++){
            const float z=obs[181+axis]+gate*(candidate[axis]-obs[181+axis]);
            q[axis]=std::tanh(z);norm2+=q[axis]*q[axis];dot+=q[axis]*target[axis];
        }
        return coefficient*(1.0f-dot/std::sqrt(norm2));
    };
    float max_error=0.0f;
    for(uint32_t axis=0;axis<3;axis++){
        auto plus=means,minus=means;plus[axis]+=epsilon;minus[axis]-=epsilon;
        const float finite_difference=(loss(plus)-loss(minus))/(2.0f*epsilon);
        max_error=std::max(max_error,std::fabs(finite_difference-gradient[axis]));
    }
    require(std::fabs(gradient[3])<1.0e-8f && max_error<2.0e-5f,
            "Metal geodesic direction gradient differs from finite differences");
    std::cout<<"geodesic direction gradient PASS finite_difference_max_error="<<max_error<<" yaw_gradient="<<gradient[3]<<"\n";
}

struct SimRun {
    float hidden[16],motors[4],previous_nav[4],desired_velocity[3],yaw;
    uint32_t rng,steps;
    float path,elapsed,peak_speed,min_clearance,initial_distance;
    uint32_t successes,collisions,timeouts,episodes;
    float success_time,total_path,total_elapsed,final_progress,reference_position[3];
};
struct SimConfig {
    uint32_t n=128,tick=0,mode=0,family=0,substeps=5,sensor_period=1,sensor_delay=0,command_delay=0,max_steps=200,seed=42,eval=0;
    float speed=2,distance=4,wind=0,depth_noise=0,dropout=0,risk_coef=0,entropy_coef=.005f,learning_rate=.0003f;uint32_t velocity_contract=1,geometry_memory=0;
};
static_assert(sizeof(SimRun)==184 && sizeof(SimConfig)==84,"sim layout mismatch");
static_assert(fixed_ppo::actor_obs_dim==181||fixed_ppo::actor_obs_dim==184||
              fixed_ppo::actor_obs_dim==661||fixed_ppo::actor_obs_dim==824,
              "navigation actor dimension must match a supported fixed observation contract");
using BufferBinding=std::pair<id<MTLBuffer>,size_t>;
[[maybe_unused]] static void encode(Metal& m,id<MTLCommandBuffer> cb,const char* name,size_t count,std::initializer_list<BufferBinding> bindings,size_t group=128) {
    auto p=m.pipeline(name);auto e=[cb computeCommandEncoder];[e setComputePipelineState:p];uint j=0;for(auto binding:bindings)[e setBuffer:binding.first offset:binding.second atIndex:j++];
    [e dispatchThreads:MTLSizeMake(count,1,1) threadsPerThreadgroup:MTLSizeMake(std::min(group,size_t(p.maxTotalThreadsPerThreadgroup)),1,1)];[e endEncoding];
}
// Append after SimRun, SimConfig and load_raptor() are declared. This command
// runs only the CPU path; it does not create or dispatch Metal command buffers.
#include "cpu_reference.hpp"

static cpu_reference::Metrics cpu_reference_benchmark(uint32_t rollouts=1,
                                                       uint32_t family=0,
                                                       uint32_t env_count=128,
                                                       uint32_t horizon=32,
                                                       float speed=1.0f,
                                                       float distance=3.0f) {
    SimConfig cfg;cfg.n=env_count;cfg.family=family;cfg.mode=0;cfg.seed=42;
    cfg.substeps=5;cfg.sensor_period=1;cfg.sensor_delay=0;cfg.command_delay=0;
    cfg.max_steps=200;cfg.eval=0;cfg.speed=speed;cfg.distance=distance;
    cfg.wind=0;cfg.depth_noise=0;cfg.dropout=0;
    const RaptorWeights raptor=load_raptor();
    const RLPhysicsParams physics=rl_physics_crazyflie_default();
    cpu_reference::Trainer trainer(cfg,horizon,raptor,physics);
    require(trainer.validate_batched_forward(),"CPU Accelerate forward differs from scalar reference");
    const auto result=trainer.run(rollouts);
    const double total_episodes=double(result.episodes);
    std::cout<<"CPU reference device=Accelerate/CPU family="<<family<<" envs="<<env_count
             <<" horizon="<<horizon<<" rollouts="<<rollouts<<" transitions="<<uint64_t(env_count)*horizon*rollouts
             <<" native_ticks="<<uint64_t(env_count)*horizon*rollouts*cfg.substeps
             <<" wall_s="<<result.wall_seconds<<" steps_per_s="
             <<(result.wall_seconds?double(env_count)*horizon*rollouts/result.wall_seconds:0)
             <<" native_steps_per_s="<<(result.wall_seconds?double(env_count)*horizon*rollouts*cfg.substeps/result.wall_seconds:0)
             <<" optimizer_step="<<result.updates<<" episodes="<<result.episodes
             <<" success="<<(total_episodes?double(result.successes)/total_episodes:0)
             <<" collision="<<(total_episodes?double(result.collisions)/total_episodes:0)
             <<" timeout="<<(total_episodes?double(result.timeouts)/total_episodes:0)
             <<" progress="<<(total_episodes?result.final_progress/total_episodes:0)
             <<" mean_goal_time="<<(result.successes?result.success_time/result.successes:0)
             <<" policy_loss="<<result.policy_loss<<" value_loss="<<result.value_loss
             <<" entropy_loss="<<result.entropy_loss<<" ratio="<<result.ratio<<"\n";
    return result;
}

struct ChallengeBankControl {
    uint32_t enabled=0, bank_count=0, schedule_stride=0, horizon=0;
};
static_assert(sizeof(ChallengeBankControl)==16,"challenge bank control layout");

struct Sim {
    Metal& m;SimConfig cfg;uint32_t horizon;
    id<MTLBuffer> states,runs,worlds,sensors,poses,commands,raptor,physics,actor,critic,obs,co,actions,logp,values,rewards,next_values,terminated,truncated,advantages,returns;
    std::vector<id<MTLBuffer>> configs;
    id<MTLComputePipelineState> reset_p,depth_p,observe_p,act_p,advance_p,actor_p;
    id<MTLBuffer> actor_workspace,env_count,memory_points,memory_clearances;
    id<MTLBuffer> bank_worlds,bank_schedule,bank_control,bank_active_ids,bank_transition_ids;
    id<MTLBuffer> environment_physics,runtime_control,task_states,task_control;
    id<MTLBuffer> potential_fields,potential_spec,potential_control,potential_hash;
    id<MTLBuffer> direction_aux_targets=nil,direction_aux_level_ids=nil;
    id<MTLComputePipelineState> direction_aux_target_p=nil;
    bool direction_aux_enabled=false;
    id<MTLComputePipelineState> memory_points_p,memory_candidates_p;
    bool simd_actor=true;
    Sim(Metal& metal,SimConfig config,uint32_t steps):m(metal),cfg(config),horizon(steps) {
        auto w=load_raptor();auto p=rl_physics_crazyflie_default();uint n=cfg.n;size_t rows=n*size_t(horizon);
        NavigationRuntimeConfig runtime{};runtime.domain_range=rl_physics_domain_stress_range();
        NavigationTaskControl tasks{};
        TrainingPotentialControl shaping{};TrainingPotentialGridSpec grid{};
        potential_fields=m.buffer(4);potential_spec=m.buffer(sizeof(grid),&grid);potential_control=m.buffer(sizeof(shaping),&shaping);potential_hash=m.buffer(64);
        std::memset(potential_hash.contents,0,64);
        environment_physics=m.buffer(n*sizeof(RLPhysicsParams));runtime_control=m.buffer(sizeof(runtime),&runtime);
        task_states=m.buffer(n*sizeof(NavigationTaskState));task_control=m.buffer(sizeof(tasks),&tasks);
        states=m.buffer(n*sizeof(RLPhysicsState));runs=m.buffer(n*sizeof(SimRun));worlds=m.buffer(n*sizeof(WWorld));sensors=m.buffer(n*8*320*4);poses=m.buffer(n*8*12*4);commands=m.buffer(n*8*4*4);raptor=m.buffer(sizeof(w),&w);physics=m.buffer(sizeof(p),&p);
        // Disabled bank buffers keep ordinary procedural training unchanged.
        // Bank training uploads a fixed table and a complete rollout reset schedule.
        ChallengeBankControl bank_defaults;
        bank_worlds=m.buffer(sizeof(WWorld));
        bank_schedule=m.buffer(size_t(n)*(horizon+1)*sizeof(uint32_t));
        bank_control=m.buffer(sizeof(bank_defaults),&bank_defaults);
        bank_active_ids=m.buffer(size_t(n)*sizeof(uint32_t));
        bank_transition_ids=m.buffer(rows*sizeof(uint32_t));
        fixed_ppo::ActorParams a;fixed_ppo::CriticParams c;uint seed=1789;
        auto normal=[&](){float u=std::max(wurand(seed),1e-7f);return std::sqrt(-2*std::log(u))*std::cos(6.2831853f*wurand(seed));};
        for(size_t i=0;i<fixed_ppo::actor_b1_offset;i++)a.values[i]=normal()*0.04f;
        for(size_t i=fixed_ppo::actor_w2_offset;i<fixed_ppo::actor_b2_offset;i++)a.values[i]=normal()*0.01f;
        for(size_t i=0;i<4;i++)a.values[fixed_ppo::actor_log_std_offset+i]=-1.0f;
        for(size_t i=0;i<fixed_ppo::critic_b1_offset;i++)c.values[i]=normal()*0.15f;
        for(size_t i=fixed_ppo::critic_w2_offset;i<fixed_ppo::critic_b2_offset;i++)c.values[i]=normal()*0.1f;
        actor=m.buffer(sizeof(a),&a);critic=m.buffer(sizeof(c),&c);obs=m.buffer(rows*fixed_ppo::actor_obs_dim*4);co=m.buffer(rows*fixed_ppo::critic_obs_dim*4);actions=m.buffer(rows*4*4);logp=m.buffer(rows*4);values=m.buffer(rows*4);rewards=m.buffer(rows*4);next_values=m.buffer(rows*4);terminated=m.buffer(rows);truncated=m.buffer(rows);advantages=m.buffer(rows*4);returns=m.buffer(rows*4);
        for(uint t=0;t<horizon;t++){cfg.tick=t;configs.push_back(m.buffer(sizeof(cfg),&cfg));}cfg.tick=0;
        simd_actor=!(std::getenv("METAL_NAV_SCALAR_ACTOR") && std::string(std::getenv("METAL_NAV_SCALAR_ACTOR"))=="1");
        actor_p=m.pipeline(simd_actor?"ppo_actor_forward_simd_fused":"ppo_actor_forward");
        require(!simd_actor || (fixed_ppo::hidden_dim==64 && actor_p.threadExecutionWidth==32 && actor_p.maxTotalThreadsPerThreadgroup>=256),"SIMD actor requires64hidden/M3-style32lane/256threadgroups");
        actor_workspace=m.buffer(cfg.n*fixed_ppo::hidden_dim*4);env_count=m.buffer(sizeof(cfg.n),&cfg.n);
        memory_points=m.buffer(size_t(n)*640*4*4);memory_clearances=m.buffer(size_t(n)*85*4);memory_points_p=m.pipeline("nav_memory_build_points");memory_candidates_p=m.pipeline("nav_memory_candidate_clearance");
        reset_p=m.pipeline("sim_reset");depth_p=m.pipeline("sim_depth");observe_p=m.pipeline("sim_observe");act_p=m.pipeline("sim_act");advance_p=m.pipeline("sim_advance");reset();
    }
    void set_config(SimConfig config){require(config.n==cfg.n,"cannot resize simulator");cfg=config;for(uint t=0;t<horizon;t++){cfg.tick=t;std::memcpy(configs[t].contents,&cfg,sizeof(cfg));}cfg.tick=0;}
    void reset(){auto cb=[m.queue commandBuffer];m.dispatch(cb,reset_p,cfg.n,{states,runs,worlds,sensors,commands,raptor,physics,configs[0],poses,bank_worlds,bank_schedule,bank_control,bank_active_ids,environment_physics,runtime_control,task_states,task_control},64);m.finish(cb);}
    void enable_direction_auxiliary(){
        require(fixed_ppo::actor_obs_dim==184,"geodesic direction auxiliary requires the 184D guided actor");
        direction_aux_targets=m.buffer(size_t(cfg.n)*horizon*16*sizeof(float));
        direction_aux_level_ids=m.buffer(size_t(cfg.n)*horizon*sizeof(uint32_t));
        direction_aux_target_p=m.pipeline("ppo_geodesic_direction_target");
        direction_aux_enabled=true;
    }
    void collect(id<MTLCommandBuffer> cb,uint count=0){
        if(count==0)count=horizon;
        for(uint t=0;t<count;t++){
            auto c=configs[t%horizon];
            m.dispatch(cb,depth_p,cfg.n*320,{states,runs,worlds,sensors,physics,c,poses});
            if(cfg.geometry_memory){
                m.dispatch(cb,memory_points_p,cfg.n*640,{states,runs,sensors,poses,memory_points,physics,c});
                m.dispatch(cb,memory_candidates_p,cfg.n*85,{states,worlds,memory_points,memory_clearances,c});
            }
            m.dispatch(cb,observe_p,cfg.n,{states,runs,worlds,sensors,obs,co,physics,c,poses,memory_clearances},64);
            const size_t row_offset=size_t(t%horizon)*cfg.n;
            encode(m,cb,simd_actor?"ppo_actor_forward_simd_fused":"ppo_actor_forward",
                   simd_actor?((size_t(cfg.n)+7)/8)*256:cfg.n,
                   {{obs,row_offset*fixed_ppo::actor_obs_dim*4},{actor,0},{actor_workspace,0},
                    {actions,row_offset*fixed_ppo::action_dim*4},{env_count,0}},simd_actor?256:64);
            m.dispatch(cb,act_p,cfg.n,{states,runs,worlds,obs,co,actor,critic,actions,logp,values,commands,physics,c},64);
            if(direction_aux_enabled){
                const size_t target_offset=row_offset*16*sizeof(float);
                const size_t id_offset=row_offset*sizeof(uint32_t);
                encode(m,cb,"ppo_geodesic_direction_target",cfg.n,
                    {{states,0},{worlds,0},{bank_active_ids,0},{potential_fields,0},{potential_spec,0},
                     {direction_aux_targets,target_offset},{direction_aux_level_ids,id_offset},{env_count,0}},64);
            }
            m.dispatch(cb,advance_p,cfg.n,{states,runs,worlds,sensors,commands,raptor,critic,rewards,next_values,terminated,truncated,physics,c,bank_worlds,bank_schedule,bank_control,bank_active_ids,bank_transition_ids,environment_physics,runtime_control,task_states,task_control,potential_fields,potential_spec,potential_control},64);
        }
    }
    void report(const std::string& label,double wall){const SimRun* r=(const SimRun*)runs.contents;uint64_t success=0,collision=0,timeout=0,ep=0;double time=0,path=0,total=0,progress=0;float clearance=12,peak=0;
        for(uint i=0;i<cfg.n;i++){success+=r[i].successes;collision+=r[i].collisions;timeout+=r[i].timeouts;ep+=r[i].episodes;time+=r[i].success_time;path+=r[i].total_path;total+=r[i].total_elapsed;progress+=r[i].final_progress;clearance=fmin(clearance,r[i].min_clearance);peak=fmax(peak,r[i].peak_speed);}
        std::cout<<label<<" episodes="<<ep<<" success="<<(ep?double(success)/ep:0)<<" collision="<<(ep?double(collision)/ep:0)<<" timeout="<<(ep?double(timeout)/ep:0)<<" progress="<<(ep?progress/ep:0)<<" mean_goal_time="<<(success?time/success:0)<<" mean_speed="<<(total?path/total:0)<<" peak_speed="<<peak<<" min_clearance="<<clearance<<" wall_s="<<wall<<"\n";
    }
};
struct EvalScore { double success=0,collision=0,timeout=1,goal_time=1e9; };
static EvalScore sim_evaluate(Metal& m,uint family,uint mode,const float* actor=nullptr,float speed=1,float distance=3,uint32_t seed=700001,uint32_t sensor_delay=0,float wind=0,float noise=0,float dropout=0,uint32_t command_delay=0,uint32_t velocity_contract=1,uint32_t memory=0,uint32_t max_steps=0) {
    require((family!=10 && family!=11 && family!=13) || (std::fabs(speed-1.5f)<1e-6f && std::fabs(distance-4.0f)<1e-6f),"flying-threat evaluation requires1.5m/s and4m goal");
    require(max_steps==0 || (max_steps>=1 && max_steps<=2000),"evaluation episode length unsupported");
    SimConfig cfg;cfg.n=128;cfg.family=family;cfg.mode=mode;cfg.eval=1;cfg.seed=seed;cfg.distance=distance;cfg.speed=speed;cfg.sensor_delay=sensor_delay;cfg.wind=wind;cfg.depth_noise=noise;cfg.dropout=dropout;cfg.command_delay=command_delay;cfg.velocity_contract=velocity_contract;cfg.geometry_memory=memory||mode>=13;cfg.max_steps=max_steps?max_steps:((family>=14 && family<=16)?400:200);require(sensor_delay<=6 && command_delay<=7,"delay rings support <=6 sensor frames and <=7 command steps");
    if(mode==18||mode==19){require(actor!=nullptr&&(fixed_ppo::actor_obs_dim==184||fixed_ppo::actor_obs_dim==824),"modes 18/19 require a guided checkpoint");std::cout<<"EVAL_ABLATION mode="<<mode<<(mode==18?" current-depth-only, newest-frame geometry memory":" fixed-speed, mode-17 guided direction and yaw")<<"; same checkpoint versus mode17; inference ablation, not a matched retrain\n";}
    std::cout<<"eval_budget_s="<<cfg.max_steps*.05f<<" speed_intent_cap="<<cfg.speed<<"\n";
    Sim sim(m,cfg,32);if(actor)std::memcpy(sim.actor.contents,actor,fixed_ppo::actor_param_count*4);
    double start=seconds();auto cb=[m.queue commandBuffer];sim.collect(cb,cfg.max_steps);double gpu=m.finish(cb);sim.report("eval family="+std::to_string(family)+" mode="+std::to_string(mode),seconds()-start);std::cout<<"eval_GPU_s="<<gpu<<"\n";
    const auto* r=(const SimRun*)sim.runs.contents;double success=0,collision=0,timeout=0,time=0;for(uint i=0;i<cfg.n;i++){require(r[i].episodes==1,"eval must finish exactly one episode per seed");success+=r[i].successes;collision+=r[i].collisions;timeout+=r[i].timeouts;time+=r[i].success_time;}return {success/cfg.n,collision/cfg.n,timeout/cfg.n,success?time/success:1e9};
}

static void mixed_domain_tests(Metal& m) {
    float worst=0;
    for(uint32_t family:{12u,13u})for(uint32_t eval:{0u,1u}) {
        SimConfig cfg;cfg.n=16;cfg.family=family;cfg.eval=eval;cfg.speed=1.5f;cfg.distance=4;cfg.geometry_memory=1;cfg.sensor_delay=2;cfg.command_delay=1;cfg.wind=.5f;cfg.depth_noise=.05f;cfg.dropout=.1f;
        Sim sim(m,cfg,1);cpu_reference::Trainer cpu(cfg,1,load_raptor(),rl_physics_crazyflie_default());
        auto cb=[m.queue commandBuffer];auto c=sim.configs[0];
        m.dispatch(cb,sim.depth_p,cfg.n*320,{sim.states,sim.runs,sim.worlds,sim.sensors,sim.physics,c,sim.poses});
        m.dispatch(cb,sim.memory_points_p,cfg.n*640,{sim.states,sim.runs,sim.sensors,sim.poses,sim.memory_points,sim.physics,c});
        m.dispatch(cb,sim.memory_candidates_p,cfg.n*85,{sim.states,sim.worlds,sim.memory_points,sim.memory_clearances,c});
        m.dispatch(cb,sim.observe_p,cfg.n,{sim.states,sim.runs,sim.worlds,sim.sensors,sim.obs,sim.co,sim.physics,c,sim.poses,sim.memory_clearances},64);m.finish(cb);
        const auto* worlds=(const WWorld*)sim.worlds.contents;const auto* states=(const RLPhysicsState*)sim.states.contents;const float* observations=(const float*)sim.obs.contents;
        const auto cpu_initial_states=cpu.states;const auto cpu_initial_worlds=cpu.worlds;
        cpu.run(1); // The horizon1 rollout retains the pre-advance observation row.
        for(uint32_t n=0;n<cfg.n;n++) {
            const float expected_wind=(eval==0 && n%2==0)?0:cfg.wind;
            require(worlds[n].wind[0]==expected_wind && cpu_initial_worlds[n].wind[0]==expected_wind,"mixed-domain wind selection");
            const float expected_v=(worlds[n].family==10||worlds[n].family==11)?cfg.speed:0;
            require(states[n].linear_velocity[0]==expected_v && cpu_initial_states[n].linear_velocity[0]==expected_v,"flying-threat reset velocity");
            for(uint32_t k=0;k<fixed_ppo::actor_obs_dim;k++)worst=std::max(worst,std::fabs(observations[n*fixed_ppo::actor_obs_dim+k]-cpu.observations[n*fixed_ppo::actor_obs_dim+k]));
        }
    }
    require(worst<3e-5f,"mixed-domain CPU/GPU observations differ");
    std::cout<<"mixed domains PASS clean/stress and flying reset CPU/GPU observation_error="<<worst<<"\n";
}

// Camera-profile fixture. Verifies that (a) the legacy ray helpers are
// bit-identical to the frozen world.hpp model, (b) the shared header and the
// guidance grid agree at both profiles, and (c) the Metal depth/observation
// path matches the CPU reference at the calibrated profile. Emits the ray
// table used by the native-fixture comparison.
struct ProfileParity { float sensor_error=0,observation_error=0; };
static ProfileParity profile_parity(Metal& m) {
    SimConfig cfg;cfg.n=16;cfg.family=12;cfg.eval=0;cfg.mode=17;
    cfg.speed=1.5f;cfg.distance=4;cfg.geometry_memory=1;cfg.sensor_delay=0;cfg.command_delay=0;
    Sim sim(m,cfg,1);cpu_reference::Trainer cpu(cfg,1,load_raptor(),rl_physics_crazyflie_default());
    auto cb=[m.queue commandBuffer];auto c=sim.configs[0];
    m.dispatch(cb,sim.depth_p,cfg.n*320,{sim.states,sim.runs,sim.worlds,sim.sensors,sim.physics,c,sim.poses});
    m.dispatch(cb,sim.memory_points_p,cfg.n*640,{sim.states,sim.runs,sim.sensors,sim.poses,sim.memory_points,sim.physics,c});
    m.dispatch(cb,sim.memory_candidates_p,cfg.n*85,{sim.states,sim.worlds,sim.memory_points,sim.memory_clearances,c});
    m.dispatch(cb,sim.observe_p,cfg.n,{sim.states,sim.runs,sim.worlds,sim.sensors,sim.obs,sim.co,sim.physics,c,sim.poses,sim.memory_clearances},64);m.finish(cb);
    cpu.run(1);
    ProfileParity result;
    const float* gpu_sensors=static_cast<const float*>(sim.sensors.contents);
    for(size_t i=0;i<cpu.sensors.size();i++)
        result.sensor_error=std::max(result.sensor_error,std::fabs(gpu_sensors[i]-cpu.sensors[i]));
    const float* gpu_obs=static_cast<const float*>(sim.obs.contents);
    for(size_t i=0;i<cpu.observations.size();i++)
        result.observation_error=std::max(result.observation_error,std::fabs(gpu_obs[i]-cpu.observations[i]));
    return result;
}

// Geometry-only four-frame pose fixture, not a flight or learning result.
// Test the optimized cache directly: guidance output alone can miss a wrong
// vertical projection when its selected command is insensitive to that point.
static void sensor_memory_cache_profile_check(Metal& metal) {
    SimConfig config;config.n=4;config.family=12;config.eval=1;config.mode=17;
    config.geometry_memory=1;config.sensor_period=1;
    Sim sim(metal,config,1);
    auto* states=static_cast<RLPhysicsState*>(sim.states.contents);
    auto* runs=static_cast<SimRun*>(sim.runs.contents);
    auto* worlds=static_cast<WWorld*>(sim.worlds.contents);
    for(uint tick=0;tick<4;tick++) {
        for(uint n=0;n<config.n;n++) {
            const float yaw=.17f*float(tick+n);
            states[n].position[0]=.1f*tick;states[n].position[1]=.13f*tick;states[n].position[2]=1.5f+.03f*tick;
            states[n].orientation_wxyz[0]=std::cos(yaw/2);states[n].orientation_wxyz[1]=states[n].orientation_wxyz[2]=0;states[n].orientation_wxyz[3]=std::sin(yaw/2);
            states[n].linear_velocity[0]=.3f;states[n].linear_velocity[1]=.2f;states[n].linear_velocity[2]=.1f;
            runs[n].steps=tick;runs[n].elapsed=.05f*tick;
        }
        auto cb=[metal.queue commandBuffer];
        metal.dispatch(cb,sim.depth_p,config.n*320,{sim.states,sim.runs,sim.worlds,sim.sensors,sim.physics,sim.configs[0],sim.poses});
        metal.finish(cb);
    }
    auto cb=[metal.queue commandBuffer];
    metal.dispatch(cb,sim.memory_points_p,config.n*640,{sim.states,sim.runs,sim.sensors,sim.poses,sim.memory_points,sim.physics,sim.configs[0]});
    metal.dispatch(cb,sim.memory_candidates_p,config.n*85,{sim.states,sim.worlds,sim.memory_points,sim.memory_clearances,sim.configs[0]});
    metal.finish(cb);
    const float* ranges=static_cast<const float*>(sim.sensors.contents);
    const float* poses=static_cast<const float*>(sim.poses.contents);
    const float* points=static_cast<const float*>(sim.memory_points.contents);
    const float* clearances=static_cast<const float*>(sim.memory_clearances.contents);
    float point_error=0,clearance_error=0;
    for(uint n=0;n<config.n;n++) {
        float rotation[9];cpu_reference::rotation(states[n].orientation_wxyz,rotation);
        float body_pose[12];for(uint j=0;j<3;j++)body_pose[j]=states[n].position[j];for(uint j=0;j<9;j++)body_pose[j+3]=rotation[j];
        for(uint back=0;back<4;back++)for(uint cell=0;cell<80;cell++) {
            const uint frame=3-back,px=(cell/10)*40+(cell%10)*2;
            const float* image=ranges+(n*8+frame)*320;
            uint hit=px;for(uint candidate:{px+1,px+20,px+21})if(image[candidate]<image[hit])hit=candidate;
            if(image[hit]<=.01f||image[hit]>=11.9f)continue;
            const WVec ray=nav_sensor_pixel_ray(1,NAV_SENSOR_ACTIVE_TAN_V,hit);
            const float* pose=poses+(n*8+frame)*12;float delta[3];
            for(uint j=0;j<3;j++)delta[j]=pose[j]+image[hit]*(pose[3+j*3]*ray.x+pose[4+j*3]*ray.y+pose[5+j*3]*ray.z)-body_pose[j];
            const float* actual=points+(n*640+back*80+cell)*4;
            for(uint j=0;j<3;j++) {
                const float expected=rotation[j]*delta[0]+rotation[3+j]*delta[1]+rotation[6+j]*delta[2];
                point_error=std::max(point_error,std::fabs(expected-actual[j]));
            }
        }
        float goal[3],velocity[3];const float dx=worlds[n].goal[0]-body_pose[0],dy=worlds[n].goal[1]-body_pose[1],dz=worlds[n].goal[2]-body_pose[2];
        const float distance=std::sqrt(dx*dx+dy*dy+dz*dz);
        for(uint j=0;j<3;j++) {
            goal[j]=(rotation[j]*dx+rotation[3+j]*dy+rotation[6+j]*dz)/distance;
            velocity[j]=rotation[j]*.3f+rotation[3+j]*.2f+rotation[6+j]*.1f;
        }
        for(uint candidate=0;candidate<85;candidate++) {
            float direction[3]={0,0,0};
            if(candidate<80)nav_ray(candidate/10,candidate%10,NAV_SENSOR_ACTIVE_TAN_V,direction);
            else if(candidate==80)std::copy(goal,goal+3,direction);
            else if(candidate<83)direction[2]=candidate==81?1:-1;
            else direction[1]=candidate==83?1:-1;
            float sweep[3],norm=0;for(uint j=0;j<3;j++){sweep[j]=1.2f*direction[j]+.35f*velocity[j];norm+=sweep[j]*sweep[j];}
            for(float& value:sweep)value/=std::sqrt(norm);
            const float expected=nav_memory_clearance(sweep,ranges+n*8*320,poses+n*8*12,body_pose,3,4,.05f,NAV_SENSOR_ACTIVE_TAN_V);
            clearance_error=std::max(clearance_error,std::fabs(expected-clearances[n*85+candidate]));
        }
    }
    std::cout<<"profile="<<NAV_SENSOR_ACTIVE_NAME<<" point_error_m="<<point_error<<" clearance_error_m="<<clearance_error<<'\n';
    require(point_error<3e-5f&&clearance_error<3e-4f,"optimized sensor-memory cache differs from direct reference");
}

static void sensor_profile_check(Metal& m,const std::string& ray_csv) {
    // (a) Legacy helper must reproduce the frozen world.hpp camera bit-exactly.
    uint32_t mismatched_bits=0;
    for(uint32_t k=0;k<NAV_SENSOR_PIXELS;k++) {
        const WVec expected=wcamera(k);
        const WVec actual=nav_sensor_pixel_ray(NAV_SENSOR_TAN_H,NAV_SENSOR_LEGACY_TAN_V,k);
        mismatched_bits+=std::memcmp(&expected,&actual,sizeof(WVec))!=0;
    }
    require(mismatched_bits==0,"legacy pixel ray differs from world.hpp::wcamera");
    // (b) The header pooled grid and the guidance grid must agree at both profiles.
    for(float tan_v:{NAV_SENSOR_LEGACY_TAN_V,NAV_SENSOR_NATIVE_TAN_V}) {
        for(uint32_t r=0;r<NAV_SENSOR_POOL_ROWS;r++)for(uint32_t c=0;c<NAV_SENSOR_POOL_COLS;c++) {
            const WVec a=nav_sensor_pooled_ray(NAV_SENSOR_TAN_H,tan_v,r,c);
            float b[3];nav_ray(r,c,tan_v,b);
            require(std::memcmp(&a,&b,sizeof(float)*3)==0,"pooled ray differs from guidance nav_ray");
        }
    }
    std::cout<<"sensor profile PASS legacy_pixel_ray_bit_exact=1 pooled_grid_agreement=1 profiles=2\n";
    // (c) Metal/CPU depth and observation parity for the compiled profile.
    sensor_memory_cache_profile_check(m);
    const ProfileParity parity=profile_parity(m);
    require(parity.sensor_error<3e-5f && parity.observation_error<3e-5f,
            std::string("CPU/GPU profile parity failed for ")+NAV_SENSOR_ACTIVE_NAME);
    std::cout<<"profile_parity profile="<<NAV_SENSOR_ACTIVE_NAME
             <<" sensor_error="<<parity.sensor_error<<" observation_error="<<parity.observation_error<<"\n";
    if(ray_csv.empty())return;
    std::ofstream out(ray_csv);require(bool(out),"cannot write sensor ray fixture: "+ray_csv);
    out<<std::setprecision(9)<<"kind,profile,pixel,row,col,dir_x,dir_y,dir_z,dir_norm,ray_x,ray_y,ray_z\n";
    for(const auto& profile:{nav_sensor::legacy(),nav_sensor::native()}) {
        const char* name=nav_sensor::name(profile.id);
        for(uint32_t k=0;k<NAV_SENSOR_PIXELS;k++) {
            const uint32_t row=k/20,col=k%20;
            const WVec d=nav_sensor_pixel_direction(NAV_SENSOR_TAN_H,profile.tan_v,row,col);
            const WVec ray=nav_sensor_pixel_ray(NAV_SENSOR_TAN_H,profile.tan_v,k);
            out<<"pixel,"<<name<<','<<k<<','<<row<<','<<col<<','<<d.x<<','<<d.y<<','<<d.z<<','
               <<nav_sensor_pixel_norm(NAV_SENSOR_TAN_H,profile.tan_v,row,col)<<','
               <<ray.x<<','<<ray.y<<','<<ray.z<<'\n';
        }
        for(uint32_t r=0;r<NAV_SENSOR_POOL_ROWS;r++)for(uint32_t c=0;c<NAV_SENSOR_POOL_COLS;c++) {
            const WVec ray=nav_sensor_pooled_ray(NAV_SENSOR_TAN_H,profile.tan_v,r,c);
            out<<"pooled,"<<name<<",-1,"<<r<<','<<c<<",0,0,0,0,"
               <<ray.x<<','<<ray.y<<','<<ray.z<<'\n';
        }
    }
    std::cout<<"sensor_profile_rays="<<ray_csv<<" rows="<<(2*NAV_SENSOR_PIXELS+2*80)<<"\n";
}

static void deployed_action_map_test(Metal& metal) {
    if(fixed_ppo::actor_obs_dim!=184)return;
    SimConfig config;config.n=32;config.family=4;config.eval=1;
    config.speed=1.5f;config.distance=4;config.geometry_memory=1;
    config.mode=17;Sim deployed(metal,config,1);
    config.mode=22;Sim stochastic(metal,config,1);
    for(uint32_t axis=0;axis<4;axis++)
        static_cast<float*>(deployed.actor.contents)[fixed_ppo::actor_log_std_offset+axis]=-20;
    std::memcpy(stochastic.actor.contents,deployed.actor.contents,deployed.actor.length);
    auto commands=[metal.queue commandBuffer];
    deployed.collect(commands,1);stochastic.collect(commands,1);metal.finish(commands);
    const auto* expected=static_cast<const SimRun*>(deployed.runs.contents);
    const auto* actual=static_cast<const SimRun*>(stochastic.runs.contents);
    const float* observations=static_cast<const float*>(deployed.obs.contents);
    float error=0;bool gate_active=false;
    for(uint32_t env=0;env<config.n;env++) {
        for(uint32_t pixel=0;pixel<20;pixel++)
            gate_active|=observations[env*184+pixel]*12<=2.5f;
        for(uint32_t axis=0;axis<4;axis++)
            error=std::max(error,std::fabs(expected[env].previous_nav[axis]-actual[env].previous_nav[axis]));
    }
    require(gate_active && error<1e-6f,"stochastic deployment map differs from mode17 at negligible exploration");
    std::cout<<"deployed action map PASS mode22 versus17 command_error="<<error<<"\n";
}

static void closed_loop_tests(Metal& m) {
    SimConfig cfg;cfg.n=32;cfg.mode=2;cfg.family=0;cfg.eval=1;cfg.speed=.5f;cfg.distance=4;
    Sim sim(m,cfg,32);std::vector<RLPhysicsState> states(cfg.n);std::vector<SimRun> runs(cfg.n);std::vector<WWorld> worlds(cfg.n);
    std::memcpy(states.data(),sim.states.contents,states.size()*sizeof(RLPhysicsState));std::memcpy(runs.data(),sim.runs.contents,runs.size()*sizeof(SimRun));std::memcpy(worlds.data(),sim.worlds.contents,worlds.size()*sizeof(WWorld));
    auto weights=load_raptor();auto params=rl_physics_crazyflie_default();
    for(uint n=0;n<cfg.n;n++)for(uint tick=0;tick<32;tick++) {
        auto& s=states[n];auto& run=runs[n];const auto& world=worlds[n];float rotation[9];raptor_quaternion_matrix(s.orientation_wxyz,rotation);
        WVec delta=wv(world.goal[0]-s.position[0],world.goal[1]-s.position[1],world.goal[2]-s.position[2]);WVec d=wm(wn(delta),fmin(wl(delta),cfg.speed));
        float body[3];for(uint j=0;j<3;j++)body[j]=(rotation[j]*d.x+rotation[3+j]*d.y+rotation[6+j]*d.z);if(cfg.velocity_contract==0)body[2]*=.5f;
        float velocity[3];for(uint j=0;j<3;j++)velocity[j]=rotation[j*3]*body[0]+rotation[j*3+1]*body[1]+rotation[j*3+2]*body[2];
        const float qtarget[4]={1,0,0,0},wind[3]={0,0,0};
        for(uint sub=0;sub<cfg.substeps;sub++) {
            float target[3],obs[22];for(uint j=0;j<3;j++){run.reference_position[j]+=velocity[j]*params.dt;target[j]=run.reference_position[j];}
            raptor_pack_observation(s.position,s.orientation_wxyz,s.linear_velocity,s.angular_velocity_body,target,qtarget,velocity,run.motors,obs);
            raptor_forward(weights,obs,run.hidden,run.motors);raptor_clip_action(run.motors);RLPhysicsState next;rl_physics_step(s,run.motors,wind,params,next);s=next;
        }
    }
    auto cb=[m.queue commandBuffer];sim.collect(cb);double gpu=m.finish(cb);float error=0,hidden_error=0;const auto* actual=(const RLPhysicsState*)sim.states.contents;const auto* actualrun=(const SimRun*)sim.runs.contents;
    for(uint n=0;n<cfg.n;n++) {require(actualrun[n].episodes==0,"closed-loop parity seed terminated early");for(uint j=0;j<17;j++)error=fmax(error,fabs(((const float*)&actual[n])[j]-((const float*)&states[n])[j]));for(uint j=0;j<16;j++)hidden_error=fmax(hidden_error,fabs(actualrun[n].hidden[j]-runs[n].hidden[j]));}
    require(error<5e-4f&&hidden_error<5e-4f,"closed-loop CPU/GPU trajectory mismatch");std::cout<<"closed loop PASS N=32 native_steps=160 state_error="<<error<<" hidden_error="<<hidden_error<<" GPU_s="<<gpu<<"\n";
}

static volatile float raptor_benchmark_sink = 0.0f;
static std::vector<float> read_raptor_fixture_observations() {
    std::ifstream f(std::string(SOURCE_DIR)+"/assets/raptor.bin", std::ios::binary);
    require(bool(f), "RAPTOR benchmark fixture open");
    f.seekg(16 + sizeof(RaptorWeights));
    char header[20]; f.read(header, sizeof(header));
    require(f && std::memcmp(header, "RPFIX1\0\0", 8)==0, "RAPTOR benchmark fixture header");
    std::vector<float> obs(16 * 22);
    f.read(reinterpret_cast<char*>(obs.data()), obs.size()*sizeof(float));
    require(bool(f), "RAPTOR benchmark observation read");
    return obs;
}
static void raptor_benchmark(Metal& m) {
    const RaptorWeights weights = load_raptor();
    const std::vector<float> fixture_obs = read_raptor_fixture_observations();
    auto wb = m.buffer(sizeof(weights), &weights);
    auto ob = m.buffer(fixture_obs.size()*sizeof(float), fixture_obs.data());
    const auto pipeline = m.pipeline("raptor_fixture");
    std::cout << "N,gpu_kernel_ms,gpu_native_inferences_per_s,cpu_single_thread_ms,cpu_native_inferences_per_s,cpu_over_gpu_device_ratio\n";
    for (size_t n : {size_t(1),size_t(32),size_t(128),size_t(512),size_t(2048),size_t(8192),size_t(32768)}) {
        auto ab = m.buffer(n*16*4*sizeof(float));
        auto hb = m.buffer(n*16*sizeof(float));
        for (int i=0;i<4;i++) {
            auto cb=[m.queue commandBuffer];
            m.dispatch(cb,pipeline,n,{wb,ob,ab,hb});
            m.finish(cb);
        }
        std::vector<double> gpu_times;
        for (int i=0;i<7;i++) {
            auto cb=[m.queue commandBuffer];
            m.dispatch(cb,pipeline,n,{wb,ob,ab,hb});
            gpu_times.push_back(m.finish(cb));
        }
        std::sort(gpu_times.begin(),gpu_times.end());
        const double gpu_s=gpu_times.front();
        const size_t native_inferences=n*16;
        float cpu_actions[4],cpu_hidden[16],checksum=0;
        auto cpu_run=[&]() {
            checksum=0;
            for (size_t env=0;env<n;env++) {
                raptor_reset(weights,cpu_hidden);
                for (size_t t=0;t<16;t++) {
                    raptor_forward(weights,fixture_obs.data()+t*22,cpu_hidden,cpu_actions);
                    checksum += cpu_actions[(t+env)&3] + cpu_hidden[(t+env)&15];
                }
            }
            raptor_benchmark_sink=checksum;
        };
        for (int i=0;i<4;i++) cpu_run();
        std::vector<double> cpu_times;
        for (int i=0;i<7;i++) {
            const double start=seconds();cpu_run();cpu_times.push_back(seconds()-start);
        }
        std::sort(cpu_times.begin(),cpu_times.end());
        const double cpu_s=cpu_times.front();
        const double gpu_ips=double(native_inferences)/gpu_s;
        const double cpu_ips=double(native_inferences)/cpu_s;
        std::cout<<n<<","<<gpu_s*1000<<","<<gpu_ips<<","<<cpu_s*1000<<","<<cpu_ips<<","<<cpu_s/gpu_s<<"\n";
        (void)raptor_benchmark_sink;
    }
}

static void loop_benchmark(Metal& m) {
    std::cout<<"N,batch_gpu_ms,batch_wall_ms,sync_each_wall_ms,nav_transitions_per_sec\n";
    for(uint n:{1u,32u,128u,512u,2048u,8192u,32768u}) {
        SimConfig cfg;cfg.n=n;cfg.family=1;cfg.mode=0;cfg.distance=4;cfg.speed=1;
        Sim sim(m,cfg,8);double batch=1e9,wall=1e9,sync=1e9;
        for(uint repeat=0;repeat<3;repeat++) {
            sim.reset();double start=seconds();auto cb=[m.queue commandBuffer];sim.collect(cb);double g=m.finish(cb);batch=std::min(batch,g);wall=std::min(wall,seconds()-start);
            sim.reset();start=seconds();for(uint t=0;t<8;t++){cb=[m.queue commandBuffer];auto c=sim.configs[t];m.dispatch(cb,sim.depth_p,n*320,{sim.states,sim.runs,sim.worlds,sim.sensors,sim.physics,c,sim.poses});m.dispatch(cb,sim.observe_p,n,{sim.states,sim.runs,sim.worlds,sim.sensors,sim.obs,sim.co,sim.physics,c,sim.poses,sim.memory_clearances},64);encode(m,cb,sim.simd_actor?"ppo_actor_forward_simd_fused":"ppo_actor_forward",sim.simd_actor?((size_t(n)+7)/8)*256:n,{{sim.obs,size_t(t)*n*fixed_ppo::actor_obs_dim*4},{sim.actor,0},{sim.actor_workspace,0},{sim.actions,size_t(t)*n*4*4},{sim.env_count,0}},sim.simd_actor?256:64);m.dispatch(cb,sim.act_p,n,{sim.states,sim.runs,sim.worlds,sim.obs,sim.co,sim.actor,sim.critic,sim.actions,sim.logp,sim.values,sim.commands,sim.physics,c},64);m.dispatch(cb,sim.advance_p,n,{sim.states,sim.runs,sim.worlds,sim.sensors,sim.commands,sim.raptor,sim.critic,sim.rewards,sim.next_values,sim.terminated,sim.truncated,sim.physics,c,sim.bank_worlds,sim.bank_schedule,sim.bank_control,sim.bank_active_ids,sim.bank_transition_ids,sim.environment_physics,sim.runtime_control,sim.task_states,sim.task_control,sim.potential_fields,sim.potential_spec,sim.potential_control},64);m.finish(cb);}sync=std::min(sync,seconds()-start);
        }
        std::cout<<n<<","<<batch*1000<<","<<wall*1000<<","<<sync*1000<<","<<double(n)*8/wall<<"\n";
    }
}

#include <random>
#include <cstdio>
#include <cstdint>
#include <filesystem>
#include <limits>

// Host trainer for the fixed PPO kernels in ppo.metal. Append after Sim and
// sim_evaluate are defined. All rollout, loss, gradient, and optimizer work
// stays on Metal; one command-buffer sync completes each rollout and update.
static const char* PPO_TRAINER_MSL = R"MSL(
kernel void ppo_grad_scale_factor(device const float* grad [[buffer(0)]],
                                  device float* factor [[buffer(1)]],
                                  constant uint& count [[buffer(2)]],
                                  constant float& max_norm [[buffer(3)]],
                                  uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]], uint width [[threads_per_simdgroup]], uint3 group_size [[threads_per_threadgroup]]) {
    threadgroup float partials[32];
    float sum = 0.0f;
    for(uint i=tid;i<count;i+=group_size.x)sum+=grad[i]*grad[i];
    float group_sum=simd_sum(sum);if(lane==0)partials[sg]=group_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(sg==0){float total=simd_sum(lane<group_size.x/width?partials[lane]:0.0f);if(lane==0)factor[0]=min(1.0f,max_norm/sqrt(total+1e-20f));}
}
kernel void ppo_apply_grad_scale(device float* grad [[buffer(0)]],
                                 device const float* factor [[buffer(1)]],
                                 constant uint& count [[buffer(2)]],
                                 uint i [[thread_position_in_grid]]) {
    if (i < count) grad[i] *= factor[0];
}
kernel void ppo_clip_log_std(device float* actor [[buffer(0)]],
                             constant uint& count [[buffer(1)]],
                             constant float& lo [[buffer(2)]],
                             constant float& hi [[buffer(3)]],
                             uint i [[thread_position_in_grid]]) {
    if (i < count) actor[PPO_ACTOR_LOG_STD + i] = clamp(actor[PPO_ACTOR_LOG_STD + i], lo, hi);
}
kernel void ppo_metrics_batch(device const float* losses [[buffer(0)]],
                              device float* metrics [[buffer(1)]],
                              constant uint& batch_size [[buffer(2)]],
                              uint tid [[thread_position_in_grid]]) {
    if (tid != 0) return;
    float sums[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint n = 0; n < batch_size; ++n)
        for (uint k = 0; k < 4; ++k) sums[k] += losses[n * 4 + k];
    sums[3] /= float(batch_size);
    for (uint k = 0; k < 4; ++k) metrics[k] = sums[k];
}
kernel void ppo_metrics_mean(device const float* metrics [[buffer(0)]],
                             device float* mean [[buffer(1)]],
                             constant uint& update_count [[buffer(2)]],
                             uint tid [[thread_position_in_grid]]) {
    if (tid != 0) return;
    for (uint k = 0; k < 4; ++k) mean[k] = 0.0f;
    if (update_count == 0) return;
    for (uint n = 0; n < update_count; ++n)
        for (uint k = 0; k < 4; ++k) mean[k] += metrics[n * 4 + k];
    for (uint k = 0; k < 4; ++k) mean[k] /= float(update_count);
}
kernel void ppo_grad_norm(device const float* grad [[buffer(0)]],
                          device float* out [[buffer(1)]],
                          constant uint& count [[buffer(2)]],
                          uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]], uint width [[threads_per_simdgroup]], uint3 group_size [[threads_per_threadgroup]]) {
    threadgroup float partials[32];
    float sum = 0.0f;
    for(uint i=tid;i<count;i+=group_size.x)sum+=grad[i]*grad[i];
    float group_sum=simd_sum(sum);if(lane==0)partials[sg]=group_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(sg==0){float total=simd_sum(lane<group_size.x/width?partials[lane]:0.0f);if(lane==0)out[0]=sqrt(total+1e-20f);}
}
)MSL";

struct PpoAdamHostConfig { float learning_rate, beta1, beta2, epsilon, weight_decay; uint32_t step; };
static_assert(sizeof(PpoAdamHostConfig) == 24, "PPO Adam config layout mismatch");
struct PpoCheckpointHeader {
    char magic[8];
    uint32_t version, actor_count, critic_count, horizon, n, family;
    uint32_t base_seed, completed_rollouts;
    uint64_t optimizer_step;
    SimConfig config;
};
static_assert(sizeof(PpoCheckpointHeader) == 136, "PPO checkpoint header layout mismatch");

static PpoCheckpointHeader read_checkpoint_header(std::ifstream& file) {
    PpoCheckpointHeader h{};static_assert(offsetof(PpoCheckpointHeader,config)==48,"checkpoint prefix");
    file.read((char*)&h,48);require(file && std::memcmp(h.magic,"PPOFIX1",7)==0 && (h.version>=3&&h.version<=10),"checkpoint prefix/version");
    file.read((char*)&h.config,64);
    if(h.version>=4){file.read((char*)&h+112,16);if(h.version>=6)file.read((char*)&h+128,8);else h.config.geometry_memory=0;if(h.version==4)h.config.velocity_contract=0;}
    else{h.config.risk_coef=0;h.config.entropy_coef=h.family==7?.002f:.005f;h.config.learning_rate=.0003f;h.config.velocity_contract=0;h.config.geometry_memory=0;}
    require(bool(file),"checkpoint header truncated");return h;
}

// Append after PpoCheckpointHeader and read_checkpoint_header are defined.
// Export only the selected guided 184-input actor; the returned file has no
// critic, optimizer, Metal resource, or Objective-C dependency.
#include "deployment.hpp"

static std::array<float,184> deployment_probe_observation(uint32_t sample) {
    std::array<float,184> obs{};
    for(uint32_t i=0;i<160;i++)obs[i]=0.20f+0.01f*float((i*13+sample*7)%70);
    const uint32_t c=160;
    const float angle=-0.7f+0.2f*float(sample);
    obs[c+0]=std::cos(angle);obs[c+1]=std::sin(angle);obs[c+2]=0.1f;
    obs[c+3]=0.3f+0.04f*float(sample);
    obs[c+4]=0.1f;obs[c+5]=-0.05f;obs[c+6]=0.02f;
    obs[c+7]=0.01f;obs[c+8]=-0.02f;obs[c+9]=0.03f;
    obs[c+10]=0.0f;obs[c+11]=0.0f;obs[c+12]=1.0f;
    obs[c+13]=0.05f;obs[c+14]=-0.03f;obs[c+15]=0.0f;obs[c+16]=0.02f;
    obs[c+17]=0.01f;obs[c+18]=0.02f;obs[c+19]=-0.01f;obs[c+20]=0.0f;
    obs[181]=0.08f;obs[182]=-0.04f;obs[183]=0.02f;
    return obs;
}

template<size_t CompiledActorDim>
static float deployment_fixed_ppo_parity(const float* weights,
                                         const nav_deployment::NavigationPolicy& policy) {
    if constexpr(CompiledActorDim==184) {
        fixed_ppo::ActorParams reference{};
        std::copy(weights,weights+nav_deployment::actor_weight_count,reference.values.begin());
        float max_error=0.0f;
        for(uint32_t sample=0;sample<8;sample++) {
            const auto obs=deployment_probe_observation(sample);
            float hidden[fixed_ppo::hidden_dim],mean[fixed_ppo::action_dim];
            fixed_ppo::actor_forward(obs.data(),reference,hidden,mean);
            float exported[nav_deployment::action_count];
            require(policy.raw_mean(obs.data(),exported),"deployment actor inference failed");
            for(uint32_t a=0;a<nav_deployment::action_count;a++)
                max_error=std::max(max_error,std::fabs(mean[a]-exported[a]));
            nav_deployment::NavigationAction command;
            require(policy.infer(obs.data(),command),"deployment command inference failed");
            const float speed=std::sqrt(command.body_velocity_mps[0]*command.body_velocity_mps[0]+
                                        command.body_velocity_mps[1]*command.body_velocity_mps[1]+
                                        command.body_velocity_mps[2]*command.body_velocity_mps[2]);
            require(speed<=policy.metadata().max_speed_mps+1.0e-5f &&
                    std::fabs(command.yaw_rate_rps)<=0.5f+1.0e-6f,
                    "deployment command exceeded its velocity or yaw limit");
        }
        require(max_error<2.0e-5f,"exported actor differs from fixed PPO actor");
        return max_error;
    } else {
        (void)weights;(void)policy;
        return 0.0f;
    }
}

static void export_navigation_checkpoint(const std::string& source_checkpoint,
                                         const std::string& output_path="assets/navigation.bin") {
    std::ifstream file(source_checkpoint,std::ios::binary);
    require(bool(file),"cannot open guided checkpoint: "+source_checkpoint);
    const PpoCheckpointHeader h=read_checkpoint_header(file);
    require(h.version>=6 && h.version<=10,"navigation export requires checkpoint version6..10");
    require(h.actor_count==nav_deployment::actor_weight_count,
            "navigation export requires a 184-input,64-hidden,four-action actor");
    require(h.config.geometry_memory==1 && h.config.velocity_contract==1,
            "navigation export requires geometry-memory guidance and spherical velocity contract");
    require(std::fabs(h.config.speed-1.5f)<1.0e-6f,
            "navigation export requires the selected 1.5 m/s checkpoint");
    require(h.config.sensor_period==1 && h.config.substeps==5,
            "navigation export expects 20 Hz observations and 100 Hz motor control");
    std::array<float,nav_deployment::actor_weight_count> actor_weights{};
    file.read(reinterpret_cast<char*>(actor_weights.data()),actor_weights.size()*sizeof(float));
    require(bool(file),"guided checkpoint actor weights are truncated");

    nav_deployment::Metadata metadata;
    metadata.max_speed_mps=h.config.speed;
    const RLPhysicsParams dynamics=rl_physics_crazyflie_default();
    metadata.navigation_period_s=dynamics.dt*float(h.config.substeps);
    metadata.native_period_s=dynamics.dt;
    std::string error;
    require(nav_deployment::NavigationPolicy::write_file(output_path,actor_weights.data(),
                actor_weights.size(),metadata,source_checkpoint,&error),error);
    nav_deployment::NavigationPolicy policy;
    require(policy.load(output_path,&error),error);
    const float parity=deployment_fixed_ppo_parity<fixed_ppo::actor_obs_dim>(actor_weights.data(),policy);
    std::error_code ec;const auto bytes=std::filesystem::file_size(output_path,ec);
    require(!ec,"cannot stat exported navigation policy");
    std::cout<<"NAV export="<<output_path<<" source="<<source_checkpoint<<" bytes="<<bytes
             <<" observations="<<metadata.observation_count<<" weights="<<actor_weights.size()
             <<" mode="<<metadata.policy_mode<<" speed="<<metadata.max_speed_mps
             <<" checkpoint_fnv64="<<std::hex<<policy.source_checkpoint_hash()<<std::dec
             <<" max_actor_error="<<parity<<"\n";
}

static void benchmark_navigation_policy(const std::string& path) {
    nav_deployment::NavigationPolicy policy;std::string error;
    require(policy.load(path,&error),error);
    auto obs=deployment_probe_observation(0);nav_deployment::NavigationAction action;
    constexpr uint32_t iterations=10000;double checksum=0,start=seconds();
    for(uint32_t i=0;i<iterations;i++) {
        obs[163]=.3f+.0001f*float(i%1000);
        require(policy.infer(obs.data(),action),"navigation policy inference failed");
        for(uint32_t j=0;j<3;j++)obs[173+j]=action.body_velocity_mps[j]/policy.metadata().max_speed_mps;
        checksum+=action.body_velocity_mps[0]+action.yaw_rate_rps;
    }
    const double elapsed=seconds()-start;
    std::cout<<"NAV CPU batch=1 iterations="<<iterations<<" actor_only_us="<<elapsed*1e6/iterations
             <<" excludes=depth,pose_memory,RAPTOR checksum="<<checksum<<"\n";
}

static void evaluate_navigation_policy(Metal& m,const std::string& path,uint32_t family,uint32_t seed) {
    require(fixed_ppo::actor_obs_dim==184,"exported policy evaluation requires the guided binary");
    nav_deployment::NavigationPolicy policy;std::string error;require(policy.load(path,&error),error);
    sim_evaluate(m,family,policy.metadata().policy_mode,policy.weights().data(),
                 policy.metadata().max_speed_mps,4,seed,0,0,0,0,0,1,1);
}

struct PPOTrainer {
    id<MTLBuffer> grad_norm_rows;
    id<MTLComputePipelineState> grad_norm_p;
    Sim& sim;
    Metal& metal;
    uint32_t minibatch = 256;
    uint32_t rows, minibatches, updates_per_rollout;
    // PPO epochs per rollout. Preserved default keeps historical checkpoints
    // bit-identical; only an explicit training flag may lower it.
    uint32_t epochs = 2;
    uint64_t optimizer_step = 0;
    uint32_t completed_rollouts = 0;
    std::vector<uint32_t> starts;
    id<MTLBuffer> actor_hidden, actor_means, critic_hidden, predicted_values;
    id<MTLBuffer> d_means, d_log_stds, d_values, losses;
    id<MTLBuffer> actor_hidden_delta, actor_grad, critic_hidden_delta, critic_grad;
    id<MTLBuffer> actor_m, actor_v, critic_m, critic_v;
    // Optional replay evidence. Copy before PPO normalization in the same
    // command buffer; inspect only after the rollout and update complete.
    id<MTLBuffer> raw_advantages=nil;
    id<MTLBuffer> actor_scale, critic_scale, metric_rows, metric_mean;
    // Training-only standardized-advantage winsorization radius. FLT_MAX is
    // the no-op default; only bank training may narrow it via a CLI flag.
    id<MTLBuffer> adv_clip_b;
    // Training-only actor anchor: frozen copy of the actor weights at capture
    // time plus the pull strength toward that reference. lambda 0 disables it.
    id<MTLBuffer> anchor_ref_b, anchor_lambda_b;
    id<MTLBuffer> direction_aux_coefficient_b=nil;
    id<MTLComputePipelineState> anchor_p;
    id<MTLComputePipelineState> direction_aux_grad_p=nil;
    float anchor_lambda = 0.0f;
    float anchor_radius = 0.0f;
    float anchor_pull = 0.0f;
    float direction_aux_coefficient = 0.0f;
    id<MTLBuffer> rows_b, envs_b, horizon_b, gamma_b, lambda_b, gae_epsilon_b;
    id<MTLBuffer> clip_b, value_coef_b, entropy_coef_b, actor_count_b, critic_count_b;
    id<MTLBuffer> actor_max_norm_b, critic_max_norm_b, logstd_count_b, logstd_lo_b, logstd_hi_b;
    id<MTLBuffer> metric_count_b;
    std::vector<id<MTLBuffer>> batch_size_buffers, adam_configs;
    id<MTLComputePipelineState> gae_p, normalize_p, actor_forward_p, critic_forward_p;
    id<MTLComputePipelineState> loss_grad_p, actor_hidden_delta_p, actor_grad_direct_p, critic_hidden_delta_p, critic_grad_direct_p;
    id<MTLComputePipelineState> norm_factor_p, scale_grad_p, adam_p, clip_logstd_p;
    id<MTLComputePipelineState> metric_batch_p, metric_mean_p;

    static id<MTLBuffer> scalar(Metal& m, uint32_t value) { return m.buffer(sizeof(value), &value); }
    static id<MTLBuffer> scalar(Metal& m, float value) { return m.buffer(sizeof(value), &value); }
    void dispatch(id<MTLCommandBuffer> cb, id<MTLComputePipelineState> p, size_t count,
                  std::initializer_list<BufferBinding> bindings, size_t group=128) {
        auto e=metal.profiler.encoder(cb,metal.profiler.armed?metal.pipelineName(p):nullptr); [e setComputePipelineState:p]; uint j=0;
        for(auto b:bindings) [e setBuffer:b.first offset:b.second atIndex:j++];
        [e dispatchThreads:MTLSizeMake(count,1,1)
        threadsPerThreadgroup:MTLSizeMake(std::min(group,size_t(p.maxTotalThreadsPerThreadgroup)),1,1)];
        [e endEncoding];
    }
    PPOTrainer(Sim& s,uint32_t update_epochs=2):sim(s),metal(s.m) {
        require(update_epochs>=1 && update_epochs<=4,"PPO epochs must be1..4");
        epochs=update_epochs;
        rows=sim.cfg.n*sim.horizon;
        require(rows>0,"PPO requires at least one rollout row");
        minibatches=(rows+minibatch-1)/minibatch;
        updates_per_rollout=epochs*minibatches;
        starts.reserve(minibatches);for(uint32_t s0=0;s0<rows;s0+=minibatch)starts.push_back(s0);
        actor_hidden=metal.buffer(size_t(minibatch)*fixed_ppo::hidden_dim*4);
        actor_means=metal.buffer(size_t(minibatch)*fixed_ppo::action_dim*4);
        critic_hidden=metal.buffer(size_t(minibatch)*fixed_ppo::hidden_dim*4);
        predicted_values=metal.buffer(size_t(minibatch)*4);
        d_means=metal.buffer(size_t(minibatch)*fixed_ppo::action_dim*4);
        d_log_stds=metal.buffer(size_t(minibatch)*fixed_ppo::action_dim*4);
        d_values=metal.buffer(size_t(minibatch)*4);
        losses=metal.buffer(size_t(minibatch)*4*4);
        actor_hidden_delta=metal.buffer(size_t(minibatch)*fixed_ppo::hidden_dim*4);
        actor_grad=metal.buffer(fixed_ppo::actor_param_count*4);
        critic_hidden_delta=metal.buffer(size_t(minibatch)*fixed_ppo::hidden_dim*4);
        critic_grad=metal.buffer(fixed_ppo::critic_param_count*4);
        actor_m=metal.buffer(fixed_ppo::actor_param_count*4); actor_v=metal.buffer(fixed_ppo::actor_param_count*4);
        critic_m=metal.buffer(fixed_ppo::critic_param_count*4); critic_v=metal.buffer(fixed_ppo::critic_param_count*4);
        actor_scale=metal.buffer(4); critic_scale=metal.buffer(4);
        grad_norm_rows=metal.buffer(size_t(updates_per_rollout)*2*4);
        grad_norm_p=metal.pipeline("ppo_grad_norm");
        metric_rows=metal.buffer(size_t(updates_per_rollout)*4*4); metric_mean=metal.buffer(4*4);
        adv_clip_b=scalar(metal,std::numeric_limits<float>::max());
        anchor_lambda_b=scalar(metal,0.0f);
        anchor_ref_b=metal.buffer(sim.actor.length);
        std::memcpy(anchor_ref_b.contents,sim.actor.contents,sim.actor.length);
        anchor_p=metal.pipeline("ppo_anchor_grad");
        rows_b=scalar(metal,rows); envs_b=scalar(metal,sim.cfg.n); horizon_b=scalar(metal,sim.horizon);
        gamma_b=scalar(metal,.99f); lambda_b=scalar(metal,.95f); gae_epsilon_b=scalar(metal,1.0e-8f);
        clip_b=scalar(metal,.2f); value_coef_b=scalar(metal,.5f); entropy_coef_b=scalar(metal,sim.cfg.entropy_coef);
        actor_count_b=scalar(metal,uint32_t(fixed_ppo::actor_param_count));
        critic_count_b=scalar(metal,uint32_t(fixed_ppo::critic_param_count));
        actor_max_norm_b=scalar(metal,.5f); critic_max_norm_b=scalar(metal,.5f);
        logstd_count_b=scalar(metal,uint32_t(fixed_ppo::action_dim)); logstd_lo_b=scalar(metal,-2.0f); logstd_hi_b=scalar(metal,.5f);
        metric_count_b=scalar(metal,updates_per_rollout);
        for(uint32_t b=0;b<=minibatch;b++) batch_size_buffers.push_back(scalar(metal,b));
        for(uint32_t i=0;i<updates_per_rollout;i++) adam_configs.push_back(metal.buffer(sizeof(PpoAdamHostConfig)));
        gae_p=metal.pipeline("ppo_gae"); normalize_p=metal.pipeline("ppo_normalize_advantages");
        actor_forward_p=metal.pipeline(sim.simd_actor?"ppo_actor_forward_simd_fused":"ppo_actor_forward"); critic_forward_p=metal.pipeline("ppo_critic_forward");
        loss_grad_p=metal.pipeline("ppo_sample_loss_grad"); actor_hidden_delta_p=metal.pipeline("ppo_actor_hidden_delta");
        actor_grad_direct_p=metal.pipeline("ppo_actor_grad_direct"); critic_hidden_delta_p=metal.pipeline("ppo_critic_hidden_delta");
        critic_grad_direct_p=metal.pipeline("ppo_critic_grad_direct");
        norm_factor_p=metal.pipeline("ppo_grad_scale_factor"); scale_grad_p=metal.pipeline("ppo_apply_grad_scale");
        adam_p=metal.pipeline("ppo_adam_update"); clip_logstd_p=metal.pipeline("ppo_clip_log_std");
        metric_batch_p=metal.pipeline("ppo_metrics_batch"); metric_mean_p=metal.pipeline("ppo_metrics_mean");
        std::memset(actor_m.contents,0,fixed_ppo::actor_param_count*4);std::memset(actor_v.contents,0,fixed_ppo::actor_param_count*4);
        std::memset(critic_m.contents,0,fixed_ppo::critic_param_count*4);std::memset(critic_v.contents,0,fixed_ppo::critic_param_count*4);
    }
    void load_checkpoint(const std::string& path,uint32_t family,uint32_t horizon,uint32_t n,uint32_t seed) {
        std::ifstream f(path,std::ios::binary); if(!f)return;
        PpoCheckpointHeader h=read_checkpoint_header(f);
        require(f && std::memcmp(h.magic,"PPOFIX1",7)==0 && (h.version>=3&&h.version<=10),"invalid PPO checkpoint header");
        require(h.actor_count==fixed_ppo::actor_param_count && h.critic_count==fixed_ppo::critic_param_count,"PPO checkpoint dimensions mismatch");
        require(h.family==family && h.horizon==horizon && h.n==n && h.base_seed==seed,"PPO checkpoint config mismatch");
        require(h.version>=10 || NAV_SENSOR_ACTIVE_ID==NAV_SENSOR_LEGACY_ID,
                "legacy checkpoint cannot resume with calibrated sensors; use an explicit parameter warmstart");
        require(h.config.family==sim.cfg.family && h.config.n==sim.cfg.n && h.config.substeps==sim.cfg.substeps &&
                h.config.sensor_period==sim.cfg.sensor_period && h.config.sensor_delay==sim.cfg.sensor_delay &&
                h.config.command_delay==sim.cfg.command_delay && h.config.max_steps==sim.cfg.max_steps &&
                h.config.mode==sim.cfg.mode && h.config.speed==sim.cfg.speed && h.config.distance==sim.cfg.distance &&
                h.config.wind==sim.cfg.wind && h.config.depth_noise==sim.cfg.depth_noise && h.config.dropout==sim.cfg.dropout && h.config.risk_coef==sim.cfg.risk_coef && h.config.entropy_coef==sim.cfg.entropy_coef && h.config.learning_rate==sim.cfg.learning_rate && h.config.velocity_contract==sim.cfg.velocity_contract && h.config.geometry_memory==sim.cfg.geometry_memory,
                "PPO checkpoint simulator settings mismatch");
        auto readbuf=[&](id<MTLBuffer> b,size_t count){f.read((char*)b.contents,count*4);require(bool(f),"truncated PPO checkpoint");};
        readbuf(sim.actor,fixed_ppo::actor_param_count); readbuf(sim.critic,fixed_ppo::critic_param_count);
        readbuf(actor_m,fixed_ppo::actor_param_count); readbuf(actor_v,fixed_ppo::actor_param_count);
        readbuf(critic_m,fixed_ppo::critic_param_count); readbuf(critic_v,fixed_ppo::critic_param_count);
        auto readbytes=[&](id<MTLBuffer> b){f.read((char*)b.contents,b.length);require(bool(f),"truncated PPO simulator checkpoint");};
        readbytes(sim.states);readbytes(sim.runs);readbytes(sim.worlds);readbytes(sim.sensors);if(h.version>=6)readbytes(sim.poses);else std::memset(sim.poses.contents,0,sim.poses.length);readbytes(sim.commands);
        if(h.version>=7) {
            NavigationRuntimeConfig saved_runtime{};NavigationTaskControl saved_tasks{};
            f.read((char*)&saved_runtime,sizeof(saved_runtime));
            f.read((char*)&saved_tasks,h.version>=8?sizeof(saved_tasks):144);
            require(h.version>=8 || saved_tasks.enabled==0,"task-generator semantics changed; use an explicit parameter warmstart from version7");
            require(bool(f),"truncated navigation runtime configuration");
            require(std::memcmp(&saved_runtime,sim.runtime_control.contents,sizeof(saved_runtime))==0 &&
                    std::memcmp(&saved_tasks,sim.task_control.contents,sizeof(saved_tasks))==0,
                    "checkpoint dynamics/task configuration mismatch");
            readbytes(sim.environment_physics);readbytes(sim.task_states);
            RLPhysicsParams saved_nominal{};RaptorWeights saved_raptor{};
            f.read((char*)&saved_nominal,sizeof(saved_nominal));f.read((char*)&saved_raptor,sizeof(saved_raptor));
            require(bool(f),"truncated checkpoint nominal plant/controller");
            require(std::memcmp(&saved_nominal,sim.physics.contents,sizeof(saved_nominal))==0 &&
                    std::memcmp(&saved_raptor,sim.raptor.contents,sizeof(saved_raptor))==0,
                    "checkpoint nominal plant/controller mismatch");
        } else {
            require(((const NavigationRuntimeConfig*)sim.runtime_control.contents)->enabled==0 &&
                    ((const NavigationTaskControl*)sim.task_control.contents)->enabled==0,
                    "legacy resume cannot infer new dynamics/task state; use explicit parameter warmstart");
        }
        if(h.version>=9) {
            TrainingPotentialControl saved_control{};TrainingPotentialGridSpec saved_spec{};char saved_hash[64];
            f.read((char*)&saved_control,sizeof(saved_control));f.read((char*)&saved_spec,sizeof(saved_spec));f.read(saved_hash,64);
            require(bool(f),"truncated shaping configuration");
            require(std::memcmp(&saved_control,sim.potential_control.contents,sizeof(saved_control))==0 &&
                    std::memcmp(&saved_spec,sim.potential_spec.contents,sizeof(saved_spec))==0 &&
                    std::memcmp(saved_hash,sim.potential_hash.contents,64)==0,"checkpoint shaping configuration or field hash mismatch");
        } else require(((const TrainingPotentialControl*)sim.potential_control.contents)->enabled==0,
                       "legacy resume cannot infer shaping settings; use explicit parameter warmstart");
        sim.set_config(h.config);
        // Version 10 binds the complete simulator resume to the calibrated
        // camera. Older files have the historical legacy camera contract.
        if(h.version>=10) {
            nav_sensor::Profile saved{};f.read((char*)&saved,sizeof(saved));
            const auto expected=nav_sensor::active();
            require(bool(f) && saved.id==expected.id && saved.tan_h==expected.tan_h &&
                    saved.tan_v==expected.tan_v && saved.mount_x_m==expected.mount_x_m &&
                    saved.min_range_m==expected.min_range_m && saved.max_range_m==expected.max_range_m,
                    "checkpoint camera contract mismatch; use an explicit parameter warmstart");
        } else require(NAV_SENSOR_ACTIVE_ID==NAV_SENSOR_LEGACY_ID,
                       "legacy checkpoint cannot resume with calibrated sensors; use an explicit parameter warmstart");
        char extra; require(!f.read(&extra,1),"PPO checkpoint has trailing data");
        optimizer_step=h.optimizer_step;completed_rollouts=h.completed_rollouts;
        require(optimizer_step<=uint64_t(std::numeric_limits<uint32_t>::max()),"PPO checkpoint Adam step is too large");
        std::cout<<"PPO resume rollouts="<<h.completed_rollouts<<" optimizer_step="<<optimizer_step<<"\n";
    }
    void save_checkpoint(const std::string& path,uint32_t family,uint32_t seed,uint32_t completed_rollouts) {
        if(path.empty())return;
        const std::filesystem::path checkpoint_path(path);
        if(checkpoint_path.has_parent_path())std::filesystem::create_directories(checkpoint_path.parent_path());
        PpoCheckpointHeader h;
        // The binary header has alignment padding. Initialize those bytes too
        // so paired checkpoint hashes do not depend on the host stack.
        std::memset(&h,0,sizeof(h));std::memcpy(h.magic,"PPOFIX1",7);h.version=((const TrainingPotentialControl*)sim.potential_control.contents)->enabled ? 9 : (((const NavigationRuntimeConfig*)sim.runtime_control.contents)->enabled || ((const NavigationTaskControl*)sim.task_control.contents)->enabled ? 8 : 6);
        if(NAV_SENSOR_ACTIVE_ID==NAV_SENSOR_NATIVE_ID)h.version=10;
        h.actor_count=fixed_ppo::actor_param_count;h.critic_count=fixed_ppo::critic_param_count;
        h.horizon=sim.horizon;h.n=sim.cfg.n;h.family=family;h.base_seed=seed;h.completed_rollouts=completed_rollouts;
        h.optimizer_step=optimizer_step;h.config=sim.cfg;h.config.tick=0;
        const std::string temp=path+".tmp";std::ofstream f(temp,std::ios::binary|std::ios::trunc);
        require(bool(f),"cannot write PPO checkpoint: "+temp); f.write(reinterpret_cast<const char*>(&h),sizeof(h));
        auto writebuf=[&](id<MTLBuffer> b,size_t count){f.write((const char*)b.contents,count*4);};
        writebuf(sim.actor,fixed_ppo::actor_param_count);writebuf(sim.critic,fixed_ppo::critic_param_count);
        writebuf(actor_m,fixed_ppo::actor_param_count);writebuf(actor_v,fixed_ppo::actor_param_count);
        writebuf(critic_m,fixed_ppo::critic_param_count);writebuf(critic_v,fixed_ppo::critic_param_count);
        auto writebytes=[&](id<MTLBuffer> b){f.write((const char*)b.contents,b.length);};
        writebytes(sim.states);writebytes(sim.runs);writebytes(sim.worlds);writebytes(sim.sensors);writebytes(sim.poses);writebytes(sim.commands);
        if(h.version>=7) {
            writebytes(sim.runtime_control);writebytes(sim.task_control);
            writebytes(sim.environment_physics);writebytes(sim.task_states);
            writebytes(sim.physics);writebytes(sim.raptor);
        }
        if(h.version>=9) {
            writebytes(sim.potential_control);writebytes(sim.potential_spec);writebytes(sim.potential_hash);
        }
        if(h.version>=10) {
            const auto camera=nav_sensor::active();
            static_assert(sizeof(camera)==24,"checkpoint camera layout");
            f.write((const char*)&camera,sizeof(camera));
        }
        f.flush();require(bool(f),"PPO checkpoint write failed: "+temp);f.close();
        require(std::rename(temp.c_str(),path.c_str())==0,"cannot replace PPO checkpoint: "+path);
    }
    void rollout_update(id<MTLCommandBuffer> cb,uint32_t rollout_index) {
        if(anchor_lambda>0)refresh_anchor_pull();
        dispatch(cb,gae_p,sim.cfg.n,{{sim.rewards,0},{sim.values,0},{sim.next_values,0},{sim.terminated,0},{sim.truncated,0},
                 {sim.advantages,0},{sim.returns,0},{horizon_b,0},{envs_b,0},{gamma_b,0},{lambda_b,0}},64);
        if(raw_advantages) {
            auto copy=[cb blitCommandEncoder];
            [copy copyFromBuffer:sim.advantages sourceOffset:0 toBuffer:raw_advantages destinationOffset:0 size:sim.advantages.length];
            [copy endEncoding];
        }
        dispatch(cb,normalize_p,1,{{sim.advantages,0},{rows_b,0},{gae_epsilon_b,0},{adv_clip_b,0}},1);
        // Resume reconstructs this vector. Restore its canonical order before
        // shuffling so the rollout seed fully determines minibatch order.
        for(uint32_t batch=0;batch<starts.size();batch++)starts[batch]=batch*minibatch;
        std::mt19937 rng(0x9e3779b9u+rollout_index*0x85ebca6bu);std::shuffle(starts.begin(),starts.end(),rng);
        uint32_t update=0;
        for(uint32_t epoch=0;epoch<epochs;epoch++) {
            if(epoch==1)std::shuffle(starts.begin(),starts.end(),rng);
            for(uint32_t begin:starts) {
                const uint32_t b=std::min(minibatch,rows-begin);const size_t obs_offset=size_t(begin)*fixed_ppo::actor_obs_dim*4;
                const size_t co_offset=size_t(begin)*fixed_ppo::critic_obs_dim*4;
                const size_t action_offset=size_t(begin)*fixed_ppo::action_dim*4, row_offset=size_t(begin)*4;
                const auto bb=batch_size_buffers[b];
                dispatch(cb,actor_forward_p,sim.simd_actor?((size_t(b)+7)/8)*256:b,{{sim.obs,obs_offset},{sim.actor,0},{actor_hidden,0},{actor_means,0},{bb,0}},sim.simd_actor?256:64);
                dispatch(cb,critic_forward_p,b,{{sim.co,co_offset},{sim.critic,0},{critic_hidden,0},{predicted_values,0},{bb,0}},64);
                dispatch(cb,loss_grad_p,b,{{sim.actions,action_offset},{actor_means,0},{sim.actor,fixed_ppo::actor_log_std_offset*4},
                    {sim.logp,row_offset},{predicted_values,0},{sim.values,row_offset},{sim.advantages,row_offset},{sim.returns,row_offset},
                    {d_means,0},{d_log_stds,0},{d_values,0},{losses,0},{bb,0},{clip_b,0},{value_coef_b,0},{entropy_coef_b,0}},64);
                if(direction_aux_coefficient>0.0f) {
                    const size_t target_offset=size_t(begin)*16*sizeof(float);
                    dispatch(cb,direction_aux_grad_p,b,{{sim.obs,obs_offset},{actor_means,0},
                        {sim.direction_aux_targets,target_offset},{d_means,0},{bb,0},{direction_aux_coefficient_b,0}},64);
                }
                dispatch(cb,metric_batch_p,1,{{losses,0},{metric_rows,size_t(update)*4*4},{bb,0}},1);
                dispatch(cb,actor_hidden_delta_p,size_t(b)*fixed_ppo::hidden_dim,{{sim.actor,0},{d_means,0},{actor_hidden,0},{actor_hidden_delta,0},{bb,0}},128);
                dispatch(cb,actor_grad_direct_p,fixed_ppo::actor_param_count,{{sim.obs,obs_offset},{actor_hidden,0},{d_means,0},{d_log_stds,0},{actor_hidden_delta,0},{actor_grad,0},{bb,0}},128);
                if(anchor_lambda>0)
                    dispatch(cb,anchor_p,fixed_ppo::actor_param_count,{{actor_grad,0},{sim.actor,0},{anchor_ref_b,0},{anchor_lambda_b,0},{actor_count_b,0}},128);
                dispatch(cb,critic_hidden_delta_p,size_t(b)*fixed_ppo::hidden_dim,{{sim.critic,0},{d_values,0},{critic_hidden,0},{critic_hidden_delta,0},{bb,0}},128);
                dispatch(cb,critic_grad_direct_p,fixed_ppo::critic_param_count,{{sim.co,co_offset},{critic_hidden,0},{d_values,0},{critic_hidden_delta,0},{critic_grad,0},{bb,0}},128);
                require(optimizer_step<uint64_t(std::numeric_limits<uint32_t>::max()),"PPO Adam step overflow");
                const uint32_t step=uint32_t(++optimizer_step);
                PpoAdamHostConfig ac{sim.cfg.learning_rate,.9f,.999f,1.0e-8f,0.0f,step};
                std::memcpy(adam_configs[update].contents,&ac,sizeof(ac));
                dispatch(cb,grad_norm_p,256,{{actor_grad,0},{grad_norm_rows,size_t(update)*2*4},{actor_count_b,0}},256);
                dispatch(cb,norm_factor_p,256,{{actor_grad,0},{actor_scale,0},{actor_count_b,0},{actor_max_norm_b,0}},256);
                dispatch(cb,scale_grad_p,fixed_ppo::actor_param_count,{{actor_grad,0},{actor_scale,0},{actor_count_b,0}},128);
                dispatch(cb,grad_norm_p,256,{{critic_grad,0},{grad_norm_rows,(size_t(update)*2+1)*4},{critic_count_b,0}},256);
                dispatch(cb,norm_factor_p,256,{{critic_grad,0},{critic_scale,0},{critic_count_b,0},{critic_max_norm_b,0}},256);
                dispatch(cb,scale_grad_p,fixed_ppo::critic_param_count,{{critic_grad,0},{critic_scale,0},{critic_count_b,0}},128);
                dispatch(cb,adam_p,fixed_ppo::actor_param_count,{{sim.actor,0},{actor_grad,0},{actor_m,0},{actor_v,0},{actor_count_b,0},{adam_configs[update],0}},128);
                dispatch(cb,adam_p,fixed_ppo::critic_param_count,{{sim.critic,0},{critic_grad,0},{critic_m,0},{critic_v,0},{critic_count_b,0},{adam_configs[update],0}},128);
                dispatch(cb,clip_logstd_p,fixed_ppo::action_dim,{{sim.actor,0},{logstd_count_b,0},{logstd_lo_b,0},{logstd_hi_b,0}},4);
                ++update;
            }
        }
        require(update==updates_per_rollout,"PPO update count mismatch");
        dispatch(cb,metric_mean_p,1,{{metric_rows,0},{metric_mean,0},{metric_count_b,0}},1);
    }
    // Anchor reference must be captured after any warmstart/resume load so it
    // points at the actual launch policy, not the fresh initialization.
    void recapture_anchor_reference() {
        std::memcpy(anchor_ref_b.contents,sim.actor.contents,sim.actor.length);
    }
    void set_direction_auxiliary(float coefficient) {
        require(std::isfinite(coefficient)&&coefficient>=0.0f&&coefficient<=1.0f,
                "geodesic direction auxiliary coefficient must be in0..1");
        if(coefficient>0.0f) {
            require(sim.direction_aux_enabled,"direction targets must be enabled before PPO updates");
            direction_aux_coefficient_b=scalar(metal,coefficient);
            direction_aux_grad_p=metal.pipeline("ppo_actor_geodesic_direction_grad");
        }
        direction_aux_coefficient=coefficient;
    }
    void set_anchor(float lambda,float radius=0.0f) {
        require(std::isfinite(lambda)&&lambda>=0.0f,"anchor lambda must be finite and non-negative");
        require(std::isfinite(radius)&&radius>=0.0f,"anchor radius must be finite and non-negative");
        anchor_lambda=lambda;anchor_radius=radius;
        refresh_anchor_pull();
    }
    // Parameter-space hinge penalty: within anchor_radius of the reference the
    // actor has no anchor force; beyond it, the gradient of
    // lambda/2*(distance-radius)^2 restores the actor toward the reference.
    // Refresh once per rollout from shared host memory.
    void refresh_anchor_pull() {
        float effective=0.0f;
        if(anchor_lambda>0) {
            const double distance=actor_drift_l2();
            double scale=1.0;
            if(anchor_radius>0)scale=distance>1e-12?std::max(0.0,1.0-double(anchor_radius)/distance):0.0;
            effective=anchor_lambda*float(scale);
            anchor_pull=effective;
        }
        std::memcpy(anchor_lambda_b.contents,&effective,sizeof(float));
    }
    float current_anchor_pull()const{return anchor_pull;}
    struct GradNormSummary {
        float actor_mean=0.0f,actor_max=0.0f,actor_clip_frac=0.0f;
        float critic_mean=0.0f,critic_max=0.0f,critic_clip_frac=0.0f;
    };
    GradNormSummary grad_norm_summary()const {
        GradNormSummary summary;
        const float* rows=static_cast<const float*>(grad_norm_rows.contents);
        const float actor_limit=static_cast<const float*>(actor_max_norm_b.contents)[0];
        const float critic_limit=static_cast<const float*>(critic_max_norm_b.contents)[0];
        double actor_sum=0.0,critic_sum=0.0;uint32_t actor_clipped=0,critic_clipped=0;
        for(uint32_t update=0;update<updates_per_rollout;update++) {
            const float actor_norm=rows[update*2],critic_norm=rows[update*2+1];
            require(std::isfinite(actor_norm)&&std::isfinite(critic_norm),"non-finite pre-clip gradient norm");
            actor_sum+=actor_norm;critic_sum+=critic_norm;
            actor_clipped+=actor_norm>actor_limit?1u:0u;
            critic_clipped+=critic_norm>critic_limit?1u:0u;
            summary.actor_max=std::max(summary.actor_max,actor_norm);
            summary.critic_max=std::max(summary.critic_max,critic_norm);
        }
        if(updates_per_rollout>0) {
            summary.actor_mean=float(actor_sum/double(updates_per_rollout));
            summary.critic_mean=float(critic_sum/double(updates_per_rollout));
            summary.actor_clip_frac=float(actor_clipped)/float(updates_per_rollout);
            summary.critic_clip_frac=float(critic_clipped)/float(updates_per_rollout);
        }
        return summary;
    }
    double actor_drift_l2()const {
        const float* actor=static_cast<const float*>(sim.actor.contents);
        const float* reference=static_cast<const float*>(anchor_ref_b.contents);
        const size_t count=fixed_ppo::actor_param_count;
        double acc=0;
        for(size_t i=0;i<count;i++){const double d=double(actor[i])-double(reference[i]);acc+=d*d;}
        return std::sqrt(acc);
    }
};

static float verify_raw_depth_lift(Metal& metal,const float* old_weights,
                                   const float* lifted_weights,const std::string& old_checkpoint,
                                   const std::string& lifted_checkpoint,const std::string& raw_policy_path,
                                   const PpoCheckpointHeader& source_header) {
    const auto dynamics=rl_physics_crazyflie_default();
    nav_deployment::Metadata old_metadata;old_metadata.max_speed_mps=source_header.config.speed;
    old_metadata.navigation_period_s=dynamics.dt*float(source_header.config.substeps);
    old_metadata.native_period_s=dynamics.dt;
    const std::string old_policy_path=raw_policy_path+".legacy-v1-probe";
    std::string error;
    require(nav_deployment::NavigationPolicy::write_file(old_policy_path,old_weights,
                nav_deployment::actor_weight_count,old_metadata,old_checkpoint,&error),error);
    nav_deployment::NavigationPolicy old_policy;
    require(old_policy.load(old_policy_path,&error),error);
    nav_deployment::Metadata raw_metadata;
    raw_metadata.version=2;raw_metadata.observation_count=nav_deployment::raw_depth_actor_observation_count;
    raw_metadata.weight_count=nav_deployment::raw_depth_actor_weight_count;
    raw_metadata.max_speed_mps=source_header.config.speed;
    raw_metadata.navigation_period_s=dynamics.dt*float(source_header.config.substeps);
    raw_metadata.native_period_s=dynamics.dt;
    require(nav_deployment::RawDepthNavigationPolicy::write_file(raw_policy_path,lifted_weights,
                nav_deployment::raw_depth_actor_weight_count,raw_metadata,lifted_checkpoint,&error),error);
    nav_deployment::RawDepthNavigationPolicy raw_policy;
    require(raw_policy.load(raw_policy_path,&error),error);
    nav_deployment::NavigationPolicy wrong_v1_reader;
    nav_deployment::RawDepthNavigationPolicy wrong_v2_reader;
    require(!wrong_v1_reader.load(raw_policy_path,&error),"v1 deployment reader accepted a v2 raw-depth file");
    require(!wrong_v2_reader.load(old_policy_path,&error),"v2 deployment reader accepted a v1 pooled file");

    float max_deployment_error=0.0f,max_metal_error=0.0f,max_pack_error=0.0f;
    constexpr uint32_t batch=8;
    std::vector<float> metal_observations(size_t(batch)*nav_deployment::raw_depth_actor_observation_count);
    std::vector<float> expected_means(size_t(batch)*fixed_ppo::action_dim);
    fixed_ppo::ActorParams lifted{};
    std::copy(lifted_weights,lifted_weights+nav_deployment::raw_depth_actor_weight_count,lifted.values.begin());
    for(uint32_t sample=0;sample<16;sample++) {
        const auto old_obs=deployment_probe_observation(sample);
        std::array<float,320> current_raw{},previous_raw{};
        for(uint32_t ray=0;ray<320;ray++) {
            current_raw[ray]=float((ray*17+sample*11)%119)/10.0f;
            previous_raw[ray]=float((ray*29+sample*7)%117)/10.0f;
        }
        std::array<float,nav_deployment::raw_depth_actor_observation_count> raw_obs{};
        std::copy(old_obs.begin(),old_obs.begin()+181,raw_obs.begin());
        std::copy(current_raw.begin(),current_raw.end(),raw_obs.begin()+nav_deployment::raw_current_range_offset);
        std::copy(previous_raw.begin(),previous_raw.end(),raw_obs.begin()+nav_deployment::raw_previous_range_offset);
        std::copy(old_obs.begin()+181,old_obs.end(),raw_obs.begin()+nav_deployment::raw_geometry_hint_offset);
        float old_mean[4],raw_mean[4];
        require(old_policy.raw_mean(old_obs.data(),old_mean),"legacy actor mean probe failed at sample="+std::to_string(sample)+
                " loaded="+std::to_string(old_policy.loaded())+" finite="+
                std::to_string(std::all_of(old_obs.begin(),old_obs.end(),[](float x){return std::isfinite(x);} )));
        require(raw_policy.raw_mean(raw_obs.data(),raw_mean),"raw-depth actor mean probe failed");
        nav_deployment::NavigationAction old_command,raw_command;
        require(old_policy.infer(old_obs.data(),old_command)&&raw_policy.infer(raw_obs.data(),raw_command),
                "lifted actor command probe failed");
        for(uint32_t action=0;action<4;action++)
            max_deployment_error=std::max(max_deployment_error,std::fabs(old_mean[action]-raw_mean[action]));
        for(uint32_t axis=0;axis<3;axis++) {
            max_deployment_error=std::max(max_deployment_error,
                std::fabs(old_command.body_velocity_mps[axis]-raw_command.body_velocity_mps[axis]));
            max_deployment_error=std::max(max_deployment_error,
                std::fabs(old_command.normalized_intent[axis]-raw_command.normalized_intent[axis]));
        }
        max_deployment_error=std::max(max_deployment_error,
            std::fabs(old_command.yaw_rate_rps-raw_command.yaw_rate_rps));
        // Different raw images must have no effect before those channels are learned.
        for(uint32_t ray=0;ray<320;ray++)raw_obs[nav_deployment::raw_current_range_offset+ray]=1.0f-current_raw[ray]/12.0f;
        nav_deployment::NavigationAction alternate;
        require(raw_policy.infer(raw_obs.data(),alternate),"alternate raw-image policy probe failed");
        for(uint32_t axis=0;axis<3;axis++)max_deployment_error=std::max(max_deployment_error,
            std::fabs(alternate.body_velocity_mps[axis]-old_command.body_velocity_mps[axis]));

        if(sample<batch) {
            // The shared native packer must preserve the v1 prefix and append
            // both raw channels and the hint at their declared version-2 offsets.
            float pooled[80],previous_pooled[80],context[21],hint[3],packed[824];
            for(uint32_t i=0;i<80;i++){pooled[i]=old_obs[i]*12.0f;previous_pooled[i]=old_obs[80+i]*12.0f;}
            std::copy(old_obs.begin()+160,old_obs.begin()+181,context);
            std::copy(old_obs.begin()+181,old_obs.end(),hint);
            nav_deployment::pack_raw_depth_observation(pooled,previous_pooled,context,
                current_raw.data(),previous_raw.data(),hint,packed);
            for(uint32_t i=0;i<181;i++)max_pack_error=std::max(max_pack_error,std::fabs(packed[i]-old_obs[i]));
            for(uint32_t i=0;i<320;i++) {
                max_pack_error=std::max(max_pack_error,std::fabs(packed[181+i]-current_raw[i]/12.0f));
                max_pack_error=std::max(max_pack_error,std::fabs(packed[501+i]-previous_raw[i]/12.0f));
            }
            for(uint32_t i=0;i<3;i++)max_pack_error=std::max(max_pack_error,std::fabs(packed[821+i]-hint[i]));
        }
        if(sample<batch) {
            std::copy(raw_obs.begin(),raw_obs.end(),metal_observations.begin()+size_t(sample)*824);
            float hidden[fixed_ppo::hidden_dim],mean[fixed_ppo::action_dim];
            fixed_ppo::actor_forward(raw_obs.data(),lifted,hidden,mean);
            std::copy(mean,mean+fixed_ppo::action_dim,expected_means.begin()+size_t(sample)*fixed_ppo::action_dim);
        }
    }
    const uint32_t count=batch;
    auto obs_buffer=metal.buffer(metal_observations.size()*sizeof(float),metal_observations.data());
    auto param_buffer=metal.buffer(nav_deployment::raw_depth_actor_weight_count*sizeof(float),lifted_weights);
    auto hidden_buffer=metal.buffer(size_t(batch)*fixed_ppo::hidden_dim*sizeof(float));
    auto mean_buffer=metal.buffer(expected_means.size()*sizeof(float));
    auto batch_buffer=metal.buffer(sizeof(count),&count);
    auto cb=[metal.queue commandBuffer];
    encode(metal,cb,"ppo_actor_forward_simd_fused",256,
        {{obs_buffer,0},{param_buffer,0},{hidden_buffer,0},{mean_buffer,0},{batch_buffer,0}},256);
    metal.finish(cb);
    const float* metal_means=static_cast<const float*>(mean_buffer.contents);
    for(size_t i=0;i<expected_means.size();i++)max_metal_error=std::max(max_metal_error,std::fabs(metal_means[i]-expected_means[i]));
    require(max_deployment_error<2.0e-5f&&max_pack_error<2.0e-7f&&max_metal_error<2.0e-5f,
            "raw-depth lifted policy or frontend parity failed");
    std::remove(old_policy_path.c_str());
    std::cout<<"raw-depth lift PASS deployment_command_error="<<max_deployment_error
             <<" pack_error="<<max_pack_error<<" cpu_metal_mean_error="<<max_metal_error<<"\n";
    return std::max(max_deployment_error,std::max(max_pack_error,max_metal_error));
}

static void raw_depth_canary_check(Metal& metal,const std::string& native_shadow_csv) {
    require(fixed_ppo::actor_obs_dim==824,"raw-depth canary check requires the824-input build");
    std::ifstream input(native_shadow_csv);
    require(bool(input),"cannot open native raw-depth shadow CSV: "+native_shadow_csv);
    std::string line;require(bool(std::getline(input,line)),"raw-depth shadow CSV has no header");
    std::vector<std::string> header;{
        std::stringstream row(line);std::string field;
        while(std::getline(row,field,','))header.push_back(field);
    }
    std::array<int,nav_deployment::raw_depth_actor_observation_count> observation_columns{};
    for(uint32_t i=0;i<observation_columns.size();i++) {
        const std::string name="observation_"+std::to_string(i);
        const auto found=std::find(header.begin(),header.end(),name);
        require(found!=header.end(),"native shadow is missing "+name);
        observation_columns[i]=int(found-header.begin());
    }
    std::array<float,824> selected_observation{};float largest_raw_pair_delta=0.0f;
    uint32_t selected_pixel=0;std::string selected_step;
    while(std::getline(input,line)) {
        std::stringstream row(line);std::string field;std::vector<std::string> values;
        while(std::getline(row,field,','))values.push_back(field);
        if(values.size()!=header.size())continue;
        std::array<float,824> observation{};
        for(uint32_t i=0;i<observation.size();i++)observation[i]=std::stof(values[observation_columns[i]]);
        for(uint32_t ray=0;ray<320;ray++) {
            const float difference=std::fabs(observation[181+ray]-observation[501+ray]);
            if(difference>largest_raw_pair_delta) {
                largest_raw_pair_delta=difference;selected_pixel=ray;selected_observation=observation;
                const auto step_column=std::find(header.begin(),header.end(),"step");
                selected_step=step_column==header.end()?"unknown":values[size_t(step_column-header.begin())];
            }
        }
    }
    require(largest_raw_pair_delta>0.1f,"native raw current/previous frames did not expose a canary difference");
    fixed_ppo::ActorParams canary{};
    canary.values[nav_deployment::raw_current_range_offset+selected_pixel]=1.0f;
    canary.values[fixed_ppo::actor_obs_dim+nav_deployment::raw_previous_range_offset+selected_pixel]=1.0f;
    canary.values[fixed_ppo::actor_w2_offset+0*fixed_ppo::hidden_dim+0]=1.0f;
    canary.values[fixed_ppo::actor_w2_offset+1*fixed_ppo::hidden_dim+1]=1.0f;
    for(uint32_t j=0;j<3;j++)selected_observation[821+j]=0.0f;
    constexpr uint32_t batch=3;
    std::array<float,batch*824> observations{};
    for(uint32_t i=0;i<824;i++)observations[i]=selected_observation[i];
    std::copy(selected_observation.begin(),selected_observation.end(),observations.begin()+824);
    observations[824+nav_deployment::raw_current_range_offset+selected_pixel]=
        selected_observation[nav_deployment::raw_previous_range_offset+selected_pixel];
    std::copy(selected_observation.begin(),selected_observation.end(),observations.begin()+2*824);
    observations[2*824+nav_deployment::raw_previous_range_offset+selected_pixel]=
        selected_observation[nav_deployment::raw_current_range_offset+selected_pixel];
    std::array<float,batch*4> cpu_means{};
    for(uint32_t sample=0;sample<batch;sample++) {
        float hidden[fixed_ppo::hidden_dim],mean[4];
        fixed_ppo::actor_forward(observations.data()+size_t(sample)*824,canary,hidden,mean);
        std::copy(mean,mean+4,cpu_means.begin()+sample*4);
    }
    const uint32_t count=batch;
    auto obs_buffer=metal.buffer(observations.size()*sizeof(float),observations.data());
    auto actor_buffer=metal.buffer(sizeof(canary),&canary);
    auto hidden_buffer=metal.buffer(size_t(batch)*fixed_ppo::hidden_dim*sizeof(float));
    auto mean_buffer=metal.buffer(cpu_means.size()*sizeof(float));
    auto count_buffer=metal.buffer(sizeof(count),&count);
    auto cb=[metal.queue commandBuffer];
    encode(metal,cb,"ppo_actor_forward_simd_fused",256,
        {{obs_buffer,0},{actor_buffer,0},{hidden_buffer,0},{mean_buffer,0},{count_buffer,0}},256);
    metal.finish(cb);
    const float* gpu_means=static_cast<const float*>(mean_buffer.contents);
    float cpu_metal_error=0.0f;
    for(size_t i=0;i<cpu_means.size();i++)cpu_metal_error=std::max(cpu_metal_error,std::fabs(cpu_means[i]-gpu_means[i]));
    const float current_axis_delta=std::fabs(cpu_means[0]-cpu_means[4]);
    const float previous_axis_delta=std::fabs(cpu_means[1]-cpu_means[9]);
    const float current_cross_delta=std::fabs(cpu_means[1]-cpu_means[5]);
    const float previous_cross_delta=std::fabs(cpu_means[0]-cpu_means[8]);
    std::cout<<"raw_depth_canary_probe pixel="<<selected_pixel<<" current_previous_delta="<<largest_raw_pair_delta
             <<" current_effect="<<current_axis_delta<<" previous_effect="<<previous_axis_delta
             <<" cross="<<std::max(current_cross_delta,previous_cross_delta)
             <<" means="<<cpu_means[0]<<','<<cpu_means[1]<<';'<<cpu_means[4]<<','<<cpu_means[5]
             <<';'<<cpu_means[8]<<','<<cpu_means[9]
             <<" cpu_metal="<<cpu_metal_error<<"\n";
    require(cpu_metal_error<2.0e-5f&&current_axis_delta>0.05f&&previous_axis_delta>0.05f&&
            current_cross_delta<1.0e-7f&&previous_cross_delta<1.0e-7f,
            "raw current/previous canary channels failed independent CPU/Metal checks");
    std::cout<<"raw_depth_canary PASS native_step="<<selected_step<<" pixel="<<selected_pixel
             <<" current_previous_delta="<<largest_raw_pair_delta
             <<" current_axis_effect="<<current_axis_delta<<" previous_axis_effect="<<previous_axis_delta
             <<" cross_effect="<<std::max(current_cross_delta,previous_cross_delta)
             <<" cpu_metal_mean_error="<<cpu_metal_error<<"\n";
}

static void export_raw_depth_checkpoint(const std::string& checkpoint_path,const std::string& raw_policy_path) {
    require(fixed_ppo::actor_obs_dim==824,"raw-depth export requires the824-input build");
    std::ifstream file(checkpoint_path,std::ios::binary);
    require(bool(file),"cannot open raw-depth checkpoint");
    const PpoCheckpointHeader header=read_checkpoint_header(file);
    require(header.version>=3&&header.version<=10&&header.actor_count==nav_deployment::raw_depth_actor_weight_count,
            "raw-depth export requires an 824-input checkpoint");
    std::vector<float> actor(header.actor_count);
    file.read(reinterpret_cast<char*>(actor.data()),std::streamsize(actor.size()*sizeof(float)));
    require(bool(file),"raw-depth checkpoint actor is truncated");
    const RLPhysicsParams dynamics=rl_physics_crazyflie_default();
    nav_deployment::Metadata metadata;metadata.version=2;
    metadata.observation_count=nav_deployment::raw_depth_actor_observation_count;
    metadata.weight_count=nav_deployment::raw_depth_actor_weight_count;
    metadata.max_speed_mps=header.config.speed;
    metadata.navigation_period_s=dynamics.dt*float(header.config.substeps);
    metadata.native_period_s=dynamics.dt;
    std::string error;
    require(nav_deployment::RawDepthNavigationPolicy::write_file(raw_policy_path,actor.data(),actor.size(),
                metadata,checkpoint_path,&error),error);
    nav_deployment::RawDepthNavigationPolicy policy;require(policy.load(raw_policy_path,&error),error);
    std::cout<<"RAW NAV export="<<raw_policy_path<<" source="<<checkpoint_path
             <<" observations="<<metadata.observation_count<<" weights="<<metadata.weight_count
             <<" mode="<<metadata.policy_mode<<" speed="<<metadata.max_speed_mps
             <<" source_fnv64="<<std::hex<<policy.source_checkpoint_hash()<<std::dec<<"\n";
}

// Export an 824-input checkpoint under the calibrated native sensor contract.
// The actor weights are unchanged (param-only); only the declared camera
// profile differs, and the file carries it so the controller can verify it.
static void export_calibrated_checkpoint(const std::string& checkpoint_path,const std::string& policy_path) {
    require(fixed_ppo::actor_obs_dim==824,"calibrated export requires the824-input build");
    require(NAV_SENSOR_ACTIVE_ID==NAV_SENSOR_NATIVE_ID,
            "calibrated export requires a binary built for the native sensor profile");
    std::ifstream file(checkpoint_path,std::ios::binary);
    require(bool(file),"cannot open calibrated checkpoint");
    const PpoCheckpointHeader header=read_checkpoint_header(file);
    require(header.version>=3&&header.version<=10&&header.actor_count==nav_deployment::raw_depth_actor_weight_count,
            "calibrated export requires an 824-input checkpoint");
    std::vector<float> actor(header.actor_count);
    file.read(reinterpret_cast<char*>(actor.data()),std::streamsize(actor.size()*sizeof(float)));
    require(bool(file),"calibrated checkpoint actor is truncated");
    const RLPhysicsParams dynamics=rl_physics_crazyflie_default();
    nav_deployment::Metadata metadata;metadata.version=3;
    metadata.observation_count=nav_deployment::raw_depth_actor_observation_count;
    metadata.weight_count=nav_deployment::raw_depth_actor_weight_count;
    metadata.max_speed_mps=header.config.speed;
    metadata.navigation_period_s=dynamics.dt*float(header.config.substeps);
    metadata.native_period_s=dynamics.dt;
    nav_deployment::SensorContract contract;
    contract.profile_id=NAV_SENSOR_NATIVE_ID;
    contract.tan_h=NAV_SENSOR_TAN_H;
    contract.tan_v=NAV_SENSOR_NATIVE_TAN_V;
    contract.mount_x_m=NAV_SENSOR_NATIVE_MOUNT_X;
    contract.min_range_m=NAV_SENSOR_MIN_RANGE_M;
    contract.max_range_m=NAV_SENSOR_MAX_RANGE_M;
    contract.noise_m=0.0f;
    std::string error;
    require(nav_deployment::RawDepthNavigationPolicy::write_calibrated_file(policy_path,actor.data(),actor.size(),
                metadata,contract,checkpoint_path,&error),error);
    nav_deployment::RawDepthNavigationPolicy policy;require(policy.load(policy_path,&error),error);
    require(policy.has_sensor_contract()&&nav_deployment::valid_native_contract(policy.sensor_contract()),
            "calibrated export did not round-trip its sensor contract");
    std::cout<<"CALIBRATED NAV export="<<policy_path<<" source="<<checkpoint_path
             <<" profile="<<policy.sensor_contract().profile_id
             <<" tan_v="<<policy.sensor_contract().tan_v
             <<" mount_x_m="<<policy.sensor_contract().mount_x_m
             <<" observations="<<metadata.observation_count<<" weights="<<metadata.weight_count
             <<" mode="<<metadata.policy_mode<<" speed="<<metadata.max_speed_mps
             <<" source_fnv64="<<std::hex<<policy.source_checkpoint_hash()<<std::dec<<"\n";
}

static void lift_guided_checkpoint_to_raw_depth(Metal& metal,const std::string& old_checkpoint,
                                                const std::string& new_checkpoint,
                                                const std::string& raw_policy_path) {
    require(fixed_ppo::actor_obs_dim==824,"raw-depth lift requires the824-input build");
    std::ifstream source(old_checkpoint,std::ios::binary);
    require(bool(source),"cannot open v1 guided checkpoint");
    const PpoCheckpointHeader old_header=read_checkpoint_header(source);
    require(old_header.version>=3&&old_header.version<=10&&old_header.actor_count==nav_deployment::actor_weight_count&&
            old_header.critic_count==fixed_ppo::critic_param_count,
            "raw-depth lift accepts only a v1 184-input guided checkpoint");
    require(old_header.config.velocity_contract==1&&old_header.config.geometry_memory==1&&
            old_header.config.sensor_period==1&&old_header.config.substeps==5&&
            std::fabs(old_header.config.speed-1.5f)<1.0e-6f,
            "raw-depth lift requires the selected guided sensor/motion contract");
    std::vector<float> old_actor(old_header.actor_count);
    std::vector<float> critic(old_header.critic_count);
    source.read(reinterpret_cast<char*>(old_actor.data()),std::streamsize(old_actor.size()*sizeof(float)));
    source.read(reinterpret_cast<char*>(critic.data()),std::streamsize(critic.size()*sizeof(float)));
    require(bool(source),"source guided actor or critic is truncated");
    std::vector<float> raw_actor(nav_deployment::raw_depth_actor_weight_count);
    std::string error;
    require(nav_deployment::lift_guided_actor_to_raw_depth(old_actor.data(),old_actor.size(),
                raw_actor.data(),raw_actor.size(),&error),error);

    metal.compile(base_source()+PPO_TRAINER_MSL);
    SimConfig config=old_header.config;
    require(config.n>0&&config.n<=32768&&old_header.horizon>0,
            "source guided checkpoint simulator dimensions are unsupported");
    Sim sim(metal,config,old_header.horizon);
    std::memcpy(sim.actor.contents,raw_actor.data(),raw_actor.size()*sizeof(float));
    std::memcpy(sim.critic.contents,critic.data(),critic.size()*sizeof(float));
    PPOTrainer trainer(sim,2);
    trainer.save_checkpoint(new_checkpoint,old_header.family,old_header.base_seed,0);
    export_raw_depth_checkpoint(new_checkpoint,raw_policy_path);
    const float parity=verify_raw_depth_lift(metal,old_actor.data(),raw_actor.data(),old_checkpoint,
                                              new_checkpoint,raw_policy_path,old_header);
    std::cout<<"raw_depth_lift old="<<old_checkpoint<<" new_checkpoint="<<new_checkpoint
             <<" asset="<<raw_policy_path<<" weights="<<raw_actor.size()
             <<" parity_error="<<parity<<"\n";
}

static EvalScore evaluate_training_policy(Metal& m,const SimConfig& cfg,uint32_t mode,const float* actor) {
    if(cfg.family!=12 && cfg.family!=13)return sim_evaluate(m,cfg.family,mode,actor,cfg.speed,cfg.distance,700001,cfg.sensor_delay,cfg.wind,cfg.depth_noise,cfg.dropout,cfg.command_delay,cfg.velocity_contract,cfg.geometry_memory);
    std::vector<uint32_t> families={4,5,7};if(cfg.family==13){families.push_back(10);families.push_back(11);}
    EvalScore result{1,0,0,0};uint32_t profiles=0;
    for(uint32_t family:families)for(bool stressed:{false,true}) {
        const auto score=sim_evaluate(m,family,mode,actor,cfg.speed,cfg.distance,700001,stressed?cfg.sensor_delay:0,stressed?cfg.wind:0,stressed?cfg.depth_noise:0,stressed?cfg.dropout:0,stressed?cfg.command_delay:0,cfg.velocity_contract,cfg.geometry_memory);
        result.success=std::min(result.success,score.success);result.collision=std::max(result.collision,score.collision);result.timeout=std::max(result.timeout,score.timeout);result.goal_time+=score.goal_time;profiles++;
    }
    result.goal_time/=profiles;
    std::cout<<"selection profiles="<<profiles<<" minimum_success="<<result.success<<" worst_collision="<<result.collision<<" mode="<<mode<<"\n";
    return result;
}

static void train_navigation(Metal& m,uint32_t iterations,uint32_t family,const std::string& checkpoint,const std::string& warmstart="",float speed=1,float distance=3,float risk=0,float entropy=-1,float learning_rate=.0003f,uint32_t velocity_contract=1,uint32_t memory=0,uint32_t sensor_delay=0,float wind=0,float noise=0,float dropout=0,uint32_t command_delay=0,uint32_t evaluation_mode=4,float warmstart_log_std=-1) {
    require(iterations>0,"train_navigation needs at least one rollout");
    require(std::isfinite(warmstart_log_std) && warmstart_log_std>=-2 && warmstart_log_std<=.5f,"warmstart log std outside supported range");
    require((family!=10 && family!=11 && family!=13) || (std::fabs(speed-1.5f)<1e-6f && std::fabs(distance-4.0f)<1e-6f),"flying-threat families require 1.5m/s and a4m goal");
    require(sensor_delay<=6 && command_delay<=7,"training delay exceeds the sensor/command rings");
    require(std::isfinite(speed) && speed>0 && std::isfinite(distance) && distance>1 && std::isfinite(wind) && std::isfinite(noise) && noise>=0 && std::isfinite(dropout) && dropout>=0 && dropout<=1,"invalid training scene/corruption parameters");
    require(evaluation_mode==4 || ((fixed_ppo::actor_obs_dim==184||fixed_ppo::actor_obs_dim==824) && evaluation_mode==17),"training selection supports learned mean4 or guided mode17");
    m.compile(base_source()+PPO_TRAINER_MSL);
    constexpr uint32_t horizon=32,base_seed=42;
    uint32_t n=128;
    if(const char* value=std::getenv("METAL_NAV_TRAIN_ENVS")) {
        size_t parsed=0;const auto requested=std::stoul(value,&parsed);
        require(parsed==std::strlen(value) && requested>0 && requested<=32768,"METAL_NAV_TRAIN_ENVS must be1..32768");n=uint32_t(requested);
    } else if(!checkpoint.empty() && std::filesystem::exists(checkpoint)) {
        std::ifstream file(checkpoint,std::ios::binary);n=read_checkpoint_header(file).n;
        require(n>0 && n<=32768,"checkpoint environment count unsupported");
    }
    SimConfig cfg;cfg.n=n;cfg.family=family;cfg.mode=0;cfg.seed=base_seed;cfg.distance=distance;cfg.speed=speed;cfg.max_steps=(family>=14 && family<=16)?400:200;cfg.risk_coef=risk;cfg.entropy_coef=entropy>=0?entropy:(family==7?.002f:.005f);cfg.learning_rate=learning_rate;cfg.velocity_contract=velocity_contract;cfg.geometry_memory=memory;cfg.sensor_delay=sensor_delay;cfg.wind=wind;cfg.depth_noise=noise;cfg.dropout=dropout;cfg.command_delay=command_delay;
    Sim sim(m,cfg,horizon);PPOTrainer trainer(sim);trainer.load_checkpoint(checkpoint,family,horizon,n,base_seed);
    if(!warmstart.empty() && trainer.completed_rollouts==0) {
        std::ifstream f(warmstart,std::ios::binary);PpoCheckpointHeader h=read_checkpoint_header(f);
        require(f && (h.version>=3&&h.version<=10) && h.actor_count==fixed_ppo::actor_param_count && h.critic_count==fixed_ppo::critic_param_count,"warmstart checkpoint format mismatch");
        f.read((char*)sim.actor.contents,fixed_ppo::actor_param_count*4);f.read((char*)sim.critic.contents,fixed_ppo::critic_param_count*4);require(bool(f),"warmstart read failed");
        for(uint j=0;j<4;j++)((float*)sim.actor.contents)[fixed_ppo::actor_log_std_offset+j]=warmstart_log_std;
        std::cout<<"warmstart parameters from "<<warmstart<<"; reset optimizer, exploration_log_std="<<warmstart_log_std<<"\n";
    }
    EvalScore best;
    if(!checkpoint.empty() && std::filesystem::exists(checkpoint+".best")) {
        std::ifstream f(checkpoint+".best",std::ios::binary);PpoCheckpointHeader h=read_checkpoint_header(f);std::vector<float> a(fixed_ppo::actor_param_count);f.read((char*)a.data(),a.size()*4);require(f && h.actor_count==fixed_ppo::actor_param_count,"best checkpoint read/dimensions");best=evaluate_training_policy(m,cfg,evaluation_mode,a.data());
    }
    if(!checkpoint.empty() && !std::filesystem::exists(checkpoint+".best")) {
        best=evaluate_training_policy(m,cfg,evaluation_mode,(const float*)sim.actor.contents);
        trainer.save_checkpoint(checkpoint+".best",family,base_seed,trainer.completed_rollouts);
    }
    std::filesystem::create_directories("results");
    std::ofstream history("results/training.tsv",std::ios::app);
    const double run_start=seconds();
    const uint32_t start_rollout=trainer.completed_rollouts;
    const uint32_t finish_rollout=start_rollout+iterations;
    std::cout<<"PPO training device="<<m.device.name.UTF8String<<" family="<<family<<" envs="<<n<<" horizon="<<horizon<<" speed="<<cfg.speed<<" distance="<<cfg.distance<<" risk="<<cfg.risk_coef<<" entropy="<<cfg.entropy_coef<<" lr="<<cfg.learning_rate<<" sensor_delay="<<cfg.sensor_delay<<" command_delay="<<cfg.command_delay<<" wind_accel="<<cfg.wind<<" depth_noise="<<cfg.depth_noise<<" dropout="<<cfg.dropout<<" selection_mode="<<evaluation_mode<<"\n";
    for(uint32_t r=start_rollout;r<finish_rollout;r++) {@autoreleasepool{
        if(m.profiler.available)m.profiler.arm();const double rollout_start=seconds();auto cb=[m.queue commandBuffer];sim.collect(cb,horizon);trainer.rollout_update(cb,r);
        const double gpu=m.finish(cb);
        const float* met=(const float*)trainer.metric_mean.contents;
        if((r+1)%((family==12||family==13)?25:10)==0 || r+1==finish_rollout) {
            std::cout<<"train rollout="<<(r+1)<<" policy_loss="<<met[0]<<" value_loss="<<met[1]<<" entropy_loss="<<met[2]<<" ratio="<<met[3]<<" optimizer_step="<<trainer.optimizer_step<<" gpu_s="<<gpu<<"\n";
            sim.report("train",0.0);
            EvalScore score=evaluate_training_policy(m,cfg,evaluation_mode,(const float*)sim.actor.contents);
            history<<checkpoint<<'\t'<<r+1<<'\t'<<seconds()-run_start<<'\t'<<gpu<<'\t'<<score.success<<'\t'<<score.collision<<'\t'<<score.timeout<<'\t'<<score.goal_time<<'\n';history.flush();
            if(score.success>best.success || (score.success==best.success && score.goal_time<best.goal_time)) {
                best=score;trainer.completed_rollouts=r+1;trainer.save_checkpoint(checkpoint+".best",family,base_seed,r+1);
            }
            trainer.completed_rollouts=r+1;trainer.save_checkpoint(checkpoint,family,base_seed,r+1);
        }
        (void)rollout_start;
    }}
    trainer.completed_rollouts=finish_rollout;trainer.save_checkpoint(checkpoint,family,base_seed,finish_rollout);
}

static void evaluate_checkpoint(Metal& m,const std::string& checkpoint,uint mode,uint family,uint32_t seed=800001,float speed=1,float distance=3,uint32_t sensor_delay=0,float wind=0,float noise=0,float dropout=0,uint32_t command_delay=0,uint32_t max_steps=0) {
    std::ifstream f(checkpoint,std::ios::binary);PpoCheckpointHeader h=read_checkpoint_header(f);require(f && (h.version>=3&&h.version<=10) && h.actor_count==fixed_ppo::actor_param_count,"evaluation checkpoint format");std::vector<float>a(h.actor_count);f.read((char*)a.data(),a.size()*4);require(bool(f),"evaluation policy read");sim_evaluate(m,family,mode,a.data(),speed,distance,seed,sensor_delay,wind,noise,dropout,command_delay,h.config.velocity_contract,h.config.geometry_memory,max_steps);
}

// Depends on Sim, SimRun, SimConfig, PpoCheckpointHeader and
// read_checkpoint_header above.
#include "challenge_evaluation.hpp"
#include "challenge_training.hpp"
#include "training_potential.hpp"
#include "navigation_training.hpp"
#include "navigation_contact_audit.hpp"
#include "reward_audit.hpp"
#if FIXED_PPO_ACTOR_OBS_DIM==184
#include "webots/simulator_comparison.hpp"
#endif

// Non-default PPO update guards live beside each checkpoint. Legacy checkpoints
// need no sidecar when the update settings are the historical defaults.
struct PpoSafeguardConfig {
    uint32_t epochs;
    float advantage_clip_sd;
    float value_coefficient;
    float anchor_lambda;
    float anchor_radius;
    float direction_aux_coefficient;
    std::string direction_field_sha256;
};
struct PpoSafeguardHeaderV1 {
    char magic[8];
    uint32_t version, epochs, actor_count, anchor_reference_bytes, completed_rollouts;
    float advantage_clip_sd, value_coefficient, anchor_lambda, anchor_radius;
    char checkpoint_sha256[64], anchor_reference_sha256[64];
};
struct PpoSafeguardHeader {
    char magic[8];
    uint32_t version, epochs, actor_count, anchor_reference_bytes, completed_rollouts;
    float advantage_clip_sd, value_coefficient, anchor_lambda, anchor_radius, direction_aux_coefficient;
    char checkpoint_sha256[64], anchor_reference_sha256[64];
    char direction_field_sha256[64];
};
static_assert(sizeof(PpoSafeguardHeaderV1)==172,"PPO safeguard v1 sidecar layout changed");
static_assert(sizeof(PpoSafeguardHeader)==240,"PPO safeguard v2 sidecar layout changed");

static bool ppo_safeguard_is_default(const PpoSafeguardConfig& config) {
    return config.epochs==2 && config.advantage_clip_sd==0.0f &&
           config.value_coefficient==0.5f && config.anchor_lambda==0.0f &&
           config.anchor_radius==0.0f && config.direction_aux_coefficient==0.0f;
}

static std::string ppo_safeguard_sha256(const void* data,size_t size) {
    unsigned char digest[CC_SHA256_DIGEST_LENGTH];
    CC_SHA256(static_cast<const unsigned char*>(data),CC_LONG(size),digest);
    static const char hex[]="0123456789abcdef";
    std::string output(64,'0');
    for(size_t i=0;i<sizeof(digest);i++){
        output[2*i]=hex[digest[i]>>4];output[2*i+1]=hex[digest[i]&15];
    }
    return output;
}

static std::string load_ppo_safeguard_state(const std::string& checkpoint,
                                            const PpoSafeguardConfig& expected,
                                            uint32_t completed_rollouts,
                                            void* anchor_reference_destination=nullptr) {
    const std::string state_path=checkpoint+".safeguard";
    if(!std::filesystem::exists(state_path)) {
        require(ppo_safeguard_is_default(expected),
                "non-default PPO safeguards require a matching .safeguard sidecar; use an explicit warmstart");
        return std::string(64,'0');
    }
    std::ifstream state(state_path,std::ios::binary);
    require(bool(state),"cannot open PPO safeguard state: "+state_path);
    PpoSafeguardHeader header{};
    struct Prefix { char magic[8]; uint32_t version; } prefix{};
    state.read(reinterpret_cast<char*>(&prefix),sizeof(prefix));
    require(bool(state) && std::memcmp(prefix.magic,"PPOSFG1\0",8)==0,
            "invalid PPO safeguard sidecar");
    state.clear();state.seekg(0);
    if(prefix.version==1) {
        PpoSafeguardHeaderV1 legacy{};
        state.read(reinterpret_cast<char*>(&legacy),sizeof(legacy));
        require(bool(state) && expected.direction_aux_coefficient==0.0f,
                "legacy PPO safeguard cannot resume a geodesic auxiliary run");
        std::memcpy(header.magic,legacy.magic,8);header.version=legacy.version;
        header.epochs=legacy.epochs;header.actor_count=legacy.actor_count;
        header.anchor_reference_bytes=legacy.anchor_reference_bytes;
        header.completed_rollouts=legacy.completed_rollouts;
        header.advantage_clip_sd=legacy.advantage_clip_sd;
        header.value_coefficient=legacy.value_coefficient;header.anchor_lambda=legacy.anchor_lambda;
        header.anchor_radius=legacy.anchor_radius;header.direction_aux_coefficient=0.0f;
        std::memcpy(header.checkpoint_sha256,legacy.checkpoint_sha256,64);
        std::memcpy(header.anchor_reference_sha256,legacy.anchor_reference_sha256,64);
        std::memset(header.direction_field_sha256,'0',64);
    } else if(prefix.version==2) {
        state.read(reinterpret_cast<char*>(&header),sizeof(header));
        require(bool(state),"truncated PPO safeguard v2 sidecar");
    } else {
        require(false,"unsupported PPO safeguard sidecar version");
    }
    // Safeguard v1 predates the geodesic auxiliary and has no field hash.
    // A v1 file may resume only with auxiliary coefficient zero; potential
    // shaping, when enabled, is independently bound by PPO checkpoint v9.
    const bool legacy_without_direction_hash=prefix.version==1u && expected.direction_aux_coefficient==0.0f;
    const bool direction_hash_matches=legacy_without_direction_hash ||
        (expected.direction_field_sha256.size()==64 &&
         std::memcmp(header.direction_field_sha256,expected.direction_field_sha256.data(),64)==0);
    require(header.epochs==expected.epochs && header.advantage_clip_sd==expected.advantage_clip_sd &&
            header.value_coefficient==expected.value_coefficient && header.anchor_lambda==expected.anchor_lambda &&
            header.anchor_radius==expected.anchor_radius &&
            header.direction_aux_coefficient==expected.direction_aux_coefficient &&
            direction_hash_matches,
            "PPO safeguard settings differ from the saved run; use an explicit warmstart");
    const std::string checkpoint_hash=challenge_evaluation::sha256_file(checkpoint);
    require(header.actor_count==fixed_ppo::actor_param_count &&
            header.completed_rollouts==completed_rollouts &&
            checkpoint_hash.size()==64 && std::memcmp(header.checkpoint_sha256,checkpoint_hash.data(),64)==0,
            "PPO safeguard sidecar does not match checkpoint or rollout");
    const uint32_t expected_reference_bytes=expected.anchor_lambda>0.0f
        ? uint32_t(fixed_ppo::actor_param_count*sizeof(float)):0u;
    require(header.anchor_reference_bytes==expected_reference_bytes,
            "PPO safeguard sidecar anchor-reference layout mismatch");
    std::string reference_hash(64,'0');
    if(expected_reference_bytes>0) {
        const std::string reference_path=checkpoint+".anchorref";
        std::ifstream reference(reference_path,std::ios::binary);
        require(bool(reference),"active PPO anchor is missing its saved .anchorref: "+reference_path);
        std::vector<char> bytes(expected_reference_bytes);
        reference.read(bytes.data(),bytes.size());
        require(bool(reference) && reference.peek()==EOF,"active PPO anchor reference is truncated or has trailing bytes");
        reference_hash=ppo_safeguard_sha256(bytes.data(),bytes.size());
        require(std::memcmp(header.anchor_reference_sha256,reference_hash.data(),64)==0,
                "PPO anchor reference hash differs from safeguard sidecar");
        if(anchor_reference_destination)std::memcpy(anchor_reference_destination,bytes.data(),bytes.size());
    } else {
        const std::string zeros(64,'0');
        const std::array<char,64> empty_hash{};
        require(std::memcmp(header.anchor_reference_sha256,zeros.data(),64)==0 ||
                std::memcmp(header.anchor_reference_sha256,empty_hash.data(),64)==0,
                "disabled PPO anchor has an unexpected reference hash");
    }
    char extra;
    require(!state.read(&extra,1),"PPO safeguard sidecar has trailing bytes");
    return reference_hash;
}

static void save_ppo_safeguard_state(const PPOTrainer& trainer,
                                     const std::string& checkpoint,
                                     const PpoSafeguardConfig& config,
                                     uint32_t completed_rollouts) {
    const std::string state_path=checkpoint+".safeguard";
    const bool persist=!ppo_safeguard_is_default(config)||std::filesystem::exists(state_path);
    if(!persist)return;
    PpoSafeguardHeader header{};
    std::memcpy(header.magic,"PPOSFG1\0",8);header.version=2;
    header.epochs=config.epochs;header.actor_count=uint32_t(fixed_ppo::actor_param_count);
    header.completed_rollouts=completed_rollouts;
    header.advantage_clip_sd=config.advantage_clip_sd;header.value_coefficient=config.value_coefficient;
    header.anchor_lambda=config.anchor_lambda;header.anchor_radius=config.anchor_radius;
    header.direction_aux_coefficient=config.direction_aux_coefficient;
    require(config.direction_field_sha256.size()==64,"PPO direction field hash must be SHA-256 hex");
    std::memcpy(header.direction_field_sha256,config.direction_field_sha256.data(),64);
    std::memset(header.anchor_reference_sha256,'0',64);
    const std::string checkpoint_hash=challenge_evaluation::sha256_file(checkpoint);
    std::memcpy(header.checkpoint_sha256,checkpoint_hash.data(),64);
    if(config.anchor_lambda>0.0f) {
        const size_t bytes=fixed_ppo::actor_param_count*sizeof(float);
        require(bytes<=UINT32_MAX,"PPO anchor reference is too large for safeguard sidecar");
        header.anchor_reference_bytes=uint32_t(bytes);
        const void* reference=trainer.anchor_ref_b.contents;
        const std::string reference_hash=ppo_safeguard_sha256(reference,bytes);
        std::memcpy(header.anchor_reference_sha256,reference_hash.data(),64);
        const std::string reference_path=checkpoint+".anchorref";
        const std::string reference_temporary=reference_path+".tmp";
        std::ofstream reference_file(reference_temporary,std::ios::binary|std::ios::trunc);
        require(bool(reference_file),"cannot write PPO anchor reference: "+reference_temporary);
        reference_file.write(static_cast<const char*>(reference),std::streamsize(bytes));
        reference_file.flush();require(bool(reference_file),"PPO anchor reference write failed");
        reference_file.close();
        require(std::rename(reference_temporary.c_str(),reference_path.c_str())==0,
                "cannot replace PPO anchor reference: "+reference_path);
    }
    const std::string temporary=state_path+".tmp";
    if(std::filesystem::path(state_path).has_parent_path())
        std::filesystem::create_directories(std::filesystem::path(state_path).parent_path());
    std::ofstream state(temporary,std::ios::binary|std::ios::trunc);
    require(bool(state),"cannot write PPO safeguard sidecar: "+temporary);
    state.write(reinterpret_cast<const char*>(&header),sizeof(header));
    state.flush();require(bool(state),"PPO safeguard sidecar write failed");state.close();
    require(std::rename(temporary.c_str(),state_path.c_str())==0,
            "cannot replace PPO safeguard sidecar: "+state_path);
}

static void compare_webots_scenes(Metal& metal,const std::string& actor_path,
                                 const std::string& metadata_directory,
                                 const std::string& output_path) {
#if FIXED_PPO_ACTOR_OBS_DIM==184
    const auto actor=simulator_comparison::load_deployed_actor(actor_path);
    std::vector<std::filesystem::path> scenes;
    for(const auto& entry:std::filesystem::directory_iterator(metadata_directory)) {
        const std::string name=entry.path().filename().string();
        if(entry.is_regular_file() && name.size()>13 && name.substr(name.size()-13)=="_bounded.json")
            scenes.push_back(entry.path());
    }
    std::sort(scenes.begin(),scenes.end());require(!scenes.empty(),"no bounded Webots scenes found");
    const std::filesystem::path destination(output_path);
    if(destination.has_parent_path())std::filesystem::create_directories(destination.parent_path());
    std::ofstream output(output_path);require(bool(output),"cannot write paired scene results");
    output<<"scene,family,seed,actor_sha256,scene_sha256,mode,success,collision,timeout,steps,time_s,path_m,goal_error_m,peak_speed_mps,min_clearance_m,budget_s,success_rule\n";
    const auto actor_hash=challenge_evaluation::sha256_file(actor_path);
    for(const auto& path:scenes) {
        const auto scene=simulator_comparison::load_bounded_webots_scene(path.string());
        const auto outcomes=simulator_comparison::compare_modes(metal,scene,actor,1.5f,160);
        for(const auto& episode:outcomes) {
            output<<path.filename().string()<<','<<episode.family_name<<','<<episode.seed<<','<<actor_hash
                  <<','<<challenge_evaluation::sha256_file(path.string())<<','<<episode.mode
                  <<','<<episode.success<<','<<episode.collision<<','<<episode.timeout<<','<<episode.steps
                  <<','<<episode.time_s<<','<<episode.path_m<<','<<episode.final_error_m
                  <<','<<episode.peak_speed_mps<<','<<episode.min_clearance_m<<",8,first_goal_region_entry\n";
        }
        output.flush();require(bool(output),"paired scene result write failed");
    }
    std::cout<<"paired_geometry Metal episodes="<<scenes.size()*2<<" output="<<output_path
             <<"; same geometry/bounds, remaining sensor/startup contracts audited separately\n";
#else
    (void)metal;(void)actor_path;(void)metadata_directory;(void)output_path;
    throw std::runtime_error("paired Webots comparison requires the184-input binary");
#endif
}


static challenge_evaluation::BankScore raw_depth_bank_evaluate(
        Metal& metal,const std::string& checkpoint_path,const std::string& bank_path,
        const std::string& split,const std::string& output_path,uint32_t mode=17,
        float speed=1.5f,uint32_t max_steps=400) {
    require(fixed_ppo::has_raw_depth_extension,"raw bank evaluation requires the raw guided build");
    return challenge_evaluation::run(metal,checkpoint_path,bank_path,split,output_path,mode,speed,max_steps);
}

static challenge_evaluation::BankScore run_navigation_bank_evaluation(
        Metal& metal,const std::string& checkpoint,const std::string& bank,
        const std::string& split,const std::string& output,uint32_t mode=17,
        float speed=1.5f,uint32_t max_steps=400) {
    if(fixed_ppo::actor_obs_dim==824)
        return raw_depth_bank_evaluate(metal,checkpoint,bank,split,output,mode,speed,max_steps);
    return challenge_evaluation::run(metal,checkpoint,bank,split,output,mode,speed,max_steps);
}

static void challenge_bank_cli(Metal& m,int argc,char** argv) {
    require(argc>=6,"bank-eval CHECKPOINT BANK_JSONL SPLIT OUTPUT_CSV [MODE=17] [SPEED=1.5] [MAX_STEPS=400]");
    const uint32_t mode=argc>6?uint32_t(std::stoul(argv[6])):17u;
    const float speed=argc>7?std::stof(argv[7]):1.5f;
    const uint32_t max_steps=argc>8?uint32_t(std::stoul(argv[8])):400u;
    run_navigation_bank_evaluation(m,argv[2],argv[3],argv[4],argv[5],mode,speed,max_steps);
}

struct GeodesicAuxTargetCheck { uint32_t valid=0;float max_body_error=0.0f; };
static GeodesicAuxTargetCheck validate_geodesic_aux_targets(
        const Sim& sim,const training_potential::TrainingPotentialFields& fields) {
    require(sim.direction_aux_enabled,"geodesic target buffer is disabled");
    const float* targets=static_cast<const float*>(sim.direction_aux_targets.contents);
    const uint32_t* levels=static_cast<const uint32_t*>(sim.direction_aux_level_ids.contents);
    const size_t rows=size_t(sim.cfg.n)*sim.horizon;
    GeodesicAuxTargetCheck result;
    uint32_t compared=0;
    for(size_t row=0;row<rows;row++) {
        const float* target=targets+row*16;
        if(!(target[3]>0.5f))continue;
        result.valid++;
        require(levels[row]<fields.spec.level_count,"geodesic target level id is outside the train field table");
        if(compared>=16)continue;
        const float* position=target+4;
        float world_direction[3];
        require(training_potential::training_potential_direction(fields.phi.data(),fields.spec,
                    levels[row],position,world_direction),"CPU geodesic direction rejected a valid GPU label");
        const float* rotation=target+7;
        float body[3]={rotation[0]*world_direction[0]+rotation[3]*world_direction[1]+rotation[6]*world_direction[2],
                       rotation[1]*world_direction[0]+rotation[4]*world_direction[1]+rotation[7]*world_direction[2],
                       rotation[2]*world_direction[0]+rotation[5]*world_direction[1]+rotation[8]*world_direction[2]};
        const float length=std::sqrt(body[0]*body[0]+body[1]*body[1]+body[2]*body[2]);
        require(length>1.0e-5f,"CPU body direction has zero length");
        for(uint32_t axis=0;axis<3;axis++)
            result.max_body_error=std::max(result.max_body_error,
                std::fabs(body[axis]/length-target[axis]));
        compared++;
    }
    require(compared>0,"no valid geodesic targets were available for CPU/GPU parity");
    require(result.max_body_error<2.0e-5f,"CPU/GPU geodesic body-direction mismatch");
    return result;
}

static void geodesic_direction_preflight(Metal& metal,const std::string& warmstart,
                                         const std::string& bank_path,const std::string& output_path) {
    require(fixed_ppo::actor_obs_dim==184,"geodesic direction preflight requires the184-input actor");
    metal.compile(base_source()+PPO_TRAINER_MSL);
    constexpr uint32_t envs=128,horizon=1;
    challenge_training::Settings settings;settings.environment_count=envs;settings.horizon=horizon;
    settings.sampler_seed=42;settings.selection=challenge_training::SelectionMode::UniformBank;
    settings.focus_family=14;settings.rehearsal_environments=0;
    challenge_training::ChallengeTraining sampler(bank_path,
        challenge_evaluation::sha256_file(std::string(SOURCE_DIR)+"/world.hpp"),settings);
    const auto& levels=sampler.levels();const auto& worlds=sampler.worlds();
    auto fields=training_potential::build_training_potential_fields(levels,
        challenge_evaluation::sha256_file(bank_path),
        challenge_evaluation::sha256_file(std::string(SOURCE_DIR)+"/world.hpp"),
        bank_path+".potential-v2.cache");
    const std::string field_hash=training_potential::sha256_hex(fields.phi.data(),fields.phi.size()*sizeof(float));

    SimConfig config;config.n=envs;config.seed=42;config.family=14;config.mode=22;
    config.speed=1.5f;config.distance=8.0f;config.max_steps=400;config.geometry_memory=1;
    config.risk_coef=0.0f;config.entropy_coef=0.0f;config.learning_rate=3.0e-4f;
    Sim sim(metal,config,horizon);
    sim.bank_worlds=metal.buffer(worlds.size()*sizeof(WWorld),worlds.data());
    const auto bank_control=sampler.control();std::memcpy(sim.bank_control.contents,&bank_control,sizeof(bank_control));
    const auto schedule=sampler.initial_schedule();
    std::memcpy(sim.bank_schedule.contents,schedule.data(),schedule.size()*sizeof(uint32_t));
    sim.potential_fields=metal.buffer(fields.phi.size()*sizeof(float),fields.phi.data());
    std::memcpy(sim.potential_spec.contents,&fields.spec,sizeof(fields.spec));
    sim.enable_direction_auxiliary();sim.reset();
    std::ifstream source(warmstart,std::ios::binary);
    require(bool(source),"cannot open geodesic preflight warmstart");
    const PpoCheckpointHeader header=read_checkpoint_header(source);
    require(header.actor_count==fixed_ppo::actor_param_count&&header.critic_count==fixed_ppo::critic_param_count,
            "geodesic preflight warmstart dimensions mismatch");
    source.read(static_cast<char*>(sim.actor.contents),sim.actor.length);
    source.read(static_cast<char*>(sim.critic.contents),sim.critic.length);
    require(bool(source),"geodesic preflight warmstart is truncated");

    auto commands=[metal.queue commandBuffer];sim.collect(commands,horizon);metal.finish(commands);
    const auto target_check=validate_geodesic_aux_targets(sim,fields);
    const float* target=static_cast<const float*>(sim.direction_aux_targets.contents);
    const uint32_t* target_levels=static_cast<const uint32_t*>(sim.direction_aux_level_ids.contents);
    const float* obs=static_cast<const float*>(sim.obs.contents);
    std::vector<uint8_t> seen(levels.size(),0);
    struct LevelMeasure { size_t id;bool target_valid;float cosine,min_clearance;std::array<float,3> witness,field; };
    std::vector<LevelMeasure> measurements;
    double inverse_error=0.0;uint32_t inverse_count=0;
    std::vector<double> actor_losses_before;
    for(uint32_t env=0;env<envs;env++) {
        const uint32_t level=target_levels[env];
        if(level>=levels.size() || seen[level])continue;
        seen[level]=1;
        const float* label=target+size_t(env)*16;
        const challenge_evaluation::Level& entry=levels[level];
        require(entry.split=="train"&&entry.family==14,
                "geodesic preflight schedule must contain family14 TRAIN worlds only");
        require(entry.witness_route.size()>=2,"family14 training level has no witness segment");
        const auto& a=entry.witness_route[0];const auto& b=entry.witness_route[1];
        std::array<float,3> witness{{b[0]-a[0],b[1]-a[1],b[2]-a[2]}};
        const float witness_norm=std::sqrt(witness[0]*witness[0]+witness[1]*witness[1]+witness[2]*witness[2]);
        require(witness_norm>1.0e-5f,"witness first segment is empty");
        for(float& component:witness)component/=witness_norm;
        std::array<float,3> field_world{};float min_clearance=12.0f;
        if(label[3]>0.5f) {
            const float* r=label+7;
            for(uint32_t axis=0;axis<3;axis++)
                field_world[axis]=r[axis*3]*label[0]+r[axis*3+1]*label[1]+r[axis*3+2]*label[2];
            for(uint32_t sample=1;sample<=12;sample++) {
                const float d=.05f*float(sample);
                min_clearance=std::min(min_clearance,wclearance(entry.world,
                    wv(label[4]+field_world[0]*d,label[5]+field_world[1]*d,label[6]+field_world[2]*d),0.0f));
            }
        }
        const float cosine=label[3]>0.5f
            ? witness[0]*field_world[0]+witness[1]*field_world[1]+witness[2]*field_world[2] : -1.0f;
        measurements.push_back({level,label[3]>0.5f,cosine,min_clearance,witness,field_world});

        if(label[3]>0.5f) {
            const float* row_obs=obs+size_t(env)*fixed_ppo::actor_obs_dim;
            float gate_clear=0;for(uint32_t ray=0;ray<20;ray++)gate_clear+=row_obs[ray]*12.0f>2.5f;
            const float gate=.25f+.75f*gate_clear/20.0f;
            float max_error=0;
            for(uint32_t axis=0;axis<3;axis++) {
                const float prior=row_obs[181+axis];
                const float desired=(1.2f/1.5f)*label[axis];
                const float raw=prior+(std::atanh(desired)-prior)/gate;
                const float reconstructed=std::tanh(prior+gate*(raw-prior));
                max_error=std::max(max_error,std::fabs(reconstructed-desired)*1.5f);
            }
            inverse_error+=max_error;inverse_count++;
        }
    }
    require(!measurements.empty(),"geodesic preflight found no family14 train levels in initial schedule");

    const auto mean_direction_loss=[&](){
        fixed_ppo::ActorParams parameters;
        std::memcpy(parameters.values.data(),sim.actor.contents,sim.actor.length);
        double loss=0;uint32_t count=0;
        for(uint32_t env=0;env<envs;env++) {
            const float* label=target+size_t(env)*16;if(!(label[3]>0.5f))continue;
            const float* row_obs=obs+size_t(env)*fixed_ppo::actor_obs_dim;
            thread_local float hidden[fixed_ppo::hidden_dim],mean[fixed_ppo::action_dim];
            fixed_ppo::actor_forward(row_obs,parameters,hidden,mean);
            float overhead=0;for(uint32_t ray=0;ray<20;ray++)overhead+=row_obs[ray]*12.0f>2.5f;
            const float gate=.25f+.75f*overhead/20.0f;
            float q[3],norm2=0,dot=0;
            for(uint32_t axis=0;axis<3;axis++){
                const float prior=row_obs[181+axis];q[axis]=std::tanh(prior+gate*(mean[axis]-prior));
                norm2+=q[axis]*q[axis];dot+=q[axis]*label[axis];
            }
            if(norm2>1.0e-10f){loss+=1.0-double(dot/std::sqrt(norm2));count++;}
        }
        require(count>0,"geodesic preflight has no valid actor-loss rows");
        return loss/double(count);
    };
    const double loss_before=mean_direction_loss();
    std::memset(sim.rewards.contents,0,sim.rewards.length);
    std::memset(sim.values.contents,0,sim.values.length);
    std::memset(sim.next_values.contents,0,sim.next_values.length);
    PPOTrainer trainer(sim,1);trainer.set_direction_auxiliary(.1f);
    auto update=[metal.queue commandBuffer];trainer.rollout_update(update,0);metal.finish(update);
    const double loss_after=mean_direction_loss();
    float cosine_sum=0.0f;uint32_t cosine_count=0;float positive_y=0.0f;
    std::vector<float> cosines;
    for(const auto& row:measurements)if(row.target_valid){cosine_sum+=row.cosine;cosines.push_back(row.cosine);positive_y+=row.field[1]>0.0f;cosine_count++;}
    require(cosine_count>0,"geodesic field produced no valid initial direction targets");
    const float mean_cosine=cosine_sum/float(cosine_count);
    std::sort(cosines.begin(),cosines.end());
    const float median_cosine=cosines[cosines.size()/2];
    const float positive_y_fraction=positive_y/float(cosine_count);

    const std::filesystem::path output(output_path);
    if(output.has_parent_path())std::filesystem::create_directories(output.parent_path());
    std::ofstream report(output_path);require(bool(report),"cannot write geodesic direction preflight report");
    report<<std::setprecision(9)<<"{\n\"status\":\"preflight_complete\",\n\"bank_sha256\":\""
          <<challenge_evaluation::sha256_file(bank_path)<<"\",\n\"world_sha256\":\""
          <<sampler.world_hash()<<"\",\n\"field_sha256\":\""<<field_hash
          <<"\",\n\"warmstart_sha256\":\""<<challenge_evaluation::sha256_file(warmstart)
          <<"\",\n\"scope\":\"family14 mirrored TRAIN states only; no dev/final fields or labels\",\n"
          <<"\"valid_gpu_targets\":"<<target_check.valid<<",\n\"initial_levels_checked\":"<<measurements.size()
          <<",\n\"cpu_gpu_body_direction_max_error\":"<<target_check.max_body_error
          <<",\n\"target_alignment_to_first_witness_segment_mean_cosine\":"<<mean_cosine
          <<",\n\"target_alignment_to_first_witness_segment_median_cosine\":"<<median_cosine
          <<",\n\"valid_initial_targets\":"<<cosine_count
          <<",\n\"target_positive_y_fraction\":"<<positive_y_fraction
          <<",\n\"witness_inverse_reconstruction_max_mps\":"<<(inverse_count?inverse_error/inverse_count:0)
          <<",\n\"actor_direction_loss_before_one_aux_only_update\":"<<loss_before
          <<",\n\"actor_direction_loss_after_one_aux_only_update\":"<<loss_after
          <<",\n\"aux_only_optimizer_updates\":1,\n\"auxiliary_coefficient\":0.1,\n"
          <<"\"reconstruction_speed_mps\":1.2,\n\"levels\":[";
    for(size_t index=0;index<measurements.size();index++) {
        const auto& row=measurements[index];if(index)report<<',';
        report<<"{\"failure_id\":\""<<levels[row.id].failure_id<<"\",\"target_valid\":"
              <<(row.target_valid?"true":"false")<<",\"witness_clearance_m\":"<<levels[row.id].witness_clearance
              <<",\"first_segment_cosine\":"<<row.cosine<<",\"target_segment_clearance_m\":"<<row.min_clearance
              <<",\"witness_unit_world\":["<<row.witness[0]<<','<<row.witness[1]<<','<<row.witness[2]
              <<"],\"field_unit_world\":["<<row.field[0]<<','<<row.field[1]<<','<<row.field[2]<<"]}";
    }
    report<<"]\n}\n";report.flush();require(bool(report),"geodesic direction preflight report write failed");
    std::cout<<"geodesic_preflight bank="<<bank_path<<" train_levels="<<measurements.size()
             <<" valid_targets="<<target_check.valid<<" witness_cos_mean="<<mean_cosine
             <<" witness_cos_median="<<median_cosine<<" inverse_error_mps="<<(inverse_count?inverse_error/inverse_count:0)
             <<" actor_direction_loss="<<loss_before<<"->"<<loss_after<<" report="<<output_path<<"\n";
}

// Controlled bank training uses the same PPO implementation and rollout size.
// Only level selection changes between uniform and priority arms. The final
// split is never loaded by the sampler or checkpoint selection.
static void train_challenge_bank(Metal& metal, uint32_t iterations,
                                 const std::string& bank_path,
                                 const std::string& checkpoint,
                                 const std::string& warmstart,
                                 bool prioritized,uint32_t sampler_seed=42,uint32_t rehearsal_environments=0,float potential_scale=0,float risk_coef=.1f,uint32_t focus_family=0,float learning_rate=.0001f,float entropy=.0005f,float warm_logstd=-1,uint32_t epochs=2,float adv_clip=0,float value_coef=.5f,float anchor=0,float anchor_radius=0,float direction_aux_coefficient=0.0f) {
    require(iterations>0 && (fixed_ppo::actor_obs_dim==184||fixed_ppo::actor_obs_dim==824),"bank training needs a guided build and positive rollouts");
    require(epochs>=1 && epochs<=4,"bank epochs must be1..4");
    require(std::isfinite(adv_clip) && (adv_clip==0.0f || (adv_clip>=0.5f && adv_clip<=100.0f)),
            "bank advantage clip must be0 (off) or0.5..100 standard deviations");
    require(std::isfinite(value_coef) && value_coef>0 && value_coef<=4,"bank value coefficient must be in0..4");
    require(std::isfinite(anchor) && anchor>=0 && anchor<=1000,"bank anchor lambda must be in0..1000");
    require(std::isfinite(anchor_radius) && anchor_radius>=0 && anchor_radius<=100,"bank anchor radius must be in0..100");
    constexpr uint32_t environments=128,horizon=32;
    metal.compile(base_source()+PPO_TRAINER_MSL);
    const auto selection=prioritized?challenge_training::SelectionMode::FailureWeighted:
                                     challenge_training::SelectionMode::UniformBank;
    challenge_training::Settings settings;
    settings.environment_count=environments;settings.horizon=horizon;
    settings.sampler_seed=sampler_seed;settings.selection=selection;
    settings.rehearsal_environments=rehearsal_environments;settings.focus_family=focus_family;
    challenge_training::ChallengeTraining sampler(bank_path,
        challenge_evaluation::sha256_file(std::string(SOURCE_DIR)+"/world.hpp"),settings);
    SimConfig config;
    config.n=environments;config.seed=sampler_seed;config.family=14;config.mode=22;config.distance=8;config.speed=1.5f;
    require(std::isfinite(risk_coef)&&risk_coef>=0&&risk_coef<=1,"bank risk coefficient must be0..1");
    config.max_steps=400;config.geometry_memory=1;config.risk_coef=risk_coef;
    require(learning_rate>0 && learning_rate<=.001f && entropy>=0 && entropy<=.01f && warm_logstd>=-2 && warm_logstd<=.5f,"invalid focused training settings");
    config.entropy_coef=entropy;config.learning_rate=learning_rate;
    Sim simulator(metal,config,horizon);
    const auto& worlds=sampler.worlds();
    simulator.bank_worlds=metal.buffer(worlds.size()*sizeof(WWorld),worlds.data());
    const auto control=sampler.control();
    std::memcpy(simulator.bank_control.contents,&control,sizeof(control));
    require(std::isfinite(potential_scale)&&potential_scale>=0&&potential_scale<=16,"potential scale must be0..16");
    require(std::isfinite(direction_aux_coefficient)&&direction_aux_coefficient>=0.0f&&direction_aux_coefficient<=1.0f,
            "geodesic direction auxiliary coefficient must be in0..1");
    training_potential::TrainingPotentialFields direction_fields;
    std::string direction_field_hash(64,'0');
    if(potential_scale>0 || direction_aux_coefficient>0.0f) {
        const auto& levels=sampler.levels();
        require(levels.size()==worlds.size(),"potential fields must match the train sampler");
        for(size_t index=0;index<levels.size();index++)
            require(std::memcmp(&levels[index].world,&worlds[index],sizeof(WWorld))==0,"potential level order differs from sampler");
        direction_fields=training_potential::build_training_potential_fields(levels,
            challenge_evaluation::sha256_file(bank_path),challenge_evaluation::sha256_file(std::string(SOURCE_DIR)+"/world.hpp"),
            bank_path+".potential-v2.cache");
        simulator.potential_fields=metal.buffer(direction_fields.phi.size()*sizeof(float),direction_fields.phi.data());
        std::memcpy(simulator.potential_spec.contents,&direction_fields.spec,sizeof(direction_fields.spec));
        direction_field_hash=training_potential::sha256_hex(direction_fields.phi.data(),direction_fields.phi.size()*sizeof(float));
        std::memcpy(simulator.potential_hash.contents,direction_field_hash.data(),64);
        if(potential_scale>0.0f) {
            const TrainingPotentialControl shaping{1,potential_scale,.99f,training_potential::kVersion};
            std::memcpy(simulator.potential_control.contents,&shaping,sizeof(shaping));
            std::cout<<"training_only_potential scale="<<potential_scale<<" gamma=.99 field_sha256="
                     <<direction_field_hash<<" bytes="<<simulator.potential_fields.length<<"\n";
        }
        if(direction_aux_coefficient>0.0f) {
            simulator.enable_direction_auxiliary();
            std::cout<<"training_only_geodesic_direction coefficient="<<direction_aux_coefficient
                     <<" field_sha256="<<direction_field_hash<<" source=TRAIN_bank_phi=-distance/cap\n";
        }
    }
    PPOTrainer trainer(simulator,epochs);
    trainer.set_direction_auxiliary(direction_aux_coefficient);
    const PpoSafeguardConfig safeguard_config{epochs,adv_clip,value_coef,anchor,anchor_radius,
        direction_aux_coefficient,direction_field_hash};
    std::string anchor_reference_hash(64,'0');
    trainer.raw_advantages=metal.buffer(simulator.advantages.length);
    if(adv_clip>0)std::memcpy(trainer.adv_clip_b.contents,&adv_clip,sizeof(float));
    if(value_coef!=.5f)std::memcpy(trainer.value_coef_b.contents,&value_coef,sizeof(float));
    trainer.set_anchor(anchor,anchor_radius);
    if(adv_clip>0 || epochs!=2 || value_coef!=.5f || anchor>0 || direction_aux_coefficient>0.0f)
        std::cout<<"bank_update_guard advantage_clip_sd="<<(adv_clip>0?std::to_string(adv_clip):"off")
                 <<" epochs="<<epochs<<" value_coef="<<value_coef
                 <<" anchor_lambda="<<anchor<<" anchor_radius="<<anchor_radius
                 <<" geodesic_direction_aux="<<direction_aux_coefficient<<"\n";
    const std::string replay_path=checkpoint+".replay.state";
    const bool resume=std::filesystem::exists(checkpoint);
    require(resume==std::filesystem::exists(replay_path),"bank checkpoint and replay state must both exist or both be absent");
    std::vector<uint32_t> schedule;
    if(resume) {
        trainer.load_checkpoint(checkpoint,config.family,horizon,environments,config.seed);
        anchor_reference_hash=load_ppo_safeguard_state(checkpoint,safeguard_config,
            trainer.completed_rollouts,trainer.anchor_ref_b.contents);
        if(anchor==0.0f)trainer.recapture_anchor_reference();
        sampler.load_state(replay_path,challenge_evaluation::sha256_file(checkpoint),trainer.completed_rollouts);
        const auto& active=sampler.active_ids();
        std::memcpy(simulator.bank_active_ids.contents,active.data(),active.size()*sizeof(uint32_t));
    } else {
        std::ifstream source(warmstart,std::ios::binary);
        const auto header=read_checkpoint_header(source);
        require(header.actor_count==fixed_ppo::actor_param_count && header.critic_count==fixed_ppo::critic_param_count,"bank warmstart dimensions");
        source.read(static_cast<char*>(simulator.actor.contents),simulator.actor.length);
        source.read(static_cast<char*>(simulator.critic.contents),simulator.critic.length);
        require(bool(source),"bank warmstart read failed");
        for(uint32_t axis=0;axis<4;axis++)
            static_cast<float*>(simulator.actor.contents)[fixed_ppo::actor_log_std_offset+axis]=warm_logstd;
        // This buffer is also the diagnostic drift reference when anchoring is
        // disabled. Capture the actual warmstart, not the fresh random actor
        // copied when PPOTrainer was constructed.
        trainer.recapture_anchor_reference();
        if(anchor>0.0f) {
            anchor_reference_hash=ppo_safeguard_sha256(trainer.anchor_ref_b.contents,trainer.anchor_ref_b.length);
        }
        schedule=sampler.initial_schedule();
        std::memcpy(simulator.bank_schedule.contents,schedule.data(),schedule.size()*sizeof(uint32_t));
        simulator.reset();
    }
    if(resume&&anchor>0.0f)std::cout<<"anchor_reference validated and loaded "<<checkpoint<<".anchorref\n";
    // Episode counters accumulate for the whole process, so diagnostics take a
    // per-rollout delta. Read-only: no checkpoint, sampler, or optimizer state.
    std::vector<SimRun> previous_runs(environments);
    std::memcpy(previous_runs.data(),simulator.runs.contents,environments*sizeof(SimRun));
    const auto save=[&](const std::string& path,uint32_t rollout) {
        trainer.save_checkpoint(path,config.family,config.seed,rollout);
        save_ppo_safeguard_state(trainer,path,safeguard_config,rollout);
        const uint32_t* active=static_cast<const uint32_t*>(simulator.bank_active_ids.contents);
        sampler.save_state(path+".replay.state",challenge_evaluation::sha256_file(path),rollout,
                           std::vector<uint32_t>(active,active+environments));
    };
    const auto selection_score=[&](const challenge_evaluation::BankScore& score) {
        if(focus_family) {
            const size_t slot=focus_family-14;
            return score.family_episodes[slot]?double(score.family_successes[slot])/score.family_episodes[slot]:0.0;
        }
        return score.worst_family_success();
    };
    auto best=challenge_evaluation::BankScore{};
    if(std::filesystem::exists(checkpoint+".best")) {
        std::ifstream best_checkpoint(checkpoint+".best",std::ios::binary);
        const auto best_header=read_checkpoint_header(best_checkpoint);
        const std::string best_reference_hash=load_ppo_safeguard_state(checkpoint+".best",safeguard_config,
            best_header.completed_rollouts,nullptr);
        require(best_reference_hash==anchor_reference_hash,
                "best checkpoint PPO anchor reference differs from resumed run");
        best=run_navigation_bank_evaluation(metal,checkpoint+".best",bank_path,"dev",checkpoint+".best-dev.csv");
    } else {
        save(checkpoint+".best",trainer.completed_rollouts);
        best=run_navigation_bank_evaluation(metal,checkpoint+".best",bank_path,"dev",checkpoint+".best-dev.csv");
    }
    std::ofstream history(checkpoint+".history.csv",std::ios::app);
    require(bool(history),"cannot write bank history");
    if(!resume)history<<"rollout,transitions,wall_s,gpu_s,success,worst_family_success,collision,timeout\n";
    std::ofstream diag(checkpoint+".diag.csv",std::ios::app);
    require(bool(diag),"cannot write bank diagnostics");
    if(!resume)diag<<"rollout,transitions,gpu_s,rew_mean,rew_std,rew_min,rew_max,term_n,term_mean,term_min,term_max,term_pos_n,adv_mean,adv_std,adv_min,adv_max,adv_gt3_frac,ret_mean,ret_std,ret_max,val_mean,val_std,pol_loss,val_loss,ent_term,ratio,actor_scale,critic_scale,logstd0,logstd1,logstd2,logstd3,ep_success,ep_collision,ep_timeout,ep_path_m,ep_time_s,train_speed_mps,inflight_clear_min,actor_drift_l2\n";
    diag<<std::setprecision(9);
    const double started=seconds();
    const uint32_t start_rollout=trainer.completed_rollouts;
    const uint32_t finish=trainer.completed_rollouts+iterations;
    std::cout<<"bank_train selection="<<(prioritized?"priority":"uniform")
             <<" train_levels="<<worlds.size()<<" seed="<<sampler_seed<<" rehearsal_envs="<<rehearsal_environments<<" envs=128 horizon=32 start="<<trainer.completed_rollouts<<"\n";
    for(uint32_t rollout=trainer.completed_rollouts;rollout<finish;rollout++) {@autoreleasepool {
        if(resume || rollout>0) {
            const uint32_t* active=static_cast<const uint32_t*>(simulator.bank_active_ids.contents);
            schedule=sampler.next_schedule(std::vector<uint32_t>(active,active+environments),rollout);
            std::memcpy(simulator.bank_schedule.contents,schedule.data(),schedule.size()*sizeof(uint32_t));
        }
        auto commands=[metal.queue commandBuffer];
        simulator.collect(commands,horizon);
        double collect_gpu=0;
        if(potential_scale>0) {
            collect_gpu=metal.finish(commands);
            const auto* reward=(const float*)simulator.rewards.contents;
            for(uint32_t row=0;row<environments*horizon;row++)require(std::isfinite(reward[row]),"invalid shaping reward");
            commands=[metal.queue commandBuffer];
        }
        trainer.rollout_update(commands,rollout);
        const double gpu=collect_gpu+metal.finish(commands);
        sampler.observe_raw_gae(static_cast<const float*>(trainer.raw_advantages.contents),
            static_cast<const uint8_t*>(simulator.terminated.contents),
            static_cast<const uint8_t*>(simulator.truncated.contents),
            static_cast<const uint32_t*>(simulator.bank_transition_ids.contents),rollout);
        if(direction_aux_coefficient>0.0f && rollout==start_rollout) {
            const auto check=validate_geodesic_aux_targets(simulator,direction_fields);
            require(check.valid>0,"geodesic field produced no usable directions in the first rollout");
            const size_t target_rows=size_t(environments)*horizon;
            std::cout<<"geodesic_targets cpu_gpu_parity=PASS valid="<<check.valid<<"/"<<target_rows
                     <<" fraction="<<double(check.valid)/target_rows<<" max_body_error="<<check.max_body_error<<"\n";
        }
        trainer.completed_rollouts=rollout+1;
        {
            constexpr size_t rows=size_t(environments)*horizon;
            const float* rew=static_cast<const float*>(simulator.rewards.contents);
            const float* adv=static_cast<const float*>(trainer.raw_advantages.contents);
            const float* ret=static_cast<const float*>(simulator.returns.contents);
            const float* val=static_cast<const float*>(simulator.values.contents);
            const auto* term=static_cast<const uint8_t*>(simulator.terminated.contents);
            const auto moment=[&](const float* values,size_t count){
                double mean=0;float lo=std::numeric_limits<float>::max(),hi=-lo;
                for(size_t i=0;i<count;i++){const float v=values[i];require(std::isfinite(v),"diagnostic buffer is non-finite");mean+=v;lo=std::min(lo,v);hi=std::max(hi,v);}
                if(count==0){lo=0;hi=0;}
                mean/=double(count?count:1);
                double var=0;for(size_t i=0;i<count;i++){const double d=double(values[i])-mean;var+=d*d;}
                return std::array<double,4>{mean,count?std::sqrt(var/double(count)):0.0,lo,hi};
            };
            const auto rw=moment(rew,rows),av=moment(adv,rows),rt=moment(ret,rows),vl=moment(val,rows);
            size_t term_n=0,term_pos=0,adv_gt3=0;
            double term_sum=0,term_lo=std::numeric_limits<double>::max(),term_hi=-term_lo;
            for(size_t i=0;i<rows;i++) {
                if(term[i]) {
                    const double v=rew[i];term_n++;term_sum+=v;term_lo=std::min(term_lo,v);term_hi=std::max(term_hi,v);
                    if(v>0)term_pos++;
                }
                if(std::fabs(adv[i])>3.0f)adv_gt3++;
            }
            if(term_n==0){term_lo=0;term_hi=0;}
            const auto* runs_now=static_cast<const SimRun*>(simulator.runs.contents);
            uint64_t d_success=0,d_collision=0,d_timeout=0,d_episodes=0;
            double d_path=0,d_elapsed=0;
            float clear_min=12.0f;
            for(uint32_t env=0;env<environments;env++) {
                const SimRun& now=runs_now[env];const SimRun& before=previous_runs[env];
                require(now.successes>=before.successes && now.collisions>=before.collisions &&
                        now.timeouts>=before.timeouts && now.episodes>=before.episodes,"diagnostic episode counters went backwards");
                d_success+=now.successes-before.successes;d_collision+=now.collisions-before.collisions;
                d_timeout+=now.timeouts-before.timeouts;d_episodes+=now.episodes-before.episodes;
                d_path+=double(now.total_path)-before.total_path;d_elapsed+=double(now.total_elapsed)-before.total_elapsed;
                clear_min=std::min(clear_min,now.min_clearance);
            }
            std::memcpy(previous_runs.data(),simulator.runs.contents,environments*sizeof(SimRun));
            const float* met=static_cast<const float*>(trainer.metric_mean.contents);
            const float* actor_params=static_cast<const float*>(simulator.actor.contents);
            const double drift_l2=trainer.actor_drift_l2();
            diag<<(rollout+1)<<','<<uint64_t(rollout+1)*environments*horizon<<','<<gpu
                <<','<<rw[0]<<','<<rw[1]<<','<<rw[2]<<','<<rw[3]
                <<','<<term_n<<','<<(term_n?term_sum/double(term_n):0.0)<<','<<term_lo<<','<<term_hi<<','<<term_pos
                <<','<<av[0]<<','<<av[1]<<','<<av[2]<<','<<av[3]<<','<<double(adv_gt3)/rows
                <<','<<rt[0]<<','<<rt[1]<<','<<rt[3]
                <<','<<vl[0]<<','<<vl[1]
                <<','<<met[0]<<','<<met[1]<<','<<met[2]<<','<<met[3]
                <<','<<*static_cast<const float*>(trainer.actor_scale.contents)<<','<<*static_cast<const float*>(trainer.critic_scale.contents);
            for(uint32_t axis=0;axis<fixed_ppo::action_dim;axis++)diag<<','<<actor_params[fixed_ppo::actor_log_std_offset+axis];
            diag<<','<<double(d_success)/double(d_episodes?d_episodes:1)
                <<','<<double(d_collision)/double(d_episodes?d_episodes:1)
                <<','<<double(d_timeout)/double(d_episodes?d_episodes:1)
                <<','<<(d_episodes?d_path/double(d_episodes):0.0)
                <<','<<(d_episodes?d_elapsed/double(d_episodes):0.0)
                <<','<<(d_elapsed?d_path/d_elapsed:0.0)
                <<','<<clear_min<<','<<drift_l2<<'\n';
            diag.flush();require(bool(diag),"bank diagnostics write failed");
            if((rollout+1)%50==0 || rollout+1==finish)
                std::cout<<"bank_diag rollout="<<rollout+1<<" rew="<<rw[0]<<" adv_std="<<av[1]
                         <<" term_n="<<term_n<<" term_mean="<<(term_n?term_sum/double(term_n):0.0)
                         <<" ratio="<<met[3]<<" ent_term="<<met[2]<<" pol="<<met[0]<<" val="<<met[1]
                         <<" ep_success="<<(d_episodes?double(d_success)/double(d_episodes):0.0)
                         <<" ep_collision="<<(d_episodes?double(d_collision)/double(d_episodes):0.0)
                         <<" clear_min="<<clear_min<<" drift_l2="<<drift_l2
                         <<" anchor_pull="<<trainer.current_anchor_pull()<<"\n";
        }
        if((rollout+1)%50==0 || rollout+1==finish) {
            save(checkpoint,rollout+1);
            const auto score=run_navigation_bank_evaluation(metal,checkpoint,bank_path,"dev",checkpoint+".dev.csv");
            history<<rollout+1<<','<<uint64_t(rollout+1)*environments*horizon<<','<<seconds()-started
                   <<','<<gpu<<','<<score.success_rate()<<','<<score.worst_family_success()
                   <<','<<double(score.collisions)/score.episodes<<','<<double(score.timeouts)/score.episodes<<'\n';
            history.flush();require(bool(history),"bank history write failed");
            if(selection_score(score)>selection_score(best) ||
               (selection_score(score)==selection_score(best) && score.success_rate()>best.success_rate())) {
                best=score;save(checkpoint+".best",rollout+1);
                std::filesystem::copy_file(checkpoint+".dev.csv",checkpoint+".best-dev.csv",
                                           std::filesystem::copy_options::overwrite_existing);
            }
            std::cout<<"bank_train rollout="<<rollout+1<<" wall_s="<<seconds()-started
                     <<" success="<<score.success_rate()<<" worst_family="<<score.worst_family_success()<<" focus_success="<<selection_score(score)<<"\n";
            if(direction_aux_coefficient>0.0f) {
                const float* targets=static_cast<const float*>(simulator.direction_aux_targets.contents);
                uint32_t valid=0;for(size_t row=0;row<size_t(environments)*horizon;row++)valid+=targets[row*16+3]>0.5f;
                std::cout<<"geodesic_aux coefficient="<<direction_aux_coefficient
                         <<" current_rollout_targets="<<valid<<"/"<<size_t(environments)*horizon<<"\n";
            }
        }
    }}
}

static void trace_checkpoint(Metal& metal, const std::string& checkpoint, uint32_t family,
                             uint32_t mode, uint32_t seed,
                             const std::string& output_prefix="results/trace",
                             int32_t selected_environment=-1, bool clean_conditions=false) {
    std::ifstream file(checkpoint,std::ios::binary);
    const auto header=read_checkpoint_header(file);
    require(header.actor_count==fixed_ppo::actor_param_count,"trace actor dimensions");
    std::vector<float> actor(header.actor_count);
    file.read(reinterpret_cast<char*>(actor.data()),actor.size()*sizeof(float));
    require(bool(file),"trace actor read");
    require(std::all_of(actor.begin(),actor.end(),[](float value){return std::isfinite(value);}),"trace actor contains nonfinite parameters");

    SimConfig config=header.config;
    config.family=family;config.mode=mode;config.eval=1;config.seed=seed;
    if(family>=14 && family<=16)config.max_steps=400;
    if(selected_environment>=0){config.n=1;config.seed=seed+uint32_t(selected_environment)*747796405u;}
    else config.n=std::min(config.n,128u);
    config.geometry_memory=header.config.geometry_memory || mode>=13;
    if(clean_conditions){config.sensor_delay=0;config.command_delay=0;config.wind=0;config.depth_noise=0;config.dropout=0;}
    Sim simulator(metal,config,config.max_steps);
    std::memcpy(simulator.actor.contents,actor.data(),actor.size()*sizeof(float));
    auto commands=[metal.queue commandBuffer];simulator.collect(commands);metal.finish(commands);

    const auto* episodes=static_cast<const SimRun*>(simulator.runs.contents);
    uint32_t environment=0;
    if(selected_environment<0)while(environment+1<config.n && !episodes[environment].collisions)environment++;
    const uint32_t logical_environment=selected_environment>=0?uint32_t(selected_environment):environment;
    const auto& episode=episodes[environment];
    const auto& world=static_cast<const WWorld*>(simulator.worlds.contents)[environment];
    const auto& final_state=static_cast<const RLPhysicsState*>(simulator.states.contents)[environment];
    const auto& physics=*static_cast<const RLPhysicsParams*>(simulator.physics.contents);
    const float navigation_dt=physics.dt*config.substeps;
    const float* privileged=static_cast<const float*>(simulator.co.contents);
    const float* observations=static_cast<const float*>(simulator.obs.contents);
    const uint32_t depth_cells=fixed_ppo::actor_obs_dim==661?320u:80u;
    const uint32_t context=depth_cells*2;

    const std::filesystem::path output_path(output_prefix);
    if(output_path.has_parent_path())std::filesystem::create_directories(output_path.parent_path());
    std::ofstream metadata(output_prefix+".json");require(bool(metadata),"cannot write episode metadata");
    auto array=[&](const float* values,size_t count){metadata<<'[';for(size_t i=0;i<count;i++){if(i)metadata<<',';metadata<<values[i];}metadata<<']';};
    metadata<<std::setprecision(9)<<"{\n\"schema_version\":1,\"checkpoint\":"<<std::quoted(checkpoint)
            <<",\"family\":"<<family<<",\"episode_family\":"<<world.family<<",\"mode\":"<<mode
            <<",\"seed\":"<<seed<<",\"environment_index\":"<<logical_environment<<",\"world_seed\":"<<world.seed
            <<",\"frame\":\"world XYZ, Z up; body FLU; metres and seconds; quaternion wxyz; body-to-world rotation\""
            <<",\"geometry_use\":\"ground-truth visualization and scoring only; not actor inputs\""
            <<",\"dt_s\":"<<navigation_dt<<",\"native_dt_s\":"<<physics.dt
            <<",\"sensor_interval_s\":"<<navigation_dt*config.sensor_period
            <<",\"sensor_delay_frames\":"<<config.sensor_delay<<",\"command_delay_steps\":"<<config.command_delay
            <<",\"wind_accel_x\":"<<config.wind<<",\"depth_noise_m\":"<<config.depth_noise<<",\"pixel_dropout\":"<<config.dropout
            <<",\"depth_rows\":"<<(depth_cells==320?16:8)<<",\"depth_columns\":"<<(depth_cells==320?20:10)
            <<",\"depth_encoding\":\"row-major ray-range metres; 2x2 minimum pooling for8x10; misses12m\""
            <<",\"depth_pose_note\":\"capture pose is trace pose at depth_capture_t_s; stale startup frames marked sensor_ready0\""
            <<",\"room_bounds\":[[-2,14],[-5,5],[0,5]],\"drone_radius_m\":0.18,\"goal\":";
    array(world.goal,3);
    metadata<<",\"success\":"<<episode.successes<<",\"collision\":"<<episode.collisions
            <<",\"timeout\":"<<episode.timeouts<<",\"duration_s\":"<<episode.elapsed
            <<",\"path_m\":"<<episode.path<<",\"native_min_clearance_m\":"<<episode.min_clearance<<",\"obstacles\":[";
    for(uint32_t i=0;i<world.count;i++){
        if(i)metadata<<',';const auto& obstacle=world.obstacles[i];
        metadata<<"{\"kind\":"<<obstacle.kind<<",\"center_t0\":";array(obstacle.center,3);
        metadata<<",\"size\":";array(obstacle.size,3);metadata<<",\"velocity\":";array(obstacle.velocity,3);metadata<<'}';
    }
    metadata<<"],\"obstacle_encoding\":\"kind0 AABB half extents XYZ; kind1 sphere radius size0; kind2 vertical cylinder radius size0, half-height size2; center(t)=center_t0+velocity*t\""
            <<",\"csv_note\":\"linear velocity,reference andcommands are world XYZ; angular velocity is body XYZ; clearance is instantaneous body SDF,not cumulative minimum; terminal_state is last100Hz state\",\"terminal_state\":{\"position\":";
    array(final_state.position,3);metadata<<",\"quaternion_wxyz\":";array(final_state.orientation_wxyz,4);
    metadata<<",\"linear_velocity\":";array(final_state.linear_velocity,3);metadata<<",\"angular_velocity_body\":";array(final_state.angular_velocity_body,3);
    metadata<<",\"reference\":";array(episode.reference_position,3);metadata<<"}}\n";
    require(bool(metadata),"episode metadata write failed");

    std::ofstream trace(output_prefix+".csv");require(bool(trace),"cannot write episode trace");
    trace<<"t_s,x,y,z,qw,qx,qy,qz,vx,vy,vz,wx,wy,wz,ref_x,ref_y,ref_z,cmd_vx,cmd_vy,cmd_vz,cmd_yaw_rate,clearance_m,sensor_ready,depth_capture_t_s";
    for(uint32_t cell=0;cell<depth_cells;cell++)trace<<",depth_"<<cell;
    if(fixed_ppo::actor_obs_dim==824) {
        for(uint32_t ray=0;ray<320;ray++)trace<<",raw_current_"<<ray;
        for(uint32_t ray=0;ray<320;ray++)trace<<",raw_previous_"<<ray;
    }
    trace<<'\n'<<std::setprecision(9);
    for(uint32_t tick=0;tick<episode.steps;tick++){
        const size_t row=size_t(tick)*config.n+environment;
        const float* state=privileged+row*32;
        const float* observation=observations+row*fixed_ppo::actor_obs_dim;
        float applied[4];
        for(uint32_t axis=0;axis<4;axis++)applied[axis]=tick+1<episode.steps?observations[(row+config.n)*fixed_ppo::actor_obs_dim+context+13+axis]:episode.previous_nav[axis];
        float velocity[3]={applied[0]*config.speed,applied[1]*config.speed,applied[2]*config.speed*(config.velocity_contract==0?.5f:1.0f)};
        const float magnitude=std::sqrt(velocity[0]*velocity[0]+velocity[1]*velocity[1]+velocity[2]*velocity[2]);
        const float scale=config.mode==19?(magnitude>1e-8f?config.speed/magnitude:1.0f):(config.velocity_contract==1?std::min(1.0f,config.speed/std::max(magnitude,1e-8f)):1.0f);
        for(float& component:velocity)component*=scale;
        float rotation[9];raptor_quaternion_matrix(state+9,rotation);
        trace<<tick*navigation_dt;
        for(uint32_t axis=0;axis<3;axis++)trace<<','<<state[13+axis]*10;
        for(uint32_t axis=0;axis<4;axis++)trace<<','<<state[9+axis];
        for(uint32_t axis=0;axis<3;axis++)trace<<','<<state[3+axis]*4;
        for(uint32_t axis=0;axis<3;axis++)trace<<','<<state[6+axis]*4;
        for(uint32_t axis=0;axis<3;axis++)trace<<','<<state[13+axis]*10+state[16+axis]*.5f;
        for(uint32_t axis=0;axis<3;axis++)trace<<','<<rotation[axis*3]*velocity[0]+rotation[axis*3+1]*velocity[1]+rotation[axis*3+2]*velocity[2];
        trace<<','<<applied[3]*.5f<<','<<state[19]*5<<','<<(tick/config.sensor_period>=config.sensor_delay)<<','<<tick*navigation_dt-observation[context+17];
        for(uint32_t cell=0;cell<depth_cells;cell++)trace<<','<<observation[cell]*12;
        if(fixed_ppo::actor_obs_dim==824) {
            for(uint32_t ray=0;ray<320;ray++)trace<<','<<observation[181+ray]*12;
            for(uint32_t ray=0;ray<320;ray++)trace<<','<<observation[501+ray]*12;
        }
        trace<<'\n';
    }
    require(bool(trace),"episode trace write failed");
    std::cout<<"episode trace="<<output_prefix<<" environment="<<logical_environment<<" success="<<episode.successes<<" collision="<<episode.collisions<<" timeout="<<episode.timeouts<<" final="<<final_state.position[0]<<','<<final_state.position[1]<<','<<final_state.position[2]<<"\n";
}

static void gpu_training_benchmark(Metal& m,uint32_t n,uint32_t rollouts,uint32_t family=1) {
    m.compile(base_source()+PPO_TRAINER_MSL);SimConfig cfg;cfg.n=n;cfg.family=family;cfg.speed=1;cfg.distance=3;Sim sim(m,cfg,32);PPOTrainer trainer(sim);
    double gpu=0,start=seconds();for(uint32_t r=0;r<rollouts;r++){@autoreleasepool{auto cb=[m.queue commandBuffer];sim.collect(cb);trainer.rollout_update(cb,r);gpu+=m.finish(cb);}}
    double wall=seconds()-start;const auto* metrics=(const float*)trainer.metric_mean.contents;
    std::cout<<"GPU training family="<<family<<" n="<<n<<" rollouts="<<rollouts<<" transitions="<<uint64_t(n)*32*rollouts<<" gpu_s="<<gpu<<" wall_s="<<wall<<" optimizer_step="<<trainer.optimizer_step<<" policy="<<metrics[0]<<" value="<<metrics[1]<<" entropy="<<metrics[2]<<" ratio="<<metrics[3]<<"\n";
}

static void evaluate_threat(Metal& m,const std::string& checkpoint,uint32_t mode,uint32_t kind,float threat_speed,float nominal_ttc,uint32_t seed,uint32_t sensor_delay,uint32_t command_delay) {
    require(kind<=1 && sensor_delay<=6 && command_delay<=7,"invalid threat kind or delay");
    std::ifstream f(checkpoint,std::ios::binary);auto h=read_checkpoint_header(f);
    require(h.actor_count==fixed_ppo::actor_param_count,"threat actor dimensions");
    std::vector<float> actor(h.actor_count);f.read((char*)actor.data(),actor.size()*4);require(bool(f),"threat actor read");
    SimConfig cfg;cfg.mode=mode;cfg.family=0;cfg.eval=1;cfg.seed=seed;cfg.speed=1.5f;cfg.distance=4;
    cfg.sensor_delay=sensor_delay;cfg.command_delay=command_delay;cfg.velocity_contract=h.config.velocity_contract;cfg.geometry_memory=h.config.geometry_memory||mode>=13;
    Sim sim(m,cfg,32);std::memcpy(sim.actor.contents,actor.data(),actor.size()*4);
    auto* worlds=(WWorld*)sim.worlds.contents;auto* states=(RLPhysicsState*)sim.states.contents;auto* runs=(SimRun*)sim.runs.contents;
    for(uint32_t n=0;n<cfg.n;n++) {
        uint32_t obstacle=0;require(add_threat_sphere(worlds[n],seed+n*747796405u,ThreatKind(kind),cfg.speed,threat_speed,nominal_ttc,obstacle),"invalid or overlapping threat scene");
        states[n].linear_velocity[0]=cfg.speed;runs[n].desired_velocity[0]=cfg.speed;runs[n].previous_nav[0]=1;
    }
    std::cout<<"threat kind="<<kind<<" speed="<<threat_speed<<" nominal_ttc="<<nominal_ttc<<" initial_vx="<<cfg.speed<<" seed="<<seed<<" sensor_delay="<<sensor_delay<<" command_delay="<<command_delay<<"\n";
    const double start=seconds();auto cb=[m.queue commandBuffer];sim.collect(cb,cfg.max_steps);double gpu=m.finish(cb);
    sim.report("eval family=0 mode="+std::to_string(mode),seconds()-start);std::cout<<"eval_GPU_s="<<gpu<<"\n";
}

static void measure_reaction_latency(Metal& m,const std::string& checkpoint,uint32_t sensor_delay,uint32_t command_delay) {
    require(fixed_ppo::actor_obs_dim==184 && sensor_delay<=6 && command_delay<=7,"latency requires guided actor and valid delays");
    std::ifstream file(checkpoint,std::ios::binary);auto h=read_checkpoint_header(file);
    require(h.actor_count==fixed_ppo::actor_param_count,"latency actor dimensions");
    std::vector<float> actor(h.actor_count);file.read((char*)actor.data(),actor.size()*4);require(bool(file),"latency actor read");
    SimConfig cfg;cfg.n=1;cfg.mode=17;cfg.family=0;cfg.eval=1;cfg.seed=800001;cfg.speed=1.5f;cfg.distance=10;
    cfg.sensor_delay=sensor_delay;cfg.command_delay=command_delay;cfg.velocity_contract=h.config.velocity_contract;cfg.geometry_memory=1;
    Sim control(m,cfg,32),event(m,cfg,32);std::memcpy(control.actor.contents,actor.data(),actor.size()*4);
    auto warm=[m.queue commandBuffer];control.collect(warm,20);m.finish(warm);
    for(auto pair:{std::make_pair(control.states,event.states),std::make_pair(control.runs,event.runs),std::make_pair(control.worlds,event.worlds),std::make_pair(control.sensors,event.sensors),std::make_pair(control.poses,event.poses),std::make_pair(control.commands,event.commands),std::make_pair(control.actor,event.actor),std::make_pair(control.critic,event.critic)})std::memcpy(pair.second.contents,pair.first.contents,pair.first.length);
    auto paired_step=[&](){auto cb=[m.queue commandBuffer];control.collect(cb,1);event.collect(cb,1);m.finish(cb);};
    paired_step();
    auto* baseline=(SimRun*)control.runs.contents;auto* changed=(SimRun*)event.runs.contents;
    for(uint j=0;j<4;j++)require(std::fabs(baseline[0].motors[j]-changed[0].motors[j])<1e-6f,"latency paired baseline diverged before event");
    const auto state=((RLPhysicsState*)event.states.contents)[0];float rotation[9];raptor_quaternion_matrix(state.orientation_wxyz,rotation);
    const WVec forward=wv(rotation[0],rotation[3],rotation[6]);const WVec at_event=wa(wv(state.position[0],state.position[1],state.position[2]),wm(forward,2.0f));
    const WVec velocity=wm(forward,-1.0f);const float event_time=changed[0].elapsed;
    auto& world=((WWorld*)event.worlds.contents)[0];wadd(world,1,ws(at_event,wm(velocity,event_time)),wv(.35f,.35f,.35f),velocity);
    int command_tick=0,motor_tick=0;constexpr float delta_threshold=1e-4f;
    for(int tick=1;tick<=20;tick++) {
        paired_step();float command_delta=0,motor_delta=0;
        for(uint j=0;j<4;j++){command_delta=std::max(command_delta,std::fabs(baseline[0].previous_nav[j]-changed[0].previous_nav[j]));motor_delta=std::max(motor_delta,std::fabs(baseline[0].motors[j]-changed[0].motors[j]));}
        if(command_tick==0 && command_delta>delta_threshold)command_tick=tick;
        if(motor_tick==0 && motor_delta>delta_threshold)motor_tick=tick;
        if(command_tick && motor_tick)break;
        require(changed[0].episodes==0 && baseline[0].episodes==0,"episode ended before measurable reaction");
    }
    std::cout<<"reaction sensor_frames="<<sensor_delay<<" command_steps="<<command_delay<<" applied_command_upper_ms="<<command_tick*50<<" motor_upper_ms="<<motor_tick*50<<" resolution_ms=50 delta_threshold="<<delta_threshold<<" detected="<<bool(command_tick&&motor_tick)<<"\n";
}

int main(int argc,char** argv){@autoreleasepool{try{
    // Explicit camera-profile opt-in. Default is the legacy source model; the
    // flag is stripped so it can appear anywhere on the command line.
    std::vector<char*> args(argv,argv+argc);
    for(size_t i=1;i<args.size();) {
        if(std::string(args[i])=="--sensor-profile"&&i+1<args.size()) {
            uint32_t id=0;require(nav_sensor::parse(args[i+1],id),"sensor profile must be legacy or native");
            require(id==NAV_SENSOR_ACTIVE_ID,
                    std::string("this binary was built for the ")+NAV_SENSOR_ACTIVE_NAME+
                    " sensor profile; rebuild with the matching profile instead of mixing contracts");
            args.erase(args.begin()+i,args.begin()+i+2);
        } else ++i;
    }
    argc=int(args.size());argv=args.data();
    std::cout<<"sensor_profile="<<NAV_SENSOR_ACTIVE_NAME
             <<" tan_h="<<NAV_SENSOR_TAN_H<<" tan_v="<<NAV_SENSOR_ACTIVE_TAN_V
             <<" mount_x_m="<<NAV_SENSOR_ACTIVE_MOUNT_X<<"\n";
    std::string command=argc>1?argv[1]:"test";if(command=="export"){require(argc>=3,"export CHECKPOINT [OUTPUT]");require(fixed_ppo::actor_obs_dim==184,"export requires the guided binary");export_navigation_checkpoint(argv[2],argc>3?argv[3]:"assets/navigation.bin");return 0;}if(command=="export-raw"){require(argc>=4,"export-raw CHECKPOINT OUTPUT_RAW_NAV");export_raw_depth_checkpoint(argv[2],argv[3]);return 0;}if(command=="export-calibrated"){require(argc>=4,"export-calibrated CHECKPOINT_824 OUTPUT_CAL_NAV");export_calibrated_checkpoint(argv[2],argv[3]);return 0;}if(command=="lift-guided"){require(argc>=5,"lift-guided CHECKPOINT_184 OUTPUT_CHECKPOINT_824 OUTPUT_RAW_NAV");Metal metal;metal.compile(base_source()+PPO_TRAINER_MSL);lift_guided_checkpoint_to_raw_depth(metal,argv[2],argv[3],argv[4]);return 0;}if(command=="policy-bench"){benchmark_navigation_policy(argc>2?argv[2]:"assets/navigation.bin");return 0;}if(command=="cpu-bench"){cpu_reference_benchmark(argc>2?std::stoul(argv[2]):3,argc>4?std::stoul(argv[4]):1,argc>3?std::stoul(argv[3]):128);return 0;}Metal m(command=="profile");m.compile(base_source());std::cout<<"device="<<m.device.name.UTF8String<<" FP32 safe/precise\n";
    if(command=="test"){world_tests(m);raptor_tests(m);physics_tests(m);raptor_px4_adapter_tests();require(fixed_ppo::run_cpu_self_tests(),"PPO CPU self tests");ppo_tests(m);geodesic_direction_gradient_test(m);closed_loop_tests(m);mixed_domain_tests(m);deployed_action_map_test(m);}else if(command=="raw-depth-canary-check"){require(argc>=3,"raw-depth-canary-check NATIVE_SHADOW_CSV");raw_depth_canary_check(m,argv[2]);}else if(command=="geodesic-preflight"){require(argc>=5,"geodesic-preflight WARMSTART BANK_JSONL OUTPUT_JSON");geodesic_direction_preflight(m,argv[2],argv[3],argv[4]);}else if(command=="reaction-latency"){require(argc>=3,"reaction-latency CHECKPOINT [SENSOR_DELAY] [COMMAND_DELAY]");measure_reaction_latency(m,argv[2],argc>3?std::stoul(argv[3]):0,argc>4?std::stoul(argv[4]):0);}else if(command=="threat-eval"){require(argc>=7,"threat-eval CHECKPOINT MODE KIND SPEED TTC [SEED] [SENSOR_DELAY] [COMMAND_DELAY]");evaluate_threat(m,argv[2],std::stoul(argv[3]),std::stoul(argv[4]),std::stof(argv[5]),std::stof(argv[6]),argc>7?std::stoul(argv[7]):800001,argc>8?std::stoul(argv[8]):0,argc>9?std::stoul(argv[9]):0);}else if(command=="bench-depth")depth_benchmark(m);else if(command=="sensor-profile-check"){require(argc>=3,"sensor-profile-check RAY_CSV");sensor_profile_check(m,argv[2]);}else if(command=="eval-policy")evaluate_navigation_policy(m,argc>2?argv[2]:"assets/navigation.bin",argc>3?std::stoul(argv[3]):8,argc>4?std::stoul(argv[4]):800001);else if(command=="compare-webots") {
        require(argc>=5,"compare-webots NAV_ACTOR BOUNDED_METADATA_DIR OUTPUT_CSV");
        compare_webots_scenes(m,argv[2],argv[3],argv[4]);
    }else if(command=="reward-audit") {
        require(argc>=5,"reward-audit CHECKPOINT BANK_JSONL OUTPUT_CSV [LEVELS_PER_FAMILY=6] [MAX_STEPS=400]");
        reward_audit::run(m,argv[2],argv[3],argv[4],argc>5?std::stoul(argv[5]):6,argc>6?std::stoul(argv[6]):400);
    }else if(command=="contact-audit") {
        require(argc>=4,"contact-audit CHECKPOINT OUTPUT_CSV");
        audit_navigation_checkpoint_contacts(m,argv[2],argv[3]);
    }else if(command=="task-eval") {
        require(argc>=7,"task-eval CHECKPOINT OUTPUT_CSV STAGE FAMILY DOMAIN_AMPLITUDE [SEED] [MODE]");
        SimConfig config;Sim policy(m,config,32);navigation_training::load_actor(policy,argv[2]);
        navigation_training::evaluate(m,(const float*)policy.actor.contents,std::stoul(argv[4]),std::stoul(argv[5]),std::stof(argv[6]),
                                     argc>7?std::stoul(argv[7]):800001,argv[3],argc>8?std::stoul(argv[8]):17);
    }else if(command=="train-tasks") {
        require(argc>=8,"train-tasks ROLLOUTS CHECKPOINT WARMSTART STAGE FAMILY DOMAIN_AMPLITUDE [SEED]");
        navigation_training::train(m,std::stoul(argv[2]),argv[3],argv[4],std::stoul(argv[5]),std::stoul(argv[6]),std::stof(argv[7]),
                                   argc>8?std::stoul(argv[8]):42);
    }else if(command=="bank-witness") {
        require(argc>=5,"bank-witness BANK_JSONL train|dev OUTPUT_CSV [SPEED=1] [MAX_STEPS=1200] [LOCAL_POLICY_CHECKPOINT|goal-script]");
        challenge_evaluation::run_witness(m,argv[2],argv[3],argv[4],argc>5?std::stof(argv[5]):1.0f,argc>6?std::stoul(argv[6]):1200,argc>7?argv[7]:"");
    }else if(command=="train-bank") {
        require(argc>=7,"train-bank ROLLOUTS BANK_JSONL CHECKPOINT WARMSTART uniform|priority [--seed N] [--rehearsal N] [--potential-scale LAMBDA] [--direction-aux COEFFICIENT] [--risk COEFFICIENT] [--focus-family 14|15|16] [--epochs 1..4] [--adv-clip SD] [--value-coef V] [--anchor LAMBDA] [--anchor-radius DISTANCE]");
        const std::string selection=argv[6];
        require(selection=="uniform" || selection=="priority","bank selection must be uniform or priority");
        uint32_t seed=42,rehearsal=0,focus_family=0,epochs=2;float potential_scale=0,direction_aux=0,risk_coef=.1f,learning_rate=.0001f,entropy=.0005f,warm_logstd=-1,adv_clip=0,value_coef=.5f,anchor=0,anchor_radius=0;
        for(int argument=7;argument<argc;argument+=2) {
            require(argument+1<argc,"bank option requires a value");
            const std::string option=argv[argument];
            if(option=="--seed")seed=std::stoul(argv[argument+1]);
            else if(option=="--rehearsal")rehearsal=std::stoul(argv[argument+1]);
            else if(option=="--potential-scale")potential_scale=std::stof(argv[argument+1]);
            else if(option=="--direction-aux")direction_aux=std::stof(argv[argument+1]);
            else if(option=="--risk")risk_coef=std::stof(argv[argument+1]);
            else if(option=="--focus-family")focus_family=std::stoul(argv[argument+1]);
            else if(option=="--learning-rate")learning_rate=std::stof(argv[argument+1]);
            else if(option=="--entropy")entropy=std::stof(argv[argument+1]);
            else if(option=="--warm-logstd")warm_logstd=std::stof(argv[argument+1]);
            else if(option=="--epochs")epochs=std::stoul(argv[argument+1]);
            else if(option=="--adv-clip")adv_clip=std::stof(argv[argument+1]);
            else if(option=="--value-coef")value_coef=std::stof(argv[argument+1]);
            else if(option=="--anchor")anchor=std::stof(argv[argument+1]);
            else if(option=="--anchor-radius")anchor_radius=std::stof(argv[argument+1]);
            else throw std::runtime_error("unknown bank option "+option);
        }
        train_challenge_bank(m,std::stoul(argv[2]),argv[3],argv[4],argv[5],selection=="priority",seed,rehearsal,potential_scale,risk_coef,focus_family,learning_rate,entropy,warm_logstd,epochs,adv_clip,value_coef,anchor,anchor_radius,direction_aux);
    }else if(command=="bank-eval")challenge_bank_cli(m,argc,argv);else if(command=="eval")evaluate_checkpoint(m,argc>2?argv[2]:"results/open.bin",argc>3?std::stoul(argv[3]):4,argc>4?std::stoul(argv[4]):0,argc>5?std::stoul(argv[5]):800001,argc>6?std::stof(argv[6]):1,argc>7?std::stof(argv[7]):3,argc>8?std::stoul(argv[8]):0,argc>9?std::stof(argv[9]):0,argc>10?std::stof(argv[10]):0,argc>11?std::stof(argv[11]):0,argc>12?std::stoul(argv[12]):0,argc>13?std::stoul(argv[13]):0);else if(command=="trace")trace_checkpoint(m,argv[2],argc>3?std::stoul(argv[3]):5,argc>4?std::stoul(argv[4]):10,argc>5?std::stoul(argv[5]):800001,argc>6?argv[6]:"results/trace",argc>7?std::stoi(argv[7]):-1,argc>8?bool(std::stoi(argv[8])):false);else if(command=="gpu-bench")gpu_training_benchmark(m,argc>2?std::stoul(argv[2]):128,argc>3?std::stoul(argv[3]):3,argc>4?std::stoul(argv[4]):1);else if(command=="profile")train_navigation(m,1,0,"results/profile.bin");else if(command=="train")train_navigation(m,argc>2?std::stoul(argv[2]):100,argc>3?std::stoul(argv[3]):0,argc>4?argv[4]:"results/checkpoint.bin",argc>5?argv[5]:"",argc>6?std::stof(argv[6]):1,argc>7?std::stof(argv[7]):3,argc>8?std::stof(argv[8]):0,argc>9?std::stof(argv[9]):-1,argc>10?std::stof(argv[10]):.0003f,argc>11?std::stoul(argv[11]):1,argc>12?std::stoul(argv[12]):0,argc>13?std::stoul(argv[13]):0,argc>14?std::stof(argv[14]):0,argc>15?std::stof(argv[15]):0,argc>16?std::stof(argv[16]):0,argc>17?std::stoul(argv[17]):0,argc>18?std::stoul(argv[18]):4,argc>19?std::stof(argv[19]):-1);else if(command=="bench-loop")loop_benchmark(m);else if(command=="bench-raptor")raptor_benchmark(m);else if(command=="sim"){for(uint f=0;f<4;f++)sim_evaluate(m,f,2);}else throw std::runtime_error("Unknown command "+command);
    return 0;
}catch(const std::exception& e){std::cerr<<"ERROR: "<<e.what()<<"\n";return 1;}}}
