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
#include "raptor.hpp"
#include "physics.hpp"
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
    RLPhysicsState s=states[n],next;float a[4],wind[3];for(uint j=0;j<4;j++)a[j]=actions[n*4+j];for(uint j=0;j<3;j++)wind[j]=winds[n*3+j];rl_physics_step(s,a,wind,params,next);out[n]=next;
}
)MSL";
static std::string base_source() {
    std::string root=SOURCE_DIR;
    const std::string actor_obs_define="#define FIXED_PPO_ACTOR_OBS_DIM "+std::to_string(fixed_ppo::actor_obs_dim)+"\n";
    return "#include <metal_stdlib>\nusing namespace metal;\n"+read_text(root+"/world.hpp")+world_kernels+read_text(root+"/raptor.metal")+read_text(root+"/physics.metal")+actor_obs_define+read_text(root+"/ppo.metal")+read_text(root+"/guidance.hpp")+read_text(root+"/sim.metal")+control_test_kernels;
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
    std::cout << "RAPTOR adapter PASS PX4 target-frame transform + 0.5 s velocity preview, max_error=" << max_error << "\n";
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
    std::cout<<"ppo PASS CPU/Metal forward, terminal+truncation GAE, PPO gradients, reduction, Adam; toy target error "<<start_error<<" -> "<<final_error<<"\n";
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
    float speed=2,distance=4,wind=0,depth_noise=0,dropout=0,risk_coef=0,entropy_coef=.005f,learning_rate=.0003f;uint32_t velocity_contract=1;
};
static_assert(sizeof(SimRun)==184 && sizeof(SimConfig)==80,"sim layout mismatch");
static_assert((fixed_ppo::actor_obs_dim==181||fixed_ppo::actor_obs_dim==184) || fixed_ppo::actor_obs_dim==661,
              "navigation actor supports only pooled-181 or raw-661 observations");
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

struct Sim {
    Metal& m;SimConfig cfg;uint32_t horizon;
    id<MTLBuffer> states,runs,worlds,sensors,commands,raptor,physics,actor,critic,obs,co,actions,logp,values,rewards,next_values,terminated,truncated,advantages,returns;
    std::vector<id<MTLBuffer>> configs;
    id<MTLComputePipelineState> reset_p,depth_p,observe_p,act_p,advance_p,actor_p;
    id<MTLBuffer> actor_workspace,env_count;
    bool simd_actor=true;
    Sim(Metal& metal,SimConfig config,uint32_t steps):m(metal),cfg(config),horizon(steps) {
        auto w=load_raptor();auto p=rl_physics_crazyflie_default();uint n=cfg.n;size_t rows=n*size_t(horizon);
        states=m.buffer(n*sizeof(RLPhysicsState));runs=m.buffer(n*sizeof(SimRun));worlds=m.buffer(n*sizeof(WWorld));sensors=m.buffer(n*8*320*4);commands=m.buffer(n*8*4*4);raptor=m.buffer(sizeof(w),&w);physics=m.buffer(sizeof(p),&p);
        fixed_ppo::ActorParams a;fixed_ppo::CriticParams c;uint seed=1789;
        auto normal=[&](){float u=std::max(wurand(seed),1e-7f);return std::sqrt(-2*std::log(u))*std::cos(6.2831853f*wurand(seed));};
        for(size_t i=0;i<fixed_ppo::actor_b1_offset;i++)a.values[i]=normal()*0.04f;
        for(size_t i=fixed_ppo::actor_w2_offset;i<fixed_ppo::actor_b2_offset;i++)a.values[i]=normal()*0.01f;
        for(size_t i=0;i<4;i++)a.values[fixed_ppo::actor_log_std_offset+i]=-1.0f;
        for(size_t i=0;i<fixed_ppo::critic_b1_offset;i++)c.values[i]=normal()*0.15f;
        for(size_t i=fixed_ppo::critic_w2_offset;i<fixed_ppo::critic_b2_offset;i++)c.values[i]=normal()*0.1f;
        actor=m.buffer(sizeof(a),&a);critic=m.buffer(sizeof(c),&c);obs=m.buffer(rows*fixed_ppo::actor_obs_dim*4);co=m.buffer(rows*32*4);actions=m.buffer(rows*4*4);logp=m.buffer(rows*4);values=m.buffer(rows*4);rewards=m.buffer(rows*4);next_values=m.buffer(rows*4);terminated=m.buffer(rows);truncated=m.buffer(rows);advantages=m.buffer(rows*4);returns=m.buffer(rows*4);
        for(uint t=0;t<horizon;t++){cfg.tick=t;configs.push_back(m.buffer(sizeof(cfg),&cfg));}cfg.tick=0;
        simd_actor=!(std::getenv("METAL_NAV_SCALAR_ACTOR") && std::string(std::getenv("METAL_NAV_SCALAR_ACTOR"))=="1");
        actor_p=m.pipeline(simd_actor?"ppo_actor_forward_simd_fused":"ppo_actor_forward");
        require(!simd_actor || (fixed_ppo::hidden_dim==64 && actor_p.threadExecutionWidth==32 && actor_p.maxTotalThreadsPerThreadgroup>=256),"SIMD actor requires64hidden/M3-style32lane/256threadgroups");
        actor_workspace=m.buffer(cfg.n*fixed_ppo::hidden_dim*4);env_count=m.buffer(sizeof(cfg.n),&cfg.n);
        reset_p=m.pipeline("sim_reset");depth_p=m.pipeline("sim_depth");observe_p=m.pipeline("sim_observe");act_p=m.pipeline("sim_act");advance_p=m.pipeline("sim_advance");reset();
    }
    void set_config(SimConfig config){require(config.n==cfg.n,"cannot resize simulator");cfg=config;for(uint t=0;t<horizon;t++){cfg.tick=t;std::memcpy(configs[t].contents,&cfg,sizeof(cfg));}cfg.tick=0;}
    void reset(){auto cb=[m.queue commandBuffer];m.dispatch(cb,reset_p,cfg.n,{states,runs,worlds,sensors,commands,raptor,physics,configs[0]},64);m.finish(cb);}
    void collect(id<MTLCommandBuffer> cb,uint count=0){if(count==0)count=horizon;for(uint t=0;t<count;t++){auto c=configs[t%horizon];m.dispatch(cb,depth_p,cfg.n*320,{states,runs,worlds,sensors,physics,c});m.dispatch(cb,observe_p,cfg.n,{states,runs,worlds,sensors,obs,co,physics,c},64);encode(m,cb,simd_actor?"ppo_actor_forward_simd_fused":"ppo_actor_forward",simd_actor?((size_t(cfg.n)+7)/8)*256:cfg.n,{{obs,size_t(t%horizon)*cfg.n*fixed_ppo::actor_obs_dim*4},{actor,0},{actor_workspace,0},{actions,size_t(t%horizon)*cfg.n*4*4},{env_count,0}},simd_actor?256:64);m.dispatch(cb,act_p,cfg.n,{states,runs,worlds,obs,co,actor,critic,actions,logp,values,commands,physics,c},64);m.dispatch(cb,advance_p,cfg.n,{states,runs,worlds,sensors,commands,raptor,critic,rewards,next_values,terminated,truncated,physics,c},64);}}
    void report(const std::string& label,double wall){const SimRun* r=(const SimRun*)runs.contents;uint64_t success=0,collision=0,timeout=0,ep=0;double time=0,path=0,total=0,progress=0;float clearance=12,peak=0;
        for(uint i=0;i<cfg.n;i++){success+=r[i].successes;collision+=r[i].collisions;timeout+=r[i].timeouts;ep+=r[i].episodes;time+=r[i].success_time;path+=r[i].total_path;total+=r[i].total_elapsed;progress+=r[i].final_progress;clearance=fmin(clearance,r[i].min_clearance);peak=fmax(peak,r[i].peak_speed);}
        std::cout<<label<<" episodes="<<ep<<" success="<<(ep?double(success)/ep:0)<<" collision="<<(ep?double(collision)/ep:0)<<" timeout="<<(ep?double(timeout)/ep:0)<<" progress="<<(ep?progress/ep:0)<<" mean_goal_time="<<(success?time/success:0)<<" mean_speed="<<(total?path/total:0)<<" peak_speed="<<peak<<" min_clearance="<<clearance<<" wall_s="<<wall<<"\n";
    }
};
struct EvalScore { double success=0,collision=0,timeout=1,goal_time=1e9; };
static EvalScore sim_evaluate(Metal& m,uint family,uint mode,const float* actor=nullptr,float speed=1,float distance=3,uint32_t seed=700001,uint32_t sensor_delay=0,float wind=0,float noise=0,float dropout=0,uint32_t command_delay=0,uint32_t velocity_contract=1) {
    SimConfig cfg;cfg.n=128;cfg.family=family;cfg.mode=mode;cfg.eval=1;cfg.seed=seed;cfg.distance=distance;cfg.speed=speed;cfg.sensor_delay=sensor_delay;cfg.wind=wind;cfg.depth_noise=noise;cfg.dropout=dropout;cfg.command_delay=command_delay;cfg.velocity_contract=velocity_contract;require(sensor_delay<=6 && command_delay<=7,"delay rings support <=6 sensor frames and <=7 command steps");
    Sim sim(m,cfg,32);if(actor)std::memcpy(sim.actor.contents,actor,fixed_ppo::actor_param_count*4);
    double start=seconds();auto cb=[m.queue commandBuffer];sim.collect(cb,cfg.max_steps);double gpu=m.finish(cb);sim.report("eval family="+std::to_string(family)+" mode="+std::to_string(mode),seconds()-start);std::cout<<"eval_GPU_s="<<gpu<<"\n";
    const auto* r=(const SimRun*)sim.runs.contents;double success=0,collision=0,timeout=0,time=0;for(uint i=0;i<cfg.n;i++){require(r[i].episodes==1,"eval must finish exactly one episode per seed");success+=r[i].successes;collision+=r[i].collisions;timeout+=r[i].timeouts;time+=r[i].success_time;}return {success/cfg.n,collision/cfg.n,timeout/cfg.n,success?time/success:1e9};
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
            sim.reset();start=seconds();for(uint t=0;t<8;t++){cb=[m.queue commandBuffer];auto c=sim.configs[t];m.dispatch(cb,sim.depth_p,n*320,{sim.states,sim.runs,sim.worlds,sim.sensors,sim.physics,c});m.dispatch(cb,sim.observe_p,n,{sim.states,sim.runs,sim.worlds,sim.sensors,sim.obs,sim.co,sim.physics,c},64);encode(m,cb,sim.simd_actor?"ppo_actor_forward_simd_fused":"ppo_actor_forward",sim.simd_actor?((size_t(n)+7)/8)*256:n,{{sim.obs,size_t(t)*n*fixed_ppo::actor_obs_dim*4},{sim.actor,0},{sim.actor_workspace,0},{sim.actions,size_t(t)*n*4*4},{sim.env_count,0}},sim.simd_actor?256:64);m.dispatch(cb,sim.act_p,n,{sim.states,sim.runs,sim.worlds,sim.obs,sim.co,sim.actor,sim.critic,sim.actions,sim.logp,sim.values,sim.commands,sim.physics,c},64);m.dispatch(cb,sim.advance_p,n,{sim.states,sim.runs,sim.worlds,sim.sensors,sim.commands,sim.raptor,sim.critic,sim.rewards,sim.next_values,sim.terminated,sim.truncated,sim.physics,c},64);m.finish(cb);}sync=std::min(sync,seconds()-start);
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
static_assert(sizeof(PpoCheckpointHeader) == 128, "PPO checkpoint header layout mismatch");

static PpoCheckpointHeader read_checkpoint_header(std::ifstream& file) {
    PpoCheckpointHeader h{};static_assert(offsetof(PpoCheckpointHeader,config)==48,"checkpoint prefix");
    file.read((char*)&h,48);require(file && std::memcmp(h.magic,"PPOFIX1",7)==0 && (h.version>=3&&h.version<=5),"checkpoint prefix/version");
    file.read((char*)&h.config,64);if(h.version>=4){file.read((char*)&h+112,16);if(h.version==4)h.config.velocity_contract=0;}else{h.config.risk_coef=0;h.config.entropy_coef=h.family==7?.002f:.005f;h.config.learning_rate=.0003f;h.config.velocity_contract=0;}
    require(bool(file),"checkpoint header truncated");return h;
}

struct PPOTrainer {
    Sim& sim;
    Metal& metal;
    uint32_t minibatch = 256;
    uint32_t rows, minibatches, updates_per_rollout;
    uint64_t optimizer_step = 0;
    uint32_t completed_rollouts = 0;
    std::vector<uint32_t> starts;
    id<MTLBuffer> actor_hidden, actor_means, critic_hidden, predicted_values;
    id<MTLBuffer> d_means, d_log_stds, d_values, losses;
    id<MTLBuffer> actor_hidden_delta, actor_grad, critic_hidden_delta, critic_grad;
    id<MTLBuffer> actor_m, actor_v, critic_m, critic_v;
    id<MTLBuffer> actor_scale, critic_scale, metric_rows, metric_mean;
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
    PPOTrainer(Sim& s):sim(s),metal(s.m) {
        rows=sim.cfg.n*sim.horizon;
        require(rows>0,"PPO requires at least one rollout row");
        minibatches=(rows+minibatch-1)/minibatch;
        updates_per_rollout=2*minibatches;
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
        metric_rows=metal.buffer(size_t(updates_per_rollout)*4*4); metric_mean=metal.buffer(4*4);
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
        require(f && std::memcmp(h.magic,"PPOFIX1",7)==0 && (h.version>=3&&h.version<=5),"invalid PPO checkpoint header");
        require(h.actor_count==fixed_ppo::actor_param_count && h.critic_count==fixed_ppo::critic_param_count,"PPO checkpoint dimensions mismatch");
        require(h.family==family && h.horizon==horizon && h.n==n && h.base_seed==seed,"PPO checkpoint config mismatch");
        require(h.config.family==sim.cfg.family && h.config.n==sim.cfg.n && h.config.substeps==sim.cfg.substeps &&
                h.config.sensor_period==sim.cfg.sensor_period && h.config.sensor_delay==sim.cfg.sensor_delay &&
                h.config.command_delay==sim.cfg.command_delay && h.config.max_steps==sim.cfg.max_steps &&
                h.config.mode==sim.cfg.mode && h.config.speed==sim.cfg.speed && h.config.distance==sim.cfg.distance &&
                h.config.wind==sim.cfg.wind && h.config.depth_noise==sim.cfg.depth_noise && h.config.dropout==sim.cfg.dropout && h.config.risk_coef==sim.cfg.risk_coef && h.config.entropy_coef==sim.cfg.entropy_coef && h.config.learning_rate==sim.cfg.learning_rate && h.config.velocity_contract==sim.cfg.velocity_contract,
                "PPO checkpoint simulator settings mismatch");
        auto readbuf=[&](id<MTLBuffer> b,size_t count){f.read((char*)b.contents,count*4);require(bool(f),"truncated PPO checkpoint");};
        readbuf(sim.actor,fixed_ppo::actor_param_count); readbuf(sim.critic,fixed_ppo::critic_param_count);
        readbuf(actor_m,fixed_ppo::actor_param_count); readbuf(actor_v,fixed_ppo::actor_param_count);
        readbuf(critic_m,fixed_ppo::critic_param_count); readbuf(critic_v,fixed_ppo::critic_param_count);
        auto readbytes=[&](id<MTLBuffer> b){f.read((char*)b.contents,b.length);require(bool(f),"truncated PPO simulator checkpoint");};
        readbytes(sim.states);readbytes(sim.runs);readbytes(sim.worlds);readbytes(sim.sensors);readbytes(sim.commands);
        sim.set_config(h.config);
        char extra; require(!f.read(&extra,1),"PPO checkpoint has trailing data");
        optimizer_step=h.optimizer_step;completed_rollouts=h.completed_rollouts;
        require(optimizer_step<=uint64_t(std::numeric_limits<uint32_t>::max()),"PPO checkpoint Adam step is too large");
        std::cout<<"PPO resume rollouts="<<h.completed_rollouts<<" optimizer_step="<<optimizer_step<<"\n";
    }
    void save_checkpoint(const std::string& path,uint32_t family,uint32_t seed,uint32_t completed_rollouts) {
        if(path.empty())return;
        const std::filesystem::path checkpoint_path(path);
        if(checkpoint_path.has_parent_path())std::filesystem::create_directories(checkpoint_path.parent_path());
        PpoCheckpointHeader h{};std::memcpy(h.magic,"PPOFIX1",7);h.version=5;
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
        writebytes(sim.states);writebytes(sim.runs);writebytes(sim.worlds);writebytes(sim.sensors);writebytes(sim.commands);
        f.flush();require(bool(f),"PPO checkpoint write failed: "+temp);f.close();
        require(std::rename(temp.c_str(),path.c_str())==0,"cannot replace PPO checkpoint: "+path);
    }
    void rollout_update(id<MTLCommandBuffer> cb,uint32_t rollout_index) {
        dispatch(cb,gae_p,sim.cfg.n,{{sim.rewards,0},{sim.values,0},{sim.next_values,0},{sim.terminated,0},{sim.truncated,0},
                 {sim.advantages,0},{sim.returns,0},{horizon_b,0},{envs_b,0},{gamma_b,0},{lambda_b,0}},64);
        dispatch(cb,normalize_p,1,{{sim.advantages,0},{rows_b,0},{gae_epsilon_b,0}},1);
        std::mt19937 rng(0x9e3779b9u+rollout_index*0x85ebca6bu);std::shuffle(starts.begin(),starts.end(),rng);
        uint32_t update=0;
        for(uint32_t epoch=0;epoch<2;epoch++) {
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
                dispatch(cb,metric_batch_p,1,{{losses,0},{metric_rows,size_t(update)*4*4},{bb,0}},1);
                dispatch(cb,actor_hidden_delta_p,size_t(b)*fixed_ppo::hidden_dim,{{sim.actor,0},{d_means,0},{actor_hidden,0},{actor_hidden_delta,0},{bb,0}},128);
                dispatch(cb,actor_grad_direct_p,fixed_ppo::actor_param_count,{{sim.obs,obs_offset},{actor_hidden,0},{d_means,0},{d_log_stds,0},{actor_hidden_delta,0},{actor_grad,0},{bb,0}},128);
                dispatch(cb,critic_hidden_delta_p,size_t(b)*fixed_ppo::hidden_dim,{{sim.critic,0},{d_values,0},{critic_hidden,0},{critic_hidden_delta,0},{bb,0}},128);
                dispatch(cb,critic_grad_direct_p,fixed_ppo::critic_param_count,{{sim.co,co_offset},{critic_hidden,0},{d_values,0},{critic_hidden_delta,0},{critic_grad,0},{bb,0}},128);
                require(optimizer_step<uint64_t(std::numeric_limits<uint32_t>::max()),"PPO Adam step overflow");
                const uint32_t step=uint32_t(++optimizer_step);
                PpoAdamHostConfig ac{sim.cfg.learning_rate,.9f,.999f,1.0e-8f,0.0f,step};
                std::memcpy(adam_configs[update].contents,&ac,sizeof(ac));
                dispatch(cb,norm_factor_p,256,{{actor_grad,0},{actor_scale,0},{actor_count_b,0},{actor_max_norm_b,0}},256);
                dispatch(cb,scale_grad_p,fixed_ppo::actor_param_count,{{actor_grad,0},{actor_scale,0},{actor_count_b,0}},128);
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
};

static void train_navigation(Metal& m,uint32_t iterations,uint32_t family,const std::string& checkpoint,const std::string& warmstart="",float speed=1,float distance=3,float risk=0,float entropy=-1,float learning_rate=.0003f,uint32_t velocity_contract=1) {
    require(iterations>0,"train_navigation needs at least one rollout");
    m.compile(base_source()+PPO_TRAINER_MSL);
    constexpr uint32_t n=128,horizon=32,base_seed=42;
    SimConfig cfg;cfg.n=n;cfg.family=family;cfg.mode=0;cfg.seed=base_seed;cfg.distance=distance;cfg.speed=speed;cfg.max_steps=200;cfg.risk_coef=risk;cfg.entropy_coef=entropy>=0?entropy:(family==7?.002f:.005f);cfg.learning_rate=learning_rate;cfg.velocity_contract=velocity_contract;
    Sim sim(m,cfg,horizon);PPOTrainer trainer(sim);trainer.load_checkpoint(checkpoint,family,horizon,n,base_seed);
    if(!warmstart.empty() && trainer.completed_rollouts==0) {
        std::ifstream f(warmstart,std::ios::binary);PpoCheckpointHeader h=read_checkpoint_header(f);
        require(f && (h.version>=3&&h.version<=5) && h.actor_count==fixed_ppo::actor_param_count && h.critic_count==fixed_ppo::critic_param_count,"warmstart checkpoint format mismatch");
        f.read((char*)sim.actor.contents,fixed_ppo::actor_param_count*4);f.read((char*)sim.critic.contents,fixed_ppo::critic_param_count*4);require(bool(f),"warmstart read failed");
        for(uint j=0;j<4;j++)((float*)sim.actor.contents)[fixed_ppo::actor_log_std_offset+j]=-1.0f;
        std::cout<<"warmstart parameters from "<<warmstart<<"; reset optimizer and exploration\n";
    }
    EvalScore best;
    if(!checkpoint.empty() && std::filesystem::exists(checkpoint+".best")) {
        std::ifstream f(checkpoint+".best",std::ios::binary);PpoCheckpointHeader h=read_checkpoint_header(f);std::vector<float> a(fixed_ppo::actor_param_count);f.read((char*)a.data(),a.size()*4);require(f && h.actor_count==fixed_ppo::actor_param_count,"best checkpoint read/dimensions");best=sim_evaluate(m,family,4,a.data(),cfg.speed,cfg.distance);
    }
    std::filesystem::create_directories("results");
    std::ofstream history("results/training.tsv",std::ios::app);
    const double run_start=seconds();
    const uint32_t start_rollout=trainer.completed_rollouts;
    const uint32_t finish_rollout=start_rollout+iterations;
    std::cout<<"PPO training device="<<m.device.name.UTF8String<<" family="<<family<<" envs="<<n<<" horizon="<<horizon<<" speed="<<cfg.speed<<" distance="<<cfg.distance<<" risk="<<cfg.risk_coef<<" entropy="<<cfg.entropy_coef<<" lr="<<cfg.learning_rate<<"\n";
    for(uint32_t r=start_rollout;r<finish_rollout;r++) {@autoreleasepool{
        if(m.profiler.available)m.profiler.arm();const double rollout_start=seconds();auto cb=[m.queue commandBuffer];sim.collect(cb,horizon);trainer.rollout_update(cb,r);
        const double gpu=m.finish(cb);
        const float* met=(const float*)trainer.metric_mean.contents;
        if((r+1)%10==0 || r+1==finish_rollout) {
            std::cout<<"train rollout="<<(r+1)<<" policy_loss="<<met[0]<<" value_loss="<<met[1]<<" entropy_loss="<<met[2]<<" ratio="<<met[3]<<" optimizer_step="<<trainer.optimizer_step<<" gpu_s="<<gpu<<"\n";
            sim.report("train",0.0);
            EvalScore score=sim_evaluate(m,family,4,(const float*)sim.actor.contents,cfg.speed,cfg.distance);
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

static void evaluate_checkpoint(Metal& m,const std::string& checkpoint,uint mode,uint family,uint32_t seed=800001,float speed=1,float distance=3,uint32_t sensor_delay=0,float wind=0,float noise=0,float dropout=0,uint32_t command_delay=0) {
    std::ifstream f(checkpoint,std::ios::binary);PpoCheckpointHeader h=read_checkpoint_header(f);require(f && (h.version>=3&&h.version<=5) && h.actor_count==fixed_ppo::actor_param_count,"evaluation checkpoint format");std::vector<float>a(h.actor_count);f.read((char*)a.data(),a.size()*4);require(bool(f),"evaluation policy read");sim_evaluate(m,family,mode,a.data(),speed,distance,seed,sensor_delay,wind,noise,dropout,command_delay,h.config.velocity_contract);
}

static void gpu_training_benchmark(Metal& m,uint32_t n,uint32_t rollouts,uint32_t family=1) {
    m.compile(base_source()+PPO_TRAINER_MSL);SimConfig cfg;cfg.n=n;cfg.family=family;cfg.speed=1;cfg.distance=3;Sim sim(m,cfg,32);PPOTrainer trainer(sim);
    double gpu=0,start=seconds();for(uint32_t r=0;r<rollouts;r++){@autoreleasepool{auto cb=[m.queue commandBuffer];sim.collect(cb);trainer.rollout_update(cb,r);gpu+=m.finish(cb);}}
    double wall=seconds()-start;const auto* metrics=(const float*)trainer.metric_mean.contents;
    std::cout<<"GPU training family="<<family<<" n="<<n<<" rollouts="<<rollouts<<" transitions="<<uint64_t(n)*32*rollouts<<" gpu_s="<<gpu<<" wall_s="<<wall<<" optimizer_step="<<trainer.optimizer_step<<" policy="<<metrics[0]<<" value="<<metrics[1]<<" entropy="<<metrics[2]<<" ratio="<<metrics[3]<<"\n";
}

int main(int argc,char** argv){@autoreleasepool{try{
    std::string command=argc>1?argv[1]:"test";if(command=="cpu-bench"){cpu_reference_benchmark(argc>2?std::stoul(argv[2]):3,argc>4?std::stoul(argv[4]):1,argc>3?std::stoul(argv[3]):128);return 0;}Metal m(command=="profile");m.compile(base_source());std::cout<<"device="<<m.device.name.UTF8String<<" FP32 safe/precise\n";
    if(command=="test"){world_tests(m);raptor_tests(m);physics_tests(m);raptor_px4_adapter_tests();require(fixed_ppo::run_cpu_self_tests(),"PPO CPU self tests");ppo_tests(m);closed_loop_tests(m);}else if(command=="bench-depth")depth_benchmark(m);else if(command=="eval")evaluate_checkpoint(m,argc>2?argv[2]:"results/open.bin",argc>3?std::stoul(argv[3]):4,argc>4?std::stoul(argv[4]):0,argc>5?std::stoul(argv[5]):800001,argc>6?std::stof(argv[6]):1,argc>7?std::stof(argv[7]):3,argc>8?std::stoul(argv[8]):0,argc>9?std::stof(argv[9]):0,argc>10?std::stof(argv[10]):0,argc>11?std::stof(argv[11]):0,argc>12?std::stoul(argv[12]):0);else if(command=="gpu-bench")gpu_training_benchmark(m,argc>2?std::stoul(argv[2]):128,argc>3?std::stoul(argv[3]):3,argc>4?std::stoul(argv[4]):1);else if(command=="profile")train_navigation(m,1,0,"results/profile.bin");else if(command=="train")train_navigation(m,argc>2?std::stoul(argv[2]):100,argc>3?std::stoul(argv[3]):0,argc>4?argv[4]:"results/checkpoint.bin",argc>5?argv[5]:"",argc>6?std::stof(argv[6]):1,argc>7?std::stof(argv[7]):3,argc>8?std::stof(argv[8]):0,argc>9?std::stof(argv[9]):-1,argc>10?std::stof(argv[10]):.0003f,argc>11?std::stoul(argv[11]):1);else if(command=="bench-loop")loop_benchmark(m);else if(command=="bench-raptor")raptor_benchmark(m);else if(command=="sim"){for(uint f=0;f<4;f++)sim_evaluate(m,f,2);}else throw std::runtime_error("Unknown command "+command);
    return 0;
}catch(const std::exception& e){std::cerr<<"ERROR: "<<e.what()<<"\n";return 1;}}}
