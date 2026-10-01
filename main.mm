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
    Metal() {
        device=MTLCreateSystemDefaultDevice();require(device!=nil,"No Metal device"); queue=[device newCommandQueue];
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
    void dispatch(id<MTLCommandBuffer> cb,id<MTLComputePipelineState> p,size_t count,std::initializer_list<id<MTLBuffer>> buffers,size_t group=128) {
        id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];[e setComputePipelineState:p];uint j=0;for(auto b:buffers)[e setBuffer:b offset:0 atIndex:j++];
        [e dispatchThreads:MTLSizeMake(count,1,1) threadsPerThreadgroup:MTLSizeMake(std::min(group,size_t(p.maxTotalThreadsPerThreadgroup)),1,1)];[e endEncoding];
    }
    double finish(id<MTLCommandBuffer> cb) { [cb commit];[cb waitUntilCompleted];require(cb.status!=MTLCommandBufferStatusError,cb.error?cb.error.localizedDescription.UTF8String:"GPU failure");return cb.GPUEndTime-cb.GPUStartTime; }
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
    return "#include <metal_stdlib>\nusing namespace metal;\n"+read_text(root+"/world.hpp")+world_kernels+read_text(root+"/raptor.metal")+read_text(root+"/physics.metal")+read_text(root+"/ppo.metal")+read_text(root+"/sim.metal")+control_test_kernels;
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
        for(uint32_t n=0;n<B;++n)toy_oldlp[n]=gaussian_log_prob(toy_actions.data()+n*action_dim,means+n*action_dim,toy.values.data()+actor_log_std_offset);
        std::memcpy(toy_oldlp_b.contents,toy_oldlp.data(),B*4);
        toy_optimizer.step=step;std::memcpy(toy_opt_b.contents,&toy_optimizer,sizeof(toy_optimizer));
        cb=[m.queue commandBuffer];
        m.dispatch(cb,m.pipeline("ppo_sample_loss_grad"),B,{toy_actions_b,toy_means_b,toy_log_std_b,toy_oldlp_b,toy_values_b,toy_oldv_b,toy_adv_b,toy_ret_b,toy_dmu_b,toy_dstd_b,toy_dv_b,toy_loss_b,batch_b,clip_b,vc_b,ec_b});
        m.dispatch(cb,m.pipeline("ppo_actor_backward"),B,{zobs_b,toy_hidden_b,toy_ap_b,toy_dmu_b,toy_dstd_b,toy_part_b,batch_b});
        m.dispatch(cb,m.pipeline("ppo_reduce_grads"),actor_param_count,{toy_part_b,toy_grad_b,toy_params_count_b,batch_b});
        m.dispatch(cb,m.pipeline("ppo_adam_update"),actor_param_count,{toy_ap_b,toy_grad_b,toy_m_b,toy_v_b,toy_params_count_b,toy_opt_b});
        m.dispatch(cb,m.pipeline("ppo_actor_forward"),B,{zobs_b,toy_ap_b,toy_hidden_b,toy_means_b,batch_b});m.finish(cb);
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
    float success_time,total_path,total_elapsed,final_progress;
};
struct SimConfig {
    uint32_t n=128,tick=0,mode=0,family=0,substeps=5,sensor_period=1,sensor_delay=0,command_delay=0,max_steps=200,seed=42,eval=0;
    float speed=2,distance=4,wind=0,depth_noise=0,dropout=0;
};
static_assert(sizeof(SimRun)==172 && sizeof(SimConfig)==64,"sim layout mismatch");
using BufferBinding=std::pair<id<MTLBuffer>,size_t>;
static void encode(Metal& m,id<MTLCommandBuffer> cb,const char* name,size_t count,std::initializer_list<BufferBinding> bindings,size_t group=128) {
    auto p=m.pipeline(name);auto e=[cb computeCommandEncoder];[e setComputePipelineState:p];uint j=0;for(auto binding:bindings)[e setBuffer:binding.first offset:binding.second atIndex:j++];
    [e dispatchThreads:MTLSizeMake(count,1,1) threadsPerThreadgroup:MTLSizeMake(std::min(group,size_t(p.maxTotalThreadsPerThreadgroup)),1,1)];[e endEncoding];
}
struct Sim {
    Metal& m;SimConfig cfg;uint32_t horizon;
    id<MTLBuffer> states,runs,worlds,sensors,commands,raptor,physics,actor,critic,obs,co,actions,logp,values,rewards,next_values,terminated,truncated,advantages,returns;
    std::vector<id<MTLBuffer>> configs;
    id<MTLComputePipelineState> reset_p,depth_p,observe_p,act_p,advance_p;
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
        actor=m.buffer(sizeof(a),&a);critic=m.buffer(sizeof(c),&c);obs=m.buffer(rows*660*4);co=m.buffer(rows*32*4);actions=m.buffer(rows*4*4);logp=m.buffer(rows*4);values=m.buffer(rows*4);rewards=m.buffer(rows*4);next_values=m.buffer(rows*4);terminated=m.buffer(rows);truncated=m.buffer(rows);advantages=m.buffer(rows*4);returns=m.buffer(rows*4);
        for(uint t=0;t<horizon;t++){cfg.tick=t;configs.push_back(m.buffer(sizeof(cfg),&cfg));}cfg.tick=0;
        reset_p=m.pipeline("sim_reset");depth_p=m.pipeline("sim_depth");observe_p=m.pipeline("sim_observe");act_p=m.pipeline("sim_act");advance_p=m.pipeline("sim_advance");reset();
    }
    void set_config(SimConfig config){require(config.n==cfg.n,"cannot resize simulator");cfg=config;for(uint t=0;t<horizon;t++){cfg.tick=t;std::memcpy(configs[t].contents,&cfg,sizeof(cfg));}cfg.tick=0;}
    void reset(){auto cb=[m.queue commandBuffer];m.dispatch(cb,reset_p,cfg.n,{states,runs,worlds,sensors,commands,raptor,physics,configs[0]},64);m.finish(cb);}
    void collect(id<MTLCommandBuffer> cb,uint count=0){if(count==0)count=horizon;for(uint t=0;t<count;t++){auto c=configs[t%horizon];m.dispatch(cb,depth_p,cfg.n*320,{states,runs,worlds,sensors,physics,c});m.dispatch(cb,observe_p,cfg.n,{states,runs,worlds,sensors,obs,co,physics,c},64);m.dispatch(cb,act_p,cfg.n,{states,runs,worlds,obs,co,actor,critic,actions,logp,values,commands,physics,c},64);m.dispatch(cb,advance_p,cfg.n,{states,runs,worlds,sensors,commands,raptor,critic,rewards,next_values,terminated,truncated,physics,c},64);}}
    void report(const std::string& label,double wall){const SimRun* r=(const SimRun*)runs.contents;uint64_t success=0,collision=0,timeout=0,ep=0;double time=0,path=0,total=0,progress=0;float clearance=12,peak=0;
        for(uint i=0;i<cfg.n;i++){success+=r[i].successes;collision+=r[i].collisions;timeout+=r[i].timeouts;ep+=r[i].episodes;time+=r[i].success_time;path+=r[i].total_path;total+=r[i].total_elapsed;progress+=r[i].final_progress;clearance=fmin(clearance,r[i].min_clearance);peak=fmax(peak,r[i].peak_speed);}
        std::cout<<label<<" episodes="<<ep<<" success="<<(ep?double(success)/ep:0)<<" collision="<<(ep?double(collision)/ep:0)<<" timeout="<<(ep?double(timeout)/ep:0)<<" progress="<<(ep?progress/ep:0)<<" mean_goal_time="<<(success?time/success:0)<<" mean_speed="<<(total?path/total:0)<<" peak_speed="<<peak<<" min_clearance="<<clearance<<" wall_s="<<wall<<"\n";
    }
};
static void sim_evaluate(Metal& m,uint family,uint mode,const float* actor=nullptr) {
    SimConfig cfg;cfg.n=128;cfg.family=family;cfg.mode=mode;cfg.eval=1;cfg.seed=700001;cfg.distance=4;
    Sim sim(m,cfg,32);if(actor)std::memcpy(sim.actor.contents,actor,fixed_ppo::actor_param_count*4);
    double start=seconds();auto cb=[m.queue commandBuffer];sim.collect(cb,cfg.max_steps);double gpu=m.finish(cb);sim.report("eval family="+std::to_string(family)+" mode="+std::to_string(mode),seconds()-start);std::cout<<"eval_GPU_s="<<gpu<<"\n";
    const auto* r=(const SimRun*)sim.runs.contents;for(uint i=0;i<cfg.n;i++)require(r[i].episodes==1,"eval must finish exactly one episode per seed");
}

static void closed_loop_tests(Metal& m) {
    SimConfig cfg;cfg.n=32;cfg.mode=2;cfg.family=0;cfg.eval=1;cfg.speed=.5f;cfg.distance=4;
    Sim sim(m,cfg,32);std::vector<RLPhysicsState> states(cfg.n);std::vector<SimRun> runs(cfg.n);std::vector<WWorld> worlds(cfg.n);
    std::memcpy(states.data(),sim.states.contents,states.size()*sizeof(RLPhysicsState));std::memcpy(runs.data(),sim.runs.contents,runs.size()*sizeof(SimRun));std::memcpy(worlds.data(),sim.worlds.contents,worlds.size()*sizeof(WWorld));
    auto weights=load_raptor();auto params=rl_physics_crazyflie_default();
    for(uint n=0;n<cfg.n;n++)for(uint tick=0;tick<32;tick++) {
        auto& s=states[n];auto& run=runs[n];const auto& world=worlds[n];float rotation[9];raptor_quaternion_matrix(s.orientation_wxyz,rotation);
        WVec delta=wv(world.goal[0]-s.position[0],world.goal[1]-s.position[1],world.goal[2]-s.position[2]);WVec d=wm(wn(delta),fmin(wl(delta),cfg.speed));
        float body[3];for(uint j=0;j<3;j++)body[j]=(rotation[j]*d.x+rotation[3+j]*d.y+rotation[6+j]*d.z);body[2]*=.5f;
        float velocity[3];for(uint j=0;j<3;j++)velocity[j]=rotation[j*3]*body[0]+rotation[j*3+1]*body[1]+rotation[j*3+2]*body[2];
        const float qtarget[4]={1,0,0,0},wind[3]={0,0,0};
        for(uint sub=0;sub<cfg.substeps;sub++) {
            float target[3],obs[22];for(uint j=0;j<3;j++)target[j]=s.position[j]+velocity[j]*.5f;
            raptor_pack_observation(s.position,s.orientation_wxyz,s.linear_velocity,s.angular_velocity_body,target,qtarget,velocity,run.motors,obs);
            raptor_forward(weights,obs,run.hidden,run.motors);raptor_clip_action(run.motors);RLPhysicsState next;rl_physics_step(s,run.motors,wind,params,next);s=next;
        }
    }
    auto cb=[m.queue commandBuffer];sim.collect(cb);double gpu=m.finish(cb);float error=0,hidden_error=0;const auto* actual=(const RLPhysicsState*)sim.states.contents;const auto* actualrun=(const SimRun*)sim.runs.contents;
    for(uint n=0;n<cfg.n;n++) {require(actualrun[n].episodes==0,"closed-loop parity seed terminated early");for(uint j=0;j<17;j++)error=fmax(error,fabs(((const float*)&actual[n])[j]-((const float*)&states[n])[j]));for(uint j=0;j<16;j++)hidden_error=fmax(hidden_error,fabs(actualrun[n].hidden[j]-runs[n].hidden[j]));}
    require(error<5e-4f&&hidden_error<5e-4f,"closed-loop CPU/GPU trajectory mismatch");std::cout<<"closed loop PASS N=32 native_steps=160 state_error="<<error<<" hidden_error="<<hidden_error<<" GPU_s="<<gpu<<"\n";
}

int main(int argc,char** argv){@autoreleasepool{try{
    Metal m;m.compile(base_source());std::cout<<"device="<<m.device.name.UTF8String<<" FP32 safe/precise\n";std::string command=argc>1?argv[1]:"test";
    if(command=="test"){world_tests(m);raptor_tests(m);physics_tests(m);raptor_px4_adapter_tests();require(fixed_ppo::run_cpu_self_tests(),"PPO CPU self tests");ppo_tests(m);closed_loop_tests(m);}else if(command=="bench-depth")depth_benchmark(m);else if(command=="sim"){for(uint f=0;f<4;f++)sim_evaluate(m,f,2);}else throw std::runtime_error("Unknown command "+command);
    return 0;
}catch(const std::exception& e){std::cerr<<"ERROR: "<<e.what()<<"\n";return 1;}}}
