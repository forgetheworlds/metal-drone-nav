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
#include <cstring>
#include "world.hpp"

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
    Metal() {
        device=MTLCreateSystemDefaultDevice();require(device!=nil,"No Metal device"); queue=[device newCommandQueue];
    }
    void compile(std::string source) {
        MTLCompileOptions* options=[MTLCompileOptions new];options.mathMode=MTLMathModeSafe;options.mathFloatingPointFunctions=MTLMathFloatingPointFunctionsPrecise;
        NSError* e=nil;library=[device newLibraryWithSource:[NSString stringWithUTF8String:source.c_str()] options:options error:&e];
        require(library!=nil,e?e.localizedDescription.UTF8String:"Metal compile failed");
    }
    id<MTLComputePipelineState> pipeline(const char* name) {
        NSError* e=nil;id<MTLFunction> fn=[library newFunctionWithName:[NSString stringWithUTF8String:name]];require(fn!=nil,std::string("Missing kernel ")+name);
        id<MTLComputePipelineState> p=[device newComputePipelineStateWithFunction:fn error:&e];require(p!=nil,e?e.localizedDescription.UTF8String:"Pipeline failed");return p;
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

static std::string base_source() { return "#include <metal_stdlib>\nusing namespace metal;\n"+read_text(std::string(SOURCE_DIR)+"/world.hpp")+world_kernels; }
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
int main(int argc,char** argv){@autoreleasepool{try{
    Metal m;m.compile(base_source());std::cout<<"device="<<m.device.name.UTF8String<<" FP32 safe/precise\n";std::string command=argc>1?argv[1]:"test";
    if(command=="test")world_tests(m);else if(command=="bench-depth")depth_benchmark(m);else throw std::runtime_error("Unknown command "+command);
    return 0;
}catch(const std::exception& e){std::cerr<<"ERROR: "<<e.what()<<"\n";return 1;}}}
