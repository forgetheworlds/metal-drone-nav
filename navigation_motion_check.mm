// CPU/Metal parity for bounded moving spheres and their collision geometry.
#define main embedded_navigation_main
#include "main.mm"
#undef main

static const char* motion_kernel=R"MSL(
kernel void motion_check(device const WWorld* worlds [[buffer(0)]],
                         device const float* times [[buffer(1)]],
                         device float* result [[buffer(2)]],
                         uint i [[thread_position_in_grid]]) {
    WVec center=wc(worlds[i].obstacles[0],times[i]);
    result[i*5]=center.x;result[i*5+1]=center.y;result[i*5+2]=center.z;
    WVec point=wv(center.x+worlds[i].obstacles[0].size[0]+.18f+1,center.y,center.z);
    result[i*5+3]=wclearance(worlds[i],point,times[i]);
    result[i*5+4]=wray(worlds[i],point,wv(-1,0,0),times[i]);
}
)MSL";

int main(int argc,char** argv) {@autoreleasepool {try {
    require(argc==2,"navigation_motion_check OUT_JSON");constexpr uint count=1001;
    std::vector<WWorld> worlds(count);std::vector<float> times(count),expected(count*5);
    for(uint i=0;i<count;i++) {
        auto& world=worlds[i];world.count=1;auto& o=world.obstacles[0];o.kind=3;o.center[0]=6;o.center[2]=1.5f;
        o.size[0]=.28f;o.size[1]=2.7f;o.size[2]=-1.57079632679f;
        o.velocity[1]=.8f;times[i]=i*.02f;
        WVec center=wc(world.obstacles[0],times[i]);
        expected[i*5]=center.x;expected[i*5+1]=center.y;expected[i*5+2]=center.z;
        WVec point=wv(center.x+o.size[0]+.18f+1,center.y,center.z);
        expected[i*5+3]=wclearance(world,point,times[i]);expected[i*5+4]=wray(world,point,wv(-1,0,0),times[i]);
        require(std::fabs(expected[i*5+3]-1)<1e-5f,"bounded sphere collision radius is wrong");
        require(std::fabs(expected[i*5+4]-1.18f)<1e-5f,"bounded sphere depth radius is wrong");
    }
    Metal metal;metal.compile(base_source()+PPO_TRAINER_MSL+motion_kernel);
    auto bodies=metal.buffer(worlds.size()*sizeof(WWorld),worlds.data());
    auto clock=metal.buffer(times.size()*4,times.data()),out=metal.buffer(expected.size()*4);
    auto commands=[metal.queue commandBuffer];metal.dispatch(commands,metal.pipeline("motion_check"),count,{bodies,clock,out},64);metal.finish(commands);
    const float* actual=static_cast<const float*>(out.contents);float worst=0;
    for(size_t i=0;i<expected.size();i++)worst=std::max(worst,std::fabs(actual[i]-expected[i]));
    require(worst<1e-5f,"bounded motion/collision CPU-Metal mismatch");
    std::ofstream report(argv[1]);require(bool(report),"cannot write motion proof");
    report<<"{\"samples\":1001,\"time_end_s\":20,\"max_cpu_metal_error\":"<<worst
          <<",\"sphere_clearance_m\":1,\"sphere_depth_m\":1.18,\"passed\":true}\n";
    std::cout<<"PASS bounded motion/collision/depth; max error="<<worst<<'\n';return 0;
}catch(const std::exception& e){std::cerr<<"ERROR: "<<e.what()<<'\n';return 1;}}}
