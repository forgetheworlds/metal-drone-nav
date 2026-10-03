#ifndef TRAINING_POTENTIAL_HPP
#define TRAINING_POTENTIAL_HPP

#ifdef __METAL_VERSION__
typedef uint TPIndex;
#else
#include "challenge_evaluation.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <queue>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <CommonCrypto/CommonDigest.h>
using TPIndex = uint32_t;
#endif

// Shared CPU/MSL buffer ABI. Values use [level][z][y][x], with x contiguous.
// Each cell stores bounded Phi=-min(static geodesic distance, cap)/cap.
// Exactly -1 means unreachable. Reachable cells at the cap use the next float
// above -1 so lookup can distinguish them from the no-path sentinel.
#ifndef TRAINING_POTENTIAL_SHARED_TYPES
#define TRAINING_POTENTIAL_SHARED_TYPES
struct TrainingPotentialGridSpec {
    TPIndex nx, ny, nz, level_count, level_stride, version;
    float origin[3];
    float spacing_m;
    float distance_cap_m;
};

// Training-only shaping settings. `version` must match the potential grid.
struct TrainingPotentialControl {
    TPIndex enabled;
    float scale;
    float gamma;
    TPIndex version;
};

#endif

#ifndef __METAL_VERSION__
static_assert(sizeof(TrainingPotentialGridSpec) == 44, "training-potential Metal ABI changed");
static_assert(sizeof(TrainingPotentialControl) == 16, "training-potential control ABI changed");
#endif

#ifdef __METAL_VERSION__
constant float TRAINING_POTENTIAL_TERMINAL = 0.0f;
#else
constexpr float TRAINING_POTENTIAL_TERMINAL = 0.0f;
#endif

#ifdef __METAL_VERSION__
inline uint training_potential_index(constant TrainingPotentialGridSpec& spec,
                                    uint x, uint y, uint z) {
    return (z*spec.ny+y)*spec.nx+x;
}

inline float training_potential_lookup(device const float* bank,
                                       constant TrainingPotentialGridSpec& spec,
                                       uint level, thread const float* position) {
    if (level>=spec.level_count || !(spec.spacing_m>0.0f) ||
        !isfinite(position[0]) || !isfinite(position[1]) || !isfinite(position[2])) return -1.0f;
    const float gx=(position[0]-spec.origin[0])/spec.spacing_m;
    const float gy=(position[1]-spec.origin[1])/spec.spacing_m;
    const float gz=(position[2]-spec.origin[2])/spec.spacing_m;
    if (gx<0.0f || gy<0.0f || gz<0.0f || gx>float(spec.nx-1) ||
        gy>float(spec.ny-1) || gz>float(spec.nz-1)) return -1.0f;
    const uint x0=uint(floor(gx)),y0=uint(floor(gy)),z0=uint(floor(gz));
    const uint x1=min(x0+1,spec.nx-1),y1=min(y0+1,spec.ny-1),z1=min(z0+1,spec.nz-1);
    const float tx=gx-float(x0),ty=gy-float(y0),tz=gz-float(z0);
    const uint base=level*spec.level_stride;
    const float values[8]={bank[base+training_potential_index(spec,x0,y0,z0)],bank[base+training_potential_index(spec,x1,y0,z0)],
        bank[base+training_potential_index(spec,x0,y1,z0)],bank[base+training_potential_index(spec,x1,y1,z0)],
        bank[base+training_potential_index(spec,x0,y0,z1)],bank[base+training_potential_index(spec,x1,y0,z1)],
        bank[base+training_potential_index(spec,x0,y1,z1)],bank[base+training_potential_index(spec,x1,y1,z1)]};
    const float weights[8]={(1-tx)*(1-ty)*(1-tz),tx*(1-ty)*(1-tz),
        (1-tx)*ty*(1-tz),tx*ty*(1-tz),(1-tx)*(1-ty)*tz,tx*(1-ty)*tz,
        (1-tx)*ty*tz,tx*ty*tz};
    float weighted_phi=0.0f,total_weight=0.0f;
    for(uint i=0;i<8;i++)if(values[i]>-1.0f){weighted_phi+=values[i]*weights[i];total_weight+=weights[i];}
    if(!(total_weight>0.0f))return -1.0f;
    return clamp(weighted_phi/total_weight,-1.0f,0.0f);
}

// Direction of increasing Phi (toward the goal because Phi=-distance/cap).
// Refuse sentinel, flat, and non-finite neighborhoods instead of inventing a
// direction where the grid has no local route information.
inline bool training_potential_direction(device const float* bank,
                                         constant TrainingPotentialGridSpec& spec,
                                         uint level, thread const float* position,
                                         thread float* direction) {
    const float center=training_potential_lookup(bank,spec,level,position);
    if(!(center>-1.0f) || !isfinite(center))return false;
    float gradient[3];
    for(uint axis=0;axis<3;axis++) {
        float plus[3]={position[0],position[1],position[2]};
        float minus[3]={position[0],position[1],position[2]};
        plus[axis]+=spec.spacing_m;minus[axis]-=spec.spacing_m;
        const float p=training_potential_lookup(bank,spec,level,plus);
        const float m=training_potential_lookup(bank,spec,level,minus);
        if(!(p>-1.0f) || !(m>-1.0f) || !isfinite(p) || !isfinite(m))return false;
        gradient[axis]=(p-m)/(2.0f*spec.spacing_m);
    }
    const float magnitude=sqrt(gradient[0]*gradient[0]+gradient[1]*gradient[1]+gradient[2]*gradient[2]);
    if(!(magnitude>1.0e-5f) || !isfinite(magnitude))return false;
    for(uint axis=0;axis<3;axis++)direction[axis]=gradient[axis]/magnitude;
    float ahead[3]={position[0]+direction[0]*spec.spacing_m,
                    position[1]+direction[1]*spec.spacing_m,
                    position[2]+direction[2]*spec.spacing_m};
    const float uphill=training_potential_lookup(bank,spec,level,ahead);
    return uphill>center+1.0e-6f && isfinite(uphill);
}

#else

namespace training_potential {

constexpr uint32_t kVersion = 2;
constexpr uint32_t kNx = 81, kNy = 51, kNz = 26;
constexpr float kSpacingM = 0.20f;
constexpr float kDistanceCapM = 30.0f;
constexpr float kReachableSaturatedPhi = -0.99999994f;
constexpr float kGoalRadiusM = 0.35f;
constexpr float kBodyRadiusM = 0.18f;
constexpr float kCellClearanceM = 0.02f;
constexpr uint8_t kConnectedEdges = 6;
constexpr double kInfinity = 1.0e30;

struct TrainingPotentialFields {
    TrainingPotentialGridSpec spec{};
    std::vector<float> phi;
    std::string cache_key;
    std::string bank_sha256;
    std::string world_sha256;

    float lookup(uint32_t level,const std::array<float,3>& position) const;

    const float* data() const { return phi.data(); }
    size_t bytes() const { return phi.size()*sizeof(float); }
};

inline uint32_t cell_index(uint32_t x,uint32_t y,uint32_t z) {
    return (z*kNy+y)*kNx+x;
}
inline std::array<uint32_t,3> cell_xyz(uint32_t index) {
    const uint32_t x=index%kNx;
    const uint32_t y=(index/kNx)%kNy;
    const uint32_t z=index/(kNx*kNy);
    return {x,y,z};
}
inline std::array<float,3> cell_position(uint32_t x,uint32_t y,uint32_t z) {
    return {-2.0f+float(x)*kSpacingM,-5.0f+float(y)*kSpacingM,float(z)*kSpacingM};
}
inline uint32_t nearest_cell(const std::array<float,3>& point) {
    const auto clamp_index=[](float p,float origin,uint32_t count) {
        const float grid=(p-origin)/kSpacingM;
        return uint32_t(std::max(0.0f,std::min(float(count-1),std::floor(grid+0.5f))));
    };
    return cell_index(clamp_index(point[0],-2.0f,kNx),
                      clamp_index(point[1],-5.0f,kNy),
                      clamp_index(point[2],0.0f,kNz));
}
inline float point_clearance(const WWorld& world,const std::array<float,3>& p) {
    return wclearance(world,wv(p[0],p[1],p[2]),0.0f);
}
inline bool static_world(const WWorld& world) {
    for(uint32_t obstacle=0;obstacle<world.count;obstacle++)
        for(float velocity:world.obstacles[obstacle].velocity)
            if(std::fabs(velocity)>1.0e-8f)return false;
    return true;
}
inline float trilinear(const float* bank,const TrainingPotentialGridSpec& spec,
                       uint32_t level,const std::array<float,3>& position) {
    if(level>=spec.level_count || !std::isfinite(position[0]) ||
       !std::isfinite(position[1]) || !std::isfinite(position[2]))return -1.0f;
    const float gx=(position[0]-spec.origin[0])/spec.spacing_m;
    const float gy=(position[1]-spec.origin[1])/spec.spacing_m;
    const float gz=(position[2]-spec.origin[2])/spec.spacing_m;
    if(gx<0 || gy<0 || gz<0 || gx>float(spec.nx-1) || gy>float(spec.ny-1) || gz>float(spec.nz-1))return -1.0f;
    const uint32_t x0=uint32_t(std::floor(gx)),y0=uint32_t(std::floor(gy)),z0=uint32_t(std::floor(gz));
    const uint32_t x1=std::min(x0+1,spec.nx-1),y1=std::min(y0+1,spec.ny-1),z1=std::min(z0+1,spec.nz-1);
    const float tx=gx-float(x0),ty=gy-float(y0),tz=gz-float(z0);
    const size_t base=size_t(level)*spec.level_stride;
    const auto at=[&](uint32_t x,uint32_t y,uint32_t z){return bank[base+(z*spec.ny+y)*spec.nx+x];};
    const float values[8]={at(x0,y0,z0),at(x1,y0,z0),at(x0,y1,z0),at(x1,y1,z0),
        at(x0,y0,z1),at(x1,y0,z1),at(x0,y1,z1),at(x1,y1,z1)};
    const float weights[8]={(1-tx)*(1-ty)*(1-tz),tx*(1-ty)*(1-tz),
        (1-tx)*ty*(1-tz),tx*ty*(1-tz),(1-tx)*(1-ty)*tz,tx*(1-ty)*tz,
        (1-tx)*ty*tz,tx*ty*tz};
    float weighted_phi=0.0f,total_weight=0.0f;
    for(uint32_t i=0;i<8;i++)if(values[i]>-1.0f){weighted_phi+=values[i]*weights[i];total_weight+=weights[i];}
    if(!(total_weight>0.0f))return -1.0f;
    return std::max(-1.0f,std::min(0.0f,weighted_phi/total_weight));
}

inline float TrainingPotentialFields::lookup(uint32_t level,const std::array<float,3>& position) const {
    return trilinear(phi.data(),spec,level,position);
}

// Host twin of the MSL lookup above; both use the same cell layout, bounds,
// valid-corner interpolation and finite fallback.
inline float training_potential_lookup(const float* bank,const TrainingPotentialGridSpec& spec,
                                       uint32_t level,const float position[3]) {
    return trilinear(bank,spec,level,{position[0],position[1],position[2]});
}

inline bool training_potential_direction(const float* bank,const TrainingPotentialGridSpec& spec,
                                         uint32_t level,const float position[3],float direction[3]) {
    const float center=training_potential_lookup(bank,spec,level,position);
    if(!(center>-1.0f) || !std::isfinite(center))return false;
    float gradient[3];
    for(uint32_t axis=0;axis<3;axis++) {
        float plus[3]={position[0],position[1],position[2]};
        float minus[3]={position[0],position[1],position[2]};
        plus[axis]+=spec.spacing_m;minus[axis]-=spec.spacing_m;
        const float p=training_potential_lookup(bank,spec,level,plus);
        const float m=training_potential_lookup(bank,spec,level,minus);
        if(!(p>-1.0f) || !(m>-1.0f) || !std::isfinite(p) || !std::isfinite(m))return false;
        gradient[axis]=(p-m)/(2.0f*spec.spacing_m);
    }
    const float magnitude=std::sqrt(gradient[0]*gradient[0]+gradient[1]*gradient[1]+gradient[2]*gradient[2]);
    if(!(magnitude>1.0e-5f) || !std::isfinite(magnitude))return false;
    for(uint32_t axis=0;axis<3;axis++)direction[axis]=gradient[axis]/magnitude;
    const float ahead[3]={position[0]+direction[0]*spec.spacing_m,
                          position[1]+direction[1]*spec.spacing_m,
                          position[2]+direction[2]*spec.spacing_m};
    const float uphill=training_potential_lookup(bank,spec,level,ahead);
    return uphill>center+1.0e-6f && std::isfinite(uphill);
}

inline bool nearest_reachable_cell(const std::vector<uint8_t>& free_cells,
                                   const std::vector<float>& distance,
                                   const std::array<float,3>& point,
                                   float max_radius) {
    const uint32_t center=nearest_cell(point);
    const auto c=cell_xyz(center);
    const int radius=int(std::ceil(max_radius/kSpacingM));
    float nearest_squared=max_radius*max_radius;
    bool found=false;
    for(int dz=-radius;dz<=radius;dz++)for(int dy=-radius;dy<=radius;dy++)for(int dx=-radius;dx<=radius;dx++){
        const int x=int(c[0])+dx,y=int(c[1])+dy,z=int(c[2])+dz;
        if(x<0||x>=int(kNx)||y<0||y>=int(kNy)||z<0||z>=int(kNz))continue;
        const uint32_t index=cell_index(uint32_t(x),uint32_t(y),uint32_t(z));
        if(!free_cells[index]||distance[index]>=float(kInfinity))continue;
        const auto p=cell_position(uint32_t(x),uint32_t(y),uint32_t(z));
        const float d2=(p[0]-point[0])*(p[0]-point[0])+(p[1]-point[1])*(p[1]-point[1])+(p[2]-point[2])*(p[2]-point[2]);
        if(d2<=nearest_squared){nearest_squared=d2;found=true;}
    }
    return found;
}

inline std::vector<float> build_one_field(const challenge_evaluation::Level& level) {
    if(level.world.family<14||level.world.family>16)
        throw std::runtime_error("potential field accepts only static families14–16");
    if(!static_world(level.world))
        throw std::runtime_error("potential field refuses moving obstacles; use a separate transient layer");
    const uint32_t cells=kNx*kNy*kNz;
    std::vector<uint8_t> free_cells(cells,0),edge_masks(cells,0);
    for(uint32_t z=0;z<kNz;z++)for(uint32_t y=0;y<kNy;y++)for(uint32_t x=0;x<kNx;x++){
        const uint32_t index=cell_index(x,y,z);
        const float value=point_clearance(level.world,cell_position(x,y,z));
        if(!std::isfinite(value))throw std::runtime_error("potential grid has a non-finite clearance");
        free_cells[index]=value>=kCellClearanceM;
    }
    constexpr int delta[3][3]={{1,0,0},{0,1,0},{0,0,1}};
    for(uint32_t z=0;z<kNz;z++)for(uint32_t y=0;y<kNy;y++)for(uint32_t x=0;x<kNx;x++){
        const uint32_t index=cell_index(x,y,z);
        if(!free_cells[index])continue;
        const auto p=cell_position(x,y,z);
        for(uint32_t axis=0;axis<3;axis++){
            const int xx=int(x)+delta[axis][0],yy=int(y)+delta[axis][1],zz=int(z)+delta[axis][2];
            if(xx>=int(kNx)||yy>=int(kNy)||zz>=int(kNz))continue;
            const uint32_t neighbor=cell_index(uint32_t(xx),uint32_t(yy),uint32_t(zz));
            if(!free_cells[neighbor])continue;
            const auto q=cell_position(uint32_t(xx),uint32_t(yy),uint32_t(zz));
            const std::array<float,3> midpoint{{(p[0]+q[0])*0.5f,(p[1]+q[1])*0.5f,(p[2]+q[2])*0.5f}};
            if(point_clearance(level.world,midpoint)>=kCellClearanceM)edge_masks[index]|=uint8_t(1u<<axis);
        }
    }

    std::vector<float> distance(cells,float(kInfinity));
    using QueueNode=std::pair<float,uint32_t>;
    std::priority_queue<QueueNode,std::vector<QueueNode>,std::greater<QueueNode>> open;
    const auto goal=std::array<float,3>{{level.world.goal[0],level.world.goal[1],level.world.goal[2]}};
    uint32_t seeds=0;
    const float goal_r2=kGoalRadiusM*kGoalRadiusM;
    for(uint32_t index=0;index<cells;index++)if(free_cells[index]){
        const auto p=cell_position(cell_xyz(index)[0],cell_xyz(index)[1],cell_xyz(index)[2]);
        const float dx=p[0]-goal[0],dy=p[1]-goal[1],dz=p[2]-goal[2];
        if(dx*dx+dy*dy+dz*dz<=goal_r2){distance[index]=0;open.push({0,index});seeds++;}
    }
    if(!seeds)throw std::runtime_error("no free grid cell lies in the 0.35m goal region: "+level.failure_id);

    constexpr int directions[6][4]={{1,0,0,0},{-1,0,0,0},{0,1,0,1},{0,-1,0,1},{0,0,1,2},{0,0,-1,2}};
    while(!open.empty()){
        const QueueNode node=open.top();open.pop();
        if(node.first!=distance[node.second])continue;
        const auto c=cell_xyz(node.second);
        for(const auto& direction:directions){
            const int x=int(c[0])+direction[0],y=int(c[1])+direction[1],z=int(c[2])+direction[2];
            if(x<0||x>=int(kNx)||y<0||y>=int(kNy)||z<0||z>=int(kNz))continue;
            const uint32_t neighbor=cell_index(uint32_t(x),uint32_t(y),uint32_t(z));
            if(!free_cells[neighbor])continue;
            const uint32_t axis=uint32_t(direction[3]);
            const uint32_t edge_index=(direction[0]+direction[1]+direction[2]>0)?node.second:neighbor;
            if((edge_masks[edge_index]&(1u<<axis))==0)continue;
            const float candidate=node.first+kSpacingM;
            if(candidate<distance[neighbor]){distance[neighbor]=candidate;open.push({candidate,neighbor});}
        }
    }

    const std::array<float,3> start{{0,0,1.5f}};
    if(!nearest_reachable_cell(free_cells,distance,start,0.40f))
        throw std::runtime_error("grid start is not connected to a goal: "+level.failure_id);
    if(!nearest_reachable_cell(free_cells,distance,goal,0.40f))
        throw std::runtime_error("grid goal is not connected: "+level.failure_id);

    for(size_t segment=0;segment+1<level.witness_route.size();segment++){
        const auto& a=level.witness_route[segment];const auto& b=level.witness_route[segment+1];
        const float dx=b[0]-a[0],dy=b[1]-a[1],dz=b[2]-a[2];
        const float length=std::sqrt(dx*dx+dy*dy+dz*dz);
        const uint32_t samples=std::max(1u,uint32_t(std::ceil(length/0.10f)));
        for(uint32_t sample=0;sample<=samples;sample++){
            const float t=float(sample)/samples;
            const std::array<float,3> p{{a[0]+dx*t,a[1]+dy*t,a[2]+dz*t}};
            if(!nearest_reachable_cell(free_cells,distance,p,0.40f))
                throw std::runtime_error("0.2m grid misses witness passage; refine resolution: "+level.failure_id);
        }
    }

    std::vector<float> phi(cells,-1.0f);
    for(uint32_t index=0;index<cells;index++)if(free_cells[index]&&distance[index]<float(kInfinity))
        phi[index]=distance[index]>=kDistanceCapM?kReachableSaturatedPhi:-distance[index]/kDistanceCapM;
    TrainingPotentialGridSpec single_level_spec{};
    single_level_spec.nx=kNx;single_level_spec.ny=kNy;single_level_spec.nz=kNz;
    single_level_spec.level_count=1;single_level_spec.level_stride=cells;single_level_spec.version=kVersion;
    single_level_spec.origin[0]=-2.0f;single_level_spec.origin[1]=-5.0f;single_level_spec.origin[2]=0.0f;
    single_level_spec.spacing_m=kSpacingM;single_level_spec.distance_cap_m=kDistanceCapM;
    if(trilinear(phi.data(),single_level_spec,0,start)<=-1.0f||
       trilinear(phi.data(),single_level_spec,0,goal)<=-1.0f)
        throw std::runtime_error("interpolated grid lookup misses start or goal: "+level.failure_id);
    for(size_t segment=0;segment+1<level.witness_route.size();segment++){
        const auto& a=level.witness_route[segment];const auto& b=level.witness_route[segment+1];
        const float dx=b[0]-a[0],dy=b[1]-a[1],dz=b[2]-a[2];
        const float length=std::sqrt(dx*dx+dy*dy+dz*dz);
        const uint32_t samples=std::max(1u,uint32_t(std::ceil(length/0.10f)));
        for(uint32_t sample=0;sample<=samples;sample++){
            const float t=float(sample)/samples;
            const std::array<float,3> point{{a[0]+dx*t,a[1]+dy*t,a[2]+dz*t}};
            if(trilinear(phi.data(),single_level_spec,0,point)<=-1.0f)
                throw std::runtime_error("interpolated grid lookup misses witness route: "+level.failure_id);
        }
    }
    return phi;
}

inline TrainingPotentialGridSpec default_grid_spec(uint32_t level_count) {
    TrainingPotentialGridSpec spec{};
    spec.nx=kNx;spec.ny=kNy;spec.nz=kNz;spec.level_count=level_count;
    spec.level_stride=kNx*kNy*kNz;spec.version=kVersion;
    spec.origin[0]=-2.0f;spec.origin[1]=-5.0f;spec.origin[2]=0.0f;
    spec.spacing_m=kSpacingM;spec.distance_cap_m=kDistanceCapM;
    return spec;
}

inline std::string sha256_hex(const void* data,size_t size) {
    unsigned char digest[CC_SHA256_DIGEST_LENGTH];
    CC_SHA256(static_cast<const unsigned char*>(data),CC_LONG(size),digest);
    static const char hex[]="0123456789abcdef";std::string key(64,'0');
    for(size_t i=0;i<sizeof(digest);i++){key[2*i]=hex[digest[i]>>4];key[2*i+1]=hex[digest[i]&15];}
    return key;
}
inline std::string cache_key(const std::string& bank_hash,const std::string& world_hash) {
    const std::string params="training-potential-v2|grid=81x51x26|origin=-2,-5,0|spacing=0.2|body=0.18|cell_margin=0.02|goal_radius=0.35|cap=30|connectivity=6|reachable_cap=-0.99999994";
    const std::string material=bank_hash+"|"+world_hash+"|"+params;
    return sha256_hex(material.data(),material.size());
}

inline void write_cache(const TrainingPotentialFields& fields,const std::string& path) {
    const std::string temporary=path+".tmp";
    if(std::filesystem::path(path).has_parent_path())std::filesystem::create_directories(std::filesystem::path(path).parent_path());
    std::ofstream out(temporary,std::ios::binary|std::ios::trunc);
    if(!out)throw std::runtime_error("cannot write potential cache: "+temporary);
    const char magic[8]={'T','P','H','I','2','\0','\0','\0'};
    const uint64_t count=fields.phi.size();
    if(fields.phi.size()>UINT32_MAX/sizeof(float))throw std::runtime_error("potential cache exceeds single-hash size limit");
    const std::string payload_hash=sha256_hex(fields.phi.data(),fields.phi.size()*sizeof(float));
    out.write(magic,sizeof(magic));out.write(reinterpret_cast<const char*>(&fields.spec),sizeof(fields.spec));
    out.write(fields.cache_key.data(),64);out.write(reinterpret_cast<const char*>(&count),sizeof(count));out.write(payload_hash.data(),64);
    out.write(reinterpret_cast<const char*>(fields.phi.data()),std::streamsize(fields.phi.size()*sizeof(float)));
    out.flush();if(!out)throw std::runtime_error("potential cache write failed");out.close();
    if(std::rename(temporary.c_str(),path.c_str())!=0)throw std::runtime_error("cannot replace potential cache: "+path);
}

inline bool try_load_cache(TrainingPotentialFields& fields,const std::string& path) {
    if(path.empty()||!std::filesystem::exists(path))return false;
    std::ifstream in(path,std::ios::binary);if(!in)throw std::runtime_error("cannot open potential cache: "+path);
    char magic[8];TrainingPotentialGridSpec spec{};char key[64],payload_hash[64];uint64_t count=0;
    in.read(magic,sizeof(magic));in.read(reinterpret_cast<char*>(&spec),sizeof(spec));in.read(key,sizeof(key));in.read(reinterpret_cast<char*>(&count),sizeof(count));in.read(payload_hash,sizeof(payload_hash));
    const char expected_magic[8]={'T','P','H','I','2','\0','\0','\0'};
    if(!in||std::memcmp(magic,expected_magic,sizeof(magic))!=0||
       std::memcmp(&spec,&fields.spec,sizeof(spec))!=0||std::memcmp(key,fields.cache_key.data(),64)!=0||
       count!=uint64_t(fields.spec.level_count)*fields.spec.level_stride)return false;
    fields.phi.resize(size_t(count));in.read(reinterpret_cast<char*>(fields.phi.data()),std::streamsize(fields.phi.size()*sizeof(float)));
    if(!in)throw std::runtime_error("potential cache payload is truncated");
    char extra;if(in.read(&extra,1))throw std::runtime_error("potential cache has trailing bytes");
    if(fields.phi.size()>UINT32_MAX/sizeof(float)||sha256_hex(fields.phi.data(),fields.phi.size()*sizeof(float))!=std::string(payload_hash,64))
        throw std::runtime_error("potential cache payload checksum mismatch");
    for(float value:fields.phi)if(!std::isfinite(value)||value < -1.0f||value>0.0f)
        throw std::runtime_error("potential cache contains invalid Phi value");
    return true;
}

inline TrainingPotentialFields build_training_potential_fields(
        const std::vector<challenge_evaluation::Level>& train_levels,
        const std::string& bank_hash,const std::string& world_hash,
        const std::string& cache_path={}) {
    if(train_levels.empty())throw std::runtime_error("cannot build potential fields for an empty bank");
    TrainingPotentialFields fields;
    fields.spec=default_grid_spec(uint32_t(train_levels.size()));
    fields.cache_key=cache_key(bank_hash,world_hash);
    fields.bank_sha256=bank_hash;fields.world_sha256=world_hash;
    if(try_load_cache(fields,cache_path))return fields;
    fields.phi.resize(size_t(fields.spec.level_count)*fields.spec.level_stride);
    for(uint32_t level=0;level<train_levels.size();level++){
        const auto one=build_one_field(train_levels[level]);
        std::copy(one.begin(),one.end(),fields.phi.begin()+size_t(level)*fields.spec.level_stride);
    }
    if(!cache_path.empty())write_cache(fields,cache_path);
    return fields;
}

} // namespace training_potential
#endif // __METAL_VERSION__

#endif // TRAINING_POTENTIAL_HPP
