#pragma once

// CPU-only deployable interface for the fixed guided navigation actor.
// Input observations are prepared by the sensor/geometry frontend; this class
// runs only the actor MLP and maps its result to a bounded navigation command.
// Observation contract: [0,80) is the current 8x10 min-pooled range image and
// [80,160) is the previous image. Both use row-major FLU camera order and
// metres divided by 12. Features 160..180 hold goal unit xyz, distance/10,
// body velocity/4, body rates/4, body up, previous navigation command, sensor
// age in seconds, and clipped body-frame reference error. Features 181..183
// hold the geometry-memory guidance hint in atanh velocity space. The caller
// owns depth capture, pose history, pooling, and hint construction.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>

namespace nav_deployment {

constexpr uint32_t actor_observation_count=184;
constexpr uint32_t hidden_count=64;
constexpr uint32_t action_count=4;
constexpr uint32_t actor_weight_count=12104;
constexpr uint32_t metadata_bytes=64;
constexpr uint32_t trailer_bytes=16;
constexpr uint32_t body_frame_flu=0x00554c46u; // serialized little-endian bytes: "FLU\0"
constexpr char file_magic[8]={'N','A','V','P','O','L','1','\0'};
constexpr char provenance_magic[8]={'N','A','V','S','R','C','1','\0'};

struct Metadata {
    uint32_t version=1;
    uint32_t observation_count=actor_observation_count;
    uint32_t hidden_count_value=hidden_count;
    uint32_t action_count_value=action_count;
    uint32_t policy_mode=17;
    uint32_t weight_count=actor_weight_count;
    float max_speed_mps=1.5f;
    uint32_t sensor_rows=16;
    uint32_t sensor_columns=20;
    float range_max_m=12.0f;
    float navigation_period_s=0.05f;
    float native_period_s=0.01f;
    uint32_t memory_frames=8;
    uint32_t body_frame=body_frame_flu;
};
static_assert(sizeof(Metadata)==56,"NAV metadata fields must stay 56 bytes");

struct NavigationAction {
    float body_velocity_mps[3]={0,0,0};
    float yaw_rate_rps=0;
    // The trained observation stores applied tanh commands before speed projection.
    // Retain them for the next observation; they are not motor commands.
    float normalized_intent[4]={0,0,0,0};
};

inline uint64_t fnv1a64_file(const std::string& path,bool& ok) {
    std::ifstream f(path,std::ios::binary);
    if(!f){ok=false;return 0;}
    uint64_t hash=14695981039346656037ull;
    char buffer[16384];
    while(f) {
        f.read(buffer,sizeof(buffer));
        const std::streamsize count=f.gcount();
        for(std::streamsize i=0;i<count;i++) {
            hash^=static_cast<uint8_t>(buffer[i]);
            hash*=1099511628211ull;
        }
    }
    ok=f.eof();
    return hash;
}

class NavigationPolicy {
public:
    using Weights=std::array<float,actor_weight_count>;

    bool load(const std::string& path,std::string* error=nullptr) {
        loaded_=false;
        std::ifstream f(path,std::ios::binary);
        if(!f)return fail(error,"cannot open navigation policy: "+path);
        char magic[8];if(!read_bytes(f,magic,sizeof(magic))||std::memcmp(magic,file_magic,8)!=0)
            return fail(error,"navigation policy magic mismatch");
        Metadata m{};
        if(!read_u32(f,m.version)||!read_u32(f,m.observation_count)||!read_u32(f,m.hidden_count_value)||
           !read_u32(f,m.action_count_value)||!read_u32(f,m.policy_mode)||!read_u32(f,m.weight_count)||
           !read_float(f,m.max_speed_mps)||!read_u32(f,m.sensor_rows)||!read_u32(f,m.sensor_columns)||
           !read_float(f,m.range_max_m)||!read_float(f,m.navigation_period_s)||!read_float(f,m.native_period_s)||
           !read_u32(f,m.memory_frames)||!read_u32(f,m.body_frame))
            return fail(error,"truncated navigation policy metadata");
        if(!valid_metadata(m))return fail(error,"unsupported navigation policy metadata");
        for(float& value:weights_) {
            if(!read_float(f,value)||!std::isfinite(value))return fail(error,"invalid navigation actor weight");
        }
        char provenance_tag[8];uint64_t source_hash=0;
        if(!read_bytes(f,provenance_tag,sizeof(provenance_tag))||std::memcmp(provenance_tag,provenance_magic,8)!=0||
           !read_u64(f,source_hash))return fail(error,"navigation source provenance is missing");
        char extra;if(f.read(&extra,1))return fail(error,"navigation policy has trailing bytes");
        if(!f.eof())return fail(error,"navigation policy read failed");
        metadata_=m;source_hash_=source_hash;loaded_=true;
        if(error)error->clear();
        return true;
    }

    static bool write_file(const std::string& path,const float* actor_weights,size_t count,
                           const Metadata& metadata,const std::string& source_checkpoint,
                           std::string* error=nullptr) {
        if(!actor_weights||count!=actor_weight_count||!valid_metadata(metadata))
            return fail(error,"invalid NAV actor export dimensions or metadata");
        for(size_t i=0;i<count;i++)if(!std::isfinite(actor_weights[i]))
            return fail(error,"actor has a non-finite parameter");
        bool source_ok=false;const uint64_t source_hash=fnv1a64_file(source_checkpoint,source_ok);
        if(!source_ok)return fail(error,"cannot hash source checkpoint: "+source_checkpoint);
        const std::filesystem::path output(path);
        if(output.has_parent_path()) {
            std::error_code ec;std::filesystem::create_directories(output.parent_path(),ec);
            if(ec)return fail(error,"cannot create policy output directory: "+ec.message());
        }
        const std::string temp=path+".tmp";
        std::ofstream f(temp,std::ios::binary|std::ios::trunc);
        if(!f)return fail(error,"cannot create navigation policy: "+temp);
        f.write(file_magic,sizeof(file_magic));
        write_u32(f,metadata.version);write_u32(f,metadata.observation_count);
        write_u32(f,metadata.hidden_count_value);write_u32(f,metadata.action_count_value);
        write_u32(f,metadata.policy_mode);write_u32(f,metadata.weight_count);
        write_float(f,metadata.max_speed_mps);write_u32(f,metadata.sensor_rows);
        write_u32(f,metadata.sensor_columns);write_float(f,metadata.range_max_m);
        write_float(f,metadata.navigation_period_s);write_float(f,metadata.native_period_s);
        write_u32(f,metadata.memory_frames);write_u32(f,metadata.body_frame);
        for(size_t i=0;i<count;i++)write_float(f,actor_weights[i]);
        f.write(provenance_magic,sizeof(provenance_magic));write_u64(f,source_hash);
        f.flush();
        if(!f){f.close();std::remove(temp.c_str());return fail(error,"navigation policy write failed");}
        f.close();
        if(std::rename(temp.c_str(),path.c_str())!=0) {
            std::remove(temp.c_str());return fail(error,"cannot replace navigation policy: "+path);
        }
        if(error)error->clear();
        return true;
    }

    bool loaded() const { return loaded_; }
    const Metadata& metadata() const { return metadata_; }
    uint64_t source_checkpoint_hash() const { return source_hash_; }
    const Weights& weights() const { return weights_; }

    // Returns the guided pre-gate mean: the MLP xyz output plus the three
    // atanh-space prior features stored at observations[181..183].
    bool raw_mean(const float* observation,float mean[action_count]) const {
        if(mean)std::fill(mean,mean+action_count,0.0f);
        if(!loaded_||!observation||!mean)return false;
        for(uint32_t i=0;i<actor_observation_count;i++)
            if(!std::isfinite(observation[i]))return false;
        float hidden[hidden_count];
        constexpr size_t w1=0,b1=size_t(hidden_count)*actor_observation_count;
        constexpr size_t w2=b1+hidden_count,b2=w2+action_count*hidden_count;
        for(uint32_t h=0;h<hidden_count;h++) {
            float z=weights_[b1+h];const size_t row=size_t(h)*actor_observation_count;
            for(uint32_t i=0;i<actor_observation_count;i++)z+=weights_[w1+row+i]*observation[i];
            hidden[h]=std::tanh(z);
        }
        for(uint32_t a=0;a<action_count;a++) {
            float z=weights_[b2+a];const size_t row=w2+size_t(a)*hidden_count;
            for(uint32_t h=0;h<hidden_count;h++)z+=weights_[row+h]*hidden[h];
            mean[a]=z;
        }
        for(uint32_t j=0;j<3;j++)mean[j]+=observation[181+j];
        return true;
    }

    bool infer(const float* observation,NavigationAction& command) const {
        command=NavigationAction{};
        float mean[action_count];
        if(!raw_mean(observation,mean))return false;
        uint32_t overhead=0;
        for(uint32_t k=0;k<20;k++)if(observation[k]*metadata_.range_max_m>2.5f)overhead++;
        const float scale=0.25f+0.75f*(float(overhead)/20.0f);
        for(uint32_t j=0;j<3;j++)mean[j]=observation[181+j]+scale*(mean[j]-observation[181+j]);
        mean[3]*=scale;
        float velocity[3]={std::tanh(mean[0]),std::tanh(mean[1]),std::tanh(mean[2])};
        for(uint32_t j=0;j<3;j++)command.normalized_intent[j]=velocity[j];
        command.normalized_intent[3]=std::tanh(mean[3]);
        const float norm=std::sqrt(velocity[0]*velocity[0]+velocity[1]*velocity[1]+velocity[2]*velocity[2]);
        const float speed_scale=norm>1.0f?1.0f/norm:1.0f;
        for(uint32_t j=0;j<3;j++)command.body_velocity_mps[j]=velocity[j]*speed_scale*metadata_.max_speed_mps;
        command.yaw_rate_rps=0.5f*std::tanh(mean[3]);
        return true;
    }

private:
    Metadata metadata_{};
    Weights weights_{};
    uint64_t source_hash_=0;
    bool loaded_=false;

    static bool valid_metadata(const Metadata& m) {
        return m.version==1&&m.observation_count==actor_observation_count&&m.hidden_count_value==hidden_count&&
            m.action_count_value==action_count&&m.policy_mode==17&&m.weight_count==actor_weight_count&&
            std::isfinite(m.max_speed_mps)&&m.max_speed_mps>0&&m.sensor_rows==16&&m.sensor_columns==20&&
            std::fabs(m.range_max_m-12.0f)<1.0e-6f&&
            std::fabs(m.navigation_period_s-0.05f)<1.0e-6f&&
            std::fabs(m.native_period_s-0.01f)<1.0e-6f&&
            m.memory_frames==8&&m.body_frame==body_frame_flu;
    }
    static bool fail(std::string* error,const std::string& message) { if(error)*error=message;return false; }
    static bool read_bytes(std::istream& f,void* out,size_t n) {
        f.read(static_cast<char*>(out),std::streamsize(n));return f.good();
    }
    static bool read_u32(std::istream& f,uint32_t& out) {
        uint8_t b[4];if(!read_bytes(f,b,4))return false;
        out=uint32_t(b[0])|(uint32_t(b[1])<<8)|(uint32_t(b[2])<<16)|(uint32_t(b[3])<<24);return true;
    }
    static bool read_u64(std::istream& f,uint64_t& out) {
        uint8_t b[8];if(!read_bytes(f,b,8))return false;out=0;
        for(uint32_t i=0;i<8;i++)out|=uint64_t(b[i])<<(8*i);return true;
    }
    static bool read_float(std::istream& f,float& out) {
        uint32_t bits;if(!read_u32(f,bits))return false;std::memcpy(&out,&bits,sizeof(out));return true;
    }
    static void write_u32(std::ostream& f,uint32_t v) {
        const uint8_t b[4]={uint8_t(v),uint8_t(v>>8),uint8_t(v>>16),uint8_t(v>>24)};f.write(reinterpret_cast<const char*>(b),4);
    }
    static void write_u64(std::ostream& f,uint64_t v) {
        uint8_t b[8];for(uint32_t i=0;i<8;i++)b[i]=uint8_t(v>>(8*i));f.write(reinterpret_cast<const char*>(b),8);
    }
    static void write_float(std::ostream& f,float v) {
        uint32_t bits;std::memcpy(&bits,&v,sizeof(bits));write_u32(f,bits);
    }
};

static_assert(sizeof(float)==4&&std::numeric_limits<float>::is_iec559,
              "NAV deployment format requires IEEE-754 binary32");

} // namespace nav_deployment
