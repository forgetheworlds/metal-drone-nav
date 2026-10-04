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
constexpr uint32_t raw_depth_actor_observation_count=824;
constexpr uint32_t hidden_count=64;
constexpr uint32_t action_count=4;
constexpr uint32_t actor_weight_count=12104;
constexpr uint32_t raw_depth_actor_weight_count=53064;
constexpr uint32_t raw_current_range_offset=181;
constexpr uint32_t raw_previous_range_offset=501;
constexpr uint32_t raw_geometry_hint_offset=821;
static_assert(raw_current_range_offset+320==raw_previous_range_offset&&
              raw_previous_range_offset+320==raw_geometry_hint_offset&&
              raw_geometry_hint_offset+3==raw_depth_actor_observation_count,
              "raw-depth version-2 feature offsets changed");
constexpr uint32_t metadata_bytes=64;
constexpr uint32_t trailer_bytes=16;
constexpr uint32_t body_frame_flu=0x00554c46u; // serialized little-endian bytes: "FLU\0"
constexpr char file_magic[8]={'N','A','V','P','O','L','1','\0'};
constexpr char provenance_magic[8]={'N','A','V','S','R','C','1','\0'};
constexpr char raw_depth_file_magic[8]={'N','A','V','R','A','W','2','\0'};
constexpr char raw_depth_provenance_magic[8]={'N','A','V','S','R','C','2','\0'};
// Calibrated native-profile raw-depth actor. Separate magic and an explicit
// sensor contract so a legacy consumer cannot load it and a calibrated
// consumer cannot silently accept the legacy 0.75/body-origin contract.
constexpr char calibrated_file_magic[8]={'N','A','V','C','A','L','3','\0'};
constexpr char calibrated_provenance_magic[8]={'N','A','V','S','C','C','3','\0'};

// Declared camera contract of a calibrated actor. Values are the measured
// native RangeFinder geometry from sensor_profile.hpp.
struct SensorContract {
    uint32_t profile_id=0;      // nav_sensor profile id: 2 = native
    float tan_h=0.0f;
    float tan_v=0.0f;
    float mount_x_m=0.0f;
    float min_range_m=0.0f;
    float max_range_m=0.0f;
    float noise_m=0.0f;
    uint32_t reserved=0;
};
static_assert(sizeof(SensorContract)==32,"sensor contract layout");

inline bool valid_native_contract(const SensorContract& c) {
    return c.profile_id==2u && std::fabs(c.tan_h-1.0f)<1e-6f && std::fabs(c.tan_v-0.8f)<1e-6f &&
        std::fabs(c.mount_x_m-0.08f)<1e-6f && std::fabs(c.min_range_m-0.03f)<1e-6f &&
        std::fabs(c.max_range_m-12.0f)<1e-6f && std::fabs(c.noise_m)<1e-6f && c.reserved==0;
}

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

// Shared Webots/native-v2 observation packer. Input ranges are metres; the
// actor stores all range features in units of the 12m sensor maximum. Context
// values are already normalized and retain v1 indices 160..180.
inline void pack_raw_depth_observation(const float current_pooled_m[80],
                                       const float previous_pooled_m[80],
                                       const float context[21],
                                       const float current_raw_m[320],
                                       const float previous_raw_m[320],
                                       const float geometry_hint[3],
                                       float observation[raw_depth_actor_observation_count]) {
    std::fill(observation,observation+raw_depth_actor_observation_count,0.0f);
    for(uint32_t i=0;i<80;i++) {
        observation[i]=current_pooled_m[i]/12.0f;
        observation[80+i]=previous_pooled_m[i]/12.0f;
    }
    std::copy(context,context+21,observation+160);
    for(uint32_t i=0;i<320;i++) {
        observation[raw_current_range_offset+i]=current_raw_m[i]/12.0f;
        observation[raw_previous_range_offset+i]=previous_raw_m[i]/12.0f;
    }
    std::copy(geometry_hint,geometry_hint+3,observation+raw_geometry_hint_offset);
}

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

// Version-2 raw-depth policy. Its first 181 inputs are identical to v1's
// pooled policy, raw current/previous ranges occupy 181..820, and geometry
// guidance remains in the final three inputs. It has a separate file magic so
// a v1 consumer cannot silently load this larger actor.
class RawDepthNavigationPolicy {
public:
    using Weights=std::array<float,raw_depth_actor_weight_count>;

    bool load(const std::string& path,std::string* error=nullptr) {
        loaded_=false;has_contract_=false;
        std::ifstream f(path,std::ios::binary);
        if(!f)return fail(error,"cannot open raw-depth navigation policy: "+path);
        char magic[8];if(!read_bytes(f,magic,sizeof(magic)))
            return fail(error,"raw-depth policy magic read failed");
        const bool calibrated=std::memcmp(magic,calibrated_file_magic,8)==0;
        if(!calibrated&&std::memcmp(magic,raw_depth_file_magic,8)!=0)
            return fail(error,"raw-depth policy magic mismatch; expected policy version2 or3");
        Metadata m{};
        if(!read_metadata(f,m))return fail(error,"truncated raw-depth policy metadata");
        if(calibrated){if(!valid_calibrated_metadata(m))return fail(error,"unsupported calibrated policy metadata");}
        else if(!valid_metadata(m))return fail(error,"unsupported raw-depth navigation policy metadata");
        SensorContract contract{};
        if(calibrated) {
            if(!read_contract(f,contract))return fail(error,"truncated calibrated sensor contract");
            if(!valid_native_contract(contract))return fail(error,"calibrated policy does not declare the native sensor profile");
        }
        for(float& value:weights_) {
            if(!read_float(f,value)||!std::isfinite(value))return fail(error,"invalid raw-depth actor weight");
        }
        char provenance[8];uint64_t source_hash=0;
        const char* expected_provenance=calibrated?calibrated_provenance_magic:raw_depth_provenance_magic;
        if(!read_bytes(f,provenance,sizeof(provenance))||std::memcmp(provenance,expected_provenance,8)!=0||
           !read_u64(f,source_hash))return fail(error,"raw-depth source provenance is missing");
        char extra;if(f.read(&extra,1))return fail(error,"raw-depth policy has trailing bytes");
        if(!f.eof())return fail(error,"raw-depth policy read failed");
        metadata_=m;contract_=contract;has_contract_=calibrated;source_hash_=source_hash;loaded_=true;
        if(error)error->clear();return true;
    }

    // Write the calibrated (native-profile) variant. The legacy v2 writer is
    // unchanged, so old raw-depth assets keep loading through the same class.
    static bool write_calibrated_file(const std::string& path,const float* actor_weights,size_t count,
                                      const Metadata& metadata,const SensorContract& contract,
                                      const std::string& source_checkpoint,std::string* error=nullptr) {
        if(!actor_weights||count!=raw_depth_actor_weight_count||!valid_calibrated_metadata(metadata)||
           !valid_native_contract(contract))
            return fail(error,"invalid calibrated NAV actor dimensions, metadata, or sensor contract");
        for(size_t i=0;i<count;i++)if(!std::isfinite(actor_weights[i]))
            return fail(error,"calibrated actor has a non-finite parameter");
        bool source_ok=false;const uint64_t source_hash=fnv1a64_file(source_checkpoint,source_ok);
        if(!source_ok)return fail(error,"cannot hash calibrated source checkpoint: "+source_checkpoint);
        const std::filesystem::path output(path);
        if(output.has_parent_path()) {
            std::error_code ec;std::filesystem::create_directories(output.parent_path(),ec);
            if(ec)return fail(error,"cannot create calibrated policy directory: "+ec.message());
        }
        const std::string temp=path+".tmp";
        std::ofstream f(temp,std::ios::binary|std::ios::trunc);
        if(!f)return fail(error,"cannot create calibrated navigation policy: "+temp);
        f.write(calibrated_file_magic,sizeof(calibrated_file_magic));write_metadata(f,metadata);
        write_contract(f,contract);
        for(size_t i=0;i<count;i++)write_float(f,actor_weights[i]);
        f.write(calibrated_provenance_magic,sizeof(calibrated_provenance_magic));write_u64(f,source_hash);
        f.flush();
        if(!f){f.close();std::remove(temp.c_str());return fail(error,"calibrated policy write failed");}
        f.close();
        if(std::rename(temp.c_str(),path.c_str())!=0) {
            std::remove(temp.c_str());return fail(error,"cannot replace calibrated policy: "+path);
        }
        if(error)error->clear();return true;
    }

    static bool write_file(const std::string& path,const float* actor_weights,size_t count,
                           const Metadata& metadata,const std::string& source_checkpoint,
                           std::string* error=nullptr) {
        if(!actor_weights||count!=raw_depth_actor_weight_count||!valid_metadata(metadata))
            return fail(error,"invalid raw-depth NAV actor dimensions or metadata");
        for(size_t i=0;i<count;i++)if(!std::isfinite(actor_weights[i]))
            return fail(error,"raw-depth actor has a non-finite parameter");
        bool source_ok=false;const uint64_t source_hash=fnv1a64_file(source_checkpoint,source_ok);
        if(!source_ok)return fail(error,"cannot hash raw-depth source checkpoint: "+source_checkpoint);
        const std::filesystem::path output(path);
        if(output.has_parent_path()) {
            std::error_code ec;std::filesystem::create_directories(output.parent_path(),ec);
            if(ec)return fail(error,"cannot create raw-depth policy directory: "+ec.message());
        }
        const std::string temp=path+".tmp";
        std::ofstream f(temp,std::ios::binary|std::ios::trunc);
        if(!f)return fail(error,"cannot create raw-depth navigation policy: "+temp);
        f.write(raw_depth_file_magic,sizeof(raw_depth_file_magic));write_metadata(f,metadata);
        for(size_t i=0;i<count;i++)write_float(f,actor_weights[i]);
        f.write(raw_depth_provenance_magic,sizeof(raw_depth_provenance_magic));write_u64(f,source_hash);
        f.flush();
        if(!f){f.close();std::remove(temp.c_str());return fail(error,"raw-depth policy write failed");}
        f.close();
        if(std::rename(temp.c_str(),path.c_str())!=0) {
            std::remove(temp.c_str());return fail(error,"cannot replace raw-depth policy: "+path);
        }
        if(error)error->clear();return true;
    }

    bool loaded()const{return loaded_;}
    const Metadata& metadata()const{return metadata_;}
    // True only for the calibrated variant; the contract is the declared
    // native sensor profile the caller must already be running.
    bool has_sensor_contract()const{return has_contract_;}
    const SensorContract& sensor_contract()const{return contract_;}
    uint64_t source_checkpoint_hash()const{return source_hash_;}
    const Weights& weights()const{return weights_;}

    bool raw_mean(const float* observation,float mean[action_count])const{
        if(mean)std::fill(mean,mean+action_count,0.0f);
        if(!loaded_||!observation||!mean)return false;
        for(uint32_t i=0;i<raw_depth_actor_observation_count;i++)
            if(!std::isfinite(observation[i]))return false;
        constexpr size_t b1=size_t(hidden_count)*raw_depth_actor_observation_count;
        constexpr size_t w2=b1+hidden_count,b2=w2+action_count*hidden_count;
        float hidden[hidden_count];
        for(uint32_t h=0;h<hidden_count;h++) {
            float z=weights_[b1+h];const size_t row=size_t(h)*raw_depth_actor_observation_count;
            for(uint32_t i=0;i<raw_depth_actor_observation_count;i++)z+=weights_[row+i]*observation[i];
            hidden[h]=std::tanh(z);
        }
        for(uint32_t a=0;a<action_count;a++) {
            float z=weights_[b2+a];const size_t row=w2+size_t(a)*hidden_count;
            for(uint32_t h=0;h<hidden_count;h++)z+=weights_[row+h]*hidden[h];
            if(a<3)z+=observation[raw_geometry_hint_offset+a];
            mean[a]=z;
        }
        return true;
    }

    bool infer(const float* observation,NavigationAction& command)const {
        command=NavigationAction{};float mean[action_count];
        if(!raw_mean(observation,mean))return false;
        uint32_t overhead=0;
        for(uint32_t k=0;k<20;k++)if(observation[k]*metadata_.range_max_m>2.5f)overhead++;
        const float gate=.25f+.75f*float(overhead)/20.0f;
        for(uint32_t j=0;j<3;j++)mean[j]=observation[raw_geometry_hint_offset+j]+gate*(mean[j]-observation[raw_geometry_hint_offset+j]);
        mean[3]*=gate;
        float velocity[3]={std::tanh(mean[0]),std::tanh(mean[1]),std::tanh(mean[2])};
        for(uint32_t j=0;j<3;j++)command.normalized_intent[j]=velocity[j];
        command.normalized_intent[3]=std::tanh(mean[3]);
        const float norm=std::sqrt(velocity[0]*velocity[0]+velocity[1]*velocity[1]+velocity[2]*velocity[2]);
        const float scale=norm>1.0f?1.0f/norm:1.0f;
        for(uint32_t j=0;j<3;j++)command.body_velocity_mps[j]=velocity[j]*scale*metadata_.max_speed_mps;
        command.yaw_rate_rps=.5f*std::tanh(mean[3]);return true;
    }

private:
    Metadata metadata_{};SensorContract contract_{};Weights weights_{};
    uint64_t source_hash_=0;bool loaded_=false;bool has_contract_=false;

    static bool valid_metadata(const Metadata& m) {
        return m.version==2&&m.observation_count==raw_depth_actor_observation_count&&
            m.hidden_count_value==hidden_count&&m.action_count_value==action_count&&m.policy_mode==17&&
            m.weight_count==raw_depth_actor_weight_count&&std::isfinite(m.max_speed_mps)&&m.max_speed_mps>0&&
            m.sensor_rows==16&&m.sensor_columns==20&&std::fabs(m.range_max_m-12.0f)<1e-6f&&
            std::fabs(m.navigation_period_s-.05f)<1e-6f&&std::fabs(m.native_period_s-.01f)<1e-6f&&
            m.memory_frames==8&&m.body_frame==body_frame_flu;
    }
    static bool valid_calibrated_metadata(const Metadata& m) {
        Metadata v=m;v.version=2;return m.version==3 && valid_metadata(v);
    }
    static bool read_contract(std::istream& f,SensorContract& c) {
        return read_u32(f,c.profile_id)&&read_float(f,c.tan_h)&&read_float(f,c.tan_v)&&
            read_float(f,c.mount_x_m)&&read_float(f,c.min_range_m)&&read_float(f,c.max_range_m)&&
            read_float(f,c.noise_m)&&read_u32(f,c.reserved);
    }
    static void write_contract(std::ostream& f,const SensorContract& c) {
        write_u32(f,c.profile_id);write_float(f,c.tan_h);write_float(f,c.tan_v);
        write_float(f,c.mount_x_m);write_float(f,c.min_range_m);write_float(f,c.max_range_m);
        write_float(f,c.noise_m);write_u32(f,c.reserved);
    }
    static bool fail(std::string* error,const std::string& message){if(error)*error=message;return false;}
    static bool read_metadata(std::istream& f,Metadata& m) {
        return read_u32(f,m.version)&&read_u32(f,m.observation_count)&&read_u32(f,m.hidden_count_value)&&
            read_u32(f,m.action_count_value)&&read_u32(f,m.policy_mode)&&read_u32(f,m.weight_count)&&
            read_float(f,m.max_speed_mps)&&read_u32(f,m.sensor_rows)&&read_u32(f,m.sensor_columns)&&
            read_float(f,m.range_max_m)&&read_float(f,m.navigation_period_s)&&read_float(f,m.native_period_s)&&
            read_u32(f,m.memory_frames)&&read_u32(f,m.body_frame);
    }
    static void write_metadata(std::ostream& f,const Metadata& m) {
        write_u32(f,m.version);write_u32(f,m.observation_count);write_u32(f,m.hidden_count_value);
        write_u32(f,m.action_count_value);write_u32(f,m.policy_mode);write_u32(f,m.weight_count);
        write_float(f,m.max_speed_mps);write_u32(f,m.sensor_rows);write_u32(f,m.sensor_columns);
        write_float(f,m.range_max_m);write_float(f,m.navigation_period_s);write_float(f,m.native_period_s);
        write_u32(f,m.memory_frames);write_u32(f,m.body_frame);
    }
    static bool read_bytes(std::istream& f,void* out,size_t n){f.read(static_cast<char*>(out),std::streamsize(n));return f.good();}
    static bool read_u32(std::istream& f,uint32_t& out){
        uint8_t b[4];if(!read_bytes(f,b,4))return false;
        out=uint32_t(b[0])|(uint32_t(b[1])<<8)|(uint32_t(b[2])<<16)|(uint32_t(b[3])<<24);return true;
    }
    static bool read_u64(std::istream& f,uint64_t& out){
        uint8_t b[8];if(!read_bytes(f,b,8))return false;out=0;
        for(uint32_t i=0;i<8;i++)out|=uint64_t(b[i])<<(8*i);return true;
    }
    static bool read_float(std::istream& f,float& out){uint32_t bits;if(!read_u32(f,bits))return false;std::memcpy(&out,&bits,4);return true;}
    static void write_u32(std::ostream& f,uint32_t v){
        const uint8_t b[4]={uint8_t(v),uint8_t(v>>8),uint8_t(v>>16),uint8_t(v>>24)};f.write(reinterpret_cast<const char*>(b),4);
    }
    static void write_u64(std::ostream& f,uint64_t v){uint8_t b[8];for(uint32_t i=0;i<8;i++)b[i]=uint8_t(v>>(8*i));f.write(reinterpret_cast<const char*>(b),8);}
    static void write_float(std::ostream& f,float value){uint32_t bits;std::memcpy(&bits,&value,4);write_u32(f,bits);}
};

// Copy a v1 guided actor into a v2 raw-depth actor. Its pooled/context inputs
// retain their indices; the three prior skip inputs move to the final slots.
inline bool lift_guided_actor_to_raw_depth(const float* old_weights,size_t old_count,
                                           float* raw_weights,size_t raw_count,
                                           std::string* error=nullptr) {
    constexpr size_t old_count_expected=actor_weight_count;
    if(!old_weights||!raw_weights||old_count!=old_count_expected||raw_count!=raw_depth_actor_weight_count) {
        if(error)*error="guided actor lift dimensions mismatch";return false;
    }
    if(!std::all_of(old_weights,old_weights+old_count,[](float x){return std::isfinite(x);})) {
        if(error)*error="guided actor lift source contains non-finite weights";return false;
    }
    std::fill(raw_weights,raw_weights+raw_count,0.0f);
    constexpr size_t old_w1=0,old_b1=hidden_count*actor_observation_count;
    constexpr size_t old_w2=old_b1+hidden_count,old_b2=old_w2+action_count*hidden_count;
    constexpr size_t old_logstd=old_b2+action_count;
    constexpr size_t new_w1=0,new_b1=hidden_count*raw_depth_actor_observation_count;
    constexpr size_t new_w2=new_b1+hidden_count,new_b2=new_w2+action_count*hidden_count;
    constexpr size_t new_logstd=new_b2+action_count;
    for(size_t h=0;h<hidden_count;h++) {
        const size_t old_row=old_w1+h*actor_observation_count;
        const size_t new_row=new_w1+h*raw_depth_actor_observation_count;
        std::copy(old_weights+old_row,old_weights+old_row+181,raw_weights+new_row);
        for(size_t hint=0;hint<3;hint++)
            raw_weights[new_row+raw_geometry_hint_offset+hint]=old_weights[old_row+181+hint];
        raw_weights[new_b1+h]=old_weights[old_b1+h];
    }
    std::copy(old_weights+old_w2,old_weights+old_w2+action_count*hidden_count,raw_weights+new_w2);
    std::copy(old_weights+old_b2,old_weights+old_b2+action_count,raw_weights+new_b2);
    std::copy(old_weights+old_logstd,old_weights+old_logstd+action_count,raw_weights+new_logstd);
    if(error)error->clear();return true;
}

static_assert(sizeof(float)==4&&std::numeric_limits<float>::is_iec559,
              "NAV deployment format requires IEEE-754 binary32");

} // namespace nav_deployment
