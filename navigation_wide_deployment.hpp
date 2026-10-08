#pragma once

// Width-aware guided actor for independent simulator validation. The critic is
// not serialized. Legacy NAVPOL1/NAVRAW2/NAVCAL3 readers remain unchanged.
#include "deployment.hpp"
#include <vector>

namespace nav_deployment {

constexpr char wide_file_magic[8]={'N','A','V','W','I','D','4','\0'};
constexpr char wide_provenance_magic[8]={'N','A','V','S','R','C','4','\0'};

class WideNavigationPolicy {
public:
    bool load(const std::string& path,std::string* error=nullptr) {
        loaded_=false;weights_.clear();hidden_.clear();
        std::ifstream file(path,std::ios::binary);
        if(!file)return fail(error,"cannot open wide policy");
        char magic[8];file.read(magic,8);
        if(!file||std::memcmp(magic,wide_file_magic,8))return fail(error,"wide policy magic mismatch");
        uint32_t fields[14];
        for(uint32_t& field:fields)if(!read_u32(file,field))return fail(error,"truncated wide metadata");
        Metadata m;std::memcpy(&m,fields,sizeof(m));
        uint32_t camera[8];
        for(uint32_t& field:camera)if(!read_u32(file,field))return fail(error,"truncated camera contract");
        SensorContract sensor;std::memcpy(&sensor,camera,sizeof(sensor));
        if(!valid_metadata(m)||!valid_sensor(sensor))return fail(error,"unsupported wide policy contract");
        weights_.resize(m.weight_count);hidden_.resize(m.hidden_count_value);
        for(float& weight:weights_) {
            uint32_t bits;if(!read_u32(file,bits))return fail(error,"truncated wide weights");
            std::memcpy(&weight,&bits,4);
            if(!std::isfinite(weight))return fail(error,"non-finite wide weight");
        }
        char trailer[8];file.read(trailer,8);
        uint32_t low,high;
        if(!file||std::memcmp(trailer,wide_provenance_magic,8)||
           !read_u32(file,low)||!read_u32(file,high))return fail(error,"missing wide provenance");
        char extra;if(file.read(&extra,1)||!file.eof())return fail(error,"wide policy has trailing bytes");
        metadata_=m;sensor_=sensor;source_hash_=uint64_t(low)|(uint64_t(high)<<32);loaded_=true;
        if(error)error->clear();return true;
    }

    bool loaded()const{return loaded_;}
    const Metadata& metadata()const{return metadata_;}
    const SensorContract& sensor_contract()const{return sensor_;}
    uint64_t source_checkpoint_hash()const{return source_hash_;}

    // Scratch is allocated once at load. One inference thread per instance.
    bool raw_mean(const float* observation,float mean[4])const {
        if(mean)std::fill(mean,mean+4,0.0f);
        if(!loaded_||!observation||!mean)return false;
        for(uint32_t i=0;i<184;i++)if(!std::isfinite(observation[i]))return false;
        const size_t width=metadata_.hidden_count_value;
        const size_t b1=width*184,w2=b1+width,b2=w2+4*width;
        for(size_t h=0;h<width;h++) {
            float value=weights_[b1+h];
            for(size_t i=0;i<184;i++)value+=weights_[h*184+i]*observation[i];
            hidden_[h]=std::tanh(value);
        }
        for(size_t a=0;a<4;a++) {
            float value=weights_[b2+a];
            for(size_t h=0;h<width;h++)value+=weights_[w2+a*width+h]*hidden_[h];
            mean[a]=value;
        }
        for(size_t a=0;a<3;a++)mean[a]+=observation[181+a];
        for(size_t a=0;a<4;a++)if(!std::isfinite(mean[a]))return false;
        return true;
    }

    bool infer(const float* observation,NavigationAction& command)const {
        command=NavigationAction{};float mean[4];
        if(!raw_mean(observation,mean))return false;
        // Keep the established mode17 operation order, including prior add/sub.
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
        command.yaw_rate_rps=0.5f*std::tanh(mean[3]);return true;
    }

private:
    Metadata metadata_{};SensorContract sensor_{};
    std::vector<float> weights_;mutable std::vector<float> hidden_;
    uint64_t source_hash_=0;bool loaded_=false;
    static bool valid_metadata(const Metadata& m) {
        return m.version==4&&m.observation_count==184&&m.hidden_count_value>=64&&m.hidden_count_value<=8192&&
            m.action_count_value==4&&m.policy_mode==17&&m.weight_count==189*m.hidden_count_value+8&&
            std::isfinite(m.max_speed_mps)&&m.max_speed_mps>0&&m.max_speed_mps<=10&&
            m.sensor_rows==16&&m.sensor_columns==20&&std::fabs(m.range_max_m-12)<1e-6f&&
            std::fabs(m.navigation_period_s-.05f)<1e-6f&&std::fabs(m.native_period_s-.01f)<1e-6f&&
            m.memory_frames==8&&m.body_frame==body_frame_flu;
    }
    static bool valid_sensor(const SensorContract& s) {
        const bool legacy=s.profile_id==1&&std::fabs(s.tan_h-1)<1e-6f&&
            std::fabs(s.tan_v-.75f)<1e-6f&&std::fabs(s.mount_x_m)<1e-6f;
        return (legacy||valid_native_contract(s))&&std::fabs(s.min_range_m-.03f)<1e-6f&&
            std::fabs(s.max_range_m-12)<1e-6f&&s.noise_m==0&&s.reserved==0;
    }
    static bool read_u32(std::istream& file,uint32_t& out) {
        uint8_t bytes[4];file.read((char*)bytes,4);if(!file)return false;
        out=uint32_t(bytes[0])|(uint32_t(bytes[1])<<8)|(uint32_t(bytes[2])<<16)|(uint32_t(bytes[3])<<24);return true;
    }
    static bool fail(std::string* error,const char* message) {if(error)*error=message;return false;}
};
}
