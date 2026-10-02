#pragma once

#include "physics.hpp"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <string>

// Independent ODE state measurements versus a free-running L2F reference.
// Both receive the same recorded motor actions. The reference is initialized
// once after hover warm-up; it is never corrected toward later measurements.
class PhysicsComparison {
public:
    PhysicsComparison(const RLPhysicsParams& parameters,
                      const std::filesystem::path& directory)
        : parameters_(parameters), directory_(directory) {}

    void observe(double time,const float position[3],const float quaternion[4],
                 const float velocity[3],const float rates[3],const float rotor_state[4]) {
        if(!started_) {
            for(int i=0;i<3;i++) {
                predicted_.position[i]=position[i];
                predicted_.linear_velocity[i]=velocity[i];
                predicted_.angular_velocity_body[i]=rates[i];
            }
            for(int i=0;i<4;i++) {
                predicted_.orientation_wxyz[i]=quaternion[i];
                predicted_.rpm[i]=rotor_state[i];
            }
            start_time_=time;started_=true;
            trace_.open(directory_/"physics-comparison.csv");
            if(!trace_)throw std::runtime_error("cannot write physics comparison");
            trace_<<"time_s,position_error_m,velocity_error_mps,attitude_error_rad,body_rate_error_rps";
            for(const char* prefix:{"measured_p","predicted_p","measured_v","predicted_v","measured_w","predicted_w"})
                for(int i=0;i<3;i++)trace_<<','<<prefix<<i;
            for(int i=0;i<4;i++)trace_<<",last_motor_action"<<i;
            trace_<<'\n'<<std::setprecision(10);
        }
        relative_time_=time-start_time_;
        const double errors[4]={norm_difference(position,predicted_.position),
            norm_difference(velocity,predicted_.linear_velocity),
            attitude_difference(quaternion,predicted_.orientation_wxyz),
            norm_difference(rates,predicted_.angular_velocity_body)};
        trace_<<relative_time_;
        for(int i=0;i<4;i++) {
            if(!std::isfinite(errors[i]))throw std::runtime_error("nonfinite physics comparison");
            squared_errors_[i]+=errors[i]*errors[i];
            max_errors_[i]=std::max(max_errors_[i],errors[i]);
            trace_<<','<<errors[i];
        }
        const float* channels[6]={position,predicted_.position,velocity,predicted_.linear_velocity,rates,predicted_.angular_velocity_body};
        for(const float* values:channels)
            for(int i=0;i<3;i++)trace_<<','<<values[i];
        for(float action:last_action_)trace_<<','<<action;
        trace_<<'\n';samples_++;
    }

    // Small paired motor pulses excite each torque axis without a navigation
    // policy or a ground-truth path. RAPTOR continues updating its hidden state.
    void apply_pulse(const std::string& profile,float hover_action,float action[4]) const {
        if(!started_ || relative_time_>=.35)return;
        float pulse=relative_time_>=.10 && relative_time_<.16?.05f:
                    relative_time_>=.16 && relative_time_<.22?-.05f:0.0f;
        float signs[4]={0,0,0,0};
        if(profile=="collective")for(float& sign:signs)sign=1;
        else if(profile=="roll") {signs[0]=-1;signs[1]=-1;signs[2]=1;signs[3]=1;}
        else if(profile=="pitch") {signs[0]=-1;signs[1]=1;signs[2]=1;signs[3]=-1;}
        else if(profile=="yaw") {signs[0]=-1;signs[1]=1;signs[2]=-1;signs[3]=1;}
        else if(profile!="hover")throw std::runtime_error("unknown physics comparison profile");
        for(int i=0;i<4;i++)action[i]=std::clamp(hover_action+pulse*signs[i],-1.0f,1.0f);
    }

    void advance(const float action[4]) {
        if(!started_)return;
        RLPhysicsState next;
        const float force[3]={0,0,0};
        rl_physics_step(predicted_,action,force,parameters_,next);
        predicted_=next;
        std::copy(action,action+4,last_action_);
    }

    void save(const std::string& profile,const std::string& motor_sampling) {
        if(!started_ || !samples_)throw std::runtime_error("physics comparison did not start");
        trace_.flush();if(!trace_)throw std::runtime_error("physics comparison trace failed");
        trace_.close();
        std::ofstream result(directory_/"physics-comparison.json");
        result<<std::setprecision(10)<<"{\"profile\":\""<<profile
              <<"\",\"motor_sampling\":\""<<motor_sampling
              <<"\",\"reference\":\"CPU L2F model; Metal numerical parity separately checked\""
              <<",\"plant\":\"Webots ODE and configured Propeller actuators\""
              <<",\"same_motor_commands\":true,\"warmup_s\":"<<start_time_
              <<",\"duration_s\":"<<relative_time_<<",\"samples\":"<<samples_;
        const char* names[4]={"position_m","velocity_mps","attitude_rad","body_rate_rps"};
        for(int i=0;i<4;i++)result<<",\"rms_"<<names[i]<<"\":"<<std::sqrt(squared_errors_[i]/samples_)
                                <<",\"max_"<<names[i]<<"\":"<<max_errors_[i];
        result<<",\"initial_rotor_state\":\"controller normalized actuator state, not measured RPM\""
              <<",\"audit_velocity\":\"instantaneous ODE Supervisor; never policy input\",\"hardware_validation\":false}\n";
        result.flush();if(!result)throw std::runtime_error("physics comparison summary failed");
    }

private:
    static double norm_difference(const float* a,const float* b) {
        double sum=0;for(int i=0;i<3;i++){const double delta=double(a[i])-b[i];sum+=delta*delta;}
        return std::sqrt(sum);
    }
    static double attitude_difference(const float* a,const float* b) {
        double dot=0,aa=0,bb=0;
        for(int i=0;i<4;i++){dot+=double(a[i])*b[i];aa+=double(a[i])*a[i];bb+=double(b[i])*b[i];}
        return 2*std::acos(std::clamp(std::fabs(dot)/std::sqrt(aa*bb),0.0,1.0));
    }
    RLPhysicsParams parameters_;
    RLPhysicsState predicted_{};
    std::filesystem::path directory_;
    std::ofstream trace_;
    double start_time_=0,relative_time_=0,squared_errors_[4]{},max_errors_[4]{};
    float last_action_[4]{};
    uint32_t samples_=0;
    bool started_=false;
};
