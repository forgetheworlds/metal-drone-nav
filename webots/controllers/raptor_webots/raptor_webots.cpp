#include <webots/robot.h>
#include <webots/motor.h>
#include <webots/range_finder.h>
#include <webots/gps.h>
#include <webots/inertial_unit.h>
#include <webots/gyro.h>
#include <webots/supervisor.h>
#include <webots/contact_point.h>

#include "deployment.hpp"
#include "guidance.hpp"
#include "physics.hpp"
#include "raptor.hpp"
#include "physics_comparison.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace {
constexpr int kNavEvery = 5;             // 20 Hz depth/navigation over 100 Hz RAPTOR.
constexpr float kSensorDt = 0.05f;
constexpr float kRangeMax = 12.0f;
constexpr float kRotorThrustK = 4.0e-5f; // Webots T = k * |omega| * omega.
constexpr int kMotorSign[4] = {-1, 1, -1, 1};
const char* kMotorNames[4] = {"motor_fr", "motor_br", "motor_bl", "motor_fl"};

struct Config {
    std::string phase="raptor-hover";
    std::string physics_profile="hover";
    std::string motor_sampling="end";
    std::string goal_objective="entry";
    std::string movie_file;
    std::string view_snapshot_file;
    std::string diagnostic_prefix;
    std::string raw_depth_shadow_path;
    float diagnostic_nav_scale=1.0f;
    std::string policy="../assets/navigation.bin";
    uint32_t policy_version=1;
    bool raw_depth_shadow=false;
    uint32_t seed=1;
    uint32_t max_steps=800;
    float speed=1.5f;
    float distance=4.0f;
    float range_plane_x=2.0f;
    bool sensor_audit=false;
    bool capture_trajectory=false;
    bool velocity_reference_is_world=false;
    float goal[3]={4,0,1.5f};
    float velocity[3]={0,0,0};
};

bool parse_float(const std::string& text,float& out) {
    char* end=nullptr;const float value=std::strtof(text.c_str(),&end);
    if(end==text.c_str()||*end!='\0'||!std::isfinite(value))return false;
    out=value;return true;
}
Config parse_config(const char* custom) {
    Config c;
    if(!custom)return c;
    std::string data(custom);size_t begin=0;
    while(begin<data.size()) {
        size_t end=data.find(';',begin);if(end==std::string::npos)end=data.size();
        const std::string item=data.substr(begin,end-begin);const size_t eq=item.find('=');
        if(eq!=std::string::npos) {
            const std::string key=item.substr(0,eq),value=item.substr(eq+1);
            if(key=="phase")c.phase=value;
            else if(key=="profile")c.physics_profile=value;
            else if(key=="motor_sampling")c.motor_sampling=value;
            else if(key=="goal_objective")c.goal_objective=value;
            else if(key=="movie_file")c.movie_file=value;
            else if(key=="view_snapshot_file")c.view_snapshot_file=value;
            else if(key=="diagnostic_prefix")c.diagnostic_prefix=value;
            else if(key=="raw_depth_shadow_path")c.raw_depth_shadow_path=value;
            else if(key=="diagnostic_nav_scale")parse_float(value,c.diagnostic_nav_scale);
            else if(key=="policy")c.policy=value;
            else if(key=="policy_version")c.policy_version=uint32_t(std::strtoul(value.c_str(),nullptr,10));
            else if(key=="raw_depth_shadow")c.raw_depth_shadow=(value=="1"||value=="true");
            else if(key=="seed")c.seed=uint32_t(std::strtoul(value.c_str(),nullptr,10));
            else if(key=="max_steps")c.max_steps=uint32_t(std::strtoul(value.c_str(),nullptr,10));
            else if(key=="speed")parse_float(value,c.speed);
            else if(key=="distance")parse_float(value,c.distance);
            else if(key=="range_plane_x")parse_float(value,c.range_plane_x);
            else if(key=="sensor_audit")c.sensor_audit=(value=="1"||value=="true");
            else if(key=="capture_trajectory")c.capture_trajectory=(value=="1"||value=="true");
            else if(key=="velocity_frame")c.velocity_reference_is_world=(value=="world");
            else if(key=="goal") {
                size_t at=0;
                for(int j=0;j<3;j++){
                    const size_t comma=value.find(',',at);
                    const std::string component=value.substr(at,comma==std::string::npos?comma:comma-at);
                    parse_float(component,c.goal[j]);
                    if(comma==std::string::npos)break;
                    at=comma+1;
                }
            }
            else if(key=="velocity") {
                size_t at=0;for(int j=0;j<3;j++){size_t comma=value.find(',',at);std::string v=value.substr(at,comma==std::string::npos?comma:comma-at);parse_float(v,c.velocity[j]);if(comma==std::string::npos)break;at=comma+1;}
            }
        }
        begin=end+1;
    }
    return c;
}

float clampf(float x,float lo,float hi){return std::fmax(lo,std::fmin(hi,x));}
void normalize_ranges(const float* raw,int width,int height,float hfov,float* ranges) {
    const float tan_h=std::tan(0.5f*hfov);
    const float tan_v_native=tan_h*float(height)/float(width);
    const float tan_v_model=0.75f;
    for(int y=0;y<height;y++)for(int x=0;x<width;x++) {
        const float u=(2.0f*(float(x)+0.5f)/float(width)-1.0f)*tan_h;
        const float v=(1.0f-2.0f*(float(y)+0.5f)/float(height))*tan_v_model;
        const float source_y=(1.0f-v/tan_v_native)*0.5f*float(height)-0.5f;
        const int y0=std::max(0,std::min(height-1,int(std::floor(source_y))));
        const int y1=std::max(0,std::min(height-1,int(std::ceil(source_y))));
        const float axial=std::fmin(raw[y0*width+x],raw[y1*width+x]);
        const float range=std::isfinite(axial)?axial*std::sqrt(1.0f+u*u+v*v):kRangeMax;
        ranges[y*width+x]=clampf(range,0.03f,kRangeMax);
    }
}
void pool_ranges(const float* ranges,float pooled[80]) {
    for(int r=0;r<8;r++)for(int c=0;c<10;c++) {
        const int i=(2*r)*20+2*c;
        pooled[r*10+c]=std::fmin(std::fmin(ranges[i],ranges[i+1]),std::fmin(ranges[i+20],ranges[i+21]));
    }
}
void quat_to_yaw(float yaw,float q[4]) {q[0]=std::cos(yaw*0.5f);q[1]=0;q[2]=0;q[3]=std::sin(yaw*0.5f);}
void rotate_body_to_world(const float r[9],const float b[3],float w[3]) {
    for(int i=0;i<3;i++)w[i]=r[3*i]*b[0]+r[3*i+1]*b[1]+r[3*i+2]*b[2];
}
void rotate_world_to_body(const float r[9],const float w[3],float b[3]) {
    for(int i=0;i<3;i++)b[i]=r[i]*w[0]+r[3+i]*w[1]+r[6+i]*w[2];
}
float vector_norm3(const float v[3]) {return std::sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]);}
void update_recording_view(WbFieldRef position_field,WbFieldRef orientation_field,const float target[3]) {
    const float offset[3]={-1.2f,-1.6f,0.8f};
    float camera[3]={target[0]+offset[0],target[1]+offset[1],target[2]+offset[2]};
    float forward[3]={target[0]-camera[0],target[1]-camera[1],target[2]-camera[2]};
    const float forward_norm=vector_norm3(forward);
    for(float& value:forward)value/=forward_norm;
    float left[3]={-forward[1],forward[0],0.0f};
    const float left_norm=vector_norm3(left);
    for(float& value:left)value/=left_norm;
    float up[3]={forward[1]*left[2]-forward[2]*left[1],
                 forward[2]*left[0]-forward[0]*left[2],
                 forward[0]*left[1]-forward[1]*left[0]};
    // R2025a Viewpoint axes are local +X forward, +Y left, +Z up.
    const float m00=forward[0],m01=left[0],m02=up[0];
    const float m10=forward[1],m11=left[1],m12=up[1];
    const float m20=forward[2],m21=left[2],m22=up[2];
    float qw,qx,qy,qz;const float trace=m00+m11+m22;
    if(trace>0.0f) {
        const float s=std::sqrt(trace+1.0f)*2.0f;
        qw=0.25f*s;qx=(m21-m12)/s;qy=(m02-m20)/s;qz=(m10-m01)/s;
    } else if(m00>m11&&m00>m22) {
        const float s=std::sqrt(1.0f+m00-m11-m22)*2.0f;
        qw=(m21-m12)/s;qx=0.25f*s;qy=(m01+m10)/s;qz=(m02+m20)/s;
    } else if(m11>m22) {
        const float s=std::sqrt(1.0f+m11-m00-m22)*2.0f;
        qw=(m02-m20)/s;qx=(m01+m10)/s;qy=0.25f*s;qz=(m12+m21)/s;
    } else {
        const float s=std::sqrt(1.0f+m22-m00-m11)*2.0f;
        qw=(m10-m01)/s;qx=(m02+m20)/s;qy=(m12+m21)/s;qz=0.25f*s;
    }
    const float qnorm=std::sqrt(qw*qw+qx*qx+qy*qy+qz*qz);
    qw/=qnorm;qx/=qnorm;qy/=qnorm;qz/=qnorm;
    if(qw<0.0f){qw=-qw;qx=-qx;qy=-qy;qz=-qz;}
    const float angle=2.0f*std::acos(std::clamp(qw,-1.0f,1.0f));
    const float sin_half=std::sqrt(std::max(0.0f,1.0f-qw*qw));
    double position[3]={camera[0],camera[1],camera[2]};
    double rotation[4]={sin_half>1e-6f?qx/sin_half:0.0,
                        sin_half>1e-6f?qy/sin_half:0.0,
                        sin_half>1e-6f?qz/sin_half:1.0,angle};
    wb_supervisor_field_set_sf_vec3f(position_field,position);
    wb_supervisor_field_set_sf_rotation(orientation_field,rotation);
}
std::unordered_map<std::string,std::string> data_fields(const char* raw) {
    std::unordered_map<std::string,std::string> fields;
    if(!raw)return fields;
    std::string s(raw);size_t p=0;
    while(p<s.size()){size_t q=s.find(';',p);if(q==std::string::npos)q=s.size();size_t e=s.find('=',p);if(e<q)fields[s.substr(p,e-p)]=s.substr(e+1,q-e-1);p=q+1;}
    return fields;
}
}

int main(int argc,char** argv) {
    wb_robot_init();
    const int physics_step_ms=int(std::round(wb_robot_get_basic_time_step()));
    if(physics_step_ms<1 || 10%physics_step_ms!=0) {
        std::fprintf(stderr,"physics step must divide the100Hz controller period\n");
        wb_robot_cleanup();return 2;
    }
    const int step_ms=10; // RAPTOR remains100Hz with finer ODE integration.
    const Config config=parse_config(wb_robot_get_custom_data());
    if(config.policy_version!=1&&config.policy_version!=2) {
        std::fprintf(stderr,"policy_version must be1 or2\n");wb_robot_cleanup();return 2;
    }
    if(config.goal_objective!="entry"&&config.goal_objective!="hold") {
        std::fprintf(stderr,"goal_objective must be entry or hold\n");wb_robot_cleanup();return 2;
    }
    if(config.diagnostic_nav_scale<=0.0f||config.diagnostic_nav_scale>1.0f) {
        std::fprintf(stderr,"diagnostic_nav_scale must be in (0,1]\n");wb_robot_cleanup();return 2;
    }
    if(!config.movie_file.empty()&&std::filesystem::path(config.movie_file).extension()!=".mp4") {
        std::fprintf(stderr,"movie_file must use the .mp4 extension\n");wb_robot_cleanup();return 2;
    }
    std::printf("WEBOTS_START phase=%s objective=%s seed=%u step_ms=%d custom=%s\n",config.phase.c_str(),config.goal_objective.c_str(),config.seed,step_ms,wb_robot_get_custom_data()?wb_robot_get_custom_data():"");
    std::fflush(stdout);
    const auto fields=data_fields(wb_robot_get_custom_data());
    const float* goal=config.goal;

    WbDeviceTag motors[4];
    for(int i=0;i<4;i++){motors[i]=wb_robot_get_device(kMotorNames[i]);if(!motors[i]){std::fprintf(stderr,"missing motor %s\n",kMotorNames[i]);wb_robot_cleanup();return 2;}wb_motor_set_position(motors[i],INFINITY);}
    const WbDeviceTag gps=wb_robot_get_device("gps"),imu=wb_robot_get_device("imu"),gyro=wb_robot_get_device("gyro"),depth=wb_robot_get_device("depth");
    if(!gps||!imu||!gyro||!depth){std::fprintf(stderr,"missing Webots ego/depth sensor\n");wb_robot_cleanup();return 2;}
    wb_gps_enable(gps,step_ms);wb_inertial_unit_enable(imu,step_ms);wb_gyro_enable(gyro,step_ms);wb_range_finder_enable(depth,50);
    const int width=wb_range_finder_get_width(depth),height=wb_range_finder_get_height(depth);
    if(width!=20||height!=16){std::fprintf(stderr,"expected 20x16 RangeFinder, got %dx%d\n",width,height);wb_robot_cleanup();return 2;}
    const float hfov=float(wb_range_finder_get_fov(depth));
    const char* model=wb_robot_get_model();(void)model;
    WbNodeRef self=wb_supervisor_node_get_self();
    const WbNodeRef depth_node=wb_supervisor_node_get_from_device(depth);
    wb_supervisor_node_enable_contact_points_tracking(self,step_ms,true);
    std::printf("WEBOTS_SENSORS range=%dx%d fov=%.3f self=%p\n",width,height,hfov,(void*)self);std::fflush(stdout);

    std::filesystem::path root=wb_robot_get_project_path()?wb_robot_get_project_path():".";
    const std::filesystem::path result_dir=root/"results";
    std::filesystem::create_directories(result_dir);
    if(config.raw_depth_shadow) {
        if(config.phase!="navigation"||config.policy_version!=1||config.raw_depth_shadow_path.empty()) {
            std::fprintf(stderr,"raw_depth_shadow requires navigation phase, policy_version=1, and raw_depth_shadow_path\n");
            wb_robot_cleanup();return 2;
        }
    }
    std::ofstream trace(result_dir/"last-run-trace.csv",std::ios::trunc);
    trace<<"step,time_s,x,y,z,vx,vy,vz,target_vx,target_vy,target_vz,goal_error";
    if(config.capture_trajectory)trace<<",q_w,q_x,q_y,q_z,goal_dwell_s,actual_world_speed_mps";
    trace<<'\n';
    const bool diagnostics=!config.diagnostic_prefix.empty();
    std::ofstream dense_diagnostic,nav_diagnostic,contact_diagnostic;
    std::ofstream raw_depth_shadow;
    uint32_t raw_depth_shadow_rows=0;
    if(config.raw_depth_shadow) {
        std::filesystem::path shadow_path=config.raw_depth_shadow_path;
        if(shadow_path.is_relative())shadow_path=root/shadow_path;
        if(shadow_path.has_parent_path())std::filesystem::create_directories(shadow_path.parent_path());
        raw_depth_shadow.open(shadow_path,std::ios::trunc);
        if(!raw_depth_shadow){std::fprintf(stderr,"cannot open raw depth shadow output: %s\n",shadow_path.string().c_str());wb_robot_cleanup();return 2;}
        raw_depth_shadow<<"step,time_s,current_frame,previous_frame,current_capture_time_s,previous_capture_time_s,layout_max_abs_error";
        for(int i=0;i<12;i++)raw_depth_shadow<<",current_camera_pose_"<<i;
        for(int i=0;i<12;i++)raw_depth_shadow<<",previous_camera_pose_"<<i;
        for(int i=0;i<320;i++)raw_depth_shadow<<",range_current_m_"<<i;
        for(int i=0;i<320;i++)raw_depth_shadow<<",range_previous_m_"<<i;
        for(int i=0;i<824;i++)raw_depth_shadow<<",observation_"<<i;
        raw_depth_shadow<<'\n'<<std::setprecision(9);
    }
    if(diagnostics) {
        dense_diagnostic.open(result_dir/(config.diagnostic_prefix+"-dense.csv"),std::ios::trunc);
        nav_diagnostic.open(result_dir/(config.diagnostic_prefix+"-nav.csv"),std::ios::trunc);
        contact_diagnostic.open(result_dir/(config.diagnostic_prefix+"-contacts.csv"),std::ios::trunc);
        dense_diagnostic<<"step,time_s,x,y,z,vx,vy,vz,q_w,q_x,q_y,q_z,body_rate_x,body_rate_y,body_rate_z,target_x,target_y,target_z,target_vx,target_vy,target_vz,motor_u0,motor_u1,motor_u2,motor_u3,motor_state0,motor_state1,motor_state2,motor_state3,goal_error,contact_count\n";
        nav_diagnostic<<"step,time_s,goal_x,goal_y,goal_z,goal_distance,body_vx,body_vy,body_vz,prior_x,prior_y,prior_z,prev_intent_x,prev_intent_y,prev_intent_z,prev_intent_yaw,action_x,action_y,action_z,action_yaw,target_body_vx,target_body_vy,target_body_vz,current_ranges,previous_ranges\n";
        contact_diagnostic<<"step,time_s,point_x,point_y,point_z,node_id,node_def\n";
    }
    if(config.phase=="geometry-calibrate"){
        std::ofstream geometry(result_dir/"geometry-axis-calibration.json",std::ios::trunc);
        geometry<<"{\"obstacles\":[";
        bool first=true;
        for(int index=0;index<8;index++){
            char def_name[40];std::snprintf(def_name,sizeof(def_name),"ChallengeObstacle%02d",index);
            const WbNodeRef obstacle=wb_supervisor_node_get_from_def(def_name);
            if(!obstacle)continue;
            const double* obstacle_position=wb_supervisor_node_get_position(obstacle);
            const double* obstacle_rotation=wb_supervisor_node_get_orientation(obstacle);
            if(!first)geometry<<',';
            first=false;
            geometry<<"{\"def\":\""<<def_name<<"\",\"position_xyz_m\":["
                    <<obstacle_position[0]<<','<<obstacle_position[1]<<','<<obstacle_position[2]
                    <<"],\"local_z_world\":["<<obstacle_rotation[2]<<','<<obstacle_rotation[5]<<','<<obstacle_rotation[8]<<"]}";
            std::printf("WEBOTS_GEOMETRY_CAL def=%s xyz=%g,%g,%g local_z_world=%g,%g,%g\n",
                        def_name,obstacle_position[0],obstacle_position[1],obstacle_position[2],
                        obstacle_rotation[2],obstacle_rotation[5],obstacle_rotation[8]);
        }
        geometry<<"]}\n";geometry.flush();geometry.close();
        std::fflush(stdout);
        wb_supervisor_simulation_quit(0);
        wb_robot_cleanup();
        return 0;
    }
    nav_deployment::NavigationPolicy navigation;
    nav_deployment::RawDepthNavigationPolicy raw_depth_navigation;
    const bool policy_mode=config.phase=="navigation";
    if(policy_mode){
        std::filesystem::path policy_path=config.policy;
        if(policy_path.is_relative())policy_path=root/policy_path;
        std::string error;
        const bool loaded=config.policy_version==1
            ?navigation.load(policy_path.string(),&error)
            :raw_depth_navigation.load(policy_path.string(),&error);
        if(!loaded){std::fprintf(stderr,"policy version%u load failed: %s\n",config.policy_version,error.c_str());wb_robot_cleanup();return 2;}
    }
    RaptorWeights raptor{};bool raptor_loaded=false;
    std::filesystem::path raptor_path=root/"../assets/raptor.bin";
    if(!raptor_load_weights(raptor_path.string().c_str(),&raptor)){
        // Webots project roots may resolve from either the repository root or `webots/`.
        raptor_path=root/"assets/raptor.bin";
        if(!raptor_load_weights(raptor_path.string().c_str(),&raptor)){std::fprintf(stderr,"RAPTOR weights not found at %s\n",raptor_path.string().c_str());wb_robot_cleanup();return 2;}
    }
    raptor_loaded=true;
    std::printf("WEBOTS_MODELS raptor=%s path=%s project=%s\n",raptor_loaded?"loaded":"pending",raptor_path.string().c_str(),root.string().c_str());std::fflush(stdout);
    const RLPhysicsParams physics=rl_physics_crazyflie_default();
    PhysicsComparison physics_comparison(physics,result_dir);
    const float c0=physics.rotor_thrust_coefficients[0],c1=physics.rotor_thrust_coefficients[1],c2=physics.rotor_thrust_coefficients[2];
    const float hover_force=physics.mass*9.81f/4.0f;
    float throttle=(-c1+std::sqrt(c1*c1-4.0f*c2*(c0-hover_force)))/(2.0f*c2);
    throttle=clampf(throttle,physics.action_min,physics.action_max);
    float motor_state[4]={throttle,throttle,throttle,throttle};
    float previous_nav_intent[4]={0,0,0,0};
    float previous_action[4]={0,0,0,0},hidden[16];raptor_reset(raptor,hidden);
    float target_position[3]={0,0,1.5f},target_velocity_body[3]={0,0,0},target_velocity_world[3]={0,0,0},target_yaw=0,target_yaw_rate=0;
    if(config.phase=="velocity"){for(int j=0;j<3;j++)target_velocity_body[j]=config.velocity[j];}
    float current_ranges[320],pooled[80],previous_pooled[80],range_ring[8*320],pose_ring[8*12];
    double range_capture_times[8]{};
    std::fill(current_ranges,current_ranges+320,kRangeMax);std::fill(pooled,pooled+80,kRangeMax);std::fill(previous_pooled,previous_pooled+80,kRangeMax);
    std::fill(range_ring,range_ring+8*320,kRangeMax);std::fill(pose_ring,pose_ring+8*12,0.0f);
    uint32_t capture_count=0,latest_frame=0,valid_frames=0;
    bool sensor_audit_written=false;
    bool view_snapshot_written=false;
    const uint32_t max_steps=config.max_steps;
    uint32_t steps=0;int completed=false,success=false,collision=false,timeout=false;
    double path=0.0,tracking_squared=0.0,minimum_sensor_range=kRangeMax;
    uint32_t tracking_samples=0;
    float peak_speed=0.0f,altitude_min=1.5f,altitude_max=1.5f;
    double previous_position[3]={0,0,1.5},current_time=0.0;
    uint32_t nav_updates=0;
    bool previous_inside_goal=false;
    uint32_t goal_radius_entry_count=0;
    double goal_radius_entry_first_time_s=-1.0,goal_radius_entry_last_time_s=-1.0;
    float goal_dwell_s=0.0f,final_world_speed_mps=0.0f;
    const bool movie_requested=!config.movie_file.empty();
    bool movie_recording=false,movie_failed=false;
    WbNodeRef recording_view_node=movie_requested?wb_supervisor_node_get_from_def("RLRecordingViewpoint"):nullptr;
    WbFieldRef recording_view_position=movie_requested?wb_supervisor_node_get_field(recording_view_node,"position"):nullptr;
    WbFieldRef recording_view_orientation=movie_requested?wb_supervisor_node_get_field(recording_view_node,"orientation"):nullptr;
    if(movie_requested) {
        if(!recording_view_node||!recording_view_position||!recording_view_orientation) {
            std::fprintf(stderr,"recording requires DEF RLRecordingViewpoint with position and orientation fields\n");
            wb_robot_cleanup();return 2;
        }
        const float initial_view_target[3]={0,0,1.5f};
        update_recording_view(recording_view_position,recording_view_orientation,initial_view_target);
        if(!wb_supervisor_movie_is_ready()) {
            std::fprintf(stderr,"Webots movie recorder is busy\n");wb_robot_cleanup();return 2;
        }
        wb_supervisor_movie_start_recording(config.movie_file.c_str(),1280,720,0,95,1,false);
        if(wb_supervisor_movie_failed()) {
            std::fprintf(stderr,"Webots movie recorder failed to start: %s\n",config.movie_file.c_str());
            wb_robot_cleanup();return 2;
        }
        movie_recording=true;
        std::printf("WEBOTS_MOVIE_START file=%s resolution=1280x720 acceleration=1\n",config.movie_file.c_str());
        std::fflush(stdout);
    }

    while(true) {
        const int step_status=wb_robot_step(step_ms);
        if(step_status==-1){std::printf("WEBOTS_STEP_END steps=%u\n",steps);std::fflush(stdout);break;}
        if(steps==0){std::printf("WEBOTS_STEP first\n");std::fflush(stdout);}
        current_time=wb_robot_get_time();
        const double* gps_position=wb_gps_get_values(gps);const double* gps_world_velocity=wb_gps_get_speed_vector(gps);
        const double* imu_xyzw=wb_inertial_unit_get_quaternion(imu);const double* gyro_values=wb_gyro_get_values(gyro);
        if(!gps_position||!gps_world_velocity||!imu_xyzw||!gyro_values)continue;
        if(steps==0){std::printf("WEBOTS_READ sensors %g,%g,%g q=%g,%g,%g,%g v=%g,%g,%g\n",gps_position[0],gps_position[1],gps_position[2],imu_xyzw[0],imu_xyzw[1],imu_xyzw[2],imu_xyzw[3],gps_world_velocity[0],gps_world_velocity[1],gps_world_velocity[2]);std::fflush(stdout);}
        const float position[3]={float(gps_position[0]),float(gps_position[1]),float(gps_position[2])};
        const float world_velocity[3]={float(gps_world_velocity[0]),float(gps_world_velocity[1]),float(gps_world_velocity[2])};
        const float q[4]={float(imu_xyzw[3]),float(imu_xyzw[0]),float(imu_xyzw[1]),float(imu_xyzw[2])};
        const float body_rates[3]={float(gyro_values[0]),float(gyro_values[1]),float(gyro_values[2])};
        float rotation[9];raptor_quaternion_matrix(q,rotation);
        if(movie_requested)update_recording_view(recording_view_position,recording_view_orientation,position);
        if(!view_snapshot_written&&!config.view_snapshot_file.empty()&&steps+1>=100) {
            wb_supervisor_export_image(config.view_snapshot_file.c_str(),95);
            view_snapshot_written=true;
            std::printf("WEBOTS_VIEW_SNAPSHOT file=%s step=%u time_s=%.3f\n",config.view_snapshot_file.c_str(),steps,current_time);
            std::fflush(stdout);
        }
        if(config.phase=="physics-audit" && steps>=200) {
            // Audit scoring reads ODE state explicitly. Navigation continues
            // to use sensor APIs; no Supervisor state enters its observation.
            const double* true_velocity=wb_supervisor_node_get_velocity(self);
            const float audit_velocity[3]={float(true_velocity[0]),float(true_velocity[1]),float(true_velocity[2])};
            const float world_rates[3]={float(true_velocity[3]),float(true_velocity[4]),float(true_velocity[5])};
            float audit_rates[3];rotate_world_to_body(rotation,world_rates,audit_rates);
            physics_comparison.observe(current_time,position,q,audit_velocity,audit_rates,motor_state);
        }
        path+=std::sqrt(std::pow(position[0]-previous_position[0],2)+std::pow(position[1]-previous_position[1],2)+std::pow(position[2]-previous_position[2],2));
        for(int j=0;j<3;j++)previous_position[j]=position[j];
        const float speed=vector_norm3(world_velocity);peak_speed=std::fmax(peak_speed,speed);altitude_min=std::fmin(altitude_min,position[2]);altitude_max=std::fmax(altitude_max,position[2]);
        // Sensors are sampled at 50 ms; after wb_robot_step, steps==4 is t=50 ms.
        // Align each stored depth image with the pose measured at its capture time.
        bool new_depth=(steps%kNavEvery==kNavEvery-1 && steps>=kNavEvery-1);
        if(new_depth){
            const float* image=wb_range_finder_get_range_image(depth);
            if(image){normalize_ranges(image,width,height,hfov,current_ranges);
                if(config.sensor_audit&&!sensor_audit_written){
                    const double* camera_position=wb_supervisor_node_get_position(depth_node);
                    const double* camera_rotation=wb_supervisor_node_get_orientation(depth_node);
                    std::ofstream audit(result_dir/"range-ray-audit.json",std::ios::trunc);
                    audit<<"{\"schema\":\"webots-native-range-audit-v1\",\"step\":"<<steps
                         <<",\"time_s\":"<<current_time<<",\"width\":"<<width<<",\"height\":"<<height
                         <<",\"horizontal_fov_rad\":"<<hfov<<",\"native_vertical_tangent\":"
                         <<std::tan(0.5f*hfov)*float(height)/float(width)<<",\"body_position_gps_xyz_m\":["
                         <<position[0]<<','<<position[1]<<','<<position[2]<<"],\"camera_position_supervisor_xyz_m\":["
                         <<camera_position[0]<<','<<camera_position[1]<<','<<camera_position[2]<<"],\"body_rotation_sensor_row_major\":[";
                    for(int i=0;i<9;i++){if(i)audit<<',';audit<<rotation[i];}
                    audit<<"],\"camera_rotation_supervisor_row_major\":[";
                    for(int i=0;i<9;i++){if(i)audit<<',';audit<<camera_rotation[i];}
                    audit<<"],\"native_axial_depth_m\":[";
                    for(int i=0;i<width*height;i++){
                        if(i)audit<<',';
                        audit<<(std::isfinite(image[i])?image[i]:12.0f);
                    }
                    audit<<"],\"normalized_ray_range_m\":[";
                    for(int i=0;i<width*height;i++){if(i)audit<<',';audit<<current_ranges[i];}
                    audit<<"]}\n";audit.flush();audit.close();
                    sensor_audit_written=true;
                }
                if(config.phase=="range-calibrate"){
                    float image_min=kRangeMax;
                    for(int k=0;k<width*height;k++)if(std::isfinite(image[k]))image_min=std::fmin(image_min,image[k]);
                    const double* camera_position=wb_supervisor_node_get_position(depth_node);
                    const double* supervisor_position=wb_supervisor_node_get_position(self);
                    const double* supervisor_rotation=wb_supervisor_node_get_orientation(self);
                    float quad[4]={kRangeMax,kRangeMax,kRangeMax,kRangeMax};
                    for(int y=0;y<height;y++)for(int x=0;x<width;x++){
                        const int q=(y>=height/2?2:0)+(x>=width/2?1:0);
                        quad[q]=std::fmin(quad[q],current_ranges[y*width+x]);
                    }
                    const float native_tan_v=std::tan(0.5f*hfov)*float(height)/float(width);
                    const float u_center=(2.0f*(float(width/2)+0.5f)/float(width)-1.0f)*std::tan(0.5f*hfov);
                    const float v_center=(1.0f-2.0f*(float(height/2)+0.5f)/float(height))*native_tan_v;
                    const float ray_world_x=rotation[0]-u_center*rotation[1]+v_center*rotation[2];
                    const float expected=(config.range_plane_x-0.05f-float(camera_position[0]))/ray_world_x;
                    std::printf("WEBOTS_RANGE_CAL step=%u gps_xyz=%g,%g,%g supervisor_xyz=%g,%g,%g imu_R=%g,%g,%g,%g,%g,%g,%g,%g,%g supervisor_R=%g,%g,%g,%g,%g,%g,%g,%g,%g sensor_xyz=%g,%g,%g center_axial=%g center_ray=%g min_axial=%g quads_tl_tr_bl_br=%g,%g,%g,%g plane_expected=%g\n",
                                steps,position[0],position[1],position[2],supervisor_position[0],supervisor_position[1],supervisor_position[2],rotation[0],rotation[1],rotation[2],rotation[3],rotation[4],rotation[5],rotation[6],rotation[7],rotation[8],
                                supervisor_rotation[0],supervisor_rotation[1],supervisor_rotation[2],supervisor_rotation[3],supervisor_rotation[4],supervisor_rotation[5],supervisor_rotation[6],supervisor_rotation[7],supervisor_rotation[8],
                                camera_position[0],camera_position[1],camera_position[2],image[(height/2)*width+width/2],current_ranges[(height/2)*width+width/2],image_min,quad[0],quad[1],quad[2],quad[3],expected);
                    std::fflush(stdout);
                }
                if(capture_count>0)std::copy(pooled,pooled+80,previous_pooled);
                pool_ranges(current_ranges,pooled);
                // The L2F reset state uses the first depth frame for both history slots.
                if(capture_count==0)std::copy(pooled,pooled+80,previous_pooled);
                for(int i=0;i<80;i++)minimum_sensor_range=std::fmin(minimum_sensor_range,pooled[i]);
                const uint slot=capture_count%8;std::copy(current_ranges,current_ranges+320,range_ring+slot*320);
                range_capture_times[slot]=current_time;
                float camera_offset_body[3]={0.08f,0,0},camera_offset_world[3];
                rotate_body_to_world(rotation,camera_offset_body,camera_offset_world);
                float* pose=pose_ring+slot*12;
                for(int i=0;i<3;i++)pose[i]=position[i]+camera_offset_world[i];
                for(int i=0;i<9;i++)pose[3+i]=rotation[i];
                latest_frame=capture_count;valid_frames=std::min(capture_count+1,8u);capture_count++;
                nav_updates++;
            }
        }
        if(policy_mode && new_depth && capture_count>0){
            float body_velocity[3];rotate_world_to_body(rotation,world_velocity,body_velocity);
            float delta_world[3]={goal[0]-position[0],goal[1]-position[1],goal[2]-position[2]},goal_body[3];rotate_world_to_body(rotation,delta_world,goal_body);
            const float distance=vector_norm3(delta_world),inv_distance=distance>1e-6f?1.0f/distance:0.0f;for(int j=0;j<3;j++)goal_body[j]*=inv_distance;
            float pose[12];for(int i=0;i<3;i++)pose[i]=position[i];for(int i=0;i<9;i++)pose[3+i]=rotation[i];
            float prior[3]={0,0,0};nav_guidance_memory(pooled,previous_pooled,goal_body,distance,body_velocity,kSensorDt,range_ring,pose_ring,pose,latest_frame,valid_frames,prior);
            float nav_previous_intent[4];for(int j=0;j<4;j++)nav_previous_intent[j]=previous_nav_intent[j];
            float observation[nav_deployment::raw_depth_actor_observation_count]{};
            float context[21]{};
            context[0]=goal_body[0];context[1]=goal_body[1];context[2]=goal_body[2];context[3]=std::fmin(distance/10.0f,1.5f);
            for(int j=0;j<3;j++){context[4+j]=body_velocity[j]/4.0f;context[7+j]=body_rates[j]/4.0f;}
            context[10]=rotation[6];context[11]=rotation[7];context[12]=rotation[8];
            for(int j=0;j<4;j++)context[13+j]=previous_nav_intent[j];
            context[17]=0;
            float ref_delta[3]={target_position[0]-position[0],target_position[1]-position[1],target_position[2]-position[2]},ref_body[3];rotate_world_to_body(rotation,ref_delta,ref_body);
            for(int j=0;j<3;j++)context[18+j]=clampf(ref_body[j]*2.0f,-1.0f,1.0f);
            if(config.policy_version==2) {
                const uint32_t previous_frame=valid_frames>1?(latest_frame+7u)%8u:latest_frame;
                const float* previous_raw=range_ring+previous_frame*320;
                nav_deployment::pack_raw_depth_observation(pooled,previous_pooled,context,
                    current_ranges,previous_raw,prior,observation);
            } else {
                for(int i=0;i<80;i++){observation[i]=pooled[i]/12.0f;observation[80+i]=previous_pooled[i]/12.0f;}
                std::copy(context,context+21,observation+160);
                for(int j=0;j<3;j++)observation[181+j]=prior[j];
            }
            if(raw_depth_shadow.is_open()) {
                const uint32_t current_slot=latest_frame%8u;
                const uint32_t previous_frame=valid_frames>1?latest_frame-1u:latest_frame;
                const uint32_t previous_slot=previous_frame%8u;
                const float* current_raw=range_ring+current_slot*320;
                const float* previous_raw=range_ring+previous_slot*320;
                const float* current_pose=pose_ring+current_slot*12;
                const float* previous_pose=pose_ring+previous_slot*12;
                float shadow_observation[nav_deployment::raw_depth_actor_observation_count];
                nav_deployment::pack_raw_depth_observation(pooled,previous_pooled,context,
                    current_raw,previous_raw,prior,shadow_observation);
                float layout_error=0.0f;
                for(int i=0;i<181;i++)layout_error=std::fmax(layout_error,std::fabs(shadow_observation[i]-observation[i]));
                for(int ray=0;ray<320;ray++) {
                    layout_error=std::fmax(layout_error,std::fabs(shadow_observation[nav_deployment::raw_current_range_offset+ray]-current_raw[ray]/12.0f));
                    layout_error=std::fmax(layout_error,std::fabs(shadow_observation[nav_deployment::raw_previous_range_offset+ray]-previous_raw[ray]/12.0f));
                    layout_error=std::fmax(layout_error,std::fabs(current_raw[ray]-current_ranges[ray]));
                }
                for(int j=0;j<3;j++)layout_error=std::fmax(layout_error,
                    std::fabs(shadow_observation[nav_deployment::raw_geometry_hint_offset+j]-prior[j]));
                raw_depth_shadow<<steps<<','<<current_time<<','<<latest_frame<<','<<previous_frame<<','
                                <<range_capture_times[current_slot]<<','<<range_capture_times[previous_slot]<<','<<layout_error;
                for(int i=0;i<12;i++)raw_depth_shadow<<','<<current_pose[i];
                for(int i=0;i<12;i++)raw_depth_shadow<<','<<previous_pose[i];
                for(int i=0;i<320;i++)raw_depth_shadow<<','<<current_raw[i];
                for(int i=0;i<320;i++)raw_depth_shadow<<','<<previous_raw[i];
                for(float value:shadow_observation)raw_depth_shadow<<','<<value;
                raw_depth_shadow<<'\n';raw_depth_shadow.flush();
                if(!raw_depth_shadow){std::fprintf(stderr,"raw depth shadow write failed\n");wb_robot_cleanup();return 2;}
                raw_depth_shadow_rows++;
            }
            nav_deployment::NavigationAction action;
            const bool action_ready=config.policy_version==1
                ?navigation.infer(observation,action)
                :raw_depth_navigation.infer(observation,action);
            if(action_ready){
                const float nav_scale=diagnostics?config.diagnostic_nav_scale:1.0f;
                for(int j=0;j<3;j++)target_velocity_body[j]=nav_scale*action.body_velocity_mps[j];
                target_yaw_rate=nav_scale*action.yaw_rate_rps;
                for(int j=0;j<4;j++)previous_nav_intent[j]=action.normalized_intent[j];
            }
            if(diagnostics) {
                nav_diagnostic<<steps<<','<<current_time<<','<<goal_body[0]<<','<<goal_body[1]<<','<<goal_body[2]<<','<<distance
                              <<','<<body_velocity[0]<<','<<body_velocity[1]<<','<<body_velocity[2]
                              <<','<<prior[0]<<','<<prior[1]<<','<<prior[2]
                              <<','<<nav_previous_intent[0]<<','<<nav_previous_intent[1]<<','<<nav_previous_intent[2]<<','<<nav_previous_intent[3]
                              <<','<<action.normalized_intent[0]<<','<<action.normalized_intent[1]<<','<<action.normalized_intent[2]<<','<<action.normalized_intent[3]
                              <<','<<target_velocity_body[0]<<','<<target_velocity_body[1]<<','<<target_velocity_body[2]<<',';
                for(int i=0;i<80;i++){if(i)nav_diagnostic<<'|';nav_diagnostic<<pooled[i];}
                nav_diagnostic<<',';
                for(int i=0;i<80;i++){if(i)nav_diagnostic<<'|';nav_diagnostic<<previous_pooled[i];}
                nav_diagnostic<<'\n';
                nav_diagnostic.flush();
            }
        }
        if(steps==0||new_depth){
            if(config.phase=="velocity"&&config.velocity_reference_is_world){
                for(int j=0;j<3;j++)target_velocity_world[j]=config.velocity[j];
                rotate_world_to_body(rotation,target_velocity_world,target_velocity_body);
            } else {
                rotate_body_to_world(rotation,target_velocity_body,target_velocity_world);
            }
        }
        for(int j=0;j<3;j++)target_position[j]+=target_velocity_world[j]*(step_ms*0.001f);
        target_yaw+=target_yaw_rate*(step_ms*0.001f);
        for(int j=0;j<3;j++){const double error=world_velocity[j]-target_velocity_world[j];tracking_squared+=error*error;}
        tracking_samples++;
        float target_q[4];quat_to_yaw(target_yaw,target_q);
        float state_position[3]={position[0],position[1],position[2]};
        float raptor_observation[22];raptor_pack_observation(state_position,q,world_velocity,body_rates,target_position,target_q,target_velocity_world,previous_action,raptor_observation);
        float motor_action[4];
        if(config.phase=="plant-hover"||config.phase=="range-calibrate") {
            for(int i=0;i<4;i++)motor_action[i]=2.0f*throttle-1.0f;
        } else {
            raptor_forward(raptor,raptor_observation,hidden,motor_action);raptor_clip_action(motor_action);
        }
        if(config.phase=="physics-audit" && steps>=200) {
            physics_comparison.apply_pulse(config.physics_profile,2*throttle-1,motor_action);
            physics_comparison.advance(motor_action);
        }
        if(steps==0){std::printf("WEBOTS_RAPTOR action=%g,%g,%g,%g\n",motor_action[0],motor_action[1],motor_action[2],motor_action[3]);std::fflush(stdout);}
        const float dt=step_ms*0.001f;
        for(int i=0;i<4;i++){
            previous_action[i]=motor_action[i];const float setpoint=(motor_action[i]+1.0f)*0.5f;
            const float tau=setpoint>=motor_state[i]?physics.rotor_time_constants_rising[i]:physics.rotor_time_constants_falling[i];
            const float initial=motor_state[i];
            const float alpha=std::exp(-dt/tau);
            motor_state[i]=alpha*initial+(1.0f-alpha)*setpoint;
            const float* thrust=&physics.rotor_thrust_coefficients[3*i];
            float force=thrust[0]+thrust[1]*motor_state[i]+thrust[2]*motor_state[i]*motor_state[i];
            if(config.motor_sampling=="average") {
                // ODE holds propeller force over one native interval. Integrate
                // the same continuous first-order actuator and quadratic force
                // curve used by L2F, rather than applying the end-of-step force.
                const float delta=initial-setpoint;
                const float first_moment=-tau*std::expm1(-dt/tau)/dt;
                const float second_moment=-.5f*tau*std::expm1(-2*dt/tau)/dt;
                const float mean_state=setpoint+delta*first_moment;
                const float mean_squared=setpoint*setpoint+2*setpoint*delta*first_moment+delta*delta*second_moment;
                force=thrust[0]+thrust[1]*mean_state+thrust[2]*mean_squared;
            }
            const float omega=std::sqrt(std::fmax(force/kRotorThrustK,0.0f));wb_motor_set_velocity(motors[i],float(kMotorSign[i])*omega);
        }
        if(steps==0){std::printf("WEBOTS_SET motors=%g,%g,%g,%g throttle=%g\n",wb_motor_get_velocity(motors[0]),wb_motor_get_velocity(motors[1]),wb_motor_get_velocity(motors[2]),wb_motor_get_velocity(motors[3]),motor_state[0]);std::fflush(stdout);}
        // Contact tracking is re-enabled after the controller loop is stable.
        int contact_count=0;
        WbContactPoint* contact_points=wb_supervisor_node_get_contact_points(self,true,&contact_count);
        if(steps==0){std::printf("WEBOTS_CONTACT count=%d\n",contact_count);std::fflush(stdout);}
        if(diagnostics&&contact_count>0&&contact_points) {
            for(int i=0;i<contact_count;i++) {
                const WbNodeRef contact_node=wb_supervisor_node_get_from_id(contact_points[i].node_id);
                const char* contact_def=contact_node?wb_supervisor_node_get_def(contact_node):nullptr;
                contact_diagnostic<<steps<<','<<current_time<<','<<contact_points[i].point[0]<<','<<contact_points[i].point[1]<<','<<contact_points[i].point[2]
                                 <<','<<contact_points[i].node_id<<','<<(contact_def?contact_def:"")<<'\n';
            }
            contact_diagnostic.flush();
        }
        if(contact_count>0)collision=true;
        const float goal_error=std::sqrt(std::pow(goal[0]-position[0],2)+std::pow(goal[1]-position[1],2)+std::pow(goal[2]-position[2],2));
        const bool inside_goal=goal_error<=0.35f;
        final_world_speed_mps=vector_norm3(world_velocity);
        if(inside_goal&&!previous_inside_goal) {
            goal_radius_entry_count++;
            if(goal_radius_entry_count==1)goal_radius_entry_first_time_s=current_time;
            goal_radius_entry_last_time_s=current_time;
        }
        previous_inside_goal=inside_goal;
        if(config.goal_objective=="hold") {
            if(inside_goal&&final_world_speed_mps<=0.5f)goal_dwell_s+=step_ms*0.001f;
            else goal_dwell_s=0.0f;
        }
        if(trace && (config.capture_trajectory || steps<80 || steps%100==0)){
            trace<<steps<<','<<current_time<<','<<position[0]<<','<<position[1]<<','<<position[2]<<','
                 <<world_velocity[0]<<','<<world_velocity[1]<<','<<world_velocity[2]<<','
                 <<target_velocity_world[0]<<','<<target_velocity_world[1]<<','<<target_velocity_world[2]<<','<<goal_error;
            if(config.capture_trajectory)trace<<','<<q[0]<<','<<q[1]<<','<<q[2]<<','<<q[3]<<','<<goal_dwell_s<<','<<final_world_speed_mps;
            trace<<'\n';
            trace.flush();
        }
        if(diagnostics) {
            dense_diagnostic<<steps<<','<<current_time<<','<<position[0]<<','<<position[1]<<','<<position[2]
                            <<','<<world_velocity[0]<<','<<world_velocity[1]<<','<<world_velocity[2]
                            <<','<<q[0]<<','<<q[1]<<','<<q[2]<<','<<q[3]
                            <<','<<body_rates[0]<<','<<body_rates[1]<<','<<body_rates[2]
                            <<','<<target_position[0]<<','<<target_position[1]<<','<<target_position[2]
                            <<','<<target_velocity_world[0]<<','<<target_velocity_world[1]<<','<<target_velocity_world[2]
                            <<','<<previous_action[0]<<','<<previous_action[1]<<','<<previous_action[2]<<','<<previous_action[3]
                            <<','<<motor_state[0]<<','<<motor_state[1]<<','<<motor_state[2]<<','<<motor_state[3]
                            <<','<<goal_error<<','<<contact_count<<'\n';
            dense_diagnostic.flush();
        }
        if(config.goal_objective=="entry") {
            if(goal_error<0.35f&&!collision){success=true;completed=true;}
        } else if(inside_goal&&final_world_speed_mps<=0.5f&&goal_dwell_s>=0.2f-1e-6f&&!collision) {
            success=true;completed=true;
        }
        steps++;
        if(steps%100==0){std::printf("WEBOTS_PROGRESS step=%u pz=%g vz=%g error=%g\n",steps,position[2],world_velocity[2],goal_error);std::fflush(stdout);}
        if(collision||success) {std::printf("WEBOTS_BREAK event=collision_or_success step=%u\n",steps);completed=true;break;}
        if(steps>=max_steps){std::printf("WEBOTS_BREAK event=timeout step=%u\n",steps);timeout=true;completed=true;break;}
    }
    if(config.phase=="physics-audit")physics_comparison.save(config.physics_profile,config.motor_sampling);
    if(movie_recording) {
        wb_supervisor_movie_stop_recording();
        const auto movie_wait_start=std::chrono::steady_clock::now();
        while(!wb_supervisor_movie_is_ready() &&
              std::chrono::duration<double>(std::chrono::steady_clock::now()-movie_wait_start).count()<300.0)
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        movie_failed=wb_supervisor_movie_failed()||!wb_supervisor_movie_is_ready();
        std::printf("WEBOTS_MOVIE_DONE file=%s failed=%d wait_s=%.2f\n",config.movie_file.c_str(),movie_failed?1:0,
                    std::chrono::duration<double>(std::chrono::steady_clock::now()-movie_wait_start).count());
        std::fflush(stdout);
    }
    std::printf("WEBOTS_LOOP_EXIT steps=%u completed=%d\n",steps,completed?1:0);std::fflush(stdout);
    if(raw_depth_shadow.is_open()) {
        raw_depth_shadow.flush();raw_depth_shadow.close();
        std::printf("WEBOTS_RAW_DEPTH_SHADOW rows=%u path=%s motor_policy_version=1\n",
                    raw_depth_shadow_rows,config.raw_depth_shadow_path.c_str());std::fflush(stdout);
    }
    const double* final_pos=wb_gps_get_values(gps);const float final_error=final_pos?float(std::sqrt(std::pow(goal[0]-final_pos[0],2)+std::pow(goal[1]-final_pos[1],2)+std::pow(goal[2]-final_pos[2],2))):NAN;
    std::ofstream metrics(result_dir/"last-run.json",std::ios::trunc);
    metrics<<"{\"phase\":\""<<config.phase<<"\",\"seed\":"<<config.seed
           <<",\"success\":"<<(success?"true":"false")<<",\"collision\":"<<(collision?"true":"false")
           <<",\"timeout\":"<<(timeout?"true":"false")<<",\"steps\":"<<steps<<",\"time_s\":"<<current_time
           <<",\"path_m\":"<<path<<",\"final_error_m\":"<<final_error<<",\"peak_speed_mps\":"<<peak_speed
           <<",\"min_sensor_range_m\":"<<minimum_sensor_range<<",\"altitude_min_m\":"<<altitude_min
           <<",\"altitude_max_m\":"<<altitude_max<<",\"tracking_rms_mps\":"
           <<(tracking_samples?std::sqrt(tracking_squared/(3.0*tracking_samples)):0.0)
           <<",\"sensor_updates\":"<<nav_updates<<",\"raptor_loaded\":"<<(raptor_loaded?"true":"false")
           <<",\"navigation_loaded\":"<<(policy_mode?"true":"false")
           <<",\"policy_version\":"<<config.policy_version
           <<",\"goal_objective\":\""<<config.goal_objective<<"\",\"goal_radius_entry_count\":"<<goal_radius_entry_count
           <<",\"goal_radius_entry_first_time_s\":"<<goal_radius_entry_first_time_s
           <<",\"goal_radius_entry_last_time_s\":"<<goal_radius_entry_last_time_s
           <<",\"goal_dwell_s\":"<<goal_dwell_s<<",\"final_world_speed_mps\":"<<final_world_speed_mps
           <<",\"raw_depth_shadow\":"<<(config.raw_depth_shadow?"true":"false")
           <<",\"raw_depth_shadow_rows\":"<<raw_depth_shadow_rows
           <<",\"trajectory_trace\":"<<(config.capture_trajectory?"true":"false")
           <<",\"movie_recording\":"<<(movie_requested?"true":"false")<<",\"movie_failed\":"<<(movie_failed?"true":"false")
           <<",\"movie_file\":\""<<config.movie_file<<"\",\"view_snapshot_file\":\""<<config.view_snapshot_file
           <<"\",\"view_snapshot_written\":"<<(view_snapshot_written?"true":"false")<<"}\n";
    metrics.flush();metrics.close();trace.flush();trace.close();
    if(diagnostics) {dense_diagnostic.flush();nav_diagnostic.flush();contact_diagnostic.flush();}
    std::ofstream marker(result_dir/"last-run-exit.marker",std::ios::trunc);
    marker<<"loop_exit steps="<<steps<<" completed="<<(completed?1:0)<<"\n";
    marker.flush();marker.close();
    std::printf("WEBOTS_RESULT phase=%s objective=%s seed=%u success=%d collision=%d timeout=%d steps=%u time_s=%.4f path_m=%.4f final_error_m=%.4f mean_speed_mps=%.4f peak_speed_mps=%.4f tracking_rms_mps=%.4f min_sensor_range_m=%.4f altitude_min_m=%.4f altitude_max_m=%.4f raptor_loaded=%d nav_loaded=%d nav_updates=%u goal_entries=%u goal_dwell_s=%.3f final_speed_mps=%.4f\n",
        config.phase.c_str(),config.goal_objective.c_str(),config.seed,success?1:0,collision?1:0,timeout?1:0,steps,current_time,path,final_error,
        steps?path/(steps*step_ms*0.001):0,peak_speed,tracking_samples?std::sqrt(tracking_squared/(3.0*tracking_samples)):0,minimum_sensor_range,altitude_min,altitude_max,raptor_loaded?1:0,policy_mode?1:0,nav_updates,goal_radius_entry_count,goal_dwell_s,final_world_speed_mps);
    std::fflush(stdout);
    wb_supervisor_simulation_quit(movie_failed?2:0);
    wb_robot_cleanup();
    return movie_failed?2:(completed?0:1);
}
