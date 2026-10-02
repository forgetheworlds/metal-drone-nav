#include <webots/robot.h>
#include <webots/motor.h>
#include <webots/range_finder.h>
#include <webots/gps.h>
#include <webots/inertial_unit.h>
#include <webots/gyro.h>
#include <webots/supervisor.h>

#include "deployment.hpp"
#include "guidance.hpp"
#include "physics.hpp"
#include "raptor.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
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
    std::string policy="../assets/navigation.bin";
    uint32_t seed=1;
    uint32_t max_steps=800;
    float speed=1.5f;
    float distance=4.0f;
    float range_plane_x=2.0f;
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
            else if(key=="policy")c.policy=value;
            else if(key=="seed")c.seed=uint32_t(std::strtoul(value.c_str(),nullptr,10));
            else if(key=="max_steps")c.max_steps=uint32_t(std::strtoul(value.c_str(),nullptr,10));
            else if(key=="speed")parse_float(value,c.speed);
            else if(key=="distance")parse_float(value,c.distance);
            else if(key=="range_plane_x")parse_float(value,c.range_plane_x);
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
    const int step_ms=int(std::round(wb_robot_get_basic_time_step()));
    if(step_ms!=10){std::fprintf(stderr,"Webots basicTimeStep must be 10ms, got %d\n",step_ms);wb_robot_cleanup();return 2;}
    const Config config=parse_config(wb_robot_get_custom_data());
    std::printf("WEBOTS_START phase=%s seed=%u step_ms=%d custom=%s\n",config.phase.c_str(),config.seed,step_ms,wb_robot_get_custom_data()?wb_robot_get_custom_data():"");
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
    std::ofstream trace(result_dir/"last-run-trace.csv",std::ios::trunc);
    trace<<"step,time_s,x,y,z,vx,vy,vz,target_vx,target_vy,target_vz,goal_error\n";
    nav_deployment::NavigationPolicy navigation;
    const bool policy_mode=config.phase=="navigation";
    if(policy_mode){
        std::filesystem::path policy_path=config.policy;
        if(policy_path.is_relative())policy_path=root/policy_path;
        std::string error;
        if(!navigation.load(policy_path.string(),&error)){std::fprintf(stderr,"policy load failed: %s\n",error.c_str());wb_robot_cleanup();return 2;}
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
    std::fill(current_ranges,current_ranges+320,kRangeMax);std::fill(pooled,pooled+80,kRangeMax);std::fill(previous_pooled,previous_pooled+80,kRangeMax);
    std::fill(range_ring,range_ring+8*320,kRangeMax);std::fill(pose_ring,pose_ring+8*12,0.0f);
    uint32_t capture_count=0,latest_frame=0,valid_frames=0;
    const uint32_t max_steps=config.max_steps;
    uint32_t steps=0;int completed=false,success=false,collision=false,timeout=false;
    double path=0.0,tracking_squared=0.0,minimum_sensor_range=kRangeMax;
    uint32_t tracking_samples=0;
    float peak_speed=0.0f,altitude_min=1.5f,altitude_max=1.5f;
    double previous_position[3]={0,0,1.5},current_time=0.0;
    uint32_t nav_updates=0;

    while(true) {
        const int step_status=wb_robot_step(step_ms);
        if(step_status==-1){std::printf("WEBOTS_STEP_END steps=%u\n",steps);std::fflush(stdout);break;}
        if(config.phase=="geometry-calibrate"){
            std::printf("WEBOTS_GEOMETRY_CAL_BEGIN\n");
            for(int index=0;index<8;index++){
                char def_name[40];std::snprintf(def_name,sizeof(def_name),"ChallengeObstacle%02d",index);
                const WbNodeRef obstacle=wb_supervisor_node_get_from_def(def_name);
                if(!obstacle){if(index<2)std::printf("WEBOTS_GEOMETRY_CAL_MISSING def=%s\n",def_name);continue;}
                const double* obstacle_position=wb_supervisor_node_get_position(obstacle);
                const double* obstacle_rotation=wb_supervisor_node_get_orientation(obstacle);
                std::printf("WEBOTS_GEOMETRY_CAL def=%s xyz=%g,%g,%g local_z_world=%g,%g,%g\n",
                            def_name,obstacle_position[0],obstacle_position[1],obstacle_position[2],
                            obstacle_rotation[2],obstacle_rotation[5],obstacle_rotation[8]);
            }
            std::fflush(stdout);
            wb_supervisor_simulation_quit(0);
            wb_robot_cleanup();
            return 0;
        }
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
        path+=std::sqrt(std::pow(position[0]-previous_position[0],2)+std::pow(position[1]-previous_position[1],2)+std::pow(position[2]-previous_position[2],2));
        for(int j=0;j<3;j++)previous_position[j]=position[j];
        const float speed=vector_norm3(world_velocity);peak_speed=std::fmax(peak_speed,speed);altitude_min=std::fmin(altitude_min,position[2]);altitude_max=std::fmax(altitude_max,position[2]);
        // Sensors are sampled at 50 ms; after wb_robot_step, steps==4 is t=50 ms.
        // Align each stored depth image with the pose measured at its capture time.
        bool new_depth=(steps%kNavEvery==kNavEvery-1 && steps>=kNavEvery-1);
        if(new_depth){
            const float* image=wb_range_finder_get_range_image(depth);
            if(image){normalize_ranges(image,width,height,hfov,current_ranges);
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
            float observation[nav_deployment::actor_observation_count]{};
            for(int i=0;i<80;i++){observation[i]=pooled[i]/12.0f;observation[80+i]=previous_pooled[i]/12.0f;}
            observation[160]=goal_body[0];observation[161]=goal_body[1];observation[162]=goal_body[2];observation[163]=std::fmin(distance/10.0f,1.5f);
            for(int j=0;j<3;j++){observation[164+j]=body_velocity[j]/4.0f;observation[167+j]=body_rates[j]/4.0f;}
            observation[170]=rotation[6];observation[171]=rotation[7];observation[172]=rotation[8];
            for(int j=0;j<4;j++)observation[173+j]=previous_nav_intent[j];
            observation[177]=0;
            float ref_delta[3]={target_position[0]-position[0],target_position[1]-position[1],target_position[2]-position[2]},ref_body[3];rotate_world_to_body(rotation,ref_delta,ref_body);
            for(int j=0;j<3;j++)observation[178+j]=clampf(ref_body[j]*2.0f,-1.0f,1.0f);
            for(int j=0;j<3;j++)observation[181+j]=prior[j];
            nav_deployment::NavigationAction action;
            if(navigation.infer(observation,action)){
                for(int j=0;j<3;j++)target_velocity_body[j]=action.body_velocity_mps[j];
                target_yaw_rate=action.yaw_rate_rps;
                for(int j=0;j<4;j++)previous_nav_intent[j]=action.normalized_intent[j];
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
        if(steps==0){std::printf("WEBOTS_RAPTOR action=%g,%g,%g,%g\n",motor_action[0],motor_action[1],motor_action[2],motor_action[3]);std::fflush(stdout);}
        const float dt=step_ms*0.001f;
        for(int i=0;i<4;i++){
            previous_action[i]=motor_action[i];const float setpoint=(motor_action[i]+1.0f)*0.5f;
            const float tau=setpoint>=motor_state[i]?physics.rotor_time_constants_rising[i]:physics.rotor_time_constants_falling[i];
            const float alpha=std::exp(-dt/tau);motor_state[i]=alpha*motor_state[i]+(1.0f-alpha)*setpoint;
            const float* thrust=&physics.rotor_thrust_coefficients[3*i];const float force=thrust[0]+thrust[1]*motor_state[i]+thrust[2]*motor_state[i]*motor_state[i];
            const float omega=std::sqrt(std::fmax(force/kRotorThrustK,0.0f));wb_motor_set_velocity(motors[i],float(kMotorSign[i])*omega);
        }
        if(steps==0){std::printf("WEBOTS_SET motors=%g,%g,%g,%g throttle=%g\n",wb_motor_get_velocity(motors[0]),wb_motor_get_velocity(motors[1]),wb_motor_get_velocity(motors[2]),wb_motor_get_velocity(motors[3]),motor_state[0]);std::fflush(stdout);}
        // Contact tracking is re-enabled after the controller loop is stable.
        int contact_count=0;
        wb_supervisor_node_get_contact_points(self,true,&contact_count);
        if(steps==0){std::printf("WEBOTS_CONTACT count=%d\n",contact_count);std::fflush(stdout);}
        if(contact_count>0)collision=true;
        const float goal_error=std::sqrt(std::pow(goal[0]-position[0],2)+std::pow(goal[1]-position[1],2)+std::pow(goal[2]-position[2],2));
        if(trace && (steps<80 || steps%100==0)){
            trace<<steps<<','<<current_time<<','<<position[0]<<','<<position[1]<<','<<position[2]<<','
                 <<world_velocity[0]<<','<<world_velocity[1]<<','<<world_velocity[2]<<','
                 <<target_velocity_world[0]<<','<<target_velocity_world[1]<<','<<target_velocity_world[2]<<','<<goal_error<<'\n';
            trace.flush();
        }
        if(goal_error<0.35f && !collision){success=true;completed=true;}
        steps++;
        if(steps%100==0){std::printf("WEBOTS_PROGRESS step=%u pz=%g vz=%g error=%g\n",steps,position[2],world_velocity[2],goal_error);std::fflush(stdout);}
        if(collision||success) {std::printf("WEBOTS_BREAK event=collision_or_success step=%u\n",steps);completed=true;break;}
        if(steps>=max_steps){std::printf("WEBOTS_BREAK event=timeout step=%u\n",steps);timeout=true;completed=true;break;}
    }
    std::printf("WEBOTS_LOOP_EXIT steps=%u completed=%d\n",steps,completed?1:0);std::fflush(stdout);
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
           <<",\"navigation_loaded\":"<<(policy_mode?"true":"false")<<"}\n";
    metrics.flush();metrics.close();trace.flush();trace.close();
    std::ofstream marker(result_dir/"last-run-exit.marker",std::ios::trunc);
    marker<<"loop_exit steps="<<steps<<" completed="<<(completed?1:0)<<"\n";
    marker.flush();marker.close();
    std::printf("WEBOTS_RESULT phase=%s seed=%u success=%d collision=%d timeout=%d steps=%u time_s=%.4f path_m=%.4f final_error_m=%.4f mean_speed_mps=%.4f peak_speed_mps=%.4f tracking_rms_mps=%.4f min_sensor_range_m=%.4f altitude_min_m=%.4f altitude_max_m=%.4f raptor_loaded=%d nav_loaded=%d nav_updates=%u\n",
        config.phase.c_str(),config.seed,success?1:0,collision?1:0,timeout?1:0,steps,current_time,path,final_error,
        steps?path/(steps*step_ms*0.001):0,peak_speed,tracking_samples?std::sqrt(tracking_squared/(3.0*tracking_samples)):0,minimum_sensor_range,altitude_min,altitude_max,raptor_loaded?1:0,policy_mode?1:0,nav_updates);
    std::fflush(stdout);
    wb_supervisor_simulation_quit(0);
    wb_robot_cleanup();
    return completed?0:1;
}
