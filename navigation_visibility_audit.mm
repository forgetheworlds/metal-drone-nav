#define WAYPOINT_EMBEDDED
#include "navigation_waypoint_training.mm"
#include <array>

// Posthoc diagnostic only. These geometry IDs never reach the controller.
static int first_hit(const WWorld& world,WVec origin,WVec direction,float time,float& range) {
    range=12.0f;int id=-1;
    const float wall=wray_box(origin,direction,wv(6,0,2.5f),wv(8,5,2.5f));
    if(wall<range) {
        range=wall;const WVec p=wa(origin,wm(direction,wall));
        const float distances[6]={std::fabs(p.x+2),std::fabs(p.x-14),std::fabs(p.y+5),
                                  std::fabs(p.y-5),std::fabs(p.z),std::fabs(p.z-5)};
        id=16+int(std::min_element(distances,distances+6)-distances);
    }
    for(uint j=0;j<world.count;j++) {
        const auto& obstacle=world.obstacles[j];float distance=1000;
        const WVec center=wc(obstacle,time);
        if(obstacle.kind==0)distance=wray_box(origin,direction,center,wv(obstacle.size[0],obstacle.size[1],obstacle.size[2]));
        else if(obstacle.kind==1)distance=wray_sphere(origin,direction,center,obstacle.size[0]);
        else distance=wray_cylinder(origin,direction,center,obstacle.size[0],obstacle.size[2]);
        if(distance<range){range=distance;id=int(j);}
    }
    return id;
}

static int nearest_contact(const WWorld& world,WVec position,float time) {
    const float bounds[6]={position.x+2,14-position.x,position.y+5,5-position.y,position.z,5-position.z};
    int id=16+int(std::min_element(bounds,bounds+6)-bounds);
    float best=*std::min_element(bounds,bounds+6)-.18f;
    for(uint j=0;j<world.count;j++) {
        WWorld single=world;single.count=1;single.obstacles[0]=world.obstacles[j];
        const float distance=wclearance(single,position,time);
        if(distance<best){best=distance;id=int(j);}
    }
    return id;
}

// Conservative box/frustum exclusion: only a separated half-space proves
// the complete obstacle outside view. Overlap is inconclusive, not visibility.
static bool outside_view(const WWorld& world,int id,const std::array<float,12>& pose) {
    WVec center{},extent{};
    if(id<16) {
        const auto& o=world.obstacles[id];center=wc(o,0);extent=wv(o.size[0],o.size[1],o.size[2]);
        if(o.kind==1)extent=wv(o.size[0],o.size[0],o.size[0]);
        if(o.kind==2)extent=wv(o.size[0],o.size[0],o.size[2]);
    } else {
        center=wv(6,0,2.5f);extent=wv(8,5,2.5f);
        const int face=id-16,axis=face/2;const float sign=face%2?1:-1;
        if(axis==0){center.x+=sign*extent.x;extent.x=0;}
        else if(axis==1){center.y+=sign*extent.y;extent.y=0;}
        else {center.z+=sign*extent.z;extent.z=0;}
    }
    std::array<bool,6> separated{};separated.fill(true);
    for(int sx:{-1,1})for(int sy:{-1,1})for(int sz:{-1,1}) {
        const float d[3]={center.x+sx*extent.x-pose[0],center.y+sy*extent.y-pose[1],center.z+sz*extent.z-pose[2]};
        float b[3];for(int j=0;j<3;j++)b[j]=pose[3+j]*d[0]+pose[6+j]*d[1]+pose[9+j]*d[2];
        const float planes[6]={b[0],b[0]-b[1],b[0]+b[1],NAV_SENSOR_ACTIVE_TAN_V*b[0]-b[2],NAV_SENSOR_ACTIVE_TAN_V*b[0]+b[2],12-b[0]};
        for(int j=0;j<6;j++)separated[j]=separated[j]&&planes[j]<0;
    }
    return std::any_of(separated.begin(),separated.end(),[](bool value){return value;});
}

int main(int argc,char**argv){@autoreleasepool{try{
    require(argc==4,"visibility CHECKPOINT BANK_BIN OUTPUT_CSV");
    waypoint::BankControl control{};std::string bank_hash;
    const auto bank=waypoint::read_bank(argv[2],control,bank_hash);
    require(control.period==1&&bank.size()==128,"requires the kept128task DEV bank");
    for(const auto& entry:bank)for(uint j=0;j<entry.world.count;j++)for(float velocity:entry.world.obstacles[j].velocity)
        require(velocity==0,"frustum exclusion diagnostic requires static geometry");
    Metal metal;metal.compile(base_source()+PPO_TRAINER_MSL+waypoint::kWaypointKernels);
    SimConfig config;config.n=128;config.eval=1;config.mode=17;config.speed=1.5f;
    config.max_steps=400;config.geometry_memory=1;config.seed=700001;
    Sim sim(metal,config,32);navigation_training::load_actor(sim,argv[1],false);
    auto run=waypoint::make_local_run(sim,bank,control,metal.pipeline("waypoint_task_apply"));
    auto cb=[metal.queue commandBuffer];waypoint::local_probe(run,cb);metal.finish(cb);
    std::array<std::array<float,22>,128> first_raw{},first_pooled{},last_raw{},last_pooled{};
    for(auto& row:first_raw)row.fill(-1);for(auto& row:first_pooled)row.fill(-1);
    for(auto& row:last_raw)row.fill(-1);for(auto& row:last_pooled)row.fill(-1);
    std::array<uint,128> previous_episodes{};float maximum_ray_error=0;
    std::array<std::vector<std::array<float,12>>,128> camera_history;
    for(uint tick=0;tick<400;tick++) {
        const auto* before=(const SimRun*)sim.runs.contents;
        for(uint env=0;env<128;env++)previous_episodes[env]=before[env].episodes;
        cb=[metal.queue commandBuffer];waypoint::local_tick(run,cb,tick,false);metal.finish(cb);
        const auto* runs=(const SimRun*)sim.runs.contents;
        const float* sensors=(const float*)sim.sensors.contents;
        const float* poses=(const float*)sim.poses.contents;
        for(uint env=0;env<128;env++) {
            if(previous_episodes[env])continue;
            const uint frame=(runs[env].steps-1)%8;
            const float* pose=poses+(env*8+frame)*12;
            const float* ranges=sensors+(env*8+frame)*320;
            std::array<float,12> captured_pose;std::copy(pose,pose+12,captured_pose.begin());camera_history[env].push_back(captured_pose);
            std::array<int,320> ids{};const float time=tick*.05f;
            for(uint pixel=0;pixel<320;pixel++) {
                const WVec ray=nav_sensor_pixel_ray(1,NAV_SENSOR_ACTIVE_TAN_V,pixel);
                const WVec direction=wv(pose[3]*ray.x+pose[4]*ray.y+pose[5]*ray.z,
                    pose[6]*ray.x+pose[7]*ray.y+pose[8]*ray.z,pose[9]*ray.x+pose[10]*ray.y+pose[11]*ray.z);
                float expected;ids[pixel]=first_hit(bank[env].world,wv(pose[0],pose[1],pose[2]),direction,time,expected);
                maximum_ray_error=std::max(maximum_ray_error,std::fabs(expected-ranges[pixel]));
                if(ids[pixel]>=0){last_raw[env][ids[pixel]]=time;if(first_raw[env][ids[pixel]]<0)first_raw[env][ids[pixel]]=time;}
            }
            for(uint cell=0;cell<80;cell++) {
                const uint pixel=(cell/10)*40+(cell%10)*2;uint hit=pixel;
                for(uint candidate:{pixel+1,pixel+20,pixel+21})if(ranges[candidate]<ranges[hit])hit=candidate;
                if(ids[hit]>=0){last_pooled[env][ids[hit]]=time;if(first_pooled[env][ids[hit]]<0)first_pooled[env][ids[hit]]=time;}
            }
        }
    }
    require(maximum_ray_error<2e-4f,"ray attribution does not match measured GPU sensor");
    const auto* runs=(const SimRun*)sim.runs.contents;const auto* states=(const RLPhysicsState*)sim.states.contents;
    std::ofstream out(argv[3]);require(bool(out),"cannot write visibility records");
    out<<"env,family,success,collision,timeout,time_s,contact_geometry_id,first_raw_seen_s,first_pooled_seen_s,raw_warning_s,pooled_warning_s,last_raw_seen_s,last_pooled_seen_s,entire_collider_outside_view_all_frames,dense_counterfactual_first_seen_s\n";
    uint contacts=0,unseen_raw=0,unseen_pooled=0;
    for(uint env=0;env<128;env++) {
        require(runs[env].episodes==1,"missing terminal task");
        const auto& state=states[env];const int collider=runs[env].collisions?nearest_contact(bank[env].world,wv(state.position[0],state.position[1],state.position[2]),runs[env].elapsed):-1;
        const float raw=collider>=0?first_raw[env][collider]:-1,pooled=collider>=0?first_pooled[env][collider]:-1;
        if(collider>=0){contacts++;unseen_raw+=raw<0;unseen_pooled+=pooled<0;}
        const bool always_outside=collider>=0 && std::all_of(camera_history[env].begin(),camera_history[env].end(),[&](const auto& pose){return outside_view(bank[env].world,collider,pose);});
        float dense_first=-1;
        // Higher ray density at exactly the same captured poses and FoV.
        // This is an analytic diagnostic, never an actor input or flight claim.
        if(collider>=0&&raw<0&&!always_outside) {
            for(size_t frame=0;frame<camera_history[env].size()&&dense_first<0;frame++) {
                const auto& pose=camera_history[env][frame];
                if(outside_view(bank[env].world,collider,pose))continue;
                for(uint row=0;row<64&&dense_first<0;row++)for(uint col=0;col<80;col++) {
                    const WVec ray=wn(wv(1,1-2*(float(col)+.5f)/80,NAV_SENSOR_ACTIVE_TAN_V*(1-2*(float(row)+.5f)/64)));
                    const WVec direction=wv(pose[3]*ray.x+pose[4]*ray.y+pose[5]*ray.z,
                        pose[6]*ray.x+pose[7]*ray.y+pose[8]*ray.z,pose[9]*ray.x+pose[10]*ray.y+pose[11]*ray.z);
                    float range;
                    if(first_hit(bank[env].world,wv(pose[0],pose[1],pose[2]),direction,float(frame)*.05f,range)==collider) {
                        dense_first=float(frame)*.05f;break;
                    }
                }
            }
        }

        out<<env<<','<<bank[env].family<<','<<runs[env].successes<<','<<runs[env].collisions<<','<<runs[env].timeouts<<','<<runs[env].elapsed<<','<<collider<<','<<raw<<','<<pooled<<','<<(raw<0?-1:runs[env].elapsed-raw)<<','<<(pooled<0?-1:runs[env].elapsed-pooled)<<','<<(collider<0?-1:last_raw[env][collider])<<','<<(collider<0?-1:last_pooled[env][collider])<<','<<always_outside<<','<<dense_first<<'\n';
    }
    std::cout<<"contacts="<<contacts<<" collider_unseen_raw="<<unseen_raw<<" collider_unseen_pooled="<<unseen_pooled<<" maximum_sensor_attribution_error_m="<<maximum_ray_error<<'\n';
    return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}}
