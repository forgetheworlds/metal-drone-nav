// All buffers have fixed layouts. Policy -> reference -> RAPTOR -> RK4 runs on GPU.
struct SimRun {
    float hidden[16], motors[4], previous_nav[4], desired_velocity[3], yaw;
    uint rng, steps;
    float path, elapsed, peak_speed, min_clearance, initial_distance;
    uint successes, collisions, timeouts, episodes;
    float success_time, total_path, total_elapsed, final_progress, reference_position[3];
};
struct SimConfig {
    uint n, tick, mode, family, substeps, sensor_period, sensor_delay, command_delay, max_steps, seed, eval;
    float speed, distance, wind, depth_noise, dropout, risk_coef, entropy_coef, learning_rate;uint velocity_contract,geometry_memory;
};
struct ChallengeBankControl { uint enabled, bank_count, schedule_stride, horizon; };
inline bool sim_clean_training_env(constant SimConfig& cfg,uint n) {
    return cfg.eval==0 && (cfg.family==12 || cfg.family==13) && (n%2==0);
}
inline uint sim_sensor_delay(constant SimConfig& cfg,uint n) {
    return sim_clean_training_env(cfg,n)?0:cfg.sensor_delay;
}
inline uint sim_command_delay(constant SimConfig& cfg,uint n) {
    return sim_clean_training_env(cfg,n)?0:cfg.command_delay;
}
inline float sim_wind_param(constant SimConfig& cfg,uint n) {
    return sim_clean_training_env(cfg,n)?0.0f:cfg.wind;
}
inline float sim_depth_noise(constant SimConfig& cfg,uint n) {
    return sim_clean_training_env(cfg,n)?0.0f:cfg.depth_noise;
}
inline float sim_depth_dropout(constant SimConfig& cfg,uint n) {
    return sim_clean_training_env(cfg,n)?0.0f:cfg.dropout;
}
constant uint SIM_DEPTH_FEATURES = (PPO_ACTOR_OBS - (PPO_ACTOR_OBS==184?24:21)) / 2;
inline void sim_rotation(thread const float* q, thread float* r) {
    float w=q[0],x=q[1],y=q[2],z=q[3];
    r[0]=1-2*(y*y+z*z);r[1]=2*(x*y-w*z);r[2]=2*(x*z+w*y);
    r[3]=2*(x*y+w*z);r[4]=1-2*(x*x+z*z);r[5]=2*(y*z-w*x);
    r[6]=2*(x*z-w*y);r[7]=2*(y*z+w*x);r[8]=1-2*(x*x+y*y);
}
inline float sim_normal(thread uint& rng) { float u=max(wurand(rng),1e-7f),v=wurand(rng);return sqrt(-2*log(u))*cos(6.28318530718f*v); }
inline void sim_reset_one(device RLPhysicsState& s,device SimRun& run,device WWorld& world,device float* sensors,device float* commands,device const float* weights,constant RLPhysicsParams& p,constant SimConfig& cfg,uint n,bool first) {
    uint rng=first?(cfg.seed+n*747796405u+2891336453u):run.rng;
    uint episode_family=wtraining_family(cfg.family,rng);
    wgenerate(world,wrng(rng),episode_family,cfg.distance);
    world.wind[0]=sim_wind_param(cfg,n);world.wind[1]=0;world.wind[2]=0;
    for(uint j=0;j<3;j++){s.position[j]=j==2?1.5f:0;s.linear_velocity[j]=0;s.angular_velocity_body[j]=0;run.desired_velocity[j]=0;run.reference_position[j]=s.position[j];}
    s.orientation_wxyz[0]=1;for(uint j=1;j<4;j++)s.orientation_wxyz[j]=0;
    float target=p.mass*9.81f/4,c0=p.rotor_thrust_coefficients[0],c1=p.rotor_thrust_coefficients[1],c2=p.rotor_thrust_coefficients[2];
    float hover=clamp((-c1+sqrt(c1*c1-4*c2*(c0-target)))/(2*c2),p.action_min,p.action_max);
    float hidden[16];raptor_reset(weights,hidden);for(uint j=0;j<16;j++)run.hidden[j]=hidden[j];
    for(uint j=0;j<4;j++){s.rpm[j]=hover;run.motors[j]=0;run.previous_nav[j]=0;}
    if(world.family==10 || world.family==11) {
        s.linear_velocity[0]=cfg.speed;
        run.desired_velocity[0]=cfg.speed;
        run.previous_nav[0]=1.0f;
    }
    for(uint j=0;j<8*320;j++)sensors[n*8*320+j]=12;
    for(uint j=0;j<8*4;j++)commands[n*8*4+j]=0;
    run.yaw=0;run.rng=rng;run.steps=0;run.path=run.elapsed=run.peak_speed=0;run.min_clearance=12;run.initial_distance=length(float3(world.goal[0],world.goal[1],world.goal[2]-1.5f));
    if(first){run.successes=run.collisions=run.timeouts=run.episodes=0;run.success_time=run.total_path=run.total_elapsed=run.final_progress=0;}
}
inline void sim_apply_challenge_world(device RLPhysicsState& state,device SimRun& run,device WWorld& world,
                                      device const WWorld* bank_worlds,device const uint* bank_schedule,
                                      constant ChallengeBankControl& bank_control,device uint* bank_active_ids,
                                      constant SimConfig& cfg,uint env,uint schedule_slot) {
    if(bank_control.enabled==0 || bank_control.bank_count==0 || env>=cfg.n ||
       schedule_slot>=bank_control.schedule_stride)return;
    const uint level_id=bank_schedule[env*bank_control.schedule_stride+schedule_slot];
    if(level_id==0xffffffffu) {
        // Fixed rehearsal slots retain the verified short-task distribution.
        // Alternate static rehearsal and moving-threat rehearsal families,
        // with clean sensing in this controlled bank experiment.
        uint rng=run.rng;
        const uint family=wtraining_family(env%2==0?12u:13u,rng);
        wgenerate(world,wrng(rng),family,4.0f);
        run.rng=rng;
    } else {
        if(level_id>=bank_control.bank_count)return;
        world=bank_worlds[level_id];
    }
    world.wind[0]=sim_wind_param(cfg,env);world.wind[1]=0;world.wind[2]=0;
    run.initial_distance=length(float3(world.goal[0],world.goal[1],world.goal[2]-1.5f));
    if(world.family==10 || world.family==11) {
        state.linear_velocity[0]=cfg.speed;
        run.desired_velocity[0]=cfg.speed;
        run.previous_nav[0]=1.0f;
    }
    bank_active_ids[env]=level_id;
}
kernel void sim_reset(device RLPhysicsState* states [[buffer(0)]],device SimRun* runs [[buffer(1)]],device WWorld* worlds [[buffer(2)]],device float* sensors [[buffer(3)]],device float* commands [[buffer(4)]],device const float* weights [[buffer(5)]],constant RLPhysicsParams& p [[buffer(6)]],constant SimConfig& cfg [[buffer(7)]],device float* poses [[buffer(8)]],device const WWorld* bank_worlds [[buffer(9)]],device const uint* bank_schedule [[buffer(10)]],constant ChallengeBankControl& bank_control [[buffer(11)]],device uint* bank_active_ids [[buffer(12)]],uint n [[thread_position_in_grid]]) {
    if(n<cfg.n){sim_reset_one(states[n],runs[n],worlds[n],sensors,commands,weights,p,cfg,n,true);if(bank_control.enabled!=0)sim_apply_challenge_world(states[n],runs[n],worlds[n],bank_worlds,bank_schedule,bank_control,bank_active_ids,cfg,n,0);else bank_active_ids[n]=0xffffffffu;for(uint j=0;j<96;j++)poses[n*96+j]=0;}
}
kernel void sim_depth(device const RLPhysicsState* states [[buffer(0)]],device SimRun* runs [[buffer(1)]],device const WWorld* worlds [[buffer(2)]],device float* sensors [[buffer(3)]],constant RLPhysicsParams& p [[buffer(4)]],constant SimConfig& cfg [[buffer(5)]],device float* poses [[buffer(6)]],uint i [[thread_position_in_grid]]) {
    uint n=i/320,k=i%320;if(n>=cfg.n || (cfg.eval && runs[n].episodes) || runs[n].steps%cfg.sensor_period!=0)return;
    RLPhysicsState s=states[n];float r[9];sim_rotation(s.orientation_wxyz,r);WVec ray=wcamera(k);
    WVec d=wv(r[0]*ray.x+r[1]*ray.y+r[2]*ray.z,r[3]*ray.x+r[4]*ray.y+r[5]*ray.z,r[6]*ray.x+r[7]*ray.y+r[8]*ray.z);
    float depth=wray(worlds[n],wv(s.position[0],s.position[1],s.position[2]),d,runs[n].elapsed);
    uint rng=runs[n].rng+k*1664525u+runs[n].steps*1013904223u;
    const float noise=sim_depth_noise(cfg,n),dropout=sim_depth_dropout(cfg,n);
    depth=clamp(depth+noise*sim_normal(rng),0.0f,12.0f);if(wurand(rng)<dropout)depth=12;
    uint frame=(runs[n].steps/cfg.sensor_period)%8;if(k==0){for(uint j=0;j<3;j++)poses[(n*8+frame)*12+j]=s.position[j];for(uint j=0;j<9;j++)poses[(n*8+frame)*12+3+j]=r[j];}sensors[(n*8+frame)*320+k]=depth;
}
inline void sim_critic_obs(thread const RLPhysicsState& s,device const WWorld& w,device const SimRun& run,thread float* obs) {
    for(uint j=0;j<32;j++)obs[j]=0;
    for(uint j=0;j<3;j++){obs[j]=(w.goal[j]-s.position[j])/10;obs[j+3]=s.linear_velocity[j]/4;obs[j+6]=s.angular_velocity_body[j]/4;obs[j+13]=s.position[j]/10;obs[j+16]=(run.reference_position[j]-s.position[j])*2;}
    for(uint j=0;j<4;j++)obs[9+j]=s.orientation_wxyz[j];
    obs[30]=w.wind[0];obs[31]=w.wind[1];
    obs[19]=wclearance(w,wv(s.position[0],s.position[1],s.position[2]),run.elapsed)/5;obs[20]=float(run.steps)/200;
    for(uint j=0;j<min(w.count,3u);j++){WVec c=wc(w.obstacles[j],run.elapsed);obs[21+j*3]=(c.x-s.position[0])/10;obs[22+j*3]=(c.y-s.position[1])/10;obs[23+j*3]=(c.z-s.position[2])/10;}
}
kernel void sim_observe(device const RLPhysicsState* states [[buffer(0)]],device const SimRun* runs [[buffer(1)]],device const WWorld* worlds [[buffer(2)]],device const float* sensors [[buffer(3)]],device float* obs [[buffer(4)]],device float* critic_obs [[buffer(5)]],constant RLPhysicsParams& p [[buffer(6)]],constant SimConfig& cfg [[buffer(7)]],device float* poses [[buffer(8)]],device const float* memory_clearances [[buffer(9)]],uint n [[thread_position_in_grid]]) {
    if(n>=cfg.n || (cfg.eval && runs[n].episodes))return;RLPhysicsState s=states[n];float r[9];sim_rotation(s.orientation_wxyz,r);
    const uint sensor_delay=sim_sensor_delay(cfg,n);
    uint available=runs[n].steps/cfg.sensor_period,frame=available>sensor_delay?available-sensor_delay:0,prev=frame>0?frame-1:0;
    uint row=(cfg.tick*cfg.n+n)*PPO_ACTOR_OBS,crow=(cfg.tick*cfg.n+n)*32;
    for(uint k=0;k<SIM_DEPTH_FEATURES;k++) {
        float current=12.0f,previous=12.0f;
        if(SIM_DEPTH_FEATURES==320) {
            current=sensors[(n*8+frame%8)*320+k];
            previous=sensors[(n*8+prev%8)*320+k];
        } else {
            const uint y=(k/10)*2,x=(k%10)*2;
            for(uint dy=0;dy<2;dy++)for(uint dx=0;dx<2;dx++) {
                const uint pixel=(y+dy)*20+x+dx;
                current=min(current,sensors[(n*8+frame%8)*320+pixel]);
                previous=min(previous,sensors[(n*8+prev%8)*320+pixel]);
            }
        }
        obs[row+k]=available>=sensor_delay?current/12.0f:1.0f;
        obs[row+SIM_DEPTH_FEATURES+k]=available>=sensor_delay?previous/12.0f:1.0f;
    }
    // Evaluation ablation: remove the actor's previous-depth channel.
    if(cfg.mode==18)for(uint k=0;k<SIM_DEPTH_FEATURES;k++)obs[row+SIM_DEPTH_FEATURES+k]=obs[row+k];
    float3 delta=float3(worlds[n].goal[0]-s.position[0],worlds[n].goal[1]-s.position[1],worlds[n].goal[2]-s.position[2]);float distance=max(length(delta),1e-6f);
    const uint context=row+2*SIM_DEPTH_FEATURES;
    for(uint j=0;j<3;j++) {
        obs[context+j]=(r[j]*delta.x+r[3+j]*delta.y+r[6+j]*delta.z)/distance;
        obs[context+4+j]=(r[j]*s.linear_velocity[0]+r[3+j]*s.linear_velocity[1]+r[6+j]*s.linear_velocity[2])/4;
        obs[context+7+j]=s.angular_velocity_body[j]/4;obs[context+10+j]=r[6+j];
    }
    obs[context+3]=min(distance/10,1.5f);for(uint j=0;j<4;j++)obs[context+13+j]=runs[n].previous_nav[j];
    obs[context+17]=float(runs[n].steps-frame*cfg.sensor_period)*p.dt*cfg.substeps;float ref[3]={runs[n].reference_position[0]-s.position[0],runs[n].reference_position[1]-s.position[1],runs[n].reference_position[2]-s.position[2]};for(uint j=0;j<3;j++)obs[context+18+j]=clamp((r[j]*ref[0]+r[3+j]*ref[1]+r[6+j]*ref[2])*2,-1.0f,1.0f);
    if(PPO_ACTOR_OBS==184){float cur[80],prev[80],goal[3],vel[3],hint[3];for(uint j=0;j<80;j++){cur[j]=obs[row+j]*12;prev[j]=obs[row+80+j]*12;}for(uint j=0;j<3;j++){goal[j]=obs[context+j];vel[j]=obs[context+4+j]*4;}if(cfg.geometry_memory){float pose[12];for(uint j=0;j<3;j++)pose[j]=s.position[j];for(uint j=0;j<9;j++)pose[j+3]=r[j];uint valid=available>=sensor_delay?min(frame+1,8-sensor_delay):0;nav_guidance_memory(cur,prev,goal,distance,vel,float(cfg.sensor_period)*p.dt*cfg.substeps,sensors+n*8*320,poses+n*8*12,pose,frame,valid,hint,memory_clearances+n*85);}else nav_guidance(cur,prev,goal,distance,vel,float(cfg.sensor_period)*p.dt*cfg.substeps,hint);for(uint j=0;j<3;j++)obs[row+PPO_ACTOR_OBS-3+j]=hint[j];}
    float co[32];sim_critic_obs(s,worlds[n],runs[n],co);for(uint j=0;j<32;j++)critic_obs[crow+j]=co[j];
}
kernel void sim_act(device RLPhysicsState* states [[buffer(0)]],device SimRun* runs [[buffer(1)]],device const WWorld* worlds [[buffer(2)]],device const float* observations [[buffer(3)]],device const float* critic_obs [[buffer(4)]],device const float* actor [[buffer(5)]],device const float* critic [[buffer(6)]],device float* actions [[buffer(7)]],device float* logp [[buffer(8)]],device float* values [[buffer(9)]],device float* commands [[buffer(10)]],constant RLPhysicsParams& p [[buffer(11)]],constant SimConfig& cfg [[buffer(12)]],uint n [[thread_position_in_grid]]) {
    if(n>=cfg.n || (cfg.eval && runs[n].episodes))return;uint row=cfg.tick*cfg.n+n;float co[32],hidden[64],mean[4],a[4];
    for(uint j=0;j<32;j++)co[j]=critic_obs[row*32+j];for(uint j=0;j<4;j++)mean[j]=actions[row*4+j];
    values[row]=ppo_critic_value(critic,co,hidden);
    if(PPO_ACTOR_OBS==184 && cfg.mode>=9 && cfg.mode<=21){float scale=(cfg.mode==9||cfg.mode==13)?0.0f:((cfg.mode==10||cfg.mode==14||cfg.mode==16)?0.25f:(cfg.mode==15?1.0f:0.5f));if(cfg.mode>=17){float overhead=0;for(uint k=0;k<20;k++)overhead+=observations[row*PPO_ACTOR_OBS+k]*12>2.5f;scale=.25f+.75f*(overhead/20.0f);}if(cfg.mode==12){float near=12;for(uint k=0;k<80;k++)near=min(near,observations[row*PPO_ACTOR_OBS+k]*12);scale=clamp((near-.3f)/2,0.2f,1.0f);}for(uint j=0;j<3;j++){float prior=observations[row*PPO_ACTOR_OBS+PPO_ACTOR_OBS-3+j];mean[j]=prior+scale*(mean[j]-prior);}mean[3]*=scale;if(cfg.mode==16){uint context=row*PPO_ACTOR_OBS+160;float yaw=atan2(observations[context+1],observations[context]);mean[3]+=nav_atanh(clamp(yaw*1.5f,-.85f,.85f));}}
    float deployed_scale=1.0f;
    if(cfg.mode==22 && PPO_ACTOR_OBS==184) {
        float overhead=0;
        for(uint k=0;k<20;k++)overhead+=observations[row*PPO_ACTOR_OBS+k]*12>2.5f;
        deployed_scale=.25f+.75f*(overhead/20.0f);
    }
    uint rng=runs[n].rng;float lp=0;for(uint j=0;j<4;j++) {
        a[j]=mean[j]+((cfg.mode==0||cfg.mode==22)?exp(actor[PPO_ACTOR_LOG_STD+j])*sim_normal(rng):0);
        if(cfg.mode==1)a[j]=sim_normal(rng);
        float diff=(a[j]-mean[j])*exp(-actor[PPO_ACTOR_LOG_STD+j]);lp+=-0.5f*(diff*diff+2*actor[PPO_ACTOR_LOG_STD+j]+PPO_LOG_TWO_PI);
        actions[row*4+j]=a[j];
        float executed_latent=a[j];
        if(cfg.mode==22 && PPO_ACTOR_OBS==184) {
            // Train the deployed mode17 action map. PPO still scores the raw
            // Gaussian sample; this observation-dependent map has no weights.
            if(j<3) {
                const float prior=observations[row*PPO_ACTOR_OBS+PPO_ACTOR_OBS-3+j];
                executed_latent=prior+deployed_scale*(a[j]-prior);
            } else executed_latent=deployed_scale*a[j];
        }
        commands[(n*8+runs[n].steps%8)*4+j]=tanh(executed_latent);
    }
    logp[row]=lp;runs[n].rng=rng;
    if(cfg.mode==2 || cfg.mode==3 || cfg.mode==20) {
        float3 d=float3(worlds[n].goal[0]-states[n].position[0],worlds[n].goal[1]-states[n].position[1],worlds[n].goal[2]-states[n].position[2]);d=normalize(d)*min(length(d),cfg.speed);
        RLPhysicsState s=states[n];float r[9];sim_rotation(s.orientation_wxyz,r);
        a[0]=(r[0]*d.x+r[3]*d.y+r[6]*d.z)/cfg.speed;a[1]=(r[1]*d.x+r[4]*d.y+r[7]*d.z)/cfg.speed;a[2]=(r[2]*d.x+r[5]*d.y+r[8]*d.z)/cfg.speed;a[3]=0;
        for(uint j=0;j<4;j++)commands[(n*8+runs[n].steps%8)*4+j]=a[j];
    }
    const uint command_delay=sim_command_delay(cfg,n);
    uint applied=runs[n].steps>command_delay?runs[n].steps-command_delay:0;
    for(uint j=0;j<4;j++)runs[n].previous_nav[j]=runs[n].steps<command_delay?0:commands[(n*8+applied%8)*4+j];
    RLPhysicsState s=states[n];float r[9];sim_rotation(s.orientation_wxyz,r);float v[3]={runs[n].previous_nav[0]*cfg.speed,runs[n].previous_nav[1]*cfg.speed,runs[n].previous_nav[2]*cfg.speed*(cfg.velocity_contract==0?0.5f:1.0f)};
    if(cfg.velocity_contract==1){float scale=min(1.0f,cfg.speed/max(length(float3(v[0],v[1],v[2])),1e-8f));for(uint j=0;j<3;j++)v[j]*=scale;}
    // Fixed-speed eval ablation: scale the already-converted executed intent only.
    if(cfg.mode==19){float magnitude=length(float3(v[0],v[1],v[2]));if(magnitude>1e-8f){float scale=cfg.speed/magnitude;for(uint j=0;j<3;j++)v[j]*=scale;}}
    for(uint j=0;j<3;j++)runs[n].desired_velocity[j]=r[j*3]*v[0]+r[j*3+1]*v[1]+r[j*3+2]*v[2];runs[n].yaw+=runs[n].previous_nav[3]*p.dt*cfg.substeps*0.5f;
}
kernel void sim_advance(device RLPhysicsState* states [[buffer(0)]],device SimRun* runs [[buffer(1)]],device WWorld* worlds [[buffer(2)]],device float* sensors [[buffer(3)]],device float* commands [[buffer(4)]],device const float* weights [[buffer(5)]],device const float* critic [[buffer(6)]],device float* rewards [[buffer(7)]],device float* next_values [[buffer(8)]],device uchar* terminated [[buffer(9)]],device uchar* truncated [[buffer(10)]],constant RLPhysicsParams& p [[buffer(11)]],constant SimConfig& cfg [[buffer(12)]],device const WWorld* bank_worlds [[buffer(13)]],device const uint* bank_schedule [[buffer(14)]],constant ChallengeBankControl& bank_control [[buffer(15)]],device uint* bank_active_ids [[buffer(16)]],device uint* bank_transition_ids [[buffer(17)]],uint n [[thread_position_in_grid]]) {
    if(n>=cfg.n || (cfg.eval && runs[n].episodes))return;uint row=cfg.tick*cfg.n+n;RLPhysicsState s=states[n];float h[16],motors[4];for(uint j=0;j<16;j++)h[j]=runs[n].hidden[j];for(uint j=0;j<4;j++)motors[j]=runs[n].motors[j];
    float3 goal=float3(worlds[n].goal[0],worlds[n].goal[1],worlds[n].goal[2]);float before=length(goal-float3(s.position[0],s.position[1],s.position[2]));float yaw=runs[n].yaw,c=cos(yaw),si=sin(yaw);bool collision=false;float step_clearance=12;
    float wind[3]={worlds[n].wind[0]*p.mass,worlds[n].wind[1]*p.mass,worlds[n].wind[2]*p.mass};
    for(uint step=0;step<cfg.substeps;step++) {
        float obs[22],v[3];for(uint j=0;j<3;j++)v[j]=runs[n].desired_velocity[j];
        for(uint j=0;j<3;j++)runs[n].reference_position[j]+=v[j]*p.dt;
        float dp[3]={s.position[0]-runs[n].reference_position[0],s.position[1]-runs[n].reference_position[1],s.position[2]-runs[n].reference_position[2]},dv[3]={s.linear_velocity[0]-v[0],s.linear_velocity[1]-v[1],s.linear_velocity[2]-v[2]};
        obs[0]=clamp(c*dp[0]+si*dp[1],-.5f,.5f);obs[1]=clamp(-si*dp[0]+c*dp[1],-.5f,.5f);obs[2]=clamp(dp[2],-.5f,.5f);
        float ch=cos(yaw*.5f),sh=sin(yaw*.5f),q[4]={ch*s.orientation_wxyz[0]+sh*s.orientation_wxyz[3],ch*s.orientation_wxyz[1]+sh*s.orientation_wxyz[2],ch*s.orientation_wxyz[2]-sh*s.orientation_wxyz[1],ch*s.orientation_wxyz[3]-sh*s.orientation_wxyz[0]};
        float r[9];sim_rotation(q,r);for(uint j=0;j<9;j++)obs[3+j]=r[j];obs[12]=clamp(c*dv[0]+si*dv[1],-1.0f,1.0f);obs[13]=clamp(-si*dv[0]+c*dv[1],-1.0f,1.0f);obs[14]=clamp(dv[2],-1.0f,1.0f);
        for(uint j=0;j<3;j++)obs[15+j]=s.angular_velocity_body[j];for(uint j=0;j<4;j++)obs[18+j]=motors[j];
        raptor_forward(weights,obs,h,motors);raptor_clip_action(motors);RLPhysicsState next;rl_physics_step(s,motors,wind,p,next);
        bool valid=true;for(uint j=0;j<3;j++)valid=valid&&isfinite(next.position[j])&&isfinite(next.linear_velocity[j])&&isfinite(next.angular_velocity_body[j]);for(uint j=0;j<4;j++)valid=valid&&isfinite(next.orientation_wxyz[j])&&isfinite(next.rpm[j]);if(!valid){collision=true;break;}
        runs[n].path+=distance(float3(s.position[0],s.position[1],s.position[2]),float3(next.position[0],next.position[1],next.position[2]));s=next;runs[n].elapsed+=p.dt;
        float clearance=wclearance(worlds[n],wv(s.position[0],s.position[1],s.position[2]),runs[n].elapsed);step_clearance=min(step_clearance,clearance);runs[n].min_clearance=min(runs[n].min_clearance,clearance);runs[n].peak_speed=max(runs[n].peak_speed,length(float3(s.linear_velocity[0],s.linear_velocity[1],s.linear_velocity[2])));
        if(clearance<=0 || !isfinite(s.position[0]) || !isfinite(s.position[1]) || !isfinite(s.position[2])) {collision=true;break;}
    }
    runs[n].steps++;states[n]=s;for(uint j=0;j<16;j++)runs[n].hidden[j]=h[j];for(uint j=0;j<4;j++)runs[n].motors[j]=motors[j];
    float after=length(goal-float3(s.position[0],s.position[1],s.position[2]));bool success=after<(cfg.mode==20?.10f:.35f)&&!collision,timeout=runs[n].steps>=cfg.max_steps;
    rewards[row]=(before-after)*2-0.01f-cfg.risk_coef*clamp((0.6f-step_clearance)/0.6f,0.0f,1.0f)+(success?10.0f:0)-(collision?10.0f:0);terminated[row]=collision||success;truncated[row]=timeout&&!terminated[row];
    bank_transition_ids[row]=bank_control.enabled!=0?bank_active_ids[n]:0xffffffffu;
    float co[32],hidden[64];sim_critic_obs(s,worlds[n],runs[n],co);next_values[row]=ppo_critic_value(critic,co,hidden);
    if(success||collision||timeout) {
        runs[n].successes+=success;runs[n].collisions+=collision;runs[n].timeouts+=timeout&&!terminated[row];runs[n].episodes++;runs[n].success_time+=success?runs[n].elapsed:0;runs[n].total_path+=runs[n].path;runs[n].total_elapsed+=runs[n].elapsed;runs[n].final_progress+=1-after/max(runs[n].initial_distance,1e-4f);
        if(!cfg.eval){sim_reset_one(states[n],runs[n],worlds[n],sensors,commands,weights,p,cfg,n,false);if(bank_control.enabled!=0)sim_apply_challenge_world(states[n],runs[n],worlds[n],bank_worlds,bank_schedule,bank_control,bank_active_ids,cfg,n,cfg.tick+1);else bank_active_ids[n]=0xffffffffu;}
    }
}
