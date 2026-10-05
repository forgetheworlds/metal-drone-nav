// Frozen-policy stress probes on exact bank geometry. No training or retiming.
#define WAYPOINT_EMBEDDED
#include "navigation_critic_training.mm"


namespace stress_evaluation {
// Validate the exact plant handed to Metal, rather than trusting the requested
// sampler amplitude. Save those same bytes next to each flight table.
void validate_plant(const RLPhysicsParams& plant, bool varied) {
    const auto nominal = rl_physics_crazyflie_default();
    float values[88];
    std::memcpy(values, &plant, sizeof(plant));
    for (float value : values) require(std::isfinite(value), "non-finite sampled plant");
    if (!varied) {
        require(std::memcmp(&plant, &nominal, sizeof(plant)) == 0, "nominal plant changed");
        return;
    }
    const auto range = rl_physics_domain_stress_range();
    auto check_scale = [](float scale, float low, float high) {
        require(scale >= low - 1e-5f && scale <= high + 1e-5f,
                "sampled plant outside declared bounds");
    };
    const float mass_scale = plant.mass / nominal.mass;
    check_scale(mass_scale, range.mass_scale_min, range.mass_scale_max);
    RLPhysicsDomainSample sample{};
    sample.mass_scale = mass_scale;
    sample.thrust_to_weight = rl_physics_domain_thrust_to_weight(plant);
    for (int axis = 0; axis < 3; ++axis) {
        // The sampler scales inertia with mass as well as its independent axis draw.
        const int index = 4 * axis;
        check_scale(plant.inertia[index] / (nominal.inertia[index] * mass_scale),
                    range.inertia_axis_scale_min, range.inertia_axis_scale_max);
    }
    for (int rotor = 0; rotor < 4; ++rotor) {
        const float thrust_gain = plant.rotor_thrust_coefficients[3 * rotor + 2] /
                                  nominal.rotor_thrust_coefficients[3 * rotor + 2];
        check_scale(thrust_gain, range.thrust_gain_min, range.thrust_gain_max);
        for (int coefficient = 0; coefficient < 3; ++coefficient) {
            const int index = 3 * rotor + coefficient;
            require(std::fabs(plant.rotor_thrust_coefficients[index] -
                              nominal.rotor_thrust_coefficients[index] * thrust_gain) < 1e-5f,
                    "rotor thrust polynomial changed inconsistently");
        }
        check_scale(plant.rotor_time_constants_rising[rotor] /
                    nominal.rotor_time_constants_rising[rotor],
                    range.rising_lag_scale_min, range.rising_lag_scale_max);
        check_scale(plant.rotor_time_constants_falling[rotor] /
                    nominal.rotor_time_constants_falling[rotor],
                    range.falling_lag_scale_min, range.falling_lag_scale_max);
    }
    require(rl_physics_domain_validate(plant, sample, range), "invalid varied plant");
}

std::string shader_source() {
    return base_source() + PPO_TRAINER_MSL + waypoint::kWaypointKernels;
}

int check_capture_age() {
    Metal metal;
    metal.compile(shader_source() + R"MSL(
    kernel void check_capture_age(device float* out [[buffer(0)]], uint i [[thread_position_in_grid]]) {
        if (i < 3) out[i] = nav_memory_point_radius(2.0f, .05f, float(i) * .1f);
    }
    )MSL");
    auto output = metal.buffer(3 * sizeof(float));
    auto commands = [metal.queue commandBuffer];
    metal.dispatch(commands, metal.pipeline("check_capture_age"), 3, {output}, 3);
    metal.finish(commands);
    const auto* actual = static_cast<const float*>(output.contents);
    for (uint i = 0; i < 3; ++i) {
        const float expected = nav_memory_point_radius(2.0f, .05f, float(i) * .1f);
        require(std::fabs(actual[i] - expected) < 1e-7f, "capture-age host/Metal mismatch");
    }
    const float expected_delta = NAV_MEMORY_USE_CAPTURE_AGE ? .015f : 0.0f;
    require(std::fabs(actual[1] - actual[0] - expected_delta) < 1e-7f,
            "capture delay absent or double counted in point uncertainty");
    std::cout << "capture_age_check PASS enabled=" << NAV_MEMORY_USE_CAPTURE_AGE
              << " radius_delta_100ms=" << actual[1] - actual[0] << '\n';
    return 0;
}

// Diagnostic capture reads stored rollout buffers after execution. It cannot
// change actor observations, commands, physics or grading.
void write_trace(const std::string& path, const Sim& sim) {
    require(sim.horizon == 400, "full trace requires 400 stored navigation steps");
    std::ofstream output(path);
    require(bool(output), "cannot write stress trace");
    output << "env,step,time_s,success,collision,timeout,goal_distance_m,clearance_m,depth_age_s,min_depth_m,"
              "body_vx,body_vy,body_vz,requested_vx,requested_vy,requested_vz,"
              "applied_vx,applied_vy,applied_vz,command_gap_mps,position_x,position_y,position_z\n";
    output << std::setprecision(9);
    const auto* results = static_cast<const SimRun*>(sim.runs.contents);
    const auto* observations = static_cast<const float*>(sim.obs.contents);
    const auto* critic = static_cast<const float*>(sim.co.contents);
    const auto* actions = static_cast<const float*>(sim.actions.contents);
    const float period = sim.cfg.substeps * .01f;
    for (uint env = 0; env < sim.cfg.n; ++env) {
        const uint steps = std::min(results[env].steps, 400u);
        for (uint step = 0; step < steps; ++step) {
            const size_t row = size_t(step) * sim.cfg.n + env;
            const float* actor = observations + row * fixed_ppo::actor_obs_dim;
            const float* value = critic + row * fixed_ppo::critic_obs_dim;
            const float* context = actor + fixed_ppo::context_offset;
            auto command = [&](uint issued_step, float* velocity) {
                float squared = 0;
                for (uint axis = 0; axis < 3; ++axis) {
                    velocity[axis] = std::tanh(actions[(size_t(issued_step) * sim.cfg.n + env) * 4 + axis]);
                    squared += velocity[axis] * velocity[axis];
                }
                const float scale = sim.cfg.speed / std::max(1.0f, std::sqrt(squared));
                for (uint axis = 0; axis < 3; ++axis) velocity[axis] *= scale;
            };
            float requested[3], applied[3] = {0, 0, 0};
            command(step, requested);
            if (step >= sim.cfg.command_delay) command(step - sim.cfg.command_delay, applied);
            float gap = 0, nearest_depth = 12;
            for (uint axis = 0; axis < 3; ++axis)
                gap += (requested[axis] - applied[axis]) * (requested[axis] - applied[axis]);
            for (uint pixel = 0; pixel < 80; ++pixel)
                nearest_depth = std::min(nearest_depth, actor[pixel] * 12);
            output << env << ',' << step << ',' << step * period << ','
                   << results[env].successes << ',' << results[env].collisions << ',' << results[env].timeouts
                   << ',' << context[3] * 10 << ',' << value[19] * 5 << ',' << context[17]
                   << ',' << nearest_depth;
            for (uint axis = 0; axis < 3; ++axis) output << ',' << context[4 + axis] * 4;
            for (float velocity : requested) output << ',' << velocity;
            for (float velocity : applied) output << ',' << velocity;
            output << ',' << std::sqrt(gap);
            for (uint axis = 0; axis < 3; ++axis) output << ',' << value[13 + axis] * 10;
            output << '\n';
        }
    }
    require(bool(output), "stress trace write failed");
}

} // namespace stress_evaluation

int main(int argc,char** argv) {@autoreleasepool {try {
    if(argc==2 && std::string(argv[1])=="capture-age-check") return stress_evaluation::check_capture_age();
    require(argc==5||argc==6,"stress_eval CHECKPOINT BANK OUT_CSV PROFILE [TRACE_CSV]");
    const std::string checkpoint=argv[1],bank_path=argv[2],output=argv[3],profile=argv[4];
    waypoint::BankControl control{};std::string bank_hash;const auto bank=waypoint::read_bank(bank_path,control,bank_hash);
    require(control.period==1,"stress evaluation requires one task per environment");
    SimConfig cfg;cfg.n=control.count;cfg.eval=1;cfg.mode=17;cfg.seed=700001;cfg.speed=1.5f;cfg.max_steps=400;cfg.geometry_memory=1;
    float amplitude=0;
    if(profile=="nominal"){}
    else if(profile=="depth-noise")cfg.depth_noise=.03f;
    else if(profile=="dropout")cfg.dropout=.05f;
    else if(profile=="sensor-delay")cfg.sensor_delay=2;
    else if(profile=="command-delay")cfg.command_delay=2;
    else if(profile=="both-delay"){cfg.sensor_delay=2;cfg.command_delay=2;}
    else if(profile=="dynamics")amplitude=1;
    else if(profile=="combined"){
        cfg.depth_noise=.03f;cfg.dropout=.05f;cfg.sensor_delay=2;cfg.command_delay=2;amplitude=1;
    }else throw std::runtime_error("unknown stress profile");
    Metal metal;metal.compile(stress_evaluation::shader_source());
    Sim sim(metal,cfg,argc==6?400:32);
    // Install the nominal task recipe before explicitly enabling evaluation-only
    // domain variation. The shared training constructor retains its safety guard.
    auto run=waypoint::make_local_run(sim,bank,control,metal.pipeline("waypoint_task_apply"));
    if(amplitude>0) {
        NavigationRuntimeConfig runtime{};runtime.enabled=1;runtime.domain_amplitude=amplitude;
        runtime.domain_seed=0x3f84d5b5u;runtime.domain_range=rl_physics_domain_stress_range();
        std::memcpy(sim.runtime_control.contents,&runtime,sizeof(runtime));sim.reset();
    }
    navigation_training::load_actor(sim,checkpoint,false);
    auto probe=[metal.queue commandBuffer];waypoint::local_probe(run,probe);metal.finish(probe);
    waypoint::verify_reset(run,bank,control,cfg.sensor_delay==0&&cfg.depth_noise==0&&cfg.dropout==0);
    const auto* params=static_cast<const RLPhysicsParams*>(sim.environment_physics.contents);
    std::ofstream raw_physics(output+".physics.bin",std::ios::binary);
    require(bool(raw_physics),"cannot write exact sampled plant parameters");
    raw_physics.write(reinterpret_cast<const char*>(params),sim.environment_physics.length);
    require(bool(raw_physics),"sampled plant write failed");
    std::ofstream dynamics(output+".physics.csv");require(bool(dynamics),"cannot write sampled plant parameters");
    dynamics<<"env,mass,inertia_x,inertia_y,inertia_z\n";
    for(uint env=0;env<cfg.n;env++) {
        stress_evaluation::validate_plant(params[env], amplitude > 0);
        dynamics<<env<<','<<params[env].mass<<','<<params[env].inertia[0]<<','<<params[env].inertia[4]<<','<<params[env].inertia[8]<<'\n';
    }
    auto commands=[metal.queue commandBuffer];waypoint::local_collect(run,commands,400);metal.finish(commands);
    const auto* results=static_cast<const SimRun*>(sim.runs.contents);
    uint success=0,collision=0,timeout=0;
    for(uint env=0;env<cfg.n;env++){
        require(results[env].episodes==1,"stress eval must finish exactly one episode");
        require(results[env].successes+results[env].collisions+results[env].timeouts==1,"stress outcome must be exhaustive and exclusive");
        require(std::isfinite(results[env].elapsed)&&std::isfinite(results[env].path),"stress flight measure non-finite");
        success+=results[env].successes;collision+=results[env].collisions;timeout+=results[env].timeouts;
    }
    waypoint::write_eval_csv(output,profile,nullptr,bank,control,sim,700001,17);
    if(argc==6) stress_evaluation::write_trace(argv[5],sim);
    std::cout<<"stress="<<profile<<" tasks="<<cfg.n<<" success="<<success<<" contact="<<collision<<" timeout="<<timeout
             <<" noise_m="<<cfg.depth_noise<<" dropout="<<cfg.dropout<<" sensor_delay_ms="<<cfg.sensor_delay*50
             <<" command_delay_ms="<<cfg.command_delay*50<<" domain_amplitude="<<amplitude
             <<" ego_state=ideal wind=none bank_sha256="<<bank_hash<<'\n';
    return 0;
}catch(const std::exception& e){std::cerr<<"ERROR: "<<e.what()<<'\n';return 1;}}}
