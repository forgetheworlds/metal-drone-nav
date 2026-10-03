// PPO on the original challenge bank with valid, translated TRAIN corner starts.
// The actor sees only its normal goal/depth/ego/history observation. The
// witness route selects TRAIN reset locations and never changes the goal.
#define main embedded_navigation_main
#include "main.mm"
#undef main

namespace navigation_curriculum {

constexpr uint32_t kEnvironmentCount = 128;
constexpr uint32_t kHorizon = 32;
constexpr uint32_t kRolloutsPerStage = 100;
constexpr uint32_t kStages = 4;
constexpr uint32_t kFamilyCorner = 14;
constexpr uint32_t kFamilyRooms = 15;
constexpr uint32_t kFamilyVertical = 16;
constexpr float kSpawnX = 0.0f;
constexpr float kSpawnY = 0.0f;
constexpr float kSpawnZ = 1.5f;

struct CurriculumStage {
    uint32_t route_node;
    const char* name;
};

// The route has six verified points. These starts expose the final corner,
// then the preceding hall, then the first turn, then the original start.
constexpr CurriculumStage kCurriculumStages[kStages] = {
    {3, "after_second_corner"},
    {2, "mid_hall"},
    {1, "after_first_turn"},
    {0, "original_start"},
};

static std::array<float, 3> transform_route_point(
        const std::array<float, 3>& point, const std::array<float, 3>& anchor,
        int quarter_turn) {
    const float x = point[0] - anchor[0];
    const float y = point[1] - anchor[1];
    float transformed_x = x, transformed_y = y;
    if (quarter_turn == 1) { transformed_x = -y; transformed_y = x; }
    if (quarter_turn == -1) { transformed_x = y; transformed_y = -x; }
    return {kSpawnX + transformed_x, kSpawnY + transformed_y,
            kSpawnZ + point[2] - anchor[2]};
}

static int align_next_leg_with_room_length(
        const std::vector<std::array<float, 3>>& route, uint32_t route_node) {
    if (route_node == 0) return 0;
    float largest_y_leg = 0.0f;
    float selected_y_delta = 0.0f;
    for (size_t index = route_node; index + 1 < route.size(); index++) {
        const float dx = route[index + 1][0] - route[index][0];
        const float dy = route[index + 1][1] - route[index][1];
        if (std::fabs(dy) > std::fabs(dx) && std::fabs(dy) > largest_y_leg) {
            largest_y_leg = std::fabs(dy);
            selected_y_delta = dy;
        }
    }
    if (largest_y_leg == 0.0f) return 0;
    // Quarter turns preserve the axis-aligned box geometry exactly. Put the
    // long north/south route leg along the simulator's long x room axis.
    return selected_y_delta > 0.0f ? -1 : 1;
}

static void transform_world_to_spawn(WWorld& world,
                                     const std::array<float, 3>& anchor,
                                     int quarter_turn) {
    for (uint32_t obstacle_index = 0; obstacle_index < world.count; obstacle_index++) {
        auto& obstacle = world.obstacles[obstacle_index];
        require(obstacle.kind == 0 && obstacle.velocity[0] == 0.0f &&
                obstacle.velocity[1] == 0.0f && obstacle.velocity[2] == 0.0f,
                "corner curriculum expects static axis-aligned wall boxes");
        const auto center = transform_route_point(
            {obstacle.center[0], obstacle.center[1], obstacle.center[2]}, anchor, quarter_turn);
        obstacle.center[0] = center[0];
        obstacle.center[1] = center[1];
        obstacle.center[2] = center[2];
        if (quarter_turn != 0) std::swap(obstacle.size[0], obstacle.size[1]);
    }
    const auto goal = transform_route_point({world.goal[0], world.goal[1], world.goal[2]},
                                            anchor, quarter_turn);
    for (uint32_t axis = 0; axis < 3; axis++) world.goal[axis] = goal[axis];
}

static std::vector<WWorld> build_stage_worlds(
        const std::vector<challenge_evaluation::Level>& levels,
        const std::vector<WWorld>& source_worlds,
        const CurriculumStage& stage) {
    require(levels.size() == source_worlds.size(), "curriculum level/world count mismatch");
    std::vector<WWorld> worlds = source_worlds;
    uint32_t shifted_corner_levels = 0;
    for (size_t level_id = 0; level_id < levels.size(); level_id++) {
        const auto& level = levels[level_id];
        require(level.split == "train", "curriculum may transform TRAIN levels only");
        if (level.family != kFamilyCorner) continue;
        require(level.witness_route.size() == 6, "corner witness route must have six points");
        require(stage.route_node < level.witness_route.size(), "curriculum node outside witness route");
        const WWorld original_world = source_worlds[level_id];
        const int preferred_turn = align_next_leg_with_room_length(
            level.witness_route, stage.route_node);
        bool found_valid_start = false;
        uint32_t selected_node = stage.route_node;
        int selected_turn = 0;
        for (uint32_t route_node = stage.route_node;
             route_node < level.witness_route.size() && !found_valid_start; route_node++) {
            // Stage one was a matched all-level quarter-turn course. Later
            // stages can try other exact transforms when room bounds require it.
            std::array<int, 3> turns{{0, preferred_turn, -preferred_turn}};
            size_t turn_count = preferred_turn == 0 ? 1 : 3;
            if (stage.route_node == 3) {
                turns[0] = preferred_turn;
                turn_count = 1;
            }
            for (size_t turn_index = 0; turn_index < turn_count && !found_valid_start; turn_index++) {
                const int quarter_turn = turns[turn_index];
                WWorld candidate = original_world;
                if (route_node > 0)
                    transform_world_to_spawn(candidate, level.witness_route[route_node], quarter_turn);
                bool valid = wclearance(candidate, wv(kSpawnX, kSpawnY, kSpawnZ), 0.0f) > 0.04f;
                for (size_t segment = route_node; segment + 1 < level.witness_route.size() && valid; segment++) {
                    constexpr uint32_t samples = 20;
                    for (uint32_t sample = 0; sample <= samples; sample++) {
                        const float fraction = float(sample) / samples;
                        std::array<float, 3> point{};
                        for (uint32_t axis = 0; axis < 3; axis++)
                            point[axis] = level.witness_route[segment][axis] + fraction *
                                (level.witness_route[segment + 1][axis] - level.witness_route[segment][axis]);
                        const auto transformed = transform_route_point(
                            point, level.witness_route[route_node], quarter_turn);
                        if (wclearance(candidate, wv(transformed[0], transformed[1], transformed[2]), 0.0f) <= 0.04f) {
                            valid = false;
                            break;
                        }
                    }
                }
                const auto goal = transform_route_point(
                    {original_world.goal[0], original_world.goal[1], original_world.goal[2]},
                    level.witness_route[route_node], quarter_turn);
                valid = valid && wclearance(candidate, wv(goal[0], goal[1], goal[2]), 0.0f) > 0.04f;
                if (valid) {
                    worlds[level_id] = candidate;
                    selected_node = route_node;
                    selected_turn = quarter_turn;
                    found_valid_start = true;
                }
            }
        }
        require(found_valid_start,
                "no feasible translated corner start through the final goal: " + level.failure_id +
                " requested_node=" + std::to_string(stage.route_node));
        if (selected_node != stage.route_node || selected_turn != 0)
            std::cout << "curriculum_start_adjustment stage=" << stage.name
                      << " scene=" << level.failure_id << " requested_node=" << stage.route_node
                      << " selected_node=" << selected_node
                      << " quarter_turn=" << selected_turn << '\n';
        shifted_corner_levels++;
    }
    require(shifted_corner_levels == 30, "curriculum expects all 30 TRAIN corner levels");
    return worlds;
}

static std::array<uint64_t, 3> family_counts(const challenge_evaluation::BankScore& score) {
    std::array<uint64_t, 3> successes{};
    for (size_t family = 0; family < successes.size(); family++)
        successes[family] = score.family_successes[family];
    return successes;
}

static void load_parameter_warmstart(Sim& sim, const std::string& path) {
    std::ifstream source(path, std::ios::binary);
    require(bool(source), "cannot open curriculum warmstart: " + path);
    const auto header = read_checkpoint_header(source);
    require(header.actor_count == fixed_ppo::actor_param_count &&
            header.critic_count == fixed_ppo::critic_param_count,
            "curriculum warmstart dimensions do not match this actor build");
    source.read(static_cast<char*>(sim.actor.contents), sim.actor.length);
    source.read(static_cast<char*>(sim.critic.contents), sim.critic.length);
    require(bool(source), "curriculum warmstart parameters are truncated");
    std::cout << "parameter_warmstart=" << path
              << " actor_features=" << fixed_ppo::actor_obs_dim
              << " optimizer_state=reset\n";
}

static void validate_reset_state(const Sim& sim, const std::vector<WWorld>& worlds,
                                 const std::vector<uint32_t>& schedule) {
    require(schedule.size() == size_t(kEnvironmentCount) * (kHorizon + 1),
            "curriculum reset schedule has wrong size");
    const auto* states = static_cast<const RLPhysicsState*>(sim.states.contents);
    const auto* runs = static_cast<const SimRun*>(sim.runs.contents);
    const auto* active_ids = static_cast<const uint32_t*>(sim.bank_active_ids.contents);
    const auto* actual_worlds = static_cast<const WWorld*>(sim.worlds.contents);
    const auto* sensors = static_cast<const float*>(sim.sensors.contents);
    uint32_t corner_slots = 0, room_slots = 0, vertical_slots = 0;
    for (uint32_t env = 0; env < kEnvironmentCount; env++) {
        const uint32_t level_id = active_ids[env];
        require(level_id < worlds.size(), "reset schedule selected an invalid level");
        require(std::memcmp(&actual_worlds[env], &worlds[level_id], offsetof(WWorld, wind)) == 0 &&
                std::fabs(actual_worlds[env].wind[0] - sim.cfg.wind) < 1.0e-6f &&
                std::fabs(actual_worlds[env].wind[1]) < 1.0e-6f &&
                std::fabs(actual_worlds[env].wind[2]) < 1.0e-6f,
                "simulator reset did not load the selected stage world");
        const uint32_t family = actual_worlds[env].family;
        corner_slots += family == kFamilyCorner;
        room_slots += family == kFamilyRooms;
        vertical_slots += family == kFamilyVertical;
        require(std::fabs(states[env].position[0] - kSpawnX) < 1.0e-6f &&
                std::fabs(states[env].position[1] - kSpawnY) < 1.0e-6f &&
                std::fabs(states[env].position[2] - kSpawnZ) < 1.0e-6f,
                "simulator did not start at the normal physical spawn");
        require(std::fabs(runs[env].reference_position[0] - kSpawnX) < 1.0e-6f &&
                std::fabs(runs[env].reference_position[1] - kSpawnY) < 1.0e-6f &&
                std::fabs(runs[env].reference_position[2] - kSpawnZ) < 1.0e-6f,
                "RAPTOR reference state does not match the physical spawn");
        require(std::fabs(states[env].orientation_wxyz[0] - 1.0f) < 1.0e-6f &&
                std::fabs(states[env].orientation_wxyz[1]) < 1.0e-6f &&
                std::fabs(states[env].orientation_wxyz[2]) < 1.0e-6f &&
                std::fabs(states[env].orientation_wxyz[3]) < 1.0e-6f,
                "curriculum changed the native reset orientation");
        require(std::isfinite(states[env].rpm[0]) && states[env].rpm[0] > 0.0f,
                "curriculum reset did not initialize rotor speed");
        for (uint32_t ray = 0; ray < 8 * 320; ray++)
            require(sensors[size_t(env) * 8 * 320 + ray] == 12.0f,
                    "curriculum reset did not initialize the depth ring");
    }
    require(corner_slots >= 40 && room_slots >= 40 && vertical_slots >= 40,
            "initial schedule must sample all three TRAIN families");
    std::cout << "reset_invariants PASS family_slots=" << corner_slots << '/'
              << room_slots << '/' << vertical_slots << " physical_spawn=clear"
              << " orientation=identity rotors=initialized depth_ring=12m\n";
}

static void train(const std::string& bank_path, const std::string& warmstart,
                  const std::string& output_dir, uint32_t rollouts_per_stage,
                  uint32_t first_stage = 0, const std::string& resume_checkpoint = "") {
    require(fixed_ppo::actor_obs_dim == 824, "PPO corner curriculum requires the verified raw-depth actor");
    require(rollouts_per_stage >= 50 && rollouts_per_stage <= 500,
            "rollouts per stage must be 50..500");
    require(first_stage < kStages && uint64_t(rollouts_per_stage) * (kStages - first_stage) <= 8000,
            "curriculum PPO rollout budget exceeds 8000");

    Metal metal;
    metal.compile(base_source() + PPO_TRAINER_MSL);
    challenge_training::Settings settings;
    settings.environment_count = kEnvironmentCount;
    settings.horizon = kHorizon;
    settings.sampler_seed = 42;
    settings.selection = challenge_training::SelectionMode::UniformBank;
    settings.rehearsal_environments = 0;
    settings.focus_family = 0;
    const std::string expected_world_hash = challenge_evaluation::sha256_file(
        std::string(SOURCE_DIR) + "/world.hpp");
    challenge_training::ChallengeTraining sampler(bank_path, expected_world_hash, settings);
    const auto& levels = sampler.levels();
    const auto& source_worlds = sampler.worlds();
    require(challenge_evaluation::sha256_file(bank_path) ==
            "e2170fbe8e8c2d074ffb08de4269175a6de0bdcd410c3b60bd699ee7b5fc491e",
            "curriculum bank hash differs from the approved mirrored bank");

    SimConfig config;
    config.n = kEnvironmentCount;
    config.family = kFamilyCorner;
    config.seed = 42;
    config.mode = 22;
    config.distance = 8.0f;
    config.speed = 1.5f;
    config.max_steps = 400;
    config.geometry_memory = 1;
    config.risk_coef = 0.1f;
    config.entropy_coef = 0.0005f;
    config.learning_rate = 0.0001f;
    Sim sim(metal, config, kHorizon);
    const auto control = sampler.control();
    std::memcpy(sim.bank_control.contents, &control, sizeof(control));
    const auto initial_schedule = sampler.initial_schedule();
    std::memcpy(sim.bank_schedule.contents, initial_schedule.data(),
                initial_schedule.size() * sizeof(uint32_t));

    const auto stage0_worlds = build_stage_worlds(levels, source_worlds, kCurriculumStages[0]);
    sim.bank_worlds = metal.buffer(stage0_worlds.size() * sizeof(WWorld), stage0_worlds.data());
    sim.reset();
    validate_reset_state(sim, stage0_worlds, initial_schedule);
    PPOTrainer trainer(sim, 2);
    if (resume_checkpoint.empty()) {
        load_parameter_warmstart(sim, warmstart);
    } else {
        trainer.load_checkpoint(resume_checkpoint, kFamilyCorner, kHorizon,
                                kEnvironmentCount, 42);
        require(trainer.completed_rollouts == first_stage * rollouts_per_stage,
                "resume checkpoint rollout count does not match the next curriculum stage");
        std::cout << "ppo_resume_checkpoint=" << resume_checkpoint
                  << " completed_rollouts=" << trainer.completed_rollouts
                  << " optimizer_step=" << trainer.optimizer_step << '\n';
    }

    std::filesystem::create_directories(output_dir);
    const bool append_history = !resume_checkpoint.empty();
    std::ofstream history(output_dir + "/stages.csv",
                          append_history ? std::ios::app : std::ios::trunc);
    require(bool(history), "cannot write curriculum stage history");
    if (!append_history)
        history << "stage,name,route_node,rollouts,transitions,train_success,train_collision,train_timeout,train_mean_speed_mps,train_min_clearance_m,dev_corner,dev_rooms,dev_vertical,dev_total,dev_collision,dev_timeout,checkpoint_sha256,dev_csv_sha256\n";
    std::ofstream rollout_log(output_dir + "/rollouts.csv",
                              append_history ? std::ios::app : std::ios::trunc);
    require(bool(rollout_log), "cannot write curriculum rollout history");
    if (!append_history)
        rollout_log << "stage,rollout,transitions,gpu_s,episodes,success,collision,timeout,mean_speed_mps,min_clearance_m,policy_loss,value_loss,entropy_loss,ratio\n";

    const std::string source_sha = challenge_evaluation::sha256_file(bank_path);
    const double mission_started = seconds();
    for (uint32_t stage_index = first_stage; stage_index < kStages; stage_index++) {
        const auto& stage = kCurriculumStages[stage_index];
        const auto stage_worlds = build_stage_worlds(levels, source_worlds, stage);
        sim.bank_worlds = metal.buffer(stage_worlds.size() * sizeof(WWorld), stage_worlds.data());
        const auto schedule = sampler.initial_schedule();
        std::memcpy(sim.bank_schedule.contents, schedule.data(), schedule.size() * sizeof(uint32_t));
        sim.reset();
        validate_reset_state(sim, stage_worlds, schedule);
        std::cout << "curriculum_stage=" << stage_index + 1 << '/' << kStages
                  << " name=" << stage.name << " route_node=" << stage.route_node
                  << " start=translated_witness_node original_goal=retained"
                  << " physics=unchanged action=mode22 speed_cap=1.5 max_steps=400\n";

        const uint32_t stage_first_rollout = trainer.completed_rollouts;
        const auto* runs_before = static_cast<const SimRun*>(sim.runs.contents);
        std::vector<SimRun> previous_runs(kEnvironmentCount);
        std::memcpy(previous_runs.data(), runs_before, sizeof(SimRun) * kEnvironmentCount);
        for (uint32_t stage_rollout = 1; stage_rollout <= rollouts_per_stage; stage_rollout++) {
            if (stage_rollout > 1) {
                const auto* active_ids = static_cast<const uint32_t*>(sim.bank_active_ids.contents);
                const std::vector<uint32_t> active(active_ids, active_ids + kEnvironmentCount);
                const auto next = sampler.next_schedule(active, trainer.completed_rollouts);
                std::memcpy(sim.bank_schedule.contents, next.data(), next.size() * sizeof(uint32_t));
            }
            auto commands = [metal.queue commandBuffer];
            sim.collect(commands, kHorizon);
            const double gpu_seconds = metal.finish(commands);
            commands = [metal.queue commandBuffer];
            trainer.rollout_update(commands, trainer.completed_rollouts);
            const double update_gpu_seconds = metal.finish(commands);
            trainer.completed_rollouts++;

            const auto* current_runs = static_cast<const SimRun*>(sim.runs.contents);
            uint64_t episodes = 0, successes = 0, collisions = 0, timeouts = 0;
            double path_delta = 0.0, elapsed_delta = 0.0;
            float minimum_clearance = 12.0f;
            for (uint32_t env = 0; env < kEnvironmentCount; env++) {
                const auto& current = current_runs[env];
                const auto& previous = previous_runs[env];
                require(current.episodes >= previous.episodes &&
                        current.successes >= previous.successes &&
                        current.collisions >= previous.collisions &&
                        current.timeouts >= previous.timeouts,
                        "curriculum episode counters moved backwards inside a stage");
                episodes += current.episodes - previous.episodes;
                successes += current.successes - previous.successes;
                collisions += current.collisions - previous.collisions;
                timeouts += current.timeouts - previous.timeouts;
                path_delta += current.total_path - previous.total_path;
                elapsed_delta += current.total_elapsed - previous.total_elapsed;
                minimum_clearance = std::min(minimum_clearance, current.min_clearance);
            }
            std::memcpy(previous_runs.data(), current_runs, sizeof(SimRun) * kEnvironmentCount);
            const float* metrics = static_cast<const float*>(trainer.metric_mean.contents);
            const double mean_speed = elapsed_delta > 0.0 ? path_delta / elapsed_delta : 0.0;
            rollout_log << stage_index + 1 << ',' << stage_rollout << ','
                        << uint64_t(trainer.completed_rollouts) * kEnvironmentCount * kHorizon
                        << ',' << gpu_seconds + update_gpu_seconds << ',' << episodes << ','
                        << successes << ',' << collisions << ',' << timeouts << ','
                        << mean_speed << ',' << minimum_clearance << ','
                        << metrics[0] << ',' << metrics[1] << ',' << metrics[2] << ','
                        << metrics[3] << '\n';
            if (stage_rollout % 25 == 0 || stage_rollout == rollouts_per_stage) {
                rollout_log.flush();
                std::cout << "curriculum_rollout stage=" << stage_index + 1
                          << " rollout=" << stage_rollout
                          << " transitions=" << uint64_t(trainer.completed_rollouts) * kEnvironmentCount * kHorizon
                          << " train_success=" << (episodes ? double(successes) / episodes : 0.0)
                          << " train_collision=" << (episodes ? double(collisions) / episodes : 0.0)
                          << " train_timeout=" << (episodes ? double(timeouts) / episodes : 0.0)
                          << " mean_speed=" << mean_speed
                          << " min_clearance=" << minimum_clearance << '\n';
            }
            if (stage_rollout % 50 == 0 || stage_rollout == rollouts_per_stage) {
                const std::string checkpoint = output_dir + "/stage-" +
                    std::to_string(stage_index + 1) + "-rollout-" +
                    std::to_string(stage_rollout) + ".bin";
                trainer.save_checkpoint(checkpoint, kFamilyCorner, 42, trainer.completed_rollouts);
            }
        }

        const uint32_t stage_end = trainer.completed_rollouts;
        const auto* stage_runs = static_cast<const SimRun*>(sim.runs.contents);
        uint64_t stage_episodes = 0, stage_successes = 0, stage_collisions = 0, stage_timeouts = 0;
        double stage_path = 0.0, stage_elapsed = 0.0;
        float stage_clearance = 12.0f;
        for (uint32_t env = 0; env < kEnvironmentCount; env++) {
            stage_episodes += stage_runs[env].episodes;
            stage_successes += stage_runs[env].successes;
            stage_collisions += stage_runs[env].collisions;
            stage_timeouts += stage_runs[env].timeouts;
            stage_path += stage_runs[env].total_path;
            stage_elapsed += stage_runs[env].total_elapsed;
            stage_clearance = std::min(stage_clearance, stage_runs[env].min_clearance);
        }
        const std::string checkpoint = output_dir + "/stage-" + std::to_string(stage_index + 1) + ".bin";
        trainer.save_checkpoint(checkpoint, kFamilyCorner, 42, stage_end);
        const std::string dev_csv = output_dir + "/stage-" + std::to_string(stage_index + 1) + "-dev.csv";
        const auto score = run_navigation_bank_evaluation(
            metal, checkpoint, bank_path, "dev", dev_csv, 17, 1.5f, 400);
        const auto family_successes = family_counts(score);
        const bool eligible = family_successes[0] > 0 && family_successes[1] >= 25 &&
                              family_successes[2] >= 29;
        const uint64_t family_episodes = score.family_episodes[0] + score.family_episodes[1] +
                                         score.family_episodes[2];
        require(family_episodes == 90, "stage DEV evaluation must score all 90 original-start cases");
        history << stage_index + 1 << ',' << stage.name << ',' << stage.route_node << ','
                << (stage_end - stage_first_rollout) << ','
                << uint64_t(stage_end) * kEnvironmentCount * kHorizon << ','
                << (stage_episodes ? double(stage_successes) / stage_episodes : 0.0) << ','
                << (stage_episodes ? double(stage_collisions) / stage_episodes : 0.0) << ','
                << (stage_episodes ? double(stage_timeouts) / stage_episodes : 0.0) << ','
                << (stage_elapsed > 0.0 ? stage_path / stage_elapsed : 0.0) << ','
                << stage_clearance << ',' << family_successes[0] << ',' << family_successes[1]
                << ',' << family_successes[2] << ',' << score.successes << ',' << score.collisions
                << ',' << score.timeouts << ',' << challenge_evaluation::sha256_file(checkpoint)
                << ',' << challenge_evaluation::sha256_file(dev_csv) << '\n';
        history.flush();
        require(bool(history), "curriculum stage history write failed");
        std::cout << "curriculum_stage_end=" << stage_index + 1
                  << " reset_task_success=" << stage_successes << '/' << stage_episodes
                  << " dev_original_start=" << family_successes[0] << "/30,"
                  << family_successes[1] << "/30," << family_successes[2] << "/30"
                  << " total=" << score.successes << "/90"
                  << " eligible=" << (eligible ? "true" : "false")
                  << " checkpoint_sha256=" << challenge_evaluation::sha256_file(checkpoint)
                  << " dev_csv_sha256=" << challenge_evaluation::sha256_file(dev_csv) << '\n';
    }

    std::cout << "curriculum_complete source_bank_sha256=" << source_sha
              << " total_rollouts=" << uint64_t(rollouts_per_stage) * kStages
              << " wall_s=" << seconds() - mission_started
              << " final_split=untouched target_training=none\n";
}

static std::vector<float> checkpoint_actor(const std::string& checkpoint) {
    std::ifstream file(checkpoint, std::ios::binary);
    require(bool(file), "cannot open trace checkpoint: " + checkpoint);
    const auto header = read_checkpoint_header(file);
    require(header.actor_count == fixed_ppo::actor_param_count,
            "trace checkpoint actor dimensions do not match this build");
    std::vector<float> actor(header.actor_count);
    file.read(reinterpret_cast<char*>(actor.data()), actor.size() * sizeof(float));
    require(bool(file) && std::all_of(actor.begin(), actor.end(),
                                      [](float value) { return std::isfinite(value); }),
            "trace checkpoint actor data is invalid");
    return actor;
}

static void trace_dev_corners(const std::string& bank_path, const std::string& checkpoint,
                             const std::string& output_dir, uint32_t dev_index) {
    require(dev_index < 30, "corner DEV index must be0..29");
    Metal metal;
    metal.compile(base_source() + PPO_TRAINER_MSL);
    const std::string world_hash = challenge_evaluation::sha256_file(
        std::string(SOURCE_DIR) + "/world.hpp");
    std::string bank_hash;
    const auto levels = challenge_evaluation::load_split(bank_path, "dev", world_hash, bank_hash);
    const auto actor = checkpoint_actor(checkpoint);
    const std::string actor_hash = challenge_evaluation::sha256_file(checkpoint);
    std::filesystem::create_directories(output_dir);
    std::ofstream summary(output_dir + "/trace-summary.csv",
                          dev_index == 0 ? std::ios::trunc : std::ios::app);
    require(bool(summary), "cannot write trace summary");
    if (dev_index == 0)
        summary << "failure_id,checkpoint_sha256,success,collision,timeout,steps,elapsed_s,path_m,mean_speed_mps,peak_speed_mps,min_clearance_m,final_distance_m,trajectory_csv_sha256\n";

    uint32_t corner_index = 0;
    bool traced = false;
    for (const auto& level : levels) {
        if (level.family != kFamilyCorner) continue;
        if (corner_index++ != dev_index) continue;
        SimConfig config;
        config.n = 1;
        config.family = 0;
        config.mode = 17;
        config.eval = 1;
        config.seed = level.seed;
        config.distance = level.distance;
        config.speed = 1.5f;
        config.max_steps = 400;
        config.geometry_memory = 1;
        Sim sim(metal, config, config.max_steps);
        std::memcpy(sim.actor.contents, actor.data(), actor.size() * sizeof(float));
        auto* world = static_cast<WWorld*>(sim.worlds.contents);
        auto* run = static_cast<SimRun*>(sim.runs.contents);
        world[0] = level.world;
        run[0].rng = level.base_seed + level.environment_index * challenge_evaluation::kEnvStride +
                     challenge_evaluation::kEnvOffset;
        run[0].steps = 0;
        run[0].initial_distance = std::sqrt(level.world.goal[0] * level.world.goal[0] +
            level.world.goal[1] * level.world.goal[1] +
            (level.world.goal[2] - 1.5f) * (level.world.goal[2] - 1.5f));
        run[0].reference_position[0] = 0.0f;
        run[0].reference_position[1] = 0.0f;
        run[0].reference_position[2] = 1.5f;
        auto commands = [metal.queue commandBuffer];
        sim.collect(commands, config.max_steps);
        metal.finish(commands);
        require(run[0].episodes == 1, "DEV trace did not finish one full route attempt: " + level.failure_id);

        const std::string stem = output_dir + "/" + level.failure_id;
        std::ofstream trace(stem + ".csv");
        require(bool(trace), "cannot write DEV trace: " + stem);
        trace << "t_s,x_m,y_m,z_m,qw,qx,qy,qz,vx_mps,vy_mps,vz_mps,wx_rps,wy_rps,wz_rps,ref_x_m,ref_y_m,ref_z_m,cmd_vx_world_mps,cmd_vy_world_mps,cmd_vz_world_mps,cmd_yaw_rps,clearance_m,depth_age_s";
        for (uint32_t ray = 0; ray < 320; ray++) trace << ",raw_current_" << ray;
        for (uint32_t ray = 0; ray < 320; ray++) trace << ",raw_previous_" << ray;
        trace << '\n' << std::setprecision(9);
        const auto* critic_rows = static_cast<const float*>(sim.co.contents);
        const auto* observations = static_cast<const float*>(sim.obs.contents);
        const uint32_t context = 160;
        const auto& physics = *static_cast<const RLPhysicsParams*>(sim.physics.contents);
        const float nav_dt = physics.dt * config.substeps;
        for (uint32_t tick = 0; tick < run[0].steps; tick++) {
            const float* state = critic_rows + size_t(tick) * 32;
            const float* observation = observations + size_t(tick) * fixed_ppo::actor_obs_dim;
            float applied[4];
            for (uint32_t axis = 0; axis < 4; axis++)
                applied[axis] = tick + 1 < run[0].steps
                    ? observations[size_t(tick + 1) * fixed_ppo::actor_obs_dim + context + 13 + axis]
                    : run[0].previous_nav[axis];
            float velocity[3] = {applied[0] * config.speed, applied[1] * config.speed,
                applied[2] * config.speed};
            const float magnitude = std::sqrt(velocity[0] * velocity[0] +
                velocity[1] * velocity[1] + velocity[2] * velocity[2]);
            const float scale = config.velocity_contract == 1
                ? std::min(1.0f, config.speed / std::max(magnitude, 1.0e-8f)) : 1.0f;
            for (float& component : velocity) component *= scale;
            float rotation[9];
            raptor_quaternion_matrix(state + 9, rotation);
            trace << tick * nav_dt;
            for (uint32_t axis = 0; axis < 3; axis++) trace << ',' << state[13 + axis] * 10.0f;
            for (uint32_t axis = 0; axis < 4; axis++) trace << ',' << state[9 + axis];
            for (uint32_t axis = 0; axis < 3; axis++) trace << ',' << state[3 + axis] * 4.0f;
            for (uint32_t axis = 0; axis < 3; axis++) trace << ',' << state[6 + axis] * 4.0f;
            for (uint32_t axis = 0; axis < 3; axis++) trace << ',' << state[13 + axis] * 10.0f + state[16 + axis] * 0.5f;
            for (uint32_t axis = 0; axis < 3; axis++)
                trace << ',' << rotation[axis * 3] * velocity[0] + rotation[axis * 3 + 1] * velocity[1] + rotation[axis * 3 + 2] * velocity[2];
            trace << ',' << applied[3] * 0.5f << ',' << state[19] * 5.0f << ','
                  << tick * nav_dt - observation[context + 17];
            for (uint32_t ray = 0; ray < 320; ray++) trace << ',' << observation[181 + ray] * 12.0f;
            for (uint32_t ray = 0; ray < 320; ray++) trace << ',' << observation[501 + ray] * 12.0f;
            trace << '\n';
        }
        trace.flush();
        require(bool(trace), "DEV trace write failed: " + stem);
        const float dx = static_cast<const RLPhysicsState*>(sim.states.contents)[0].position[0] - level.world.goal[0];
        const float dy = static_cast<const RLPhysicsState*>(sim.states.contents)[0].position[1] - level.world.goal[1];
        const float dz = static_cast<const RLPhysicsState*>(sim.states.contents)[0].position[2] - level.world.goal[2];
        const float goal_distance = std::sqrt(dx * dx + dy * dy + dz * dz);
        summary << level.failure_id << ',' << actor_hash << ',' << run[0].successes << ','
                << run[0].collisions << ',' << run[0].timeouts << ',' << run[0].steps << ','
                << run[0].elapsed << ',' << run[0].path << ','
                << (run[0].elapsed > 0.0f ? run[0].path / run[0].elapsed : 0.0f) << ','
                << run[0].peak_speed << ',' << run[0].min_clearance << ',' << goal_distance << ','
                << challenge_evaluation::sha256_file(stem + ".csv") << '\n';
        summary.flush();
        std::cout << "dev_trace checkpoint=" << checkpoint << " scene=" << level.failure_id
                  << " success=" << run[0].successes << " collision=" << run[0].collisions
                  << " elapsed_s=" << run[0].elapsed << " goal_error_m=" << goal_distance
                  << " output=" << stem << ".csv\n";
        traced = true;
        break;
    }
    require(traced, "requested family-14 DEV level is missing");
    std::cout << "dev_trace_complete count=1 dev_index=" << dev_index << " split=dev bank_sha256=" << bank_hash
              << " checkpoint_sha256=" << actor_hash << " final_split=untouched\n";
}

static int main_cli(int argc, char** argv) {
    if (argc == 6 && std::string(argv[1]) == "--train") {
        const uint32_t rollouts_per_stage = uint32_t(std::stoul(argv[5]));
        std::filesystem::create_directories(argv[4]);
        train(argv[2], argv[3], argv[4], rollouts_per_stage);
        return 0;
    }
    if (argc == 6 && std::string(argv[1]) == "--trace") {
        trace_dev_corners(argv[2], argv[3], argv[4], uint32_t(std::stoul(argv[5])));
        return 0;
    }
    require(argc == 7 && std::string(argv[1]) == "--resume",
            "navigation_curriculum --train BANK WARMSTART OUTPUT_DIR ROLLOUTS_PER_STAGE | --resume BANK CHECKPOINT OUTPUT_DIR ROLLOUTS_PER_STAGE NEXT_STAGE_INDEX | --trace BANK CHECKPOINT OUTPUT_DIR DEV_CORNER_INDEX");
    const uint32_t rollouts_per_stage = uint32_t(std::stoul(argv[5]));
    const uint32_t next_stage = uint32_t(std::stoul(argv[6]));
    std::filesystem::create_directories(argv[4]);
    train(argv[2], "", argv[4], rollouts_per_stage, next_stage, argv[3]);
    return 0;
}

} // namespace navigation_curriculum

int main(int argc, char** argv) {
    @autoreleasepool {
        try {
            return navigation_curriculum::main_cli(argc, argv);
        } catch (const std::exception& error) {
            std::cerr << error.what() << '\n';
            return 1;
        }
    }
}
