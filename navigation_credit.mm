// Controlled credit-assignment experiment for the source navigation policy.
//
// One knob only: the PPO/GAE discount gamma. The control arm keeps the
// preserved source objective (gamma=.99); the treatment arm uses gamma=.995.
// Everything else is held byte-identical: challenge bank, sampler seed and
// schedule stream, uniform per-environment family mix, original-start TRAIN
// worlds (no translated curriculum starts), the raw824 parameter warmstart,
// optimizer reset, PPO epochs/minibatch/clip/value coefficient/entropy/risk,
// learning rate, rollout horizon, action map, sensors and physical budget.
//
// This is a cold runner by construction. The PPO checkpoint format does not
// record gamma, so a gamma-tagged run must never be reloaded as a full resume.
// This file therefore has no resume path; it reads only actor+critic parameter
// warmstarts and refuses to write into a directory that already holds a
// checkpoint. Every run writes a provenance sidecar that states its gamma.
#define main embedded_navigation_main
#include "main.mm"
#undef main

#include <iomanip>
#include <sstream>

namespace navigation_credit {

constexpr uint32_t kEnvironmentCount = 128;
constexpr uint32_t kHorizon = 32;
constexpr uint32_t kEpochs = 2;
constexpr float kLearningRate = 0.0001f;
constexpr float kEntropyCoefficient = 0.0005f;
constexpr float kRiskCoefficient = 0.1f;
constexpr float kSpeedCap = 1.5f;
constexpr uint32_t kMaxSteps = 400;
constexpr uint32_t kFamilyCorner = 14;
constexpr uint32_t kSamplerSeed = 42;
// The established source run used one cold curriculum stage per checkpoint; the
// approved per-arm budget for this isolated experiment is <=800 rollouts.
constexpr uint32_t kMaxRolloutsPerArm = 800;

struct Arm {
    const char* name;
    float gamma;
};

// gamma=.99 is the preserved source objective; gamma=.995 is the single
// isolated treatment. No other arm is defined here.
constexpr Arm kControlArm{"gamma0990", 0.99f};
constexpr Arm kTreatmentArm{"gamma0995", 0.995f};

// Path of the executing binary, captured in main() so provenance can hash the
// exact bytes that were run.
static std::string g_runner_binary = "build/credit";

// Reject anything that is not exactly one of the two matched arms.
static Arm arm_from_name(const std::string& name) {
    if (name == kControlArm.name) return kControlArm;
    if (name == kTreatmentArm.name) return kTreatmentArm;
    throw std::runtime_error("arm must be " + std::string(kControlArm.name) + " or " + kTreatmentArm.name);
}

// SHA-256 of a raw byte range; used to fingerprint schedules and prove two
// constructions are equal without writing files.
static std::string sha256_bytes(const void* data, size_t bytes) {
    unsigned char digest[CC_SHA256_DIGEST_LENGTH];
    CC_SHA256(static_cast<const unsigned char*>(data), CC_LONG(bytes), digest);
    std::ostringstream out;
    out << std::hex << std::setfill('0');
    for (unsigned char byte : digest) out << std::setw(2) << unsigned(byte);
    return out.str();
}

static void write_text(const std::string& path, const std::string& text) {
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    require(bool(file), "cannot write " + path);
    file << text;
    require(bool(file), "write failed for " + path);
}

// Parameter-only warmstart: copy the actor and critic weights and leave the
// optimizer moments at their constructor zeros. The warmstart header config is
// deliberately ignored; only dimensions are checked, exactly as the preserved
// source curriculum did.
static void load_parameter_warmstart(Sim& sim, const std::string& path) {
    std::ifstream source(path, std::ios::binary);
    require(bool(source), "cannot open warmstart: " + path);
    const auto header = read_checkpoint_header(source);
    require(header.actor_count == fixed_ppo::actor_param_count &&
            header.critic_count == fixed_ppo::critic_param_count,
            "warmstart dimensions do not match this 824-input build");
    source.read(static_cast<char*>(sim.actor.contents), sim.actor.length);
    source.read(static_cast<char*>(sim.critic.contents), sim.critic.length);
    require(bool(source), "warmstart parameters are truncated");
}

// Build one cold arm: sampler schedule (original starts), simulator, trainer.
// The returned Sim owns the bank buffers used by the reset kernels.
struct ArmRuntime {
    Sim sim;
    PPOTrainer trainer;
    ArmRuntime(Metal& metal, const std::vector<WWorld>& worlds,
               const std::vector<uint32_t>& schedule, challenge_training::Control control,
               const std::string& warmstart)
        : sim(metal, arm_config(), kHorizon), trainer(sim, kEpochs) {
        sim.bank_worlds = metal.buffer(worlds.size() * sizeof(WWorld), worlds.data());
        std::memcpy(sim.bank_control.contents, &control, sizeof(control));
        std::memcpy(sim.bank_schedule.contents, schedule.data(), schedule.size() * sizeof(uint32_t));
        sim.reset();
        load_parameter_warmstart(sim, warmstart);
    }
    static SimConfig arm_config() {
        SimConfig config;
        config.n = kEnvironmentCount;
        config.family = kFamilyCorner;
        config.seed = kSamplerSeed;
        config.mode = 22;
        config.distance = 8.0f;
        config.speed = kSpeedCap;
        config.max_steps = kMaxSteps;
        config.geometry_memory = 1;
        config.risk_coef = kRiskCoefficient;
        config.entropy_coef = kEntropyCoefficient;
        config.learning_rate = kLearningRate;
        return config;
    }
};

// The only permitted difference between arms. Any other divergence is a bug.
static void apply_gamma(PPOTrainer& trainer, float gamma) {
    std::memcpy(trainer.gamma_b.contents, &gamma, sizeof(float));
}

static float gamma_of(const PPOTrainer& trainer) {
    return *static_cast<const float*>(trainer.gamma_b.contents);
}

static void require_equal_bytes(const std::string& label, const void* left, const void* right, size_t bytes) {
    require(std::memcmp(left, right, bytes) == 0, label + " differs between arms");
}

static void require_equal_scalar(const std::string& label, const void* left, const void* right, size_t bytes) {
    require(std::memcmp(left, right, bytes) == 0, label + " buffer differs between arms");
}

// Prove the simulator and trainer state are identical except gamma.
static void verify_arms_match(const ArmRuntime& control, const ArmRuntime& treatment) {
    const Sim& a = control.sim;
    const Sim& b = treatment.sim;
    require_equal_bytes("actor warmstart", a.actor.contents, b.actor.contents, a.actor.length);
    require_equal_bytes("critic warmstart", a.critic.contents, b.critic.contents, a.critic.length);
    require_equal_bytes("bank worlds", a.bank_worlds.contents, b.bank_worlds.contents, a.bank_worlds.length);
    require_equal_bytes("bank schedule", a.bank_schedule.contents, b.bank_schedule.contents, a.bank_schedule.length);
    require_equal_bytes("bank control", a.bank_control.contents, b.bank_control.contents, a.bank_control.length);
    require_equal_bytes("reset states", a.states.contents, b.states.contents, a.states.length);
    require_equal_bytes("reset runs", a.runs.contents, b.runs.contents, a.runs.length);
    require_equal_bytes("reset worlds", a.worlds.contents, b.worlds.contents, a.worlds.length);
    require_equal_bytes("reset sensors", a.sensors.contents, b.sensors.contents, a.sensors.length);
    require_equal_bytes("reset commands", a.commands.contents, b.commands.contents, a.commands.length);
    require(a.cfg.n == b.cfg.n && a.cfg.mode == b.cfg.mode && a.cfg.family == b.cfg.family &&
            a.cfg.seed == b.cfg.seed && a.cfg.speed == b.cfg.speed && a.cfg.distance == b.cfg.distance &&
            a.cfg.max_steps == b.cfg.max_steps && a.cfg.risk_coef == b.cfg.risk_coef &&
            a.cfg.entropy_coef == b.cfg.entropy_coef && a.cfg.learning_rate == b.cfg.learning_rate &&
            a.cfg.geometry_memory == b.cfg.geometry_memory,
            "simulator configuration differs between arms");
    const PPOTrainer& ta = control.trainer;
    const PPOTrainer& tb = treatment.trainer;
    require(ta.epochs == tb.epochs && ta.minibatch == tb.minibatch, "PPO epochs/minibatch differ between arms");
    require_equal_scalar("lambda", ta.lambda_b.contents, tb.lambda_b.contents, ta.lambda_b.length);
    require_equal_scalar("clip", ta.clip_b.contents, tb.clip_b.contents, ta.clip_b.length);
    require_equal_scalar("value coefficient", ta.value_coef_b.contents, tb.value_coef_b.contents, ta.value_coef_b.length);
    require_equal_scalar("entropy coefficient", ta.entropy_coef_b.contents, tb.entropy_coef_b.contents, ta.entropy_coef_b.length);
    require_equal_scalar("rows", ta.rows_b.contents, tb.rows_b.contents, ta.rows_b.length);
    require_equal_scalar("envs", ta.envs_b.contents, tb.envs_b.contents, ta.envs_b.length);
    require_equal_scalar("horizon", ta.horizon_b.contents, tb.horizon_b.contents, ta.horizon_b.length);
    require_equal_scalar("logstd lo", ta.logstd_lo_b.contents, tb.logstd_lo_b.contents, ta.logstd_lo_b.length);
    require_equal_scalar("logstd hi", ta.logstd_hi_b.contents, tb.logstd_hi_b.contents, ta.logstd_hi_b.length);
    require_equal_scalar("actor max norm", ta.actor_max_norm_b.contents, tb.actor_max_norm_b.contents, ta.actor_max_norm_b.length);
    require_equal_scalar("critic max norm", ta.critic_max_norm_b.contents, tb.critic_max_norm_b.contents, ta.critic_max_norm_b.length);
    require_equal_scalar("advantage clip", ta.adv_clip_b.contents, tb.adv_clip_b.contents, ta.adv_clip_b.length);
    require_equal_scalar("anchor lambda", ta.anchor_lambda_b.contents, tb.anchor_lambda_b.contents, ta.anchor_lambda_b.length);
    require(*static_cast<const float*>(ta.lambda_b.contents) == 0.95f, "GAE lambda must stay 0.95");
    require(*static_cast<const float*>(ta.adv_clip_b.contents) == std::numeric_limits<float>::max(),
            "advantage clip must stay disabled");
    require(*static_cast<const float*>(ta.anchor_lambda_b.contents) == 0.0f, "actor anchor must stay disabled");
    require(std::memcmp(ta.actor_m.contents, tb.actor_m.contents, ta.actor_m.length) == 0,
            "actor optimizer moments differ between arms");
    require(std::memcmp(ta.critic_m.contents, tb.critic_m.contents, ta.critic_m.length) == 0,
            "critic optimizer moments differ between arms");
    require(gamma_of(ta) == kControlArm.gamma && gamma_of(tb) == kTreatmentArm.gamma, "gamma override is wrong");
}

// Physical reset invariants shared by both arms. Original-start worlds are
// required: spawn at [0,0,1.5], identity orientation, initialized rotors,
// 12 m depth ring, and balanced 14/15/16 family slots.
static void validate_reset(const Sim& sim, const std::vector<WWorld>& worlds,
                           const std::vector<uint32_t>& schedule) {
    require(schedule.size() == size_t(kEnvironmentCount) * (kHorizon + 1), "schedule has the wrong size");
    const auto* states = static_cast<const RLPhysicsState*>(sim.states.contents);
    const auto* runs = static_cast<const SimRun*>(sim.runs.contents);
    const auto* active = static_cast<const uint32_t*>(sim.bank_active_ids.contents);
    const auto* actual = static_cast<const WWorld*>(sim.worlds.contents);
    const auto* sensors = static_cast<const float*>(sim.sensors.contents);
    uint32_t corners = 0, rooms = 0, vertical = 0;
    for (uint32_t env = 0; env < kEnvironmentCount; env++) {
        const uint32_t level = active[env];
        require(level < worlds.size(), "initial schedule selected an invalid level");
        require(std::memcmp(&actual[env], &worlds[level], offsetof(WWorld, wind)) == 0,
                "reset did not load the scheduled original-start world");
        const uint32_t family = actual[env].family;
        corners += family == 14; rooms += family == 15; vertical += family == 16;
        require(std::fabs(states[env].position[0]) < 1e-6f &&
                std::fabs(states[env].position[1]) < 1e-6f &&
                std::fabs(states[env].position[2] - 1.5f) < 1e-6f,
                "reset did not start at the normal physical spawn");
        require(std::fabs(states[env].orientation_wxyz[0] - 1.0f) < 1e-6f &&
                std::fabs(states[env].orientation_wxyz[1]) < 1e-6f &&
                std::fabs(states[env].orientation_wxyz[2]) < 1e-6f &&
                std::fabs(states[env].orientation_wxyz[3]) < 1e-6f,
                "reset changed the native orientation");
        require(std::fabs(runs[env].reference_position[2] - 1.5f) < 1e-6f, "RAPTOR reference differs from spawn");
        require(std::isfinite(states[env].rpm[0]) && states[env].rpm[0] > 0.0f, "reset did not initialize rotor speed");
        for (uint32_t ray = 0; ray < 8 * 320; ray++)
            require(sensors[size_t(env) * 8 * 320 + ray] == 12.0f, "reset did not initialize the depth ring");
    }
    require(corners >= 40 && rooms >= 40 && vertical >= 40,
            "initial schedule must sample all three TRAIN families");
    std::cout << "reset_invariants PASS family_slots=" << corners << '/' << rooms << '/' << vertical
              << " physical_spawn=clear orientation=identity rotors=initialized depth_ring=12m\n";
}

static std::array<uint64_t, 3> family_counts(const challenge_evaluation::BankScore& score) {
    return {score.family_successes[0], score.family_successes[1], score.family_successes[2]};
}

// Same acceptance/retention gate the preserved source run used.
static bool eligible(const std::array<uint64_t, 3>& successes) {
    return successes[0] > 0 && successes[1] >= 25 && successes[2] >= 29;
}

static std::vector<uint32_t> family_slots(const Sim& sim, const std::vector<WWorld>& levels) {
    const auto* active = static_cast<const uint32_t*>(sim.bank_active_ids.contents);
    std::vector<uint32_t> counts(3, 0);
    for (uint32_t env = 0; env < kEnvironmentCount; env++) {
        require(active[env] < levels.size(), "active level id outside the train bank");
        counts[levels[active[env]].family - 14]++;
    }
    return counts;
}

static std::string join_counts(const std::vector<uint32_t>& counts) {
    return std::to_string(counts[0]) + "/" + std::to_string(counts[1]) + "/" + std::to_string(counts[2]);
}

// --- preflight: prove the two arms differ only by gamma and the ABI runs ----
static void check(const std::string& bank_path, const std::string& warmstart, const std::string& output_dir) {
    require(fixed_ppo::actor_obs_dim == 824, "credit experiment requires the verified raw-depth actor");
    std::filesystem::create_directories(output_dir);
    Metal metal;
    metal.compile(base_source() + PPO_TRAINER_MSL);

    challenge_training::Settings settings;
    settings.environment_count = kEnvironmentCount;
    settings.horizon = kHorizon;
    settings.sampler_seed = kSamplerSeed;
    settings.selection = challenge_training::SelectionMode::UniformBank;
    settings.focus_family = 0;
    challenge_training::ChallengeTraining sampler(
        bank_path, challenge_evaluation::sha256_file(std::string(SOURCE_DIR) + "/world.hpp"), settings);
    const auto& worlds = sampler.worlds();
    const auto control = sampler.control();
    const auto schedule = sampler.initial_schedule();

    ArmRuntime control_arm(metal, worlds, schedule, control, warmstart);
    ArmRuntime treatment_arm(metal, worlds, schedule, control, warmstart);
    apply_gamma(treatment_arm.trainer, kTreatmentArm.gamma);
    validate_reset(control_arm.sim, worlds, schedule);
    verify_arms_match(control_arm, treatment_arm);

    // Baseline reproduction: the untouched warmstart must score the preserved
    // 54/90 all-90 original-start DEV result (0 corners, 25 rooms, 29 vertical).
    const std::string baseline = output_dir + "/baseline-warmstart.bin";
    control_arm.trainer.save_checkpoint(baseline, kFamilyCorner, kSamplerSeed, 0);
    const auto baseline_score = run_navigation_bank_evaluation(
        metal, baseline, bank_path, "dev", output_dir + "/baseline-dev.csv", 17, kSpeedCap, kMaxSteps);
    const auto baseline_families = family_counts(baseline_score);
    require(baseline_score.episodes == 90, "baseline DEV must score all 90 levels");
    require(baseline_families[0] == 0 && baseline_families[1] == 25 && baseline_families[2] == 29,
            "warmstart does not reproduce the preserved 54/90 DEV baseline");
    std::cout << "baseline_DEV=" << baseline_families[0] << "/30," << baseline_families[1] << "/30,"
              << baseline_families[2] << "/30 total=" << baseline_score.successes << "/90\n";

    // One matched rollout per arm exercises GAE -> normalize -> epochs -> Adam.
    auto update = [&](ArmRuntime& arm, uint32_t index) {
        auto commands = [metal.queue commandBuffer];
        arm.sim.collect(commands, kHorizon);
        arm.trainer.rollout_update(commands, index);
        metal.finish(commands);
        arm.trainer.completed_rollouts++;
        const float* metrics = static_cast<const float*>(arm.trainer.metric_mean.contents);
        std::cout << "rollout arm=" << gamma_of(arm.trainer) << " policy=" << metrics[0] << " value=" << metrics[1]
                  << " entropy=" << metrics[2] << " ratio=" << metrics[3] << '\n';
    };
    update(control_arm, 0);
    update(treatment_arm, 0);
    require(std::memcmp(control_arm.sim.actor.contents, treatment_arm.sim.actor.contents,
                        control_arm.sim.actor.length) != 0,
            "gamma arms produced identical actor updates; the override is not reaching GAE");
    control_arm.trainer.save_checkpoint(output_dir + "/control-rollout1.bin", kFamilyCorner, kSamplerSeed, 1);
    treatment_arm.trainer.save_checkpoint(output_dir + "/treatment-rollout1.bin", kFamilyCorner, kSamplerSeed, 1);

    std::ostringstream report;
    report << "{\n  \"status\": \"preflight_pass\",\n"
           << "  \"bank_sha256\": \"" << challenge_evaluation::sha256_file(bank_path) << "\",\n"
           << "  \"warmstart_sha256\": \"" << challenge_evaluation::sha256_file(warmstart) << "\",\n"
           << "  \"world_sha256\": \"" << challenge_evaluation::sha256_file(std::string(SOURCE_DIR) + "/world.hpp") << "\",\n"
           << "  \"runner_source_sha256\": \""
           << challenge_evaluation::sha256_file(std::string(SOURCE_DIR) + "/navigation_credit.mm") << "\",\n"
           << "  \"runner_binary_sha256\": \"" << challenge_evaluation::sha256_file(g_runner_binary) << "\",\n"
           << "  \"arm_count\": " << kEnvironmentCount << ",\n  \"horizon\": " << kHorizon << ",\n"
           << "  \"sampler_seed\": " << kSamplerSeed << ",\n  \"epochs\": " << kEpochs << ",\n"
           << "  \"minibatch\": " << control_arm.trainer.minibatch << ",\n"
           << "  \"gamma_control\": " << kControlArm.gamma << ",\n  \"gamma_treatment\": " << kTreatmentArm.gamma << ",\n"
           << "  \"lambda\": " << *static_cast<const float*>(control_arm.trainer.lambda_b.contents) << ",\n"
           << "  \"learning_rate\": " << kLearningRate << ",\n"
           << "  \"entropy_coefficient\": " << kEntropyCoefficient << ",\n"
           << "  \"risk_coefficient\": " << kRiskCoefficient << ",\n"
           << "  \"baseline_dev_corners\": " << baseline_families[0] << ",\n"
           << "  \"baseline_dev_rooms\": " << baseline_families[1] << ",\n"
           << "  \"baseline_dev_vertical\": " << baseline_families[2] << ",\n"
           << "  \"baseline_dev_total\": " << baseline_score.successes << "\n}\n";
    write_text(output_dir + "/check.json", report.str());
    std::cout << "credit_preflight_complete output=" << output_dir << '\n';
}

// --- training: cold, single-arm, provenance recorded -----------------------

// Per-family episode outcomes taken from the transition rows themselves.
// bank_transition_ids[row] is written before the reset loads the next world, so
// a just-completed episode is attributed to the family it actually flew.
struct OutcomeTally {
    uint64_t episodes = 0, successes = 0, collisions = 0, timeouts = 0;
};

static std::string checkpoint_path(const std::string& output_dir, uint64_t rollout) {
    return output_dir + "/arm-rollout-" + std::to_string(rollout) + ".bin";
}

static std::string join_tally(const OutcomeTally& tally) {
    return std::to_string(tally.episodes) + "/" + std::to_string(tally.successes) + "/" +
           std::to_string(tally.collisions) + "/" + std::to_string(tally.timeouts);
}

static std::string join_family_outcomes(const std::array<OutcomeTally, 3>& tally) {
    return "corner=" + join_tally(tally[0]) + ";rooms=" + join_tally(tally[1]) +
           ";vertical=" + join_tally(tally[2]);
}

// Scan every transition row of the completed rollout. Returns the minimum
// pre-action sampled clearance over all transitions: the critic observation
// stores clearance/5 at channel 19, and `sim_observe` writes it before
// `sim_advance` executes the tick's RAPTOR substeps. It therefore does NOT
// include the within-tick substep minimum that the risk term uses; report it as
// a pre-action sample, not as the episode's true minimum clearance. Also fills
// per-family episode outcomes.
static float accumulate_train_outcomes(const Sim& sim,
                                       const std::vector<challenge_evaluation::Level>& levels,
                                       std::array<OutcomeTally, 3>& tally) {
    const auto* ids = static_cast<const uint32_t*>(sim.bank_transition_ids.contents);
    const auto* term = static_cast<const uint8_t*>(sim.terminated.contents);
    const auto* trunc = static_cast<const uint8_t*>(sim.truncated.contents);
    const auto* rewards = static_cast<const float*>(sim.rewards.contents);
    const auto* critic_obs = static_cast<const float*>(sim.co.contents);
    const size_t rows = size_t(kEnvironmentCount) * kHorizon;
    float min_clearance = 12.0f;
    for (size_t row = 0; row < rows; row++) {
        min_clearance = std::min(min_clearance, critic_obs[row * 32 + 19] * 5.0f);
        const uint32_t id = ids[row];
        if (id == 0xffffffffu) continue;
        require(id < levels.size(), "transition level id is outside the train bank");
        const uint32_t family = levels[id].family;
        require(family >= 14 && family <= 16, "transition family is outside 14..16");
        if (!term[row] && !trunc[row]) continue;
        OutcomeTally& entry = tally[family - 14];
        entry.episodes++;
        if (term[row] && rewards[row] > 0.0f) entry.successes++;
        else if (term[row]) entry.collisions++;
        else entry.timeouts++;
    }
    return min_clearance;
}

static void train_arm(const std::string& bank_path, const std::string& warmstart,
                      const std::string& output_dir, const Arm& arm, uint32_t rollouts, uint32_t eval_every) {
    require(fixed_ppo::actor_obs_dim == 824, "credit experiment requires the verified raw-depth actor");
    require(rollouts >= 1 && rollouts <= kMaxRolloutsPerArm, "per-arm rollout budget must be 1..800");
    require(eval_every >= 50, "DEV evaluation interval must be at least 50 rollouts");
    require(!std::filesystem::exists(output_dir + "/provenance.json"),
            "refusing to reuse an existing run: this runner is cold-only because gamma is not stored in the checkpoint");
    std::filesystem::create_directories(output_dir);
    std::cout << "credit_arm=" << arm.name << " gamma=" << arm.gamma
              << " lambda=0.95 horizon=" << kHorizon << " epochs=" << kEpochs
              << " minibatch=256 lr=" << kLearningRate << " entropy=" << kEntropyCoefficient
              << " risk=" << kRiskCoefficient << " rollouts=" << rollouts
              << " actionable_resume=false\n";

    Metal metal;
    metal.compile(base_source() + PPO_TRAINER_MSL);
    challenge_training::Settings settings;
    settings.environment_count = kEnvironmentCount;
    settings.horizon = kHorizon;
    settings.sampler_seed = kSamplerSeed;
    settings.selection = challenge_training::SelectionMode::UniformBank;
    settings.focus_family = 0;
    challenge_training::ChallengeTraining sampler(
        bank_path, challenge_evaluation::sha256_file(std::string(SOURCE_DIR) + "/world.hpp"), settings);
    const auto& worlds = sampler.worlds();
    const auto control = sampler.control();
    const auto schedule = sampler.initial_schedule();

    ArmRuntime runtime(metal, worlds, schedule, control, warmstart);
    apply_gamma(runtime.trainer, arm.gamma);
    validate_reset(runtime.sim, worlds, schedule);
    require(gamma_of(runtime.trainer) == arm.gamma, "gamma override failed");

    std::ofstream runs_csv(output_dir + "/credit-runs.csv", std::ios::trunc);
    require(bool(runs_csv), "cannot write credit-runs.csv");
    runs_csv << "rollout,transitions,gpu_s,episodes,success,collision,timeout,mean_speed_mps,"
                "min_preaction_sampled_clearance_m,family_slots_14_15_16,schedule_sha256,"
                "policy_loss,value_loss,entropy_loss,ratio,"
                "corner_episodes,corner_success,corner_collision,corner_timeout,"
                "rooms_episodes,rooms_success,rooms_collision,rooms_timeout,"
                "vertical_episodes,vertical_success,vertical_collision,vertical_timeout,"
                "cumulative_family_outcomes\n";
    std::ofstream dev_csv(output_dir + "/credit-stages.csv", std::ios::trunc);
    require(bool(dev_csv), "cannot write credit-stages.csv");
    dev_csv << "rollout,transitions,dev_corner,dev_rooms,dev_vertical,dev_total,dev_collision,dev_timeout,"
               "retention_gate_eligible_this_arm,checkpoint_sha256,dev_csv_sha256\n";

    std::ostringstream provenance;
    provenance << "{\n  \"arm\": \"" << arm.name << "\",\n  \"gamma\": " << arm.gamma << ",\n"
               << "  \"lambda\": 0.95,\n  \"rollouts\": " << rollouts << ",\n"
               << "  \"envs\": " << kEnvironmentCount << ",\n  \"horizon\": " << kHorizon << ",\n"
               << "  \"epochs\": " << kEpochs << ",\n  \"minibatch\": 256,\n"
               << "  \"learning_rate\": " << kLearningRate << ",\n"
               << "  \"entropy_coefficient\": " << kEntropyCoefficient << ",\n"
               << "  \"risk_coefficient\": " << kRiskCoefficient << ",\n"
               << "  \"speed_cap_mps\": " << kSpeedCap << ",\n  \"max_steps\": " << kMaxSteps << ",\n"
               << "  \"sampler_seed\": " << kSamplerSeed << ",\n  \"sample_selection\": \"uniform_bank\",\n"
               << "  \"reset_starts\": \"original_train_start\",\n"
               << "  \"objective\": \"2*(prev_dist-next_dist)-0.01-risk_coef*clamp((0.6-clearance)/0.6,0,1)+10*success-10*collision\",\n"
               << "  \"bank_sha256\": \"" << challenge_evaluation::sha256_file(bank_path) << "\",\n"
               << "  \"world_sha256\": \"" << challenge_evaluation::sha256_file(std::string(SOURCE_DIR) + "/world.hpp") << "\",\n"
               << "  \"warmstart_sha256\": \"" << challenge_evaluation::sha256_file(warmstart) << "\",\n"
               << "  \"warmstart_kind\": \"parameter_only_optimizer_reset\",\n"
               << "  \"runner_source\": \"" << std::string(SOURCE_DIR) << "/navigation_credit.mm\",\n"
               << "  \"runner_source_sha256\": \""
               << challenge_evaluation::sha256_file(std::string(SOURCE_DIR) + "/navigation_credit.mm") << "\",\n"
               << "  \"runner_binary\": \"" << g_runner_binary << "\",\n"
               << "  \"runner_binary_sha256\": \"" << challenge_evaluation::sha256_file(g_runner_binary) << "\",\n"
               << "  \"checkpoint_snapshots\": \"arm-rollout-<N>.bin per evaluated rollout\",\n"
               << "  \"checkpoint_resume\": false,\n"
               << "  \"gamma_stored_in_checkpoint\": false\n}\n";
    write_text(output_dir + "/provenance.json", provenance.str());
    std::cout << "provenance_bank_sha256=" << challenge_evaluation::sha256_file(bank_path)
              << " warmstart_sha256=" << challenge_evaluation::sha256_file(warmstart) << '\n';

    // Baseline all-90 original-start DEV before any update. Checkpoint snapshots
    // are retained per evaluated rollout so no eligible policy is overwritten.
    const std::string baseline_checkpoint = checkpoint_path(output_dir, 0);
    runtime.trainer.save_checkpoint(baseline_checkpoint, kFamilyCorner, kSamplerSeed, 0);
    auto baseline = run_navigation_bank_evaluation(
        metal, baseline_checkpoint, bank_path, "dev", output_dir + "/dev-rollout-0.csv", 17, kSpeedCap, kMaxSteps);
    auto baseline_families = family_counts(baseline);
    require(baseline.episodes == 90, "baseline DEV must score all 90 levels");
    dev_csv << 0 << ',' << 0 << ',' << baseline_families[0] << ',' << baseline_families[1] << ','
            << baseline_families[2] << ',' << baseline.successes << ',' << baseline.collisions << ','
            << baseline.timeouts << ",false,"
            << challenge_evaluation::sha256_file(baseline_checkpoint) << ','
            << challenge_evaluation::sha256_file(output_dir + "/dev-rollout-0.csv") << '\n';
    dev_csv.flush();
    std::cout << "dev_baseline corners=" << baseline_families[0] << "/30 rooms=" << baseline_families[1]
              << "/30 vertical=" << baseline_families[2] << "/30 total=" << baseline.successes << "/90\n";

    const auto* runs_before = static_cast<const SimRun*>(runtime.sim.runs.contents);
    std::vector<SimRun> previous(runs_before, runs_before + kEnvironmentCount);
    std::array<OutcomeTally, 3> cumulative{};
    uint64_t total_episodes = 0, total_successes = 0, total_collisions = 0, total_timeouts = 0;
    for (uint32_t rollout = 1; rollout <= rollouts; rollout++) {
        if (rollout > 1) {
            const auto* active_ids = static_cast<const uint32_t*>(runtime.sim.bank_active_ids.contents);
            const std::vector<uint32_t> active(active_ids, active_ids + kEnvironmentCount);
            const auto next = sampler.next_schedule(active, runtime.trainer.completed_rollouts);
            std::memcpy(runtime.sim.bank_schedule.contents, next.data(), next.size() * sizeof(uint32_t));
            require_equal_bytes("next schedule", runtime.sim.bank_schedule.contents, next.data(), next.size() * sizeof(uint32_t));
        }
        const std::string schedule_hash = sha256_bytes(runtime.sim.bank_schedule.contents, runtime.sim.bank_schedule.length);
        auto commands = [metal.queue commandBuffer];
        runtime.sim.collect(commands, kHorizon);
        const double collect_gpu = metal.finish(commands);
        commands = [metal.queue commandBuffer];
        runtime.trainer.rollout_update(commands, runtime.trainer.completed_rollouts);
        const double update_gpu = metal.finish(commands);
        runtime.trainer.completed_rollouts++;

        // Episode counters from the environment state, used to validate the
        // per-transition attribution below. Path/elapsed include the active
        // (unfinished) episodes, so the reported mean speed is not
        // completed-episodes-only.
        const auto* current = static_cast<const SimRun*>(runtime.sim.runs.contents);
        uint64_t episodes = 0, successes = 0, collisions = 0, timeouts = 0;
        double path_delta = 0, elapsed_delta = 0;
        for (uint32_t env = 0; env < kEnvironmentCount; env++) {
            require(current[env].episodes >= previous[env].episodes &&
                    current[env].successes >= previous[env].successes &&
                    current[env].collisions >= previous[env].collisions &&
                    current[env].timeouts >= previous[env].timeouts,
                    "episode counters moved backwards inside a run");
            episodes += current[env].episodes - previous[env].episodes;
            successes += current[env].successes - previous[env].successes;
            collisions += current[env].collisions - previous[env].collisions;
            timeouts += current[env].timeouts - previous[env].timeouts;
            path_delta += (current[env].total_path - previous[env].total_path) +
                          (current[env].path - previous[env].path);
            elapsed_delta += (current[env].total_elapsed - previous[env].total_elapsed) +
                             (current[env].elapsed - previous[env].elapsed);
        }
        std::memcpy(previous.data(), current, sizeof(SimRun) * kEnvironmentCount);

        std::array<OutcomeTally, 3> rollout_outcomes{};
        const float min_clearance = accumulate_train_outcomes(runtime.sim, sampler.levels(), rollout_outcomes);
        // The transition-level attribution must reproduce this rollout's
        // environment counters exactly, or the per-family TRAIN rates are not
        // trustworthy.
        const uint64_t attributed_episodes =
            rollout_outcomes[0].episodes + rollout_outcomes[1].episodes + rollout_outcomes[2].episodes;
        const uint64_t attributed_successes =
            rollout_outcomes[0].successes + rollout_outcomes[1].successes + rollout_outcomes[2].successes;
        const uint64_t attributed_collisions =
            rollout_outcomes[0].collisions + rollout_outcomes[1].collisions + rollout_outcomes[2].collisions;
        const uint64_t attributed_timeouts =
            rollout_outcomes[0].timeouts + rollout_outcomes[1].timeouts + rollout_outcomes[2].timeouts;
        require(attributed_episodes == episodes && attributed_successes == successes &&
                attributed_collisions == collisions && attributed_timeouts == timeouts,
                "per-transition outcome attribution does not match the environment counters");
        for (size_t family = 0; family < 3; family++) {
            cumulative[family].episodes += rollout_outcomes[family].episodes;
            cumulative[family].successes += rollout_outcomes[family].successes;
            cumulative[family].collisions += rollout_outcomes[family].collisions;
            cumulative[family].timeouts += rollout_outcomes[family].timeouts;
        }
        total_episodes += episodes; total_successes += successes;
        total_collisions += collisions; total_timeouts += timeouts;
        require(cumulative[0].episodes + cumulative[1].episodes + cumulative[2].episodes == total_episodes &&
                cumulative[0].successes + cumulative[1].successes + cumulative[2].successes == total_successes &&
                cumulative[0].collisions + cumulative[1].collisions + cumulative[2].collisions == total_collisions &&
                cumulative[0].timeouts + cumulative[1].timeouts + cumulative[2].timeouts == total_timeouts,
                "cumulative family outcomes diverge from the environment counters");

        const float* metrics = static_cast<const float*>(runtime.trainer.metric_mean.contents);
        const auto slots = family_slots(runtime.sim, worlds);
        const double mean_speed = elapsed_delta > 0 ? path_delta / elapsed_delta : 0;
        runs_csv << rollout << ',' << uint64_t(runtime.trainer.completed_rollouts) * kEnvironmentCount * kHorizon << ','
                 << collect_gpu + update_gpu << ',' << episodes << ',' << successes << ',' << collisions << ','
                 << timeouts << ',' << mean_speed << ',' << min_clearance << ',' << join_counts(slots) << ','
                 << schedule_hash << ',' << metrics[0] << ',' << metrics[1] << ',' << metrics[2] << ',' << metrics[3]
                 << ',' << rollout_outcomes[0].episodes << ',' << rollout_outcomes[0].successes << ','
                 << rollout_outcomes[0].collisions << ',' << rollout_outcomes[0].timeouts
                 << ',' << rollout_outcomes[1].episodes << ',' << rollout_outcomes[1].successes << ','
                 << rollout_outcomes[1].collisions << ',' << rollout_outcomes[1].timeouts
                 << ',' << rollout_outcomes[2].episodes << ',' << rollout_outcomes[2].successes << ','
                 << rollout_outcomes[2].collisions << ',' << rollout_outcomes[2].timeouts
                 << ',' << join_family_outcomes(cumulative) << '\n';
        if (rollout % 10 == 0 || rollout == rollouts) {
            runs_csv.flush();
            std::cout << "credit_rollout arm=" << arm.name << " rollout=" << rollout
                      << " train_success=" << (episodes ? double(successes) / episodes : 0.0)
                      << " train_collision=" << (episodes ? double(collisions) / episodes : 0.0)
                      << " slots=" << join_counts(slots) << " schedule=" << schedule_hash.substr(0, 12)
                      << " cum_corner=" << join_tally(cumulative[0]) << '\n';
        }
        if (rollout % eval_every == 0 || rollout == rollouts) {
            const std::string checkpoint = checkpoint_path(output_dir, runtime.trainer.completed_rollouts);
            runtime.trainer.save_checkpoint(checkpoint, kFamilyCorner, kSamplerSeed, runtime.trainer.completed_rollouts);
            const std::string dev_path = output_dir + "/dev-rollout-" + std::to_string(rollout) + ".csv";
            const auto score = run_navigation_bank_evaluation(
                metal, checkpoint, bank_path, "dev", dev_path, 17, kSpeedCap, kMaxSteps);
            const auto families = family_counts(score);
            require(score.episodes == 90, "DEV evaluation must score all 90 original-start levels");
            dev_csv << runtime.trainer.completed_rollouts << ','
                    << uint64_t(runtime.trainer.completed_rollouts) * kEnvironmentCount * kHorizon << ','
                    << families[0] << ',' << families[1] << ',' << families[2] << ',' << score.successes << ','
                    << score.collisions << ',' << score.timeouts << ',' << (eligible(families) ? "true" : "false") << ','
                    << challenge_evaluation::sha256_file(checkpoint) << ','
                    << challenge_evaluation::sha256_file(dev_path) << '\n';
            dev_csv.flush();
            std::cout << "dev_rollout arm=" << arm.name << " rollout=" << runtime.trainer.completed_rollouts
                      << " corners=" << families[0] << "/30 rooms=" << families[1] << "/30 vertical="
                      << families[2] << "/30 total=" << score.successes << "/90 eligible="
                      << (eligible(families) ? "true" : "false")
                      << " checkpoint=" << checkpoint << '\n';
        }
    }
    std::ostringstream complete;
    complete << "{\n  \"arm\": \"" << arm.name << "\",\n  \"gamma\": " << arm.gamma << ",\n"
             << "  \"rollouts\": " << runtime.trainer.completed_rollouts << ",\n"
             << "  \"baseline_dev_corners\": " << baseline_families[0] << ",\n"
             << "  \"baseline_dev_rooms\": " << baseline_families[1] << ",\n"
             << "  \"baseline_dev_vertical\": " << baseline_families[2] << ",\n"
             << "  \"baseline_dev_total\": " << baseline.successes << ",\n"
             << "  \"final_checkpoint\": \"" << checkpoint_path(output_dir, runtime.trainer.completed_rollouts) << "\",\n"
             << "  \"final_checkpoint_sha256\": \""
             << challenge_evaluation::sha256_file(checkpoint_path(output_dir, runtime.trainer.completed_rollouts)) << "\",\n"
             << "  \"cumulative_train_family_outcomes\": \"" << join_family_outcomes(cumulative) << "\",\n"
             << "  \"cumulative_train_episodes\": " << total_episodes << ",\n"
             << "  \"cumulative_train_successes\": " << total_successes << ",\n"
             << "  \"cumulative_train_collisions\": " << total_collisions << ",\n"
             << "  \"cumulative_train_timeouts\": " << total_timeouts << "\n}\n";
    write_text(output_dir + "/complete.json", complete.str());
    std::cout << "credit_complete arm=" << arm.name << " total_rollouts=" << runtime.trainer.completed_rollouts
              << " final_split=untouched webots=none\n";
}

static int main_cli(int argc, char** argv) {
    if (argc == 5 && std::string(argv[1]) == "--check") {
        check(argv[2], argv[3], argv[4]);
        return 0;
    }
    if (argc == 5 && std::string(argv[1]) == "--eval") {
        Metal metal;
        metal.compile(base_source() + PPO_TRAINER_MSL);
        run_navigation_bank_evaluation(metal, argv[2], argv[3], "dev", argv[4], 17, kSpeedCap, kMaxSteps);
        return 0;
    }
    require((argc == 7 || argc == 8) && std::string(argv[1]) == "--train",
            "navigation_credit --check BANK WARMSTART OUTPUT_DIR | --eval CHECKPOINT BANK DEV_CSV | "
            "--train BANK WARMSTART OUTPUT_DIR ARM ROLLOUTS [EVAL_EVERY]");
    train_arm(argv[2], argv[3], argv[4], arm_from_name(argv[5]), uint32_t(std::stoul(argv[6])),
              argc == 8 ? uint32_t(std::stoul(argv[7])) : 100);
    return 0;
}

} // namespace navigation_credit

int main(int argc, char** argv) {
    @autoreleasepool {
        try {
            if (argc > 0 && argv[0] != nullptr) navigation_credit::g_runner_binary = argv[0];
            return navigation_credit::main_cli(argc, argv);
        } catch (const std::exception& error) {
            std::cerr << error.what() << '\n';
            return 1;
        }
    }
}
