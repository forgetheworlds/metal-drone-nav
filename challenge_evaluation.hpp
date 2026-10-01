#pragma once

// Frozen challenge-bank evaluation. Include from Objective-C++ after Sim,
// SimRun, SimConfig, PpoCheckpointHeader and read_checkpoint_header exist.
// Ground-truth level metadata stays on the host. Only saved WWorld geometry,
// saved goal/wind and the ordinary actor observation reach the simulator.

#import <Foundation/Foundation.h>
#include <CommonCrypto/CommonDigest.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cfloat>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <locale>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace challenge_evaluation {

constexpr uint32_t kMaxBatch = 128;
constexpr uint32_t kMaxObstacles = 16;
constexpr uint32_t kMaxSteps = 400; // 400 x 50 ms = 20 s per episode.
constexpr uint32_t kEnvStride = 747796405u;
constexpr uint32_t kEnvOffset = 2891336453u;

struct Level {
    std::string failure_id, split, family_name, geometry_contract, difficulty_band;
    std::string scene_sha256, world_source_sha256;
    uint32_t family = 0, seed = 0, base_seed = 0, environment_index = 0;
    float distance = 0, difficulty = 0;
    float direct_route_clearance = 0, witness_clearance = 0;
    uint32_t witness_samples = 0;
    std::vector<std::array<float, 3>> witness_route;
    WWorld world{};
};

struct Result {
    Level level;
    uint32_t episodes = 0, successes = 0, collisions = 0, timeouts = 0;
    float progress = 0, path_m = 0, goal_time_s = 0;
    float mean_speed_mps = 0, peak_speed_mps = 0, min_clearance_m = 0;
};

[[noreturn]] inline void fail(const std::string& message) {
    throw std::runtime_error("challenge bank: " + message);
}

inline std::string sha256_file(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) fail("cannot read for SHA-256: " + path);
    CC_SHA256_CTX context;
    CC_SHA256_Init(&context);
    char buffer[64 * 1024];
    while (file) {
        file.read(buffer, sizeof(buffer));
        const std::streamsize count = file.gcount();
        if (count > 0) CC_SHA256_Update(&context, buffer, CC_LONG(count));
    }
    if (!file.eof()) fail("read failed while hashing: " + path);
    unsigned char digest[CC_SHA256_DIGEST_LENGTH];
    CC_SHA256_Final(digest, &context);
    std::ostringstream out;
    out << std::hex << std::setfill('0');
    for (unsigned char byte : digest) out << std::setw(2) << unsigned(byte);
    return out.str();
}

inline bool is_sha256(const std::string& value) {
    return value.size() == 64 && std::all_of(value.begin(), value.end(), [](unsigned char c) {
        return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
    });
}

inline id required_field(NSDictionary* object, const char* name, const std::string& where) {
    NSString* key = [NSString stringWithUTF8String:name];
    id value = [object objectForKey:key];
    if (value == nil) fail(where + " missing field " + name);
    return value;
}

inline NSDictionary* required_dict(id value, const std::string& where) {
    if (![value isKindOfClass:[NSDictionary class]]) fail(where + " must be an object");
    return (NSDictionary*)value;
}

inline NSArray* required_array(id value, const std::string& where) {
    if (![value isKindOfClass:[NSArray class]]) fail(where + " must be an array");
    return (NSArray*)value;
}

inline std::string required_string(id value, const std::string& where) {
    if (![value isKindOfClass:[NSString class]]) fail(where + " must be a string");
    const char* utf8 = [(NSString*)value UTF8String];
    if (!utf8) fail(where + " is not valid UTF-8");
    return utf8;
}

inline double required_number(id value, const std::string& where) {
    if (![value isKindOfClass:[NSNumber class]]) fail(where + " must be a number");
    const double result = [(NSNumber*)value doubleValue];
    if (!std::isfinite(result)) fail(where + " must be finite");
    return result;
}

inline uint32_t required_uint(id value, const std::string& where) {
    const double number = required_number(value, where);
    if (number < 0 || number > double(UINT32_MAX) || std::floor(number) != number)
        fail(where + " must be an unsigned 32-bit integer");
    return uint32_t(number);
}

inline float required_float(id value, const std::string& where) {
    const double number = required_number(value, where);
    if (std::fabs(number) > double(FLT_MAX)) fail(where + " is outside float32 range");
    return float(number);
}

inline std::array<float, 3> required_vec3(id value, const std::string& where) {
    NSArray* array = required_array(value, where);
    if ([array count] != 3) fail(where + " must contain exactly 3 values");
    return {required_float(array[0], where + "[0]"),
            required_float(array[1], where + "[1]"),
            required_float(array[2], where + "[2]")};
}

inline void require_room_bounds(NSDictionary* record, const std::string& where) {
    NSDictionary* bounds = required_dict(required_field(record, "room_bounds", where), where + ".room_bounds");
    const std::array<std::array<float, 2>, 3> expected{{{{-2, 14}}, {{-5, 5}}, {{0, 5}}}};
    const char* names[3] = {"x", "y", "z"};
    for (uint32_t axis = 0; axis < 3; ++axis) {
        NSArray* pair = required_array(required_field(bounds, names[axis], where + ".room_bounds"), where + ".room_bounds");
        if ([pair count] != 2) fail(where + ".room_bounds axis must have two values");
        const float lo = required_float(pair[0], where + ".room_bounds lower");
        const float hi = required_float(pair[1], where + ".room_bounds upper");
        if (std::fabs(lo - expected[axis][0]) > 1e-5f || std::fabs(hi - expected[axis][1]) > 1e-5f)
            fail(where + " uses room bounds incompatible with world.hpp");
    }
}

inline Level parse_level(NSDictionary* record, uint32_t line_number) {
    const std::string where = "JSONL line " + std::to_string(line_number);
    Level level;
    if (required_string(required_field(record, "record_type", where), where + ".record_type") != "challenge")
        fail(where + " has an unsupported record_type");
    if (required_uint(required_field(record, "schema_version", where), where + ".schema_version") != 1)
        fail(where + " has an unsupported schema_version");

    level.failure_id = required_string(required_field(record, "failure_id", where), where + ".failure_id");
    level.split = required_string(required_field(record, "split", where), where + ".split");
    if (level.split != "train" && level.split != "dev" && level.split != "final") fail(where + " has invalid split");
    level.family = required_uint(required_field(record, "family", where), where + ".family");
    if (level.family < 14 || level.family > 16) fail(where + " family must be 14, 15, or 16");
    level.family_name = required_string(required_field(record, "family_name", where), where + ".family_name");
    level.geometry_contract = required_string(required_field(record, "geometry_contract", where), where + ".geometry_contract");
    const char* expected_name = level.family == 14 ? "bent_hallway_corner" :
                               level.family == 15 ? "connected_rooms_offset_doors_furniture" :
                                                    "vertical_over_under_choice";
    const char* expected_contract = level.family == 14 ? "connected partitions force north turn, east hall, then south turn" :
                                   level.family == 15 ? "offset doors with intermediate table detour" :
                                                        "cross low barrier above, overhang below, then choose over or under slab";
    if (level.family_name != expected_name || level.geometry_contract != expected_contract)
        fail(where + " family label/geometry contract does not match its family id");
    level.seed = required_uint(required_field(record, "seed", where), where + ".seed");
    level.environment_index = required_uint(required_field(record, "environment_index", where), where + ".environment_index");
    level.difficulty = required_float(required_field(record, "difficulty", where), where + ".difficulty");
    if (level.difficulty < 0 || level.difficulty >= 1) fail(where + " difficulty must be in [0,1)");
    level.difficulty_band = required_string(required_field(record, "difficulty_band", where), where + ".difficulty_band");
    if (level.difficulty_band != "easy" && level.difficulty_band != "medium" && level.difficulty_band != "hard")
        fail(where + " has invalid difficulty_band");
    const std::string expected_band = level.difficulty < (1.0f / 3.0f) ? "easy" :
                                      level.difficulty < (2.0f / 3.0f) ? "medium" : "hard";
    if (level.difficulty_band != expected_band) fail(where + " difficulty band does not match its difficulty value");
    level.scene_sha256 = required_string(required_field(record, "scene_sha256", where), where + ".scene_sha256");
    level.world_source_sha256 = required_string(required_field(record, "world_source_sha256", where), where + ".world_source_sha256");
    if (!is_sha256(level.scene_sha256) || !is_sha256(level.world_source_sha256)) fail(where + " has malformed source hashes");

    NSDictionary* config = required_dict(required_field(record, "sim_config", where), where + ".sim_config");
    if (required_uint(required_field(config, "family", where), where + ".sim_config.family") != level.family ||
        required_uint(required_field(config, "environment_index", where), where + ".sim_config.environment_index") != level.environment_index)
        fail(where + " sim_config does not match record family/index");
    level.base_seed = required_uint(required_field(config, "seed", where), where + ".sim_config.seed");
    level.distance = required_float(required_field(config, "distance", where), where + ".sim_config.distance");
    if (level.distance < 3.0f || level.distance > 10.0f) fail(where + " distance must be in [3,10]");
    uint32_t seed_state = level.base_seed + level.environment_index * kEnvStride + kEnvOffset;
    seed_state ^= seed_state << 13; seed_state ^= seed_state >> 17; seed_state ^= seed_state << 5;
    if (seed_state != level.seed) fail(where + " seed does not match simulator environment seed derivation");

    require_room_bounds(record, where);
    const auto goal = required_vec3(required_field(record, "goal", where), where + ".goal");
    const auto wind = required_vec3(required_field(record, "wind", where), where + ".wind");
    const double body_radius = required_number(required_field(record, "body_radius_m", where), where + ".body_radius_m");
    if (std::fabs(body_radius - 0.18) > 1e-5) fail(where + " body radius differs from simulator collision model");

    level.world = WWorld{};
    level.world.family = level.family;
    level.world.seed = level.seed;
    for (uint32_t axis = 0; axis < 3; ++axis) {
        level.world.goal[axis] = goal[axis];
        level.world.wind[axis] = wind[axis];
    }
    NSArray* obstacles = required_array(required_field(record, "obstacles", where), where + ".obstacles");
    if ([obstacles count] > kMaxObstacles) fail(where + " contains more than 16 obstacles");
    level.world.count = uint32_t([obstacles count]);
    for (uint32_t i = 0; i < level.world.count; ++i) {
        NSDictionary* object = required_dict(obstacles[i], where + ".obstacles[]");
        WObstacle& obstacle = level.world.obstacles[i];
        obstacle.kind = required_uint(required_field(object, "kind", where), where + ".obstacle.kind");
        if (obstacle.kind > 2) fail(where + " obstacle kind must be AABB(0), sphere(1), or cylinder(2)");
        const auto center = required_vec3(required_field(object, "center", where), where + ".obstacle.center");
        const auto size = required_vec3(required_field(object, "half_extent", where), where + ".obstacle.half_extent");
        const auto velocity = required_vec3(required_field(object, "velocity", where), where + ".obstacle.velocity");
        for (uint32_t axis = 0; axis < 3; ++axis) {
            if (size[axis] < 0) fail(where + " obstacle size cannot be negative");
            obstacle.center[axis] = center[axis];
            obstacle.size[axis] = size[axis];
            obstacle.velocity[axis] = velocity[axis];
        }
        if ((obstacle.kind == 0 && (size[0] <= 0 || size[1] <= 0 || size[2] <= 0)) ||
            (obstacle.kind == 1 && size[0] <= 0) ||
            (obstacle.kind == 2 && (size[0] <= 0 || size[2] <= 0)))
            fail(where + " obstacle has a non-positive collision dimension");
        for (uint32_t axis = 0; axis < 3; ++axis) {
            const float lo = axis == 0 ? -2.0f : axis == 1 ? -5.0f : 0.0f;
            const float hi = axis == 0 ? 14.0f : axis == 1 ? 5.0f : 5.0f;
            const float extent = obstacle.kind == 0 ? size[axis] :
                                 obstacle.kind == 1 ? size[0] :
                                 axis == 2 ? size[2] : size[0];
            if (center[axis] - extent < lo - 1e-4f || center[axis] + extent > hi + 1e-4f)
                fail(where + " obstacle extends outside the fixed room bounds");
        }
    }

    const std::array<float, 3> start{{0, 0, 1.5f}};
    if (wclearance(level.world, wv(start[0], start[1], start[2]), 0) <= 0.04f ||
        wclearance(level.world, wv(goal[0], goal[1], goal[2]), 0) <= 0.04f)
        fail(where + " has an unsafe start or goal for the 0.18 m vehicle");
    level.direct_route_clearance = required_float(required_field(record, "direct_route_clearance_m", where), where + ".direct_route_clearance_m");
    level.witness_clearance = required_float(required_field(record, "witness_min_clearance_m", where), where + ".witness_min_clearance_m");
    level.witness_samples = required_uint(required_field(record, "witness_sample_count", where), where + ".witness_sample_count");
    if (level.direct_route_clearance >= -0.02f || level.witness_clearance <= 0.04f || level.witness_samples == 0)
        fail(where + " does not satisfy the stored obstructed-route/witness-clearance contract");
    NSArray* route = required_array(required_field(record, "witness_route", where), where + ".witness_route");
    if ([route count] < 2) fail(where + " witness_route needs at least two points");
    for (NSUInteger i = 0; i < [route count]; ++i) level.witness_route.push_back(required_vec3(route[i], where + ".witness_route[]"));
    for (uint32_t axis = 0; axis < 3; ++axis) {
        if (std::fabs(level.witness_route.front()[axis] - start[axis]) > 1e-5f ||
            std::fabs(level.witness_route.back()[axis] - goal[axis]) > 1e-5f)
            fail(where + " witness route endpoints do not match the simulator start and stored goal");
    }
    return level;
}

inline std::vector<Level> load_split(const std::string& path, const std::string& split,
                                    const std::string& expected_world_hash,
                                    std::string& bank_hash) {
    if (split != "train" && split != "dev" && split != "final") fail("split must be train, dev, or final");
    bank_hash = sha256_file(path);
    std::ifstream input(path);
    if (!input) fail("cannot open bank JSONL: " + path);
    std::vector<Level> selected;
    std::set<std::string> failure_ids;
    std::string line;
    uint32_t line_number = 0;
    while (std::getline(input, line)) {
        ++line_number;
        if (line.empty()) fail("blank JSONL row at line " + std::to_string(line_number));
        NSData* data = [NSData dataWithBytes:line.data() length:line.size()];
        NSError* error = nil;
        id json = [NSJSONSerialization JSONObjectWithData:data options:NSJSONReadingFragmentsAllowed error:&error];
        if (json == nil || ![json isKindOfClass:[NSDictionary class]])
            fail("invalid JSON object at line " + std::to_string(line_number) +
                 (error ? std::string(": ") + [[error localizedDescription] UTF8String] : ""));
        Level level = parse_level((NSDictionary*)json, line_number);
        if (level.world_source_sha256 != expected_world_hash) fail("bank world.hpp hash does not match the running binary source");
        if (!failure_ids.insert(level.failure_id).second) fail("duplicate failure_id: " + level.failure_id);
        if (level.split == split) selected.push_back(std::move(level));
        if (line_number > 1000000) fail("bank exceeds one million records");
    }
    if (!input.eof()) fail("failed while reading bank JSONL");
    if (selected.empty()) fail("bank contains no records for requested split " + split);
    return selected;
}

inline std::string csv_cell(const std::string& value) {
    if (value.find_first_of(",\"\r\n") == std::string::npos) return value;
    std::string escaped = "\"";
    for (char c : value) { if (c == '"') escaped += '"'; escaped += c; }
    return escaped + '"';
}

inline std::string number_cell(double value) {
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << std::setprecision(9) << value;
    return out.str();
}

inline std::string uint_cell(uint64_t value) { return std::to_string(value); }

inline std::string witness_route_cell(const Level& level) {
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << std::setprecision(9) << '[';
    for (size_t i = 0; i < level.witness_route.size(); ++i) {
        if (i) out << ',';
        out << '[' << level.witness_route[i][0] << ',' << level.witness_route[i][1]
            << ',' << level.witness_route[i][2] << ']';
    }
    out << ']';
    return out.str();
}

inline void write_csv_row(std::ofstream& file, const std::vector<std::string>& cells) {
    for (size_t i = 0; i < cells.size(); ++i) {
        if (i) file << ',';
        file << csv_cell(cells[i]);
    }
    file << '\n';
}

inline void run(Metal& metal, const std::string& checkpoint_path,
                const std::string& bank_path, const std::string& split,
                const std::string& output_path, uint32_t mode = 17,
                float speed = 1.5f, uint32_t max_steps = 400) {
    if (mode != 2 && mode != 13 && mode != 17) fail("mode must be 2, 13, or 17");
    if (!std::isfinite(speed) || speed <= 0 || speed > 3.0f) fail("speed must be finite and in (0,3]");
    if (max_steps == 0 || max_steps > kMaxSteps) fail("max_steps must be in 1..400 (maximum 20 seconds)");
    if (mode >= 13 && fixed_ppo::actor_obs_dim != 184) fail("guided modes require the 184-input checkpoint build");

    const std::string checkpoint_hash = sha256_file(checkpoint_path);
    std::ifstream checkpoint(checkpoint_path, std::ios::binary);
    if (!checkpoint) fail("cannot open checkpoint: " + checkpoint_path);
    const PpoCheckpointHeader header = read_checkpoint_header(checkpoint);
    if (!checkpoint || header.actor_count != fixed_ppo::actor_param_count ||
        header.critic_count != fixed_ppo::critic_param_count || header.version < 3 || header.version > 6)
        fail("checkpoint header/dimensions are unsupported");
    std::vector<float> actor(header.actor_count), critic(header.critic_count);
    checkpoint.read(reinterpret_cast<char*>(actor.data()), std::streamsize(actor.size() * sizeof(float)));
    checkpoint.read(reinterpret_cast<char*>(critic.data()), std::streamsize(critic.size() * sizeof(float)));
    if (!checkpoint) fail("checkpoint is missing actor or critic weights");
    const auto finite_weights = [](float value) { return std::isfinite(value); };
    if (!std::all_of(actor.begin(), actor.end(), finite_weights) ||
        !std::all_of(critic.begin(), critic.end(), finite_weights))
        fail("checkpoint contains non-finite weights");

    const std::string world_path = std::string(SOURCE_DIR) + "/world.hpp";
    const std::string world_hash = sha256_file(world_path);
    std::string bank_hash;
    std::vector<Level> levels = load_split(bank_path, split, world_hash, bank_hash);
    if (split == "final")
        std::cerr << "FINAL HOLDOUT: do not use this split for training, checkpoint selection, or curriculum design.\n";
    const std::filesystem::path output_file(output_path);
    if (output_file.has_parent_path()) std::filesystem::create_directories(output_file.parent_path());
    const std::filesystem::path temporary_output = output_file.string() + ".tmp";
    std::ofstream csv(temporary_output, std::ios::out | std::ios::trunc);
    if (!csv) fail("cannot write output CSV: " + temporary_output.string());
    csv.imbue(std::locale::classic());
    write_csv_row(csv, {"failure_id", "scene_sha256", "split", "family", "family_name", "seed",
        "base_seed", "environment_index", "distance_m", "geometry_contract", "difficulty", "difficulty_band", "mode",
        "speed_mps", "max_steps", "wind_x", "wind_y", "wind_z", "obstacle_count", "witness_route_json",
        "episodes", "success", "collision", "timeout", "progress", "path_m", "elapsed_s",
        "goal_time_s", "mean_speed_mps", "peak_speed_mps", "min_clearance_m", "physics_invalid",
        "direct_route_clearance_m", "witness_min_clearance_m", "witness_sample_count",
        "checkpoint_sha256", "bank_sha256", "world_source_sha256"});

    uint64_t total_success = 0, total_collision = 0, total_timeout = 0;
    for (size_t begin = 0; begin < levels.size(); begin += kMaxBatch) {
        const uint32_t count = uint32_t(std::min<size_t>(kMaxBatch, levels.size() - begin));
        SimConfig config;
        config.n = count;
        config.family = 0; // Geometry is overwritten from the bank before any observation.
        config.mode = mode;
        config.seed = levels[begin].seed;
        config.eval = 1;
        config.speed = speed;
        config.distance = levels[begin].distance;
        config.max_steps = max_steps;
        config.sensor_delay = 0;
        config.command_delay = 0;
        config.wind = 0;
        config.depth_noise = 0;
        config.dropout = 0;
        config.velocity_contract = header.config.velocity_contract;
        config.geometry_memory = header.config.geometry_memory;
        Sim sim(metal, config, 32);
        std::memcpy(sim.actor.contents, actor.data(), actor.size() * sizeof(float));
        std::memcpy(sim.critic.contents, critic.data(), critic.size() * sizeof(float));

        WWorld* worlds = static_cast<WWorld*>(sim.worlds.contents);
        SimRun* runs = static_cast<SimRun*>(sim.runs.contents);
        for (uint32_t n = 0; n < count; ++n) {
            const Level& level = levels[begin + n];
            worlds[n] = level.world;
            // Match reset's deterministic policy sampling seed for this bank
            // environment, independent of its current GPU batch slot.
            runs[n].rng = level.base_seed + level.environment_index * kEnvStride + kEnvOffset;
            runs[n].steps = 0;
            runs[n].initial_distance = std::sqrt(
                level.world.goal[0] * level.world.goal[0] +
                level.world.goal[1] * level.world.goal[1] +
                (level.world.goal[2] - 1.5f) * (level.world.goal[2] - 1.5f));
            runs[n].reference_position[0] = 0;
            runs[n].reference_position[1] = 0;
            runs[n].reference_position[2] = 1.5f;
        }
        auto command_buffer = [metal.queue commandBuffer];
        sim.collect(command_buffer, max_steps);
        metal.finish(command_buffer);

        const RLPhysicsState* states = static_cast<const RLPhysicsState*>(sim.states.contents);
        for (uint32_t n = 0; n < count; ++n) {
            const Level& level = levels[begin + n];
            const SimRun& run = runs[n];
            if (run.episodes != 1 || run.successes + run.collisions + run.timeouts != 1)
                fail("level did not produce exactly one terminal outcome: " + level.failure_id);
            const float progress = run.final_progress;
            const float mean_speed = run.elapsed > 0 ? run.path / run.elapsed : 0;
            const float state_values[] = {progress, run.path, run.elapsed, run.success_time,
                                          mean_speed, run.peak_speed, run.min_clearance};
            if (!std::all_of(std::begin(state_values), std::end(state_values), [](float x) { return std::isfinite(x); }))
                fail("non-finite episode metric for " + level.failure_id);
            for (uint32_t j = 0; j < 3; ++j)
                if (!std::isfinite(states[n].position[j]) || !std::isfinite(states[n].linear_velocity[j]))
                    fail("non-finite terminal simulator state for " + level.failure_id);

            total_success += run.successes;
            total_collision += run.collisions;
            total_timeout += run.timeouts;
            write_csv_row(csv, {level.failure_id, level.scene_sha256, level.split,
                uint_cell(level.family), level.family_name, uint_cell(level.seed),
                uint_cell(level.base_seed), uint_cell(level.environment_index), number_cell(level.distance),
                level.geometry_contract, number_cell(level.difficulty), level.difficulty_band, uint_cell(mode), number_cell(speed),
                uint_cell(max_steps), number_cell(level.world.wind[0]), number_cell(level.world.wind[1]),
                number_cell(level.world.wind[2]), uint_cell(level.world.count), witness_route_cell(level),
                uint_cell(run.episodes),
                uint_cell(run.successes), uint_cell(run.collisions), uint_cell(run.timeouts),
                number_cell(progress), number_cell(run.path), number_cell(run.elapsed),
                run.successes ? number_cell(run.success_time) : "", number_cell(mean_speed),
                number_cell(run.peak_speed), number_cell(run.min_clearance), "not_instrumented",
                number_cell(level.direct_route_clearance), number_cell(level.witness_clearance),
                uint_cell(level.witness_samples), checkpoint_hash, bank_hash, level.world_source_sha256});
        }
        csv.flush();
        if (!csv) fail("write failed for output CSV: " + temporary_output.string());
    }
    csv.close();
    if (!csv) fail("close failed for output CSV: " + temporary_output.string());
    std::error_code rename_error;
    std::filesystem::rename(temporary_output, output_file, rename_error);
    if (rename_error) fail("cannot atomically publish output CSV: " + rename_error.message());
    std::cout << "bank_eval split=" << split << " levels=" << levels.size()
              << " mode=" << mode << " speed=" << speed << " max_steps=" << max_steps
              << " budget_s=" << (double(max_steps) * 0.05)
              << " success=" << double(total_success) / levels.size()
              << " collision=" << double(total_collision) / levels.size()
              << " timeout=" << double(total_timeout) / levels.size()
              << " output=" << output_path << " bank_sha256=" << bank_hash
              << " checkpoint_sha256=" << checkpoint_hash << "\n";
}

} // namespace challenge_evaluation
