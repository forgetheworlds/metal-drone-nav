#pragma once

// Narrow train-only bank sampler and per-level failure score state.
// Include after challenge_evaluation.hpp in Objective-C++ host code.
#include "challenge_evaluation.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace challenge_training {

constexpr uint32_t kInvalidLevel = UINT32_MAX;
constexpr uint32_t kStateVersion = 3;

enum class SelectionMode : uint32_t {
    UniformBank = 0,
    FailureWeighted = 1,
};

// Layout mirrors the four-u32 constant buffer read by sim.metal.
struct Control {
    uint32_t enabled = 1;
    uint32_t bank_count = 0;
    uint32_t schedule_stride = 0; // horizon + 1, env-major schedule slots
    uint32_t horizon = 0;
};
static_assert(sizeof(Control) == 16, "challenge bank control layout");

struct Settings {
    uint32_t environment_count = 128;
    uint32_t horizon = 32;
    uint32_t sampler_seed = 1;
    uint32_t rehearsal_environments = 0;
    SelectionMode selection = SelectionMode::UniformBank;
    float uniform_floor = 0.50f;
    float score_staleness_mix = 0.50f;
    uint32_t focus_family = 0; // 0 balances14/15/16; a family pins all bank slots.
};

// This sampler owns frozen train levels, family quotas, replay priorities,
// and partial episode GAE scores. It does not own simulation or PPO buffers.
class ChallengeTraining {
public:
    ChallengeTraining(const std::string& bank_jsonl,
                      const std::string& expected_world_hash,
                      Settings settings)
        : settings_(settings) {
        if (settings_.environment_count == 0 || settings_.horizon == 0)
            throw std::runtime_error("challenge training needs nonzero environments and horizon");
        if (settings_.environment_count<3 || settings_.rehearsal_environments>settings_.environment_count-3)
            throw std::runtime_error("rehearsal must leave at least three bank environments");
        if (settings_.focus_family!=0u && settings_.focus_family!=14u &&
            settings_.focus_family!=15u && settings_.focus_family!=16u)
            throw std::runtime_error("focused challenge family must be14,15,16 or0 for balanced");
        if (settings_.selection != SelectionMode::UniformBank &&
            settings_.selection != SelectionMode::FailureWeighted)
            throw std::runtime_error("unsupported challenge level selection mode");
        if (!std::isfinite(settings_.uniform_floor) || settings_.uniform_floor < 0.5f || settings_.uniform_floor > 1.0f)
            throw std::runtime_error("failure-weighted replay uniform floor must be in [0.5,1]");
        if (!std::isfinite(settings_.score_staleness_mix) || settings_.score_staleness_mix < 0.0f || settings_.score_staleness_mix > 1.0f)
            throw std::runtime_error("score/staleness mix must be in [0,1]");

        levels_ = challenge_evaluation::load_split(bank_jsonl, "train", expected_world_hash, bank_hash_);
        world_hash_ = expected_world_hash;
        for (uint32_t i = 0; i < levels_.size(); ++i) {
            if (levels_[i].split != "train")
                throw std::runtime_error("train loader returned a non-train challenge level");
            family_levels_[levels_[i].family].push_back(i);
            worlds_.push_back(levels_[i].world);
        }
        if (family_levels_.size() < 3)
            throw std::runtime_error("train bank must contain each family 14, 15, and 16");
        for (uint32_t family : {14u, 15u, 16u}) {
            auto found = family_levels_.find(family);
            if (found == family_levels_.end() || found->second.empty())
                throw std::runtime_error("train bank is missing family " + std::to_string(family));
            families_.push_back(family);
        }
        scores_.assign(levels_.size(), 0.0f);
        last_visit_rollout_.assign(levels_.size(), 0);
        last_score_rollout_.assign(levels_.size(), 0);
        completed_episodes_.assign(levels_.size(), 0);
        episode_abs_gae_.assign(settings_.environment_count, 0.0);
        episode_gae_count_.assign(settings_.environment_count, 0);
        episode_level_id_.assign(settings_.environment_count, kInvalidLevel);
        rng_state_ = settings_.sampler_seed ? settings_.sampler_seed : 1;
    }

    const std::vector<challenge_evaluation::Level>& levels() const { return levels_; }
    const std::vector<WWorld>& worlds() const { return worlds_; }
    const std::string& bank_hash() const { return bank_hash_; }
    const std::string& world_hash() const { return world_hash_; }
    Control control() const {
        return {1u, uint32_t(levels_.size()), settings_.horizon + 1, settings_.horizon};
    }

    // Use only for a new training run. Each family receives an equal quota.
    std::vector<uint32_t> initial_schedule() {
        std::vector<uint32_t> schedule(size_t(settings_.environment_count) * (settings_.horizon + 1));
        refresh_priorities();
        for (uint32_t env = 0; env < settings_.environment_count; ++env) {
            if(env < settings_.rehearsal_environments) {
                for(uint32_t tick=0;tick<=settings_.horizon;tick++)
                    schedule[size_t(env)*(settings_.horizon+1)+tick]=kInvalidLevel;
                continue;
            }
            const uint32_t family = family_for_env(env);
            for (uint32_t tick = 0; tick <= settings_.horizon; ++tick)
                schedule[size_t(env) * (settings_.horizon + 1) + tick] = sample_level_for_family(family);
        }
        active_ids_.resize(settings_.environment_count);
        for (uint32_t env = 0; env < settings_.environment_count; ++env)
            active_ids_[env] = schedule[size_t(env) * (settings_.horizon + 1)];
        return schedule;
    }

    // Preserve the level for every live episode in slot 0. Slots 1..H are
    // consumed only when that env completes an episode during this rollout.
    std::vector<uint32_t> next_schedule(const std::vector<uint32_t>& active_level_ids,
                                        uint64_t rollout_index) {
        if (active_level_ids.size() != settings_.environment_count)
            throw std::runtime_error("challenge active-level id count differs from environment count");
        last_rollout_ = rollout_index;
        refresh_priorities();
        std::vector<uint32_t> schedule(size_t(settings_.environment_count) * (settings_.horizon + 1));
        for (uint32_t env = 0; env < settings_.environment_count; ++env) {
            const uint32_t active = active_level_ids[env];
            validate_active_id(env,active);
            schedule[size_t(env) * (settings_.horizon + 1)] = active;
        }
        active_ids_ = active_level_ids;
        for (uint32_t env = 0; env < settings_.environment_count; ++env) {
            if(env < settings_.rehearsal_environments) {
                for(uint32_t tick=1;tick<=settings_.horizon;tick++)
                    schedule[size_t(env)*(settings_.horizon+1)+tick]=kInvalidLevel;
                continue;
            }
            const uint32_t family = family_for_env(env);
            for (uint32_t tick = 1; tick <= settings_.horizon; ++tick)
                schedule[size_t(env) * (settings_.horizon + 1) + tick] = sample_level_for_family(family);
        }
        return schedule;
    }

    const std::vector<uint32_t>& active_ids() const { return active_ids_; }

    // GAE is the raw, unnormalized rollout tensor in [tick][env] row order.
    // Accumulate mean absolute rollout-GAE segments until the episode ends.
    // PPO still bootstraps at each rollout boundary; this is not full-episode GAE.
    void observe_raw_gae(const float* raw_gae,
                         const uint8_t* terminated,
                         const uint8_t* truncated,
                         const uint32_t* transition_level_ids,
                         uint64_t rollout_index) {
        const size_t rows = size_t(settings_.environment_count) * settings_.horizon;
        if (!raw_gae || !terminated || !truncated || !transition_level_ids)
            throw std::runtime_error("challenge rollout evidence buffer is null");
        for (uint32_t tick = 0; tick < settings_.horizon; ++tick) {
            for (uint32_t env = 0; env < settings_.environment_count; ++env) {
                const size_t row = size_t(tick) * settings_.environment_count + env;
                const uint32_t level = transition_level_ids[row];
                if(env < settings_.rehearsal_environments) {
                    if(level!=kInvalidLevel)throw std::runtime_error("rehearsal transition contains a bank level");
                    continue;
                }
                if (level >= levels_.size())
                    throw std::runtime_error("transition level id is outside the frozen train bank");
                uint32_t& active = episode_level_id_[env];
                if (active == kInvalidLevel) active = level;
                if (active != level)
                    throw std::runtime_error("active challenge level changed before episode completion");
                const float value = raw_gae[row];
                if (!std::isfinite(value)) throw std::runtime_error("raw GAE is non-finite");
                episode_abs_gae_[env] += std::fabs(double(value));
                ++episode_gae_count_[env];
                last_visit_rollout_[level] = rollout_index;
                if (terminated[row] || truncated[row]) {
                    const uint32_t count = episode_gae_count_[env];
                    if (count == 0) throw std::runtime_error("completed challenge episode has no GAE samples");
                    scores_[level] = float(episode_abs_gae_[env] / count);
                    last_score_rollout_[level] = rollout_index;
                    ++completed_episodes_[level];
                    episode_abs_gae_[env] = 0.0;
                    episode_gae_count_[env] = 0;
                    active = kInvalidLevel;
                }
            }
        }
        last_rollout_ = rollout_index;
        (void)rows;
    }

    float level_score(uint32_t level) const { return scores_.at(level); }
    uint64_t level_completed_episodes(uint32_t level) const { return completed_episodes_.at(level); }

    // Bind this sidecar to the exact PPO checkpoint and rollout boundary. The
    // caller reads active_level_ids from the completed GPU rollout and passes
    // them here; load_state returns the same IDs for simulator restoration.
    void save_state(const std::string& path,
                    const std::string& checkpoint_sha256,
                    uint64_t completed_rollouts,
                    const std::vector<uint32_t>& active_level_ids) {
        if (active_level_ids.size() != settings_.environment_count)
            throw std::runtime_error("challenge active-level id count differs from environment count");
        for (uint32_t env=0;env<settings_.environment_count;env++) validate_active_id(env,active_level_ids[env]);
        active_ids_ = active_level_ids;
        DiskHeader header = make_header(checkpoint_sha256, completed_rollouts);
        const std::string temporary = path + ".tmp";
        if (std::filesystem::path(path).has_parent_path())
            std::filesystem::create_directories(std::filesystem::path(path).parent_path());
        std::ofstream file(temporary, std::ios::binary | std::ios::trunc);
        if (!file) throw std::runtime_error("cannot write challenge sampler state: " + temporary);
        write_exact(file, &header, sizeof(header));
        write_exact(file, scores_.data(), scores_.size() * sizeof(float));
        write_exact(file, last_visit_rollout_.data(), last_visit_rollout_.size() * sizeof(uint64_t));
        write_exact(file, last_score_rollout_.data(), last_score_rollout_.size() * sizeof(uint64_t));
        write_exact(file, completed_episodes_.data(), completed_episodes_.size() * sizeof(uint64_t));
        write_exact(file, episode_abs_gae_.data(), episode_abs_gae_.size() * sizeof(double));
        write_exact(file, episode_gae_count_.data(), episode_gae_count_.size() * sizeof(uint32_t));
        write_exact(file, episode_level_id_.data(), episode_level_id_.size() * sizeof(uint32_t));
        write_exact(file, active_ids_.data(), active_ids_.size() * sizeof(uint32_t));
        file.flush();
        if (!file) throw std::runtime_error("challenge sampler state write failed: " + temporary);
        file.close();
        if (std::rename(temporary.c_str(), path.c_str()) != 0)
            throw std::runtime_error("cannot replace challenge sampler state: " + path);
    }

    std::vector<uint32_t> load_state(const std::string& path,
                                     const std::string& expected_checkpoint_sha256,
                                     uint64_t expected_completed_rollouts) {
        std::ifstream file(path, std::ios::binary);
        if (!file) throw std::runtime_error("cannot open challenge sampler state: " + path);
        DiskHeader header{};
        read_exact(file, &header, sizeof(header));
        validate_header(header, expected_checkpoint_sha256, expected_completed_rollouts);
        read_exact(file, scores_.data(), scores_.size() * sizeof(float));
        read_exact(file, last_visit_rollout_.data(), last_visit_rollout_.size() * sizeof(uint64_t));
        read_exact(file, last_score_rollout_.data(), last_score_rollout_.size() * sizeof(uint64_t));
        read_exact(file, completed_episodes_.data(), completed_episodes_.size() * sizeof(uint64_t));
        read_exact(file, episode_abs_gae_.data(), episode_abs_gae_.size() * sizeof(double));
        read_exact(file, episode_gae_count_.data(), episode_gae_count_.size() * sizeof(uint32_t));
        read_exact(file, episode_level_id_.data(), episode_level_id_.size() * sizeof(uint32_t));
        active_ids_.resize(settings_.environment_count);
        read_exact(file, active_ids_.data(), active_ids_.size() * sizeof(uint32_t));
        if (!file) throw std::runtime_error("challenge sampler state is truncated");
        char extra;
        if (file.read(&extra, 1)) throw std::runtime_error("challenge sampler state has trailing bytes");
        rng_state_ = header.rng_state;
        last_rollout_ = header.last_rollout;
        for (size_t i=0;i<scores_.size();++i) {
            if (!std::isfinite(scores_[i]) || scores_[i] < 0.0f ||
                last_visit_rollout_[i] > last_rollout_ || last_score_rollout_[i] > last_rollout_)
                throw std::runtime_error("challenge sampler sidecar has invalid priority data");
        }
        for (uint32_t env=0;env<settings_.environment_count;env++) validate_active_id(env,active_ids_[env]);
        for (uint32_t env = 0; env < settings_.environment_count; ++env) {
            const uint32_t partial = episode_level_id_[env];
            if (!std::isfinite(episode_abs_gae_[env]) || episode_abs_gae_[env] < 0.0 ||
                ((episode_gae_count_[env] == 0) != (partial == kInvalidLevel)))
                throw std::runtime_error("challenge sampler sidecar has invalid partial-episode state");
            if (partial != kInvalidLevel && partial != active_ids_[env])
                throw std::runtime_error("challenge active id does not match partial episode score state");
        }
        return active_ids_;
    }

private:
    struct DiskHeader {
        char magic[8];
        uint32_t version, selection, environment_count, horizon, level_count, rng_state, focus_family, reserved;
        uint64_t last_rollout, completed_rollouts;
        float uniform_floor, score_staleness_mix;
        char bank_hash[64], world_hash[64], checkpoint_hash[64];
    };
    static_assert(sizeof(DiskHeader) == 256, "challenge sidecar header layout");

    Settings settings_;
    std::vector<challenge_evaluation::Level> levels_;
    std::vector<WWorld> worlds_;
    std::vector<uint32_t> families_;
    std::map<uint32_t, std::vector<uint32_t>> family_levels_;
    std::map<uint32_t, std::vector<double>> family_priority_weights_;
    std::vector<float> scores_;
    std::vector<uint64_t> last_visit_rollout_, last_score_rollout_, completed_episodes_;
    std::vector<double> episode_abs_gae_;
    std::vector<uint32_t> episode_gae_count_, episode_level_id_;
    std::vector<uint32_t> active_ids_;
    std::string bank_hash_, world_hash_;
    uint32_t rng_state_ = 1;
    uint64_t last_rollout_ = 0;

    uint32_t random_u32() {
        uint32_t x = rng_state_ ? rng_state_ : 1;
        x ^= x << 13; x ^= x >> 17; x ^= x << 5;
        rng_state_ = x ? x : 1;
        return rng_state_;
    }
    double uniform() { return double(random_u32()) / 4294967296.0; }
    uint32_t uniform_index(const std::vector<uint32_t>& candidates) {
        if (candidates.empty()) throw std::runtime_error("challenge family quota has no levels");
        return candidates[(uint64_t(random_u32()) * candidates.size()) >> 32];
    }
    uint32_t family_for_env(uint32_t env) const {
        if(settings_.focus_family!=0u)return settings_.focus_family;
        return families_[(env-settings_.rehearsal_environments) % families_.size()];
    }
    void validate_active_id(uint32_t env,uint32_t id) const {
        if(env < settings_.rehearsal_environments) {
            if(id!=kInvalidLevel)throw std::runtime_error("rehearsal environment contains a bank level");
            return;
        }
        if (id >= levels_.size()) throw std::runtime_error("active challenge id is outside the train bank");
        if (levels_[id].family != family_for_env(env))
            throw std::runtime_error("active challenge id violates its fixed per-environment family quota");
    }
    void refresh_priorities() {
        family_priority_weights_.clear();
        if (settings_.selection != SelectionMode::FailureWeighted) return;
        std::vector<uint32_t> score_position(levels_.size()), age_position(levels_.size());
        for (uint32_t family : families_) {
            const auto& candidates = family_levels_.at(family);
            std::vector<uint32_t> score_rank = candidates, age_rank = candidates;
            std::stable_sort(score_rank.begin(), score_rank.end(), [&](uint32_t a, uint32_t b) {
                if (scores_[a] != scores_[b]) return scores_[a] > scores_[b];
                return a < b;
            });
            std::stable_sort(age_rank.begin(), age_rank.end(), [&](uint32_t a, uint32_t b) {
                if (last_visit_rollout_[a] != last_visit_rollout_[b]) return last_visit_rollout_[a] < last_visit_rollout_[b];
                return a < b;
            });
            for (uint32_t rank = 0; rank < candidates.size(); ++rank) {
                score_position[score_rank[rank]] = rank;
                age_position[age_rank[rank]] = rank;
            }
            auto& weights = family_priority_weights_[family];
            weights.resize(candidates.size());
            for (size_t i = 0; i < candidates.size(); ++i) {
                const uint32_t level = candidates[i];
                weights[i] = settings_.score_staleness_mix / double(score_position[level] + 1) +
                             (1.0 - settings_.score_staleness_mix) / double(age_position[level] + 1);
            }
        }
    }
    uint32_t sample_level_for_family(uint32_t family) {
        const auto& candidates = family_levels_.at(family);
        if (settings_.selection == SelectionMode::UniformBank || uniform() < settings_.uniform_floor)
            return uniform_index(candidates);
        const auto& weights = family_priority_weights_.at(family);
        double total = 0;
        for (double weight : weights) total += weight;
        double target = uniform() * total;
        for (size_t i = 0; i < candidates.size(); ++i) {
            target -= weights[i];
            if (target <= 0) return candidates[i];
        }
        return candidates.back();
    }

    DiskHeader make_header(const std::string& checkpoint_hash, uint64_t completed_rollouts) const {
        if (checkpoint_hash.size() != 64) throw std::runtime_error("challenge state requires a SHA-256 checkpoint hash");
        DiskHeader header{};
        std::memcpy(header.magic, "CHTRN1\0", 8);
        header.version = kStateVersion;
        header.selection = uint32_t(settings_.selection);
        header.environment_count = settings_.environment_count;
        header.horizon = settings_.horizon;
        header.level_count = uint32_t(levels_.size());
        header.reserved = settings_.rehearsal_environments;
        header.rng_state = rng_state_;
        header.focus_family = settings_.focus_family;
        header.last_rollout = last_rollout_;
        header.completed_rollouts = completed_rollouts;
        header.uniform_floor = settings_.uniform_floor;
        header.score_staleness_mix = settings_.score_staleness_mix;
        std::memcpy(header.bank_hash, bank_hash_.data(), 64);
        std::memcpy(header.world_hash, world_hash_.data(), 64);
        std::memcpy(header.checkpoint_hash, checkpoint_hash.data(), 64);
        return header;
    }
    void validate_header(const DiskHeader& header, const std::string& checkpoint_hash,
                         uint64_t completed_rollouts) const {
        const bool rollout_pair_matches = completed_rollouts == 0
            ? header.last_rollout == 0
            : header.last_rollout + 1 == completed_rollouts;
        const bool known_version=header.version==1u||header.version==2u||header.version==kStateVersion;
        const bool focus_matches=header.version==kStateVersion
            ? header.focus_family==settings_.focus_family
            : header.focus_family==0u&&settings_.focus_family==0u;
        if (std::memcmp(header.magic, "CHTRN1\0", 8) != 0 || !known_version || !focus_matches ||
            header.selection != uint32_t(settings_.selection) ||
            header.environment_count != settings_.environment_count || header.horizon != settings_.horizon ||
            header.level_count != levels_.size() || header.completed_rollouts != completed_rollouts || !rollout_pair_matches ||
            (header.version==1 && header.reserved!=0) || header.reserved != settings_.rehearsal_environments || header.rng_state == 0 ||
            std::fabs(header.uniform_floor-settings_.uniform_floor)>1e-7f ||
            std::fabs(header.score_staleness_mix-settings_.score_staleness_mix)>1e-7f ||
            std::memcmp(header.bank_hash,bank_hash_.data(),64)!=0 ||
            std::memcmp(header.world_hash,world_hash_.data(),64)!=0 ||
            !challenge_evaluation::is_sha256(checkpoint_hash) ||
            std::memcmp(header.checkpoint_hash,checkpoint_hash.data(),64)!=0)
            throw std::runtime_error("challenge sampler sidecar does not match bank, world, PPO checkpoint, or rollout");
    }
    template <typename T>
    static void write_exact(std::ofstream& file, const T* values, size_t bytes) {
        file.write(reinterpret_cast<const char*>(values), std::streamsize(bytes));
        if (!file) throw std::runtime_error("challenge sampler state write failed");
    }
    template <typename T>
    static void read_exact(std::ifstream& file, T* values, size_t bytes) {
        file.read(reinterpret_cast<char*>(values), std::streamsize(bytes));
        if (!file) throw std::runtime_error("challenge sampler state is truncated");
    }
};

} // namespace challenge_training
