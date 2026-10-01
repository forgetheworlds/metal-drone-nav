#pragma once

// Small, fixed-shape PPO operators for the RL project. The CPU functions are
// the reference implementation for the matching kernels in ppo.metal.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>

#ifndef FIXED_PPO_ACTOR_OBS_DIM
#define FIXED_PPO_ACTOR_OBS_DIM 661
#endif
#ifndef FIXED_PPO_CRITIC_OBS_DIM
#define FIXED_PPO_CRITIC_OBS_DIM 32
#endif
#ifndef FIXED_PPO_HIDDEN_DIM
#define FIXED_PPO_HIDDEN_DIM 64
#endif
#ifndef FIXED_PPO_ACTION_DIM
#define FIXED_PPO_ACTION_DIM 4
#endif

namespace fixed_ppo {

constexpr std::size_t actor_obs_dim = FIXED_PPO_ACTOR_OBS_DIM;
constexpr std::size_t critic_obs_dim = FIXED_PPO_CRITIC_OBS_DIM;
constexpr std::size_t hidden_dim = FIXED_PPO_HIDDEN_DIM;
constexpr std::size_t action_dim = FIXED_PPO_ACTION_DIM;
constexpr std::size_t actor_w1_offset = 0;
constexpr std::size_t actor_b1_offset = hidden_dim * actor_obs_dim;
constexpr std::size_t actor_w2_offset = actor_b1_offset + hidden_dim;
constexpr std::size_t actor_b2_offset = actor_w2_offset + action_dim * hidden_dim;
constexpr std::size_t actor_log_std_offset = actor_b2_offset + action_dim;
constexpr std::size_t actor_param_count = actor_log_std_offset + action_dim;
constexpr std::size_t critic_w1_offset = 0;
constexpr std::size_t critic_b1_offset = hidden_dim * critic_obs_dim;
constexpr std::size_t critic_w2_offset = critic_b1_offset + hidden_dim;
constexpr std::size_t critic_b2_offset = critic_w2_offset + hidden_dim;
constexpr std::size_t critic_param_count = critic_b2_offset + 1;
constexpr float log_two_pi = 1.8378770664093453f;

struct ActorParams { std::array<float, actor_param_count> values{}; };
struct CriticParams { std::array<float, critic_param_count> values{}; };
struct AdamState {
    std::array<float, actor_param_count> first{};
    std::array<float, actor_param_count> second{};
    std::uint64_t step = 0;
};

struct PpoLoss {
    float policy = 0.0f;
    float value = 0.0f;
    float entropy = 0.0f;
    float ratio = 1.0f;
};

inline float clamp(float x, float lo, float hi) { return std::max(lo, std::min(hi, x)); }

inline void actor_forward(const float* obs, const ActorParams& p, float* hidden, float* mean) {
    const auto& w = p.values;
    for (std::size_t h = 0; h < hidden_dim; ++h) {
        float z = w[actor_b1_offset + h];
        const std::size_t row = actor_w1_offset + h * actor_obs_dim;
        for (std::size_t i = 0; i < actor_obs_dim; ++i) z += w[row + i] * obs[i];
        hidden[h] = std::tanh(z);
    }
    for (std::size_t a = 0; a < action_dim; ++a) {
        float z = w[actor_b2_offset + a];
        const std::size_t row = actor_w2_offset + a * hidden_dim;
        for (std::size_t h = 0; h < hidden_dim; ++h) z += w[row + h] * hidden[h];
        mean[a] = z;
    }
}

inline float critic_forward(const float* obs, const CriticParams& p, float* hidden) {
    const auto& w = p.values;
    for (std::size_t h = 0; h < hidden_dim; ++h) {
        float z = w[critic_b1_offset + h];
        const std::size_t row = critic_w1_offset + h * critic_obs_dim;
        for (std::size_t i = 0; i < critic_obs_dim; ++i) z += w[row + i] * obs[i];
        hidden[h] = std::tanh(z);
    }
    float value = w[critic_b2_offset];
    for (std::size_t h = 0; h < hidden_dim; ++h) value += w[critic_w2_offset + h] * hidden[h];
    return value;
}

inline float gaussian_log_prob(const float* action, const float* mean, const float* log_std) {
    float result = 0.0f;
    for (std::size_t a = 0; a < action_dim; ++a) {
        const float inv_std = std::exp(-log_std[a]);
        const float d = (action[a] - mean[a]) * inv_std;
        result += -0.5f * (d * d + 2.0f * log_std[a] + log_two_pi);
    }
    return result;
}

// Caller writes rewards, values, next_values and boundary flags in time-major
// [time, environment] order. A truncation bootstraps from next_value, then
// stops the GAE carry so it cannot cross into the next episode.
inline void compute_gae(std::size_t horizon, std::size_t env_count,
                        const float* rewards, const float* values, const float* next_values,
                        const std::uint8_t* terminated, const std::uint8_t* truncated,
                        float gamma, float lambda, float* advantages, float* returns) {
    for (std::size_t env = 0; env < env_count; ++env) {
        float carry = 0.0f;
        for (std::size_t rev = 0; rev < horizon; ++rev) {
            const std::size_t t = horizon - 1 - rev;
            const std::size_t k = t * env_count + env;
            const bool term = terminated[k] != 0;
            const bool boundary = term || truncated[k] != 0;
            const float bootstrap = term ? 0.0f : next_values[k];
            const float delta = rewards[k] + gamma * bootstrap - values[k];
            carry = delta + (boundary ? 0.0f : gamma * lambda * carry);
            advantages[k] = carry;
            returns[k] = carry + values[k];
        }
    }
}

inline void normalize_advantages(float* advantages, std::size_t count, float epsilon = 1.0e-8f) {
    if (count == 0) return;
    double sum = 0.0;
    for (std::size_t i = 0; i < count; ++i) sum += advantages[i];
    const float mean = static_cast<float>(sum / static_cast<double>(count));
    double sq = 0.0;
    for (std::size_t i = 0; i < count; ++i) {
        const double d = static_cast<double>(advantages[i] - mean);
        sq += d * d;
    }
    const float inv = 1.0f / std::sqrt(static_cast<float>(sq / count) + epsilon);
    for (std::size_t i = 0; i < count; ++i) advantages[i] = (advantages[i] - mean) * inv;
}

// d_* are derivatives of the minibatch-mean objective. The policy derivative
// follows PPO's selected min branch; clipped samples have zero policy gradient.
inline PpoLoss sample_loss_and_grad(const float* action, const float* mean, const float* log_std,
                                    float old_logp, float old_value, float value,
                                    float advantage, float target_return, float batch_size,
                                    float clip_epsilon, float value_coefficient,
                                    float entropy_coefficient, float* d_mean,
                                    float* d_log_std, float* d_value) {
    const float logp = gaussian_log_prob(action, mean, log_std);
    const float raw_log_ratio = logp - old_logp;
    const float log_ratio = clamp(raw_log_ratio, -20.0f, 20.0f);
    const float ratio = std::exp(log_ratio);
    const float clipped = clamp(ratio, 1.0f - clip_epsilon, 1.0f + clip_epsilon);
    const float chosen = std::min(ratio * advantage, clipped * advantage);
    const bool ratio_in_range = raw_log_ratio >= -20.0f && raw_log_ratio <= 20.0f;
    const bool policy_active = ratio_in_range && (advantage >= 0.0f ? ratio <= 1.0f + clip_epsilon
                                                                    : ratio >= 1.0f - clip_epsilon);
    const float scale = 1.0f / std::max(batch_size, 1.0f);
    PpoLoss loss;
    loss.policy = -chosen * scale;
    loss.value = 0.5f * value_coefficient * (value - target_return) * (value - target_return) * scale;
    loss.ratio = ratio;
    for (std::size_t a = 0; a < action_dim; ++a) {
        const float inv_var = std::exp(-2.0f * log_std[a]);
        const float diff = action[a] - mean[a];
        if (policy_active) {
            const float dlogp = -ratio * advantage * scale;
            d_mean[a] = dlogp * diff * inv_var;
            d_log_std[a] = dlogp * (diff * diff * inv_var - 1.0f);
        } else {
            d_mean[a] = 0.0f;
            d_log_std[a] = 0.0f;
        }
        d_log_std[a] -= entropy_coefficient * scale;
        loss.entropy += -entropy_coefficient * (log_std[a] + 0.5f * (1.0f + log_two_pi)) * scale;
    }
    *d_value = value_coefficient * (value - target_return) * scale;
    (void)old_value; // retained in the ABI for rollout diagnostics and parity.
    return loss;
}

inline float sample_objective(const float* action, const float* mean, const float* log_std,
                              float old_logp, float value, float advantage, float target_return,
                              float clip_epsilon, float value_coefficient,
                              float entropy_coefficient) {
    const float ratio = std::exp(clamp(gaussian_log_prob(action, mean, log_std) - old_logp, -20.0f, 20.0f));
    const float clipped = clamp(ratio, 1.0f - clip_epsilon, 1.0f + clip_epsilon);
    float entropy = 0.0f;
    for (std::size_t a = 0; a < action_dim; ++a)
        entropy += log_std[a] + 0.5f * (1.0f + log_two_pi);
    const float error = value - target_return;
    return -std::min(ratio * advantage, clipped * advantage)
         + 0.5f * value_coefficient * error * error
         - entropy_coefficient * entropy;
}

inline void adam_update(float* params, const float* grad, float* first, float* second,
                        std::size_t count, std::uint64_t step, float learning_rate,
                        float beta1, float beta2, float epsilon, float weight_decay);

// Optional quick CPU checks for a host harness. They cover the active PPO
// branches, clipped policy gradients, value/entropy derivatives, and Adam.
inline bool run_cpu_self_tests(float tolerance = 2.0e-3f) {
    float action[action_dim]{};
    float mean[action_dim]{};
    float log_std[action_dim]{};
    action[0] = 0.4f;
    mean[0] = 0.1f;
    const float base_logp = gaussian_log_prob(action, mean, log_std);
    constexpr float clip = 0.2f;
    constexpr float value_coef = 0.7f;
    constexpr float entropy_coef = 0.03f;
    const float value = 0.25f;
    const float target = -0.3f;
    float d_mean[action_dim]{};
    float d_log_std[action_dim]{};
    float d_value = 0.0f;
    const float h = 1.0e-3f;
    const float old_values[] = {base_logp - std::log(1.1f), base_logp - std::log(0.9f),
                                base_logp - std::log(1.3f), base_logp - std::log(0.7f)};
    const float advantages[] = {1.0f, -1.0f, 1.0f, -1.0f};
    for (std::size_t branch = 0; branch < 4; ++branch) {
        sample_loss_and_grad(action, mean, log_std, old_values[branch], value, value,
                             advantages[branch], target, 1.0f, clip, value_coef,
                             entropy_coef, d_mean, d_log_std, &d_value);
        float saved = mean[0];
        mean[0] = saved + h;
        const float plus_mean = sample_objective(action, mean, log_std, old_values[branch], value,
                                                  advantages[branch], target, clip, value_coef, entropy_coef);
        mean[0] = saved - h;
        const float minus_mean = sample_objective(action, mean, log_std, old_values[branch], value,
                                                   advantages[branch], target, clip, value_coef, entropy_coef);
        mean[0] = saved;
        const float numeric_mean = (plus_mean - minus_mean) / (2.0f * h);
        if (std::fabs(numeric_mean - d_mean[0]) > tolerance) return false;

        saved = log_std[0];
        log_std[0] = saved + h;
        const float plus_std = sample_objective(action, mean, log_std, old_values[branch], value,
                                                 advantages[branch], target, clip, value_coef, entropy_coef);
        log_std[0] = saved - h;
        const float minus_std = sample_objective(action, mean, log_std, old_values[branch], value,
                                                  advantages[branch], target, clip, value_coef, entropy_coef);
        log_std[0] = saved;
        const float numeric_std = (plus_std - minus_std) / (2.0f * h);
        if (std::fabs(numeric_std - d_log_std[0]) > tolerance) return false;
    }
    sample_loss_and_grad(action, mean, log_std, base_logp, value, value, 0.0f, target,
                         1.0f, clip, value_coef, entropy_coef, d_mean, d_log_std, &d_value);
    const float numeric_value = value_coef * (value - target);
    if (std::fabs(numeric_value - d_value) > tolerance) return false;

    float parameter = 1.0f;
    float gradient = 0.25f;
    float first = 0.0f;
    float second = 0.0f;
    adam_update(&parameter, &gradient, &first, &second, 1, 1, 0.01f,
                0.9f, 0.999f, 1.0e-8f, 0.0f);
    const float expected_first = 0.1f * gradient;
    const float expected_second = 0.001f * gradient * gradient;
    const float expected_parameter = 1.0f - 0.01f * (expected_first / 0.1f)
                                  / (std::sqrt(expected_second / 0.001f) + 1.0e-8f);
    return std::fabs(first - expected_first) <= tolerance
        && std::fabs(second - expected_second) <= tolerance
        && std::fabs(parameter - expected_parameter) <= tolerance;
}

inline void actor_backward_sample(const float* obs, const float* hidden,
                                  const float* d_mean, const float* d_log_std,
                                  const ActorParams& p, float* out_grad) {
    std::fill(out_grad, out_grad + actor_param_count, 0.0f);
    const auto& w = p.values;
    for (std::size_t a = 0; a < action_dim; ++a) {
        out_grad[actor_b2_offset + a] = d_mean[a];
        const std::size_t row = actor_w2_offset + a * hidden_dim;
        for (std::size_t h = 0; h < hidden_dim; ++h) {
            out_grad[row + h] = d_mean[a] * hidden[h];
            const float d_hidden = d_mean[a] * w[row + h] * (1.0f - hidden[h] * hidden[h]);
            out_grad[actor_b1_offset + h] += d_hidden;
            const std::size_t first_row = actor_w1_offset + h * actor_obs_dim;
            for (std::size_t i = 0; i < actor_obs_dim; ++i) out_grad[first_row + i] += d_hidden * obs[i];
        }
    }
    for (std::size_t a = 0; a < action_dim; ++a)
        out_grad[actor_log_std_offset + a] = d_log_std[a];
}

inline void critic_backward_sample(const float* obs, const float* hidden,
                                   float d_value, const CriticParams& p, float* out_grad) {
    std::fill(out_grad, out_grad + critic_param_count, 0.0f);
    const auto& w = p.values;
    out_grad[critic_b2_offset] = d_value;
    for (std::size_t h = 0; h < hidden_dim; ++h) {
        out_grad[critic_w2_offset + h] = d_value * hidden[h];
        const float d_hidden = d_value * w[critic_w2_offset + h] * (1.0f - hidden[h] * hidden[h]);
        out_grad[critic_b1_offset + h] = d_hidden;
        const std::size_t row = critic_w1_offset + h * critic_obs_dim;
        for (std::size_t i = 0; i < critic_obs_dim; ++i) out_grad[row + i] = d_hidden * obs[i];
    }
}

template <std::size_t P>
inline void reduce_sample_grads(const float* per_sample, std::size_t batch_size, float* out) {
    std::fill(out, out + P, 0.0f);
    for (std::size_t n = 0; n < batch_size; ++n)
        for (std::size_t p = 0; p < P; ++p) out[p] += per_sample[n * P + p];
}

inline void adam_update(float* params, const float* grad, float* first, float* second,
                        std::size_t count, std::uint64_t step, float learning_rate,
                        float beta1 = 0.9f, float beta2 = 0.999f,
                        float epsilon = 1.0e-8f, float weight_decay = 0.0f) {
    const float b1_correction = 1.0f - std::pow(beta1, static_cast<float>(step));
    const float b2_correction = 1.0f - std::pow(beta2, static_cast<float>(step));
    for (std::size_t i = 0; i < count; ++i) {
        const float g = grad[i] + weight_decay * params[i];
        first[i] = beta1 * first[i] + (1.0f - beta1) * g;
        second[i] = beta2 * second[i] + (1.0f - beta2) * g * g;
        const float m_hat = first[i] / b1_correction;
        const float v_hat = second[i] / b2_correction;
        params[i] -= learning_rate * m_hat / (std::sqrt(v_hat) + epsilon);
    }
}

} // namespace fixed_ppo
