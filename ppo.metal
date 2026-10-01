#include <metal_stdlib>
using namespace metal;

// Keep these values and flattened parameter offsets in sync with ppo.hpp.
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

constant uint PPO_ACTOR_OBS = FIXED_PPO_ACTOR_OBS_DIM;
constant uint PPO_CRITIC_OBS = FIXED_PPO_CRITIC_OBS_DIM;
constant uint PPO_HIDDEN = FIXED_PPO_HIDDEN_DIM;
constant uint PPO_ACTIONS = FIXED_PPO_ACTION_DIM;
constant uint PPO_ACTOR_B1 = PPO_HIDDEN * PPO_ACTOR_OBS;
constant uint PPO_ACTOR_W2 = PPO_ACTOR_B1 + PPO_HIDDEN;
constant uint PPO_ACTOR_B2 = PPO_ACTOR_W2 + PPO_ACTIONS * PPO_HIDDEN;
constant uint PPO_ACTOR_LOG_STD = PPO_ACTOR_B2 + PPO_ACTIONS;
constant uint PPO_ACTOR_PARAMS = PPO_ACTOR_LOG_STD + PPO_ACTIONS;
constant uint PPO_CRITIC_B1 = PPO_HIDDEN * PPO_CRITIC_OBS;
constant uint PPO_CRITIC_W2 = PPO_CRITIC_B1 + PPO_HIDDEN;
constant uint PPO_CRITIC_B2 = PPO_CRITIC_W2 + PPO_HIDDEN;
constant uint PPO_CRITIC_PARAMS = PPO_CRITIC_B2 + 1;
constant float PPO_LOG_TWO_PI = 1.8378770664093453f;

struct PpoAdamConfig {
    float learning_rate;
    float beta1;
    float beta2;
    float epsilon;
    float weight_decay;
    uint step; // one-based update number
};

inline float ppo_clamp(float x, float lo, float hi) { return min(max(x, lo), hi); }

// These helpers can also be called from a fused rollout kernel after including
// this source in the same Metal compilation unit.
inline void ppo_actor_mean(device const float* params, thread const float* obs,
                           thread float* hidden, thread float* mean) {
    for (uint h = 0; h < PPO_HIDDEN; ++h) {
        float z = params[PPO_ACTOR_B1 + h];
        const uint row = h * PPO_ACTOR_OBS;
        for (uint i = 0; i < PPO_ACTOR_OBS; ++i) z += params[row + i] * obs[i];
        hidden[h] = tanh(z);
    }
    for (uint a = 0; a < PPO_ACTIONS; ++a) {
        float z = params[PPO_ACTOR_B2 + a];
        const uint row = PPO_ACTOR_W2 + a * PPO_HIDDEN;
        for (uint h = 0; h < PPO_HIDDEN; ++h) z += params[row + h] * hidden[h];
        if(PPO_ACTOR_OBS==184 && a<3)z+=obs[PPO_ACTOR_OBS-3+a];
        mean[a] = z;
    }
}

// Scalar reference that reads observations directly from device storage. Use
// this form in inference/rollout kernels so they do not reserve a 661-float
// thread-local observation array merely to call the actor.


inline float ppo_critic_value(device const float* params, thread const float* obs,
                              thread float* hidden) {
    for (uint h = 0; h < PPO_HIDDEN; ++h) {
        float z = params[PPO_CRITIC_B1 + h];
        const uint row = h * PPO_CRITIC_OBS;
        for (uint i = 0; i < PPO_CRITIC_OBS; ++i) z += params[row + i] * obs[i];
        hidden[h] = tanh(z);
    }
    float value = params[PPO_CRITIC_B2];
    for (uint h = 0; h < PPO_HIDDEN; ++h) value += params[PPO_CRITIC_W2 + h] * hidden[h];
    return value;
}

kernel void ppo_actor_forward(device const float* observations [[buffer(0)]],
                              device const float* params [[buffer(1)]],
                              device float* hidden_out [[buffer(2)]],
                              device float* means [[buffer(3)]],
                              constant uint& batch_size [[buffer(4)]],
                              uint n [[thread_position_in_grid]]) {
    if (n >= batch_size) return;
    thread float hidden[PPO_HIDDEN];
    thread float obs[PPO_ACTOR_OBS];
    const uint base = n * PPO_ACTOR_OBS;
    for (uint i = 0; i < PPO_ACTOR_OBS; ++i) obs[i] = observations[base + i];
    thread float mean[PPO_ACTIONS];
    ppo_actor_mean(params, obs, hidden, mean);
    for (uint h = 0; h < PPO_HIDDEN; ++h) hidden_out[n * PPO_HIDDEN + h] = hidden[h];
    for (uint a = 0; a < PPO_ACTIONS; ++a) means[n * PPO_ACTIONS + a] = mean[a];
}



// Batched FP32 SIMD-group matmul for the actor's first layer. Each SIMD group
// computes one [8 samples, 8 hidden units] tile of observations[B,661] times
// W1-transpose[661,64]. The final four observation features are zero-masked.




// Fused batched actor forward. Eight SIMD groups produce all 64 hidden units
// for eight observations, then the threadgroup applies the 4-unit output head
// while the hidden tile is still local. This avoids the 661-float observation
// copy, a second dispatch, and a global hidden-to-head round trip.
kernel void ppo_actor_forward_simd_fused(device const float* observations [[buffer(0)]],
                                         device const float* params [[buffer(1)]],
                                         device float* hidden_out [[buffer(2)]],
                                         device float* means [[buffer(3)]],
                                         constant uint& batch_size [[buffer(4)]],
                                         uint tid [[thread_index_in_threadgroup]],
                                         uint sg [[simdgroup_index_in_threadgroup]],
                                         uint3 group_position [[threadgroup_position_in_grid]]) {
    constexpr uint tile = 8;
    constexpr uint simdgroups_per_threadgroup = 8;
    const uint sample_base = group_position.x * tile;
    threadgroup float tile_a[64];
    threadgroup float tile_b[simdgroups_per_threadgroup * 64];
    threadgroup float tile_c[simdgroups_per_threadgroup * 64];
    threadgroup float tile_h[simdgroups_per_threadgroup * 64];
    simdgroup_float8x8 acc(0.0f);

    for (uint k_base = 0; k_base < PPO_ACTOR_OBS; k_base += tile) {
        for (uint cell = tid; cell < 64; cell += 256) {
            const uint row = cell / tile;
            const uint col = cell % tile;
            const uint sample = sample_base + row;
            const uint feature = k_base + col;
            tile_a[cell] = sample < batch_size && feature < PPO_ACTOR_OBS
                         ? observations[sample * PPO_ACTOR_OBS + feature] : 0.0f;
        }
        for (uint index = tid; index < simdgroups_per_threadgroup * 64; index += 256) {
            const uint target_sg = index / 64;
            const uint cell = index % 64;
            const uint feature = k_base + cell / tile;
            const uint hidden = target_sg * tile + cell % tile;
            tile_b[index] = feature < PPO_ACTOR_OBS && hidden < PPO_HIDDEN
                          ? params[hidden * PPO_ACTOR_OBS + feature] : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_float8x8 a, b;
        simdgroup_load(a, tile_a, 8);
        simdgroup_load(b, tile_b + sg * 64, 8);
        simdgroup_multiply_accumulate(acc, a, b, acc);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    simdgroup_store(acc, tile_c + sg * 64, 8);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint index = tid; index < simdgroups_per_threadgroup * 64; index += 256) {
        const uint hidden_tile = index / 64;
        const uint cell = index % 64;
        const uint sample_local = cell / tile;
        const uint hidden = hidden_tile * tile + cell % tile;
        const uint sample = sample_base + sample_local;
        const float activation = tanh(tile_c[index] + params[PPO_ACTOR_B1 + hidden]);
        tile_h[sample_local * PPO_HIDDEN + hidden] = activation;
        if (sample < batch_size) hidden_out[sample * PPO_HIDDEN + hidden] = activation;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < tile * PPO_ACTIONS) {
        const uint sample_local = tid / PPO_ACTIONS;
        const uint action = tid % PPO_ACTIONS;
        const uint sample = sample_base + sample_local;
        float mean = params[PPO_ACTOR_B2 + action];
        const uint row = PPO_ACTOR_W2 + action * PPO_HIDDEN;
        for (uint h = 0; h < PPO_HIDDEN; ++h)
            mean += params[row + h] * tile_h[sample_local * PPO_HIDDEN + h];
        if(PPO_ACTOR_OBS==184 && action<3 && sample<batch_size)mean+=observations[sample*PPO_ACTOR_OBS+PPO_ACTOR_OBS-3+action];
        if (sample < batch_size) means[sample * PPO_ACTIONS + action] = mean;
    }
}

kernel void ppo_critic_forward(device const float* observations [[buffer(0)]],
                               device const float* params [[buffer(1)]],
                               device float* hidden_out [[buffer(2)]],
                               device float* values [[buffer(3)]],
                               constant uint& batch_size [[buffer(4)]],
                               uint n [[thread_position_in_grid]]) {
    if (n >= batch_size) return;
    thread float hidden[PPO_HIDDEN];
    thread float obs[PPO_CRITIC_OBS];
    const uint base = n * PPO_CRITIC_OBS;
    for (uint i = 0; i < PPO_CRITIC_OBS; ++i) obs[i] = observations[base + i];
    values[n] = ppo_critic_value(params, obs, hidden);
    for (uint h = 0; h < PPO_HIDDEN; ++h) hidden_out[n * PPO_HIDDEN + h] = hidden[h];
}

// Input arrays use time-major [time, environment] order. A truncation uses its
// supplied next value for the TD residual but clears the recursive GAE carry.
kernel void ppo_gae(device const float* rewards [[buffer(0)]],
                    device const float* values [[buffer(1)]],
                    device const float* next_values [[buffer(2)]],
                    device const uchar* terminated [[buffer(3)]],
                    device const uchar* truncated [[buffer(4)]],
                    device float* advantages [[buffer(5)]],
                    device float* returns [[buffer(6)]],
                    constant uint& horizon [[buffer(7)]],
                    constant uint& env_count [[buffer(8)]],
                    constant float& gamma [[buffer(9)]],
                    constant float& lambda [[buffer(10)]],
                    uint env [[thread_position_in_grid]]) {
    if (env >= env_count) return;
    float carry = 0.0f;
    for (uint rev = 0; rev < horizon; ++rev) {
        const uint t = horizon - 1 - rev;
        const uint k = t * env_count + env;
        const bool term = terminated[k] != 0;
        const bool boundary = term || truncated[k] != 0;
        const float bootstrap = term ? 0.0f : next_values[k];
        const float delta = rewards[k] + gamma * bootstrap - values[k];
        carry = delta + (boundary ? 0.0f : gamma * lambda * carry);
        advantages[k] = carry;
        returns[k] = carry + values[k];
    }
}

// A single-thread reduction keeps normalization deterministic and avoids
// requiring floating-point atomics. Rollout batches are small enough for this.
kernel void ppo_normalize_advantages(device float* advantages [[buffer(0)]],
                                     constant uint& count [[buffer(1)]],
                                     constant float& epsilon [[buffer(2)]],
                                     uint tid [[thread_position_in_grid]]) {
    if (tid != 0 || count == 0) return;
    float mean = 0.0f;
    for (uint i = 0; i < count; ++i) mean += advantages[i];
    mean /= float(count);
    float variance = 0.0f;
    for (uint i = 0; i < count; ++i) {
        const float d = advantages[i] - mean;
        variance += d * d;
    }
    variance /= float(count);
    const float inv_std = rsqrt(variance + epsilon);
    for (uint i = 0; i < count; ++i) advantages[i] = (advantages[i] - mean) * inv_std;
}

// Emits gradients of the minibatch-mean loss. Binding order:
// action, mean, log_std, old_logp, value, old_value, advantage, return,
// d_mean, d_log_std, d_value, losses[policy,value,entropy,ratio per sample],
// batch_size, clip_epsilon, value_coefficient, entropy_coefficient.
kernel void ppo_sample_loss_grad(device const float* actions [[buffer(0)]],
                                 device const float* means [[buffer(1)]],
                                 device const float* log_std [[buffer(2)]],
                                 device const float* old_logp [[buffer(3)]],
                                 device const float* values [[buffer(4)]],
                                 device const float* old_values [[buffer(5)]],
                                 device const float* advantages [[buffer(6)]],
                                 device const float* target_returns [[buffer(7)]],
                                 device float* d_means [[buffer(8)]],
                                 device float* d_log_stds [[buffer(9)]],
                                 device float* d_values [[buffer(10)]],
                                 device float* losses [[buffer(11)]],
                                 constant uint& batch_size [[buffer(12)]],
                                 constant float& clip_epsilon [[buffer(13)]],
                                 constant float& value_coefficient [[buffer(14)]],
                                 constant float& entropy_coefficient [[buffer(15)]],
                                 uint n [[thread_position_in_grid]]) {
    if (n >= batch_size) return;
    float logp = 0.0f;
    float entropy = 0.0f;
    const uint base = n * PPO_ACTIONS;
    for (uint a = 0; a < PPO_ACTIONS; ++a) {
        const float diff = actions[base + a] - means[base + a];
        const float inv_std = exp(-log_std[a]);
        const float z = diff * inv_std;
        logp += -0.5f * (z * z + 2.0f * log_std[a] + PPO_LOG_TWO_PI);
    }
    const float raw_log_ratio = logp - old_logp[n];
    const float log_ratio = ppo_clamp(raw_log_ratio, -20.0f, 20.0f);
    const float ratio = exp(log_ratio);
    const float adv = advantages[n];
    const float clipped_ratio = ppo_clamp(ratio, 1.0f - clip_epsilon, 1.0f + clip_epsilon);
    const float chosen = min(ratio * adv, clipped_ratio * adv);
    const bool ratio_in_range = raw_log_ratio >= -20.0f && raw_log_ratio <= 20.0f;
    const bool policy_active = ratio_in_range && (adv >= 0.0f ? ratio <= 1.0f + clip_epsilon
                                                              : ratio >= 1.0f - clip_epsilon);
    const float scale = 1.0f / float(max(batch_size, 1u));
    for (uint a = 0; a < PPO_ACTIONS; ++a) {
        const float diff = actions[base + a] - means[base + a];
        const float inv_var = exp(-2.0f * log_std[a]);
        if (policy_active) {
            const float dlogp = -ratio * adv * scale;
            d_means[base + a] = dlogp * diff * inv_var;
            d_log_stds[base + a] = dlogp * (diff * diff * inv_var - 1.0f);
        } else {
            d_means[base + a] = 0.0f;
            d_log_stds[base + a] = 0.0f;
        }
        d_log_stds[base + a] -= entropy_coefficient * scale;
        entropy += -entropy_coefficient * (log_std[a] + 0.5f * (1.0f + PPO_LOG_TWO_PI)) * scale;
    }
    const float vd = values[n] - target_returns[n];
    d_values[n] = value_coefficient * vd * scale;
    losses[n * 4 + 0] = -chosen * scale;
    losses[n * 4 + 1] = 0.5f * value_coefficient * vd * vd * scale;
    losses[n * 4 + 2] = entropy;
    losses[n * 4 + 3] = ratio;
    (void)old_values;
}

// Partial gradients use parameter-major [parameter, sample] layout so the
// subsequent reduction reads contiguous values. Buffer sizes are P * batch.
kernel void ppo_actor_backward(device const float* observations [[buffer(0)]],
                               device const float* hidden [[buffer(1)]],
                               device const float* params [[buffer(2)]],
                               device const float* d_means [[buffer(3)]],
                               device const float* d_log_stds [[buffer(4)]],
                               device float* partial_grad [[buffer(5)]],
                               constant uint& batch_size [[buffer(6)]],
                               uint n [[thread_position_in_grid]]) {
    if (n >= batch_size) return;
    const uint obs_base = n * PPO_ACTOR_OBS;
    const uint hid_base = n * PPO_HIDDEN;
    const uint action_base = n * PPO_ACTIONS;
    for (uint a = 0; a < PPO_ACTIONS; ++a) {
        const float dm = d_means[action_base + a];
        partial_grad[(PPO_ACTOR_B2 + a) * batch_size + n] = dm;
        const uint row = PPO_ACTOR_W2 + a * PPO_HIDDEN;
        for (uint h = 0; h < PPO_HIDDEN; ++h) {
            partial_grad[(row + h) * batch_size + n] = dm * hidden[hid_base + h];
            const float d_hidden = dm * params[row + h] * (1.0f - hidden[hid_base + h] * hidden[hid_base + h]);
            partial_grad[(PPO_ACTOR_B1 + h) * batch_size + n] =
                (a == 0 ? 0.0f : partial_grad[(PPO_ACTOR_B1 + h) * batch_size + n]) + d_hidden;
            const uint first_row = h * PPO_ACTOR_OBS;
            for (uint i = 0; i < PPO_ACTOR_OBS; ++i) {
                const uint k = first_row + i;
                partial_grad[k * batch_size + n] =
                    (a == 0 ? 0.0f : partial_grad[k * batch_size + n]) + d_hidden * observations[obs_base + i];
            }
        }
    }
    for (uint a = 0; a < PPO_ACTIONS; ++a)
        partial_grad[(PPO_ACTOR_LOG_STD + a) * batch_size + n] = d_log_stds[action_base + a];
}

kernel void ppo_critic_backward(device const float* observations [[buffer(0)]],
                                device const float* hidden [[buffer(1)]],
                                device const float* params [[buffer(2)]],
                                device const float* d_values [[buffer(3)]],
                                device float* partial_grad [[buffer(4)]],
                                constant uint& batch_size [[buffer(5)]],
                                uint n [[thread_position_in_grid]]) {
    if (n >= batch_size) return;
    const float dv = d_values[n];
    const uint obs_base = n * PPO_CRITIC_OBS;
    const uint hid_base = n * PPO_HIDDEN;
    partial_grad[PPO_CRITIC_B2 * batch_size + n] = dv;
    for (uint h = 0; h < PPO_HIDDEN; ++h) {
        partial_grad[(PPO_CRITIC_W2 + h) * batch_size + n] = dv * hidden[hid_base + h];
        const float dh = dv * params[PPO_CRITIC_W2 + h] * (1.0f - hidden[hid_base + h] * hidden[hid_base + h]);
        partial_grad[(PPO_CRITIC_B1 + h) * batch_size + n] = dh;
        const uint row = h * PPO_CRITIC_OBS;
        for (uint i = 0; i < PPO_CRITIC_OBS; ++i)
            partial_grad[(row + i) * batch_size + n] = dh * observations[obs_base + i];
    }
}

kernel void ppo_reduce_grads(device const float* partial_grad [[buffer(0)]],
                             device float* grad [[buffer(1)]],
                             constant uint& parameter_count [[buffer(2)]],
                             constant uint& batch_size [[buffer(3)]],
                             uint p [[thread_position_in_grid]]) {
    if (p >= parameter_count) return;
    float sum = 0.0f;
    const uint base = p * batch_size;
    for (uint n = 0; n < batch_size; ++n) sum += partial_grad[base + n];
    grad[p] = sum;
}

// Direct batch-reduced gradients. These avoid a [parameter, sample] scratch
// tensor, which is 43 MB for the actor at batch 256. Each output parameter is
// owned by one thread, so there are no floating-point atomics.
kernel void ppo_actor_hidden_delta(device const float* params [[buffer(0)]],
                                   device const float* d_means [[buffer(1)]],
                                   device const float* hidden [[buffer(2)]],
                                   device float* hidden_delta [[buffer(3)]],
                                   constant uint& batch_size [[buffer(4)]],
                                   uint i [[thread_position_in_grid]]) {
    const uint count = batch_size * PPO_HIDDEN;
    if (i >= count) return;
    const uint n = i / PPO_HIDDEN;
    const uint h = i % PPO_HIDDEN;
    float delta = 0.0f;
    for (uint a = 0; a < PPO_ACTIONS; ++a)
        delta += d_means[n * PPO_ACTIONS + a] * params[PPO_ACTOR_W2 + a * PPO_HIDDEN + h];
    const float activation = hidden[i];
    hidden_delta[i] = delta * (1.0f - activation * activation);
}

kernel void ppo_actor_grad_direct(device const float* observations [[buffer(0)]],
                                 device const float* hidden [[buffer(1)]],
                                 device const float* d_means [[buffer(2)]],
                                 device const float* d_log_stds [[buffer(3)]],
                                 device const float* hidden_delta [[buffer(4)]],
                                 device float* grad [[buffer(5)]],
                                 constant uint& batch_size [[buffer(6)]],
                                 uint p [[thread_position_in_grid]]) {
    if (p >= PPO_ACTOR_PARAMS) return;
    float sum = 0.0f;
    if (p < PPO_ACTOR_B1) {
        const uint h = p / PPO_ACTOR_OBS;
        const uint obs_i = p % PPO_ACTOR_OBS;
        for (uint n = 0; n < batch_size; ++n)
            sum += hidden_delta[n * PPO_HIDDEN + h] * observations[n * PPO_ACTOR_OBS + obs_i];
    } else if (p < PPO_ACTOR_W2) {
        const uint h = p - PPO_ACTOR_B1;
        for (uint n = 0; n < batch_size; ++n) sum += hidden_delta[n * PPO_HIDDEN + h];
    } else if (p < PPO_ACTOR_B2) {
        const uint q = p - PPO_ACTOR_W2;
        const uint a = q / PPO_HIDDEN;
        const uint h = q % PPO_HIDDEN;
        for (uint n = 0; n < batch_size; ++n)
            sum += d_means[n * PPO_ACTIONS + a] * hidden[n * PPO_HIDDEN + h];
    } else if (p < PPO_ACTOR_LOG_STD) {
        const uint a = p - PPO_ACTOR_B2;
        for (uint n = 0; n < batch_size; ++n) sum += d_means[n * PPO_ACTIONS + a];
    } else {
        const uint a = p - PPO_ACTOR_LOG_STD;
        for (uint n = 0; n < batch_size; ++n) sum += d_log_stds[n * PPO_ACTIONS + a];
    }
    grad[p] = sum;
}

// The direct first-layer matrix product uses row-major [sample, hidden_delta]
// and [sample, feature] buffers. Four SIMD groups cooperate per threadgroup;
// each group accumulates one 8x8 [hidden, feature] tile over sample chunks.
// The input feature tail is masked at 661; parameter storage stays row-major.
kernel void ppo_actor_grad_tiled(device const float* observations [[buffer(0)]],
                                device const float* hidden_delta [[buffer(1)]],
                                device float* grad [[buffer(2)]],
                                constant uint& batch_size [[buffer(3)]],
                                uint tid [[thread_index_in_threadgroup]],
                                uint sg [[simdgroup_index_in_threadgroup]],
                                uint3 group_position [[threadgroup_position_in_grid]]) {
    constexpr uint tile = 8;
    constexpr uint input_tiles = (PPO_ACTOR_OBS + tile - 1) / tile;
    constexpr uint simdgroups_per_threadgroup = 4;
    const uint group = group_position.x;
    const uint output_tile = group * simdgroups_per_threadgroup + sg;

    threadgroup float tile_a[simdgroups_per_threadgroup * 64];
    threadgroup float tile_b[simdgroups_per_threadgroup * 64];
    threadgroup float tile_c[simdgroups_per_threadgroup * 64];
    simdgroup_float8x8 acc(0.0f);

    for (uint k_base = 0; k_base < batch_size; k_base += tile) {
        for (uint index = tid; index < simdgroups_per_threadgroup * 64; index += 128) {
            const uint target_sg = index / 64;
            const uint cell = index % 64;
            const uint row = cell / tile;
            const uint col = cell % tile;
            const uint target_tile = group * simdgroups_per_threadgroup + target_sg;
            const uint target_hidden_base = (target_tile / input_tiles) * tile;
            const uint target_input_base = (target_tile % input_tiles) * tile;
            const uint sample = k_base + col;
            const uint hidden_index = target_hidden_base + row;
            tile_a[index] = (sample < batch_size && hidden_index < PPO_HIDDEN)
                          ? hidden_delta[sample * PPO_HIDDEN + hidden_index] : 0.0f;
            const uint input_index = target_input_base + col;
            const uint observation_sample = k_base + row;
            tile_b[index] = (observation_sample < batch_size && input_index < PPO_ACTOR_OBS)
                          ? observations[observation_sample * PPO_ACTOR_OBS + input_index] : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_float8x8 a, b;
        simdgroup_load(a, tile_a + sg * 64, 8);
        simdgroup_load(b, tile_b + sg * 64, 8);
        simdgroup_multiply_accumulate(acc, a, b, acc);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    simdgroup_store(acc, tile_c + sg * 64, 8);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint index = tid; index < simdgroups_per_threadgroup * 64; index += 128) {
        const uint local_sg = index / 64;
        const uint cell = index % 64;
        const uint row = cell / tile;
        const uint col = cell % tile;
        const uint output_tile_index = group * simdgroups_per_threadgroup + local_sg;
        const uint h_tile = output_tile_index / input_tiles;
        const uint i_tile = output_tile_index % input_tiles;
        const uint h = h_tile * tile + row;
        const uint i = i_tile * tile + col;
        if (h < PPO_HIDDEN && i < PPO_ACTOR_OBS)
            grad[h * PPO_ACTOR_OBS + i] = tile_c[index];
    }
}

// Compute W2, b2, and log-standard-deviation gradients. The tiled kernel above
// owns W1, so the trainer can write the head into the matching parameter slice.
kernel void ppo_actor_grad_head(device const float* hidden [[buffer(0)]],
                               device const float* hidden_delta [[buffer(1)]],
                               device const float* d_means [[buffer(2)]],
                               device const float* d_log_stds [[buffer(3)]],
                               device float* grad [[buffer(4)]],
                               constant uint& batch_size [[buffer(5)]],
                               uint q [[thread_position_in_grid]]) {
    constexpr uint head_count = PPO_ACTOR_PARAMS - PPO_ACTOR_B1;
    if (q >= head_count) return;
    float sum = 0.0f;
    if (q < PPO_HIDDEN) {
        for (uint n = 0; n < batch_size; ++n) sum += hidden_delta[n * PPO_HIDDEN + q];
    } else if (q < PPO_HIDDEN + PPO_ACTIONS * PPO_HIDDEN) {
        const uint w2q = q - PPO_HIDDEN;
        const uint a = w2q / PPO_HIDDEN;
        const uint h = w2q % PPO_HIDDEN;
        for (uint n = 0; n < batch_size; ++n)
            sum += d_means[n * PPO_ACTIONS + a] * hidden[n * PPO_HIDDEN + h];
    } else if (q < PPO_HIDDEN + PPO_ACTIONS * PPO_HIDDEN + PPO_ACTIONS) {
        const uint a = q - PPO_HIDDEN - PPO_ACTIONS * PPO_HIDDEN;
        for (uint n = 0; n < batch_size; ++n) sum += d_means[n * PPO_ACTIONS + a];
    } else {
        const uint a = q - PPO_HIDDEN - PPO_ACTIONS * PPO_HIDDEN - PPO_ACTIONS;
        for (uint n = 0; n < batch_size; ++n) sum += d_log_stds[n * PPO_ACTIONS + a];
    }
    grad[PPO_ACTOR_B1 + q] = sum;
}


kernel void ppo_critic_hidden_delta(device const float* params [[buffer(0)]],
                                    device const float* d_values [[buffer(1)]],
                                    device const float* hidden [[buffer(2)]],
                                    device float* hidden_delta [[buffer(3)]],
                                    constant uint& batch_size [[buffer(4)]],
                                    uint i [[thread_position_in_grid]]) {
    const uint count = batch_size * PPO_HIDDEN;
    if (i >= count) return;
    const uint n = i / PPO_HIDDEN;
    const uint h = i % PPO_HIDDEN;
    const float activation = hidden[i];
    hidden_delta[i] = d_values[n] * params[PPO_CRITIC_W2 + h]
                    * (1.0f - activation * activation);
}

kernel void ppo_critic_grad_direct(device const float* observations [[buffer(0)]],
                                  device const float* hidden [[buffer(1)]],
                                  device const float* d_values [[buffer(2)]],
                                  device const float* hidden_delta [[buffer(3)]],
                                  device float* grad [[buffer(4)]],
                                  constant uint& batch_size [[buffer(5)]],
                                  uint p [[thread_position_in_grid]]) {
    if (p >= PPO_CRITIC_PARAMS) return;
    float sum = 0.0f;
    if (p < PPO_CRITIC_B1) {
        const uint h = p / PPO_CRITIC_OBS;
        const uint obs_i = p % PPO_CRITIC_OBS;
        for (uint n = 0; n < batch_size; ++n)
            sum += hidden_delta[n * PPO_HIDDEN + h] * observations[n * PPO_CRITIC_OBS + obs_i];
    } else if (p < PPO_CRITIC_W2) {
        const uint h = p - PPO_CRITIC_B1;
        for (uint n = 0; n < batch_size; ++n) sum += hidden_delta[n * PPO_HIDDEN + h];
    } else if (p < PPO_CRITIC_B2) {
        const uint h = p - PPO_CRITIC_W2;
        for (uint n = 0; n < batch_size; ++n) sum += d_values[n] * hidden[n * PPO_HIDDEN + h];
    } else {
        for (uint n = 0; n < batch_size; ++n) sum += d_values[n];
    }
    grad[p] = sum;
}

kernel void ppo_adam_update(device float* params [[buffer(0)]],
                            device const float* grad [[buffer(1)]],
                            device float* first [[buffer(2)]],
                            device float* second [[buffer(3)]],
                            constant uint& parameter_count [[buffer(4)]],
                            constant PpoAdamConfig& config [[buffer(5)]],
                            uint p [[thread_position_in_grid]]) {
    if (p >= parameter_count || config.step == 0) return;
    const float g = grad[p] + config.weight_decay * params[p];
    const float m = config.beta1 * first[p] + (1.0f - config.beta1) * g;
    const float v = config.beta2 * second[p] + (1.0f - config.beta2) * g * g;
    first[p] = m;
    second[p] = v;
    const float m_hat = m / (1.0f - pow(config.beta1, float(config.step)));
    const float v_hat = v / (1.0f - pow(config.beta2, float(config.step)));
    params[p] -= config.learning_rate * m_hat / (sqrt(v_hat) + config.epsilon);
}
