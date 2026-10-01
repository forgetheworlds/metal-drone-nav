// Offline fixture generator. Compile against RAPTOR's pinned rl-tools submodule:
// clang++ -std=c++17 -O2 -I/path/to/rl-tools/include reference.cpp -o reference
// ./reference assets/physics.bin
#include "physics.hpp"
#include <rl_tools/operations/cpu.h>
#include <rl_tools/rl/environments/l2f/operations_generic.h>
#include <rl_tools/rl/environments/l2f/operations_cpu.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <iostream>

namespace rlt = rl_tools;
using T = float;
using TI = std::size_t;
using DEVICE = rlt::devices::DefaultCPU;
using FACTORY = rlt::rl::environments::l2f::parameters::DEFAULT_PARAMETERS_FACTORY<T, TI, 5>;
using ENV = rlt::rl::environments::Multirotor<rlt::rl::environments::l2f::Specification<T, TI, FACTORY::STATIC_PARAMETERS>>;
using ACTION_SPEC = rlt::matrix::Specification<T, TI, 1, 4, false>;

struct FixtureHeader {
    char magic[8];
    uint32_t version;
    uint32_t cases;
    uint32_t horizon;
};
static_assert(sizeof(FixtureHeader) == 20);

static RLPhysicsParams pack_params(const ENV::Parameters& source) {
    RLPhysicsParams p{};
    const auto& d = source.dynamics;
    for (int r = 0; r < 4; ++r) {
        for (int i = 0; i < 3; ++i) {
            p.rotor_positions[3*r+i] = d.rotor_positions[r][i];
            p.rotor_thrust_directions[3*r+i] = d.rotor_thrust_directions[r][i];
            p.rotor_torque_directions[3*r+i] = d.rotor_torque_directions[r][i];
            p.rotor_thrust_coefficients[3*r+i] = d.rotor_thrust_coefficients[r][i];
        }
        p.rotor_torque_constants[r] = d.rotor_torque_constants[r];
        p.rotor_time_constants_rising[r] = d.rotor_time_constants_rising[r];
        p.rotor_time_constants_falling[r] = d.rotor_time_constants_falling[r];
    }
    for (int i = 0; i < 3; ++i) {
        p.gravity_world[i] = d.gravity[i];
        for (int j = 0; j < 3; ++j) {
            p.inertia[3*i+j] = d.J[i][j];
            p.inertia_inverse[3*i+j] = d.J_inv[i][j];
        }
    }
    p.mass = d.mass;
    p.action_min = d.action_limit.min;
    p.action_max = d.action_limit.max;
    p.dt = source.integration.dt;
    p.position_limit = FACTORY::STATIC_PARAMETERS::STATE_LIMIT_POSITION;
    p.velocity_limit = FACTORY::STATIC_PARAMETERS::STATE_LIMIT_VELOCITY;
    p.angular_velocity_limit = FACTORY::STATIC_PARAMETERS::STATE_LIMIT_ANGULAR_VELOCITY;
    return p;
}

static RLPhysicsState pack_state(const ENV::State& s) {
    RLPhysicsState out{};
    for (int i = 0; i < 3; ++i) {
        out.position[i] = s.position[i];
        out.linear_velocity[i] = s.linear_velocity[i];
        out.angular_velocity_body[i] = s.angular_velocity[i];
    }
    for (int i = 0; i < 4; ++i) {
        out.orientation_wxyz[i] = s.orientation[i];
        out.rpm[i] = s.rpm[i];
    }
    return out;
}

static void write_exact(FILE* f, const void* data, size_t size) {
    if (std::fwrite(data, 1, size, f) != size) {
        std::perror("fixture write");
        std::exit(2);
    }
}

int main(int argc, char** argv) {
    const char* path = argc > 1 ? argv[1] : "assets/physics.bin";
    FILE* f = std::fopen(path, "wb");
    if (!f) { std::perror(path); return 1; }
    constexpr uint32_t CASES = 64;
    constexpr uint32_t HORIZON = 8;
    FixtureHeader header{{'R','L','P','H','Y','S','1','\0'}, 1, CASES, HORIZON};

    DEVICE device;
    rlt::init(device);
    ENV env;
    auto params = FACTORY::nominal_parameters;
    RLPhysicsParams packed_params = pack_params(params);
    const RLPhysicsParams named_profile = rl_physics_crazyflie_default();
    const float* upstream_values = reinterpret_cast<const float*>(&packed_params);
    const float* profile_values = reinterpret_cast<const float*>(&named_profile);
    for (size_t i = 0; i < sizeof(RLPhysicsParams) / sizeof(float); ++i) {
        if (std::fabs(upstream_values[i] - profile_values[i]) > 1e-6f) {
            std::cerr << "Named Crazyflie profile differs from RAPTOR-pinned L2F at scalar " << i << "\n";
            return 3;
        }
    }
    write_exact(f, &header, sizeof(header));
    write_exact(f, &packed_params, sizeof(packed_params));

    rlt::Matrix<ACTION_SPEC> action;
    using RNG = DEVICE::SPEC::RANDOM::ENGINE<>;
    RNG rng;
    rlt::init(device, rng, TI(13));
    for (uint32_t c = 0; c < CASES; ++c) {
        ENV::State state{};
        state.position[0] = .05f * c - .12f;
        state.position[1] = -.03f * c;
        state.position[2] = .08f;
        const float angle = .11f * (c + 1);
        state.orientation[0] = std::cos(angle / 2);
        state.orientation[1] = std::sin(angle / 2) * .4f;
        state.orientation[2] = std::sin(angle / 2) * .7f;
        state.orientation[3] = std::sin(angle / 2) * .2f;
        const float qnorm = std::sqrt(state.orientation[0]*state.orientation[0] + state.orientation[1]*state.orientation[1] +
                                      state.orientation[2]*state.orientation[2] + state.orientation[3]*state.orientation[3]);
        for (int i = 0; i < 4; ++i) state.orientation[i] /= qnorm;
        state.linear_velocity[0] = .1f * (int(c % 9) - 3);
        state.linear_velocity[1] = -.07f * (c % 13);
        state.linear_velocity[2] = .02f * (c % 7);
        state.angular_velocity[0] = .13f * (int(c % 9)) - .3f;
        state.angular_velocity[1] = .04f * (int(c % 5) - 2);
        state.angular_velocity[2] = -.09f * (c % 11);
        for (int i = 0; i < 4; ++i) state.rpm[i] = .35f + .07f * ((c + i) % 5);
        for (int i = 0; i < 3; ++i) state.force[i] = .02f * (int(c % 7) - i - 1);
        state.rotor_history_step = 0;

        for (uint32_t t = 0; t < HORIZON; ++t) {
            float a[4];
            for (int i = 0; i < 4; ++i) {
                a[i] = (float((c*7 + t*5 + i*3) % 17) - 8.0f) / 8.0f;
                rlt::set(action, 0, i, a[i]);
            }
            const RLPhysicsState packed_state = pack_state(state);
            float wind[3] = {state.force[0], state.force[1], state.force[2]};
            write_exact(f, &packed_state, sizeof(packed_state));
            write_exact(f, a, sizeof(a));
            write_exact(f, wind, sizeof(wind));
            ENV::State next{};
            rlt::step(device, env, params, state, action, next, rng);
            const RLPhysicsState expected = pack_state(next);
            write_exact(f, &expected, sizeof(expected));
            state = next;
        }
    }
    std::fclose(f);
    std::cout << "Wrote official L2F CPU fixtures: " << path << " (" << CASES << " cases x " << HORIZON << " steps)\n";
    return 0;
}
