#pragma once

#ifndef __METAL_VERSION__
#include "physics_domain.hpp"
#endif

// Runtime options for episode-level navigation physics variation.
// The ranges are declared stress assumptions, not identified uncertainties.
struct NavigationRuntimeConfig {
    unsigned int enabled;
    float domain_amplitude;
    unsigned int domain_seed;
    RLPhysicsDomainRange domain_range;
};

// Fixed CPU/MSL layouts for optional training-only geodesic reward shaping.
// The field is immutable and is never part of the deployed actor input.
#define TRAINING_POTENTIAL_SHARED_TYPES
struct TrainingPotentialGridSpec {
    unsigned int nx, ny, nz, level_count, level_stride, version;
    float origin[3];
    float spacing_m, distance_cap_m;
};
struct TrainingPotentialControl {
    unsigned int enabled;
    float scale, gamma;
    unsigned int version;
};
