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
