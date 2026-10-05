// Build the exact swept-point set once per environment, then query it for
// each navigation candidate. Keep shapes fixed for the current 8x10 pooled
// camera grid and eight-frame ring.
constant uint NAV_MEMORY_FRAMES = 8;
constant uint NAV_MEMORY_CELLS = 80;
constant uint NAV_MEMORY_POINTS = NAV_MEMORY_FRAMES * NAV_MEMORY_CELLS;
constant uint NAV_MEMORY_CANDIDATES = 85;

kernel void nav_memory_build_points(device const RLPhysicsState* states [[buffer(0)]],
                                    device const SimRun* runs [[buffer(1)]],
                                    device const float* sensors [[buffer(2)]],
                                    device const float* poses [[buffer(3)]],
                                    device float* points [[buffer(4)]],
                                    constant RLPhysicsParams& physics [[buffer(5)]],
                                    constant SimConfig& cfg [[buffer(6)]],
                                    uint i [[thread_position_in_grid]]) {
    const uint n = i / NAV_MEMORY_POINTS;
    const uint local = i % NAV_MEMORY_POINTS;
    if (n >= cfg.n) return;

    const uint back = local / NAV_MEMORY_CELLS;
    const uint cell = local % NAV_MEMORY_CELLS;
    const uint row = cell / 10;
    const uint col = cell % 10;
    const uint available = runs[n].steps / cfg.sensor_period;
    const uint sensor_delay = sim_sensor_delay(cfg,n);
    const uint latest = available > sensor_delay ? available - sensor_delay : 0;
    const uint valid = available >= sensor_delay ? min(latest + 1, NAV_MEMORY_FRAMES - sensor_delay) : 0;
    // Mode 18 tests temporal depth while keeping geometry guidance enabled.
    const uint memory_valid = cfg.mode == 18 ? min(valid, 1u) : valid;
    const uint out = (n * NAV_MEMORY_POINTS + local) * 4;

    if (back >= memory_valid) {
        points[out + 0] = 0.0f;
        points[out + 1] = 0.0f;
        points[out + 2] = 0.0f;
        points[out + 3] = -1.0f;
        return;
    }

    const uint frame = (latest + NAV_MEMORY_FRAMES - back) % NAV_MEMORY_FRAMES;
    const device float* ranges = sensors + (n * NAV_MEMORY_FRAMES + frame) * 320;
    const uint px = row * 2 * 20 + col * 2;
    uint hit = px;
    float range = ranges[px];
    const uint neighbors[3] = {px + 1, px + 20, px + 21};
    for (uint j = 0; j < 3; ++j) {
        const float candidate = ranges[neighbors[j]];
        if (candidate < range) { range = candidate; hit = neighbors[j]; }
    }

    if (range <= 0.01f || range >= 11.9f) {
        points[out + 0] = 0.0f;
        points[out + 1] = 0.0f;
        points[out + 2] = 0.0f;
        points[out + 3] = -1.0f;
        return;
    }

    const device float* old_pose = poses + (n * NAV_MEMORY_FRAMES + frame) * 12;
    const float ry = 1.0f - (float(hit % 20) + 0.5f) / 10.0f;
    const float rz = NAV_SENSOR_ACTIVE_TAN_V * (1.0f - (float(hit / 20) + 0.5f) / 8.0f);
    const float inv = 1.0f / sqrt(1.0f + ry * ry + rz * rz);
    const float ray[3] = {inv, ry * inv, rz * inv};
    const float wx = old_pose[3] * ray[0] + old_pose[4] * ray[1] + old_pose[5] * ray[2];
    const float wy = old_pose[6] * ray[0] + old_pose[7] * ray[1] + old_pose[8] * ray[2];
    const float wz = old_pose[9] * ray[0] + old_pose[10] * ray[1] + old_pose[11] * ray[2];

    const RLPhysicsState state = states[n];
    float current_rotation[9];
    sim_rotation(state.orientation_wxyz, current_rotation);
    const float dx = old_pose[0] + range * wx - state.position[0];
    const float dy = old_pose[1] + range * wy - state.position[1];
    const float dz = old_pose[2] + range * wz - state.position[2];
    points[out + 0] = current_rotation[0] * dx + current_rotation[3] * dy + current_rotation[6] * dz;
    points[out + 1] = current_rotation[1] * dx + current_rotation[4] * dy + current_rotation[7] * dz;
    points[out + 2] = current_rotation[2] * dx + current_rotation[5] * dy + current_rotation[8] * dz;
    const float age = float(back) * float(cfg.sensor_period) * physics.dt * float(cfg.substeps);
    // A delayed newest frame is already old before any history lookback.
    const float capture_age = float(runs[n].steps - latest * cfg.sensor_period) *
                              physics.dt * float(cfg.substeps);
    points[out + 3] = nav_memory_point_radius(range, age, capture_age);
}

kernel void nav_memory_candidate_clearance(device const RLPhysicsState* states [[buffer(0)]],
                                           device const WWorld* worlds [[buffer(1)]],
                                           device const float* points [[buffer(2)]],
                                           device float* clearances [[buffer(3)]],
                                           constant SimConfig& cfg [[buffer(4)]],
                                           uint i [[thread_position_in_grid]]) {
    const uint n = i / NAV_MEMORY_CANDIDATES;
    const uint candidate = i % NAV_MEMORY_CANDIDATES;
    if (n >= cfg.n) return;

    const RLPhysicsState state = states[n];
    float rotation[9];
    sim_rotation(state.orientation_wxyz, rotation);
    const float dx = worlds[n].goal[0] - state.position[0];
    const float dy = worlds[n].goal[1] - state.position[1];
    const float dz = worlds[n].goal[2] - state.position[2];
    const float distance = max(sqrt(dx * dx + dy * dy + dz * dz), 1.0e-6f);
    const float goal[3] = {
        (rotation[0] * dx + rotation[3] * dy + rotation[6] * dz) / distance,
        (rotation[1] * dx + rotation[4] * dy + rotation[7] * dz) / distance,
        (rotation[2] * dx + rotation[5] * dy + rotation[8] * dz) / distance
    };
    const float goal_norm = sqrt(goal[0] * goal[0] + goal[1] * goal[1] + goal[2] * goal[2]);
    const float goal_unit[3] = {goal[0] / max(goal_norm, 1.0e-6f),
                                goal[1] / max(goal_norm, 1.0e-6f),
                                goal[2] / max(goal_norm, 1.0e-6f)};
    const float body_velocity[3] = {
        (rotation[0] * state.linear_velocity[0] + rotation[3] * state.linear_velocity[1] + rotation[6] * state.linear_velocity[2]) / 4.0f * 4.0f,
        (rotation[1] * state.linear_velocity[0] + rotation[4] * state.linear_velocity[1] + rotation[7] * state.linear_velocity[2]) / 4.0f * 4.0f,
        (rotation[2] * state.linear_velocity[0] + rotation[5] * state.linear_velocity[1] + rotation[8] * state.linear_velocity[2]) / 4.0f * 4.0f
    };

    float direction[3];
    if (candidate < NAV_MEMORY_CELLS) {
        float pooled_ray[3];
        nav_ray(candidate / 10, candidate % 10, NAV_SENSOR_ACTIVE_TAN_V, pooled_ray);
        direction[0] = pooled_ray[0]; direction[1] = pooled_ray[1]; direction[2] = pooled_ray[2];
    } else if (candidate == 80) {
        direction[0] = goal_unit[0]; direction[1] = goal_unit[1]; direction[2] = goal_unit[2];
    } else if (candidate == 81) {
        direction[0] = 0.0f; direction[1] = 0.0f; direction[2] = 1.0f;
    } else if (candidate == 82) {
        direction[0] = 0.0f; direction[1] = 0.0f; direction[2] = -1.0f;
    } else if (candidate == 83) {
        direction[0] = 0.0f; direction[1] = 1.0f; direction[2] = 0.0f;
    } else {
        direction[0] = 0.0f; direction[1] = -1.0f; direction[2] = 0.0f;
    }

    float sweep[3] = {1.2f * direction[0] + 0.35f * body_velocity[0],
                      1.2f * direction[1] + 0.35f * body_velocity[1],
                      1.2f * direction[2] + 0.35f * body_velocity[2]};
    const float sweep_norm = sqrt(sweep[0] * sweep[0] + sweep[1] * sweep[1] + sweep[2] * sweep[2]);
    for (uint j = 0; j < 3; ++j) sweep[j] /= max(sweep_norm, 1.0e-6f);

    float nearest = 3.0f;
    const device float* env_points = points + n * NAV_MEMORY_POINTS * 4;
    for (uint j = 0; j < NAV_MEMORY_POINTS; ++j) {
        const device float* point = env_points + j * 4;
        const float radius = point[3];
        if (radius < 0.0f) continue;
        const float along = point[0] * sweep[0] + point[1] * sweep[1] + point[2] * sweep[2];
        if (along <= 0.0f || along > 3.0f + radius) continue;
        const float lateral2 = max(point[0] * point[0] + point[1] * point[1] + point[2] * point[2] - along * along, 0.0f);
        const float radius2 = radius * radius;
        if (lateral2 < radius2) nearest = min(nearest, max(along - sqrt(radius2 - lateral2), 0.0f));
    }
    clearances[n * NAV_MEMORY_CANDIDATES + candidate] = nearest;
}
