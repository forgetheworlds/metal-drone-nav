cmake_minimum_required(VERSION 3.20)
project(metal_nav LANGUAGES CXX OBJCXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_OBJCXX_STANDARD 17)
function(add_navigation_target target actor_observation_dim)
    # Optional third argument selects the camera profile (sensor_profile.hpp):
    # 1 = legacy source model (default), 2 = calibrated native RangeFinder.
    set(sensor_profile 1)
    if(ARGC GREATER 2)
        set(sensor_profile ${ARGV2})
    endif()
    add_executable(${target} main.mm)
    target_compile_options(${target} PRIVATE -O3 -fobjc-arc -Wall -Wextra)
    target_compile_definitions(${target} PRIVATE
        SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
        FIXED_PPO_ACTOR_OBS_DIM=${actor_observation_dim}
        NAV_SENSOR_PROFILE=${sensor_profile})
    target_link_libraries(${target} PRIVATE "-framework Foundation" "-framework Metal" "-framework Accelerate")
endfunction()

# Preserve the full sensor input as the default trainer. The pooled trainer
# uses the same rays and physics with 2x2 min pooling before the actor MLP.
add_navigation_target(metal_nav 661)
add_navigation_target(metal_nav_pooled 181)

# Learned residual over a deployable depth/goal geometric prior.
add_navigation_target(metal_nav_guided 184)

# Calibrated camera profile for the selected 184-input guided actor, so the
# frozen legacy checkpoint can be scored against the native sensor contract
# without changing its weights.
add_navigation_target(metal_nav_guided_native 184 2)

# Preserve the guided 184-feature prefix and append both full-resolution
# 16x20 depth frames before the same final three geometry hints.
add_navigation_target(metal_nav_raw_guided 824)

# Opt-in calibrated camera profile (sensor_profile.hpp): identical actor,
# guidance, physics and world, but the 16x20 rays, vertical half-tangent and
# sensor mount match the measured native Webots RangeFinder. Separate binary so
# the legacy target above stays bit-identical and no contract can mix silently.
add_navigation_target(metal_nav_raw_guided_native 824 2)

# Cold experiment: source-only expert demonstrations and actor warmstarts.
add_executable(metal_nav_imitation navigation_imitation.mm)
target_compile_options(metal_nav_imitation PRIVATE -O3 -fobjc-arc -Wall -Wextra)
target_compile_definitions(metal_nav_imitation PRIVATE
    SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}" FIXED_PPO_ACTOR_OBS_DIM=184)
target_link_libraries(metal_nav_imitation PRIVATE
    "-framework Foundation" "-framework Metal" "-framework Accelerate")

# Local waypoint training remains an experimental component; defaults unchanged.
add_executable(metal_nav_waypoint navigation_waypoint_training.mm)
target_compile_options(metal_nav_waypoint PRIVATE -O3 -fobjc-arc -Wall -Wextra)
target_compile_definitions(metal_nav_waypoint PRIVATE
    SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}" FIXED_PPO_ACTOR_OBS_DIM=184)
target_link_libraries(metal_nav_waypoint PRIVATE
    "-framework Foundation" "-framework Metal" "-framework Accelerate")
