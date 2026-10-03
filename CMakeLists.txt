cmake_minimum_required(VERSION 3.20)
project(metal_nav LANGUAGES CXX OBJCXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_OBJCXX_STANDARD 17)
function(add_navigation_target target actor_observation_dim)
    add_executable(${target} main.mm)
    target_compile_options(${target} PRIVATE -O3 -fobjc-arc -Wall -Wextra)
    target_compile_definitions(${target} PRIVATE
        SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
        FIXED_PPO_ACTOR_OBS_DIM=${actor_observation_dim})
    target_link_libraries(${target} PRIVATE "-framework Foundation" "-framework Metal" "-framework Accelerate")
endfunction()

# Preserve the full sensor input as the default trainer. The pooled trainer
# uses the same rays and physics with 2x2 min pooling before the actor MLP.
add_navigation_target(metal_nav 661)
add_navigation_target(metal_nav_pooled 181)

# Learned residual over a deployable depth/goal geometric prior.
add_navigation_target(metal_nav_guided 184)

# Preserve the guided 184-feature prefix and append both full-resolution
# 16x20 depth frames before the same final three geometry hints.
add_navigation_target(metal_nav_raw_guided 824)

# Cold experiment: source-only expert demonstrations and actor warmstarts.
add_executable(metal_nav_imitation navigation_imitation.mm)
target_compile_options(metal_nav_imitation PRIVATE -O3 -fobjc-arc -Wall -Wextra)
target_compile_definitions(metal_nav_imitation PRIVATE
    SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}" FIXED_PPO_ACTOR_OBS_DIM=184)
target_link_libraries(metal_nav_imitation PRIVATE
    "-framework Foundation" "-framework Metal" "-framework Accelerate")
