// Offline attribution of retained measured rays to the terminal contact object.
// This geometry truth is a grader input, never a policy input.
#include "world.hpp"
#include "sensor_profile.hpp"
#include <cstdint>
#include <iostream>

struct VisibilityQuery {
    WWorld world;
    int32_t object, boundary_face;
    float capture_time;
    float pose[12];
    float measured[320];
};
static_assert(sizeof(VisibilityQuery) == 2016, "visibility wire layout changed");

static float target_range(const VisibilityQuery& query, WVec origin, WVec ray) {
    if (query.object >= 0) {
        const WObstacle& obstacle = query.world.obstacles[query.object];
        const WVec center = wc(obstacle, query.capture_time);
        if (obstacle.kind == 0)
            return wray_box(origin, ray, center, wv(obstacle.size[0], obstacle.size[1], obstacle.size[2]));
        if (obstacle.kind == 2)
            return wray_cylinder(origin, ray, center, obstacle.size[0], obstacle.size[2]);
        return wray_sphere(origin, ray, center, obstacle.size[0]);
    }
    // Attribute room contacts to the contacted face, not any visible room wall.
    const int axis = query.boundary_face / 2;
    const float coordinates[3] = {origin.x, origin.y, origin.z};
    const float directions[3] = {ray.x, ray.y, ray.z};
    const float planes[6] = {-2, 14, -5, 5, 0, 5};
    if (std::fabs(directions[axis]) < 1e-8f) return 1000;
    const float distance = (planes[query.boundary_face] - coordinates[axis]) / directions[axis];
    return distance >= 0 ? distance : 1000;
}

int main() {
    VisibilityQuery query{};
    while (std::cin.read(reinterpret_cast<char*>(&query), sizeof(query))) {
        if (query.world.count > 16 || query.object >= int(query.world.count) || query.object < -1 ||
            (query.object == -1 && (query.boundary_face < 0 || query.boundary_face > 5))) return 1;
        bool attributed[320]{};
        uint32_t geometric_hits = 0, measured_hits = 0, pooled_hits = 0;
        float maximum_range_error = 0;
        const WVec origin = wv(query.pose[0], query.pose[1], query.pose[2]);
        for (uint32_t pixel = 0; pixel < 320; ++pixel) {
            const WVec local = nav_sensor_pixel_ray(NAV_SENSOR_TAN_H, NAV_SENSOR_ACTIVE_TAN_V, pixel);
            const WVec ray = wv(query.pose[3]*local.x + query.pose[4]*local.y + query.pose[5]*local.z,
                                query.pose[6]*local.x + query.pose[7]*local.y + query.pose[8]*local.z,
                                query.pose[9]*local.x + query.pose[10]*local.y + query.pose[11]*local.z);
            const float target = target_range(query, origin, ray);
            const float first = wray(query.world, origin, ray, query.capture_time);
            maximum_range_error = std::max(maximum_range_error, std::fabs(first-query.measured[pixel]));
            const bool hits = target < 11.9f && std::fabs(target-first) < .001f;
            geometric_hits += hits;
            // Combined profile has 3 cm Gaussian noise; this attribution permits
            // 15 cm error, but excludes missing/max-range or invalid depth.
            attributed[pixel] = hits && query.measured[pixel] > .01f &&
                query.measured[pixel] < 11.9f && std::fabs(query.measured[pixel]-target) <= .15f;
            measured_hits += attributed[pixel];
        }
        for (uint32_t row = 0; row < 8; ++row) for (uint32_t col = 0; col < 10; ++col) {
            const uint32_t pixel = row*40 + col*2;
            uint32_t chosen = pixel;
            for (uint32_t neighbor : {pixel+1, pixel+20, pixel+21})
                if (query.measured[neighbor] < query.measured[chosen]) chosen = neighbor;
            pooled_hits += attributed[chosen];
        }
        // Counterfactual angular density at the SAME measured pose and FOV.
        // These 5,120 geometric rays are not sensor readings or new flights.
        uint32_t dense_geometric_hits = 0;
        for (uint32_t row = 0; row < 64; ++row) for (uint32_t col = 0; col < 80; ++col) {
            const WVec local = wn(wv(1, NAV_SENSOR_TAN_H*(1-(float(col)+.5f)/40),
                                       NAV_SENSOR_ACTIVE_TAN_V*(1-(float(row)+.5f)/32)));
            const WVec ray = wv(query.pose[3]*local.x + query.pose[4]*local.y + query.pose[5]*local.z,
                                query.pose[6]*local.x + query.pose[7]*local.y + query.pose[8]*local.z,
                                query.pose[9]*local.x + query.pose[10]*local.y + query.pose[11]*local.z);
            const float target = target_range(query, origin, ray);
            const float first = wray(query.world, origin, ray, query.capture_time);
            dense_geometric_hits += target < 11.9f && std::fabs(target-first) < .001f;
        }
        std::cout << geometric_hits << ',' << measured_hits << ',' << pooled_hits
                  << ',' << dense_geometric_hits << ',' << maximum_range_error << '\n';
    }
    return std::cin.eof() && std::cin.gcount() == 0 ? 0 : 1;
}
