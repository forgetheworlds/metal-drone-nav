// Offline collision attribution. Uses the same SDF and bounded-motion source
// as Metal; it never feeds contact-object identity back into the actor.
#include "world.hpp"
#include <cstdint>
#include <iostream>

struct ContactQuery {
    WWorld world;
    float position[3];
    float time;
};
static_assert(sizeof(ContactQuery) == 692, "contact-query wire layout changed");

int main() {
    ContactQuery query{};
    while (std::cin.read(reinterpret_cast<char*>(&query), sizeof(query))) {
        if (query.world.count > 16) return 1;
        const WVec position = wv(query.position[0], query.position[1], query.position[2]);
        WWorld room = query.world;
        room.count = 0;
        float nearest = wclearance(room, position, query.time);
        int object = -1, kind = -1;
        for (uint32_t index = 0; index < query.world.count; ++index) {
            WWorld isolated = room;
            isolated.count = 1;
            isolated.obstacles[0] = query.world.obstacles[index];
            const float clearance = wclearance(isolated, position, query.time);
            if (clearance < nearest) {
                nearest = clearance;
                object = int(index);
                kind = int(isolated.obstacles[0].kind);
            }
        }
        const float actual = wclearance(query.world, position, query.time);
        if (!std::isfinite(actual) || std::fabs(actual - nearest) > 1e-6f) return 1;
        std::cout << object << ',' << kind << ',' << actual << '\n';
    }
    return std::cin.eof() && std::cin.gcount() == 0 ? 0 : 1;
}
