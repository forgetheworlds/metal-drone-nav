// Offline task grading uses the simulator's exact collision geometry.
#include "world.hpp"
#include <iostream>
#include <vector>

int main() {
    static_assert(sizeof(WWorld)==676,"task bank world ABI changed");
    WWorld world{};
    uint32_t count=0;
    while(std::cin.read(reinterpret_cast<char*>(&world),sizeof(world))) {
        if(!std::cin.read(reinterpret_cast<char*>(&count),sizeof(count))||count>100000)return 1;
        std::vector<float> queries(size_t(count)*4),clearances(count);
        if(!std::cin.read(reinterpret_cast<char*>(queries.data()),queries.size()*sizeof(float)))return 1;
        for(uint32_t i=0;i<count;i++) {
            const float* q=queries.data()+i*4;
            clearances[i]=wclearance(world,wv(q[0],q[1],q[2]),q[3]);
        }
        std::cout.write(reinterpret_cast<const char*>(clearances.data()),clearances.size()*sizeof(float));
    }
    return std::cin.eof()&&std::cin.gcount()==0?0:1;
}
