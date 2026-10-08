#pragma once
#ifdef __METAL_VERSION__
#define CG_DEVICE device
#define CG_THREAD thread
#else
#define CG_DEVICE
#define CG_THREAD
#endif

// Training-only features. The actor does not receive this vector.
// Keep the same builder at V(s) and V(s_next); time and stored motion phase
// describe bounded movers without mistaking peak velocity for current velocity.
inline void critic_geometry_features(CG_DEVICE const WWorld& world,
                                     CG_THREAD const float* position,
                                     float elapsed,
                                     CG_THREAD float* output) {
    for (uint j=0;j<162;j++) output[j]=0.0f;
    output[0]=float(world.count)/16.0f;
    output[1]=elapsed/20.0f;
    for (uint i=0;i<world.count && i<16;i++) {
        CG_DEVICE const WObstacle& obstacle=world.obstacles[i];
        const uint offset=2+i*10;
        output[offset]=float(obstacle.kind)/3.0f;
        for (uint axis=0;axis<3;axis++) {
            output[offset+1+axis]=(obstacle.center[axis]-position[axis])/10.0f;
            output[offset+4+axis]=obstacle.size[axis]/5.0f;
            output[offset+7+axis]=obstacle.velocity[axis]/4.0f;
        }
        // Moving-sphere size[2] is phase in radians, not a spatial extent.
        if (obstacle.kind==3) output[offset+6]=obstacle.size[2]/6.28318530718f;
    }
}
#undef CG_DEVICE
#undef CG_THREAD
