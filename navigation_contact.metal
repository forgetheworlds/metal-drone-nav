// Independent pose-aware geometry audit. This kernel does not change policy
// observations, reward, episode termination, or training collision handling.
kernel void navigation_contact_audit(
    device const RLPhysicsState* states [[buffer(0)]],
    device const WWorld* worlds [[buffer(1)]],
    device const float* elapsed [[buffer(2)]],
    constant float& safety_margin_m [[buffer(3)]],
    device VGResult* results [[buffer(4)]],
    device float* legacy_sphere_clearance [[buffer(5)]],
    constant uint& emit_legacy_sphere_clearance [[buffer(6)]],
    uint n [[thread_position_in_grid]]) {
    const RLPhysicsState state=states[n];
    thread float position[3];
    thread float orientation_wxyz[4];
    for(uint axis=0;axis<3;axis++)position[axis]=state.position[axis];
    for(uint axis=0;axis<4;axis++)orientation_wxyz[axis]=state.orientation_wxyz[axis];

    VGResult result;
    vg_world_contact(position,orientation_wxyz,worlds[n],elapsed[n],safety_margin_m,result);
    results[n]=result;
    if(emit_legacy_sphere_clearance!=0) {
        legacy_sphere_clearance[n]=wclearance(
            worlds[n],wv(position[0],position[1],position[2]),elapsed[n]);
    }
}
