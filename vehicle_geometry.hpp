#pragma once

// Declared Crazyflie-derived mechanical contact model shared by CPU and Metal.
// Dimensions match NavigationQuadrotor.proto. This is not a verified hardware
// CAD model: it omits the camera/decks/landing features and uses the L2F
// profile's simplified +/-28 mm rotor positions.

#ifndef __METAL_VERSION__
#include "world.hpp"
#endif

#ifdef __METAL_VERSION__
#define VG_INLINE inline
#define VG_THREAD thread
#define VG_DEVICE device
#define VG_ISFINITE(value) isfinite(value)
#else
#include <cmath>
#include <cstdint>
#define VG_INLINE inline
#define VG_THREAD
#define VG_DEVICE
#define VG_ISFINITE(value) std::isfinite(value)
using std::fabs;
using std::fmax;
using std::fmin;
using std::sqrt;
using uint = uint32_t;
#endif

// Keep these dimensions in sync with NavigationQuadrotor.proto.
// Chassis and arms are taken from the existing visible model. Bitcraze lists
// 45 mm props; rotor centers and dynamics remain the pinned L2F values in
// physics.hpp. Those +/-28 mm centers imply 79.2 mm diagonal motor spacing,
// unlike Bitcraze's 92 x 92 mm CF2.1 mechanical specification. This is a
// dynamics-coherent declared profile, not a measured CF2.1 CAD fit. Rotor disc
// thickness is an explicit 1 mm assumption.
// Bitcraze references: https://www.bitcraze.io/crazyflie-2-1/ and
// https://www.bitcraze.io/documentation/hardware/propellers/propellers-datasheet.pdf
#define VG_DEFAULT_SAFETY_MARGIN_M 0.04f

struct VGVec3 { float x,y,z; };
struct VGResult {
    float physical_clearance_m; // Nonnegative gap; zero for contact/overlap.
    float safety_clearance_m;   // physical_clearance_m - caller margin.
    uint contact;               // Declared mechanical contact, independent of margin.
    uint ambiguous;             // Numerical/unsupported case; handled conservatively.
};

struct VGConvexShape {
    uint kind; // 0: oriented box, 1: finite cylinder.
    VGVec3 center;
    VGVec3 half_extent;
    float radius;
    float half_height;
};

struct VGSimplex {
    VGVec3 points[4];
    uint count;
};

VG_INLINE VGVec3 vg_v(float x,float y,float z) { return {x,y,z}; }
VG_INLINE VGVec3 vg_add(VGVec3 a,VGVec3 b) { return vg_v(a.x+b.x,a.y+b.y,a.z+b.z); }
VG_INLINE VGVec3 vg_sub(VGVec3 a,VGVec3 b) { return vg_v(a.x-b.x,a.y-b.y,a.z-b.z); }
VG_INLINE VGVec3 vg_mul(VGVec3 a,float s) { return vg_v(a.x*s,a.y*s,a.z*s); }
VG_INLINE float vg_dot(VGVec3 a,VGVec3 b) { return a.x*b.x+a.y*b.y+a.z*b.z; }
VG_INLINE float vg_len2(VGVec3 a) { return vg_dot(a,a); }
VG_INLINE float vg_len(VGVec3 a) { return sqrt(vg_len2(a)); }
VG_INLINE float vg_component(VGVec3 a,uint axis) { return axis==0?a.x:(axis==1?a.y:a.z); }
VG_INLINE VGVec3 vg_axis(uint axis,float sign) {
    return axis==0?vg_v(sign,0,0):(axis==1?vg_v(0,sign,0):vg_v(0,0,sign));
}
VG_INLINE float vg_sign(float x) { return x<0.0f?-1.0f:(x>0.0f?1.0f:0.0f); }

// Quaternion is scalar-first and rotates body-frame vectors into world frame.
VG_INLINE VGVec3 vg_rotate(const VG_THREAD float* q,VGVec3 v) {
    const VGVec3 qv=vg_v(q[1],q[2],q[3]);
    const VGVec3 t=vg_mul(vg_v(qv.y*v.z-qv.z*v.y,
                                qv.z*v.x-qv.x*v.z,
                                qv.x*v.y-qv.y*v.x),2.0f);
    const VGVec3 q_cross_t=vg_v(qv.y*t.z-qv.z*t.y,
                                qv.z*t.x-qv.x*t.z,
                                qv.x*t.y-qv.y*t.x);
    return vg_add(v,vg_add(vg_mul(t,q[0]),q_cross_t));
}
VG_INLINE VGVec3 vg_rotate_inverse(const VG_THREAD float* q,VGVec3 v) {
    const float inverse[4]={q[0],-q[1],-q[2],-q[3]};
    return vg_rotate(inverse,v);
}

// Support point of one declared vehicle component in a world direction.
VG_INLINE VGVec3 vg_vehicle_support(VG_THREAD const VGConvexShape& shape,
                                   const VG_THREAD float* position,
                                   const VG_THREAD float* orientation_wxyz,
                                   VGVec3 direction_world) {
    const VGVec3 direction_body=vg_rotate_inverse(orientation_wxyz,direction_world);
    VGVec3 point=shape.center;
    if(shape.kind==0) {
        point.x+=vg_sign(direction_body.x)*shape.half_extent.x;
        point.y+=vg_sign(direction_body.y)*shape.half_extent.y;
        point.z+=vg_sign(direction_body.z)*shape.half_extent.z;
    } else {
        const float radial=sqrt(direction_body.x*direction_body.x+
                                direction_body.y*direction_body.y);
        if(radial>1e-12f) {
            point.x+=shape.radius*direction_body.x/radial;
            point.y+=shape.radius*direction_body.y/radial;
        }
        point.z+=vg_sign(direction_body.z)*shape.half_height;
    }
    const VGVec3 offset=vg_rotate(orientation_wxyz,point);
    return vg_add(vg_v(position[0],position[1],position[2]),offset);
}

// Support point for each existing WObstacle kind: AABB, sphere, vertical
// finite cylinder. Unknown kinds are rejected by the public query.
VG_INLINE VGVec3 vg_obstacle_support(VG_THREAD const WObstacle& obstacle,float time,
                                    VGVec3 direction) {
    const VGVec3 center=vg_v(obstacle.center[0]+obstacle.velocity[0]*time,
                             obstacle.center[1]+obstacle.velocity[1]*time,
                             obstacle.center[2]+obstacle.velocity[2]*time);
    if(obstacle.kind==0) {
        return vg_v(center.x+vg_sign(direction.x)*obstacle.size[0],
                    center.y+vg_sign(direction.y)*obstacle.size[1],
                    center.z+vg_sign(direction.z)*obstacle.size[2]);
    }
    if(obstacle.kind==1) {
        const float length=vg_len(direction);
        if(length<=1e-12f)return center;
        return vg_add(center,vg_mul(direction,obstacle.size[0]/length));
    }
    const float radial=sqrt(direction.x*direction.x+direction.y*direction.y);
    VGVec3 point=center;
    if(radial>1e-12f) {
        point.x+=obstacle.size[0]*direction.x/radial;
        point.y+=obstacle.size[0]*direction.y/radial;
    }
    point.z+=vg_sign(direction.z)*obstacle.size[2];
    return point;
}

VG_INLINE VGVec3 vg_minkowski_support(VG_THREAD const VGConvexShape& vehicle,
                                     const VG_THREAD float* position,
                                     const VG_THREAD float* orientation_wxyz,
                                     VG_THREAD const WObstacle& obstacle,float time,
                                     VGVec3 direction) {
    const VGVec3 a=vg_vehicle_support(vehicle,position,orientation_wxyz,direction);
    const VGVec3 b=vg_obstacle_support(obstacle,time,vg_mul(direction,-1.0f));
    return vg_sub(a,b);
}

VG_INLINE float vg_shape_bounding_radius(VG_THREAD const VGConvexShape& shape) {
    return shape.kind==0?vg_len(shape.half_extent):sqrt(shape.radius*shape.radius+shape.half_height*shape.half_height);
}
VG_INLINE float vg_obstacle_bounding_radius(VG_THREAD const WObstacle& obstacle) {
    if(obstacle.kind==0)return sqrt(obstacle.size[0]*obstacle.size[0]+
                                    obstacle.size[1]*obstacle.size[1]+
                                    obstacle.size[2]*obstacle.size[2]);
    if(obstacle.kind==1)return obstacle.size[0];
    return sqrt(obstacle.size[0]*obstacle.size[0]+obstacle.size[2]*obstacle.size[2]);
}

// Solve a small KKT system for the closest point in the affine hull of the
// selected simplex vertices. A singular subset is skipped; lower-dimensional
// subsets are considered separately by vg_reduce_simplex.
VG_INLINE bool vg_affine_weights(const VG_THREAD VGVec3* points,
                                 const VG_THREAD uint* selected,uint count,
                                 VG_THREAD float* weights) {
    float matrix[5][6];
    const uint size=count+1;
    for(uint row=0;row<size;row++)for(uint col=0;col<=size;col++)matrix[row][col]=0.0f;
    for(uint row=0;row<count;row++) {
        for(uint col=0;col<count;col++)
            matrix[row][col]=vg_dot(points[selected[row]],points[selected[col]]);
        matrix[row][count]=1.0f;
        matrix[row][size]=0.0f;
        matrix[count][row]=1.0f;
    }
    matrix[count][count]=0.0f;
    matrix[count][size]=1.0f;
    for(uint col=0;col<size;col++) {
        uint pivot=col;
        for(uint row=col+1;row<size;row++)
            if(fabs(matrix[row][col])>fabs(matrix[pivot][col]))pivot=row;
        if(fabs(matrix[pivot][col])<1e-12f)return false;
        if(pivot!=col)for(uint j=col;j<=size;j++) {
            const float swap=matrix[col][j];matrix[col][j]=matrix[pivot][j];matrix[pivot][j]=swap;
        }
        const float scale=matrix[col][col];
        for(uint j=col;j<=size;j++)matrix[col][j]/=scale;
        for(uint row=0;row<size;row++)if(row!=col) {
            const float factor=matrix[row][col];
            for(uint j=col;j<=size;j++)matrix[row][j]-=factor*matrix[col][j];
        }
    }
    float total=0.0f;
    for(uint i=0;i<count;i++) {
        weights[i]=matrix[i][size];
        if(weights[i]<-1e-5f)return false;
        if(weights[i]<0.0f)weights[i]=0.0f;
        total+=weights[i];
    }
    if(total<1e-10f)return false;
    for(uint i=0;i<count;i++)weights[i]/=total;
    return true;
}

// Exhaust all faces/edges/vertices of a <=4 point simplex, then retain the
// smallest feasible convex subset. This avoids vertex-only contact tests.
VG_INLINE bool vg_reduce_simplex(VG_THREAD VGSimplex& simplex,VG_THREAD VGVec3& closest) {
    float best_distance2=1e30f;
    uint best_selected[4]={0,0,0,0},best_count=0;
    float best_weights[4]={0,0,0,0};
    const uint masks=1u<<simplex.count;
    for(uint mask=1;mask<masks;mask++) {
        uint selected[4]={0,0,0,0},count=0;
        for(uint i=0;i<simplex.count;i++)if(mask&(1u<<i))selected[count++]=i;
        float weights[4]={0,0,0,0};
        if(!vg_affine_weights(simplex.points,selected,count,weights))continue;
        VGVec3 candidate=vg_v(0,0,0);
        for(uint i=0;i<count;i++)candidate=vg_add(candidate,vg_mul(simplex.points[selected[i]],weights[i]));
        const float distance2=vg_len2(candidate);
        if(distance2<best_distance2) {
            best_distance2=distance2;closest=candidate;best_count=count;
            for(uint i=0;i<count;i++){best_selected[i]=selected[i];best_weights[i]=weights[i];}
        }
    }
    if(best_count==0)return false;
    VGVec3 reduced[4];uint count=0;
    for(uint i=0;i<best_count;i++)if(best_weights[i]>1e-6f)reduced[count++]=simplex.points[best_selected[i]];
    if(count==0)return false;
    for(uint i=0;i<count;i++)simplex.points[i]=reduced[i];
    simplex.count=count;
    return true;
}

// GJK distance for two convex shapes. It returns nonnegative separation; when
// shapes overlap it reports contact with distance zero (no penetration depth).
// Non-convergence is conservatively treated as contact and marked ambiguous.
VG_INLINE void vg_convex_distance(VG_THREAD const VGConvexShape& vehicle,
                                  const VG_THREAD float* position,
                                  const VG_THREAD float* orientation_wxyz,
                                  VG_THREAD const WObstacle& obstacle,float time,
                                  VG_THREAD float& distance,
                                  VG_THREAD uint& contact,
                                  VG_THREAD uint& ambiguous) {
    const VGVec3 vehicle_center=vg_add(vg_v(position[0],position[1],position[2]),
                                      vg_rotate(orientation_wxyz,vehicle.center));
    const VGVec3 obstacle_center=vg_v(obstacle.center[0]+obstacle.velocity[0]*time,
                                      obstacle.center[1]+obstacle.velocity[1]*time,
                                      obstacle.center[2]+obstacle.velocity[2]*time);
    VGVec3 direction=vg_sub(obstacle_center,vehicle_center);
    if(vg_len2(direction)<1e-12f)direction=vg_v(1,0,0);
    VGSimplex simplex{};simplex.count=1;
    simplex.points[0]=vg_minkowski_support(vehicle,position,orientation_wxyz,obstacle,time,direction);
    VGVec3 closest=simplex.points[0];
    for(uint iteration=0;iteration<32;iteration++) {
        if(!vg_reduce_simplex(simplex,closest)) {
            distance=0.0f;contact=1;ambiguous=1;return;
        }
        const float distance2=vg_len2(closest);
        if(distance2<=1e-10f) { distance=0.0f;contact=1;return; }
        direction=vg_mul(closest,-1.0f);
        const VGVec3 support=vg_minkowski_support(vehicle,position,orientation_wxyz,obstacle,time,direction);
        const float improvement=distance2-vg_dot(closest,support);
        if(improvement<=1e-7f*fmax(1.0f,distance2)) {
            distance=sqrt(distance2);contact=distance<=1e-5f;return;
        }
        bool duplicate=false;
        for(uint i=0;i<simplex.count;i++)if(vg_len2(vg_sub(simplex.points[i],support))<=1e-12f)duplicate=true;
        if(duplicate) { distance=sqrt(distance2);contact=distance<=1e-5f;return; }
        if(simplex.count>=4) {
            distance=0.0f;contact=1;ambiguous=1;return;
        }
        simplex.points[simplex.count++]=support;
    }
    distance=0.0f;contact=1;ambiguous=1;
}

VG_INLINE void vg_vehicle_shape(uint index,VG_THREAD VGConvexShape& shape) {
    shape.center=vg_v(0,0,0);shape.half_extent=vg_v(0,0,0);shape.radius=0.0f;shape.half_height=0.0f;
    if(index==0) { // 60 x 60 x 14 mm chassis, declared from visible model.
        shape.kind=0;shape.half_extent=vg_v(0.030f,0.030f,0.007f);
    } else if(index==1) { // 84 x 8 x 8 mm arm bar.
        shape.kind=0;shape.half_extent=vg_v(0.042f,0.004f,0.004f);
    } else if(index==2) {
        shape.kind=0;shape.half_extent=vg_v(0.004f,0.042f,0.004f);
    } else {
        shape.kind=1;
        const float sx=(index==3||index==6)?1.0f:-1.0f;
        const float sy=(index==5||index==6)?1.0f:-1.0f;
        shape.center=vg_v(sx*0.028f,sy*0.028f,0.0f);
        shape.radius=0.0225f; // Bitcraze nominal 45 mm propeller diameter.
        shape.half_height=0.0005f; // Declared 1 mm swept-disc thickness.
    }
}

// Existing simulator room bounds from world.hpp: x[-2,14], y[-5,5], z[0,5].
VG_INLINE void vg_world_contact(const VG_THREAD float* position,
                                const VG_THREAD float* orientation_wxyz,
                                VG_DEVICE const WWorld& world,float time,
                                float safety_margin_m,
                                VG_THREAD VGResult& result) {
    result.physical_clearance_m=1e30f;result.safety_clearance_m=1e30f;
    result.contact=0;result.ambiguous=0;
    if(world.count>16 || !VG_ISFINITE(time) || !VG_ISFINITE(safety_margin_m)) {
        result.physical_clearance_m=0.0f;result.safety_clearance_m=-fmax(0.0f,safety_margin_m);
        result.contact=1;result.ambiguous=1;return;
    }
    const float qnorm=sqrt(orientation_wxyz[0]*orientation_wxyz[0]+
                           orientation_wxyz[1]*orientation_wxyz[1]+
                           orientation_wxyz[2]*orientation_wxyz[2]+
                           orientation_wxyz[3]*orientation_wxyz[3]);
    if(!VG_ISFINITE(qnorm)||qnorm<1e-6f||!VG_ISFINITE(position[0])||!VG_ISFINITE(position[1])||!VG_ISFINITE(position[2])) {
        result.physical_clearance_m=0.0f;result.safety_clearance_m=-fmax(0.0f,safety_margin_m);
        result.contact=1;result.ambiguous=1;return;
    }
    for(uint obstacle_index=0;obstacle_index<world.count;obstacle_index++) {
        const WObstacle obstacle=world.obstacles[obstacle_index];
        if(obstacle.kind>2) {
            result.physical_clearance_m=0.0f;result.safety_clearance_m=-fmax(0.0f,safety_margin_m);
            result.contact=1;result.ambiguous=1;return;
        }
        for(uint axis=0;axis<3;axis++) {
            const bool valid=VG_ISFINITE(obstacle.center[axis])&&VG_ISFINITE(obstacle.size[axis])&&
                             VG_ISFINITE(obstacle.velocity[axis])&&obstacle.size[axis]>=0.0f;
            if(!valid) {
                result.physical_clearance_m=0.0f;result.safety_clearance_m=-fmax(0.0f,safety_margin_m);
                result.contact=1;result.ambiguous=1;return;
            }
        }
    }
    const float q[4]={orientation_wxyz[0]/qnorm,orientation_wxyz[1]/qnorm,
                      orientation_wxyz[2]/qnorm,orientation_wxyz[3]/qnorm};
    const float lower[3]={-2.0f,-5.0f,0.0f},upper[3]={14.0f,5.0f,5.0f};
    for(uint part=0;part<7;part++) {
        VGConvexShape shape;vg_vehicle_shape(part,shape);
        for(uint axis=0;axis<3;axis++) {
            const float maximum=vg_component(vg_vehicle_support(shape,position,q,vg_axis(axis,1.0f)),axis);
            const float minimum=vg_component(vg_vehicle_support(shape,position,q,vg_axis(axis,-1.0f)),axis);
            const float lower_gap=minimum-lower[axis],upper_gap=upper[axis]-maximum;
            const float plane_gap=fmin(lower_gap,upper_gap);
            result.physical_clearance_m=fmin(result.physical_clearance_m,plane_gap);
            if(plane_gap<=1e-5f) {
                result.physical_clearance_m=0.0f;result.safety_clearance_m=-fmax(0.0f,safety_margin_m);
                result.contact=1;return;
            }
        }
        for(uint obstacle_index=0;obstacle_index<world.count;obstacle_index++) {
            const WObstacle obstacle=world.obstacles[obstacle_index];
            const VGVec3 vehicle_center=vg_add(vg_v(position[0],position[1],position[2]),
                                               vg_rotate(q,shape.center));
            const VGVec3 obstacle_center=vg_v(obstacle.center[0]+obstacle.velocity[0]*time,
                                              obstacle.center[1]+obstacle.velocity[1]*time,
                                              obstacle.center[2]+obstacle.velocity[2]*time);
            const float lower_bound=vg_len(vg_sub(vehicle_center,obstacle_center))-
                                    vg_shape_bounding_radius(shape)-vg_obstacle_bounding_radius(obstacle);
            // Bounding spheres give a valid lower bound. Skip pairs that cannot
            // improve the current nearest gap or produce contact.
            if(lower_bound>=result.physical_clearance_m)continue;
            float gap=0.0f;uint contact=0,ambiguous=0;
            vg_convex_distance(shape,position,q,obstacle,time,gap,contact,ambiguous);
            if(contact) {
                result.physical_clearance_m=0.0f;result.safety_clearance_m=-fmax(0.0f,safety_margin_m);
                result.contact=1;result.ambiguous|=ambiguous;return;
            }
            result.physical_clearance_m=fmin(result.physical_clearance_m,gap);
            result.ambiguous|=ambiguous;
        }
    }
    if(result.physical_clearance_m<0.0f) {
        result.physical_clearance_m=0.0f;result.contact=1;
    }
    result.safety_clearance_m=result.physical_clearance_m-fmax(0.0f,safety_margin_m);
}
