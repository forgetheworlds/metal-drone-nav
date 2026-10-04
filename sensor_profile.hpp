#pragma once
// Camera-only geometry shared by the Metal trainer, the CPU reference, the
// deployment frontend, and the native Webots controller.
//
// This header is deliberately separate from world.hpp. The frozen challenge
// bank records a SHA-256 over world.hpp and refuses to run when it changes
// (challenge_bank.py, challenge_evaluation.hpp::load_split), so world.hpp and
// its legacy camera model (world.hpp::wcamera) stay byte-identical. Everything
// that is only camera geometry lives here instead, so the consumers above
// cannot drift into different sensor contracts.
//
// Frame convention (FLU): +X forward, +Y left, +Z up. A pixel index is
// row*20+col with row 0 the top (up) row and column 0 the left (+Y) column,
// matching the Webots `range_finder_get_range_image` layout.
//
// Profiles
// --------
// legacy (NAV_SENSOR_PROFILE=1, default): the source model, identical to
//        world.hpp::wcamera. Vertical half-tangent 0.75, rays cast from the
//        body reference point, depth is ray range in metres.
// native (NAV_SENSOR_PROFILE=2): the measured Webots R2025a 20x16 planar
//        RangeFinder. Horizontal half-tangent 1.0, vertical half-tangent
//        0.8 = tan(45 deg)*16/20, pixel value is AXIAL depth along the +X
//        optical axis, and the sensor sits 0.08 m forward of the body
//        reference point. Ray range = axial * |d|, |d| = sqrt(1 + u^2 + v^2).
//
// The selection is a compile-time constant carried into the Metal source text,
// so calibrated simulation uses a separate binary. Full native checkpoints
// carry the camera contract in version 10; parameter-only warmstarts and camera
// sensitivity evaluations remain explicit choices. SimConfig stays unchanged.
//
// Requires world.hpp to be included first (for WVec/wv/wn/uint) unless this
// header is being concatenated into the Metal translation unit.

#ifndef __METAL_VERSION__
#include "world.hpp"
#endif

#define NAV_SENSOR_WIDTH 20u
#define NAV_SENSOR_HEIGHT 16u
#define NAV_SENSOR_PIXELS 320u
#define NAV_SENSOR_POOL_ROWS 8u
#define NAV_SENSOR_POOL_COLS 10u
#define NAV_SENSOR_TAN_H 1.0f
#define NAV_SENSOR_LEGACY_TAN_V 0.75f
#define NAV_SENSOR_LEGACY_MOUNT_X 0.0f
#define NAV_SENSOR_NATIVE_TAN_V 0.8f
#define NAV_SENSOR_NATIVE_MOUNT_X 0.08f
#define NAV_SENSOR_MIN_RANGE_M 0.03f
#define NAV_SENSOR_MAX_RANGE_M 12.0f
#define NAV_SENSOR_LEGACY_ID 1u
#define NAV_SENSOR_NATIVE_ID 2u

#ifndef NAV_SENSOR_PROFILE
#define NAV_SENSOR_PROFILE NAV_SENSOR_LEGACY_ID
#endif
#if NAV_SENSOR_PROFILE == NAV_SENSOR_NATIVE_ID
#define NAV_SENSOR_ACTIVE_ID NAV_SENSOR_NATIVE_ID
#define NAV_SENSOR_ACTIVE_TAN_V NAV_SENSOR_NATIVE_TAN_V
#define NAV_SENSOR_ACTIVE_MOUNT_X NAV_SENSOR_NATIVE_MOUNT_X
#define NAV_SENSOR_ACTIVE_NAME "native"
#elif NAV_SENSOR_PROFILE == NAV_SENSOR_LEGACY_ID
#define NAV_SENSOR_ACTIVE_ID NAV_SENSOR_LEGACY_ID
#define NAV_SENSOR_ACTIVE_TAN_V NAV_SENSOR_LEGACY_TAN_V
#define NAV_SENSOR_ACTIVE_MOUNT_X NAV_SENSOR_LEGACY_MOUNT_X
#define NAV_SENSOR_ACTIVE_NAME "legacy"
#else
#error "NAV_SENSOR_PROFILE must be 1 (legacy) or 2 (native)"
#endif

// world.hpp undefines its WP/WF/WT helpers at end of file, so this header
// carries its own inline qualifier.
#ifdef __METAL_VERSION__
#define NAV_SENSOR_INLINE inline
#else
#define NAV_SENSOR_INLINE inline
#endif

// Unnormalized FLU direction of one image pixel. At (tan_h=1, tan_v=0.75) this
// is bit-identical to world.hpp::wcamera, which keeps legacy runs unchanged.
NAV_SENSOR_INLINE WVec nav_sensor_pixel_direction(float tan_h,float tan_v,uint row,uint col) {
    const float y=((float(col)+0.5f)/20.0f*2.0f-1.0f)*tan_h;
    const float z=((float(row)+0.5f)/16.0f*2.0f-1.0f)*tan_v;
    return wv(1.0f,-y,-z);
}
NAV_SENSOR_INLINE WVec nav_sensor_pixel_direction_index(float tan_h,float tan_v,uint pixel) {
    return nav_sensor_pixel_direction(tan_h,tan_v,pixel/20u,pixel%20u);
}
NAV_SENSOR_INLINE WVec nav_sensor_pixel_ray(float tan_h,float tan_v,uint pixel) {
    return wn(nav_sensor_pixel_direction_index(tan_h,tan_v,pixel));
}
// 8x10 pooled direction at the centroid of each 2x2 pixel block. Matches the
// legacy guidance grid exactly at (tan_h=1, tan_v=0.75).
NAV_SENSOR_INLINE WVec nav_sensor_pooled_ray(float tan_h,float tan_v,uint row,uint col) {
    const float y=(0.9f-0.2f*float(col))*tan_h;
    const float z=(0.875f-0.25f*float(row))*tan_v;
    const float inv=1.0f/sqrt(1.0f+y*y+z*z);
    return wv(inv,y*inv,z*inv);
}
// |unnormalized pixel direction| = sqrt(1 + u^2 + v^2).
NAV_SENSOR_INLINE float nav_sensor_pixel_norm(float tan_h,float tan_v,uint row,uint col) {
    const WVec d=nav_sensor_pixel_direction(tan_h,tan_v,row,col);
    return sqrt(1.0f+d.y*d.y+d.z*d.z);
}
// The native RangeFinder reports axial depth along the optical axis; the source
// model carries ray range. Convert with the pixel's ray norm.
NAV_SENSOR_INLINE float nav_sensor_axial_to_ray(float tan_h,float tan_v,uint row,uint col,float axial) {
    return axial*nav_sensor_pixel_norm(tan_h,tan_v,row,col);
}

#ifndef __METAL_VERSION__
#include <cstdint>
#include <string>
namespace nav_sensor {

struct Profile {
    uint32_t id;
    float tan_h,tan_v,mount_x_m,min_range_m,max_range_m;
};

inline Profile legacy() {
    return {NAV_SENSOR_LEGACY_ID,NAV_SENSOR_TAN_H,NAV_SENSOR_LEGACY_TAN_V,
            NAV_SENSOR_LEGACY_MOUNT_X,NAV_SENSOR_MIN_RANGE_M,NAV_SENSOR_MAX_RANGE_M};
}
inline Profile native() {
    return {NAV_SENSOR_NATIVE_ID,NAV_SENSOR_TAN_H,NAV_SENSOR_NATIVE_TAN_V,
            NAV_SENSOR_NATIVE_MOUNT_X,NAV_SENSOR_MIN_RANGE_M,NAV_SENSOR_MAX_RANGE_M};
}
inline Profile by_id(uint32_t id) { return id==NAV_SENSOR_NATIVE_ID?native():legacy(); }
inline Profile active() { return by_id(NAV_SENSOR_ACTIVE_ID); }
inline const char* name(uint32_t id) { return id==NAV_SENSOR_NATIVE_ID?"native":"legacy"; }
inline bool parse(const char* text,uint32_t& id) {
    if(text&&std::string(text)=="legacy"){id=NAV_SENSOR_LEGACY_ID;return true;}
    if(text&&std::string(text)=="native"){id=NAV_SENSOR_NATIVE_ID;return true;}
    return false;
}

} // namespace nav_sensor
#endif
