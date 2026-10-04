/*
 * Independent sensor-geometry audit controller (owner: results/omp-sensor-transfer).
 *
 * Static calibration rig. It does NOT fly, does NOT load a policy, and does NOT
 * touch the production adapter. It poses the rig at a few frozen locations, waits
 * for real native RangeFinder refreshes, and records the raw 320-pixel image plus
 * the measured device/body pose, live sensor settings and timestamps.
 *
 * Build:  make -C webots/controllers/sensor_geometry_audit
 * Run:    see docs/SENSOR_GEOMETRY_AUDIT.md
 */

#include <webots/gyro.h>
#include <webots/gps.h>
#include <webots/inertial_unit.h>
#include <webots/range_finder.h>
#include <webots/robot.h>
#include <webots/supervisor.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define WIDTH 20
#define HEIGHT 16
#define PIXELS (WIDTH * HEIGHT)

typedef struct {
  const char *name;
  double x, y, z, yaw;
  int wait_steps; /* steps to advance before reading */
  int kind;       /* 0 = settled static pose, 1 = first-refresh latency probe */
} Case;

static const Case kCases[] = {
    {"front_near", 0.0, 0.0, 1.5, 0.0, 10, 0},
    {"front_mid", 0.8, 0.0, 1.5, 0.0, 10, 0},
    {"front_far", -1.0, 0.4, 1.5, 0.0, 10, 0},
    {"yaw_left", 0.0, 0.0, 1.5, 0.55, 10, 0},
    {"yaw_right", 0.0, 0.0, 1.5, -0.55, 10, 0},
    {"side_look", 0.0, 0.0, 1.5, 1.5707963267948966, 10, 0},
    {"floor_low", 0.0, -0.5, 1.0, 0.0, 10, 0},
    {"discontinuity", -0.6, 0.0, 1.5, 0.0, 10, 0},
    {"latency_A", 0.0, 0.0, 1.5, 0.0, 5, 1},
    {"latency_B", 0.6, 0.0, 1.5, 0.5, 5, 1},
    {"latency_A", 0.0, 0.0, 1.5, 0.0, 5, 1},
    {"latency_B", 0.6, 0.0, 1.5, 0.5, 5, 1},
    {"latency_A", 0.0, 0.0, 1.5, 0.0, 5, 1},
    {"latency_B", 0.6, 0.0, 1.5, 0.5, 5, 1},
};
static const int kCaseCount = (int)(sizeof(kCases) / sizeof(kCases[0]));

static void write_doubles(FILE *f, const char *tag, const double *v, int n) {
  fprintf(f, "\"%s\":[", tag);
  for (int i = 0; i < n; i++) {
    if (i) fputc(',', f);
    fprintf(f, "%.17g", v[i]);
  }
  fputc(']', f);
}

int main(void) {
  wb_robot_init();
  const int dt = (int)wb_robot_get_basic_time_step();
  const char *args = wb_robot_get_custom_data();
  const char *project = wb_robot_get_project_path();
  char out_path[4096];
  out_path[0] = '\0';
  {
    const char *key = strstr(args, "out=");
    if (key) {
      key += 4;
      const char *end = strpbrk(key, "; \t\r\n");
      size_t n = end ? (size_t)(end - key) : strlen(key);
      if (n >= sizeof(out_path)) n = sizeof(out_path) - 1;
      memcpy(out_path, key, n);
      out_path[n] = '\0';
    }
  }
  if (out_path[0] == '/') {
    /* absolute: unchanged */
  } else {
    char relative[4096];
    snprintf(relative, sizeof(relative), "%s", out_path);
    snprintf(out_path, sizeof(out_path), "%s/%s", project ? project : ".", relative);
  }

  WbNodeRef rig = wb_supervisor_node_get_from_def("RIG");
  WbDeviceTag depth = wb_robot_get_device("depth");
  WbDeviceTag gps = wb_robot_get_device("gps");
  WbDeviceTag imu = wb_robot_get_device("imu");
  WbDeviceTag gyro = wb_robot_get_device("gyro");
  if (!rig || !depth || !gps || !imu || !gyro) {
    fprintf(stderr, "sensor_geometry_audit: missing RIG or device\n");
    wb_robot_cleanup();
    return 2;
  }
  WbNodeRef depth_node = wb_supervisor_node_get_from_device(depth);
  WbFieldRef rig_translation = wb_supervisor_node_get_field(rig, "translation");
  WbFieldRef rig_rotation = wb_supervisor_node_get_field(rig, "rotation");
  if (!depth_node || !rig_translation || !rig_rotation) {
    fprintf(stderr, "sensor_geometry_audit: missing RIG pose fields\n");
    wb_robot_cleanup();
    return 2;
  }
  wb_gps_enable(gps, dt);
  wb_inertial_unit_enable(imu, dt);
  wb_gyro_enable(gyro, dt);
  wb_range_finder_enable(depth, 50);

  const int width = wb_range_finder_get_width(depth);
  const int height = wb_range_finder_get_height(depth);
  const double fov = wb_range_finder_get_fov(depth);
  const double min_range = wb_range_finder_get_min_range(depth);
  const double max_range = wb_range_finder_get_max_range(depth);
  const int sampling = wb_range_finder_get_sampling_period(depth);
  if (width != WIDTH || height != HEIGHT) {
    fprintf(stderr, "sensor_geometry_audit: expected %dx%d RangeFinder, got %dx%d\n", WIDTH, HEIGHT, width, height);
    wb_robot_cleanup();
    return 2;
  }

  FILE *f = fopen(out_path, "w");
  int nonfinite_total = 0;
  if (!f) {
    fprintf(stderr, "sensor_geometry_audit: cannot open %s\n", out_path);
    wb_robot_cleanup();
    return 2;
  }
  fprintf(f,
          "{\"record\":\"settings\",\"schema\":\"webots-sensor-geometry-audit-v1\",\"project_path\":\"%s\","
          "\"world\":\"sensor-geometry-audit.wbt\",\"width\":%d,\"height\":%d,\"field_of_view_rad\":%.17g,"
          "\"min_range_m\":%.17g,\"max_range_m\":%.17g,\"sampling_period_ms\":%d,"
          "\"mount_translation_m\":[0.08,0,0],\"mount_rotation_axis_angle\":[0,0,1,0],"
          "\"declared_near_m\":0.03,\"declared_projection\":\"planar\",\"declared_resolution_m\":0.001,"
          "\"declared_vertical_half_tangent_aspect\":%.17g}\n",
          project ? project : "", width, height, fov, min_range, max_range, sampling,
          tan(0.5 * fov) * (double)height / (double)width);
  fflush(f);

  /* Settle the simulation and the first sensor refresh before any capture. */
  {
    const double init[3] = {kCases[0].x, kCases[0].y, kCases[0].z};
    const double init_rotation[4] = {0, 0, 1, kCases[0].yaw};
    wb_supervisor_field_set_sf_vec3f(rig_translation, init);
    wb_supervisor_field_set_sf_rotation(rig_rotation, init_rotation);
    for (int i = 0; i < 30; i++) wb_robot_step(dt);
  }

  for (int c = 0; c < kCaseCount; c++) {
    const Case *k = &kCases[c];
    const double t[3] = {k->x, k->y, k->z};
    const double rotation[4] = {0, 0, 1, k->yaw};
    wb_supervisor_field_set_sf_vec3f(rig_translation, t);
    wb_supervisor_field_set_sf_rotation(rig_rotation, rotation);
    for (int i = 0; i < k->wait_steps; i++) wb_robot_step(dt);
    const int steps = (int)((wb_robot_get_time() + 1e-9) / (dt * 1e-3));

    const float *image = wb_range_finder_get_range_image(depth);
    if (!image) {
      fprintf(stderr, "sensor_geometry_audit: null image at case %s\n", k->name);
      fclose(f);
      wb_robot_cleanup();
      return 3;
    }
    double raw[PIXELS];
    int nonfinite = 0;
    for (int i = 0; i < PIXELS; i++) {
      raw[i] = (double)image[i];
      if (!isfinite(raw[i])) nonfinite++;
    }
    if (nonfinite) nonfinite_total += nonfinite;

    const double *rig_pos = wb_supervisor_node_get_position(rig);
    const double *rig_rot = wb_supervisor_node_get_orientation(rig);
    double rig_position[3] = {rig_pos[0], rig_pos[1], rig_pos[2]};
    double rig_orientation[9];
    for (int i = 0; i < 9; i++) rig_orientation[i] = rig_rot[i];
    const double *cam_pos = wb_supervisor_node_get_position(depth_node);
    const double *cam_rot = wb_supervisor_node_get_orientation(depth_node);
    double cam_position[3] = {cam_pos[0], cam_pos[1], cam_pos[2]};
    double cam_orientation[9];
    for (int i = 0; i < 9; i++) cam_orientation[i] = cam_rot[i];
    const double *gps_values = wb_gps_get_values(gps);
    double gps_copy[3] = {gps_values[0], gps_values[1], gps_values[2]};
    const double *rpy = wb_inertial_unit_get_roll_pitch_yaw(imu);
    double rpy_copy[3] = {rpy[0], rpy[1], rpy[2]};
    const double *gyro_values = wb_gyro_get_values(gyro);
    double gyro_copy[3] = {gyro_values[0], gyro_values[1], gyro_values[2]};

    fprintf(f, "{\"record\":\"capture\",\"case\":\"%s\",\"kind\":%d,\"capture\":%d,\"step\":%d,\"time_s\":%.17g,",
            k->name, k->kind, c, steps, wb_robot_get_time());
    fprintf(f, "\"requested_position\":[%.17g,%.17g,%.17g],\"requested_yaw\":%.17g,", k->x, k->y, k->z, k->yaw);
    write_doubles(f, "rig_position", rig_position, 3);
    fputc(',', f);
    write_doubles(f, "rig_orientation", rig_orientation, 9);
    fputc(',', f);
    write_doubles(f, "depth_position", cam_position, 3);
    fputc(',', f);
    write_doubles(f, "depth_orientation", cam_orientation, 9);
    fputc(',', f);
    write_doubles(f, "gps", gps_copy, 3);
    fputc(',', f);
    write_doubles(f, "imu_roll_pitch_yaw", rpy_copy, 3);
    fputc(',', f);
    write_doubles(f, "gyro", gyro_copy, 3);
    fputc(',', f);
    fprintf(f, "\"raw\":[");
    for (int i = 0; i < PIXELS; i++) {
      if (i) fputc(',', f);
      if (isfinite(raw[i]))
        fprintf(f, "%.9g", raw[i]);
      else
        fprintf(f, "null");
    }
    fprintf(f, "]}\n");
    fflush(f);
    printf("AUDIT_CAPTURE case=%s step=%d cam=%.6f,%.6f,%.6f yaw_req=%.4f depth_pass=%s\n", k->name, steps,
           cam_position[0], cam_position[1], cam_position[2], k->yaw, isfinite(raw[8 * WIDTH + 10]) ? "finite" : "inf");
    fflush(stdout);
  }

  fclose(f);
  printf("AUDIT_DONE captures=%d nonfinite_pixels=%d out=%s\n", kCaseCount, nonfinite_total, out_path);
  if (nonfinite_total)
    printf("AUDIT_NONFINITE_REJECTED expected 0 non-finite rays (ambiguous maxRange grazing)\n");
  fflush(stdout);
  /* Deterministic termination: do not leave the simulation (and the shared GPU lock) alive. */
  wb_supervisor_simulation_quit(nonfinite_total ? EXIT_FAILURE : EXIT_SUCCESS);
  wb_robot_cleanup();
  return nonfinite_total ? 4 : 0;
}
