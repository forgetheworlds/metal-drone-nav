// New matched controller-state critic study. Historical dynamic snapshot is preserved.
// Local dynamic-threat task module (mission: reusable local avoidance of
// moving threats). OWNED BY the research/experiment mission; isolated to this
// file plus the byte-identical `dynamic_snapshot/` copy of the published root
// runner (SNAPSHOT.json proves every snapshot file matches ROOT). None of the
// old shaping modifications are reachable from here: SOURCE_DIR points at the
// snapshot, whose sim.metal/training_potential.hpp are the published bytes,
// and potential shaping can never activate on this path (bank_control stays
// disabled).
//
// Commands
//   dynamic-test                       snapshot contract + core gates + parity gates
//   dynamic-bank --train|--dev OUT     generate -> label -> RAPTOR-witness -> accept
//   dynamic-train ROLLOUTS CKPT WARM   mixture training (dynamic+static, ablate 0/1)
//   dynamic-eval CKPT OUT ...          one bank pass (mode 17/13/4/2, ablate 0/1)
//
// Experiment arms differ in exactly one variable (decision.md §6): the ablated
// arm dispatches sim_observe with a byte-identical config whose only change is
// mode=18 (the source's own "remove previous-depth channel" branch), so the
// actor previous-depth input and the geometry-prior previous frame receive the
// current frame. sim_act/sim_advance always keep the real config.
#define main dynamic_nav_reserved_main
#include "main.mm"
#undef main

namespace dyn {

// ---------------------------------------------------------------- constants
constexpr uint32_t kMixturePeriod = 12;   // slots 0-3 dynamic, 4-11 static(8 fams)
constexpr uint32_t kDynamicSlots = 4;
constexpr uint32_t kStaticSlots = 8;
constexpr uint32_t kTrainDynamicTasks = 512; // 128 envs * 4 slots
constexpr uint32_t kDevDynamicTasks = 128;
constexpr float kSphereRadius = 0.35f;
constexpr float kCombinedRadius = 0.18f + kSphereRadius;
constexpr float kHoverMarginM = 0.15f;
constexpr float kWarningMinS = 0.25f;
constexpr float kGoalLosSlackM = 0.10f;
constexpr float kDetourOffsetMinM = 0.8f;
constexpr float kDetourOffsetMaxM = 2.2f;
constexpr float kTaskSpeedCapMps = 1.5f;

// Start/goal boxes follow the published local-waypoint contract.
static const float kStartLow[3] = {-1.4f, -3.0f, 0.8f};
static const float kStartHigh[3] = {1.0f, 3.0f, 2.6f};
static const float kGoalLow[3] = {-1.7f, -4.7f, 0.3f};
static const float kGoalHigh[3] = {13.7f, 4.7f, 4.7f};

enum RejectReason : uint32_t {
    REJ_NONE = 0,
    REJ_START_BLOCKED = 1,
    REJ_GOAL_BLOCKED = 2,
    REJ_GOAL_OUT_OF_BOUNDS = 3,
    REJ_GOAL_NOT_GROUNDED = 4,   // frustum projection or LOS failed
    REJ_STATIC_STRAIGHT_BLOCKED = 5,
    REJ_NO_ROUTE = 6,
    REJ_THREAT_OVERLAP = 7,      // source guards: room/path/start overlap
    REJ_DRAW_EXHAUSTED = 8,
    REJ_NOT_CAUSAL = 9,          // straight flight did not contact the threat
    REJ_STATIC_COLLISION = 10,   // straight flight hit static geometry
    REJ_INSUFFICIENT_WARNING = 11,
    REJ_SINGLE_ACTION = 12,      // fewer than 2 plant-feasible evasions
};

static const char* reject_reason_name(uint32_t reason) {
    switch (reason) {
        case REJ_NONE: return "accepted";
        case REJ_START_BLOCKED: return "start_blocked";
        case REJ_GOAL_BLOCKED: return "goal_blocked";
        case REJ_GOAL_OUT_OF_BOUNDS: return "goal_out_of_bounds";
        case REJ_GOAL_NOT_GROUNDED: return "goal_not_grounded";
        case REJ_STATIC_STRAIGHT_BLOCKED: return "static_straight_blocked";
        case REJ_NO_ROUTE: return "no_route";
        case REJ_THREAT_OVERLAP: return "threat_overlap";
        case REJ_DRAW_EXHAUSTED: return "draw_exhausted";
        case REJ_NOT_CAUSAL: return "not_causal";
        case REJ_STATIC_COLLISION: return "static_collision_on_straight";
        case REJ_INSUFFICIENT_WARNING: return "insufficient_warning";
        case REJ_SINGLE_ACTION: return "single_action";
        default: return "unknown";
    }
}

// Scripted witness controllers (device kernel mode ids).
enum ScriptMode : uint32_t {
    SCRIPT_STRAIGHT = 0,   // blind full-speed straight (causal counterfactual)
    SCRIPT_WAIT = 1,       // hold at start until T, then straight
    SCRIPT_RETREAT = 2,    // fly backward until T, then straight
    SCRIPT_DETROUT_PLUS = 3,  // follow route_plus polyline
    SCRIPT_DETROUT_MINUS = 4, // follow route_minus polyline
};

// ---------------------------------------------------------------- bank ABI
// One dynamic local task: static scene + one moving sphere + local goal.
// All label fields are generator/diagnostic truth; NONE enter the actor.
// Layout is all-4-byte members in a fixed order so the host natural layout
// matches the MSL declaration byte for byte (no packing pragma anywhere).
struct DynamicEntry {
    WWorld world;                    // static obstacles + threat sphere + goal
    float start_position[3];
    float start_velocity[3];
    float goal_position[3];
    float start_yaw;
    float clearance_start;           // body clearance at start (static+threat t0)
    float clearance_goal;
    float route_length;              // straight start->goal length (route is straight)
    float straight_min_clearance;    // min body clearance sampled along the straight route
    float threat_speed_mps;
    float nominal_ttc_s;             // source-style construction parameter
    float wait_until_s;              // host-computed wait departure time
    float retreat_until_s;           // host-computed retreat phase time
    float warning_s;                 // first in-frustum+LOS time (-1 if none)
    float causal_collision_s;        // measured straight-flight contact time
    float measured_min_ttc_s;        // min state-based TTC over straight flight
    float retreat_point[3];
    float route_plus[4][3];          // polyline incl. first point only (see route_counts)
    float route_minus[4][3];
    uint32_t route_plus_count;
    uint32_t route_minus_count;
    uint32_t family;                 // static base family 0/1/2/4/5
    uint32_t scene_seed;
    uint32_t threat_kind;            // 0 approach, 1 crossing
    uint32_t start_stratum;          // 0 stopped, 1 toward-goal, 2 lateral
    uint32_t draw_index;             // global draw index (stratification)
    uint32_t labels;                 // bit flags below
    uint32_t witness_mask;           // controller outcomes: bit=success per mode
    uint32_t rejected_reason;        // RejectReason, REJ_NONE = accepted
};
static_assert(sizeof(DynamicEntry) % 4 == 0, "dynamic entry must stay 4-byte aligned");

// label bits
constexpr uint32_t LBL_GROUNDED = 1u << 0;
constexpr uint32_t LBL_LOS_CLEAR = 1u << 1;
constexpr uint32_t LBL_HOVER_SAFE = 1u << 2;
constexpr uint32_t LBL_WARNING_OK = 1u << 3;
constexpr uint32_t LBL_CAUSAL = 1u << 4;       // straight contacted the threat

struct DynBankHeader {           // "DYBANK1\0"
    char magic[8];
    uint32_t version;
    uint32_t period;             // slots per env in training mixture (1 for dev)
    uint32_t count;
    uint32_t entry_bytes;
    char entry_sha256[64];
};
static_assert(sizeof(DynBankHeader) == 88, "dynamic bank header ABI");

// Read-only mirror of the published local-waypoint bank entry (WPBANK1).
struct StaticEntry {
    WWorld world;
    float start_position[3];
    float start_velocity[3];
    float goal_position[3];
    float start_yaw;
    float clearance_start, clearance_goal, direct_clearance, witness_clearance;
    float witness_length, initial_distance;
    uint32_t family, scene_seed, route_class, attempts;
};
static_assert(sizeof(StaticEntry) == 756, "static bank entry ABI must match WPBANK1");

// Per-environment scripted controller parameters (witness flights only).
struct DynScriptControl {
    uint32_t mode;       // ScriptMode
    uint32_t entry_index;
    uint32_t enabled;    // 0 = fall back to nothing (unused paths must set 1)
    float wait_until_s;
};
static_assert(sizeof(DynScriptControl) == 16, "script control ABI");

// ------------------------------------------------------------------ helpers

// Sensor-profile frustum grounding, exactly the mission predicate:
// body_x > 0 && |y/x| <= tan_h && |z/x| <= tan_v, yaw-only body frame.
static bool goal_in_frustum(const float start[3], float yaw,
                            const float goal[3]) {
    const float dx = goal[0] - start[0], dy = goal[1] - start[1],
                dz = goal[2] - start[2];
    const float c = std::cos(yaw), s = std::sin(yaw);
    const float bx = c * dx + s * dy;
    if (!(bx > 0.0f)) return false;
    const float by = -s * dx + c * dy;
    if (std::fabs(by / bx) > NAV_SENSOR_TAN_H) return false;
    if (std::fabs(dz / bx) > NAV_SENSOR_ACTIVE_TAN_V) return false;
    return true;
}

static bool inside_goal_box(const float p[3]) {
    return p[0] >= kGoalLow[0] && p[0] <= kGoalHigh[0] &&
           p[1] >= kGoalLow[1] && p[1] <= kGoalHigh[1] &&
           p[2] >= kGoalLow[2] && p[2] <= kGoalHigh[2];
}

static float distance3(const float a[3], const float b[3]) {
    const float dx = b[0] - a[0], dy = b[1] - a[1], dz = b[2] - a[2];
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

// Grounding check: frustum projection AND line of sight to the goal through
// the world at t=0 (static geometry and the parked threat both count).
static bool goal_los_clear(const WWorld& world, const float start[3],
                           const float goal[3]) {
    float dx = goal[0] - start[0], dy = goal[1] - start[1], dz = goal[2] - start[2];
    const float span = std::sqrt(dx * dx + dy * dy + dz * dz);
    if (!(span > 1e-6f)) return false;
    dx /= span; dy /= span; dz /= span;
    const float hit = wray(world, wv(start[0], start[1], start[2]),
                           wv(dx, dy, dz), 0.0f);
    return hit + kGoalLosSlackM >= span;
}

// Threat visibility from the START pose: in-frustum at the sampled time and
// the ray reaches the sphere itself (static geometry may not block it).
// Returns first visible time or -1. Scan horizon: until `until_s`.
static float first_threat_visible(const WWorld& world, const float start[3],
                                  float yaw, uint32_t threat_index,
                                  float until_s) {
    const WObstacle& o = world.obstacles[threat_index];
    const uint32_t ticks = uint32_t(std::ceil(until_s / 0.05f));
    for (uint32_t i = 0; i <= ticks; i++) {
        const float t = float(i) * 0.05f;
        const float center[3] = {o.center[0] + t * o.velocity[0],
                                 o.center[1] + t * o.velocity[1],
                                 o.center[2] + t * o.velocity[2]};
        if (!goal_in_frustum(start, yaw, center)) continue;
        const float dist = distance3(start, center);
        if (dist > 12.0f) continue;
        float dx = (center[0] - start[0]) / dist, dy = (center[1] - start[1]) / dist,
              dz = (center[2] - start[2]) / dist;
        const float hit = wray(world, wv(start[0], start[1], start[2]),
                               wv(dx, dy, dz), t);
        if (hit + 0.02f >= dist - kSphereRadius) return t;  // first hit is the threat
    }
    return -1.0f;
}

// Host kinematics (declared, offline): where is the threat at time t?
static void threat_position(const WWorld& world, uint32_t threat_index, float t,
                            float out[3]) {
    const WObstacle& o = world.obstacles[threat_index];
    out[0] = o.center[0] + t * o.velocity[0];
    out[1] = o.center[1] + t * o.velocity[1];
    out[2] = o.center[2] + t * o.velocity[2];
}

// Estimate the earliest wait-departure time: the threat must stay at least
// combined+margin away from the start pose for the whole wait AND stay off
// the straight corridor from departure onward (constant-velocity model; the
// plant witness is the authority).
static bool estimate_wait_time(const WWorld& world, uint32_t threat_index,
                               const float start[3], const float goal[3],
                               float traverse_estimate_s, float& out_t) {
    const float safe = kCombinedRadius + kHoverMarginM;
    const uint32_t horizon = uint32_t(std::ceil((6.0f + traverse_estimate_s) / 0.05f));
    bool start_safe_so_far = true;
    for (uint32_t i = 0; i <= horizon; i++) {
        const float t = float(i) * 0.05f;
        float tp[3];
        threat_position(world, threat_index, t, tp);
        if (distance3(start, tp) < safe) start_safe_so_far = false;
        if (!start_safe_so_far) continue;
        // candidate departure at t: threat must clear the straight corridor
        // for the whole estimated traverse.
        bool clear = true;
        for (uint32_t j = 0; j <= uint32_t(traverse_estimate_s / 0.05f) + 2; j++) {
            const float td = t + float(j) * 0.05f;
            float tp2[3];
            threat_position(world, threat_index, td, tp2);
            // distance from threat to the start->goal segment
            const float ex = goal[0] - start[0], ey = goal[1] - start[1],
                        ez = goal[2] - start[2];
            const float len2 = ex * ex + ey * ey + ez * ez;
            float s = len2 > 1e-9f
                ? ((tp2[0] - start[0]) * ex + (tp2[1] - start[1]) * ey +
                   (tp2[2] - start[2]) * ez) / len2 : 0.0f;
            s = std::max(0.0f, std::min(1.0f, s));
            const float px = start[0] + s * ex, py = start[1] + s * ey,
                        pz = start[2] + s * ez;
            const float d2 = (tp2[0] - px) * (tp2[0] - px) +
                             (tp2[1] - py) * (tp2[1] - py) +
                             (tp2[2] - pz) * (tp2[2] - pz);
            if (d2 < (kCombinedRadius + 0.10f) * (kCombinedRadius + 0.10f)) {
                clear = false;
                break;
            }
        }
        if (clear) { out_t = t; return true; }
    }
    return false;
}
// ------------------------------------------------------------- MSL kernels
static const char* kDynamicMsl = R"MSL(
struct DynEntry {
    WWorld world;
    float start_position[3];
    float start_velocity[3];
    float goal_position[3];
    float start_yaw;
    float clearance_start;
    float clearance_goal;
    float route_length;
    float straight_min_clearance;
    float threat_speed_mps;
    float nominal_ttc_s;
    float wait_until_s;
    float retreat_until_s;
    float warning_s;
    float causal_collision_s;
    float measured_min_ttc_s;
    float retreat_point[3];
    float route_plus[4][3];
    float route_minus[4][3];
    uint route_plus_count;
    uint route_minus_count;
    uint family;
    uint scene_seed;
    uint threat_kind;
    uint start_stratum;
    uint draw_index;
    uint labels;
    uint witness_mask;
    uint rejected_reason;
};
struct StEntry {
    WWorld world;
    float start_position[3];
    float start_velocity[3];
    float goal_position[3];
    float start_yaw;
    float clearance_start, clearance_goal, direct_clearance, witness_clearance;
    float witness_length, initial_distance;
    uint family, scene_seed, route_class, attempts;
};
struct DynBankControl {
    uint dyn_count, static_count, dyn_slots, static_slots, period, pad;
};
struct DynScriptControl { uint mode; uint entry_index; uint enabled; float wait_until; };

// Install one mixture slot for this environment. Slot = episodes % period.
// Slots [0, dyn_slots) are dynamic tasks, the rest are static bank slots.
// Zeroes the episode clock, so the moving threat's trajectory always starts
// at t=0 for every episode (fresh install and auto-reset install alike).
kernel void dynamic_task_apply(device RLPhysicsState* states [[buffer(0)]],
                               device SimRun* runs [[buffer(1)]],
                               device WWorld* worlds [[buffer(2)]],
                               device NavigationTaskState* tasks [[buffer(3)]],
                               device const DynEntry* dyn_bank [[buffer(4)]],
                               device const StEntry* static_bank [[buffer(5)]],
                               constant DynBankControl& control [[buffer(6)]],
                               constant NavigationTaskControl& task_control [[buffer(7)]],
                               constant SimConfig& cfg [[buffer(8)]],
                               uint n [[thread_position_in_grid]]) {
    if(n>=cfg.n)return;
    if(runs[n].steps!=0u)return;
    const uint period=max(control.period,1u);
    const uint slot=runs[n].episodes%period;
    NavigationTaskState task=NavigationTaskState{};
    task.generation_status=NAV_TASK_GENERATION_READY;task.valid=1u;task.generation_attempts=1u;
    task.stage=task_control.config.stage;task.objective=task_control.config.objective;
    task.max_nav_steps=task_control.config.max_nav_steps;task.step_count=0u;task.stable_ticks=0u;
    task.was_inside_goal=0u;task.waypoint_event_latched=0u;task.terminal=0u;
    task.difficulty=task_control.config.difficulty;
    task.stable_time_s=0.0f;task.witness_point_count=0u;
    float start[3],velocity[3],goal[3],yaw;
    if(slot<control.dyn_slots) {
        const uint index=n*control.dyn_slots+slot;
        if(index>=control.dyn_count)return;
        const DynEntry e=dyn_bank[index];
        WWorld world=e.world;
        world.wind[0]=0.0f;world.wind[1]=0.0f;world.wind[2]=0.0f;
        worlds[n]=world;
        task.family=e.family;
        for(uint axis=0;axis<3;axis++) {
            start[axis]=e.start_position[axis];velocity[axis]=e.start_velocity[axis];
            goal[axis]=e.goal_position[axis];
            task.start_position[axis]=e.start_position[axis];
            task.start_velocity_world[axis]=e.start_velocity[axis];
            task.goal_position[axis]=e.goal_position[axis];
            task.last_executed_world_command[axis]=e.start_velocity[axis];
        }
        yaw=e.start_yaw;
        task.start_yaw_rad=e.start_yaw;
        task.initial_distance_m=e.route_length;task.previous_distance_m=e.route_length;
        task.start_clearance_m=e.clearance_start;task.goal_clearance_m=e.clearance_goal;
        task.direct_segment_clearance_m=e.straight_min_clearance;
        task.witness_min_clearance_m=e.straight_min_clearance;task.witness_length_m=e.route_length;
        runs[n].initial_distance=e.route_length;
    } else {
        const uint index=n*control.static_slots+(slot-control.dyn_slots);
        if(index>=control.static_count)return;
        const StEntry e=static_bank[index];
        WWorld world=e.world;
        world.wind[0]=0.0f;world.wind[1]=0.0f;world.wind[2]=0.0f;
        worlds[n]=world;
        task.family=e.family;
        for(uint axis=0;axis<3;axis++) {
            start[axis]=e.start_position[axis];velocity[axis]=e.start_velocity[axis];
            goal[axis]=e.goal_position[axis];
            task.start_position[axis]=e.start_position[axis];
            task.start_velocity_world[axis]=e.start_velocity[axis];
            task.goal_position[axis]=e.goal_position[axis];
            task.last_executed_world_command[axis]=e.start_velocity[axis];
        }
        yaw=e.start_yaw;
        task.start_yaw_rad=e.start_yaw;
        task.initial_distance_m=e.initial_distance;task.previous_distance_m=e.initial_distance;
        task.start_clearance_m=e.clearance_start;task.goal_clearance_m=e.clearance_goal;
        task.direct_segment_clearance_m=e.direct_clearance;
        task.witness_min_clearance_m=e.witness_clearance;task.witness_length_m=e.witness_length;
        runs[n].initial_distance=e.initial_distance;
    }
    task.last_executed_world_command[3]=0.0f;
    tasks[n]=task;
    RLPhysicsState state=states[n];
    for(uint axis=0;axis<3;axis++) {
        state.position[axis]=start[axis];
        state.linear_velocity[axis]=velocity[axis];
        state.angular_velocity_body[axis]=0.0f;
        runs[n].reference_position[axis]=start[axis];
        runs[n].desired_velocity[axis]=velocity[axis];
    }
    const float half_yaw=yaw*0.5f;
    state.orientation_wxyz[0]=cos(half_yaw);state.orientation_wxyz[1]=0.0f;
    state.orientation_wxyz[2]=0.0f;state.orientation_wxyz[3]=sin(half_yaw);
    states[n]=state;
    runs[n].yaw=yaw;
    runs[n].path=0.0f;runs[n].elapsed=0.0f;runs[n].peak_speed=0.0f;runs[n].min_clearance=12.0f;
    (void)goal;
}

// Scripted witness controller (no actor, no observation). Command convention
// matches sim_act mode 2: body-frame commands scaled by cfg.speed, delay 0.
kernel void dynamic_script_act(device RLPhysicsState* states [[buffer(0)]],
                               device SimRun* runs [[buffer(1)]],
                               device const WWorld* worlds [[buffer(2)]],
                               device const DynEntry* dyn_bank [[buffer(3)]],
                               device const DynScriptControl* script [[buffer(4)]],
                               device float* commands [[buffer(5)]],
                               constant SimConfig& cfg [[buffer(6)]],
                               uint n [[thread_position_in_grid]]) {
    if(n>=cfg.n || (cfg.eval && runs[n].episodes))return;
    const DynScriptControl sc=script[n];
    if(sc.enabled==0u)return;
    const DynEntry e=dyn_bank[sc.entry_index];
    const float elapsed=runs[n].elapsed;
    const float pos[3]={states[n].position[0],states[n].position[1],states[n].position[2]};
    float target[3]={e.goal_position[0],e.goal_position[1],e.goal_position[2]};
    float phase_speed=1.5f;
    bool hold=false;
    if(sc.mode==1u) hold = elapsed < e.wait_until_s;
    else if(sc.mode==2u) {
        if(elapsed < e.retreat_until_s) {
            for(uint axis=0;axis<3;axis++)target[axis]=e.retreat_point[axis];
            phase_speed=1.0f;
        }
    } else if(sc.mode==3u || sc.mode==4u) {
        // stateless polyline walk: active = first stored point not yet reached
        const uint count = sc.mode==3u ? e.route_plus_count : e.route_minus_count;
        uint active=0;
        for(uint i=0;i<count;i++) {
            float pt[3];
            if(sc.mode==3u) {
                pt[0]=e.route_plus[i][0];pt[1]=e.route_plus[i][1];pt[2]=e.route_plus[i][2];
            } else {
                pt[0]=e.route_minus[i][0];pt[1]=e.route_minus[i][1];pt[2]=e.route_minus[i][2];
            }
            const float dx=pt[0]-pos[0],dy=pt[1]-pos[1],dz=pt[2]-pos[2];
            const float span=sqrt(dx*dx+dy*dy+dz*dz);
            if(span<=0.35f && i+1<count) continue;   // intermediate point reached
            active=i;
            for(uint axis=0;axis<3;axis++)target[axis]=pt[axis];
            break;
        }
    }
    float wvel[3]={0,0,0};
    if(!hold) {
        float dx=target[0]-pos[0],dy=target[1]-pos[1],dz=target[2]-pos[2];
        const float span=sqrt(dx*dx+dy*dy+dz*dz);
        if(span>1e-6f) {
            float speed=phase_speed;
            const bool final_leg = (sc.mode!=3u && sc.mode!=4u) ||
                                   true;  // stored polylines end at the goal
            if(final_leg && span<=0.35f) speed=0.0f;
            else if(final_leg && span<=0.6f) speed=0.4f;
            speed=min(speed,span/0.05f);   // never overshoot the target point
            wvel[0]=dx/span*speed;wvel[1]=dy/span*speed;wvel[2]=dz/span*speed;
        }
    }
    RLPhysicsState s=states[n];float r[9];sim_rotation(s.orientation_wxyz,r);
    float body[3];
    for(uint k=0;k<3;k++) body[k]=r[k]*wvel[0]+r[3+k]*wvel[1]+r[6+k]*wvel[2];
    const uint slot_idx=(n*8u+runs[n].steps%8u)*4u;
    for(uint j=0;j<4;j++) {
        const float cmd = j<3 ? body[j]/cfg.speed : 0.0f;
        commands[slot_idx+j]=cmd;
        runs[n].previous_nav[j]=cmd;
    }
    float v[3]={runs[n].previous_nav[0]*cfg.speed,runs[n].previous_nav[1]*cfg.speed,
                runs[n].previous_nav[2]*cfg.speed};
    if(cfg.velocity_contract==1) {
        float magnitude=sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]);
        float scale=min(1.0f,cfg.speed/max(magnitude,1e-8f));
        for(uint j=0;j<3;j++)v[j]*=scale;
    }
    for(uint j=0;j<3;j++) runs[n].desired_velocity[j]=r[j*3]*v[0]+r[j*3+1]*v[1]+r[j*3+2]*v[2];
}
)MSL";

// ------------------------------------------------------------- run plumbing
struct DynBankControlHost {
    uint32_t dyn_count, static_count, dyn_slots, static_slots, period, pad;
};
static_assert(sizeof(DynBankControlHost) == 24, "dyn bank control ABI");

struct DynamicRun {
    Sim& sim;
    id<MTLBuffer> dyn_bank, static_bank, control, script;
    id<MTLComputePipelineState> apply_p, script_p;
    std::vector<id<MTLBuffer>> obs_configs;  // per-tick mode-18 variants (ablated arm)
};

static std::vector<id<MTLBuffer>> make_obs_configs(Sim& sim, bool ablate) {
    std::vector<id<MTLBuffer>> out;
    if (!ablate) return out;
    for (uint32_t t = 0; t < sim.horizon; t++) {
        SimConfig alt{};
        std::memcpy(&alt, sim.configs[t].contents, sizeof(SimConfig));
        alt.mode = 18;  // source's own previous-depth ablation branch
        out.push_back(sim.m.buffer(sizeof(SimConfig), &alt));
    }
    return out;
}

static DynamicRun make_dynamic_run(Sim& sim,
                                   const std::vector<DynamicEntry>& dyn_bank,
                                   const std::vector<StaticEntry>& static_bank,
                                   const DynBankControlHost& control,
                                   id<MTLComputePipelineState> apply_p,
                                   id<MTLComputePipelineState> script_p,
                                   bool ablate) {
    DynamicRun run{sim, nullptr, nullptr, nullptr, nullptr, apply_p, script_p, {}};
    // Metal buffers cannot be zero-length: keep a one-entry dummy when a bank
    // is unused for this run (its count guard makes it unreachable).
    DynamicEntry dyn_dummy{};
    StaticEntry static_dummy{};
    run.dyn_bank = sim.m.buffer(std::max<size_t>(dyn_bank.size(), 1) * sizeof(DynamicEntry),
                                dyn_bank.empty() ? static_cast<const void*>(&dyn_dummy)
                                                 : static_cast<const void*>(dyn_bank.data()));
    run.static_bank = sim.m.buffer(std::max<size_t>(static_bank.size(), 1) * sizeof(StaticEntry),
                                   static_bank.empty() ? static_cast<const void*>(&static_dummy)
                                                       : static_cast<const void*>(static_bank.data()));
    run.control = sim.m.buffer(sizeof(control), &control);
    std::vector<DynScriptControl> scripts(std::max<uint32_t>(sim.cfg.n, 1), DynScriptControl{0, 0, 0, 0.0f});
    run.script = sim.m.buffer(scripts.size() * sizeof(DynScriptControl), scripts.data());
    run.obs_configs = make_obs_configs(sim, ablate);
    return run;
}

// Witness flights map env n -> entry n of the current slice, so one uniform
// mode array suffices; per-controller timing lives in each entry.
static void set_script(DynamicRun& run, uint32_t mode) {
    std::vector<DynScriptControl> scripts(run.sim.cfg.n);
    for (uint32_t n = 0; n < run.sim.cfg.n; n++)
        scripts[n] = DynScriptControl{mode, n, 1u, 0.0f};
    std::memcpy(run.script.contents, scripts.data(),
                scripts.size() * sizeof(DynScriptControl));
}

static void dynamic_install(DynamicRun& run, id<MTLCommandBuffer> cb, id<MTLBuffer> c) {
    run.sim.m.dispatch(cb, run.apply_p, run.sim.cfg.n,
        {run.sim.states, run.sim.runs, run.sim.worlds, run.sim.task_states,
         run.dyn_bank, run.static_bank, run.control, run.sim.task_control, c}, 64);
}

// One training/eval tick. `ablate` swaps ONLY the observe config (mode 18);
// actor, act and advance always keep the real config. `scripted` replaces
// actor+act with the witness controller (witness flights only).
static void dynamic_tick(DynamicRun& run, id<MTLCommandBuffer> cb, uint32_t tick,
                         bool ablate, bool scripted, bool install_before) {
    Sim& sim = run.sim;
    auto c = sim.configs[tick % sim.horizon];
    if (install_before) dynamic_install(run, cb, c);
    sim.m.dispatch(cb, sim.depth_p, sim.cfg.n * 320,
                   {sim.states, sim.runs, sim.worlds, sim.sensors, sim.physics, c, sim.poses});
    auto c_obs = ablate ? run.obs_configs[tick % sim.horizon] : c;
    if (sim.cfg.geometry_memory) {
        // Corrected depth-history ablation (decision-corrected-history §2):
        // with ablate=1 the swept-point memory is built/queried under the
        // mode-18 config, which caps memory_valid to the CURRENT frame, so
        // past-frame range hits cannot reach the clearance values that feed
        // the geometry prior. ablate=0 keeps c — byte-identical to the
        // original code path (verified by Gate G2 per-flight parity).
        auto c_mem = ablate ? c_obs : c;
        sim.m.dispatch(cb, sim.memory_points_p, sim.cfg.n * 640,
            {sim.states, sim.runs, sim.sensors, sim.poses, sim.memory_points, sim.physics, c_mem});
        sim.m.dispatch(cb, sim.memory_candidates_p, sim.cfg.n * 85,
            {sim.states, sim.worlds, sim.memory_points, sim.memory_clearances, c_mem});
    }
    if (!scripted) {
        sim.m.dispatch(cb, sim.observe_p, sim.cfg.n,
            {sim.states, sim.runs, sim.worlds, sim.sensors, sim.obs, sim.co, sim.physics,
             c_obs, sim.poses, sim.memory_clearances}, 64);
        const size_t row_offset = size_t(tick % sim.horizon) * sim.cfg.n;
        encode(sim.m, cb, sim.simd_actor ? "ppo_actor_forward_simd_fused" : "ppo_actor_forward",
               sim.simd_actor ? ((size_t(sim.cfg.n) + 7) / 8) * 256 : sim.cfg.n,
               {{sim.obs, row_offset * fixed_ppo::actor_obs_dim * 4},
                {sim.actor, 0}, {sim.actor_workspace, 0},
                {sim.actions, row_offset * fixed_ppo::action_dim * 4},
                {sim.env_count, 0}}, sim.simd_actor ? 256 : 64);
        sim.m.dispatch(cb, sim.act_p, sim.cfg.n,
            {sim.states, sim.runs, sim.worlds, sim.obs, sim.co, sim.actor, sim.critic,
             sim.actions, sim.logp, sim.values, sim.commands, sim.physics, c}, 64);
    } else {
        sim.m.dispatch(cb, run.script_p, sim.cfg.n,
            {sim.states, sim.runs, sim.worlds, run.dyn_bank, run.script, sim.commands, c}, 64);
    }
    sim.m.dispatch(cb, sim.advance_p, sim.cfg.n,
        {sim.states, sim.runs, sim.worlds, sim.sensors, sim.commands, sim.raptor, sim.critic,
         sim.rewards, sim.next_values, sim.terminated, sim.truncated, sim.physics, c,
         sim.bank_worlds, sim.bank_schedule, sim.bank_control, sim.bank_active_ids,
         sim.bank_transition_ids, sim.environment_physics, sim.runtime_control,
         sim.task_states, sim.task_control, sim.potential_fields, sim.potential_spec,
         sim.potential_control}, 64);
    dynamic_install(run, cb, c);  // catch auto-resets before the next tick
}

static void dynamic_collect(DynamicRun& run, id<MTLCommandBuffer> cb, uint32_t count,
                            bool ablate) {
    const uint32_t base = run.sim.cfg.tick;
    for (uint32_t step = 0; step < count; step++)
        dynamic_tick(run, cb, base + step, ablate, false, step == 0);
}

// Fresh-episode install without advancing: reset-contract checks read states,
// task records and the episode clock at a true episode start.
static void dynamic_probe(DynamicRun& run, id<MTLCommandBuffer> cb) {
    Sim& sim = run.sim;
    auto c = sim.configs[0];
    dynamic_install(run, cb, c);
}
// -------------------------------------------------------- bank generation
struct HostRng {
    uint32_t s;
    explicit HostRng(uint32_t seed) : s(seed ? seed : 1u) {}
    uint32_t raw() { return wrng(s); }
    float uniform() { return float(raw() >> 8) * (1.0f / 16777216.0f); }
    float symmetric() { return 2.0f * uniform() - 1.0f; }
    WVec unit() {
        const float z = symmetric();
        const float angle = 6.28318530718f * uniform();
        const float radial = std::sqrt(std::max(0.0f, 1.0f - z * z));
        return wv(radial * std::cos(angle), radial * std::sin(angle), z);
    }
};

// threat_inside_room comes from the snapshot (threat_evaluation.hpp) — the
// same source guard the published controlled-threat matrix used.

// TRAIN speeds are the source set; dev adds unseen values inside the same
// declared range (decision.md §7).
static float train_speed(uint32_t g) {
    static const float k[3] = {0.5f, 1.0f, 2.0f};
    return k[(g / 2) % 3];
}
static float dev_speed(uint32_t g) {
    static const float k[5] = {0.5f, 0.75f, 1.0f, 1.5f, 2.0f};
    return k[(g / 2) % 5];
}

// One draw. Returns REJ_NONE with a fully labeled host-eligible entry, or the
// host-side reject reason. RNG advances on every draw regardless (rejections
// are part of the recorded stream).
// Path guards for a placed threat (used by draw and by the pair mirror after
// the velocity flip): room containment and static-geometry sweep over the
// nominal-TTC window.
static uint32_t threat_path_guards(const WWorld& world, const float center[3],
                                   const float velocity[3], float ttc) {
    const WVec c = wv(center[0], center[1], center[2]);
    const WVec v = wv(velocity[0], velocity[1], velocity[2]);
    if (!threat_inside_room(c, kCombinedRadius)) return REJ_THREAT_OVERLAP;
    for (uint32_t sample = 0; sample <= 8; sample++) {
        const float t = ttc * (float(sample) / 8.0f);
        const WVec p = wa(c, wm(v, t));
        if (!threat_inside_room(p, kCombinedRadius)) return REJ_THREAT_OVERLAP;
        if (wclearance(world, p, t) <= kSphereRadius) return REJ_THREAT_OVERLAP;
    }
    return REJ_NONE;
}

// Trajectory-dependent labels/homework for one entry whose threat obstacle is
// already the last obstacle in entry.world: hover-safety label, kinematic
// wait-departure time, retreat point and retreat departure time. No RNG —
// safe to recompute for a mirrored pair member.
static void compute_threat_timings(DynamicEntry& out) {
    const uint32_t threat_index = out.world.count - 1;
    const float* start = out.start_position;
    const float* goal = out.goal_position;
    float ex = goal[0] - start[0], ey = goal[1] - start[1], ez = goal[2] - start[2];
    const float el = std::sqrt(ex * ex + ey * ey + ez * ez);
    ex /= std::max(el, 1e-6f); ey /= std::max(el, 1e-6f); ez /= std::max(el, 1e-6f);

    // hover-safety label (kinematic; the plant wait flight is the authority)
    out.labels &= ~LBL_HOVER_SAFE;
    out.wait_until_s = 1e9f;
    float closest = 1e9f;
    for (uint32_t i = 0; i <= 120; i++) {
        float tp[3];
        threat_position(out.world, threat_index, float(i) * 0.05f, tp);
        closest = std::min(closest, distance3(start, tp));
    }
    if (closest > kCombinedRadius + kHoverMarginM) {
        out.labels |= LBL_HOVER_SAFE;
        float wait_t = 1e9f;
        if (estimate_wait_time(out.world, threat_index, start, goal,
                               out.route_length / 1.2f + 1.0f, wait_t))
            out.wait_until_s = wait_t;
    }

    // retreat point (backward along the corridor, room-clamped, must be clear)
    for (uint32_t axis = 0; axis < 3; axis++) out.retreat_point[axis] = 0.0f;
    for (float back : {1.2f, 0.8f, 0.6f}) {
        float rp[3] = {start[0] - ex * back, start[1] - ey * back, start[2] - ez * back};
        rp[0] = std::max(-1.7f, rp[0]); rp[1] = std::max(-4.7f, std::min(4.7f, rp[1]));
        rp[2] = std::max(0.3f, std::min(4.7f, rp[2]));
        if (wclearance(out.world, wv(rp[0], rp[1], rp[2]), 0.0f) > 0.30f) {
            for (uint32_t axis = 0; axis < 3; axis++) out.retreat_point[axis] = rp[axis];
            break;
        }
    }
    // retreat departure: after the threat passes the original start pose
    out.retreat_until_s = 1e9f;
    {
        float best_t0 = -1.0f, best_d = 1e9f;
        for (uint32_t i = 0; i <= 120; i++) {
            float tp[3];
            const float t = float(i) * 0.05f;
            threat_position(out.world, threat_index, t, tp);
            const float d = distance3(start, tp);
            if (d < best_d) { best_d = d; best_t0 = t; }
        }
        if (best_t0 >= 0.0f) {
            for (uint32_t i = uint32_t(best_t0 / 0.05f) + 1; i <= 240; i++) {
                float tp[3];
                threat_position(out.world, threat_index, float(i) * 0.05f, tp);
                if (distance3(start, tp) > kCombinedRadius + kHoverMarginM) {
                    out.retreat_until_s = float(i) * 0.05f;
                    break;
                }
            }
        }
    }
}

// pair_sign > 0: paired-generation mode — the threat is placed ON the
// corridor encounter point (shared jitter only) so both pair members share
// the identical instantaneous position/speed; the mirror member flips only
// the velocity direction (decision-corrected-history §7). force_kind >= 0
// pins threat_kind for a pair (members of a pair always share it).
static uint32_t draw_candidate(HostRng& rng, uint32_t draw_index, bool train,
                               DynamicEntry& out, int pair_sign = -2,
                               int force_kind = -1) {
    static const uint32_t kFamilies[5] = {0, 1, 2, 4, 5};
    out = DynamicEntry{};
    out.draw_index = draw_index;
    out.family = kFamilies[draw_index % 5];
    out.scene_seed = rng.raw();
    out.threat_kind = force_kind >= 0 ? uint32_t(force_kind)
                                      : uint32_t(draw_index % 2);
    out.start_stratum = draw_index % 3;
    out.warning_s = -1.0f;
    out.causal_collision_s = -1.0f;
    out.measured_min_ttc_s = 1e9f;
    out.wait_until_s = 1e9f;
    out.retreat_until_s = 1e9f;
    out.threat_speed_mps = train ? train_speed(draw_index) : dev_speed(draw_index);
    wgenerate(out.world, out.scene_seed, out.family, 4.0f);

    float start[3], goal[3];
    for (uint32_t axis = 0; axis < 3; axis++) {
        start[axis] = kStartLow[axis] + (kStartHigh[axis] - kStartLow[axis]) * rng.uniform();
        out.start_position[axis] = start[axis];
    }
    if (wclearance(out.world, wv(start[0], start[1], start[2]), 0.0f) <= 0.30f)
        return REJ_START_BLOCKED;

    const WVec direction = rng.unit();
    const float distance = 1.0f + 2.0f * rng.uniform();
    for (uint32_t axis = 0; axis < 3; axis++)
        goal[axis] = start[axis] + (axis == 0 ? direction.x : axis == 1 ? direction.y : direction.z) * distance;
    if (!inside_goal_box(goal)) return REJ_GOAL_OUT_OF_BOUNDS;
    if (wclearance(out.world, wv(goal[0], goal[1], goal[2]), 0.0f) <= 0.30f)
        return REJ_GOAL_BLOCKED;
    for (uint32_t axis = 0; axis < 3; axis++) out.goal_position[axis] = goal[axis];
    out.world.goal[0] = goal[0]; out.world.goal[1] = goal[1]; out.world.goal[2] = goal[2];

    out.start_yaw = rng.symmetric() * 3.14159265359f;
    // User clarification (decision-corrected-history §7): goal-in-frustum and
    // goal-LOS are LABELS, never rejection reasons — out-of-view goals are
    // legitimate training/evaluation utility. Obstacle observability stays a
    // gate further down (warning_s vs plant contact).
    const bool in_frustum = goal_in_frustum(start, out.start_yaw, goal);
    const bool los_clear = goal_los_clear(out.world, start, goal);
    if (in_frustum) out.labels |= LBL_GROUNDED;
    if (los_clear) out.labels |= LBL_LOS_CLEAR;

    const float straight_clear = navigation_task_segment_clearance(
        out.world, wv(start[0], start[1], start[2]), wv(goal[0], goal[1], goal[2]));
    if (straight_clear <= 0.02f) return REJ_STATIC_STRAIGHT_BLOCKED;
    out.straight_min_clearance = straight_clear;
    out.route_length = distance3(start, goal);
    out.clearance_start = wclearance(out.world, wv(start[0], start[1], start[2]), 0.0f);
    out.clearance_goal = wclearance(out.world, wv(goal[0], goal[1], goal[2]), 0.0f);

    // start velocity strata (independent of the goal direction for stratum 2)
    float dir[3] = {goal[0] - start[0], goal[1] - start[1], goal[2] - start[2]};
    const float span = std::sqrt(dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]);
    for (uint32_t axis = 0; axis < 3; axis++) dir[axis] /= std::max(span, 1e-6f);
    if (out.start_stratum == 0) {
        for (uint32_t axis = 0; axis < 3; axis++) out.start_velocity[axis] = 0.0f;
    } else if (out.start_stratum == 1) {
        const float speed = 0.3f + 0.5f * rng.uniform();
        for (uint32_t axis = 0; axis < 3; axis++) out.start_velocity[axis] = dir[axis] * speed;
    } else {
        float horiz[2] = {dir[0], dir[1]};
        const float hl = std::sqrt(horiz[0] * horiz[0] + horiz[1] * horiz[1]);
        float perp[3];
        if (hl > 0.1f) { perp[0] = -horiz[1] / hl; perp[1] = horiz[0] / hl; }
        else { perp[0] = 0.0f; perp[1] = 1.0f; }
        perp[2] = 0.0f;
        const float speed = 0.3f + 0.5f * rng.uniform();
        for (uint32_t axis = 0; axis < 3; axis++) out.start_velocity[axis] = perp[axis] * speed;
    }

    // Threat placement: source TTC construction generalized to the start->goal
    // axis (cruise reference = 1.5 m/s cap, same as the source matrix).
    out.nominal_ttc_s = 0.5f + 1.5f * rng.uniform();
    const float threat_speed = out.threat_speed_mps;
    const float cruise = kTaskSpeedCapMps;
    float ex = dir[0], ey = dir[1], ez = dir[2];
    float perp[3] = {-ey, ex, 0.0f};
    float pl = std::sqrt(perp[0] * perp[0] + perp[1] * perp[1]);
    if (pl < 1e-3f) { perp[0] = 0.0f; perp[1] = 1.0f; pl = 1.0f; }
    perp[0] /= pl; perp[1] /= pl;
    const float off_h = rng.symmetric() * 0.10f;
    const float off_v = rng.symmetric() * 0.08f;
    const float sign = pair_sign > 0 ? float(pair_sign)
                                     : (((draw_index / 2) % 2) ? 1.0f : -1.0f);
    WVec center, velocity;
    if (pair_sign > 0) {
        // Paired design: identical instantaneous threat position within the
        // pair; only the velocity direction differs between members.
        if (out.threat_kind == 0) {  // approach pair: on-corridor, mirrored bearing
            const float s = (cruise + threat_speed) * out.nominal_ttc_s + kCombinedRadius;
            float nd[3] = {-ex + perp[0] * sign, -ey + perp[1] * sign, -ez};
            const float nl = std::sqrt(nd[0] * nd[0] + nd[1] * nd[1] + nd[2] * nd[2]);
            for (uint32_t axis = 0; axis < 3; axis++) nd[axis] /= nl;
            center = wv(start[0] + ex * s + perp[0] * off_h,
                        start[1] + ey * s + perp[1] * off_h,
                        start[2] + ez * s + off_v);
            velocity = wv(nd[0] * threat_speed, nd[1] * threat_speed,
                          nd[2] * threat_speed);
        } else {  // crossing pair: sitting on the corridor line, sweeps ±perp
            const float s = cruise * out.nominal_ttc_s;
            center = wv(start[0] + ex * s + perp[0] * off_h,
                        start[1] + ey * s + perp[1] * off_h,
                        start[2] + ez * s + off_v);
            velocity = wv(perp[0] * threat_speed * sign,
                          perp[1] * threat_speed * sign, 0.0f);
        }
    } else if (out.threat_kind == 0) {  // approach: oblique front-quarter bearing
        // Attempt 1 (preserved in banks/attempt1-crossing-only + bank log)
        // showed dead-on approaches are physically one-action: the threat
        // passes through the start pose, so waiting/retreating are unsafe and
        // >=2 feasible actions can never hold. The approach now travels at a
        // 45-degree front-quarter bearing (declared correction before any
        // training, decision.md addendum): it still crosses the corridor at
        // cruise*nominal_ttc (source-consistent meeting point) but passes to
        // the side of the start pose, which is what makes yield feasible.
        const float s_meet = cruise * out.nominal_ttc_s;
        float nd[3] = {-ex + perp[0] * sign, -ey + perp[1] * sign, -ez};
        const float nl = std::sqrt(nd[0] * nd[0] + nd[1] * nd[1] + nd[2] * nd[2]);
        for (uint32_t axis = 0; axis < 3; axis++) nd[axis] /= nl;
        const float meet[3] = {start[0] + ex * s_meet, start[1] + ey * s_meet,
                               start[2] + ez * s_meet};
        const float lag = threat_speed * out.nominal_ttc_s;
        center = wv(meet[0] - nd[0] * lag, meet[1] - nd[1] * lag,
                    meet[2] - nd[2] * lag);
        velocity = wv(nd[0] * threat_speed, nd[1] * threat_speed,
                      nd[2] * threat_speed);
    } else {                     // crossing: sweeps the corridor from the side
        const float s = cruise * out.nominal_ttc_s;
        const float cross = threat_speed * out.nominal_ttc_s * sign;
        center = wv(start[0] + ex * s - perp[0] * cross + perp[0] * off_h,
                    start[1] + ey * s - perp[1] * cross + perp[1] * off_h,
                    start[2] + ez * s + off_v);
        velocity = wv(perp[0] * threat_speed * sign, perp[1] * threat_speed * sign, 0.0f);
    }
    {
        const float c3[3] = {center.x, center.y, center.z};
        const float v3[3] = {velocity.x, velocity.y, velocity.z};
        const uint32_t guard = threat_path_guards(out.world, c3, v3,
                                                  out.nominal_ttc_s);
        if (guard != REJ_NONE) return guard;
        if (distance3(start, c3) <= kCombinedRadius)
            return REJ_THREAT_OVERLAP;
    }
    wadd(out.world, 1, center, wv(kSphereRadius, kSphereRadius, kSphereRadius), velocity);
    out.clearance_start = std::min(out.clearance_start,
        wclearance(out.world, wv(start[0], start[1], start[2]), 0.0f));
    if (out.clearance_start <= 0.30f) return REJ_THREAT_OVERLAP;
    compute_threat_timings(out);

    // detour polylines: one intermediate point each side of the corridor
    float mid[3];
    for (uint32_t axis = 0; axis < 3; axis++)
        mid[axis] = 0.5f * (start[axis] + goal[axis]);
    const float offset = kDetourOffsetMinM +
        (kDetourOffsetMaxM - kDetourOffsetMinM) * rng.uniform();
    for (int32_t side = 0; side <= 1; side++) {
        const float sgn = side ? 1.0f : -1.0f;
        float mp[3] = {mid[0] + perp[0] * offset * sgn,
                       mid[1] + perp[1] * offset * sgn, mid[2]};
        mp[0] = std::max(-1.7f, std::min(13.7f, mp[0]));
        mp[1] = std::max(-4.7f, std::min(4.7f, mp[1]));
        mp[2] = std::max(0.3f, std::min(4.7f, mp[2]));
        const float c1 = navigation_task_segment_clearance(
            out.world, wv(start[0], start[1], start[2]), wv(mp[0], mp[1], mp[2]));
        const float c2 = navigation_task_segment_clearance(
            out.world, wv(mp[0], mp[1], mp[2]), wv(goal[0], goal[1], goal[2]));
        float* route = side ? &out.route_plus[0][0] : &out.route_minus[0][0];
        if (c1 > 0.02f && c2 > 0.02f) {
            route[0] = mp[0]; route[1] = mp[1]; route[2] = mp[2];
            route[3] = goal[0]; route[4] = goal[1]; route[5] = goal[2];
            if (side) out.route_plus_count = 2; else out.route_minus_count = 2;
        } else {
            route[0] = goal[0]; route[1] = goal[1]; route[2] = goal[2];
            if (side) out.route_plus_count = 1; else out.route_minus_count = 1;
        }
    }
    return REJ_NONE;
}
// Clone member A into member B: identical world/start/goal/threat position
// and speed; ONLY the threat velocity direction is mirrored across the
// corridor's lateral axis (v - 2·proj_perp(v) — exact for both threat kinds),
// then the path guards and trajectory-dependent timings are recomputed for
// the mirrored direction. Returns REJ_NONE when B is host-eligible.
static uint32_t mirror_pair_member(DynamicEntry& b) {
    require(b.world.count > 0, "pair mirror needs a threat obstacle");
    const uint32_t idx = b.world.count - 1;
    WObstacle& obstacle = b.world.obstacles[idx];
    const float* start = b.start_position;
    const float* goal = b.goal_position;
    float ex = goal[0] - start[0], ey = goal[1] - start[1], ez = goal[2] - start[2];
    const float el = std::sqrt(ex * ex + ey * ey + ez * ez);
    ex /= std::max(el, 1e-6f); ey /= std::max(el, 1e-6f); ez /= std::max(el, 1e-6f);
    float perp[3] = {-ey, ex, 0.0f};
    const float pl = std::sqrt(perp[0] * perp[0] + perp[1] * perp[1]);
    if (pl < 1e-3f) { perp[0] = 0.0f; perp[1] = 1.0f; }
    else { perp[0] /= pl; perp[1] /= pl; }
    const float velocity[3] = {obstacle.velocity[0], obstacle.velocity[1],
                               obstacle.velocity[2]};
    const float lateral = velocity[0] * perp[0] + velocity[1] * perp[1];
    obstacle.velocity[0] = velocity[0] - 2.0f * lateral * perp[0];
    obstacle.velocity[1] = velocity[1] - 2.0f * lateral * perp[1];
    obstacle.velocity[2] = velocity[2];  // crossing z is 0; approach keeps z sense
    const float center[3] = {obstacle.center[0], obstacle.center[1],
                             obstacle.center[2]};
    const float v3[3] = {obstacle.velocity[0], obstacle.velocity[1],
                         obstacle.velocity[2]};
    // Guards must not test the path against the threat itself (A's guards ran
    // before the obstacle was appended; B inherits it). The threat is the last
    // obstacle, so dropping it from the count excludes it exactly.
    WWorld static_only = b.world;
    require(static_only.count > 0, "pair mirror world lost its threat");
    static_only.count -= 1;
    const uint32_t guard = threat_path_guards(static_only, center, v3,
                                              b.nominal_ttc_s);
    if (guard != REJ_NONE) return guard;
    if (distance3(start, center) <= kCombinedRadius) return REJ_THREAT_OVERLAP;
    compute_threat_timings(b);
    return REJ_NONE;
}

// --------------------------------------------------------- plant witnesses
struct FlightResult {
    bool success = false, collision = false, threat_contact = false;
    float collision_time = -1.0f, min_ttc = 1e9f;
    uint32_t ticks = 0;
};

static NavigationTaskControl dynamic_task_recipe(float time_cost_per_s = 0.2f) {
    NavigationTaskControl control =
        navigation_training::task_settings(NAV_TASK_STAGE_OPEN_GOAL, 0u);
    control.config.goal_distance_min_m = 1.0f;
    control.config.goal_distance_max_m = 3.0f;
    control.config.time_cost_per_s = time_cost_per_s;
    return control;
}

static void enable_task_control(Sim& sim, const NavigationTaskControl& control) {
    require(std::fabs(control.config.nav_period_s - sim.cfg.substeps * 0.01f) < 1e-6f,
            "dynamic task clock must match the simulator navigation period");
    require(((const ChallengeBankControl*)sim.bank_control.contents)->enabled == 0,
            "dynamic training runs without the frozen challenge bank");
    require(((const NavigationRuntimeConfig*)sim.runtime_control.contents)->enabled == 0,
            "dynamic training runs on the nominal plant");
    std::memcpy(sim.task_control.contents, &control, sizeof(control));
}

static SimConfig witness_config(uint32_t n) {
    SimConfig config;  // host defaults: substeps 5, sensor_period 1, contract 1
    config.n = n;
    config.mode = 2;
    config.eval = 1;
    config.seed = 800001;
    config.speed = kTaskSpeedCapMps;
    config.distance = 4;
    config.max_steps = 400;
    config.geometry_memory = 0;
    config.entropy_coef = 0.001f;
    config.learning_rate = 0.0001f;
    return config;
}

// Fly one controller over a slice of entries. `per_tick` (straight only) syncs
// every tick to pin the exact contact tick, classify threat-vs-static contact,
// and measure the in-flight state-based minimum TTC. Evasions only need the
// terminal outcome and run as one command buffer, like the published evals.
static std::vector<FlightResult> fly_controller(Metal& metal,
                                                const std::vector<DynamicEntry>& slice,
                                                uint32_t mode, bool per_tick) {
    const uint32_t n = uint32_t(slice.size());
    SimConfig config = witness_config(n);
    Sim sim(metal, config, 32);
    enable_task_control(sim, dynamic_task_recipe());
    DynBankControlHost control{n, 0, 1, 0, 1, 0};
    auto run = make_dynamic_run(sim, slice, {}, control,
                                metal.pipeline("dynamic_task_apply"),
                                metal.pipeline("dynamic_script_act"), false);
    set_script(run, mode);
    std::vector<FlightResult> results(n);
    std::vector<bool> seen(n, false);
    std::vector<float> prev_range(n, -1.0f);
    auto probe = [metal.queue commandBuffer];
    dynamic_probe(run, probe);
    metal.finish(probe);
    if (!per_tick) {
        auto cb = [metal.queue commandBuffer];
        for (uint32_t t = 0; t < config.max_steps; t++)
            dynamic_tick(run, cb, t, false, true, t == 0);
        metal.finish(cb);
        const auto* runs = (const SimRun*)sim.runs.contents;
        for (uint32_t e = 0; e < n; e++) {
            results[e].success = runs[e].successes > 0;
            results[e].collision = runs[e].collisions > 0;
            results[e].ticks = runs[e].steps;
        }
        return results;
    }
    for (uint32_t t = 0; t < config.max_steps; t++) {
        auto cb = [metal.queue commandBuffer];
        dynamic_tick(run, cb, t, false, true, t == 0);
        metal.finish(cb);
        const auto* runs = (const SimRun*)sim.runs.contents;
        const auto* states = (const RLPhysicsState*)sim.states.contents;
        bool all_done = true;
        for (uint32_t e = 0; e < n; e++) {
            if (runs[e].episodes > 0) {
                if (!seen[e]) {
                    seen[e] = true;
                    FlightResult& r = results[e];
                    r.success = runs[e].successes > 0;
                    r.collision = runs[e].collisions > 0;
                    r.ticks = runs[e].steps;
                    if (r.collision) {
                        const DynamicEntry& entry = slice[e];
                        const uint32_t threat = entry.world.count - 1;
                        float tp[3];
                        threat_position(entry.world, threat, runs[e].elapsed, tp);
                        const float pos[3] = {states[e].position[0], states[e].position[1],
                                              states[e].position[2]};
                        r.collision_time = runs[e].elapsed;
                        r.threat_contact =
                            distance3(pos, tp) <= kCombinedRadius + 0.02f;
                    }
                }
                continue;
            }
            all_done = false;
            const DynamicEntry& entry = slice[e];
            const uint32_t threat = entry.world.count - 1;
            float tp[3];
            threat_position(entry.world, threat, runs[e].elapsed, tp);
            const float pos[3] = {states[e].position[0], states[e].position[1],
                                  states[e].position[2]};
            const float range = distance3(pos, tp);
            if (prev_range[e] >= 0.0f) {
                const float closing = (prev_range[e] - range) / 0.05f;
                if (closing > 1e-3f)
                    results[e].min_ttc = std::min(results[e].min_ttc, range / closing);
            }
            prev_range[e] = range;
        }
        if (all_done) break;
    }
    return results;
}

// --------------------------------------------------------------- bank I/O
static void write_dynamic_bank(const std::string& path,
                               const std::vector<DynamicEntry>& entries,
                               uint32_t period) {
    DynBankHeader header{};
    std::memcpy(header.magic, "DYBANK1", 7);
    header.version = 1;
    header.period = period;
    header.count = uint32_t(entries.size());
    header.entry_bytes = uint32_t(sizeof(DynamicEntry));
    const std::string entry_hash =
        ppo_safeguard_sha256(entries.data(), entries.size() * sizeof(DynamicEntry));
    std::memcpy(header.entry_sha256, entry_hash.data(), entry_hash.size());
    const std::filesystem::path destination(path);
    if (destination.has_parent_path())
        std::filesystem::create_directories(destination.parent_path());
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    require(bool(file), "cannot write dynamic bank " + path);
    file.write(reinterpret_cast<const char*>(&header), sizeof(header));
    file.write(reinterpret_cast<const char*>(entries.data()),
               std::streamsize(entries.size() * sizeof(DynamicEntry)));
    file.flush();
    require(bool(file), "dynamic bank write failed " + path);
    std::cout << "dyn_bank_write " << path << " entries=" << entries.size()
              << " period=" << period << " entry_sha256=" << entry_hash << "\n";
}

static std::vector<DynamicEntry> read_dynamic_bank(const std::string& path,
                                                   DynBankControlHost& control) {
    std::ifstream file(path, std::ios::binary);
    require(bool(file), "cannot open dynamic bank " + path);
    DynBankHeader header{};
    file.read(reinterpret_cast<char*>(&header), sizeof(header));
    require(bool(file) && std::memcmp(header.magic, "DYBANK1", 7) == 0 &&
               header.version == 1 && header.entry_bytes == sizeof(DynamicEntry),
            "invalid dynamic bank header " + path);
    std::vector<DynamicEntry> entries(header.count);
    file.read(reinterpret_cast<char*>(entries.data()),
              std::streamsize(entries.size() * sizeof(DynamicEntry)));
    require(bool(file), "truncated dynamic bank " + path);
    char trailing;
    require(!file.read(&trailing, 1), "dynamic bank has trailing bytes " + path);
    const std::string entry_hash =
        ppo_safeguard_sha256(entries.data(), entries.size() * sizeof(DynamicEntry));
    require(std::memcmp(header.entry_sha256, entry_hash.data(), 64) == 0,
            "dynamic bank entry hash mismatch " + path);
    control = DynBankControlHost{0, 0, 0, 0, header.period, 0};
    return entries;
}

static std::vector<StaticEntry> read_static_bank(const std::string& path,
                                                 uint32_t& period) {
    struct LocalHeader {
        char magic[8];
        uint32_t version, period, count, entry_bytes;
        char entry_sha256[64];
    };
    static_assert(sizeof(LocalHeader) == 88, "WPBANK1 header ABI");
    std::ifstream file(path, std::ios::binary);
    require(bool(file), "cannot open static bank " + path);
    LocalHeader header{};
    file.read(reinterpret_cast<char*>(&header), sizeof(header));
    require(bool(file) && std::memcmp(header.magic, "WPBANK1", 7) == 0 &&
               header.version == 1 && header.entry_bytes == sizeof(StaticEntry),
            "invalid static bank header " + path);
    std::vector<StaticEntry> entries(header.count);
    file.read(reinterpret_cast<char*>(entries.data()),
              std::streamsize(entries.size() * sizeof(StaticEntry)));
    require(bool(file), "truncated static bank " + path);
    char trailing;
    require(!file.read(&trailing, 1), "static bank has trailing bytes " + path);
    const std::string entry_hash =
        ppo_safeguard_sha256(entries.data(), entries.size() * sizeof(StaticEntry));
    require(std::memcmp(header.entry_sha256, entry_hash.data(), 64) == 0,
            "static bank entry hash mismatch " + path);
    period = header.period;
    return entries;
}
// ---------------------------------------------------------------- bank cmd
// Option helpers local to this module (the waypoint runner's are not part of
// the snapshot).
static std::string dyn_option_value(int argc, char** argv, int& index,
                                    const std::string& name) {
    require(index + 1 < argc, "option " + name + " requires a value");
    return argv[++index];
}
static float dyn_option_float(int argc, char** argv, int& index,
                              const std::string& name) {
    return std::stof(dyn_option_value(argc, argv, index, name));
}
static uint32_t dyn_option_uint(int argc, char** argv, int& index,
                                const std::string& name) {
    return uint32_t(std::stoul(dyn_option_value(argc, argv, index, name)));
}

static uint32_t finalize_acceptance(DynamicEntry& e, const FlightResult& straight,
                                    const FlightResult& wait,
                                    const FlightResult& retreat,
                                    const FlightResult& plus,
                                    const FlightResult& minus) {
    e.witness_mask = (straight.success ? 1u : 0u) | (wait.success ? 2u : 0u) |
                     (retreat.success ? 4u : 0u) | (plus.success ? 8u : 0u) |
                     (minus.success ? 16u : 0u);
    if (!straight.collision) return REJ_NOT_CAUSAL;
    if (!straight.threat_contact) return REJ_STATIC_COLLISION;
    e.labels |= LBL_CAUSAL;
    e.causal_collision_s = straight.collision_time;
    e.measured_min_ttc_s =
        straight.min_ttc >= 1e9f ? -1.0f : straight.min_ttc;
    const float warn = first_threat_visible(e.world, e.start_position, e.start_yaw,
                                            e.world.count - 1, straight.collision_time);
    e.warning_s = warn;
    if (warn < 0.0f || straight.collision_time - warn < kWarningMinS)
        return REJ_INSUFFICIENT_WARNING;
    e.labels |= LBL_WARNING_OK;
    const uint32_t actions = uint32_t(wait.success) + retreat.success +
                             plus.success + minus.success;
    if (actions < 2) return REJ_SINGLE_ACTION;
    return REJ_NONE;
}

static void write_records_csv(const std::string& path,
                              const std::vector<DynamicEntry>& records,
                              const std::vector<FlightResult>& straight_flights,
                              const std::map<uint32_t, uint32_t>& partner) {
    std::ofstream csv(path);
    require(bool(csv), "cannot write records CSV " + path);
    csv << std::setprecision(9);
    csv << "draw_index,binned,family,scene_seed,threat_kind,threat_speed_mps,"
           "nominal_ttc_s,start_stratum,route_length_m,clearance_start_m,"
           "clearance_goal_m,straight_min_clearance_m,labels,warning_s,"
           "causal_collision_s,measured_min_ttc_s,wait_until_s,retreat_until_s,"
           "witness_mask,straight_threat_contact,straight_min_ttc_flight,"
           "rejected_reason,pair_partner\n";
    for (size_t i = 0; i < records.size(); i++) {
        const DynamicEntry& e = records[i];
        const FlightResult* f =
            i < straight_flights.size() ? &straight_flights[i] : nullptr;
        const auto it = partner.find(e.draw_index);
        csv << e.draw_index << ',' << (e.rejected_reason == REJ_NONE ? 1 : 0)
            << ',' << e.family << ',' << e.scene_seed << ',' << e.threat_kind
            << ',' << e.threat_speed_mps << ',' << e.nominal_ttc_s << ','
            << e.start_stratum << ',' << e.route_length << ','
            << e.clearance_start << ',' << e.clearance_goal << ','
            << e.straight_min_clearance << ',' << e.labels << ','
            << e.warning_s << ',' << e.causal_collision_s << ','
            << e.measured_min_ttc_s << ',' << e.wait_until_s << ','
            << e.retreat_until_s << ',' << e.witness_mask << ','
            << (f ? int(f->threat_contact) : -1) << ','
            << (f && f->min_ttc < 1e9f ? f->min_ttc : -1.0f) << ','
            << reject_reason_name(e.rejected_reason) << ','
            << (it != partner.end() ? int(it->second) : -1) << '\n';
    }
    csv.flush();
    require(bool(csv), "records CSV write failed " + path);
}

static int command_bank(int argc, char** argv) {
    bool train = false;
    bool dev = false;
    bool pairs = false;
    std::string prefix;
    uint32_t seed = 0, target = 0;
    for (int index = 2; index < argc; index++) {
        const std::string option = argv[index];
        if (option == "--train") train = true;
        else if (option == "--dev") dev = true;
        else if (option == "--pairs") pairs = true;
        else if (option == "--seed") seed = dyn_option_uint(argc, argv, index, option);
        else if (option == "--target") target = dyn_option_uint(argc, argv, index, option);
        else if (prefix.empty()) prefix = option;
        else throw std::runtime_error("unknown dynamic-bank option " + option);
    }
    require(train != dev, "dynamic-bank needs exactly one of --train/--dev");
    require(!pairs || dev,
            "--pairs is a development-bank mode (the frozen train bank is not regenerated)");
    require(!prefix.empty(), "dynamic-bank needs an OUT_PREFIX");
    if (!seed) seed = train ? 20261021u
                            : (pairs ? 20261031u : 20261022u);  // decision §7
    if (!target) target = train ? kTrainDynamicTasks : kDevDynamicTasks;
    const uint32_t period = train ? kMixturePeriod : 1;
    // Safety bound only: solo acceptance is ~1.4% of draws; paired acceptance
    // measured 0.56% (same-position pairs are physically causal only for
    // slow threats — see decision-corrected-history §7 note), so pairs get a
    // 250x cap. Rejection counts are reported either way.
    const uint32_t draw_cap = target * (pairs ? 250 : 150);
    std::cout << "dynamic_bank kind=" << (train ? "train" : "dev")
              << (pairs ? "-pairs" : "") << " seed=" << seed
              << " target=" << target
              << " period=" << period << " entry_bytes=" << sizeof(DynamicEntry)
              << " draw_cap=" << draw_cap << "\n";

    Metal metal;
    metal.compile(base_source() + PPO_TRAINER_MSL + kDynamicMsl);

    HostRng rng(seed);
    std::vector<DynamicEntry> records, accepted;
    std::vector<FlightResult> record_flights;  // straight flight per record
    std::map<uint32_t, uint32_t> rejects;
    std::map<uint32_t, uint32_t> partner;      // pair member draw_index map
    std::vector<uint32_t> pending_record_indices;
    uint32_t draw_index = 0, chunks = 0, pair_attempts = 0;
    const double started = seconds();
    double draw_s = 0, plant_s = 0;

    while (accepted.size() < target && draw_index < draw_cap) {
        // --- host draws: up to 256 host-eligible candidates per chunk
        const double d0 = seconds();
        std::vector<uint32_t> chunk_record_indices;
        uint32_t host_eligible = 0;
        const uint32_t chunk_draw_start = draw_index;
        while (host_eligible < 256 && draw_index - chunk_draw_start < 4096 &&
               draw_index < draw_cap) {
            if (pairs) {
                // Pair attempt consumes two contiguous draw indices (2k,2k+1)
                // so pair identity is derivable as draw_index ^ 1 everywhere
                // (records, eval CSVs) without any ABI change.
                const uint32_t idx_a = draw_index, idx_b = draw_index + 1;
                const int force_kind = int((idx_a / 2) % 2);  // 0 approach, 1 crossing
                DynamicEntry a{};
                const uint32_t reason_a =
                    draw_candidate(rng, idx_a, train, a, +1, force_kind);
                a.rejected_reason = reason_a;
                records.push_back(a);
                record_flights.push_back(FlightResult{});
                if (reason_a == REJ_NONE) {
                    DynamicEntry b = a;               // identical instantaneous state
                    b.draw_index = idx_b;
                    const uint32_t reason_b = mirror_pair_member(b);
                    b.rejected_reason = reason_b;
                    records.push_back(b);
                    record_flights.push_back(FlightResult{});
                    partner[idx_a] = idx_b;
                    partner[idx_b] = idx_a;
                    pair_attempts++;
                    if (reason_b == REJ_NONE) {
                        chunk_record_indices.push_back(uint32_t(records.size() - 2));
                        chunk_record_indices.push_back(uint32_t(records.size() - 1));
                        host_eligible += 2;
                    } else {
                        rejects[reason_b]++;
                        chunk_record_indices.push_back(uint32_t(records.size() - 2));
                        host_eligible++;  // A continues as a labelled singleton
                    }
                } else {
                    rejects[reason_a]++;
                }
                draw_index += 2;
                continue;
            }
            DynamicEntry entry{};
            const uint32_t reason = draw_candidate(rng, draw_index, train, entry);
            entry.rejected_reason = reason;
            records.push_back(entry);
            record_flights.push_back(FlightResult{});
            if (reason == REJ_NONE) {
                chunk_record_indices.push_back(uint32_t(records.size() - 1));
                host_eligible++;
            } else {
                rejects[reason]++;
            }
            draw_index++;
        }
        draw_s += seconds() - d0;
        if (chunk_record_indices.empty()) continue;

        // --- plant flights over the chunk (slices of <= 128)
        const double p0 = seconds();
        for (size_t base = 0; base < chunk_record_indices.size(); base += 128) {
            const size_t count = std::min<size_t>(128, chunk_record_indices.size() - base);
            std::vector<DynamicEntry> slice(count);
            for (size_t i = 0; i < count; i++)
                slice[i] = records[chunk_record_indices[base + i]];
            auto straight = fly_controller(metal, slice, SCRIPT_STRAIGHT, true);
            auto wait = fly_controller(metal, slice, SCRIPT_WAIT, false);
            auto retreat = fly_controller(metal, slice, SCRIPT_RETREAT, false);
            auto plus = fly_controller(metal, slice, SCRIPT_DETROUT_PLUS, false);
            auto minus = fly_controller(metal, slice, SCRIPT_DETROUT_MINUS, false);
            for (size_t i = 0; i < count; i++) {
                const uint32_t rec = chunk_record_indices[base + i];
                record_flights[rec] = straight[i];
                DynamicEntry& e = records[rec];
                const uint32_t reason = finalize_acceptance(
                    e, straight[i], wait[i], retreat[i], plus[i], minus[i]);
                e.rejected_reason = reason;
                if (reason == REJ_NONE) {
                    if (accepted.size() < target) accepted.push_back(e);
                } else {
                    rejects[reason]++;
                }
            }
        }
        plant_s += seconds() - p0;
        chunks++;
        std::cout << "dynamic_bank_chunk chunks=" << chunks << " draws=" << draw_index
                  << " accepted=" << accepted.size() << "/" << target
                  << " record_rows=" << records.size()
                  << " draw_s=" << draw_s << " plant_s=" << plant_s << "\n";
    }

    std::cout << "dynamic_bank_done draws=" << draw_index
              << " record_rows=" << records.size()
              << " accepted=" << accepted.size() << "/" << target
              << " wall_s=" << seconds() - started
              << " draw_s=" << draw_s << " plant_s=" << plant_s << "\n";
    for (const auto& item : rejects)
        std::cout << "dynamic_bank_reject reason=" << reject_reason_name(item.first)
                  << " count=" << item.second << "\n";

    // accepted strata summary (declared axes)
    {
        uint32_t strata[3] = {0, 0, 0}, kinds[2] = {0, 0};
        std::map<uint32_t, uint32_t> speeds;
        for (const auto& e : accepted) {
            strata[e.start_stratum % 3]++;
            kinds[e.threat_kind % 2]++;
            speeds[uint32_t(e.threat_speed_mps * 100.0f)]++;
        }
        std::cout << "dynamic_bank_strata stopped=" << strata[0]
                  << " toward=" << strata[1] << " lateral=" << strata[2]
                  << " approach=" << kinds[0] << " crossing=" << kinds[1] << "\n";
        for (const auto& s : speeds)
            std::cout << "dynamic_bank_speed speed_cms=" << s.first
                      << " count=" << s.second << "\n";
    }

    // Records and manifest are ALWAYS persisted, even on shortfall, so a
    // failed generation leaves analyzable evidence instead of only a log.
    const bool complete = accepted.size() == target;
    if (complete) write_dynamic_bank(prefix + ".bin", accepted, period);
    write_records_csv(prefix + ".csv", records, record_flights, partner);

    // manifest
    uint32_t plant_accepted = 0;
    for (const auto& r : records)
        if (r.rejected_reason == REJ_NONE) plant_accepted++;
    std::map<uint32_t, uint32_t> accepted_set;
    for (const auto& r : records)
        if (r.rejected_reason == REJ_NONE) accepted_set[r.draw_index] = 1;
    uint32_t complete_pairs = 0;
    for (const auto& pr : partner)
        if (pr.first < pr.second && accepted_set.count(pr.first) &&
            accepted_set.count(pr.second))
            complete_pairs++;
    const std::string bin_sha =
        complete ? challenge_evaluation::sha256_file(prefix + ".bin") : "none";
    const std::string csv_sha = challenge_evaluation::sha256_file(prefix + ".csv");
    std::ofstream manifest(prefix + ".json");
    require(bool(manifest), "cannot write manifest " + prefix + ".json");
    manifest << std::setprecision(9)
             << "{\n  \"schema\": \"local-dynamic-bank-v2\",\n"
             << "  \"complete\": " << (complete ? "true" : "false") << ",\n"
             << "  \"kind\": \"" << (train ? "train" : "dev") << "\",\n"
             << "  \"pairs_mode\": " << (pairs ? "true" : "false") << ",\n"
             << "  \"pair_attempts\": " << pair_attempts
             << ",\n  \"accepted_complete_pairs\": " << complete_pairs
             << ",\n  \"seed\": " << seed << ",\n  \"target\": " << target
             << ",\n  \"period\": " << period
             << ",\n  \"draws\": " << draw_index
             << ",\n  \"record_rows\": " << records.size()
             << ",\n  \"accepted_binned\": " << accepted.size()
             << ",\n  \"plant_accepted_total\": " << plant_accepted
             << ",\n  \"entry_bytes\": " << sizeof(DynamicEntry)
             << ",\n  \"bin_sha256\": \"" << bin_sha
             << "\",\n  \"csv_sha256\": \"" << csv_sha
             << "\",\n  \"witness_protocol\": \"real RAPTOR plant, eval=1, "
                "max_steps=400, controllers straight(per-tick)/wait/retreat/"
                "detour+/detour-; accept iff straight collides with the threat "
                "AND >=2 of 4 evasions succeed AND warning >= 0.25s\"\n}\n";
    manifest.flush();
    require(bool(manifest), "manifest write failed " + prefix + ".json");
    std::cout << "dynamic_bank_manifest " << prefix << ".json\n";
    if (!complete) {
        throw std::runtime_error(
            "dynamic bank generation exhausted: accepted " +
            std::to_string(accepted.size()) + " of " + std::to_string(target) +
            " after " + std::to_string(draw_index) +
            " draws; records/manifest preserved at " + prefix + ".csv/.json");
    }
    return 0;
}
// --------------------------------------------------------------- evaluation
struct EvalScore {
    double success = 0, collision = 0, timeout = 0;
    double mean_time_s = 0, mean_path_m = 0, min_clearance_m = 12;
    double threat_contacts = 0, static_contacts = 0;
    double min_ttc_sum = 0, min_ttc_count = 0;
    uint32_t rows = 0;
};

// One bank pass. batched=false syncs every tick and adds threat metrics
// (contact source, in-flight measured min TTC, contact time) — used for the
// rich dynamic development/retention records. batched=false is mandatory when
// threat metrics are required and is otherwise the published eval protocol.
static EvalScore evaluate_dynamic(Metal& metal, const float* actor,
                                  const std::vector<DynamicEntry>* dyn,
                                  const std::vector<StaticEntry>* stat,
                                  const DynBankControlHost& control,
                                  uint32_t envs, uint32_t mode, bool ablate,
                                  bool per_tick, const std::string& csv_path,
                                  const std::string& split, uint32_t seed,
                                  uint32_t max_steps = 400) {
    SimConfig config;
    config.n = envs;
    config.mode = mode;
    config.eval = 1;
    config.seed = seed;
    config.speed = kTaskSpeedCapMps;
    config.distance = 4;
    config.max_steps = max_steps;
    config.geometry_memory = 1;
    config.entropy_coef = 0.001f;
    config.learning_rate = 0.0001f;
    Sim sim(metal, config, 32);
    if (actor) std::memcpy(sim.actor.contents, actor, fixed_ppo::actor_param_count * 4);
    enable_task_control(sim, dynamic_task_recipe());
    auto run = make_dynamic_run(sim, dyn ? *dyn : std::vector<DynamicEntry>{},
                                stat ? *stat : std::vector<StaticEntry>{},
                                control, metal.pipeline("dynamic_task_apply"),
                                metal.pipeline("dynamic_script_act"), ablate);
    auto probe = [metal.queue commandBuffer];
    dynamic_probe(run, probe);
    metal.finish(probe);

    std::ofstream file;
    if (!csv_path.empty()) {
        const std::filesystem::path destination(csv_path);
        if (destination.has_parent_path())
            std::filesystem::create_directories(destination.parent_path());
        file.open(csv_path);
        require(bool(file), "cannot write eval CSV " + csv_path);
        file << std::setprecision(9);
        file << "split,env,mode,ablate,family,threat_kind,threat_speed_mps,"
                "nominal_ttc_s,start_stratum,draw_index,route_length_m,"
                "labels,warning_s,success,collision,contact_source,timeout,"
                "time_s,path_m,min_clearance_m,min_ttc_flight,collision_time_s,"
                "final_distance_m\n";
    }

    EvalScore score;
    std::vector<bool> seen(envs, false);
    std::vector<float> prev_range(envs, -1.0f);
    std::vector<float> min_ttc(envs, 1e9f);
    std::vector<int> contact_source(envs, -1);   // 1 threat, 0 static, -1 none
    std::vector<float> contact_time(envs, -1.0f);
    auto write_row = [&](uint32_t e) {
        const auto* runs = (const SimRun*)sim.runs.contents;
        const auto* states = (const RLPhysicsState*)sim.states.contents;
        const DynamicEntry* entry = dyn ? &((*dyn)[std::min(e, uint32_t(dyn->size() - 1))])
                                        : nullptr;
        const StaticEntry* sentry = stat ? &((*stat)[std::min(e, uint32_t(stat->size() - 1))])
                                         : nullptr;
        const SimRun& r = runs[e];
        float final_distance = 0;
        const float* goal = entry ? entry->goal_position : sentry->goal_position;
        for (uint32_t axis = 0; axis < 3; axis++) {
            const float d = goal[axis] - states[e].position[axis];
            final_distance += d * d;
        }
        final_distance = std::sqrt(final_distance);
        if (!file.is_open()) return;
        file << split << ',' << e << ',' << mode << ',' << (ablate ? 1 : 0) << ',';
        if (entry) {
            file << entry->family << ',' << entry->threat_kind << ','
                 << entry->threat_speed_mps << ',' << entry->nominal_ttc_s << ','
                 << entry->start_stratum << ',' << entry->draw_index << ','
                 << entry->route_length << ',' << entry->labels << ','
                 << entry->warning_s;
        } else {
            file << sentry->family << ",-1,-1,-1,-1,-1," << sentry->initial_distance
                 << ",-1,-1";
        }
        file << ',' << r.successes << ',' << r.collisions << ','
             << (r.collisions ? (contact_source[e] == 1 ? "threat"
                                 : contact_source[e] == 0 ? "static" : "unknown")
                              : "none")
             << ',' << r.timeouts << ',' << r.elapsed << ',' << r.path << ','
             << r.min_clearance << ','
             << (min_ttc[e] < 1e9f ? min_ttc[e] : -1.0f) << ','
             << contact_time[e] << ',' << final_distance << '\n';
    };

    if (!per_tick) {
        auto cb = [metal.queue commandBuffer];
        for (uint32_t t = 0; t < max_steps; t++) dynamic_tick(run, cb, t, ablate, false, t == 0);
        metal.finish(cb);
        const auto* runs = (const SimRun*)sim.runs.contents;
        for (uint32_t e = 0; e < envs; e++) {
            require(runs[e].episodes == 1,
                    "eval must complete exactly one episode per environment");
            score.success += runs[e].successes;
            score.collision += runs[e].collisions;
            score.timeout += runs[e].timeouts;
            score.mean_time_s += runs[e].successes ? runs[e].elapsed : 0;
            score.mean_path_m += runs[e].path;
            score.min_clearance_m = std::min<double>(score.min_clearance_m,
                                                     runs[e].min_clearance);
            write_row(e);
        }
        score.rows = envs;
        if (score.success > 0) score.mean_time_s /= score.success;
        score.mean_path_m /= envs;
        if (file.is_open()) file.flush();
        return score;
    }

    for (uint32_t t = 0; t < max_steps; t++) {
        auto cb = [metal.queue commandBuffer];
        dynamic_tick(run, cb, t, ablate, false, t == 0);
        metal.finish(cb);
        const auto* runs = (const SimRun*)sim.runs.contents;
        const auto* states = (const RLPhysicsState*)sim.states.contents;
        bool all_done = true;
        for (uint32_t e = 0; e < envs; e++) {
            if (runs[e].episodes > 0) {
                if (!seen[e]) {
                    seen[e] = true;
                    if (runs[e].collisions && dyn) {
                        const DynamicEntry& entry =
                            (*dyn)[std::min(e, uint32_t(dyn->size() - 1))];
                        const uint32_t threat = entry.world.count - 1;
                        float tp[3];
                        threat_position(entry.world, threat, runs[e].elapsed, tp);
                        const float pos[3] = {states[e].position[0],
                                              states[e].position[1],
                                              states[e].position[2]};
                        contact_source[e] =
                            distance3(pos, tp) <= kCombinedRadius + 0.02f ? 1 : 0;
                        contact_time[e] = runs[e].elapsed;
                    }
                    write_row(e);
                    score.success += runs[e].successes;
                    score.collision += runs[e].collisions;
                    score.timeout += runs[e].timeouts;
                    score.mean_time_s += runs[e].successes ? runs[e].elapsed : 0;
                    score.mean_path_m += runs[e].path;
                    score.min_clearance_m = std::min<double>(score.min_clearance_m,
                                                             runs[e].min_clearance);
                    if (runs[e].collisions) {
                        if (contact_source[e] == 1) score.threat_contacts++;
                        else score.static_contacts++;
                    }
                    if (min_ttc[e] < 1e9f) {
                        score.min_ttc_sum += min_ttc[e];
                        score.min_ttc_count++;
                    }
                    score.rows++;
                }
                continue;
            }
            all_done = false;
            if (dyn) {
                const DynamicEntry& entry = (*dyn)[std::min(e, uint32_t(dyn->size() - 1))];
                const uint32_t threat = entry.world.count - 1;
                float tp[3];
                threat_position(entry.world, threat, runs[e].elapsed, tp);
                const float pos[3] = {states[e].position[0], states[e].position[1],
                                      states[e].position[2]};
                const float range = distance3(pos, tp);
                if (prev_range[e] >= 0.0f) {
                    const float closing = (prev_range[e] - range) / 0.05f;
                    if (closing > 1e-3f)
                        min_ttc[e] = std::min(min_ttc[e], range / closing);
                }
                prev_range[e] = range;
            }
        }
        if (all_done) break;
    }
    if (score.success > 0) score.mean_time_s /= score.success;
    score.mean_path_m /= std::max<uint32_t>(score.rows, 1);
    if (file.is_open()) file.flush();
    return score;
}

static void print_eval(const std::string& label, const EvalScore& s) {
    std::cout << "dynamic_eval " << label << " rows=" << s.rows
              << " success=" << s.success << " collision=" << s.collision
              << " timeout=" << s.timeout
              << " mean_arrival_s=" << s.mean_time_s
              << " mean_path_m=" << s.mean_path_m
              << " min_clearance_m=" << s.min_clearance_m
              << " threat_contacts=" << s.threat_contacts
              << " static_contacts=" << s.static_contacts
              << " mean_min_ttc_s="
              << (s.min_ttc_count ? s.min_ttc_sum / s.min_ttc_count : -1.0) << "\n";
}

static int command_eval(int argc, char** argv) {
    require(argc >= 4, "dynamic-eval CHECKPOINT OUT_CSV");
    const std::string checkpoint = argv[2], output = argv[3];
    std::string dyn_path, static_path, split = "dev";
    uint32_t mode = 17, seed = 700001, envs = 128;
    int ablate = -1, per_tick = -1;
    for (int index = 4; index < argc; index++) {
        const std::string option = argv[index];
        if (option == "--dyn-bank") dyn_path = dyn_option_value(argc, argv, index, option);
        else if (option == "--static-bank") static_path = dyn_option_value(argc, argv, index, option);
        else if (option == "--split") split = dyn_option_value(argc, argv, index, option);
        else if (option == "--mode") mode = dyn_option_uint(argc, argv, index, option);
        else if (option == "--seed") seed = dyn_option_uint(argc, argv, index, option);
        else if (option == "--envs") envs = dyn_option_uint(argc, argv, index, option);
        else if (option == "--ablate") ablate = int(dyn_option_uint(argc, argv, index, option));
        else if (option == "--per-tick") per_tick = int(dyn_option_uint(argc, argv, index, option));
        else throw std::runtime_error("unknown dynamic-eval option " + option);
    }
    require(!dyn_path.empty() || !static_path.empty(),
            "dynamic-eval needs --dyn-bank or --static-bank");
    if (ablate < 0) ablate = 0;
    if (per_tick < 0) per_tick = dyn_path.empty() ? 0 : 1;

    std::ifstream file(checkpoint, std::ios::binary);
    const PpoCheckpointHeader header = read_checkpoint_header(file);
    require(header.actor_count == fixed_ppo::actor_param_count,
            "eval checkpoint dimensions");
    std::vector<float> actor(header.actor_count);
    file.read(reinterpret_cast<char*>(actor.data()),
              std::streamsize(actor.size() * sizeof(float)));
    require(bool(file), "evaluation actor read failed");

    DynBankControlHost control{0, 0, 0, 0, 1, 0};
    std::vector<DynamicEntry> dyn;
    std::vector<StaticEntry> stat;
    if (!dyn_path.empty()) {
        dyn = read_dynamic_bank(dyn_path, control);
        require(dyn.size() == envs && control.period == 1,
                "dynamic eval bank must be period-1 with --envs entries");
        control = DynBankControlHost{uint32_t(dyn.size()), 0, 1, 0, 1, 0};
    } else {
        uint32_t period = 1;
        stat = read_static_bank(static_path, period);
        require(stat.size() == envs * period,
                "static eval bank does not match --envs");
        control = DynBankControlHost{0, uint32_t(stat.size()), 0, period, period, 0};
        // static-only: slot always >= dyn_slots(0) -> static index n*period_slot
        control.dyn_slots = 0;
        control.static_slots = period;
        control.period = 1;
        control.static_count = uint32_t(stat.size());
    }
    Metal metal;
    metal.compile(base_source() + PPO_TRAINER_MSL + kDynamicMsl);
    const EvalScore score = evaluate_dynamic(
        metal, actor.data(), dyn.empty() ? nullptr : &dyn,
        stat.empty() ? nullptr : &stat, control, envs, mode, ablate != 0,
        per_tick != 0, output, split, seed);
    print_eval(split + " mode=" + std::to_string(mode) +
                   " ablate=" + std::to_string(ablate),
               score);
    return 0;
}
// ---------------------------------------------------------------- train cmd
struct DynTrainContract {
    std::string sensor_profile;
    uint32_t actor_obs_dim = 0;
    std::string runner_sha256, core_sha256, snapshot_sha256, warmstart_sha256;
    std::string dyn_bank_sha256, dyn_records_sha256, static_bank_sha256;
    std::string eval_dyn_sha256, eval_static_sha256;
    uint32_t seed = 0, ablate = 0, environments = 0, horizon = 0, epochs = 0;
    uint32_t eval_every = 0, max_steps = 0;
    float learning_rate = 0, entropy_coef = 0, time_cost = 0, warmstart_log_std = 0;
};

// __FILE__ is the absolute compile path of this module; SOURCE_DIR is the
// pinned snapshot (SNAPSHOT.json hashed alongside).
static std::string dyn_runner_sha256() {
    return challenge_evaluation::sha256_file(std::string(__FILE__));
}
static std::string dyn_core_sha256() {
    const std::string source = base_source() + PPO_TRAINER_MSL + kDynamicMsl;
    return ppo_safeguard_sha256(source.data(), source.size());
}

static void write_dyn_contract(const std::string& path, const DynTrainContract& c) {
    std::ofstream out(path);
    require(bool(out), "cannot write training sidecar " + path);
    out << std::setprecision(9) << "{\n"
        << "  \"schema\": \"local-dynamic-train-v1\",\n"
        << "  \"sensor_profile\": \"" << c.sensor_profile << "\",\n"
        << "  \"actor_obs_dim\": " << c.actor_obs_dim << ",\n"
        << "  \"runner_sha256\": \"" << c.runner_sha256 << "\",\n"
        << "  \"core_sha256\": \"" << c.core_sha256 << "\",\n"
        << "  \"snapshot_sha256\": \"" << c.snapshot_sha256 << "\",\n"
        << "  \"warmstart_sha256\": \"" << c.warmstart_sha256 << "\",\n"
        << "  \"dyn_bank_sha256\": \"" << c.dyn_bank_sha256 << "\",\n"
        << "  \"dyn_records_sha256\": \"" << c.dyn_records_sha256 << "\",\n"
        << "  \"static_bank_sha256\": \"" << c.static_bank_sha256 << "\",\n"
        << "  \"eval_dyn_sha256\": \"" << c.eval_dyn_sha256 << "\",\n"
        << "  \"eval_static_sha256\": \"" << c.eval_static_sha256 << "\",\n"
        << "  \"seed\": " << c.seed << ",\n"
        << "  \"ablate\": " << c.ablate << ",\n"
        << "  \"environments\": " << c.environments << ",\n"
        << "  \"horizon\": " << c.horizon << ",\n"
        << "  \"epochs\": " << c.epochs << ",\n"
        << "  \"eval_every\": " << c.eval_every << ",\n"
        << "  \"max_steps\": " << c.max_steps << ",\n"
        << "  \"learning_rate\": " << c.learning_rate << ",\n"
        << "  \"entropy_coef\": " << c.entropy_coef << ",\n"
        << "  \"time_cost_per_s\": " << c.time_cost << ",\n"
        << "  \"warmstart_log_std\": " << c.warmstart_log_std << ",\n"
        << "  \"optimizer_state\": \"reset_on_warmstart\",\n"
        << "  \"critic_state\": \"warmstarted_from_checkpoint\"\n}\n";
    out.flush();
    require(bool(out), "training sidecar write failed: " + path);
}

// Strict flat-sidecar readers (the waypoint runner's versions are not part of
// the snapshot; these are byte-compatible in behaviour: required key, exact
// token kind, hard failure otherwise).
static std::string dyn_json_token(const std::string& text, const std::string& key) {
    const std::string needle = "\"" + key + "\"";
    size_t position = text.find(needle);
    require(position != std::string::npos,
            "resume sidecar is missing required key \"" + key + "\"");
    position += needle.size();
    while (position < text.size() &&
           std::isspace(static_cast<unsigned char>(text[position])))
        position++;
    require(position < text.size() && text[position] == ':',
            "resume sidecar key \"" + key + "\" has no value");
    position++;
    while (position < text.size() &&
           std::isspace(static_cast<unsigned char>(text[position])))
        position++;
    require(position < text.size(), "resume sidecar key \"" + key + "\" has an empty value");
    std::string token;
    if (text[position] == '"') {
        const size_t end = text.find('"', position + 1);
        require(end != std::string::npos,
                "resume sidecar key \"" + key + "\" has an unterminated string");
        token = text.substr(position + 1, end - position - 1);
    } else {
        while (position < text.size() && text[position] != ',' &&
               text[position] != '\n' && text[position] != '}')
            token += text[position++];
        while (!token.empty() && std::isspace(static_cast<unsigned char>(token.back())))
            token.pop_back();
    }
    require(!token.empty(), "resume sidecar key \"" + key + "\" has an empty value");
    return token;
}
static void dyn_require_json_string(const std::string& text, const std::string& key,
                                    const std::string& expected) {
    const std::string actual = dyn_json_token(text, key);
    require(actual == expected,
            "resume sidecar key \"" + key + "\" is \"" + actual + "\" but this run uses \"" +
                expected + "\"");
}
static void dyn_require_json_uint(const std::string& text, const std::string& key,
                                  uint32_t expected) {
    const uint32_t actual = uint32_t(std::stoul(dyn_json_token(text, key)));
    require(actual == expected,
            "resume sidecar key \"" + key + "\" is " + std::to_string(actual) +
                " but this run uses " + std::to_string(expected));
}
static void dyn_require_json_float(const std::string& text, const std::string& key,
                                   float expected) {
    const float actual = std::stof(dyn_json_token(text, key));
    require(actual == expected,
            "resume sidecar key \"" + key + "\" is " + std::to_string(actual) +
                " but this run uses " + std::to_string(expected));
}

static void require_dyn_contract(const std::string& path, const DynTrainContract& e) {
    require(std::filesystem::exists(path),
            "checkpoint has no training sidecar " + path +
            "; refusing to resume without a verifiable contract");
    const std::string text = read_text(path);
    require(dyn_json_token(text, "schema") == "local-dynamic-train-v1",
            "resume sidecar schema is not local-dynamic-train-v1");
    dyn_require_json_string(text, "sensor_profile", e.sensor_profile);
    dyn_require_json_uint(text, "actor_obs_dim", e.actor_obs_dim);
    dyn_require_json_string(text, "runner_sha256", e.runner_sha256);
    dyn_require_json_string(text, "core_sha256", e.core_sha256);
    dyn_require_json_string(text, "snapshot_sha256", e.snapshot_sha256);
    dyn_require_json_string(text, "warmstart_sha256", e.warmstart_sha256);
    dyn_require_json_string(text, "dyn_bank_sha256", e.dyn_bank_sha256);
    dyn_require_json_string(text, "dyn_records_sha256", e.dyn_records_sha256);
    dyn_require_json_string(text, "static_bank_sha256", e.static_bank_sha256);
    dyn_require_json_string(text, "eval_dyn_sha256", e.eval_dyn_sha256);
    dyn_require_json_string(text, "eval_static_sha256", e.eval_static_sha256);
    dyn_require_json_uint(text, "seed", e.seed);
    dyn_require_json_uint(text, "ablate", e.ablate);
    dyn_require_json_uint(text, "environments", e.environments);
    dyn_require_json_uint(text, "horizon", e.horizon);
    dyn_require_json_uint(text, "epochs", e.epochs);
    dyn_require_json_uint(text, "eval_every", e.eval_every);
    dyn_require_json_uint(text, "max_steps", e.max_steps);
    dyn_require_json_float(text, "learning_rate", e.learning_rate);
    dyn_require_json_float(text, "entropy_coef", e.entropy_coef);
    dyn_require_json_float(text, "time_cost_per_s", e.time_cost);
    dyn_require_json_float(text, "warmstart_log_std", e.warmstart_log_std);
    dyn_require_json_string(text, "optimizer_state", "reset_on_warmstart");
    dyn_require_json_string(text, "critic_state", "warmstarted_from_checkpoint");
    std::cout << "resume_contract PASS sidecar=" << path
              << " ablate=" << e.ablate << " seed=" << e.seed << "\n";
}

static void mixture_exposure(const Sim& sim, uint32_t envs, uint64_t& dyn_eps,
                             uint64_t& static_eps) {
    const auto* runs = (const SimRun*)sim.runs.contents;
    dyn_eps = 0;
    static_eps = 0;
    for (uint32_t e = 0; e < envs; e++) {
        const uint64_t episodes = runs[e].episodes;
        const uint64_t cycles = episodes / kMixturePeriod;
        const uint32_t rem = uint32_t(episodes % kMixturePeriod);
        dyn_eps += cycles * kDynamicSlots + std::min<uint32_t>(rem, kDynamicSlots);
        static_eps += cycles * kStaticSlots +
            (rem > kDynamicSlots ? std::min<uint32_t>(rem - kDynamicSlots, kStaticSlots) : 0);
    }
}

static void verify_dynamic_reset(const DynamicRun& run,
                                 const std::vector<DynamicEntry>& dyn,
                                 const DynBankControlHost& control) {
    const auto* states = (const RLPhysicsState*)run.sim.states.contents;
    const auto* runs = (const SimRun*)run.sim.runs.contents;
    const auto* worlds = (const WWorld*)run.sim.worlds.contents;
    const auto* tasks = (const NavigationTaskState*)run.sim.task_states.contents;
    float worst_position = 0, worst_yaw = 0, worst_reference = 0, worst_goal = 0,
          worst_distance = 0;
    const uint32_t check = std::min<uint32_t>(control.dyn_slots, 4);
    for (uint32_t e = 0; e < check; e++) {
        const DynamicEntry& entry = dyn[e * control.dyn_slots];  // slot 0 entry
        for (uint32_t axis = 0; axis < 3; axis++) {
            worst_position = std::max(worst_position,
                std::fabs(states[e].position[axis] - entry.start_position[axis]));
            worst_reference = std::max(worst_reference,
                std::fabs(runs[e].reference_position[axis] - entry.start_position[axis]));
            worst_goal = std::max(worst_goal,
                std::fabs(worlds[e].goal[axis] - entry.goal_position[axis]));
        }
        worst_yaw = std::max(worst_yaw, std::fabs(runs[e].yaw - entry.start_yaw));
        worst_distance = std::max(worst_distance,
            std::fabs(tasks[e].initial_distance_m - entry.route_length));
        require(tasks[e].valid == 1 &&
                    tasks[e].generation_status == NAV_TASK_GENERATION_READY &&
                    tasks[e].objective == NAV_TASK_OBJECTIVE_FINAL_HOLD &&
                    tasks[e].max_nav_steps == 400,
                "dynamic bank task record not installed for env " + std::to_string(e));
        require(runs[e].steps == 0 && runs[e].elapsed == 0.0f &&
                    runs[e].min_clearance == 12.0f,
                "dynamic install did not start a clean episode (clock must be 0)");
    }
    require(worst_position < 1e-6f && worst_reference < 1e-6f &&
                worst_goal < 1e-6f && worst_yaw < 1e-6f && worst_distance < 1e-5f,
            "dynamic reset pose/goal/yaw/distance invariant failed");
    std::cout << "dyn_reset_invariants PASS position=" << worst_position
              << " reference=" << worst_reference << " goal=" << worst_goal
              << " yaw=" << worst_yaw << " distance=" << worst_distance
              << " clock_zero=1 envs=" << check << "\n";
}

static int command_train(int argc, char** argv) {
    require(argc >= 5, "dynamic-train ROLLOUTS CHECKPOINT WARMSTART");
    const uint32_t rollouts = uint32_t(std::stoul(argv[2]));
    const std::string checkpoint = argv[3], warmstart = argv[4];
    std::string dyn_path, dyn_records_path, static_path, eval_dyn_path,
                eval_static_path;
    uint32_t seed = 20261012, environments = 128, horizon = 32, epochs = 2,
             eval_every = 250;
    int ablate = -1;
    float learning_rate = 0.0001f, entropy = 0.001f, time_cost = 0.2f,
          log_std = -1.0f;
    for (int index = 5; index < argc; index++) {
        const std::string option = argv[index];
        if (option == "--dyn-bank") dyn_path = dyn_option_value(argc, argv, index, option);
        else if (option == "--dyn-records") dyn_records_path = dyn_option_value(argc, argv, index, option);
        else if (option == "--static-bank") static_path = dyn_option_value(argc, argv, index, option);
        else if (option == "--eval-dyn") eval_dyn_path = dyn_option_value(argc, argv, index, option);
        else if (option == "--eval-static") eval_static_path = dyn_option_value(argc, argv, index, option);
        else if (option == "--seed") seed = dyn_option_uint(argc, argv, index, option);
        else if (option == "--ablate") ablate = int(dyn_option_uint(argc, argv, index, option));
        else if (option == "--epochs") epochs = dyn_option_uint(argc, argv, index, option);
        else if (option == "--envs") environments = dyn_option_uint(argc, argv, index, option);
        else if (option == "--horizon") horizon = dyn_option_uint(argc, argv, index, option);
        else if (option == "--eval-every") eval_every = dyn_option_uint(argc, argv, index, option);
        else if (option == "--learning-rate") learning_rate = dyn_option_float(argc, argv, index, option);
        else if (option == "--entropy") entropy = dyn_option_float(argc, argv, index, option);
        else if (option == "--time-cost") time_cost = dyn_option_float(argc, argv, index, option);
        else if (option == "--logstd") log_std = dyn_option_float(argc, argv, index, option);
        else throw std::runtime_error("unknown dynamic-train option " + option);
    }
    require(rollouts > 0 && fixed_ppo::actor_obs_dim == 184,
            "dynamic training needs positive rollouts and the 184D guided actor");
    require(ablate == 0 || ablate == 1, "dynamic-train needs --ablate 0|1");
    require(!dyn_path.empty() && !static_path.empty() && !eval_dyn_path.empty() &&
                !eval_static_path.empty(),
            "dynamic-train needs --dyn-bank --static-bank --eval-dyn --eval-static");
    require(environments == 128, "this experiment fixes 128 environments");

    // Banks first: nothing is written until a resume contract validates.
    DynBankControlHost dyn_control{};
    auto dyn = read_dynamic_bank(dyn_path, dyn_control);
    require(dyn_control.period == kMixturePeriod && dyn.size() == 4 * environments,
            "training dynamic bank must be period-12 with 4 slots per env");
    uint32_t static_period = 1;
    auto stat = read_static_bank(static_path, static_period);
    require(static_period == kStaticSlots && stat.size() == static_period * environments,
            "training static bank must be the published period-8 source bank");
    DynBankControlHost eval_control{};
    auto eval_dyn = read_dynamic_bank(eval_dyn_path, eval_control);
    require(eval_control.period == 1 && eval_dyn.size() == environments,
            "eval dynamic bank must be period-1 with one slot per env");
    uint32_t eval_static_period = 1;
    auto eval_static = read_static_bank(eval_static_path, eval_static_period);
    require(eval_static.size() == environments * eval_static_period,
            "eval static bank does not match --envs");
    const DynBankControlHost train_control{
        uint32_t(dyn.size()), uint32_t(stat.size()), kDynamicSlots, kStaticSlots,
        kMixturePeriod, 0};
    const DynBankControlHost eval_dyn_control{
        uint32_t(eval_dyn.size()), 0, 1, 0, 1, 0};
    const DynBankControlHost eval_static_control{
        0, uint32_t(eval_static.size()), 0, eval_static_period, 1, 0};

    DynTrainContract contract;
    contract.sensor_profile = NAV_SENSOR_ACTIVE_NAME;
    contract.actor_obs_dim = uint32_t(fixed_ppo::actor_obs_dim);
    contract.runner_sha256 = dyn_runner_sha256();
    contract.core_sha256 = dyn_core_sha256();
    contract.snapshot_sha256 = challenge_evaluation::sha256_file(
        std::string(SOURCE_DIR) + "/results/root-joint-critic/SNAPSHOT.json");
    contract.warmstart_sha256 = challenge_evaluation::sha256_file(warmstart);
    contract.dyn_bank_sha256 = challenge_evaluation::sha256_file(dyn_path);
    contract.dyn_records_sha256 = dyn_records_path.empty()
        ? "none" : challenge_evaluation::sha256_file(dyn_records_path);
    contract.static_bank_sha256 = challenge_evaluation::sha256_file(static_path);
    contract.eval_dyn_sha256 = challenge_evaluation::sha256_file(eval_dyn_path);
    contract.eval_static_sha256 = challenge_evaluation::sha256_file(eval_static_path);
    contract.seed = seed;
    contract.ablate = uint32_t(ablate);
    contract.environments = environments;
    contract.horizon = horizon;
    contract.epochs = epochs;
    contract.eval_every = eval_every;
    contract.max_steps = 400;
    contract.learning_rate = learning_rate;
    contract.entropy_coef = entropy;
    contract.time_cost = time_cost;
    contract.warmstart_log_std = log_std;

    const bool resume = std::filesystem::exists(checkpoint);
    if (resume) {
        require_dyn_contract(checkpoint + ".train.json", contract);
    } else {
        write_dyn_contract(checkpoint + ".train.json", contract);
        std::cout << "train_contract " << checkpoint << ".train.json"
                  << " runner=" << contract.runner_sha256.substr(0, 12)
                  << " core=" << contract.core_sha256.substr(0, 12)
                  << " snapshot=" << contract.snapshot_sha256.substr(0, 12)
                  << " dyn=" << contract.dyn_bank_sha256.substr(0, 12)
                  << " ablate=" << ablate << " seed=" << seed << "\n";
    }

    Metal metal;
    metal.compile(base_source() + PPO_TRAINER_MSL + kDynamicMsl);
    auto apply_p = metal.pipeline("dynamic_task_apply");
    auto script_p = metal.pipeline("dynamic_script_act");

    SimConfig config;
    config.n = environments;
    config.mode = 22;
    config.family = 0;
    config.seed = seed;
    config.speed = kTaskSpeedCapMps;
    config.distance = 4;
    config.max_steps = 400;
    config.geometry_memory = 1;
    config.entropy_coef = entropy;
    config.learning_rate = learning_rate;
    Sim sim(metal, config, horizon);
    auto run = make_dynamic_run(sim, dyn, stat, train_control, apply_p, script_p,
                                ablate != 0);
    enable_task_control(sim, dynamic_task_recipe(time_cost));
    PPOTrainer trainer(sim, epochs);
    if (resume) {
        trainer.load_checkpoint(checkpoint, config.family, horizon, environments, seed);
    } else {
        navigation_training::load_actor(sim, warmstart, true);
        for (uint32_t axis = 0; axis < 4; axis++)
            ((float*)sim.actor.contents)[fixed_ppo::actor_log_std_offset + axis] = log_std;
        trainer.recapture_anchor_reference();
    }
    if (resume) {
        std::cout << "resume_skips_reset_probe mid_episode_state_restored rollouts="
                  << trainer.completed_rollouts << "\n";
    } else {
        auto probe = [metal.queue commandBuffer];
        dynamic_probe(run, probe);
        metal.finish(probe);
        verify_dynamic_reset(run, dyn, train_control);
    }

    const std::filesystem::path checkpoint_path(checkpoint);
    if (checkpoint_path.has_parent_path())
        std::filesystem::create_directories(checkpoint_path.parent_path());
    std::ofstream history(checkpoint + ".history.csv",
                          resume ? std::ios::app : std::ios::trunc);
    require(bool(history), "cannot write training history");
    std::ofstream diag(checkpoint + ".diag.csv",
                       resume ? std::ios::app : std::ios::trunc);
    require(bool(diag), "cannot write training diagnostics");
    if (!resume) {
        history << "rollout,transitions,wall_s,collect_gpu_s,update_gpu_s,"
                   "policy_loss,value_loss,entropy,ratio,dyn_success,"
                   "dyn_collision,dyn_timeout,dyn_arrival,static_success,"
                   "static_collision,dyn_episodes,static_episodes\n";
        diag << "rollout,transitions,gpu_s,rew_mean,rew_std,term_n,term_mean,"
                "term_pos,adv_std,ret_mean,val_mean,pol_loss,val_loss,entropy,"
                "ratio,episodes,ep_success,ep_collision,ep_timeout,ep_path_m,"
                "ep_time_s,ep_speed_mps,inflight_clear_min,logstd0,logstd1,"
                "logstd2,logstd3,actor_drift_l2\n";
    }
    // Selection: dynamic-dev success -> fewer contacts -> shorter arrival.
    EvalScore best;
    best.success = -1;
    if (resume && std::filesystem::exists(checkpoint + ".best")) {
        std::ifstream file(checkpoint + ".best", std::ios::binary);
        const PpoCheckpointHeader header = read_checkpoint_header(file);
        std::vector<float> actor(header.actor_count);
        file.read(reinterpret_cast<char*>(actor.data()),
                  std::streamsize(actor.size() * sizeof(float)));
        require(bool(file), "best checkpoint actor read failed");
        best = evaluate_dynamic(metal, actor.data(), &eval_dyn, nullptr,
                                eval_dyn_control, environments, 17, ablate != 0,
                                false, "", "dev-dyn-best", seed);
    }

    std::vector<SimRun> previous_runs(environments);
    const double started = seconds();
    const uint32_t finish = trainer.completed_rollouts + rollouts;
    std::cout << "dynamic_train dyn_slots=" << kDynamicSlots
              << " static_slots=" << kStaticSlots
              << " environments=" << environments << " horizon=" << horizon
              << " epochs=" << epochs << " lr=" << learning_rate
              << " entropy=" << entropy << " time_cost=" << time_cost
              << " ablate=" << ablate << " seed=" << seed
              << " start=" << trainer.completed_rollouts << " finish=" << finish
              << " sensor_profile=" << NAV_SENSOR_ACTIVE_NAME << "\n";
    for (uint32_t rollout = trainer.completed_rollouts; rollout < finish; rollout++) {
        @autoreleasepool {
            std::memcpy(previous_runs.data(), sim.runs.contents,
                        environments * sizeof(SimRun));
            auto commands = [metal.queue commandBuffer];
            dynamic_collect(run, commands, horizon, ablate != 0);
            const double collect_gpu = metal.finish(commands);
            auto update = [metal.queue commandBuffer];
            trainer.rollout_update(update, rollout);
            const double update_gpu = metal.finish(update);
            trainer.completed_rollouts = rollout + 1;

            const size_t rows = size_t(environments) * horizon;
            const float* reward = (const float*)sim.rewards.contents;
            const float* advantage = (const float*)sim.advantages.contents;
            const float* returns = (const float*)sim.returns.contents;
            const float* values = (const float*)sim.values.contents;
            const auto* terminated = (const uint8_t*)sim.terminated.contents;
            const auto moment = [&](const float* buffer, size_t count) {
                double mean = 0;
                for (size_t index = 0; index < count; index++) {
                    require(std::isfinite(buffer[index]), "non-finite rollout buffer");
                    mean += buffer[index];
                }
                mean /= double(count ? count : 1);
                double var = 0;
                for (size_t index = 0; index < count; index++) {
                    const double d = double(buffer[index]) - mean;
                    var += d * d;
                }
                return std::array<double, 2>{mean, count ? std::sqrt(var / double(count)) : 0.0};
            };
            const auto reward_moment = moment(reward, rows);
            const auto advantage_moment = moment(advantage, rows);
            const auto value_moment = moment(values, rows);
            size_t terminal_count = 0;
            double terminal_sum = 0;
            for (size_t index = 0; index < rows; index++)
                if (terminated[index]) {
                    terminal_count++;
                    terminal_sum += reward[index];
                }
            const auto* runs_now = (const SimRun*)sim.runs.contents;
            uint64_t episodes = 0, successes = 0, collisions = 0, timeouts = 0;
            double path = 0, elapsed = 0;
            float clearance_min = 12;
            for (uint32_t e = 0; e < environments; e++) {
                const SimRun& now = runs_now[e];
                const SimRun& before = previous_runs[e];
                require(now.episodes >= before.episodes &&
                            now.successes >= before.successes &&
                            now.collisions >= before.collisions &&
                            now.timeouts >= before.timeouts,
                        "episode counters went backwards");
                episodes += now.episodes - before.episodes;
                successes += now.successes - before.successes;
                collisions += now.collisions - before.collisions;
                timeouts += now.timeouts - before.timeouts;
                path += double(now.total_path) - before.total_path;
                elapsed += double(now.total_elapsed) - before.total_elapsed;
                clearance_min = std::min(clearance_min, now.min_clearance);
            }
            const float* metric = (const float*)trainer.metric_mean.contents;
            const float* actor_params = (const float*)sim.actor.contents;
            diag << (rollout + 1) << ',' << uint64_t(rollout + 1) * rows << ','
                 << (collect_gpu + update_gpu) << ',' << reward_moment[0] << ','
                 << reward_moment[1] << ',' << terminal_count << ','
                 << (terminal_count ? terminal_sum / double(terminal_count) : 0.0)
                 << ',' << advantage_moment[1] << ',' << moment(returns, rows)[0]
                 << ',' << value_moment[0] << ',' << metric[0] << ',' << metric[1]
                 << ',' << metric[2] << ',' << metric[3] << ',' << episodes << ','
                 << (episodes ? double(successes) / episodes : 0) << ','
                 << (episodes ? double(collisions) / episodes : 0) << ','
                 << (episodes ? double(timeouts) / episodes : 0) << ','
                 << (episodes ? path / double(episodes) : 0) << ','
                 << (episodes ? elapsed / double(episodes) : 0) << ','
                 << (elapsed ? path / elapsed : 0) << ',' << clearance_min;
            for (uint32_t axis = 0; axis < 4; axis++)
                diag << ',' << actor_params[fixed_ppo::actor_log_std_offset + axis];
            diag << ',' << trainer.actor_drift_l2() << '\n';
            diag.flush();

            if ((rollout + 1) % eval_every == 0 || rollout + 1 == finish) {
                trainer.save_checkpoint(checkpoint, config.family, seed, rollout + 1);
                const EvalScore dyn_score = evaluate_dynamic(
                    metal, (const float*)sim.actor.contents, &eval_dyn, nullptr,
                    eval_dyn_control, environments, 17, ablate != 0, false,
                    checkpoint + ".dev.csv", "dev-dyn", seed);
                const EvalScore static_score = evaluate_dynamic(
                    metal, (const float*)sim.actor.contents, nullptr, &eval_static,
                    eval_static_control, environments, 17, ablate != 0, false,
                    checkpoint + ".static-dev.csv", "dev-a", seed);
                uint64_t dyn_eps = 0, static_eps = 0;
                mixture_exposure(sim, environments, dyn_eps, static_eps);
                history << (rollout + 1) << ',' << uint64_t(rollout + 1) * rows
                        << ',' << seconds() - started << ',' << collect_gpu << ','
                        << update_gpu << ',' << metric[0] << ',' << metric[1] << ','
                        << metric[2] << ',' << metric[3] << ',' << dyn_score.success
                        << ',' << dyn_score.collision << ',' << dyn_score.timeout
                        << ',' << dyn_score.mean_time_s << ',' << static_score.success
                        << ',' << static_score.collision << ',' << dyn_eps << ','
                        << static_eps << '\n';
                history.flush();
                const bool better = dyn_score.success > best.success ||
                    (dyn_score.success == best.success &&
                     dyn_score.collision < best.collision) ||
                    (dyn_score.success == best.success &&
                     dyn_score.collision == best.collision &&
                     (best.success < 0 || dyn_score.mean_time_s < best.mean_time_s));
                if (better) {
                    best = dyn_score;
                    trainer.save_checkpoint(checkpoint + ".best", config.family, seed,
                                            rollout + 1);
                    if (std::filesystem::exists(checkpoint + ".dev.csv"))
                        std::filesystem::copy_file(
                            checkpoint + ".dev.csv", checkpoint + ".best-dev.csv",
                            std::filesystem::copy_options::overwrite_existing);
                }
                std::cout << "dynamic_train rollout=" << (rollout + 1)
                          << " wall_s=" << seconds() - started
                          << " collect_gpu_s=" << collect_gpu
                          << " update_gpu_s=" << update_gpu
                          << " policy_loss=" << metric[0]
                          << " value_loss=" << metric[1]
                          << " dyn_success=" << dyn_score.success
                          << " dyn_collision=" << dyn_score.collision
                          << " static_success=" << static_score.success
                          << " exposure_dyn=" << dyn_eps
                          << " exposure_static=" << static_eps
                          << " clear_min=" << clearance_min << "\n";
            }
        }
    }
    std::cout << "dynamic_train_done rollouts=" << finish
              << " wall_s=" << seconds() - started << "\n";
    return 0;
}
// ---------------------------------------------------------------- test cmd
static int command_test(int argc, char** argv) {
    std::string static_bank_path = argc > 2 ? argv[2] : "";
    // 0) The snapshot runner's own production gates (world/raptor/physics/
    //    PPO/observation/action-map parity). Must pass before anything else.
    {
        char arg0[] = "metal_nav_dynamic";
        char arg1[] = "test";
        char* test_argv[2] = {arg0, arg1};
        const int rc = dynamic_nav_reserved_main(2, test_argv);
        require(rc == 0, "snapshot runner test suite failed");
        std::cout << "snapshot_runner_gates PASS\n";
    }
    std::cout << "dynamic_contract runner_sha256=" << dyn_runner_sha256()
              << " core_sha256=" << dyn_core_sha256() << " snapshot_sha256="
              << challenge_evaluation::sha256_file(std::string(SOURCE_DIR) + "/results/root-joint-critic/SNAPSHOT.json")
              << " entry_bytes=" << sizeof(DynamicEntry)
              << " static_entry_bytes=" << sizeof(StaticEntry)
              << " source_dir=" << SOURCE_DIR << "\n";
    require(sizeof(DynamicEntry) % 4 == 0 && sizeof(StaticEntry) == 756,
            "bank ABI sizes");

    // 1) Frustum grounding predicate (exact mission formula).
    {
        const float s[3] = {0, 0, 1.5f};
        const float ahead[3] = {2, 0, 1.5f};
        const float behind[3] = {-1, 0, 1.5f};
        const float wide[3] = {2, 2.5f, 1.5f};
        const float edge[3] = {2, 1.5f, 1.5f};
        const float tall[3] = {1, 0, 1.5f + 0.8f};
        const float ok_tall[3] = {1, 0, 1.5f + 0.7f};
        const float side[3] = {0, 2, 1.5f};
        require(goal_in_frustum(s, 0.0f, ahead), "straight goal must be grounded");
        require(!goal_in_frustum(s, 0.0f, behind), "behind goal must fail grounding");
        require(!goal_in_frustum(s, 0.0f, wide), "beyond tan_h must fail grounding");
        require(goal_in_frustum(s, 0.0f, edge), "goal at tan_h edge must pass");
        require(!goal_in_frustum(s, 0.0f, tall), "beyond tan_v must fail grounding");
        require(goal_in_frustum(s, 0.0f, ok_tall), "goal at tan_v edge must pass");
        require(!goal_in_frustum(s, 0.0f, side), "lateral goal fails without yaw");
        require(goal_in_frustum(s, 1.5707963f, side), "yawed goal passes after turning");
        std::cout << "frustum_grounding_test PASS tan_h=" << NAV_SENSOR_TAN_H
                  << " tan_v=" << NAV_SENSOR_ACTIVE_TAN_V << "\n";
    }

    // 2) Draw + install ABI probe: GPU-installed fields must match the host
    //    entry byte-for-byte at the observable contract (this catches any
    //    host/MSL DynamicEntry offset mismatch).
    Metal metal;
    metal.compile(base_source() + PPO_TRAINER_MSL + kDynamicMsl);
    HostRng rng(20261099u);
    std::vector<DynamicEntry> entries;
    uint32_t draw = 0;
    while (entries.size() < 8 && draw < 4000) {
        DynamicEntry entry{};
        const uint32_t reason = draw_candidate(rng, draw, true, entry);
        if (reason == REJ_NONE) entries.push_back(entry);
        draw++;
    }
    require(entries.size() == 8, "could not draw 8 host-eligible test entries");
    SimConfig config;
    config.n = 8;
    config.mode = 17;
    config.eval = 1;
    config.seed = 800001;
    config.speed = kTaskSpeedCapMps;
    config.distance = 4;
    config.max_steps = 400;
    config.geometry_memory = 1;
    Sim sim(metal, config, 32);
    enable_task_control(sim, dynamic_task_recipe());
    DynBankControlHost control{8, 0, 1, 0, 1, 0};
    auto run = make_dynamic_run(sim, entries, {}, control,
                                metal.pipeline("dynamic_task_apply"),
                                metal.pipeline("dynamic_script_act"), false);
    auto probe = [metal.queue commandBuffer];
    dynamic_probe(run, probe);
    metal.finish(probe);
    verify_dynamic_reset(run, entries, control);
    {
        const auto* worlds = (const WWorld*)sim.worlds.contents;
        const auto* tasks = (const NavigationTaskState*)sim.task_states.contents;
        const auto* states = (const RLPhysicsState*)sim.states.contents;
        for (uint32_t e = 0; e < 8; e++) {
            require(std::memcmp(&worlds[e], &entries[e].world, sizeof(WWorld)) == 0,
                    "installed world differs from host entry (DynamicEntry ABI?)");
            for (uint32_t axis = 0; axis < 3; axis++)
                require(tasks[e].goal_position[axis] == entries[e].goal_position[axis],
                        "installed goal differs from host entry");
            require(states[e].linear_velocity[0] == entries[e].start_velocity[0],
                    "installed velocity differs from host entry");
        }
        std::cout << "dynamic_abi_probe PASS entries=" << entries.size() << "\n";
    }

    // 3) Episode-clock / threat-timeline restart: run scripted straight in
    //    NON-eval mode; when an episode ends the install must zero elapsed so
    //    the threat trajectory replays from t=0 for the next episode.
    {
        SimConfig c2 = config;
        c2.eval = 0;
        c2.seed = 7;
        Sim sim2(metal, c2, 32);
        enable_task_control(sim2, dynamic_task_recipe());
        auto run2 = make_dynamic_run(sim2, entries, {}, control,
                                     metal.pipeline("dynamic_task_apply"),
                                     metal.pipeline("dynamic_script_act"), false);
        set_script(run2, SCRIPT_STRAIGHT);
        bool observed_restart = false;
        uint32_t episodes_after_restart = 0;
        float clock_after_restart = -1;
        for (uint32_t t = 0; t < 700; t++) {
            auto cb = [metal.queue commandBuffer];
            dynamic_tick(run2, cb, t, false, true, t == 0);
            metal.finish(cb);
            const auto* runs = (const SimRun*)sim2.runs.contents;
            if (runs[0].episodes >= 2 && !observed_restart) {
                observed_restart = true;
                episodes_after_restart = runs[0].episodes;
                clock_after_restart = runs[0].elapsed;
                break;
            }
        }
        require(observed_restart, "scripted straight never completed an episode");
        require(clock_after_restart >= 0.0f && clock_after_restart < 0.5f,
                "episode clock did not restart near zero (threat timeline bug)");
        std::cout << "threat_clock_restart_test PASS episodes="
                  << episodes_after_restart << " clock_s=" << clock_after_restart
                  << "\n";
    }

    // 4) Ablation wire: with identical bank/seed the two observation contracts
    //    must agree on dims 0..79 and the ablated run must duplicate them into
    //    80..159 (mode-18 semantics). Compare the LAST tick's rows — tick 0
    //    has prev==cur by construction (frame 0 has no predecessor).
    {
        constexpr uint32_t kTicks = 5;
        auto run_pair = [&](bool ablate, std::vector<float>& obs_out) {
            SimConfig c3 = config;
            c3.n = 4;
            c3.seed = 11;
            Sim sim3(metal, c3, 32);
            enable_task_control(sim3, dynamic_task_recipe());
            auto r3 = make_dynamic_run(sim3, entries, {},
                                       DynBankControlHost{4, 0, 1, 0, 1, 0},
                                       metal.pipeline("dynamic_task_apply"),
                                       metal.pipeline("dynamic_script_act"), ablate);
            for (uint32_t t = 0; t < kTicks; t++) {
                auto cb = [metal.queue commandBuffer];
                dynamic_tick(r3, cb, t, ablate, false, t == 0);
                metal.finish(cb);
            }
            const float* obs = (const float*)sim3.obs.contents;
            obs_out.assign(obs, obs + size_t(kTicks) * 4 * fixed_ppo::actor_obs_dim);
        };
        std::vector<float> normal, ablated;
        run_pair(false, normal);
        run_pair(true, ablated);
        const uint32_t last_tick = kTicks - 1;
        bool history_exists = false, dup_ok = true, front_ok = true;
        for (uint32_t e = 0; e < 4; e++) {
            const uint32_t base = (last_tick * 4 + e) * fixed_ppo::actor_obs_dim;
            for (uint32_t k = 0; k < 80; k++) {
                if (normal[base + k] != normal[base + 80 + k]) history_exists = true;
                if (ablated[base + 80 + k] != ablated[base + k]) dup_ok = false;
            }
            // Tick 0 predates any action divergence (frame 0 has prev==cur, so
            // mode 18 is a no-op there): the current-depth channel must match.
            const uint32_t first = e * fixed_ppo::actor_obs_dim;
            for (uint32_t k = 0; k < 80; k++)
                if (ablated[first + k] != normal[first + k]) front_ok = false;
        }
        require(history_exists, "test frames carry no history to ablate");
        require(dup_ok, "ablated observation did not duplicate current depth");
        require(front_ok, "ablation changed tick-0 current-depth channel");
        std::cout << "ablation_wire_test PASS ticks=" << kTicks << "\n";
    }

    // 5) Corrected depth-history gates (decision-corrected-history §4 G1/G3,
    //    on ACTUAL pose/range states): with the same collected frames, memory
    //    built under the real config keeps past-frame hits (the original
    //    leak), memory built under the mode-18 config (corrected ablate path)
    //    zeroes every back>=1 slot while keeping back==0 hits; and at tick 0
    //    FULL and CORRECTED-ABLATE observations/actions are byte-equal.
    {
        auto count_valid = [](const std::vector<float>& pts, uint32_t n,
                              uint32_t min_back) {
            uint32_t count = 0;
            for (uint32_t e = 0; e < n; e++)
                for (uint32_t j = 0; j < 160; j++)
                    if (j / 80 >= min_back &&
                        pts[(e * 160 + j) * 4 + 3] >= 0.0f)
                        count++;
            return count;
        };
        auto build_points = [&](Sim& s, id<MTLBuffer> cfg_buf,
                                std::vector<float>& out) {
            auto cb = [metal.queue commandBuffer];
            s.m.dispatch(cb, s.memory_points_p, s.cfg.n * 640,
                         {s.states, s.runs, s.sensors, s.poses, s.memory_points,
                          s.physics, cfg_buf});
            metal.finish(cb);
            const float* p = (const float*)s.memory_points.contents;
            out.assign(p, p + size_t(s.cfg.n) * 640);
        };
        const uint32_t n8 = 8;
        const uint32_t ticks = 6;
        // corrected ablate run (real path)
        SimConfig ca = config;
        ca.n = n8;
        ca.seed = 13;
        Sim simA(metal, ca, 32);
        enable_task_control(simA, dynamic_task_recipe());
        auto runA = make_dynamic_run(simA, entries, {}, control,
                                     metal.pipeline("dynamic_task_apply"),
                                     metal.pipeline("dynamic_script_act"), true);
        for (uint32_t t = 0; t < ticks; t++) {
            auto cb = [metal.queue commandBuffer];
            dynamic_tick(runA, cb, t, true, false, t == 0);
            metal.finish(cb);
        }
        std::vector<float> inpath;
        {
            const float* p = (const float*)simA.memory_points.contents;
            inpath.assign(p, p + size_t(n8) * 640);
        }
        const uint32_t inpath_back1 = count_valid(inpath, n8, 1);
        const uint32_t inpath_back0 = count_valid(inpath, n8, 0) - inpath_back1;
        require(inpath_back1 == 0,
                "corrected ablate path still exposes past-frame memory hits");
        require(inpath_back0 > 0,
                "corrected ablate path lost the current-frame memory hits");
        std::cout << "corrected_memory_path PASS back0_valid=" << inpath_back0
                  << " back1plus_valid=" << inpath_back1 << "\n";

        // G1 A/B on identical states: legacy config (original ablate arm's
        // dispatch) vs corrected config.
        std::vector<float> legacy, corrected;
        build_points(simA, simA.configs[0], legacy);            // mode 17/22
        build_points(simA, runA.obs_configs[0], corrected);     // mode 18
        const uint32_t legacy_back1 = count_valid(legacy, n8, 1);
        const uint32_t corrected_back1 = count_valid(corrected, n8, 1);
        require(legacy_back1 > 0,
                "legacy dispatch should keep past-frame hits (leak evidence)");
        require(corrected_back1 == 0,
                "mode-18 dispatch should zero past-frame hits");
        std::cout << "history_leak_ab_Pass legacy_back1plus_valid="
                  << legacy_back1 << " corrected_back1plus_valid="
                  << corrected_back1 << "\n";

        // FULL run: same frames keep history (that is the original contrast)
        SimConfig cf = config;
        cf.n = n8;
        cf.seed = 13;
        Sim simF(metal, cf, 32);
        enable_task_control(simF, dynamic_task_recipe());
        auto runF = make_dynamic_run(simF, entries, {}, control,
                                     metal.pipeline("dynamic_task_apply"),
                                     metal.pipeline("dynamic_script_act"), false);
        for (uint32_t t = 0; t < ticks; t++) {
            auto cb = [metal.queue commandBuffer];
            dynamic_tick(runF, cb, t, false, false, t == 0);
            metal.finish(cb);
        }
        std::vector<float> full_mem;
        {
            const float* p = (const float*)simF.memory_points.contents;
            full_mem.assign(p, p + size_t(n8) * 640);
        }
        const uint32_t full_back1 = count_valid(full_mem, n8, 1);
        require(full_back1 > 0,
                "FULL arm must retain past-frame geometry history");
        std::cout << "full_history_present PASS back1plus_valid=" << full_back1
                  << "\n";

        // G2(iii): tick-0 observation and sampled-action equality (identical
        // states, identical memory, so any later divergence can only come
        // from the declared channels).
        {
            const float* obsA = (const float*)simA.obs.contents;
            const float* obsF = (const float*)simF.obs.contents;
            const float* actA = (const float*)simA.actions.contents;
            const float* actF = (const float*)simF.actions.contents;
            require(std::memcmp(obsA, obsF,
                                n8 * fixed_ppo::actor_obs_dim * sizeof(float)) == 0,
                    "tick-0 observations differ between FULL and corrected ablate");
            require(std::memcmp(actA, actF, n8 * fixed_ppo::action_dim * sizeof(float)) == 0,
                    "tick-0 sampled actions differ between FULL and corrected ablate");
            // A later tick: the ablated run must duplicate 80..159 from 0..79
            // (declared channel) whenever the two runs still share states, and
            // always within its own tensor.
            const uint32_t last = (ticks - 1) * n8 * fixed_ppo::actor_obs_dim;
            bool dup_ok = true;
            for (uint32_t e = 0; e < n8; e++) {
                const uint32_t base = last + e * fixed_ppo::actor_obs_dim;
                for (uint32_t k = 0; k < 80; k++)
                    if (obsA[base + 80 + k] != obsA[base + k]) dup_ok = false;
            }
            require(dup_ok, "corrected ablate did not duplicate the prev-depth channel");
        }
        std::cout << "ablation_tick0_identity PASS\n";
    }

    // 6) Mixture slot mapping with a real static bank (optional path arg).
    if (!static_bank_path.empty()) {
        uint32_t static_period = 1;
        auto stat = read_static_bank(static_bank_path, static_period);
        require(static_period == kStaticSlots && stat.size() >= kStaticSlots * 8,
                "static bank must be period-8 with at least 8 envs of slots");
        SimConfig c4 = config;
        c4.n = 8;
        Sim sim4(metal, c4, 32);
        enable_task_control(sim4, dynamic_task_recipe());
        auto run4 = make_dynamic_run(sim4, entries, stat,
                                     DynBankControlHost{8, uint32_t(stat.size()),
                                                        4, 8, kMixturePeriod, 0},
                                     metal.pipeline("dynamic_task_apply"),
                                     metal.pipeline("dynamic_script_act"), false);
        // force slot 5 (dynamic=0..3, static slot 1) on env 0
        {
            auto* runs = (SimRun*)sim4.runs.contents;
            runs[0].episodes = 5;
        }
        auto probe4 = [metal.queue commandBuffer];
        dynamic_probe(run4, probe4);
        metal.finish(probe4);
        const auto* worlds4 = (const WWorld*)sim4.worlds.contents;
        require(std::memcmp(&worlds4[0], &stat[1].world, sizeof(WWorld)) == 0,
                "mixture slot 5 did not map to static slot 1");
        {
            auto* runs = (SimRun*)sim4.runs.contents;
            runs[0].episodes = 1;  // slot 1 -> dynamic entry 0*4+1
        }
        auto probe5 = [metal.queue commandBuffer];
        dynamic_probe(run4, probe5);
        metal.finish(probe5);
        require(std::memcmp(&worlds4[0], &entries[1].world, sizeof(WWorld)) == 0,
                "mixture slot 1 did not map to dynamic entry 1");
        std::cout << "mixture_slot_mapping_test PASS\n";
    }

    std::cout << "dynamic_test_all PASS sensor_profile=" << NAV_SENSOR_ACTIVE_NAME
              << "\n";
    return 0;
}

static void usage() {
    std::cout <<
        "commands\n"
        "  dynamic-test [STATIC_BANK]\n"
        "  dynamic-bank --train|--dev [--pairs] OUT_PREFIX [--seed N] [--target N]\n"
        "  dynamic-train ROLLOUTS CHECKPOINT WARMSTART --dyn-bank P --dyn-records P\n"
        "               --static-bank P --eval-dyn P --eval-static P --ablate 0|1\n"
        "               [--seed N] [--eval-every N] [--time-cost X] [--envs N]\n"
        "  dynamic-eval CHECKPOINT OUT_CSV (--dyn-bank P | --static-bank P) [--split NAME]\n"
        "               [--mode 17|13|4|2] [--ablate 0|1] [--per-tick 0|1] [--seed N]\n";
}

} // namespace dyn

int main(int argc, char** argv) {
    @autoreleasepool {
        try {
            std::setvbuf(stdout, nullptr, _IONBF, 0);
            std::setvbuf(stderr, nullptr, _IONBF, 0);
            if (argc < 2) {
                dyn::usage();
                return 0;
            }
            const std::string command = argv[1];
            if (command == "dynamic-test") return dyn::command_test(argc, argv);
            if (command == "dynamic-bank") return dyn::command_bank(argc, argv);
            if (command == "dynamic-train") return dyn::command_train(argc, argv);
            if (command == "dynamic-eval") return dyn::command_eval(argc, argv);
            if (command == "--help" || command == "help") {
                dyn::usage();
                return 0;
            }
            std::cerr << "unknown command " << command << "\n";
            dyn::usage();
            return 1;
        } catch (const std::exception& e) {
            std::cerr << "ERROR: " << e.what() << "\n";
            return 1;
        }
    }
}







