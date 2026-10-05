// Local waypoint PPO controller for the sensor-planned stack.
//
// Scope of this file
// ------------------
// The global-route problem is split (root decision) into a global discovery /
// memory layer and a local goal controller. This runner builds and trains only
// the local controller: given depth + ego state + ONE provided local goal
// (1-3 m), reach it, avoid contact, slow down and hold at the goal.
//
// It uses the existing production machinery unchanged -- `Sim` (depth, observe,
// actor, act, advance), `PPOTrainer` (GAE/PPO/Adam) and the frozen RAPTOR plant
// -- through the same translation unit. Nothing in main.mm/sim.metal/core is
// modified. The only new Metal kernel is `waypoint_task_apply`, dispatched
// immediately after the ordinary episode reset, which installs one kept bank
// record (world, start pose, start yaw, start velocity, goal) and the matching
// `NavigationTaskState`, exactly like `sim_reset_navigation_task` does for
// generated tasks. Task reward/termination/stable-arrival accounting stays the
// verified `navigation_task_step` path.
//
// Sources of the bank are SOURCE-only: the existing procedural TRAIN geometry
// families (0 open, 1 boxes, 2 poles, 4 doorway, 5 table/counter, 14 bent
// hallway, 15 connected rooms, 16 vertical over/under) regenerated from seeded
// scene seeds, plus host-side validated local start/goal pairs and a feasible
// witness. The actor never receives the witness, the scene seed or any
// geometry truth -- only the standard 184-float observation built by
// `sim_observe` from depth, ego state and the provided goal.
//
// Commands
//   local-bank   SPEC OUT_PREFIX              build a deterministic bank + CSV + manifest
//   local-train  ROLLOUTS CHECKPOINT WARMSTART [options]
//   local-eval   CHECKPOINT OUT_CSV [options] evaluate one bank split
//   local-trace  CHECKPOINT OUT_CSV [options] per-tick trajectory of a few envs
//
// Options are `--key value`. See usage() for the full list.
//
// SNAPSHOT NOTE (perception-support experiment): this file is the explicit
// experimental snapshot of the completed G3 runner
// `navigation_aware_training.mm` @ b696214e (preserved verbatim under
// results/omp-perception-support/provenance/archive-g3-runner/). Its one
// addition is the perception-support geometry prior: the portable module
// `perception_support.hpp` + the `ps_override_prior` kernel, dispatched only
// under `--perception-support1`. With the flag off (default) the pipelines,
// observation bytes, contracts and outputs match the original runner; the
// contract hashes now name this file and include the module so sidecars bind
// the mechanism code. New commands: `ps-probe` (causal instrument), `ps-test`
// (mechanism tests).

#include "guidance.hpp"              // nav_ray/nav_min3x3/nav_atanh for host tests
#include "perception_support.hpp"    // portable correction (host side)

#define main metal_nav_reserved_main
#include "main.mm"
#undef main

namespace waypoint {

// ---------------------------------------------------------------- bank ABI
//
// One record is one local waypoint task. The host builds it, the GPU consumes
// it verbatim after a reset. Layout is shared with the MSL mirror below.
struct BankEntry {
    WWorld world;                 // obstacle geometry, goal already local
    float start_position[3];
    float start_velocity[3];
    float goal_position[3];
    float start_yaw;
    float clearance_start;
    float clearance_goal;
    float direct_clearance;
    float witness_clearance;
    float witness_length;
    float initial_distance;
    uint32_t family;
    uint32_t scene_seed;
    uint32_t route_class;         // 0 direct segment, 1 validated dogleg
    uint32_t attempts;            // bounded search attempts used
};
static_assert(sizeof(BankEntry)==756,"waypoint bank entry ABI");

// rehearsal_weight:0 = legacy round-robin (slot = episodes mod period);
//1 = G3 fixed rehearsal weighting (decision-revision-g3-2026-10-04.md),
// period must be16: per24 episodes each env flies source slots0-7 twice
// then grounded slots8-15 once (every slot still served each cycle).
struct BankControl { uint32_t period, count, rehearsal_weight; };
static_assert(sizeof(BankControl)==12,"waypoint bank control ABI");

struct BankFileHeader {
    char magic[8];                // "WPBANK1\0"
    uint32_t version;
    uint32_t period;
    uint32_t count;
    uint32_t entry_bytes;
    char entry_sha256[64];
};
static_assert(sizeof(BankFileHeader)==88,"waypoint bank file header ABI");

// ------------------------------------------------------------ MSL mirror
static const char* kWaypointKernels=R"MSL(
struct WaypointBankEntry {
    WWorld world;
    float start_position[3];
    float start_velocity[3];
    float goal_position[3];
    float start_yaw;
    float clearance_start;
    float clearance_goal;
    float direct_clearance;
    float witness_clearance;
    float witness_length;
    float initial_distance;
    uint family;
    uint scene_seed;
    uint route_class;
    uint attempts;
};
struct WaypointBankControl { uint period; uint count; uint rehearsal_weight; };
// Install the kept bank record for this environment. Runs after the ordinary
// reset (or after an in-rollout auto-reset) while runs[n].steps==0, and leaves
// the RAPTOR recurrence, rotors, sensor ring and command ring that
// sim_reset_one already initialized untouched.
kernel void waypoint_task_apply(device RLPhysicsState* states [[buffer(0)]],
                                device SimRun* runs [[buffer(1)]],
                                device WWorld* worlds [[buffer(2)]],
                                device NavigationTaskState* tasks [[buffer(3)]],
                                device const WaypointBankEntry* bank [[buffer(4)]],
                                constant WaypointBankControl& bank_control [[buffer(5)]],
                                constant NavigationTaskControl& task_control [[buffer(6)]],
                                constant SimConfig& cfg [[buffer(7)]],
                                uint n [[thread_position_in_grid]]) {
    if(n>=cfg.n)return;
    if(runs[n].steps!=0u)return;
    const uint period=max(bank_control.period,1u);
    uint slot=runs[n].episodes%period;
    if(bank_control.rehearsal_weight==1u&&period==16u) {
        // Fixed rehearsal weighting (G3): per24-episode cycle, slots
        // [0..7,0..7,8..15] — source episodes16/24, grounded8/24, every
        // grounded slot still visited once per cycle (broad floor kept).
        const uint t=runs[n].episodes%24u;
        slot = t<8u ? t : (t<16u ? (t-8u) : (8u+(t-16u)));
    }
    const uint index=n*period+slot;
    if(index>=bank_control.count)return;
    const WaypointBankEntry entry=bank[index];
    WWorld world=entry.world;
    world.wind[0]=0.0f;world.wind[1]=0.0f;world.wind[2]=0.0f;
    worlds[n]=world;

    NavigationTaskState task=NavigationTaskState{};
    task.generation_status=NAV_TASK_GENERATION_READY;task.valid=1u;task.generation_attempts=entry.attempts;
    task.family=entry.family;task.stage=task_control.config.stage;task.objective=task_control.config.objective;
    task.max_nav_steps=task_control.config.max_nav_steps;task.step_count=0u;task.stable_ticks=0u;
    task.was_inside_goal=0u;task.waypoint_event_latched=0u;task.terminal=0u;
    for(uint axis=0;axis<3;axis++) {
        task.start_position[axis]=entry.start_position[axis];
        task.start_velocity_world[axis]=entry.start_velocity[axis];
        task.goal_position[axis]=entry.goal_position[axis];
        task.last_executed_world_command[axis]=entry.start_velocity[axis];
    }
    task.last_executed_world_command[3]=0.0f;
    task.start_yaw_rad=entry.start_yaw;
    task.initial_distance_m=entry.initial_distance;task.previous_distance_m=entry.initial_distance;
    task.difficulty=task_control.config.difficulty;
    task.start_clearance_m=entry.clearance_start;task.goal_clearance_m=entry.clearance_goal;
    task.direct_segment_clearance_m=entry.direct_clearance;
    task.witness_min_clearance_m=entry.witness_clearance;task.witness_length_m=entry.witness_length;
    task.stable_time_s=0.0f;task.witness_point_count=0u;
    tasks[n]=task;

    RLPhysicsState state=states[n];
    for(uint axis=0;axis<3;axis++) {
        state.position[axis]=entry.start_position[axis];
        state.linear_velocity[axis]=entry.start_velocity[axis];
        state.angular_velocity_body[axis]=0.0f;
        runs[n].reference_position[axis]=entry.start_position[axis];
        runs[n].desired_velocity[axis]=entry.start_velocity[axis];
    }
    const float half_yaw=entry.start_yaw*0.5f;
    state.orientation_wxyz[0]=cos(half_yaw);state.orientation_wxyz[1]=0.0f;
    state.orientation_wxyz[2]=0.0f;state.orientation_wxyz[3]=sin(half_yaw);
    states[n]=state;
    runs[n].yaw=entry.start_yaw;
    runs[n].initial_distance=entry.initial_distance;
    runs[n].path=0.0f;runs[n].elapsed=0.0f;runs[n].peak_speed=0.0f;runs[n].min_clearance=12.0f;
}
)MSL";

// ------------------------------------------------- perception support (the
// one mechanism of this experiment). The override kernel re-derives the
// geometry prior from MEASURED range rings, recorded ego poses, timestamps
// and the current frustum only, and rewrites the3-float observation tail
// (the prior channel) before the actor encodes. Network inputs, reward, PPO
// and action map are untouched; with the flag off this kernel is never
// dispatched (legacy parity).
static const char* kPerceptionKernels=R"PSMSL(
kernel void ps_override_prior(device float* obs [[buffer(0)]],
                              device SimRun* runs [[buffer(1)]],
                              device const RLPhysicsState* states [[buffer(2)]],
                              device const float* sensors [[buffer(3)]],
                              device const float* poses [[buffer(4)]],
                              constant RLPhysicsParams& p [[buffer(5)]],
                              constant SimConfig& cfg [[buffer(6)]],
                              uint n [[thread_position_in_grid]]) {
    if(n>=cfg.n || (cfg.eval && runs[n].episodes))return;
    const uint sensor_delay=sim_sensor_delay(cfg,n);
    const uint available=runs[n].steps/cfg.sensor_period;
    const uint frame=available>sensor_delay?available-sensor_delay:0;
    const uint prev_frame=frame>0?frame-1:0;
    const uint valid=available>=sensor_delay?min(frame+1,8-sensor_delay):0;
    const uint row=(cfg.tick*cfg.n+n)*PPO_ACTOR_OBS;
    const uint context=row+SIM_CONTEXT_OFFSET;
    float cur[80],previous_range[80],goal[3],vel[3],hint[3];
    for(uint k=0;k<80;k++){cur[k]=obs[row+k]*12.0f;previous_range[k]=obs[row+SIM_DEPTH_FEATURES+k]*12.0f;}
    for(uint j=0;j<3;j++){goal[j]=obs[context+j];vel[j]=obs[context+4+j]*4.0f;}
    // Same distance reconstruction the observation already carries (exact for
    // local goals <=15 m, which every local task satisfies).
    const float distance=obs[context+3]*10.0f;
    RLPhysicsState s=states[n];float rotation[9];sim_rotation(s.orientation_wxyz,rotation);
    float pose[12];
    for(uint j=0;j<3;j++)pose[j]=s.position[j];
    for(uint j=0;j<9;j++)pose[j+3]=rotation[j];
    ps_nav_guidance_memory(cur,previous_range,goal,distance,vel,
                           float(cfg.sensor_period)*p.dt*cfg.substeps,
                           NAV_SENSOR_ACTIVE_TAN_V,
                           sensors+n*8*320,poses+n*8*12,pose,frame,valid,hint,NAV_SENSOR_ACTIVE_MOUNT_X);
    for(uint j=0;j<3;j++)obs[row+PPO_ACTOR_OBS-3+j]=hint[j];
}
)PSMSL";

// Full compiled source for this experimental runner: core + the portable
// perception_support module + the override kernel. contract_core_sha256()
// hashes exactly this, so sidecars bind the mechanism code.
static std::string ps_source() {
    std::string root=SOURCE_DIR;
    return base_source()+read_text(root+"/perception_support.hpp")+kPerceptionKernels;
}

// -------------------------------------------------------------- bank specs
//
// A spec is the whole task distribution for one split: which source geometry
// families, the local goal horizon, the start velocity envelope and the
// maximum tolerated route/straight-line ratio. Everything is seeded.
struct BankSpec {
    std::string name;
    uint32_t seed=20261004;
    std::vector<uint32_t> families;
    uint32_t environments=128;
    uint32_t period=1;
    float scene_distance=4.0f;
    float goal_min_m=1.0f;
    float goal_max_m=3.0f;
    float start_velocity_max_mps=1.0f;
    float difficulty=0.5f;
    float route_length_ratio_max=2.5f;
    // Grounded capability rules (decision.md). False keeps the original
    // generator byte-for-byte: the source and exposed dev splits must never
    // shift under this flag.
    bool grounded_rules=false;
    // grounded-mix composes source slots first, grounded slots second.
    bool composite_source=false;
};

// Grounded acceptance constants (decision.md +2026-10-04 revision). All
// clearances are body-subtracted like wclearance (surface gap = value +0.18 m).
static constexpr float kVisibleSegmentClearanceM=0.12f; // flyable/visible line: surfaces >=0.30 m
static constexpr float kHalfFovRad=0.78539816340f;      // constructed-goal draw range (±45 deg)
static constexpr float kVelocityElevationRad=0.5235987756f; // stress velocity elevation ±30 deg

// Combined projected frustum from the ACTIVE sensor profile (root/contract
// correction: edge-to-edge cone, NOT independent azimuth/elevation bounds).
static constexpr float kSensorTanH=NAV_SENSOR_TAN_H;        // legacy 1.0
static constexpr float kSensorTanV=NAV_SENSOR_ACTIVE_TAN_V; // legacy 0.75
static constexpr uint32_t kRawRows=16,kRawCols=20;          //320 raw rays
static constexpr uint32_t kPooledRows=8,kPooledCols=10;     //80 min-pooled bins
static constexpr float kPassageCorridorM=0.18f;             // body radius corridor
static constexpr float kPassageRangeFactor=0.9f;            // range must reach90% of corridor

// Body-frame vector from the sensor origin at issue (yaw-only pose; mount
// offset along +x). Used for every frustum/ray label and the goal gate.
static void sensor_body_vector(float start_x,float start_y,float start_z,float start_yaw,
                               float goal_x,float goal_y,float goal_z,float out[3]) {
    const float dx=goal_x-start_x,dy=goal_y-start_y,dz=goal_z-start_z;
    const float cosine=std::cos(start_yaw),sine=std::sin(start_yaw);
    out[0]=cosine*dx+sine*dy-NAV_SENSOR_ACTIVE_MOUNT_X;
    out[1]=-sine*dx+cosine*dy;
    out[2]=dz;
}

static bool inside_projected_frustum(const float body[3]) {
    if(!(body[0]>0.0f))return false;
    return std::fabs(body[1]/body[0])<=kSensorTanH+1e-6f&&
           std::fabs(body[2]/body[0])<=kSensorTanV+1e-6f;
}

// Diagnostic only (never an acceptance gate): does the direction fall inside
// a measured pixel footprint — combined cone edges AND within half-pixel of
// the nearest ray centre. Declared tolerance: tan_h/20 by tan_v/16; legacy
// outermost centre tangents0.95 /0.703125.
static bool near_measured_ray_center(const float body[3]) {
    if(!inside_projected_frustum(body))return false;
    const float ty=body[1]/body[0],tz=body[2]/body[0];
    float column_fraction=(ty/kSensorTanH+1.0f)*0.5f*float(kRawCols)-0.5f;
    long column=lround(column_fraction);
    if(column<0)column=0;
    if(column>=long(kRawCols))column=long(kRawCols)-1;
    const float center_y=kSensorTanH*(2.0f*(float(column)+0.5f)/float(kRawCols)-1.0f);
    float row_fraction=(tz/kSensorTanV+1.0f)*0.5f*float(kRawRows)-0.5f;
    long row=lround(row_fraction);
    if(row<0)row=0;
    if(row>=long(kRawRows))row=long(kRawRows)-1;
    const float center_z=kSensorTanV*(2.0f*(float(row)+0.5f)/float(kRawRows)-1.0f);
    return std::fabs(ty-center_y)<=kSensorTanH/float(kRawCols)+1e-6f&&
           std::fabs(tz-center_z)<=kSensorTanV/float(kRawRows)+1e-6f;
}

// Raw and pooled see-through counts for the start->bend corridor (decision
// revision §2.2). Ray directions come from the sensor in the BODY frame and
// are rotated by the issue yaw before touching world geometry (contract P1:
// without the rotation the counts describe a yaw=0 hypothetical). Pooled
// range is the MIN of the pool's four raw rays — the conservative depth the
//80-bin actor observation would report.
static void passage_evidence(const WWorld& world,WVec origin,float start_yaw,WVec anchor,
                             uint32_t& raw_see,uint32_t& pooled_see) {
    raw_see=0;pooled_see=0;
    const WVec delta=ws(anchor,origin);
    const float distance=wl(delta);
    if(distance<1e-3f)return;
    const float need=kPassageRangeFactor*distance;
    const float cosine=std::cos(start_yaw),sine=std::sin(start_yaw);
    const auto to_world=[&](WVec body)->WVec {
        // Yaw-only body->world rotation (level attitude at issue).
        return wv(cosine*body.x-sine*body.y,sine*body.x+cosine*body.y,body.z);
    };
    const auto corridor=[&](WVec dir)->bool {
        const float length=wl(dir);
        if(length<1e-6f)return false;
        dir=wm(dir,1.0f/length);
        const float along=delta.x*dir.x+delta.y*dir.y+delta.z*dir.z;
        if(along<=0.0f)return false;
        const float along2=along*along;
        const float lateral_sq=std::max(0.0f,(delta.x*delta.x+delta.y*delta.y+delta.z*delta.z)-along2);
        return lateral_sq<=kPassageCorridorM*kPassageCorridorM;
    };
    for(uint32_t pixel=0;pixel<kRawRows*kRawCols;pixel++) {
        const WVec dir=to_world(nav_sensor_pixel_ray(kSensorTanH,kSensorTanV,pixel));
        if(!corridor(dir))continue;
        if(wray(world,origin,wm(dir,1.0f/std::max(1e-6f,wl(dir))),0.0f)>=need)raw_see++;
    }
    for(uint32_t pool_row=0;pool_row<kPooledRows;pool_row++)
        for(uint32_t pool_col=0;pool_col<kPooledCols;pool_col++) {
            const WVec center=to_world(nav_sensor_pooled_ray(kSensorTanH,kSensorTanV,pool_row,pool_col));
            if(!corridor(center))continue;
            float pooled_range=1e9f;
            for(uint32_t dy=0;dy<2;dy++)for(uint32_t dx=0;dx<2;dx++) {
                const uint32_t pixel=(pool_row*2+dy)*kRawCols+pool_col*2+dx;
                const WVec dir=to_world(nav_sensor_pixel_ray(kSensorTanH,kSensorTanV,pixel));
                const float length=wl(dir);
                if(length<1e-6f)continue;
                pooled_range=std::min(pooled_range,wray(world,origin,wm(dir,1.0f/length),0.0f));
            }
            if(pooled_range>=need)pooled_see++;
        }
}

// Per-accepted-task capability labels. Rows are written next to the bank CSV
// columns for grounded specs; generator truth only, never an actor input.
struct GroundedLabel {
    bool grounded=false;       // produced by the grounded rules (else rehearsal row)
    bool has_opening=false;    // blocked: geometric opening (cone bend + LOS >=0.12)
    bool visible_opening=false;// geometric AND >=1 pooled see-through bin
    bool side_fallback=false;  // blocked row accepted by the no-side escape
    float opening[3]={0,0,0};
    float opening_clearance_m=0.0f;
    int side=-1;               //0 left,1 right,2 center (|dy|<0.15 off the line)
    int vertical=-1;           //0 up,1 level,2 down (|dz|<0.30)
    int route_points=-1;       // polyline size:3 (one bend) or4 (two bends)
    float via2[3]={0,0,0};     // second bend for4-point witness routes
    int issue_kind=-1;         //0 stopped_slow U[0,0.2],1 independent_stress U[0.2,1]
    uint32_t raw_see=0,pooled_see=0; // passage evidence counts
};

// Rejection reasons and label histograms for one grounded build. Recorded,
// never silent: manifest counts must add up with the accepted rows.
struct GroundedStats {
    uint64_t start_clearance=0,goal_bounds=0,goal_clearance=0,
             goal_elevation_unreachable=0,goal_outside_projected_frustum=0,
             bend_elevation_unreachable=0,issue_yaw_unreachable=0,
             class_mismatch=0,direct_not_visible=0,opening_not_visible=0,
             opening_no_raw_evidence=0,
             no_route=0,ratio_too_long=0,side_rejected=0;
    uint64_t side_fallback_builds=0;   // builds that re-ran without side enforcement
    uint64_t grounded_accepted=0,blocked_accepted=0,direct_accepted=0;
    uint64_t issue_stopped=0,issue_stress=0;
    uint64_t side_left=0,side_right=0,side_center=0;
    uint64_t vertical_up=0,vertical_level=0,vertical_down=0;
    uint64_t velocity_toward=0,velocity_lateral=0,velocity_away=0;
    uint64_t visible_openings=0,pooled_see_zero=0;
};

static BankSpec spec_by_name(const std::string& name) {
    BankSpec spec;
    if(name=="source") {
        spec.name="source";spec.seed=20261004;spec.families={0,1,2,4,5,14,15,16};
        spec.environments=128;spec.period=8;spec.route_length_ratio_max=2.5f;
    } else if(name=="dev-a") {
        spec.name="dev-a";spec.seed=20261005;spec.families={2,4,5,14,15,16};
        spec.environments=128;spec.period=1;spec.route_length_ratio_max=2.5f;
    } else if(name=="dev-b") {
        spec.name="dev-b";spec.seed=20261007;spec.families={2,4,5,14,15,16};
        spec.environments=128;spec.period=1;spec.route_length_ratio_max=2.5f;
    } else if(name=="dev-c") {
        // Predeclared held-out split for the first experiment (audit §9):
        // fresh seed, never used for selection, built before any training run.
        spec.name="dev-c";spec.seed=20261010;spec.families={2,4,5,14,15,16};
        spec.environments=128;spec.period=1;spec.route_length_ratio_max=2.5f;
    } else if(name=="open") {
        spec.name="open";spec.seed=20261008;spec.families={0};
        spec.environments=128;spec.period=1;spec.route_length_ratio_max=3.0f;
    } else if(name=="clutter") {
        spec.name="clutter";spec.seed=20261009;spec.families={1,2,4,5,14,15,16};
        spec.environments=128;spec.period=1;spec.route_length_ratio_max=3.0f;
    } else if(name=="grounded") {
        // Grounded TRAIN component (decision.md): same families as source,
        // fresh seed, grounded rules.
        spec.name="grounded";spec.seed=20261011;spec.families={0,1,2,4,5,14,15,16};
        spec.environments=128;spec.period=8;spec.route_length_ratio_max=2.5f;
        spec.grounded_rules=true;
    } else if(name=="grounded-mix") {
        // Treatment arm bank: byte-identical source half (slots 0-7 per env)
        // plus a grounded half (slots 8-15), both present from rollout 0.
        spec.name="grounded-mix";spec.seed=20261011;spec.families={0,1,2,4,5,14,15,16};
        spec.environments=128;spec.period=16;spec.route_length_ratio_max=2.5f;
        spec.grounded_rules=true;spec.composite_source=true;
    } else if(name=="dev-g1"||name=="dev-g2") {
        // Frozen pre-learning grounded dev banks (decision revision §3). The
        // v1 seeds20261021/20261022 were exposed by the aborted pilot's
        // selector evaluations and are superseded; these fresh seeds are the
        // corrected freeze. dev-g1 is the selector; dev-g2 robustness only.
        spec.name=name;spec.seed=name=="dev-g1"?20261051u:20261052u;
        spec.families={2,4,5,14,15,16};
        spec.environments=128;spec.period=1;spec.route_length_ratio_max=2.5f;
        spec.grounded_rules=true;
    } else if(name=="dev-h") {
        // G3 fresh NON-selector bank (decision-revision-g3-2026-10-04.md):
        // frozen before any G3 run, never used for selection; scored on the
        // G3 finals plus the frozen C/T references.
        spec.name="dev-h";spec.seed=20261071u;
        spec.families={2,4,5,14,15,16};
        spec.environments=128;spec.period=1;spec.route_length_ratio_max=2.5f;
        spec.grounded_rules=true;
    } else if(name=="dev-k") {
        // Perception-support experiment: fresh NON-selector challenge bank in
        // the ORIGINAL source distribution (cold yaw, goal-directed start
        // velocity), frozen before any run; never used for selection.
        spec.name="dev-k";spec.seed=20261091u;
        spec.families={2,4,5,14,15,16};
        spec.environments=128;spec.period=1;spec.route_length_ratio_max=2.5f;
    } else {
        throw std::runtime_error("unknown bank spec '"+name+
            "' (source|dev-a|dev-b|dev-c|open|clutter|grounded|grounded-mix|dev-g1|dev-g2|dev-h|dev-k)");
    }
    return spec;
}

static const char* family_name(uint32_t family) {
    switch(family) {
        case 0:return "open";
        case 1:return "boxes";
        case 2:return "poles";
        case 4:return "doorway";
        case 5:return "table";
        case 14:return "bent_hallway";
        case 15:return "connected_rooms";
        case 16:return "vertical_choices";
        default:return "other";
    }
}

// Host RNG: the exact xorshift world.hpp uses on device, so bank construction
// is reproducible on this machine and portable to the same generator.
struct HostRng {
    uint32_t s;
    explicit HostRng(uint32_t seed):s(seed?seed:1u) {}
    uint32_t raw(){return wrng(s);}
    float uniform(){return float(raw()>>8)*(1.0f/16777216.0f);}
    float symmetric(){return 2.0f*uniform()-1.0f;}
    WVec unit() {
        const float z=symmetric();
        const float angle=6.28318530718f*uniform();
        const float radial=std::sqrt(std::max(0.0f,1.0f-z*z));
        return wv(radial*std::cos(angle),radial*std::sin(angle),z);
    }
};

static bool inside_bounds(WVec point,const float* low,const float* high) {
    return point.x>=low[0]&&point.x<=high[0]&&point.y>=low[1]&&point.y<=high[1]&&
           point.z>=low[2]&&point.z<=high[2];
}

// The bank floor for both the start and the goal: the room is x[-2,14],
// y[-5,5], z[0,5]; endpoints keep the documented 0.30 m geometry clearance.
static constexpr float kEndpointClearanceM=0.30f;
static constexpr float kMinimumWitnessClearanceM=0.02f;
// Start region: the clear approach side of the procedural field. Goals are
// then 1-3 m ahead, which places most of them inside or beyond the geometry.
static const float kStartLow[3]={-1.4f,-3.0f,0.8f};
static const float kStartHigh[3]={1.0f,3.0f,2.6f};
static const float kGoalLow[3]={-1.7f,-4.7f,0.3f};
static const float kGoalHigh[3]={13.7f,4.7f,4.7f};

// A local route is one straight segment or one intermediate waypoint. Both
// ends are validated free space; every segment keeps the 2 cm witness floor.
// `via` is the first interior bend when one exists: the grounded generator
// verifies it as the visible opening (decision.md).
struct LocalRoute {
    bool found=false;
    float length_m=0.0f;
    float clearance_m=0.0f;
    uint32_t segments=0;
    bool has_via=false;
    WVec via={0,0,0};
    bool has_via2=false;
    WVec via2={0,0,0};
};

static LocalRoute route_direct(const WWorld& world,WVec start,WVec goal) {
    LocalRoute route;
    route.clearance_m=navigation_task_segment_clearance(world,start,goal);
    route.length_m=wl(ws(goal,start));
    route.segments=1;
    route.found=true;
    return route;
}

// One intermediate waypoint on a coarse lattice between start and goal. This
// recovers routes the fixed dogleg set cannot express, in particular passing
// through a doorway gap in a full-height wall.
static LocalRoute route_via_waypoint(const WWorld& world,WVec start,WVec goal) {
    static const float fractions[3]={0.30f,0.50f,0.70f};
    static const float lateral[7]={-3.0f,-2.0f,-1.0f,0.0f,1.0f,2.0f,3.0f};
    static const float vertical[3]={-1.2f,0.0f,1.2f};
    LocalRoute best;
    for(float fraction:fractions)for(float dy:lateral)for(float dz:vertical) {
        const WVec point=wv(start.x+fraction*(goal.x-start.x),
                            start.y+fraction*(goal.y-start.y)+dy,
                            start.z+fraction*(goal.z-start.z)+dz);
        if(!inside_bounds(point,kGoalLow,kGoalHigh))continue;
        const float first=navigation_task_segment_clearance(world,start,point);
        if(first<=kMinimumWitnessClearanceM)continue;
        const float second=navigation_task_segment_clearance(world,point,goal);
        if(second<=kMinimumWitnessClearanceM)continue;
        const float length=wl(ws(point,start))+wl(ws(goal,point));
        if(!best.found||length<best.length_m) {
            best.found=true;best.length_m=length;best.clearance_m=std::min(first,second);best.segments=2;
            best.has_via=true;best.via=point;
        }
    }
    return best;
}

// Bounded local-route search: prefer the straight segment, then the validated
// geometric doglegs, then the waypoint lattice. Never returns an unvalidated
// route: an infeasible draw is rejected and resampled upstream.
static LocalRoute find_local_route(const WWorld& world,WVec start,WVec goal,
                                   const NavigationTaskConfig& witness_config,uint32_t& lattice_budget) {
    LocalRoute route=route_direct(world,start,goal);
    if(route.clearance_m>kMinimumWitnessClearanceM)return route;
    NavigationTaskWitness witness{};
    float direct_clearance=0.0f;
    if(navigation_task_find_witness(world,start,goal,witness_config,witness,direct_clearance)) {
        LocalRoute detour;
        detour.found=true;detour.length_m=witness.length_m;detour.clearance_m=witness.min_clearance_m;
        detour.segments=witness.count-1u;
        if(witness.count>=3u) {
            detour.has_via=true;detour.via=witness.points[1];
            if(witness.count>=4u) {
                detour.has_via2=true;detour.via2=witness.points[2];
            }
        }
        return detour;
    }
    if(lattice_budget>0) {
        lattice_budget--;
        return route_via_waypoint(world,start,goal);
    }
    return LocalRoute{};
}

// Bank CSV schema helpers. Grounded specs append the declared capability
// label columns (decision.md); the source schema stays byte-stable so prior
// bank CSV hashes keep verifying.
static void write_bank_csv_header(std::ostream& out,bool grounded_schema) {
    out<<"index,env,slot,family,family_name,scene_seed,route_class,attempts,"
          "start_x,start_y,start_z,start_yaw,start_velocity_x,start_velocity_y,start_velocity_z,"
          "goal_x,goal_y,goal_z,initial_distance_m,clearance_start_m,clearance_goal_m,"
          "direct_clearance_m,witness_clearance_m,witness_length_m";
    if(grounded_schema)
        out<<",grounded,issue_kind,side,side_fallback,vertical,"
              "goal_bearing_deg,goal_elevation_deg,goal_in_projected_frustum,"
              "directly_visible_goal,goal_hit_by_sensor_rays,"
              "opening_geometric,opening_x,opening_y,opening_z,opening_clearance_m,"
              "pooled_see_through_bins,raw_see_through_rays,visible_opening,"
              "route_points,via2_x,via2_y,via2_z,"
              "velocity_category,velocity_speed_mps,route_ratio_v";
    out<<'\n';
}

// Goal-directedness of a start velocity: label only, never a rejection rule.
static const char* velocity_category(const float* velocity,float goal_dx,float goal_dy) {
    const float speed=std::sqrt(velocity[0]*velocity[0]+velocity[1]*velocity[1]+velocity[2]*velocity[2]);
    if(speed<1e-6f)return "still";
    const float goal_length=std::sqrt(goal_dx*goal_dx+goal_dy*goal_dy);
    if(goal_length<1e-6f)return "lateral";
    const float alignment=(velocity[0]*goal_dx+velocity[1]*goal_dy)/(speed*goal_length);
    return alignment>0.5f?"toward":(alignment<-0.5f?"away":"lateral");
}

static void write_bank_csv_row(std::ostream& out,size_t index,uint32_t period,
                               const BankEntry& entry,const GroundedLabel& label,bool grounded_schema) {
    out<<index<<','<<index/period<<','<<index%period<<','<<entry.family<<','
       <<family_name(entry.family)<<','<<entry.scene_seed<<','<<entry.route_class<<','<<entry.attempts;
    for(int axis=0;axis<3;axis++)out<<','<<entry.start_position[axis];
    out<<','<<entry.start_yaw;
    for(int axis=0;axis<3;axis++)out<<','<<entry.start_velocity[axis];
    for(int axis=0;axis<3;axis++)out<<','<<entry.goal_position[axis];
    out<<','<<entry.initial_distance<<','<<entry.clearance_start<<','<<entry.clearance_goal
       <<','<<entry.direct_clearance<<','<<entry.witness_clearance<<','<<entry.witness_length;
    if(grounded_schema) {
        const float goal_dx=entry.goal_position[0]-entry.start_position[0];
        const float goal_dy=entry.goal_position[1]-entry.start_position[1];
        const float goal_dz=entry.goal_position[2]-entry.start_position[2];
        float goal_body[3];
        sensor_body_vector(entry.start_position[0],entry.start_position[1],entry.start_position[2],
                           entry.start_yaw,entry.goal_position[0],entry.goal_position[1],
                           entry.goal_position[2],goal_body);
        const float goal_bearing=std::remainder(std::atan2(goal_dy,goal_dx)-entry.start_yaw,
                                                6.28318530718f);
        const float goal_elevation=std::atan2(goal_dz,std::hypot(goal_dx,goal_dy));
        const bool in_cone=inside_projected_frustum(goal_body);
        const float speed=std::sqrt(entry.start_velocity[0]*entry.start_velocity[0]+
                                    entry.start_velocity[1]*entry.start_velocity[1]+
                                    entry.start_velocity[2]*entry.start_velocity[2]);
        out<<','<<(label.grounded?1:0)
           <<','<<(label.issue_kind<0?"":label.issue_kind==0?"stopped_slow":"independent_stress")
           <<','<<(label.side<0?"":label.side==0?"left":label.side==1?"right":"center")
           <<','<<(label.side_fallback?1:0)
           <<','<<(label.vertical<0?"":label.vertical==0?"up":label.vertical==1?"level":"down")
           <<','<<goal_bearing*57.29577951308232f<<','<<goal_elevation*57.29577951308232f
           <<','<<(in_cone?1:0)
           <<','<<((in_cone&&entry.direct_clearance>=kVisibleSegmentClearanceM)?1:0)
           <<','<<(near_measured_ray_center(goal_body)?1:0)
           <<','<<(label.has_opening?1:0)
           <<','<<label.opening[0]<<','<<label.opening[1]<<','<<label.opening[2]
           <<','<<label.opening_clearance_m
           <<','<<label.pooled_see<<','<<label.raw_see
           <<','<<(label.visible_opening?1:0)
           <<','<<label.route_points
           <<','<<label.via2[0]<<','<<label.via2[1]<<','<<label.via2[2]
           <<','<<velocity_category(entry.start_velocity,goal_dx,goal_dy)
           <<','<<speed
           <<','<<(entry.initial_distance>1e-6f?entry.witness_length/entry.initial_distance:0.0f);
    }
    out<<'\n';
}

// Core generation loop: fills bank+labels for one spec. csv writing and
// composition live in build_bank below.
static void generate_entries(const BankSpec& spec,std::vector<BankEntry>& bank,
                             std::vector<GroundedLabel>& labels,GroundedStats* stats) {
    NavigationTaskConfig witness_config{};
    witness_config.minimum_witness_clearance_m=kMinimumWitnessClearanceM;
    witness_config.require_detour=0u;
    HostRng rng(spec.seed);
    const size_t count=size_t(spec.environments)*spec.period;
    bank.clear();bank.reserve(count);
    labels.clear();labels.reserve(count);
    std::vector<uint32_t> route_counts(3,0),family_counts(17,0),class_counts(3,0);
    uint32_t fallbacks=0;
    // Round-robin left/right target for blocked grounded tasks: the side is
    // never hardcoded, it alternates over accepted blocked tasks.
    uint64_t blocked_accepted=0;
    // Parity counter for the issue-velocity buckets (stopped vs stress);
    // build-local so banks are identical with or without a stats sink.
    uint64_t grounded_seen=0;
    // Round-robin target over side-ful blocked accepts (center bends do not
    // consume a turn), so accepted left/right alternate exactly.
    uint32_t side_target=0;
    // class_mode: 0 any draw, 1 the straight line must be blocked (a real
    // local detour), 2 the straight line must already be clear.
    const auto try_accept=[&](const uint32_t family,const int class_mode,BankEntry& entry,
                              GroundedLabel& label,bool enforce_side) {
        uint32_t lattice_budget=8;
        for(uint32_t scene_attempt=1;scene_attempt<=16;scene_attempt++) {
            entry=BankEntry{};label=GroundedLabel{};
            const uint32_t scene_seed=rng.raw();
            wgenerate(entry.world,scene_seed,family,spec.scene_distance);
            const WVec start=wv(kStartLow[0]+(kStartHigh[0]-kStartLow[0])*rng.uniform(),
                                kStartLow[1]+(kStartHigh[1]-kStartLow[1])*rng.uniform(),
                                kStartLow[2]+(kStartHigh[2]-kStartLow[2])*rng.uniform());
            if(wclearance(entry.world,start,0.0f)<=kEndpointClearanceM) {
                if(stats)stats->start_clearance++;
                continue;
            }
            for(uint32_t direction_attempt=1;direction_attempt<=64;direction_attempt++) {
                label=GroundedLabel{};
                WVec direction=rng.unit();
                // Blocked-class draws aim into the procedural field: the
                // geometry always sits ahead of the start region, so a
                // backward draw almost never produces a detour.
                if(class_mode==1&&direction.x<0.0f&&rng.uniform()<0.75f)direction.x=-direction.x;
                const float distance=spec.goal_min_m+(spec.goal_max_m-spec.goal_min_m)*rng.uniform();
                const WVec goal=wa(start,wm(direction,distance));
                if(!inside_bounds(goal,kGoalLow,kGoalHigh)) {
                    if(stats)stats->goal_bounds++;
                    continue;
                }
                if(wclearance(entry.world,goal,0.0f)<=kEndpointClearanceM) {
                    if(stats)stats->goal_clearance++;
                    continue;
                }
                if(spec.grounded_rules) {
                    // Impossible-by-height goals are rejected before any route
                    // work: some yaw can fit the goal in the projected cone
                    // only if |z/horizontal| <= tan_v.
                    const float goal_horizontal=std::hypot(goal.x-start.x,goal.y-start.y);
                    const float goal_tangent=std::fabs(std::tan(std::atan2(goal.z-start.z,goal_horizontal)));
                    if(goal_tangent>kSensorTanV) {
                        if(stats)stats->goal_elevation_unreachable++;
                        continue;
                    }
                }
                const float direct=navigation_task_segment_clearance(entry.world,start,goal);
                const bool blocked=direct<=kMinimumWitnessClearanceM;
                if(class_mode==1&&!blocked) {
                    if(stats)stats->class_mismatch++;
                    continue;
                }
                if(class_mode==2&&blocked) {
                    if(stats)stats->class_mismatch++;
                    continue;
                }
                if(spec.grounded_rules&&!blocked&&direct<kVisibleSegmentClearanceM) {
                    // A grounded direct task needs a visibly flyable line,
                    // not just "not strictly blocked".
                    if(stats)stats->direct_not_visible++;
                    continue;
                }
                LocalRoute route;
                if(!blocked)route=route_direct(entry.world,start,goal);
                else route=find_local_route(entry.world,start,goal,witness_config,lattice_budget);
                if(!route.found) {
                    if(stats)stats->no_route++;
                    continue;
                }
                if(route.length_m>spec.route_length_ratio_max*std::max(distance,1e-3f)) {
                    if(stats)stats->ratio_too_long++;
                    continue;
                }
                if(spec.grounded_rules) {
                    // Choose the yaw AFTER the geometry so the goal and (for
                    // blocked rows) the first bend both land inside the
                    // combined projected frustum by construction (decision
                    // revision §2.1 addendum): uniform over the feasible
                    // bearing window, one draw, cone-verified below. A bend
                    // that no yaw can cover with the goal is honestly
                    // rejected (issue_yaw_unreachable / bend_elevation).
                    const float goal_world_bearing=std::atan2(goal.y-start.y,goal.x-start.x);
                    const float goal_horizontal=std::hypot(goal.x-start.x,goal.y-start.y);
                    const float goal_tangent=std::fabs(std::tan(std::atan2(goal.z-start.z,goal_horizontal)));
                    float theta_max=0.78539816340f;
                    {
                        const float cap=std::acos(std::min(1.0f,goal_tangent/kSensorTanV));
                        theta_max=std::min(theta_max,cap);
                    }
                    float theta_lo=-theta_max,theta_hi=theta_max;
                    if(blocked&&route.has_via) {
                        const WVec via=route.via;
                        const float via_horizontal=std::hypot(via.x-start.x,via.y-start.y);
                        const float via_tangent=std::fabs(std::tan(std::atan2(via.z-start.z,via_horizontal)));
                        if(via_tangent>kSensorTanV) {
                            if(stats)stats->bend_elevation_unreachable++;
                            continue;
                        }
                        const float delta=std::remainder(
                            std::atan2(via.y-start.y,via.x-start.x)-goal_world_bearing,6.28318530718f);
                        const float via_cap=std::acos(std::min(1.0f,via_tangent/kSensorTanV));
                        theta_lo=std::max(theta_lo,std::max(-0.78539816340f-delta,-via_cap-delta));
                        theta_hi=std::min(theta_hi,std::min(0.78539816340f-delta,via_cap-delta));
                    }
                    if(theta_lo>theta_hi) {
                        if(stats)stats->issue_yaw_unreachable++;
                        continue;
                    }
                    const float theta=theta_lo+(theta_hi-theta_lo)*rng.uniform();
                    entry.start_yaw=goal_world_bearing-theta;
                    while(entry.start_yaw>3.14159265359f)entry.start_yaw-=6.28318530718f;
                    while(entry.start_yaw<-3.14159265359f)entry.start_yaw+=6.28318530718f;
                    float goal_body[3];
                    sensor_body_vector(start.x,start.y,start.z,entry.start_yaw,
                                       goal.x,goal.y,goal.z,goal_body);
                    if(!inside_projected_frustum(goal_body)) {
                        if(stats)stats->goal_outside_projected_frustum++;
                        continue;
                    }
                }
                entry.family=family;entry.scene_seed=scene_seed;
                entry.start_position[0]=start.x;entry.start_position[1]=start.y;entry.start_position[2]=start.z;
                entry.goal_position[0]=goal.x;entry.goal_position[1]=goal.y;entry.goal_position[2]=goal.z;
                entry.world.goal[0]=goal.x;entry.world.goal[1]=goal.y;entry.world.goal[2]=goal.z;
                if(spec.grounded_rules) {
                    // entry.start_yaw was constructed and cone-verified above.
                    // Issue velocity buckets (decision revision §2.3): parity
                    // over accepted grounded rows — even = nominal stopped/slow
                    // issue U[0,0.2] (settled-arrival contract), odd =
                    // independent-direction stress U[0.2,1]. Direction is drawn
                    // independently in both buckets, labelled, never rejected.
                    const bool stress=(grounded_seen&1u)!=0u;
                    label.issue_kind=stress?1:0;
                    const float speed=stress?(0.2f+0.8f*rng.uniform()):(0.2f*rng.uniform());
                    const float azimuth=6.28318530718f*rng.uniform();
                    const float elevation=rng.symmetric()*kVelocityElevationRad;
                    entry.start_velocity[0]=speed*std::cos(elevation)*std::cos(azimuth);
                    entry.start_velocity[1]=speed*std::cos(elevation)*std::sin(azimuth);
                    entry.start_velocity[2]=speed*std::sin(elevation);
                    label.grounded=true;
                    if(blocked) {
                        // Geometric opening component (decision revision §2.2):
                        // first route bend inside the combined projected
                        // frustum with a flyable line-of-sight from the sensor
                        // origin. A blocker pixel or a bare bend is not an
                        // opening; pooled-passage evidence gates below.
                        if(!route.has_via) {
                            if(stats)stats->opening_not_visible++;
                            continue;
                        }
                        const WVec via=route.via;
                        if(!inside_bounds(via,kGoalLow,kGoalHigh)) {
                            if(stats)stats->opening_not_visible++;
                            continue;
                        }
                        float via_body[3];
                        sensor_body_vector(start.x,start.y,start.z,entry.start_yaw,
                                           via.x,via.y,via.z,via_body);
                        if(!inside_projected_frustum(via_body)) {
                            if(stats)stats->opening_not_visible++;
                            continue;
                        }
                        const float opening_clearance=
                            navigation_task_segment_clearance(entry.world,start,via);
                        if(opening_clearance<kVisibleSegmentClearanceM) {
                            if(stats)stats->opening_not_visible++;
                            continue;
                        }
                        // Side/vertical labels: offset of the bend from the
                        // straight line; side sign from the cross product so
                        // it is heading-invariant. Round-robin targets the
                        // side counter for EVERY accepted blocked row (the
                        // class-fallback path previously bypassed it).
                        const float line_x=goal.x-start.x,line_y=goal.y-start.y,line_z=goal.z-start.z;
                        const float length2=std::max(1e-6f,line_x*line_x+line_y*line_y);
                        const float along=std::max(0.0f,std::min(1.0f,
                            ((via.x-start.x)*line_x+(via.y-start.y)*line_y)/length2));
                        const float offset_y=via.y-(start.y+along*line_y);
                        const float offset_z=via.z-(start.z+along*line_z);
                        const float cross=line_x*(via.y-start.y)-line_y*(via.x-start.x);
                        label.side=std::fabs(offset_y)<0.15f?2:((cross>0.0f)?0:1);
                        label.vertical=std::fabs(offset_z)<0.30f?1:(offset_z>0.0f?0:2);
                        if(enforce_side&&(label.side==0||label.side==1)) {
                            const int target=int(side_target);
                            if(label.side!=target) {
                                if(stats)stats->side_rejected++;
                                continue;
                            }
                        }
                        if(!enforce_side&&(label.side==0||label.side==1))label.side_fallback=true;
                        // Sensed-passage evidence gate: at least one of the
                        //320 raw camera rays must see through the corridor
                        // (lateral <= body radius, range >=0.9*corridor) —
                        // raw-camera observability of the opening. The
                        //80-bin min-pooled count is recorded as the ACTOR-
                        // resolution diagnostic and reported, not used to
                        // filter (root policy-resolution review: measure
                        // whether min pooling loses the evidence).
                        uint32_t raw_see=0,pooled_see=0;
                        passage_evidence(entry.world,start,entry.start_yaw,via,raw_see,pooled_see);
                        if(raw_see<1u) {
                            if(stats)stats->opening_no_raw_evidence++;
                            continue;
                        }
                        label.raw_see=raw_see;label.pooled_see=pooled_see;
                        label.has_opening=true;
                        label.visible_opening=true;
                        label.opening[0]=via.x;label.opening[1]=via.y;label.opening[2]=via.z;
                        label.opening_clearance_m=opening_clearance;
                        label.route_points=int(route.segments)+1;
                        if(route.has_via2) {
                            label.via2[0]=route.via2.x;label.via2[1]=route.via2.y;label.via2[2]=route.via2.z;
                        }
                    }
                } else {
                    entry.start_yaw=rng.symmetric()*3.14159265359f;
                    const float speed=spec.start_velocity_max_mps*rng.uniform();
                    entry.start_velocity[0]=direction.x*speed;
                    entry.start_velocity[1]=direction.y*speed;
                    entry.start_velocity[2]=direction.z*speed;
                }
                entry.clearance_start=wclearance(entry.world,start,0.0f);
                entry.clearance_goal=wclearance(entry.world,goal,0.0f);
                entry.direct_clearance=direct;
                entry.witness_clearance=route.clearance_m;
                entry.witness_length=route.length_m;
                entry.initial_distance=wl(ws(goal,start));
                entry.route_class=blocked?1u:0u;
                entry.attempts=uint32_t(entry.route_class==1u?2u:1u);
                if(blocked)blocked_accepted++;
                if(label.grounded)grounded_seen++;
                if(label.side==0||label.side==1)side_target^=1u;
                if(stats&&label.grounded) {
                    stats->grounded_accepted++;
                    if(blocked)stats->blocked_accepted++;else stats->direct_accepted++;
                    if(label.issue_kind==0)stats->issue_stopped++;else stats->issue_stress++;
                    if(label.visible_opening) {
                        stats->visible_openings++;
                        if(label.pooled_see==0)stats->pooled_see_zero++;
                    }
                    if(label.has_opening) {
                        if(label.side==0)stats->side_left++;
                        else if(label.side==1)stats->side_right++;
                        else stats->side_center++;
                        if(label.vertical==0)stats->vertical_up++;
                        else if(label.vertical==1)stats->vertical_level++;
                        else stats->vertical_down++;
                    }
                    // Direction-goal independence statistics apply to the
                    // nonzero stress bucket only (stopped rows carry no
                    // meaningful direction).
                    if(label.issue_kind==1) {
                        const char* category=velocity_category(entry.start_velocity,
                                                              goal.x-start.x,goal.y-start.y);
                        if(std::strcmp(category,"toward")==0)stats->velocity_toward++;
                        else if(std::strcmp(category,"away")==0)stats->velocity_away++;
                        else stats->velocity_lateral++;
                    }
                }
                return true;
            }
        }
        return false;
    };
    for(size_t index=0;index<count;index++) {
        const uint32_t family=spec.families[index%spec.families.size()];
        // Alternate the preferred route class by cycle, not by global index,
        // so every geometry family gets both direct and detour tasks.
        const int preferred=((index/spec.families.size())%2==0)?2:1;
        BankEntry entry{};GroundedLabel label{};
        int achieved_class=preferred;
        // Side round-robin is enforced for every grounded primary attempt
        // (including the class fallback — the v1 bypass caused the dev-g1
        //25/8/15 skew); only the final bounded escape skips enforcement and
        // labels each such row side_fallback=1.
        const bool enforce=spec.grounded_rules;
        bool accepted=try_accept(family,preferred,entry,label,enforce);
        if(!accepted) { achieved_class=0; accepted=try_accept(family,0,entry,label,enforce); fallbacks++; }
        if(!accepted&&enforce) {
            accepted=try_accept(family,0,entry,label,false);
            if(accepted&&stats)stats->side_fallback_builds++;
        }
        require(accepted,"bank generation exhausted for "+spec.name+" index "+std::to_string(index)+
                " family "+std::to_string(family));
        class_counts[achieved_class]++;
        route_counts[entry.route_class]++;
        if(entry.family<family_counts.size())family_counts[entry.family]++;
        bank.push_back(entry);
        labels.push_back(label);
    }
    std::cout<<"bank_build spec="<<spec.name<<" seed="<<spec.seed<<" tasks="<<count
             <<" environments="<<spec.environments<<" period="<<spec.period
             <<" scene_distance_m="<<spec.scene_distance
             <<" goal_m=["<<spec.goal_min_m<<','<<spec.goal_max_m<<']'
             <<" start_speed_max="<<spec.start_velocity_max_mps
             <<" direct="<<route_counts[0]<<" detour="<<route_counts[1]
             <<" class_direct="<<class_counts[2]<<" class_detour="<<class_counts[1]
             <<" class_fallback="<<class_counts[0]<<" fallbacks="<<fallbacks<<"\n";
    for(uint32_t family=0;family<family_counts.size();family++)
        if(family_counts[family])std::cout<<"bank_family "<<family_name(family)<<"="<<family_counts[family]<<"\n";
}

// Build one deterministic bank (rows only; callers write the CSV header via
// write_bank_csv_header). A rejected task is retried with new scene and
// endpoint draws; a persistent failure is a hard error, never a silent
// filler. `grounded-mix` composes a byte-identical source rehearsal half
// first and a grounded half second (slots 0-7 / 8-15 per environment), so
// earlier capabilities are rehearsed from rollout 0 (decision.md).
static std::vector<BankEntry> build_bank(const BankSpec& spec,std::ostream* csv,
                                         GroundedStats* stats=nullptr,
                                         std::vector<GroundedLabel>* labels_out=nullptr) {
    std::vector<BankEntry> bank;std::vector<GroundedLabel> labels;
    if(spec.composite_source) {
        BankSpec source_spec=spec_by_name("source");
        BankSpec grounded_half=spec;
        grounded_half.name=spec.name+"-grounded-half";
        grounded_half.period=source_spec.period;grounded_half.composite_source=false;
        std::vector<BankEntry> source_bank,grounded_bank;
        std::vector<GroundedLabel> source_labels,grounded_labels;
        generate_entries(source_spec,source_bank,source_labels,nullptr);
        generate_entries(grounded_half,grounded_bank,grounded_labels,stats);
        require(grounded_bank.size()==source_bank.size(),"grounded-mix half sizes differ");
        require(source_bank.size()==size_t(spec.environments)*source_spec.period,
                "grounded-mix rehearsal half size mismatch");
        require(spec.period==source_spec.period+grounded_half.period,
                "grounded-mix period must equal the two half periods");
        // Per-env interleave (decision §6 / contract P1): slot s of every
        // environment maps to source[env*period+s] for s < source period and
        // grounded[...] after, so EVERY env rehearses both distributions.
        // (Plain concatenation would give envs 0-63 all-source and 64-127
        // all-grounded slots — the deviation the contract caught.)
        bank.reserve(source_bank.size()*2);labels.reserve(source_bank.size()*2);
        for(size_t env=0;env<size_t(spec.environments);env++) {
            for(uint32_t slot=0;slot<source_spec.period;slot++) {
                const size_t index=env*source_spec.period+slot;
                bank.push_back(source_bank[index]);labels.push_back(source_labels[index]);
            }
            for(uint32_t slot=0;slot<grounded_half.period;slot++) {
                const size_t index=env*grounded_half.period+slot;
                bank.push_back(grounded_bank[index]);labels.push_back(grounded_labels[index]);
            }
        }
        std::cout<<"bank_build spec="<<spec.name<<" seed="<<spec.seed<<" tasks="<<bank.size()
                 <<" environments="<<spec.environments<<" period="<<spec.period
                 <<" composite=per-env-interleave rehearsal:"<<source_bank.size()
                 <<" grounded:"<<grounded_bank.size()<<"\n";
    } else {
        generate_entries(spec,bank,labels,stats);
    }
    require(labels.size()==bank.size(),"bank label rows out of step");
    if(csv)
        for(size_t index=0;index<bank.size();index++)
            write_bank_csv_row(*csv,index,spec.period,bank[index],labels[index],spec.grounded_rules);
    if(labels_out)*labels_out=labels;
    return bank;
}

static void write_bank(const std::string& path,const BankSpec& spec,const std::vector<BankEntry>& bank) {
    const std::filesystem::path destination(path);
    if(destination.has_parent_path())std::filesystem::create_directories(destination.parent_path());
    BankFileHeader header{};
    std::memcpy(header.magic,"WPBANK1",7);
    header.version=1;
    header.period=spec.period;
    header.count=uint32_t(bank.size());
    header.entry_bytes=uint32_t(sizeof(BankEntry));
    const std::string entry_hash=ppo_safeguard_sha256(bank.data(),bank.size()*sizeof(BankEntry));
    std::memcpy(header.entry_sha256,entry_hash.data(),entry_hash.size());
    std::ofstream file(path,std::ios::binary|std::ios::trunc);
    require(bool(file),"cannot write bank "+path);
    file.write(reinterpret_cast<const char*>(&header),sizeof(header));
    file.write(reinterpret_cast<const char*>(bank.data()),std::streamsize(bank.size()*sizeof(BankEntry)));
    file.flush();require(bool(file),"bank write failed "+path);
    std::cout<<"bank_write "<<path<<" bytes="<<(sizeof(header)+bank.size()*sizeof(BankEntry))
             <<" entry_sha256="<<entry_hash<<"\n";
}

static std::vector<BankEntry> read_bank(const std::string& path,BankControl& control,std::string& entry_hash) {
    std::ifstream file(path,std::ios::binary);
    require(bool(file),"cannot open bank "+path);
    BankFileHeader header{};
    file.read(reinterpret_cast<char*>(&header),sizeof(header));
    require(bool(file)&&std::memcmp(header.magic,"WPBANK1",7)==0&&header.version==1&&
            header.entry_bytes==sizeof(BankEntry),"invalid bank header "+path);
    std::vector<BankEntry> bank(header.count);
    file.read(reinterpret_cast<char*>(bank.data()),std::streamsize(bank.size()*sizeof(BankEntry)));
    require(bool(file),"truncated bank "+path);
    entry_hash=ppo_safeguard_sha256(bank.data(),bank.size()*sizeof(BankEntry));
    require(std::memcmp(header.entry_sha256,entry_hash.data(),entry_hash.size())==0,
            "bank entry hash mismatch "+path);
    control.period=header.period?header.period:1;
    control.count=header.count;
    require(control.count%control.period==0,"bank count is not a multiple of its period");
    return bank;
}

// --------------------------------------------------------------- task recipe
//
// The step/reward semantics are the verified navigation-task contract. The
// generated placeholder task (stage OPEN_GOAL, family 0, empty room) only has
// to be cheap and valid: waypoint_task_apply always replaces it with the bank
// record before any observation is built.
static NavigationTaskControl local_task_recipe(float time_cost_per_s=0.2f) {
    NavigationTaskControl control=navigation_training::task_settings(NAV_TASK_STAGE_OPEN_GOAL,0u);
    control.config.goal_distance_min_m=1.0f;
    control.config.goal_distance_max_m=3.0f;
    // 0.2 is the verified navigation-task recipe. Anything else is an explicit
    // diagnostic arm and is recorded in the training provenance.
    control.config.time_cost_per_s=time_cost_per_s;
    return control;
}

static NavigationTaskControl enable_task_control(Sim& sim,const NavigationTaskControl& control) {
    require(std::fabs(control.config.nav_period_s-sim.cfg.substeps*0.01f)<1e-6f,
            "waypoint task clock must match the simulator navigation period");
    require(((const ChallengeBankControl*)sim.bank_control.contents)->enabled==0,
            "waypoint bank and frozen challenge bank cannot both own the world");
    require(((const NavigationRuntimeConfig*)sim.runtime_control.contents)->enabled==0,
            "waypoint training runs on the nominal plant; enable dynamics explicitly if needed");
    std::memcpy(sim.task_control.contents,&control,sizeof(control));
    return control;
}

// ------------------------------------------------------------- sim plumbing
struct LocalRun {
    Sim& sim;
    id<MTLBuffer> bank;
    id<MTLBuffer> bank_control;
    id<MTLComputePipelineState> apply_pipeline;
    bool support=false; // perception-support override dispatch (default off)
};

static LocalRun make_local_run(Sim& sim,const std::vector<BankEntry>& bank,const BankControl& control,
                               id<MTLComputePipelineState> apply_pipeline,float time_cost_per_s=0.2f,
                               bool support=false) {
    enable_task_control(sim,local_task_recipe(time_cost_per_s));
    LocalRun run{sim,nullptr,nullptr,apply_pipeline,support};
    run.bank=sim.m.buffer(bank.size()*sizeof(BankEntry),bank.data());
    run.bank_control=sim.m.buffer(sizeof(BankControl),&control);
    return run;
}

// One navigation tick plus the post-reset bank install, in the same order as
// Sim::collect. The install runs before the first observation of a fresh
// episode and again after advance() has auto-reset any finished environment.
static void local_tick(LocalRun& run,id<MTLCommandBuffer> cb,uint32_t tick,bool install_before) {
    Sim& sim=run.sim;
    auto c=sim.configs[tick%sim.horizon];
    const auto install=[&]() {
        sim.m.dispatch(cb,run.apply_pipeline,sim.cfg.n,
            {sim.states,sim.runs,sim.worlds,sim.task_states,run.bank,run.bank_control,sim.task_control,c},64);
    };
    if(install_before)install();
    sim.m.dispatch(cb,sim.depth_p,sim.cfg.n*320,{sim.states,sim.runs,sim.worlds,sim.sensors,sim.physics,c,sim.poses});
    if(sim.cfg.geometry_memory) {
        sim.m.dispatch(cb,sim.memory_points_p,sim.cfg.n*640,
            {sim.states,sim.runs,sim.sensors,sim.poses,sim.memory_points,sim.physics,c});
        sim.m.dispatch(cb,sim.memory_candidates_p,sim.cfg.n*85,
            {sim.states,sim.worlds,sim.memory_points,sim.memory_clearances,c});
    }
    sim.m.dispatch(cb,sim.observe_p,sim.cfg.n,
        {sim.states,sim.runs,sim.worlds,sim.sensors,sim.obs,sim.co,sim.physics,c,sim.poses,sim.memory_clearances},64);
    // Perception-support override: re-derive the geometry prior from measured
    // rings/poses/timestamps/frustum and rewrite the3-float prior tail before
    // the actor encodes. Never dispatched unless the run was flagged, so the
    // flag-off path is bit-identical to the preserved runner.
    if(run.support)
        sim.m.dispatch(cb,sim.m.pipeline("ps_override_prior"),sim.cfg.n,
            {sim.obs,sim.runs,sim.states,sim.sensors,sim.poses,sim.physics,c},64);
    const size_t row_offset=size_t(tick%sim.horizon)*sim.cfg.n;
    encode(sim.m,cb,sim.simd_actor?"ppo_actor_forward_simd_fused":"ppo_actor_forward",
           sim.simd_actor?((size_t(sim.cfg.n)+7)/8)*256:sim.cfg.n,
           {{sim.obs,row_offset*fixed_ppo::actor_obs_dim*4},{sim.actor,0},{sim.actor_workspace,0},
            {sim.actions,row_offset*fixed_ppo::action_dim*4},{sim.env_count,0}},sim.simd_actor?256:64);
    sim.m.dispatch(cb,sim.act_p,sim.cfg.n,
        {sim.states,sim.runs,sim.worlds,sim.obs,sim.co,sim.actor,sim.critic,sim.actions,sim.logp,sim.values,sim.commands,sim.physics,c},64);
    sim.m.dispatch(cb,sim.advance_p,sim.cfg.n,
        {sim.states,sim.runs,sim.worlds,sim.sensors,sim.commands,sim.raptor,sim.critic,sim.rewards,sim.next_values,
         sim.terminated,sim.truncated,sim.physics,c,sim.bank_worlds,sim.bank_schedule,sim.bank_control,sim.bank_active_ids,
         sim.bank_transition_ids,sim.environment_physics,sim.runtime_control,sim.task_states,sim.task_control,
         sim.potential_fields,sim.potential_spec,sim.potential_control},64);
    install();
}

// Install one bank record and build the matching first observation WITHOUT
// advancing the plant, so a reset contract check can read states, task records
// and observations at a true episode start.
static void local_probe(LocalRun& run,id<MTLCommandBuffer> cb) {
    Sim& sim=run.sim;
    auto c=sim.configs[0];
    sim.m.dispatch(cb,run.apply_pipeline,sim.cfg.n,
        {sim.states,sim.runs,sim.worlds,sim.task_states,run.bank,run.bank_control,sim.task_control,c},64);
    sim.m.dispatch(cb,sim.depth_p,sim.cfg.n*320,{sim.states,sim.runs,sim.worlds,sim.sensors,sim.physics,c,sim.poses});
    if(sim.cfg.geometry_memory) {
        sim.m.dispatch(cb,sim.memory_points_p,sim.cfg.n*640,
            {sim.states,sim.runs,sim.sensors,sim.poses,sim.memory_points,sim.physics,c});
        sim.m.dispatch(cb,sim.memory_candidates_p,sim.cfg.n*85,
            {sim.states,sim.worlds,sim.memory_points,sim.memory_clearances,c});
    }
    sim.m.dispatch(cb,sim.observe_p,sim.cfg.n,
        {sim.states,sim.runs,sim.worlds,sim.sensors,sim.obs,sim.co,sim.physics,c,sim.poses,sim.memory_clearances},64);
}

static void local_collect(LocalRun& run,id<MTLCommandBuffer> cb,uint32_t count) {
    // Progress ticks count from zero for a fresh run and continue from the
    // checkpointed simulator state for a resumed one.
    const uint32_t base=run.sim.cfg.tick;
    for(uint32_t step=0;step<count;step++)local_tick(run,cb,base+step,step==0);
}

// --------------------------------------------------------------- validation
//
// Confirm the physical contract of the reset: kept start pose/yaw/velocity,
// reference at the start, bank goal in the world, RAOTOR recurrence intact and
// the observation built from it exactly as the deployed stack would.
static void verify_reset(const LocalRun& run,const std::vector<BankEntry>& bank,const BankControl& control,
                         bool check_observation) {
    Sim& sim=run.sim;
    const auto* states=(const RLPhysicsState*)sim.states.contents;
    const auto* runs=(const SimRun*)sim.runs.contents;
    const auto* worlds=(const WWorld*)sim.worlds.contents;
    const auto* tasks=(const NavigationTaskState*)sim.task_states.contents;
    float worst_position=0,worst_yaw=0,worst_reference=0,worst_goal=0,worst_clearance=0;
    for(uint32_t env=0;env<sim.cfg.n;env++) {
        const BankEntry& entry=bank[size_t(env)*control.period+(runs[env].episodes%control.period)];
        for(int axis=0;axis<3;axis++) {
            worst_position=std::max(worst_position,std::fabs(states[env].position[axis]-entry.start_position[axis]));
            worst_reference=std::max(worst_reference,std::fabs(runs[env].reference_position[axis]-entry.start_position[axis]));
            worst_goal=std::max(worst_goal,std::fabs(worlds[env].goal[axis]-entry.goal_position[axis]));
        }
        worst_yaw=std::max(worst_yaw,std::fabs(runs[env].yaw-entry.start_yaw));
        require(tasks[env].valid==1&&tasks[env].generation_status==NAV_TASK_GENERATION_READY,
                "bank task record not installed for env "+std::to_string(env));
        require(tasks[env].objective==NAV_TASK_OBJECTIVE_FINAL_HOLD&&tasks[env].max_nav_steps==400,
                "bank task objective/budget mismatch");
        require(runs[env].steps==0&&runs[env].min_clearance==12.0f&&runs[env].elapsed==0.0f,
                "bank install did not start a clean episode");
        const float clearance=wclearance(worlds[env],wv(entry.start_position[0],entry.start_position[1],entry.start_position[2]),0.0f);
        worst_clearance=std::max(worst_clearance,std::fabs(clearance-entry.clearance_start));
        require(clearance>kEndpointClearanceM,"bank start pose is not clear of geometry");
        require(std::fabs(tasks[env].initial_distance_m-entry.initial_distance)<1e-5f,
                "bank task distance mismatch");
    }
    require(worst_position<1e-6f&&worst_reference<1e-6f&&worst_goal<1e-6f&&worst_yaw<1e-6f&&worst_clearance<2e-5f,
            "bank reset pose/goal/clearance invariant failed");
    std::cout<<"reset_invariants PASS position="<<worst_position<<" reference="<<worst_reference
             <<" goal="<<worst_goal<<" yaw="<<worst_yaw<<" clearance="<<worst_clearance
             <<" envs="<<sim.cfg.n<<"\n";
    if(!check_observation)return;

    // Observation contract at a fresh local-goal episode for the first four
    // environments: goal direction/distance, body velocity, reference error,
    // previous command and current depth must all match an independent CPU
    // evaluation of the same world, pose and sensor model.
    const float* obs=(const float*)sim.obs.contents;
    const uint32_t context=fixed_ppo::context_offset;
    float worst_goal_direction=0,worst_distance=0,worst_velocity=0,worst_reference_obs=0,worst_depth=0;
    for(uint32_t env=0;env<4;env++) {
        const BankEntry& entry=bank[size_t(env)*control.period];
        const RLPhysicsState& state=states[env];
        const float* r=nullptr;float rotation[9];
        const float w=state.orientation_wxyz[0],x=state.orientation_wxyz[1];
        const float y=state.orientation_wxyz[2],z=state.orientation_wxyz[3];
        rotation[0]=1-2*(y*y+z*z);rotation[1]=2*(x*y-w*z);rotation[2]=2*(x*z+w*y);
        rotation[3]=2*(x*y+w*z);rotation[4]=1-2*(x*x+z*z);rotation[5]=2*(y*z-w*x);
        rotation[6]=2*(x*z-w*y);rotation[7]=2*(y*z+w*x);rotation[8]=1-2*(x*x+y*y);
        r=rotation;
        float delta[3];
        for(int axis=0;axis<3;axis++)delta[axis]=entry.goal_position[axis]-state.position[axis];
        const float distance=std::sqrt(delta[0]*delta[0]+delta[1]*delta[1]+delta[2]*delta[2]);
        const uint32_t row=env*fixed_ppo::actor_obs_dim;
        for(int axis=0;axis<3;axis++) {
            const float expected=(r[axis]*delta[0]+r[3+axis]*delta[1]+r[6+axis]*delta[2])/std::max(distance,1e-6f);
            worst_goal_direction=std::max(worst_goal_direction,std::fabs(obs[row+context+axis]-expected));
            const float expected_velocity=(r[axis]*state.linear_velocity[0]+r[3+axis]*state.linear_velocity[1]+
                                           r[6+axis]*state.linear_velocity[2])/4.0f;
            worst_velocity=std::max(worst_velocity,std::fabs(obs[row+context+4+axis]-expected_velocity));
            worst_reference_obs=std::max(worst_reference_obs,std::fabs(obs[row+context+18+axis]));
        }
        worst_distance=std::max(worst_distance,std::fabs(obs[row+context+3]-std::min(distance/10.0f,1.5f)));
        // Depth: 2x2 min pool of the raw 16x20 range image, same sensor model
        // as sim_depth, cast from the mounted sensor origin.
        const float mount=NAV_SENSOR_ACTIVE_MOUNT_X;
        const WVec origin=wv(state.position[0]+mount*r[0],state.position[1]+mount*r[3],state.position[2]+mount*r[6]);
        for(uint32_t k=0;k<80;k++) {
            const uint32_t pool_row=k/10,pool_col=k%10;
            float expected=12.0f;
            for(uint32_t dy=0;dy<2;dy++)for(uint32_t dx=0;dx<2;dx++) {
                const uint32_t pixel=(pool_row*2+dy)*20+pool_col*2+dx;
                const WVec ray=nav_sensor_pixel_ray(NAV_SENSOR_TAN_H,NAV_SENSOR_ACTIVE_TAN_V,pixel);
                const WVec direction=wv(r[0]*ray.x+r[1]*ray.y+r[2]*ray.z,
                                        r[3]*ray.x+r[4]*ray.y+r[5]*ray.z,
                                        r[6]*ray.x+r[7]*ray.y+r[8]*ray.z);
                expected=std::min(expected,wray(worlds[env],origin,direction,0.0f));
            }
            worst_depth=std::max(worst_depth,std::fabs(obs[row+k]-expected/12.0f));
        }
    }
    require(worst_goal_direction<1e-5f&&worst_distance<1e-5f&&worst_velocity<1e-5f&&
            worst_reference_obs<1e-5f&&worst_depth<2e-5f,
            "bank observation contract failed");
    std::cout<<"observation_contract PASS goal_direction="<<worst_goal_direction
             <<" goal_distance="<<worst_distance<<" body_velocity="<<worst_velocity
             <<" reference_error="<<worst_reference_obs<<" depth="<<worst_depth<<"\n";
}

// ------------------------------------------------------------------- scores
struct BankScore {
    double success=0,collision=0,timeout=0;
    double mean_time_s=0,mean_path_m=0,mean_speed_mps=0,min_clearance_m=12;
    std::array<double,20> family_episodes{},family_successes{};
    std::array<double,20> family_collisions{};
};

static void write_eval_csv(const std::string& path,const std::string& split,const BankSpec* spec,
                           const std::vector<BankEntry>& bank,const BankControl& control,const Sim& sim,
                           uint32_t seed,uint32_t mode,
                           const std::vector<GroundedLabel>* labels=nullptr) {
    if(path.empty())return;
    const bool grounded_schema=spec&&spec->grounded_rules;
    if(grounded_schema)require(labels&&labels->size()==bank.size(),
        "grounded evaluation CSV requires the capability labels");
    const std::filesystem::path destination(path);
    if(destination.has_parent_path())std::filesystem::create_directories(destination.parent_path());
    std::ofstream file(path);
    require(bool(file),"cannot write evaluation CSV "+path);
    file<<std::setprecision(9);
    file<<"split,env,family,family_name,route_class,scene_seed,seed,mode,sensor_delay,command_delay,"
          "start_x,start_y,start_z,start_yaw,"
          "goal_x,goal_y,goal_z,goal_distance_m,direct_clearance_m,witness_clearance_m,witness_length_m,"
          "route_ratio,success,collision,timeout,time_s,path_m,mean_speed_mps,peak_speed_mps,min_clearance_m,"
          "final_x,final_y,final_z,final_distance_m,final_speed_mps,stable_hold_s";
    if(grounded_schema)
        file<<",grounded,issue_kind,side,vertical,opening_geometric,visible_opening,"
               "route_points,velocity_category,velocity_speed_mps";
    file<<'\n';
    const auto* runs=(const SimRun*)sim.runs.contents;
    const auto* states=(const RLPhysicsState*)sim.states.contents;
    const auto* tasks=(const NavigationTaskState*)sim.task_states.contents;
    for(uint32_t env=0;env<sim.cfg.n;env++) {
        const BankEntry& entry=bank[size_t(env)*control.period];
        const SimRun& run=runs[env];
        const RLPhysicsState& state=states[env];
        float distance=0;
        for(int axis=0;axis<3;axis++) {
            const float delta=entry.goal_position[axis]-state.position[axis];
            distance+=delta*delta;
        }
        distance=std::sqrt(distance);
        const float speed=std::sqrt(state.linear_velocity[0]*state.linear_velocity[0]+
                                    state.linear_velocity[1]*state.linear_velocity[1]+
                                    state.linear_velocity[2]*state.linear_velocity[2]);
        file<<(spec?spec->name:"custom")<<','<<env<<','<<entry.family<<','<<family_name(entry.family)<<','
            <<entry.route_class<<','<<entry.scene_seed<<','<<seed<<','<<mode<<','
            <<sim.cfg.sensor_delay<<','<<sim.cfg.command_delay;
        for(int axis=0;axis<3;axis++)file<<','<<entry.start_position[axis];
        file<<','<<entry.start_yaw;
        for(int axis=0;axis<3;axis++)file<<','<<entry.goal_position[axis];
        file<<','<<entry.initial_distance<<','<<entry.direct_clearance<<','<<entry.witness_clearance
            <<','<<entry.witness_length<<','<<(entry.initial_distance>1e-6f?entry.witness_length/entry.initial_distance:0)
            <<','<<run.successes<<','<<run.collisions<<','<<run.timeouts<<','<<run.elapsed<<','<<run.path
            <<','<<(run.elapsed>1e-6f?run.path/run.elapsed:0)<<','<<run.peak_speed<<','<<run.min_clearance
            <<','<<state.position[0]<<','<<state.position[1]<<','<<state.position[2]<<','<<distance<<','<<speed
            <<','<<tasks[env].stable_time_s;
        if(grounded_schema) {
            const GroundedLabel& label=(*labels)[size_t(env)*control.period];
            file<<','<<(label.grounded?1:0)
                <<','<<(label.issue_kind<0?"":label.issue_kind==0?"stopped_slow":"independent_stress")
                <<','<<(label.side<0?"":label.side==0?"left":label.side==1?"right":"center")
                <<','<<(label.vertical<0?"":label.vertical==0?"up":label.vertical==1?"level":"down")
                <<','<<(label.has_opening?1:0)
                <<','<<(label.visible_opening?1:0)
                <<','<<label.route_points
                <<','<<velocity_category(entry.start_velocity,
                                         entry.goal_position[0]-entry.start_position[0],
                                         entry.goal_position[1]-entry.start_position[1])
                <<','<<std::sqrt(entry.start_velocity[0]*entry.start_velocity[0]+
                                 entry.start_velocity[1]*entry.start_velocity[1]+
                                 entry.start_velocity[2]*entry.start_velocity[2]);
        }
        file<<'\n';
    }
}

// Run one bank split exactly once per environment with the given inference
// mode and return the aggregate score plus an optional per-episode CSV.
static BankScore evaluate_bank(Metal& metal,const float* actor,const BankSpec* spec,
                               const std::vector<BankEntry>& bank,const BankControl& control,
                               uint32_t mode,uint32_t seed,float speed,const std::string& csv_path,
                               const std::string& split,uint32_t environments=128,uint32_t max_steps=400,
                               uint32_t sensor_delay=0,uint32_t command_delay=0,
                               const std::vector<GroundedLabel>* labels=nullptr,
                               bool support=false) {
    SimConfig config;
    config.n=environments;config.mode=mode;config.family=0;config.eval=1;config.seed=seed;
    config.speed=speed;config.distance=4;config.max_steps=max_steps;config.geometry_memory=1;
    config.sensor_delay=sensor_delay;config.command_delay=command_delay;
    require(sensor_delay<=6&&command_delay<=7,"bank evaluation delay rings support <=6 sensor frames and <=7 command steps");
    require(bank.size()==size_t(environments)*control.period,"evaluation bank does not match the environment count");
    Sim sim(metal,config,32);
    if(actor)std::memcpy(sim.actor.contents,actor,fixed_ppo::actor_param_count*4);
    LocalRun run=make_local_run(sim,bank,control,metal.pipeline("waypoint_task_apply"),0.2f,support);
    auto probe=[metal.queue commandBuffer];
    local_probe(run,probe);
    metal.finish(probe);
    // The depth channel intentionally reports the initial max-range frame until
    // the delay ring is populated, so the full observation contract is only
    // asserted on the no-sensor-delay path.
    if(sensor_delay==0) {
        verify_reset(run,bank,control,true);
    } else {
        std::cout<<"observation_contract skipped sensor_delay="<<sensor_delay
                 <<" (depth channel still holds the initial max-range frame)\n";
        verify_reset(run,bank,control,false);
    }
    auto commands=[metal.queue commandBuffer];
    local_collect(run,commands,max_steps);
    const double gpu=metal.finish(commands);

    const auto* runs=(const SimRun*)sim.runs.contents;
    BankScore score;
    for(uint32_t env=0;env<config.n;env++) {
        const SimRun& run_result=runs[env];
        require(run_result.episodes==1,"bank evaluation must complete exactly one task per environment");
        const BankEntry& entry=bank[size_t(env)*control.period];
        const size_t slot=std::min<size_t>(entry.family,score.family_episodes.size()-1);
        score.success+=run_result.successes;score.collision+=run_result.collisions;
        score.timeout+=run_result.timeouts;
        score.mean_time_s+=run_result.successes?run_result.elapsed:0;
        score.mean_path_m+=run_result.path;
        score.min_clearance_m=std::min<double>(score.min_clearance_m,run_result.min_clearance);
        score.family_episodes[slot]+=1;
        score.family_successes[slot]+=run_result.successes;
        score.family_collisions[slot]+=run_result.collisions;
    }
    const double count=config.n;
    score.mean_time_s=score.success>0?score.mean_time_s/score.success:0;
    score.mean_path_m/=count;
    score.success/=count;score.collision/=count;score.timeout/=count;
    write_eval_csv(csv_path,split,spec,bank,control,sim,seed,mode,labels);
    std::cout<<"bank_eval split="<<split<<" mode="<<mode<<" seed="<<seed<<" tasks="<<config.n
             <<" success="<<score.success<<" collision="<<score.collision<<" timeout="<<score.timeout
             <<" mean_arrival_s="<<score.mean_time_s<<" mean_path_m="<<score.mean_path_m
             <<" min_clearance_m="<<score.min_clearance_m<<" gpu_s="<<gpu<<"\n";
    for(uint32_t family=0;family<score.family_episodes.size();family++)
        if(score.family_episodes[family]>0)
            std::cout<<"bank_eval_family "<<family_name(family)
                     <<" n="<<score.family_episodes[family]
                     <<" success="<<score.family_successes[family]/score.family_episodes[family]
                     <<" collision="<<score.family_collisions[family]/score.family_episodes[family]<<"\n";
    return score;
}

// -------------------------------------------------------------------- CLI
static void usage() {
    std::cout<<
        "commands\n"
        "  local-test\n"
        "  local-contract [OUT_JSON]     runner hash + compiled Metal source hash + profile\n"
        "  local-bank  SPEC OUT_PREFIX [--tasks N] [--seed N]\n"
        "  local-train ROLLOUTS CHECKPOINT WARMSTART [--spec NAME] [--bank PATH] [--seed N]\n"
        "              [--epochs N] [--envs N] [--horizon N] [--logstd X] [--eval-spec NAME]\n"
        "              [--eval-every N] [--learning-rate X] [--entropy X] [--speed X]\n"
        "              [--time-cost X] [--preflight RECEIPT.json] (required for grounded specs/selector)\n"
        "              [--rehearsal-weight 0|1] (1 = G3 fixed source-heavy schedule, grounded-mix only)\n"
        "              [--perception-support 0|1] (1 = measured-rings geometry prior, this module)\n"
        "  local-eval  CHECKPOINT OUT_CSV [--spec NAME] [--bank PATH] [--mode N] [--seed N]\n"
        "              [--envs N] [--max-steps N] [--sensor-delay N] [--command-delay N] [--selector 0|1]\n"
        "              [--perception-support 0|1]\n"
        "  ps-test     mechanism tests: module semantics, GPU/host parity, legacy parity\n"
        "  ps-probe    causal instrument: prior/command vs measured coverage at terminals\n"
        "  local-trace CHECKPOINT OUT_CSV [--spec NAME] [--bank PATH] [--mode N] [--seed N] [--envs N]\n"
        "  local-teacher CHECKPOINT DATASET [--spec NAME] [--max-ticks N]\n"
        "              observe-before-motion hybrid controller, positive RAPTOR flights only\n"
        "  local-bc DATASET OUT_CKPT WARMSTART [--updates N] [--batch N]\n"
        "              verified imitation-gradient BC; warmstart must regenerate every label\n"
        "  local-witness BANK.bin BANK.csv OUT_PREFIX\n"
        "              truth-scripted feasibility flights through RAPTOR (offline receipts)\n"
        "  eval|export reserved production commands (legacy families, deployment export)\n"
        "specs: source dev-a dev-b dev-c open clutter grounded grounded-mix dev-g1 dev-g2\n";
}

static std::string option_value(int argc,char** argv,int& index,const std::string& name) {
    require(index+1<argc,"option "+name+" requires a value");
    return argv[++index];
}

static float option_float(int argc,char** argv,int& index,const std::string& name) {
    return std::stof(option_value(argc,argv,index,name));
}
static uint32_t option_uint(int argc,char** argv,int& index,const std::string& name) {
    return uint32_t(std::stoul(option_value(argc,argv,index,name)));
}

// --------------------------------------------------------------- commands
// Production parity gates (unchanged kernels) plus the local-bank contract on
// every split: a full rollout and PPO update must stay finite.
static int command_test() {
    Metal metal;
    metal.compile(ps_source()+PPO_TRAINER_MSL+kWaypointKernels);
    world_tests(metal);
    raptor_tests(metal);
    physics_tests(metal);
    raptor_px4_adapter_tests();
    require(fixed_ppo::run_cpu_self_tests(),"PPO CPU self tests");
    ppo_tests(metal);
    geodesic_direction_gradient_test(metal);
    closed_loop_tests(metal);
    mixed_domain_tests(metal);
    deployed_action_map_test(metal);
    for(const char* name:{"source","dev-a","dev-b","open","clutter"}) {
        const BankSpec spec=spec_by_name(name);
        const auto bank=build_bank(spec,nullptr);
        const BankControl control{spec.period,uint32_t(bank.size())};
        SimConfig config;
        config.n=spec.environments;config.mode=22;config.family=0;config.seed=spec.seed;
        config.speed=1.5f;config.distance=4;config.max_steps=400;config.geometry_memory=1;
        config.entropy_coef=0.001f;config.learning_rate=0.0001f;
        Sim sim(metal,config,32);
        LocalRun run=make_local_run(sim,bank,control,metal.pipeline("waypoint_task_apply"));
        auto probe=[metal.queue commandBuffer];
        local_probe(run,probe);
        metal.finish(probe);
        verify_reset(run,bank,control,true);
        PPOTrainer trainer(sim,2);
        auto rollout=[metal.queue commandBuffer];
        local_collect(run,rollout,32);
        const double collect_gpu=metal.finish(rollout);
        auto update=[metal.queue commandBuffer];
        trainer.rollout_update(update,0);
        const double update_gpu=metal.finish(update);
        const float* reward=(const float*)sim.rewards.contents;
        const float* advantage=(const float*)sim.advantages.contents;
        const float* metric=(const float*)trainer.metric_mean.contents;
        for(size_t index=0;index<size_t(config.n)*32;index++)
            require(std::isfinite(reward[index])&&std::isfinite(advantage[index]),"non-finite rollout buffer");
        const auto* runs=(const SimRun*)sim.runs.contents;
        uint64_t episodes=0,terminals=0;
        for(uint32_t env=0;env<config.n;env++)episodes+=runs[env].episodes;
        const auto* terminated=(const uint8_t*)sim.terminated.contents;
        for(size_t index=0;index<size_t(config.n)*32;index++)terminals+=terminated[index]?1:0;
        std::cout<<"local_test split="<<spec.name<<" tasks="<<bank.size()<<" episodes="<<episodes
                 <<" terminals="<<terminals<<" policy_loss="<<metric[0]<<" value_loss="<<metric[1]
                 <<" ratio="<<metric[3]<<" collect_gpu_s="<<collect_gpu<<" update_gpu_s="<<update_gpu<<" PASS\n";
    }
    std::cout<<"local_test_all PASS sensor_profile="<<NAV_SENSOR_ACTIVE_NAME<<"\n";
    return 0;
}

static int command_bank(int argc,char** argv) {
    require(argc>=4,"local-bank SPEC OUT_PREFIX");
    std::string spec_name=argv[2],output_prefix=argv[3];
    uint32_t tasks=0,seed=0;
    for(int index=4;index<argc;index++) {
        const std::string option=argv[index];
        if(option=="--seed")seed=option_uint(argc,argv,index,option);
        else if(option=="--tasks")tasks=option_uint(argc,argv,index,option);
        else throw std::runtime_error("unknown local-bank option "+option);
    }
    BankSpec spec=spec_by_name(spec_name);
    if(seed)spec.seed=seed;
    if(tasks) {
        require(tasks%(spec.environments*spec.period)==0,"--tasks must be a multiple of environments*period");
        const uint32_t cycles=tasks/(spec.environments*spec.period);
        spec.environments*=cycles;
    }
    const std::filesystem::path prefix(output_prefix);
    if(prefix.has_parent_path())std::filesystem::create_directories(prefix.parent_path());
    std::ofstream csv(output_prefix+".csv");
    require(bool(csv),"cannot write bank CSV");
    csv<<std::setprecision(9);
    write_bank_csv_header(csv,spec.grounded_rules);
    GroundedStats grounded_stats;
    const auto bank=build_bank(spec,&csv,spec.grounded_rules?&grounded_stats:nullptr);
    if(spec.grounded_rules) {
        std::ofstream labels_manifest(output_prefix+".labels.json");
        require(bool(labels_manifest),"cannot write grounded labels manifest");
        labels_manifest<<std::setprecision(6)
            <<"{\n  \"schema\":\"grounded-bank-labels-v2\""
            <<",\n  \"spec\":\""<<spec.name<<"\""
            <<",\n  \"seed\":"<<spec.seed;
        if(spec.composite_source)
            labels_manifest<<",\n  \"layout\":\"per-env interleave: slot s<8 = source rehearsal, slot s>=8 = grounded (every environment rehearses both)\"";
        labels_manifest
            <<",\n  \"rules\":{"
            <<"\n    \"frustum\":\"combined projected cone from sensor origin: x>0 and |y/x|<=tan_h and |z/x|<=tan_v (active profile)\""
            <<",\n    \"tan_h\":"<<kSensorTanH
            <<",\n    \"tan_v\":"<<kSensorTanV
            <<",\n    \"mount_x_m\":"<<NAV_SENSOR_ACTIVE_MOUNT_X
            <<",\n    \"goal_draw_half_fov_deg\":45.0"
            <<",\n    \"ray_center_tangents\":{\"col\":0.95,\"row\":0.703125}"
            <<",\n    \"ray_center_tolerance\":{\"tangent_y\":\"tan_h/20\",\"tangent_z\":\"tan_v/16\"}"
            <<",\n    \"visible_segment_clearance_m\":"<<kVisibleSegmentClearanceM
            <<",\n    \"endpoint_clearance_m\":"<<kEndpointClearanceM
            <<",\n    \"witness_floor_m\":"<<kMinimumWitnessClearanceM
            <<",\n    \"issue_buckets\":\"parity over accepted grounded rows: stopped_slow U[0,0.2] / independent_stress U[0.2,1], direction independent, never rejected\""
            <<",\n    \"velocity_elevation_deg\":30.0"
            <<",\n    \"passage_evidence\":\"visible_opening requires raw_see_through_rays>=1 (320 raw rays, lateral<=0.18m, range>=0.9*corridor); pooled_see_through_bins recorded as the actor-resolution diagnostic (never a filter)\""
            <<",\n    \"side_enforcement\":\"round-robin left/right over EVERY accepted blocked row; bounded no-enforcement escape labelled side_fallback per row; center = |lateral offset|<0.15m\"}"
            <<",\n  \"rejections_attempts\":{"
            <<"\n    \"start_clearance\":"<<grounded_stats.start_clearance
            <<",\n    \"goal_bounds\":"<<grounded_stats.goal_bounds
            <<",\n    \"goal_clearance\":"<<grounded_stats.goal_clearance
            <<",\n    \"goal_elevation_unreachable\":"<<grounded_stats.goal_elevation_unreachable
            <<",\n    \"goal_outside_projected_frustum\":"<<grounded_stats.goal_outside_projected_frustum
            <<",\n    \"bend_elevation_unreachable\":"<<grounded_stats.bend_elevation_unreachable
            <<",\n    \"issue_yaw_unreachable\":"<<grounded_stats.issue_yaw_unreachable
            <<",\n    \"class_mismatch\":"<<grounded_stats.class_mismatch
            <<",\n    \"direct_not_visible\":"<<grounded_stats.direct_not_visible
            <<",\n    \"opening_not_visible\":"<<grounded_stats.opening_not_visible
            <<",\n    \"opening_no_raw_evidence\":"<<grounded_stats.opening_no_raw_evidence
            <<",\n    \"no_route\":"<<grounded_stats.no_route
            <<",\n    \"ratio_too_long\":"<<grounded_stats.ratio_too_long
            <<",\n    \"side_rejected\":"<<grounded_stats.side_rejected
            <<"\n  }"
            <<",\n  \"side_fallback_builds\":"<<grounded_stats.side_fallback_builds
            <<",\n  \"frozen_denominators\":{"
            <<"\n    \"n\":"<<(grounded_stats.grounded_accepted+0)
            <<",\n    \"blocked\":"<<grounded_stats.blocked_accepted
            <<",\n    \"direct\":"<<grounded_stats.direct_accepted
            <<",\n    \"issue_stopped_slow\":"<<grounded_stats.issue_stopped
            <<",\n    \"issue_independent_stress\":"<<grounded_stats.issue_stress
            <<"\n  }"
            <<",\n  \"labels\":{"
            <<"\n    \"side\":{\"left\":"<<grounded_stats.side_left
            <<",\"right\":"<<grounded_stats.side_right
            <<",\"center\":"<<grounded_stats.side_center<<"}"
            <<",\n    \"vertical\":{\"up\":"<<grounded_stats.vertical_up
            <<",\"level\":"<<grounded_stats.vertical_level
            <<",\"down\":"<<grounded_stats.vertical_down<<"}"
            <<",\n    \"visible_openings\":"<<grounded_stats.visible_openings
            <<",\n    \"pooled_see_zero_among_visible\":"<<grounded_stats.pooled_see_zero
            <<",\n    \"velocity_stress\":{\"toward\":"<<grounded_stats.velocity_toward
            <<",\"lateral\":"<<grounded_stats.velocity_lateral
            <<",\"away\":"<<grounded_stats.velocity_away<<"}}"
            <<"\n}\n";
        labels_manifest.close();
        std::cout<<"grounded_labels_manifest "<<output_prefix<<".labels.json"
                 <<" n="<<grounded_stats.grounded_accepted
                 <<" blocked="<<grounded_stats.blocked_accepted
                 <<" direct="<<grounded_stats.direct_accepted
                 <<" stopped="<<grounded_stats.issue_stopped
                 <<" stress="<<grounded_stats.issue_stress<<"\n";
    }
    csv.flush();csv.close();
    write_bank(output_prefix+".bin",spec,bank);
    std::ofstream manifest(output_prefix+".json");
    require(bool(manifest),"cannot write bank manifest");
    manifest<<std::setprecision(9)<<"{\n  \"schema\":\"local-waypoint-bank-v1\",\n"
            <<"  \"spec\":\""<<spec.name<<"\",\n  \"seed\":"<<spec.seed<<",\n"
            <<"  \"environments\":"<<spec.environments<<",\n  \"period\":"<<spec.period<<",\n"
            <<"  \"scene_distance_m\":"<<spec.scene_distance<<",\n"
            <<"  \"goal_min_m\":"<<spec.goal_min_m<<",\n  \"goal_max_m\":"<<spec.goal_max_m<<",\n"
            <<"  \"start_velocity_max_mps\":"<<spec.start_velocity_max_mps<<",\n"
            <<"  \"route_length_ratio_max\":"<<spec.route_length_ratio_max<<",\n"
            <<"  \"endpoint_clearance_m\":"<<kEndpointClearanceM<<",\n"
            <<"  \"families\":[";
    for(size_t index=0;index<spec.families.size();index++)manifest<<(index?",":"")<<spec.families[index];
    manifest<<"],\n  \"bank_file\":\""<<output_prefix<<".bin\",\n"
            <<"  \"bank_sha256\":\""<<challenge_evaluation::sha256_file(output_prefix+".bin")<<"\",\n"
            <<"  \"actor_sees\":\"depth,ego,provided_local_goal_only\"\n}\n";
    std::cout<<"bank_manifest "<<output_prefix<<".json\n";
    return 0;
}

// ------------------------------------------------------- resume contract
//
// A resumed run is only valid for the identical task bank bytes, sensor
// profile, runner source, compiled Metal source text, training parameters and
// evaluation sampler. The sidecar is read and validated BEFORE any output file
// is touched and BEFORE the checkpoint is loaded, so a mismatched invocation
// can never overwrite artifacts or silently continue a different experiment.
struct TrainContract {
    std::string sensor_profile;
    uint32_t actor_obs_dim=0;
    std::string runner_sha256,core_sha256,warmstart_sha256;
    std::string bank_spec,bank_entry_sha256;
    uint32_t bank_seed=0,bank_entries=0,bank_period=0;
    std::string eval_spec,eval_entry_sha256;
    uint32_t eval_entries=0,eval_period=0;
    uint32_t environments=0,horizon=0,epochs=0,eval_every=0,max_steps=0;
    uint32_t rehearsal_weight=0;
    uint32_t perception_support=0;
    float learning_rate=0,entropy_coef=0,risk_coef=0,value_coef=0,speed=0,time_cost=0,warmstart_log_std=0;
};

// The exact Metal source text this binary compiles at run time. Pinning only
// the runner file would leave world.hpp, sim.metal, ppo.metal, guidance.hpp and
// the sensor profile free to change under a resumed checkpoint.
static std::string contract_core_sha256() {
    const std::string source=ps_source()+PPO_TRAINER_MSL+kWaypointKernels;
    return ppo_safeguard_sha256(source.data(),source.size());
}
static std::string contract_runner_sha256() {
    return challenge_evaluation::sha256_file(std::string(SOURCE_DIR)+"/perception_support_runner.mm");
}

// Minimal strict reader for the flat sidecar object: a key must exist and its
// token must be exactly the expected kind. A sidecar without every required key
// is rejected as legacy/partial; mere existence is never enough.
static std::string json_token(const std::string& text,const std::string& key) {
    const std::string needle="\""+key+"\"";
    size_t position=text.find(needle);
    require(position!=std::string::npos,"resume sidecar is missing required key \""+key+"\"");
    position+=needle.size();
    while(position<text.size()&&std::isspace(static_cast<unsigned char>(text[position])))position++;
    require(position<text.size()&&text[position]==':',"resume sidecar key \""+key+"\" has no value");
    position++;
    while(position<text.size()&&std::isspace(static_cast<unsigned char>(text[position])))position++;
    require(position<text.size(),"resume sidecar key \""+key+"\" has an empty value");
    std::string token;
    if(text[position]=='"') {
        const size_t end=text.find('"',position+1);
        require(end!=std::string::npos,"resume sidecar key \""+key+"\" has an unterminated string");
        token=text.substr(position+1,end-position-1);
    } else {
        while(position<text.size()&&text[position]!=','&&text[position]!='\n'&&text[position]!='}')token+=text[position++];
        while(!token.empty()&&std::isspace(static_cast<unsigned char>(token.back())))token.pop_back();
    }
    require(!token.empty(),"resume sidecar key \""+key+"\" has an empty value");
    return token;
}
static uint32_t json_uint(const std::string& text,const std::string& key) {
    const std::string token=json_token(text,key);
    return uint32_t(std::stoul(token));
}
static float json_float(const std::string& text,const std::string& key) {
    const std::string token=json_token(text,key);
    return std::stof(token);
}
static void require_json_string(const std::string& text,const std::string& key,const std::string& expected) {
    const std::string actual=json_token(text,key);
    require(actual==expected,"resume sidecar key \""+key+"\" is \""+actual+"\" but this run uses \""+expected+"\"");
}
static void require_json_uint(const std::string& text,const std::string& key,uint32_t expected) {
    const uint32_t actual=json_uint(text,key);
    require(actual==expected,"resume sidecar key \""+key+"\" is "+std::to_string(actual)+
            " but this run uses "+std::to_string(expected));
}
static void require_json_float(const std::string& text,const std::string& key,float expected) {
    const float actual=json_float(text,key);
    require(actual==expected,"resume sidecar key \""+key+"\" is "+std::to_string(actual)+
            " but this run uses "+std::to_string(expected));
}

// Validate every contract field. Any missing key, any mismatch and any change
// of bank bytes, profile, compiled source or training parameter aborts before
// the checkpoint is loaded.
static void require_contract_matches(const std::string& sidecar_path,const TrainContract& expected) {
    require(std::filesystem::exists(sidecar_path),
            "checkpoint has no training sidecar "+sidecar_path+
            "; refusing to resume without a verifiable contract");
    const std::string text=read_text(sidecar_path);
    require_json_string(text,"schema","local-waypoint-train-v2");
    require_json_string(text,"sensor_profile",expected.sensor_profile);
    require_json_uint(text,"actor_obs_dim",expected.actor_obs_dim);
    require_json_string(text,"runner_sha256",expected.runner_sha256);
    require_json_string(text,"core_sha256",expected.core_sha256);
    require_json_string(text,"warmstart_sha256",expected.warmstart_sha256);
    require_json_string(text,"bank_spec",expected.bank_spec);
    require_json_uint(text,"bank_seed",expected.bank_seed);
    require_json_string(text,"bank_entry_sha256",expected.bank_entry_sha256);
    require_json_uint(text,"bank_entries",expected.bank_entries);
    require_json_uint(text,"bank_period",expected.bank_period);
    require_json_string(text,"eval_spec",expected.eval_spec);
    require_json_string(text,"eval_entry_sha256",expected.eval_entry_sha256);
    require_json_uint(text,"eval_entries",expected.eval_entries);
    require_json_uint(text,"eval_period",expected.eval_period);
    require_json_uint(text,"environments",expected.environments);
    require_json_uint(text,"horizon",expected.horizon);
    require_json_uint(text,"epochs",expected.epochs);
    require_json_uint(text,"eval_every",expected.eval_every);
    require_json_uint(text,"max_steps",expected.max_steps);
    require_json_float(text,"learning_rate",expected.learning_rate);
    require_json_float(text,"entropy_coef",expected.entropy_coef);
    require_json_float(text,"risk_coef",expected.risk_coef);
    require_json_float(text,"value_coef",expected.value_coef);
    require_json_float(text,"speed_cap_mps",expected.speed);
    require_json_float(text,"time_cost_per_s",expected.time_cost);
    require_json_float(text,"warmstart_log_std",expected.warmstart_log_std);
    // rehearsal_weight is bound when present; sidecars that predate the key
    // resume only under weight0 (legacy schedule), never under a weighted run.
    if(text.find("\"rehearsal_weight\"")!=std::string::npos)
        require_json_uint(text,"rehearsal_weight",expected.rehearsal_weight);
    else
        require(expected.rehearsal_weight==0,
                "sidecar predates rehearsal_weight but this run uses "
                "--rehearsal-weight "+std::to_string(expected.rehearsal_weight)+
                "; refusing to resume");
    // perception_support is bound the same way: a weighted/mechanism resume
    // needs the key; legacy resumes at support0 work with old sidecars.
    if(text.find("\"perception_support\"")!=std::string::npos)
        require_json_uint(text,"perception_support",expected.perception_support);
    else
        require(expected.perception_support==0,
                "sidecar predates perception_support but this run uses "
                "--perception-support "+std::to_string(expected.perception_support)+
                "; refusing to resume");
    require_json_string(text,"optimizer_state","reset_on_warmstart");
    require_json_string(text,"critic_state","warmstarted_from_checkpoint");
    std::cout<<"resume_contract PASS sidecar="<<sidecar_path
             <<" profile="<<expected.sensor_profile
             <<" bank="<<expected.bank_spec<<"@"<<expected.bank_entry_sha256.substr(0,12)
             <<" core="<<expected.core_sha256.substr(0,12)
             <<" runner="<<expected.runner_sha256.substr(0,12)<<"\n";
}

// One line per checked field so the negative tests can point at the exact key.
static void write_contract(const std::string& path,const TrainContract& contract) {
    std::ofstream out(path);
    require(bool(out),"cannot write training sidecar "+path);
    out<<std::setprecision(9)<<"{\n"
       <<"  \"schema\":\"local-waypoint-train-v2\",\n"
       <<"  \"sensor_profile\":\""<<contract.sensor_profile<<"\",\n"
       <<"  \"actor_obs_dim\":"<<contract.actor_obs_dim<<",\n"
       <<"  \"runner_sha256\":\""<<contract.runner_sha256<<"\",\n"
       <<"  \"core_sha256\":\""<<contract.core_sha256<<"\",\n"
       <<"  \"warmstart_sha256\":\""<<contract.warmstart_sha256<<"\",\n"
       <<"  \"bank_spec\":\""<<contract.bank_spec<<"\",\n"
       <<"  \"bank_seed\":"<<contract.bank_seed<<",\n"
       <<"  \"bank_entry_sha256\":\""<<contract.bank_entry_sha256<<"\",\n"
       <<"  \"bank_entries\":"<<contract.bank_entries<<",\n"
       <<"  \"bank_period\":"<<contract.bank_period<<",\n"
       <<"  \"eval_spec\":\""<<contract.eval_spec<<"\",\n"
       <<"  \"eval_entry_sha256\":\""<<contract.eval_entry_sha256<<"\",\n"
       <<"  \"eval_entries\":"<<contract.eval_entries<<",\n"
       <<"  \"eval_period\":"<<contract.eval_period<<",\n"
       <<"  \"environments\":"<<contract.environments<<",\n"
       <<"  \"horizon\":"<<contract.horizon<<",\n"
       <<"  \"epochs\":"<<contract.epochs<<",\n"
       <<"  \"eval_every\":"<<contract.eval_every<<",\n"
       <<"  \"max_steps\":"<<contract.max_steps<<",\n"
       <<"  \"learning_rate\":"<<contract.learning_rate<<",\n"
       <<"  \"entropy_coef\":"<<contract.entropy_coef<<",\n"
       <<"  \"risk_coef\":"<<contract.risk_coef<<",\n"
       <<"  \"value_coef\":"<<contract.value_coef<<",\n"
       <<"  \"speed_cap_mps\":"<<contract.speed<<",\n"
       <<"  \"time_cost_per_s\":"<<contract.time_cost<<",\n"
       <<"  \"warmstart_log_std\":"<<contract.warmstart_log_std<<",\n"
       <<"  \"rehearsal_weight\":"<<contract.rehearsal_weight<<",\n"
       <<"  \"perception_support\":"<<contract.perception_support<<",\n"
       <<"  \"optimizer_state\":\"reset_on_warmstart\",\n"
       <<"  \"critic_state\":\"warmstarted_from_checkpoint\"\n"
       <<"}\n";
    out.flush();
    require(bool(out),"training sidecar write failed: "+path);
}

// Print the identity of this build: the runner hash, the exact Metal source
// text compiled at run time, the sensor profile and the sidecar schema. Makes
// the 2000-rollout provenance verifiable after the runner source changes.
static int command_contract(int argc,char** argv) {
    const std::string runner=contract_runner_sha256();
    const std::string core=contract_core_sha256();
    std::ostringstream json;
    json<<std::setprecision(9)<<"{\n"
        <<"  \"schema\":\"local-waypoint-contract-v1\",\n"
        <<"  \"train_sidecar_schema\":\"local-waypoint-train-v2\",\n"
        <<"  \"sensor_profile\":\""<<NAV_SENSOR_ACTIVE_NAME<<"\",\n"
        <<"  \"actor_obs_dim\":"<<fixed_ppo::actor_obs_dim<<",\n"
        <<"  \"runner_file\":\"navigation_aware_training.mm\",\n"
        <<"  \"runner_sha256\":\""<<runner<<"\",\n"
        <<"  \"core_source_sha256\":\""<<core<<"\",\n"
        <<"  \"core_source_bytes\":"<<(ps_source()+PPO_TRAINER_MSL+kWaypointKernels).size()<<"\n"
        <<"}\n";
    std::cout<<json.str();
    if(argc>2) {
        std::ofstream out(argv[2]);require(bool(out),"cannot write contract "+std::string(argv[2]));
        out<<json.str();out.flush();require(bool(out),"contract write failed");
        std::cout<<"contract_write "<<argv[2]<<"\n";
    }
    return 0;
}

// Fail-closed preflight for grounded runs (decision revision §5): a PASS
// receipt published by the independent contract checker must bind this exact
// runner, training-bank bytes and selector-bank bytes. The check runs BEFORE
// any checkpoint, sidecar or bank file is created or mutated. Old source/dev
// runs never pass the option and behave exactly as before.
static void require_grounded_preflight(const std::string& receipt_path,
                                       const BankSpec& train_spec,
                                       const BankSpec& selector_spec,
                                       const std::string& train_bank_sha,
                                       const std::string& selector_bank_sha) {
    require(!receipt_path.empty(),
            "grounded run requires --preflight RECEIPT.json (independent contract PASS); no files written");
    std::ifstream file(receipt_path);
    require(bool(file),"preflight receipt not found: "+receipt_path+" (refusing grounded run)");
    const std::string text((std::istreambuf_iterator<char>(file)),std::istreambuf_iterator<char>());
    require(text.find("\"schema\":\"grounded-preflight-v1\"")!=std::string::npos,
            "preflight receipt schema mismatch (want grounded-preflight-v1)");
    require(text.find("\"verdict\":\"PASS\"")!=std::string::npos,
            "preflight verdict is not PASS (refusing grounded run)");
    const auto need=[&](const char* key,const std::string& value) {
        const std::string needle=std::string("\"")+key+"\":\""+value+"\"";
        require(text.find(needle)!=std::string::npos,
                std::string("preflight receipt mismatch for ")+key+" (want "+value+")");
    };
    need("checker","task-contract");
    need("train_spec",train_spec.name);
    need("train_bank_entry_sha256",train_bank_sha);
    need("runner_sha256",contract_runner_sha256());
    if(!selector_spec.name.empty()&&selector_spec.name!="none") {
        need("selector_spec",selector_spec.name);
        need("selector_bank_entry_sha256",selector_bank_sha);
    }
    std::cout<<"preflight_pass receipt="<<receipt_path
             <<" train="<<train_spec.name<<"@"<<train_bank_sha.substr(0,12)
             <<" selector="<<selector_spec.name<<"@"<<selector_bank_sha.substr(0,12)<<"\n";
}

struct TrainingOptions {
    BankSpec spec;
    std::string bank_path;
    std::string preflight_path;
    uint32_t rehearsal_weight=0;
    uint32_t perception_support=0;
    uint32_t seed=20261004,epochs=2,environments=128,horizon=32,eval_every=50;
    float log_std=-1.0f,learning_rate=0.0001f,entropy=0.001f,speed=1.5f,time_cost=0.2f;
    std::string eval_spec="dev-a";
    BankSpec eval_bank_spec;
};

static int command_train(int argc,char** argv) {
    require(argc>=5,"local-train ROLLOUTS CHECKPOINT WARMSTART");
    const uint32_t rollouts=uint32_t(std::stoul(argv[2]));
    const std::string checkpoint=argv[3],warmstart=argv[4];
    TrainingOptions options;
    options.spec=spec_by_name("source");
    options.eval_bank_spec=spec_by_name("dev-a");
    for(int index=5;index<argc;index++) {
        const std::string option=argv[index];
        if(option=="--spec")options.spec=spec_by_name(option_value(argc,argv,index,option));
        else if(option=="--bank")options.bank_path=option_value(argc,argv,index,option);
        else if(option=="--preflight")options.preflight_path=option_value(argc,argv,index,option);
        else if(option=="--rehearsal-weight")options.rehearsal_weight=option_uint(argc,argv,index,option);
        else if(option=="--perception-support")options.perception_support=option_uint(argc,argv,index,option);
        else if(option=="--seed")options.seed=option_uint(argc,argv,index,option);
        else if(option=="--epochs")options.epochs=option_uint(argc,argv,index,option);
        else if(option=="--envs")options.environments=option_uint(argc,argv,index,option);
        else if(option=="--horizon")options.horizon=option_uint(argc,argv,index,option);
        else if(option=="--logstd")options.log_std=option_float(argc,argv,index,option);
        else if(option=="--learning-rate")options.learning_rate=option_float(argc,argv,index,option);
        else if(option=="--entropy")options.entropy=option_float(argc,argv,index,option);
        else if(option=="--speed")options.speed=option_float(argc,argv,index,option);
        else if(option=="--time-cost")options.time_cost=option_float(argc,argv,index,option);
        else if(option=="--eval-spec")options.eval_spec=option_value(argc,argv,index,option);
        else if(option=="--eval-every")options.eval_every=option_uint(argc,argv,index,option);
        else throw std::runtime_error("unknown local-train option "+option);
    }
    require(rollouts>0&&fixed_ppo::actor_obs_dim==184,"local training requires positive rollouts and the 184D guided actor");
    require(options.spec.environments==options.environments,
            "bank environments must equal --envs; rebuild the bank with matching --tasks");
    require(std::isfinite(options.log_std)&&options.log_std>=-2&&options.log_std<=0.5f,
            "warmstart log std outside supported range");

    // Bank: assembled in memory first. Nothing is written until the resume
    // contract has been validated, so a mismatched invocation cannot overwrite
    // the bank files or the sidecar of a completed run.
    BankControl control{options.spec.period,uint32_t(options.spec.environments*options.spec.period)};
    std::vector<BankEntry> bank;
    std::string bank_sha;
    std::ostringstream bank_csv;
    if(options.bank_path.empty()) {
        bank_csv<<std::setprecision(9);
        write_bank_csv_header(bank_csv,options.spec.grounded_rules);
        bank=build_bank(options.spec,&bank_csv);
        bank_sha=ppo_safeguard_sha256(bank.data(),bank.size()*sizeof(BankEntry));
    } else {
        bank=read_bank(options.bank_path,control,bank_sha);
        require(size_t(control.count)==size_t(options.environments)*control.period,
                "bank file does not match --envs/period");
    }

    BankControl eval_control{};
    std::vector<BankEntry> eval_bank;
    std::vector<GroundedLabel> eval_labels;
    std::string eval_bank_sha="none";
    std::ostringstream eval_bank_csv;
    if(options.eval_spec!="none") {
        options.eval_bank_spec=spec_by_name(options.eval_spec);
        const BankSpec& eval_spec=options.eval_bank_spec;
        eval_bank_csv<<std::setprecision(9);
        write_bank_csv_header(eval_bank_csv,eval_spec.grounded_rules);
        eval_bank=build_bank(eval_spec,&eval_bank_csv,nullptr,
                             eval_spec.grounded_rules?&eval_labels:nullptr);
        eval_control.period=eval_spec.period;
        eval_control.count=uint32_t(eval_bank.size());
        eval_bank_sha=ppo_safeguard_sha256(eval_bank.data(),eval_bank.size()*sizeof(BankEntry));
    }

    TrainContract contract;
    contract.sensor_profile=NAV_SENSOR_ACTIVE_NAME;
    contract.actor_obs_dim=uint32_t(fixed_ppo::actor_obs_dim);
    contract.runner_sha256=contract_runner_sha256();
    contract.core_sha256=contract_core_sha256();
    contract.warmstart_sha256=challenge_evaluation::sha256_file(warmstart);
    contract.bank_spec=options.spec.name;
    contract.bank_seed=options.spec.seed;
    contract.bank_entry_sha256=bank_sha;
    contract.bank_entries=uint32_t(bank.size());
    contract.bank_period=control.period;
    contract.eval_spec=options.eval_spec;
    contract.eval_entry_sha256=eval_bank_sha;
    contract.eval_entries=uint32_t(eval_bank.size());
    contract.eval_period=eval_control.period;
    contract.environments=options.environments;
    contract.horizon=options.horizon;
    contract.epochs=options.epochs;
    contract.eval_every=options.eval_every;
    contract.max_steps=400;
    contract.learning_rate=options.learning_rate;
    contract.entropy_coef=options.entropy;
    contract.risk_coef=0.0f;
    contract.value_coef=0.5f;
    contract.speed=options.speed;
    contract.time_cost=options.time_cost;
    contract.warmstart_log_std=options.log_std;
    contract.rehearsal_weight=options.rehearsal_weight;
    contract.perception_support=options.perception_support;
    require(options.perception_support<=1,"--perception-support supports 0 or 1");
    if(options.perception_support)
        std::cout<<"perception_support=1 module=perception_support.hpp kernel=ps_override_prior "
                 <<"prior_channel=measured-rings-only unknown=not-occupied speed_cap="<<"0.5"<<"\n";

    // Rehearsal weighting is a schedule-only switch (decision-revision-g3):
    // weight0 = legacy round-robin, byte-identical to every prior run.
    require(options.rehearsal_weight<=1,"--rehearsal-weight supports 0 or 1");
    if(options.rehearsal_weight) {
        require(control.period==16,
                "--rehearsal-weight1 requires the16-slot grounded-mix bank");
        control.rehearsal_weight=options.rehearsal_weight;
        std::cout<<"rehearsal_schedule weight=1 cycle=24 source_episodes=16/24 grounded_episodes=8/24 "
                 <<"slots=0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 "
                 <<"floor=every grounded slot once per cycle\n";
    }

    // Fail-closed preflight gate: refuses grounded training/selector runs
    // before a single output file exists unless the independent contract
    // checker's PASS receipt binds this runner + these exact bank bytes.
    {
        const bool grounded_run=options.spec.grounded_rules||options.eval_bank_spec.grounded_rules;
        if(grounded_run)
            require(options.eval_spec!="none",
                    "grounded training requires a frozen selector bank (--eval-spec)");
        if(grounded_run||!options.preflight_path.empty())
            require_grounded_preflight(options.preflight_path,options.spec,options.eval_bank_spec,
                                       bank_sha,eval_bank_sha);
    }

    const bool resume=std::filesystem::exists(checkpoint);
    if(resume) {
        require_contract_matches(checkpoint+".train.json",contract);
        // Extra integrity gate: a bank file left on disk must be the same bank
        // the contract names, never a silently replaced distribution.
        if(options.bank_path.empty()&&std::filesystem::exists(checkpoint+".bank.bin")) {
            BankControl file_control{};std::string file_sha;
            const auto file_bank=read_bank(checkpoint+".bank.bin",file_control,file_sha);
            require(file_sha==bank_sha&&file_bank.size()==bank.size(),
                    "on-disk bank file does not match the resume contract; refusing to resume");
            std::cout<<"resume_bank_file PASS "<<checkpoint<<".bank.bin sha="<<file_sha.substr(0,12)<<"\n";
        }
    } else {
        if(options.bank_path.empty()) {
            std::ofstream csv(checkpoint+".bank.csv");require(bool(csv),"cannot write bank CSV");
            csv<<bank_csv.str();csv.flush();csv.close();
            write_bank(checkpoint+".bank.bin",options.spec,bank);
        }
        if(!eval_bank.empty()) {
            std::ofstream csv(checkpoint+".evalbank.csv");require(bool(csv),"cannot write eval bank CSV");
            csv<<eval_bank_csv.str();csv.flush();csv.close();
        }
        write_contract(checkpoint+".train.json",contract);
        std::cout<<"train_contract "<<checkpoint<<".train.json profile="<<contract.sensor_profile
                 <<" bank="<<contract.bank_spec<<"@"<<contract.bank_entry_sha256.substr(0,12)
                 <<" eval="<<contract.eval_spec<<"@"<<contract.eval_entry_sha256.substr(0,12)
                 <<" core="<<contract.core_sha256.substr(0,12)
                 <<" runner="<<contract.runner_sha256.substr(0,12)<<"\n";
    }

    Metal metal;
    metal.compile(ps_source()+PPO_TRAINER_MSL+kWaypointKernels);
    auto apply_pipeline=metal.pipeline("waypoint_task_apply");

    SimConfig config;
    config.n=options.environments;config.mode=22;config.family=0;config.seed=options.seed;
    config.speed=options.speed;config.distance=4;config.max_steps=400;config.geometry_memory=1;
    config.entropy_coef=options.entropy;config.learning_rate=options.learning_rate;
    Sim sim(metal,config,options.horizon);
    LocalRun run=make_local_run(sim,bank,control,apply_pipeline,options.time_cost,
                                options.perception_support!=0);
    PPOTrainer trainer(sim,options.epochs);

    // Explicit warmstart: actor AND critic parameters, fresh Adam moments,
    // fresh log std. Critic/optimizer reset state is stated in the contract.
    if(resume) {
        trainer.load_checkpoint(checkpoint,config.family,options.horizon,options.environments,options.seed);
    } else {
        navigation_training::load_actor(sim,warmstart,true);
        for(uint32_t axis=0;axis<4;axis++)
            ((float*)sim.actor.contents)[fixed_ppo::actor_log_std_offset+axis]=options.log_std;
        trainer.recapture_anchor_reference();
    }

    // A fresh environment snapshot must satisfy the reset contract before any
    // optimizer step. On resume the simulator continues mid-episode from the
    // checkpoint, so the check is only meaningful for a fresh start.
    if(resume) {
        std::cout<<"resume_skips_reset_probe mid_episode_state_restored rollouts="<<trainer.completed_rollouts<<"\n";
    } else {
        auto probe=[metal.queue commandBuffer];
        local_probe(run,probe);
        metal.finish(probe);
        verify_reset(run,bank,control,true);
    }

    const std::filesystem::path checkpoint_path(checkpoint);
    if(checkpoint_path.has_parent_path())std::filesystem::create_directories(checkpoint_path.parent_path());
    std::ofstream history(checkpoint+".history.csv",resume?std::ios::app:std::ios::trunc);
    require(bool(history),"cannot write training history");
    std::ofstream diag(checkpoint+".diag.csv",resume?std::ios::app:std::ios::trunc);
    require(bool(diag),"cannot write training diagnostics");
    if(!resume) {
        history<<"rollout,transitions,wall_s,collect_gpu_s,update_gpu_s,policy_loss,value_loss,entropy,ratio,"
                 "dev_success,dev_collision,dev_timeout,dev_arrival_s,dev_min_clearance\n";
        diag<<"rollout,transitions,gpu_s,rew_mean,rew_std,term_n,term_mean,term_pos,adv_std,ret_mean,"
              "val_mean,pol_loss,val_loss,entropy,ratio,episodes,ep_success,ep_collision,ep_timeout,"
              "ep_path_m,ep_time_s,ep_speed_mps,inflight_clear_min,logstd0,logstd1,logstd2,logstd3,"
              "actor_drift_l2\n";
    }
    BankScore best;
    best.success=-1;
    if(resume&&std::filesystem::exists(checkpoint+".best")&&!eval_bank.empty()) {
        Sim selected(metal,config,options.horizon);
        navigation_training::load_actor(selected,checkpoint+".best",false);
        best=evaluate_bank(metal,(const float*)selected.actor.contents,&options.eval_bank_spec,eval_bank,eval_control,
                           17,700001,options.speed,"",options.eval_spec,options.environments,
                           400,0,0,eval_labels.empty()?nullptr:&eval_labels,
                           options.perception_support!=0);
    }

    std::vector<SimRun> previous_runs(options.environments);
    const double started=seconds();
    const uint32_t finish=trainer.completed_rollouts+rollouts;
    std::cout<<"local_train bank="<<options.spec.name<<" environments="<<options.environments
             <<" horizon="<<options.horizon<<" epochs="<<options.epochs
             <<" lr="<<options.learning_rate<<" entropy="<<options.entropy
             <<" risk="<<config.risk_coef<<" time_cost="<<options.time_cost
             <<" rehearsal_weight="<<options.rehearsal_weight
             <<" perception_support="<<options.perception_support
             <<" start="<<trainer.completed_rollouts<<" finish="<<finish
             <<" sensor_profile="<<NAV_SENSOR_ACTIVE_NAME<<"\n";
    for(uint32_t rollout=trainer.completed_rollouts;rollout<finish;rollout++) {@autoreleasepool {
        std::memcpy(previous_runs.data(),sim.runs.contents,options.environments*sizeof(SimRun));
        auto commands=[metal.queue commandBuffer];
        local_collect(run,commands,options.horizon);
        const double collect_gpu=metal.finish(commands);
        auto update=[metal.queue commandBuffer];
        trainer.rollout_update(update,rollout);
        const double update_gpu=metal.finish(update);
        trainer.completed_rollouts=rollout+1;

        const size_t rows=size_t(options.environments)*options.horizon;
        const float* reward=(const float*)sim.rewards.contents;
        const float* advantage=(const float*)sim.advantages.contents;
        const float* returns=(const float*)sim.returns.contents;
        const float* values=(const float*)sim.values.contents;
        const auto* terminated=(const uint8_t*)sim.terminated.contents;
        const auto moment=[&](const float* buffer,size_t count){
            double mean=0;for(size_t index=0;index<count;index++){require(std::isfinite(buffer[index]),"non-finite rollout buffer");mean+=buffer[index];}
            mean/=double(count?count:1);double var=0;
            for(size_t index=0;index<count;index++){const double d=double(buffer[index])-mean;var+=d*d;}
            return std::array<double,2>{mean,count?std::sqrt(var/double(count)):0.0};
        };
        const auto reward_moment=moment(reward,rows);
        const auto advantage_moment=moment(advantage,rows);
        const auto value_moment=moment(values,rows);
        size_t terminal_count=0,terminal_positive=0;double terminal_sum=0;
        for(size_t index=0;index<rows;index++)if(terminated[index]){terminal_count++;terminal_sum+=reward[index];if(reward[index]>0)terminal_positive++;}
        const auto* runs_now=(const SimRun*)sim.runs.contents;
        uint64_t episodes=0,successes=0,collisions=0,timeouts=0;double path=0,elapsed=0;float clearance_min=12;
        for(uint32_t env=0;env<options.environments;env++) {
            const SimRun& now=runs_now[env];const SimRun& before=previous_runs[env];
            require(now.episodes>=before.episodes&&now.successes>=before.successes&&
                    now.collisions>=before.collisions&&now.timeouts>=before.timeouts,"episode counters went backwards");
            episodes+=now.episodes-before.episodes;successes+=now.successes-before.successes;
            collisions+=now.collisions-before.collisions;timeouts+=now.timeouts-before.timeouts;
            path+=double(now.total_path)-before.total_path;elapsed+=double(now.total_elapsed)-before.total_elapsed;
            clearance_min=std::min(clearance_min,now.min_clearance);
        }
        const float* metric=(const float*)trainer.metric_mean.contents;
        const float* actor_params=(const float*)sim.actor.contents;
        diag<<(rollout+1)<<','<<uint64_t(rollout+1)*rows<<','<<(collect_gpu+update_gpu)
            <<','<<reward_moment[0]<<','<<reward_moment[1]<<','<<terminal_count
            <<','<<(terminal_count?terminal_sum/double(terminal_count):0.0)<<','<<terminal_positive
            <<','<<advantage_moment[1]<<','<<moment(returns,rows)[0]<<','<<value_moment[0]
            <<','<<metric[0]<<','<<metric[1]<<','<<metric[2]<<','<<metric[3]
            <<','<<episodes<<','<<(episodes?double(successes)/episodes:0)<<','<<(episodes?double(collisions)/episodes:0)
            <<','<<(episodes?double(timeouts)/episodes:0)<<','<<(episodes?path/double(episodes):0)
            <<','<<(episodes?elapsed/double(episodes):0)<<','<<(elapsed?path/elapsed:0)<<','<<clearance_min;
        for(uint32_t axis=0;axis<4;axis++)diag<<','<<actor_params[fixed_ppo::actor_log_std_offset+axis];
        diag<<','<<trainer.actor_drift_l2()<<'\n';
        diag.flush();
        if((rollout+1)%options.eval_every==0||rollout+1==finish) {
            trainer.save_checkpoint(checkpoint,config.family,options.seed,rollout+1);
            BankScore score;
            if(!eval_bank.empty())
                score=evaluate_bank(metal,(const float*)sim.actor.contents,&options.eval_bank_spec,eval_bank,eval_control,
                                    17,700001,options.speed,checkpoint+".dev.csv",options.eval_spec,options.environments,
                                    400,0,0,eval_labels.empty()?nullptr:&eval_labels,
                                    options.perception_support!=0);
            history<<(rollout+1)<<','<<uint64_t(rollout+1)*rows<<','<<seconds()-started
                   <<','<<collect_gpu<<','<<update_gpu<<','<<metric[0]<<','<<metric[1]<<','<<metric[2]<<','<<metric[3]
                   <<','<<score.success<<','<<score.collision<<','<<score.timeout<<','<<score.mean_time_s
                   <<','<<score.min_clearance_m<<'\n';
            history.flush();
            const bool better=score.success>best.success||
                (score.success==best.success&&score.collision<best.collision)||
                (score.success==best.success&&score.collision==best.collision&&
                 (best.success<=0||score.mean_time_s<best.mean_time_s));
            if(better&&!eval_bank.empty()) {
                best=score;trainer.save_checkpoint(checkpoint+".best",config.family,options.seed,rollout+1);
                if(std::filesystem::exists(checkpoint+".dev.csv"))
                    std::filesystem::copy_file(checkpoint+".dev.csv",checkpoint+".best-dev.csv",
                                               std::filesystem::copy_options::overwrite_existing);
            }
            std::cout<<"local_train rollout="<<(rollout+1)<<" wall_s="<<seconds()-started
                     <<" collect_gpu_s="<<collect_gpu<<" update_gpu_s="<<update_gpu
                     <<" policy_loss="<<metric[0]<<" value_loss="<<metric[1]
                     <<" ep_success="<<(episodes?double(successes)/episodes:0)
                     <<" dev_success="<<score.success<<" dev_collision="<<score.collision
                     <<" dev_arrival_s="<<score.mean_time_s<<" clear_min="<<clearance_min<<"\n";
        }
    }}
    std::cout<<"local_train_done rollouts="<<finish<<" wall_s="<<seconds()-started<<"\n";
    return 0;
}

static int command_eval(int argc,char** argv) {
    require(argc>=4,"local-eval CHECKPOINT OUT_CSV");
    const std::string checkpoint=argv[2],output=argv[3];
    std::string spec_name="dev-a",bank_path;
    uint32_t mode=17,seed=700001,environments=128,max_steps=400,sensor_delay=0,command_delay=0;
    float speed=1.5f;
    bool selector=false;
    bool support_flag=false;   // perception-support override for this eval
    for(int index=4;index<argc;index++) {
        const std::string option=argv[index];
        if(option=="--spec")spec_name=option_value(argc,argv,index,option);
        else if(option=="--bank")bank_path=option_value(argc,argv,index,option);
        else if(option=="--mode")mode=option_uint(argc,argv,index,option);
        else if(option=="--seed")seed=option_uint(argc,argv,index,option);
        else if(option=="--envs")environments=option_uint(argc,argv,index,option);
        else if(option=="--max-steps")max_steps=option_uint(argc,argv,index,option);
        else if(option=="--speed")speed=option_float(argc,argv,index,option);
        else if(option=="--sensor-delay")sensor_delay=option_uint(argc,argv,index,option);
        else if(option=="--command-delay")command_delay=option_uint(argc,argv,index,option);
        else if(option=="--selector")selector=option_uint(argc,argv,index,option)!=0;
        else if(option=="--perception-support")support_flag=option_uint(argc,argv,index,option)!=0;
        else throw std::runtime_error("unknown local-eval option "+option);
    }
    Metal metal;
    metal.compile(ps_source()+PPO_TRAINER_MSL+kWaypointKernels);
    BankControl control{};std::vector<BankEntry> bank;BankSpec spec;
    std::vector<GroundedLabel> eval_labels;
    if(bank_path.empty()) {
        spec=spec_by_name(spec_name);
        bank=build_bank(spec,nullptr,nullptr,spec.grounded_rules?&eval_labels:nullptr);
        control.period=spec.period;control.count=uint32_t(bank.size());
    } else {
        std::string unused_hash;
        bank=read_bank(bank_path,control,unused_hash);
        spec.name=bank_path;
    }
    require(control.count==size_t(environments)*control.period,"evaluation bank does not match --envs");
    require(fixed_ppo::actor_obs_dim==184,"local evaluation requires the 184D guided actor");
    std::ifstream file(checkpoint,std::ios::binary);
    const PpoCheckpointHeader header=read_checkpoint_header(file);
    require(header.actor_count==fixed_ppo::actor_param_count,"evaluation checkpoint dimensions");
    std::vector<float> actor(header.actor_count);
    file.read(reinterpret_cast<char*>(actor.data()),std::streamsize(actor.size()*sizeof(float)));
    require(bool(file),"evaluation checkpoint actor read failed");
    if(selector) {
        // Selection run on the DEV split at the deployment inference mode.
        require(mode==17,"selector evaluation must use the deployed mode 17");
    }
    evaluate_bank(metal,actor.data(),&spec,bank,control,mode,seed,speed,output,spec.name,environments,max_steps,
                  sensor_delay,command_delay,eval_labels.empty()?nullptr:&eval_labels,support_flag);
    return 0;
}

static int command_trace(int argc,char** argv) {
    require(argc>=4,"local-trace CHECKPOINT OUT_CSV");
    const std::string checkpoint=argv[2],output=argv[3];
    std::string spec_name="dev-a",bank_path;
    uint32_t mode=17,seed=700001,environments=4,max_steps=400;
    float speed=1.5f;
    for(int index=4;index<argc;index++) {
        const std::string option=argv[index];
        if(option=="--spec")spec_name=option_value(argc,argv,index,option);
        else if(option=="--bank")bank_path=option_value(argc,argv,index,option);
        else if(option=="--mode")mode=option_uint(argc,argv,index,option);
        else if(option=="--seed")seed=option_uint(argc,argv,index,option);
        else if(option=="--envs")environments=option_uint(argc,argv,index,option);
        else if(option=="--max-steps")max_steps=option_uint(argc,argv,index,option);
        else if(option=="--speed")speed=option_float(argc,argv,index,option);
        else throw std::runtime_error("unknown local-trace option "+option);
    }
    Metal metal;
    metal.compile(ps_source()+PPO_TRAINER_MSL+kWaypointKernels);
    BankControl control{};std::vector<BankEntry> bank;BankSpec spec;
    if(bank_path.empty()) {
        spec=spec_by_name(spec_name);
        bank=build_bank(spec,nullptr);
        control.period=spec.period;control.count=uint32_t(bank.size());
    } else {
        std::string unused_hash;
        bank=read_bank(bank_path,control,unused_hash);
        spec.name=bank_path;
    }
    require(size_t(control.count)>=size_t(environments)*control.period,"trace bank is too small");
    control.count=uint32_t(size_t(environments)*control.period);
    SimConfig config;
    config.n=environments;config.mode=mode;config.family=0;config.eval=1;config.seed=seed;
    config.speed=speed;config.distance=4;config.max_steps=max_steps;config.geometry_memory=1;
    Sim sim(metal,config,32);
    std::ifstream file(checkpoint,std::ios::binary);
    const PpoCheckpointHeader header=read_checkpoint_header(file);
    require(header.actor_count==fixed_ppo::actor_param_count,"trace checkpoint dimensions");
    file.read((char*)sim.actor.contents,sim.actor.length);
    require(bool(file),"trace checkpoint actor read failed");
    LocalRun run=make_local_run(sim,bank,control,metal.pipeline("waypoint_task_apply"));
    const std::filesystem::path destination(output);
    if(destination.has_parent_path())std::filesystem::create_directories(destination.parent_path());
    std::ofstream trace(output);
    require(bool(trace),"cannot write trace CSV "+output);
    trace<<std::setprecision(9);
    trace<<"tick,env,family,scene_seed,time_s,x,y,z,vx,vy,vz,yaw,goal_x,goal_y,goal_z,goal_distance_m,"
           "clearance_m,steps,episodes,terminated\n";
    auto probe=[metal.queue commandBuffer];
    local_probe(run,probe);
    metal.finish(probe);
    verify_reset(run,bank,control,true);
    const uint32_t tick_base=sim.cfg.tick;
    for(uint32_t step=0;step<max_steps;step++) {
        auto commands=[metal.queue commandBuffer];
        local_tick(run,commands,tick_base+step,step==0);
        metal.finish(commands);
        const auto* states=(const RLPhysicsState*)sim.states.contents;
        const auto* runs=(const SimRun*)sim.runs.contents;
        for(uint32_t env=0;env<environments;env++) {
            const BankEntry& entry=bank[size_t(env)*control.period];
            const RLPhysicsState& state=states[env];
            float distance=0;
            for(int axis=0;axis<3;axis++){const float delta=entry.goal_position[axis]-state.position[axis];distance+=delta*delta;}
            const float clearance=wclearance(((const WWorld*)sim.worlds.contents)[env],
                                             wv(state.position[0],state.position[1],state.position[2]),runs[env].elapsed);
            trace<<step<<','<<env<<','<<entry.family<<','<<entry.scene_seed<<','<<runs[env].elapsed
                 <<','<<state.position[0]<<','<<state.position[1]<<','<<state.position[2]
                 <<','<<state.linear_velocity[0]<<','<<state.linear_velocity[1]<<','<<state.linear_velocity[2]
                 <<','<<runs[env].yaw;
            for(int axis=0;axis<3;axis++)trace<<','<<entry.goal_position[axis];
            trace<<','<<std::sqrt(distance)<<','<<clearance<<','<<runs[env].steps<<','<<runs[env].episodes<<','
                 <<(runs[env].episodes>0?1:0)<<'\n';
        }
    }
    std::cout<<"trace_write "<<output<<" envs="<<environments<<" steps="<<max_steps<<"\n";
    return 0;
}

// ------------------------------------------- observe-before-motion teacher
//
// First experiment (docs/LOCAL_DISTRIBUTION_AUDIT.md §12): a hybrid controller
// on TRAIN flights. Where the goal body bearing exceeds 35 deg while the goal
// is farther than 0.6 m (root's predeclared view-conditioning constants), the
// host overrides that tick to zero body velocity and full-rate yaw toward the
// goal; every other tick flies the checkpoint's own mode-17 command. Labels
// live in the deployed command space and are pure functions of the stored
// observation (goal body direction/distance channels plus the actor's own
// action map) — no witness, scene seed or task truth is read for any label.
// Rows are kept only from successful actual-RAPTOR episodes; failed episodes
// are counted and dropped.
struct Demonstration {
    std::array<float,fixed_ppo::actor_obs_dim> observation;
    std::array<float,4> command;
};
static_assert(sizeof(Demonstration)==(fixed_ppo::actor_obs_dim+4)*sizeof(float),"demonstration row layout");

static constexpr float kBrakeBearingRad=0.610865f; // 35 deg, inherited, no sweep
static constexpr float kBrakeDistanceM=0.6f;
static constexpr float kBrakeYawGain=4.0f;

static int command_teacher(int argc,char** argv) {
    require(argc>=4,"local-teacher CHECKPOINT DATASET [--spec NAME] [--max-ticks N]");
    const std::string checkpoint=argv[2],dataset_path=argv[3];
    std::string spec_name="source";uint32_t max_ticks=0;
    for(int index=4;index<argc;index++) {
        const std::string option=argv[index];
        if(option=="--spec")spec_name=option_value(argc,argv,index,option);
        else if(option=="--max-ticks")max_ticks=option_uint(argc,argv,index,option);
        else throw std::runtime_error("unknown local-teacher option "+option);
    }
    require(fixed_ppo::actor_obs_dim==184,"teacher requires the 184D guided actor");
    const BankSpec spec=spec_by_name(spec_name);
    Metal metal;
    metal.compile(ps_source()+PPO_TRAINER_MSL+kWaypointKernels);
    auto apply=metal.pipeline("waypoint_task_apply");
    BankControl control{};const auto bank=build_bank(spec,nullptr);
    control.period=spec.period;control.count=uint32_t(bank.size());
    SimConfig config;
    config.n=spec.environments;config.mode=17;config.eval=0;config.family=0;
    config.seed=spec.seed;config.speed=1.5f;config.distance=4;config.max_steps=400;
    config.geometry_memory=1;
    Sim sim(metal,config,32);
    navigation_training::load_actor(sim,checkpoint,false);
    LocalRun run=make_local_run(sim,bank,control,apply,1.0f);
    auto probe=[metal.queue commandBuffer];
    local_probe(run,probe);
    metal.finish(probe);
    verify_reset(run,bank,control,true);

    const uint32_t horizon=sim.horizon,envs=spec.environments,target=spec.period;
    const uint32_t context=fixed_ppo::context_offset;
    if(!max_ticks)max_ticks=spec.period*(uint32_t(config.max_steps)+80);
    const float yaw_dt=static_cast<const RLPhysicsParams*>(sim.physics.contents)->dt*float(config.substeps)*0.5f;
    require(std::fabs(yaw_dt-0.025f)<1e-6f,"teacher yaw rate contract");
    const std::string bank_sha=ppo_safeguard_sha256(bank.data(),bank.size()*sizeof(BankEntry));
    const std::filesystem::path destination(dataset_path);
    if(destination.has_parent_path())std::filesystem::create_directories(destination.parent_path());
    std::ofstream episodes_log(dataset_path+".episodes.csv");
    require(bool(episodes_log),"cannot write "+dataset_path+".episodes.csv");
    episodes_log<<"env,slot,family,scene_seed,route_class,success,ticks,rows\n";

    std::vector<std::vector<Demonstration>> pending(envs);
    std::vector<bool> brake(envs,false);
    std::vector<float> before_yaw(envs),bearing(envs),yaw_command(envs);
    std::vector<uint32_t> prev_episodes(envs,0),prev_success(envs,0);
    std::vector<Demonstration> retained;
    retained.reserve(size_t(envs)*size_t(spec.period)*120);
    uint64_t episodes_done=0,episodes_success=0,episodes_overrun=0,brake_ticks=0,follow_ticks=0;
    const double started=seconds();
    bool covered=false;uint32_t tick=0;
    for(;tick<max_ticks;tick++) {
        covered=true;
        for(uint32_t env=0;env<envs;env++)if(prev_episodes[env]<target){covered=false;break;}
        if(covered)break;
        const auto c=sim.configs[tick%horizon];
        const size_t row_offset=size_t(tick%horizon)*envs;

        // Sense and act, in the deployed order, so the host override lands
        // between act() (which consumed the observation and mutated yaw) and
        // advance() (which executes the command this tick).
        auto sense=[metal.queue commandBuffer];
        sim.m.dispatch(sense,sim.depth_p,envs*320,{sim.states,sim.runs,sim.worlds,sim.sensors,sim.physics,c,sim.poses});
        if(sim.cfg.geometry_memory) {
            sim.m.dispatch(sense,sim.memory_points_p,envs*640,
                {sim.states,sim.runs,sim.sensors,sim.poses,sim.memory_points,sim.physics,c});
            sim.m.dispatch(sense,sim.memory_candidates_p,envs*85,
                {sim.states,sim.worlds,sim.memory_points,sim.memory_clearances,c});
        }
        sim.m.dispatch(sense,sim.observe_p,envs,
            {sim.states,sim.runs,sim.worlds,sim.sensors,sim.obs,sim.co,sim.physics,c,sim.poses,sim.memory_clearances},64);
        encode(sim.m,sense,sim.simd_actor?"ppo_actor_forward_simd_fused":"ppo_actor_forward",
               sim.simd_actor?((size_t(envs)+7)/8)*256:envs,
               {{sim.obs,row_offset*fixed_ppo::actor_obs_dim*4},{sim.actor,0},{sim.actor_workspace,0},
                {sim.actions,row_offset*fixed_ppo::action_dim*4},{sim.env_count,0}},sim.simd_actor?256:64);
        metal.finish(sense);

        const float* obs=(const float*)sim.obs.contents;
        auto* runs_now=(SimRun*)sim.runs.contents;
        for(uint32_t env=0;env<envs;env++) {
            before_yaw[env]=runs_now[env].yaw;
            const float* row=obs+(row_offset+env)*fixed_ppo::actor_obs_dim;
            bearing[env]=std::atan2(row[context+1],row[context]);
            const float distance=row[context+3]*10.0f;
            brake[env]=std::fabs(bearing[env])>kBrakeBearingRad&&distance>kBrakeDistanceM;
            yaw_command[env]=brake[env]
                ?std::max(-1.0f,std::min(1.0f,kBrakeYawGain*bearing[env])):0.0f;
        }
        auto act_cb=[metal.queue commandBuffer];
        sim.m.dispatch(act_cb,sim.act_p,envs,
            {sim.states,sim.runs,sim.worlds,sim.obs,sim.co,sim.actor,sim.critic,sim.actions,sim.logp,
             sim.values,sim.commands,sim.physics,c},64);
        metal.finish(act_cb);

        // Labels: brake rows are the override itself (pure obs rule); follow
        // rows are the executed mode-17 command, spherical-capped like the
        // kernel's deployed command space.
        runs_now=(SimRun*)sim.runs.contents;
        for(uint32_t env=0;env<envs;env++) {
            if(prev_episodes[env]>=target)continue;
            Demonstration row{};
            const float* observation=obs+(row_offset+env)*fixed_ppo::actor_obs_dim;
            std::copy(observation,observation+fixed_ppo::actor_obs_dim,row.observation.begin());
            if(brake[env]) {
                const float yaw=yaw_command[env];
                row.command={0,0,0,yaw};
                SimRun& current=runs_now[env];
                for(int axis=0;axis<3;axis++) {
                    current.previous_nav[axis]=0;
                    current.desired_velocity[axis]=0;
                }
                current.previous_nav[3]=yaw;
                current.yaw=before_yaw[env]+yaw*yaw_dt;
                float* ring=(float*)sim.commands.contents+(size_t(env)*8+current.steps%8)*4;
                ring[0]=ring[1]=ring[2]=0;ring[3]=yaw;
                brake_ticks++;
            } else {
                const float* previous=runs_now[env].previous_nav;
                const float magnitude=std::sqrt(previous[0]*previous[0]+previous[1]*previous[1]+previous[2]*previous[2]);
                const float scale=1.0f/std::max(1.0f,magnitude);
                row.command={previous[0]*scale,previous[1]*scale,previous[2]*scale,previous[3]};
                follow_ticks++;
            }
            pending[env].push_back(row);
        }
        auto move_cb=[metal.queue commandBuffer];
        sim.m.dispatch(move_cb,sim.advance_p,envs,
            {sim.states,sim.runs,sim.worlds,sim.sensors,sim.commands,sim.raptor,sim.critic,sim.rewards,
             sim.next_values,sim.terminated,sim.truncated,sim.physics,c,sim.bank_worlds,sim.bank_schedule,
             sim.bank_control,sim.bank_active_ids,sim.bank_transition_ids,sim.environment_physics,
             sim.runtime_control,sim.task_states,sim.task_control,sim.potential_fields,sim.potential_spec,
             sim.potential_control},64);
        sim.m.dispatch(move_cb,run.apply_pipeline,envs,
            {sim.states,sim.runs,sim.worlds,sim.task_states,run.bank,run.bank_control,sim.task_control,c},64);
        metal.finish(move_cb);

        runs_now=(SimRun*)sim.runs.contents;
        for(uint32_t env=0;env<envs;env++) {
            if(runs_now[env].episodes==prev_episodes[env])continue;
            require(runs_now[env].episodes==prev_episodes[env]+1,"teacher expects one terminal per tick");
            const bool in_cycle=prev_episodes[env]<target;
            if(in_cycle) {
                const bool success=runs_now[env].successes>prev_success[env];
                const uint32_t slot=prev_episodes[env]%spec.period;
                const BankEntry& entry=bank[size_t(env)*spec.period+slot];
                episodes_log<<env<<','<<slot<<','<<entry.family<<','<<entry.scene_seed<<','
                            <<entry.route_class<<','<<uint32_t(success)<<','
                            <<runs_now[env].steps<<','<<pending[env].size()<<'\n';
                episodes_done++;
                if(success) {
                    episodes_success++;
                    retained.insert(retained.end(),pending[env].begin(),pending[env].end());
                }
            } else episodes_overrun++;
            pending[env].clear();
            prev_episodes[env]=runs_now[env].episodes;
            prev_success[env]=runs_now[env].successes;
        }
        episodes_log.flush();
    }
    require(covered,"teacher did not cover every bank slot; raise --max-ticks");
    episodes_log.close();
    require(retained.size()>=2000,"teacher retained too few successful-flight rows");
    require(retained.size()<=400000,"teacher dataset exceeds the row budget");

    std::ofstream dataset(dataset_path,std::ios::binary);
    require(bool(dataset),"cannot write "+dataset_path);
    const uint64_t rows=uint64_t(retained.size());
    dataset.write(reinterpret_cast<const char*>(&rows),sizeof(rows));
    dataset.write(reinterpret_cast<const char*>(retained.data()),std::streamsize(rows*sizeof(Demonstration)));
    require(bool(dataset),"dataset write failed "+dataset_path);
    dataset.close();
    const std::string dataset_sha=challenge_evaluation::sha256_file(dataset_path);
    const std::string warmstart_sha=challenge_evaluation::sha256_file(checkpoint);
    std::ofstream summary(dataset_path+".summary.json");
    require(bool(summary),"cannot write "+dataset_path+".summary.json");
    summary<<"{\"schema\":\"local-teacher-v1\",\"spec\":\""<<spec.name<<"\",\"spec_seed\":"<<spec.seed
           <<",\"bank_entries\":"<<bank.size()<<",\"bank_entry_sha256\":\""<<bank_sha
           <<"\",\"checkpoint\":\""<<checkpoint<<"\",\"checkpoint_sha256\":\""<<warmstart_sha
           <<"\",\"mode\":17,\"time_cost_per_s\":1.0,\"speed_mps\":1.5,"
           <<"\"brake_bearing_rad\":"<<kBrakeBearingRad<<",\"brake_distance_m\":"<<kBrakeDistanceM
           <<",\"yaw_gain\":"<<kBrakeYawGain<<",\"ticks\":"<<tick
           <<",\"episodes\":"<<episodes_done<<",\"successes\":"<<episodes_success
           <<",\"overrun_episodes\":"<<episodes_overrun<<",\"brake_ticks\":"<<brake_ticks
           <<",\"follow_ticks\":"<<follow_ticks<<",\"retained_rows\":"<<rows
           <<",\"dataset_sha256\":\""<<dataset_sha<<"\",\"wall_s\":"<<seconds()-started<<"}\n";
    summary.close();
    std::cout<<"teacher_done rows="<<rows<<" episodes="<<episodes_done<<" successes="<<episodes_success
             <<" overrun="<<episodes_overrun<<" brake_ticks="<<brake_ticks<<" follow_ticks="<<follow_ticks
             <<" ticks="<<tick<<" dataset_sha256="<<dataset_sha<<"\n";
    return 0;
}

// ------------------------------------------------------- verified BC block
//
// The imitation kernel, its finite-difference gate and the Adam chain are
// copied verbatim from navigation_imitation.mm (owner's proven BC objective,
// self-test at lines 150-213 of that file) so this runner stays standalone.
// The kernel already models the deployed action map (guidance gate + spherical
// cap), so unreachable raw-mean labels cannot dominate the objective.
static const char* kImitationKernel=R"MSL(
kernel void imitation_gradient(device const float* obs [[buffer(0)]],device const float* means [[buffer(1)]],
    device const float* target [[buffer(2)]],device float* gradient [[buffer(3)]],device float* losses [[buffer(4)]],
    constant uint& count [[buffer(5)]],uint n [[thread_position_in_grid]]){
    if(n>=count)return;float open=0;for(uint k=0;k<20;k++)open+=obs[n*PPO_ACTOR_OBS+k]*12>2.5f;
    float gate=.25f+.75f*open/20;float3 prior=float3(obs[n*PPO_ACTOR_OBS+PPO_ACTOR_OBS-3],obs[n*PPO_ACTOR_OBS+PPO_ACTOR_OBS-2],obs[n*PPO_ACTOR_OBS+PPO_ACTOR_OBS-1]);
    float3 mean=float3(means[n*4],means[n*4+1],means[n*4+2]);float3 q=tanh(prior+gate*(mean-prior));
    float magnitude=length(q);float3 command=q/max(1.0f,magnitude);
    float3 desired=float3(target[n*4],target[n*4+1],target[n*4+2]);float3 delta=command-desired;
    float3 derivative=delta;
    if(magnitude>1)derivative=(delta-command*dot(command,delta))/magnitude;
    derivative*=gate*(1-q*q)/float(count);
    float yaw=tanh(gate*means[n*4+3]),dyaw=yaw-target[n*4+3];
    gradient[n*4]=derivative.x;gradient[n*4+1]=derivative.y;gradient[n*4+2]=derivative.z;
    gradient[n*4+3]=dyaw*gate*(1-yaw*yaw)/float(count);
    losses[n]=.5f*(dot(delta,delta)+dyaw*dyaw);
}
)MSL";

static void verify_imitation_gradient(Metal& metal,const std::string& output) {
    constexpr uint batch=3;
    constexpr uint obs_dim=uint(fixed_ppo::actor_obs_dim);
    constexpr uint hint=obs_dim-3;
    std::array<float,batch*obs_dim> observations{};
    std::array<float,batch*4> means{{.5f,-.25f,.1f,.2f,3.f,2.f,-2.f,.5f,-.7f,.3f,.2f,-.2f}};
    std::array<float,batch*4> targets{{.2f,.4f,0,0,-.2f,.5f,.1f,0,.4f,-.3f,.1f,0}};
    for(uint n=0;n<batch;n++) {
        for(uint k=0;k<20;k++)observations[n*obs_dim+k]=n==0?.1f:(n==1?1.f:(k%2?1.f:.1f));
        observations[n*obs_dim+hint]=.2f;observations[n*obs_dim+hint+1]=-.1f;
    }
    auto obs=metal.buffer(sizeof(observations),observations.data()),mu=metal.buffer(sizeof(means),means.data());
    auto target=metal.buffer(sizeof(targets),targets.data()),gradient=metal.buffer(sizeof(means)),losses=metal.buffer(batch*4);
    auto count=PPOTrainer::scalar(metal,batch);
    auto cb=[metal.queue commandBuffer];
    metal.dispatch(cb,metal.pipeline("imitation_gradient"),batch,{obs,mu,target,gradient,losses,count},64);
    metal.finish(cb);
    const auto loss=[&](uint n){
        float gate=n==0?.25f:(n==1?1.f:.625f);
        float q[3]{};
        for(uint k=0;k<3;k++){float prior=observations[n*obs_dim+hint+k];q[k]=std::tanh(prior+gate*(means[n*4+k]-prior));}
        const float magnitude=std::sqrt(q[0]*q[0]+q[1]*q[1]+q[2]*q[2]);
        const float scale=1/std::max(1.f,magnitude);
        float value=0;
        for(uint k=0;k<3;k++){float error=q[k]*scale-targets[n*4+k];value+=error*error;}
        float yaw=std::tanh(gate*means[n*4+3])-targets[n*4+3];
        return .5f*(value+yaw*yaw)/batch;
    };
    float worst=0;constexpr float epsilon=.001f;
    for(uint n=0;n<batch;n++)for(uint k=0;k<4;k++) {
        uint i=n*4+k;float value=means[i];means[i]=value+epsilon;float plus=loss(n);
        means[i]=value-epsilon;float minus=loss(n);means[i]=value;
        worst=std::max(worst,std::fabs((plus-minus)/(2*epsilon)-static_cast<float*>(gradient.contents)[i]));
    }
    require(worst<1e-4f,"imitation deployed-command gradient differs from finite differences");
    const std::filesystem::path path(output);
    if(path.has_parent_path())std::filesystem::create_directories(path.parent_path());
    std::ofstream report(output);
    require(bool(report),"cannot write gradient verification");
    report<<"{\"passed\":true,\"cases\":3,\"checked_components\":12,\"includes_spherical_cap\":true,\"finite_difference_max_error\":"<<worst<<"}\n";
    std::cout<<"imitation_gradient PASS max_error="<<worst<<std::endl;
}

static int command_bc(int argc,char** argv) {
    require(argc>=5,"local-bc DATASET OUT_CKPT WARMSTART [--updates N] [--batch N]");
    const std::string dataset_path=argv[2],output=argv[3],warmstart=argv[4];
    uint32_t updates=1000,batch=256;
    for(int index=5;index<argc;index++) {
        const std::string option=argv[index];
        if(option=="--updates")updates=option_uint(argc,argv,index,option);
        else if(option=="--batch")batch=option_uint(argc,argv,index,option);
        else throw std::runtime_error("unknown local-bc option "+option);
    }
    require(updates>=2&&updates<=5000,"local-bc updates budget 2..5000");
    require(batch>=16&&batch<=2048,"local-bc batch budget 16..2048");
    require(fixed_ppo::actor_obs_dim==184,"local-bc requires the 184D guided actor");

    std::ifstream dataset_file(dataset_path,std::ios::binary);
    require(bool(dataset_file),"cannot open dataset "+dataset_path);
    uint64_t rows=0;
    dataset_file.read(reinterpret_cast<char*>(&rows),sizeof(rows));
    require(rows>0&&rows<=400000,"dataset row budget");
    dataset_file.seekg(0,std::ios::end);
    const auto bytes=dataset_file.tellg();
    require(bytes==std::streamoff(sizeof(rows)+rows*sizeof(Demonstration)),"dataset size mismatch "+dataset_path);
    dataset_file.seekg(sizeof(rows),std::ios::beg);
    std::vector<Demonstration> dataset(size_t(rows),Demonstration{});
    dataset_file.read(reinterpret_cast<char*>(dataset.data()),std::streamsize(rows*sizeof(Demonstration)));
    require(bool(dataset_file),"dataset read failed "+dataset_path);

    Metal metal;
    metal.compile(ps_source()+PPO_TRAINER_MSL+kWaypointKernels+kImitationKernel);
    verify_imitation_gradient(metal,dataset_path+".gradient-check.json");

    // Label/observation/action-map consistency: every stored label must be
    // regenerable from its stored observation and this warmstart alone. A
    // mismatch means the dataset was not produced by this checkpoint (or a
    // label leaked non-observation state), which would poison the BC step.
    std::ifstream warm_file(warmstart,std::ios::binary);
    require(bool(warm_file),"cannot open warmstart "+warmstart);
    const PpoCheckpointHeader header=read_checkpoint_header(warm_file);
    require(header.actor_count==fixed_ppo::actor_param_count,"warmstart checkpoint dimensions");
    fixed_ppo::ActorParams params;
    warm_file.read(reinterpret_cast<char*>(params.values.data()),
                   std::streamsize(params.values.size()*sizeof(float)));
    require(bool(warm_file),"warmstart actor read failed");
    const uint32_t context=fixed_ppo::context_offset;
    const uint32_t hint=fixed_ppo::geometry_prior_offset;
    float worst=0;uint64_t brake_rows=0;
    for(const Demonstration& demo:dataset) {
        const float* obs=demo.observation.data();
        float open=0;
        for(uint k=0;k<20;k++)if(obs[k]*12>2.5f)open+=1;
        const float gate=.25f+.75f*open/20;
        const float angle=std::atan2(obs[context+1],obs[context]);
        const float distance=obs[context+3]*10.0f;
        std::array<float,4> expected{};
        if(std::fabs(angle)>kBrakeBearingRad&&distance>kBrakeDistanceM) {
            expected={0,0,0,std::max(-1.0f,std::min(1.0f,kBrakeYawGain*angle))};
            brake_rows++;
        } else {
            float hidden[fixed_ppo::hidden_dim],mean[fixed_ppo::action_dim];
            fixed_ppo::actor_forward(obs,params,hidden,mean);
            float q[3]{};
            for(uint axis=0;axis<3;axis++) {
                const float prior=obs[hint+axis];
                q[axis]=std::tanh(prior+gate*(mean[axis]-prior));
            }
            const float magnitude=std::sqrt(q[0]*q[0]+q[1]*q[1]+q[2]*q[2]);
            const float scale=1.0f/std::max(1.0f,magnitude);
            for(uint axis=0;axis<3;axis++)expected[axis]=q[axis]*scale;
            expected[3]=std::tanh(gate*mean[3]);
        }
        for(uint axis=0;axis<4;axis++)
            worst=std::max(worst,std::fabs(expected[axis]-demo.command[axis]));
    }
    require(worst<1e-4f,"label/observation/action-map mismatch "+dataset_path+
            " versus "+warmstart+" max_error="+std::to_string(worst));
    std::cout<<"bc_label_check PASS rows="<<rows<<" brake_rows="<<brake_rows<<" max_error="<<worst<<std::endl;

    const std::filesystem::path destination(output);
    if(destination.has_parent_path())std::filesystem::create_directories(destination.parent_path());

    // BC loop after navigation_imitation::train: same buffers, same kernel
    // chain, same Adam settings. The gradient buffers for log-std stay zero,
    // so exploration parameters are untouched.
    SimConfig config;config.mode=22;config.geometry_memory=1;config.speed=1.5f;
    config.max_steps=400;config.seed=42;
    Sim policy(metal,config,32);
    navigation_training::load_actor(policy,warmstart,true);
    PPOTrainer checkpoint(policy);
    auto observations=metal.buffer(batch*fixed_ppo::actor_obs_dim*4),targets=metal.buffer(batch*4*4);
    auto hidden=metal.buffer(batch*fixed_ppo::hidden_dim*4);
    auto means=metal.buffer(batch*4*4),dmean=metal.buffer(batch*4*4),dstd=metal.buffer(batch*4*4);
    auto delta=metal.buffer(batch*fixed_ppo::hidden_dim*4);
    auto gradient=metal.buffer(fixed_ppo::actor_param_count*4),first=metal.buffer(fixed_ppo::actor_param_count*4);
    auto second=metal.buffer(fixed_ppo::actor_param_count*4);
    auto losses=metal.buffer(batch*4),count=PPOTrainer::scalar(metal,batch);
    auto parameter_count=PPOTrainer::scalar(metal,uint(fixed_ppo::actor_param_count));
    auto norm_limit=PPOTrainer::scalar(metal,.5f),factor=metal.buffer(4);
    auto optimizer=metal.buffer(sizeof(PpoAdamHostConfig));
    std::memset(first.contents,0,first.length);
    std::memset(second.contents,0,second.length);
    std::memset(dstd.contents,0,dstd.length);
    std::mt19937 random(42);
    std::uniform_int_distribution<size_t> sample(0,dataset.size()-1);
    std::ofstream history(output+".history.csv");
    require(bool(history),"cannot write "+output+".history.csv");
    history<<"update,batch_command_mse,wall_s\n";
    const double started=seconds();
    for(uint step=1;step<=updates;step++) {@autoreleasepool{
        for(uint n=0;n<batch;n++) {
            const auto& demo=dataset[sample(random)];
            std::memcpy(static_cast<float*>(observations.contents)+n*fixed_ppo::actor_obs_dim,
                        demo.observation.data(),fixed_ppo::actor_obs_dim*sizeof(float));
            std::memcpy(static_cast<float*>(targets.contents)+n*4,demo.command.data(),4*sizeof(float));
        }
        PpoAdamHostConfig settings{.0003f,.9f,.999f,1e-8f,0,step};
        std::memcpy(optimizer.contents,&settings,sizeof(settings));
        auto cb=[metal.queue commandBuffer];
        metal.dispatch(cb,metal.pipeline("ppo_actor_forward"),batch,{observations,policy.actor,hidden,means,count},64);
        metal.dispatch(cb,metal.pipeline("imitation_gradient"),batch,{observations,means,targets,dmean,losses,count},64);
        metal.dispatch(cb,metal.pipeline("ppo_actor_hidden_delta"),batch*fixed_ppo::hidden_dim,
            {policy.actor,dmean,hidden,delta,count},128);
        metal.dispatch(cb,metal.pipeline("ppo_actor_grad_direct"),fixed_ppo::actor_param_count,
            {observations,hidden,dmean,dstd,delta,gradient,count},128);
        metal.dispatch(cb,metal.pipeline("ppo_grad_scale_factor"),256,
            {gradient,factor,parameter_count,norm_limit},256);
        metal.dispatch(cb,metal.pipeline("ppo_apply_grad_scale"),fixed_ppo::actor_param_count,
            {gradient,factor,parameter_count},128);
        metal.dispatch(cb,metal.pipeline("ppo_adam_update"),fixed_ppo::actor_param_count,
            {policy.actor,gradient,first,second,parameter_count,optimizer},128);
        metal.finish(cb);
        if(step==1||step%50==0||step==updates) {
            double loss=0;
            for(uint n=0;n<batch;n++)loss+=static_cast<float*>(losses.contents)[n];
            loss/=batch;
            history<<step<<','<<loss<<','<<seconds()-started<<'\n';history.flush();
            std::cout<<"bc_update="<<step<<" batch_command_mse="<<loss<<" wall_s="<<seconds()-started<<std::endl;
            if(step==updates/2&&step!=updates) {
                std::memset(policy.task_control.contents,0,policy.task_control.length);
                std::memset(policy.runtime_control.contents,0,policy.runtime_control.length);
                std::memset(policy.potential_control.contents,0,policy.potential_control.length);
                checkpoint.save_checkpoint(output+"."+std::to_string(step)+".bin",0,42,0);
            }
        }
    }}
    // Zero control blocks keep the header version deterministic; the saved
    // optimizer state is zero by design — this is a parameter warmstart, not
    // a PPO resume receipt.
    std::memset(policy.task_control.contents,0,policy.task_control.length);
    std::memset(policy.runtime_control.contents,0,policy.runtime_control.length);
    std::memset(policy.potential_control.contents,0,policy.potential_control.length);
    checkpoint.save_checkpoint(output,0,42,0);

    // Closed-loop evidence for the saved file on the selection and held-out
    // splits (flights, not loss values).
    for(const char* split:{"dev-a","dev-c"}) {
        const BankSpec eval_spec=spec_by_name(split);
        const auto eval_bank=build_bank(eval_spec,nullptr);
        const BankControl eval_control{eval_spec.period,uint32_t(eval_bank.size())};
        std::ifstream saved(output,std::ios::binary);
        require(bool(saved),"cannot reopen "+output);
        const PpoCheckpointHeader saved_header=read_checkpoint_header(saved);
        require(saved_header.actor_count==fixed_ppo::actor_param_count,"bc checkpoint dimensions");
        std::vector<float> actor(saved_header.actor_count);
        saved.read(reinterpret_cast<char*>(actor.data()),std::streamsize(actor.size()*sizeof(float)));
        require(bool(saved),"bc checkpoint actor read back failed");
        evaluate_bank(metal,actor.data(),&eval_spec,eval_bank,eval_control,17,700001,1.5f,
                      output+".eval-"+split+".csv",eval_spec.name,eval_spec.environments,400);
    }
    const std::string checkpoint_sha=challenge_evaluation::sha256_file(output);
    std::ofstream summary(output+".summary.json");
    require(bool(summary),"cannot write "+output+".summary.json");
    summary<<"{\"schema\":\"local-bc-v1\",\"dataset\":\""<<dataset_path
           <<"\",\"dataset_sha256\":\""<<challenge_evaluation::sha256_file(dataset_path)
           <<"\",\"warmstart\":\""<<warmstart<<"\",\"warmstart_sha256\":\""
           <<challenge_evaluation::sha256_file(warmstart)
           <<"\",\"updates\":"<<updates<<",\"batch\":"<<batch<<",\"learning_rate\":0.0003"
           <<",\"label_rows\":"<<rows<<",\"label_brake_rows\":"<<brake_rows
           <<",\"label_max_error\":"<<worst<<",\"checkpoint_sha256\":\""<<checkpoint_sha
           <<"\",\"optimizer_state\":\"zero (parameter warmstart, not a PPO resume)\"}\n";
    summary.close();
    std::cout<<"bc_done checkpoint="<<output<<" sha256="<<checkpoint_sha<<std::endl;
    return 0;
}

// ---------------------------------------------------------- witness controls
//
// Truth-scripted feasibility flights (decision.md). A non-learning scripted
// pilot follows the verified route polyline (opening bend -> second bend ->
// goal) with proportional braking and yaw-to-target, through actual RAPTOR
// and rl_physics_step. Generator truth is used ONLY here, offline, as a
// measurement instrument: it is never an actor input. Passing and failing
// flights are both retained; these receipts back any "grounded task is
// physically feasible" claim instead of witness geometry alone.
struct WitnessLabelRow {
    bool has_opening=false;
    int route_points=-1;
    float opening[3]={0,0,0};
    float via2[3]={0,0,0};
    std::string side,vertical;
};

// Read grounded label columns back out of the bank CSV by header name, so the
// receipt does not depend on column order.
static std::vector<WitnessLabelRow> read_grounded_labels(const std::string& csv_path,uint64_t expected) {
    std::ifstream file(csv_path);
    require(bool(file),"cannot open bank CSV "+csv_path);
    std::string line;
    require(bool(std::getline(file,line)),"empty bank CSV "+csv_path);
    std::vector<std::string> columns;
    {
        std::stringstream stream(line);std::string cell;
        while(std::getline(stream,cell,','))columns.push_back(cell);
    }
    const auto find_column=[&](const char* name)->size_t {
        for(size_t index=0;index<columns.size();index++)
            if(columns[index]==name)return index;
        throw std::runtime_error(std::string("missing grounded label column ")+name);
    };
    const size_t open_col=find_column("opening_geometric"),route_col=find_column("route_points");
    const size_t ox=find_column("opening_x"),oy=find_column("opening_y"),oz=find_column("opening_z");
    const size_t v2x=find_column("via2_x"),v2y=find_column("via2_y"),v2z=find_column("via2_z");
    const size_t side_col=find_column("side"),vertical_col=find_column("vertical");
    std::vector<WitnessLabelRow> rows;
    while(std::getline(file,line)) {
        if(line.empty())continue;
        std::stringstream stream(line);std::vector<std::string> cells;std::string cell;
        while(std::getline(stream,cell,','))cells.push_back(cell);
        require(cells.size()>=columns.size(),"bank CSV row too short in "+csv_path);
        WitnessLabelRow row;
        row.has_opening=cells[open_col]=="1";
        row.route_points=std::stoi(cells[route_col]);
        row.opening[0]=std::stof(cells[ox]);row.opening[1]=std::stof(cells[oy]);row.opening[2]=std::stof(cells[oz]);
        row.via2[0]=std::stof(cells[v2x]);row.via2[1]=std::stof(cells[v2y]);row.via2[2]=std::stof(cells[v2z]);
        row.side=cells[side_col];row.vertical=cells[vertical_col];
        rows.push_back(row);
    }
    require(rows.size()==expected,"bank CSV label row count mismatch "+csv_path);
    return rows;
}

static int command_witness(int argc,char** argv) {
    require(argc>=5,"local-witness BANK.bin BANK.csv OUT_PREFIX");
    const std::string bank_path=argv[2],labels_path=argv[3],output=argv[4];
    BankControl control{};std::string bank_hash;
    std::vector<BankEntry> entries=read_bank(bank_path,control,bank_hash);
    require(control.period>=1&&entries.size()==size_t(control.count),"witness bank shape");
    require(control.count%control.period==0,
            "witness expects count divisible by period (envs x period entries)");
    const std::vector<WitnessLabelRow> labels=read_grounded_labels(labels_path,entries.size());
    Metal metal;
    metal.compile(ps_source()+PPO_TRAINER_MSL+kWaypointKernels);
    auto apply=metal.pipeline("waypoint_task_apply");
    const std::filesystem::path destination(output);
    if(destination.has_parent_path())std::filesystem::create_directories(destination.parent_path());
    std::ofstream flights(output+".flights.csv");
    require(bool(flights),"cannot write witness flights "+output+".flights.csv");
    flights<<std::setprecision(9);
    flights<<"slot,env,family,scene_seed,route_class,side,vertical,opening_visible,route_points,"
              "velocity_category,success,collision,timeout,steps,time_s,path_m,min_clearance_m,"
              "initial_distance_m,final_distance_m,final_speed_mps\n";
    const double started=seconds();
    uint64_t total=0,successes=0,collisions=0,timeouts=0;
    uint64_t blocked_total=0,blocked_success=0,direct_total=0,direct_success=0;
    for(uint32_t slot=0;slot<control.period;slot++) {
        // One fresh eval run per slot; the sliced bank maps env i to entry
        // i*period+slot under period=1.
        std::vector<BankEntry> slice;std::vector<WitnessLabelRow> slice_labels;
        const size_t envs=size_t(control.count)/control.period;
        slice.reserve(envs);slice_labels.reserve(envs);
        for(size_t env=0;env<envs;env++) {
            slice.push_back(entries[env*control.period+slot]);
            slice_labels.push_back(labels[env*control.period+slot]);
        }
        BankControl slice_control{1,uint32_t(slice.size())};
        SimConfig config;
        config.n=uint32_t(slice.size());config.mode=17;config.family=0;config.eval=1;config.seed=800001;
        config.speed=1.5f;config.distance=4;config.max_steps=400;config.geometry_memory=1;
        Sim sim(metal,config,32);
        LocalRun run=make_local_run(sim,slice,slice_control,apply,1.0f);
        auto install=[&](id<MTLCommandBuffer> cb,uint32_t tick) {
            const auto c=sim.configs[tick%sim.horizon];
            sim.m.dispatch(cb,run.apply_pipeline,sim.cfg.n,
                {sim.states,sim.runs,sim.worlds,sim.task_states,run.bank,run.bank_control,sim.task_control,c},64);
        };
        auto first=[metal.queue commandBuffer];
        install(first,0);
        metal.finish(first);
        std::vector<uint8_t> phase(sim.cfg.n,0),done(sim.cfg.n,0);
        for(uint32_t env=0;env<sim.cfg.n;env++)
            phase[env]=slice_labels[env].has_opening?0:1;
        uint32_t tick=0;
        for(;tick<=config.max_steps;tick++) {
            bool all_done=true;
            for(uint32_t env=0;env<sim.cfg.n;env++)if(!done[env])all_done=false;
            if(all_done)break;
            const auto* states=(const RLPhysicsState*)sim.states.contents;
            auto* runs=(SimRun*)sim.runs.contents;
            for(uint32_t env=0;env<sim.cfg.n;env++) {
                if(done[env])continue;
                const BankEntry& entry=slice[env];
                const WitnessLabelRow& label=slice_labels[env];
                float target[3];
                if(phase[env]==0&&label.has_opening) {
                    const float dx=label.opening[0]-states[env].position[0];
                    const float dy=label.opening[1]-states[env].position[1];
                    const float dz=label.opening[2]-states[env].position[2];
                    if(std::sqrt(dx*dx+dy*dy+dz*dz)<=0.4f)phase[env]=1;
                }
                if(phase[env]==0) {
                    target[0]=label.opening[0];target[1]=label.opening[1];target[2]=label.opening[2];
                } else if(label.route_points==4) {
                    // Two-bend witness: pass the second bend before the goal.
                    const float dx=label.via2[0]-states[env].position[0];
                    const float dy=label.via2[1]-states[env].position[1];
                    const float dz=label.via2[2]-states[env].position[2];
                    if(std::sqrt(dx*dx+dy*dy+dz*dz)>0.4f) {
                        target[0]=label.via2[0];target[1]=label.via2[1];target[2]=label.via2[2];
                    } else {
                        phase[env]=2;
                        target[0]=entry.goal_position[0];target[1]=entry.goal_position[1];target[2]=entry.goal_position[2];
                    }
                } else {
                    target[0]=entry.goal_position[0];target[1]=entry.goal_position[1];target[2]=entry.goal_position[2];
                }
                float command[3]={1.2f*(target[0]-states[env].position[0]),
                                  1.2f*(target[1]-states[env].position[1]),
                                  1.2f*(target[2]-states[env].position[2])};
                const float command_speed=std::sqrt(command[0]*command[0]+command[1]*command[1]+command[2]*command[2]);
                if(command_speed>1.2f) {
                    const float scale=1.2f/command_speed;
                    for(int axis=0;axis<3;axis++)command[axis]*=scale;
                }
                const float bearing=std::atan2(target[1]-states[env].position[1],
                                               target[0]-states[env].position[0]);
                const float error=std::remainder(bearing-runs[env].yaw,6.28318530718f);
                const float yaw_command=std::max(-1.0f,std::min(1.0f,4.0f*error));
                for(int axis=0;axis<3;axis++)runs[env].desired_velocity[axis]=command[axis];
                runs[env].yaw+=yaw_command*0.025f;
                const float cosine=std::cos(runs[env].yaw),sine=std::sin(runs[env].yaw);
                runs[env].previous_nav[0]=(cosine*command[0]+sine*command[1])/1.5f;
                runs[env].previous_nav[1]=(-sine*command[0]+cosine*command[1])/1.5f;
                runs[env].previous_nav[2]=command[2]/1.5f;
                runs[env].previous_nav[3]=yaw_command;
                float* ring=(float*)sim.commands.contents+(size_t(env)*8+runs[env].steps%8)*4;
                for(int axis=0;axis<4;axis++)ring[axis]=runs[env].previous_nav[axis];
            }
            auto move=[metal.queue commandBuffer];
            const auto c=sim.configs[tick%sim.horizon];
            sim.m.dispatch(move,sim.advance_p,sim.cfg.n,
                {sim.states,sim.runs,sim.worlds,sim.sensors,sim.commands,sim.raptor,sim.critic,sim.rewards,
                 sim.next_values,sim.terminated,sim.truncated,sim.physics,c,sim.bank_worlds,sim.bank_schedule,
                 sim.bank_control,sim.bank_active_ids,sim.bank_transition_ids,sim.environment_physics,
                 sim.runtime_control,sim.task_states,sim.task_control,sim.potential_fields,sim.potential_spec,
                 sim.potential_control},64);
            install(move,tick);
            metal.finish(move);
            runs=(SimRun*)sim.runs.contents;
            for(uint32_t env=0;env<sim.cfg.n;env++)
                if(!done[env]&&runs[env].episodes>0)done[env]=1;
        }
        const auto* states=(const RLPhysicsState*)sim.states.contents;
        const auto* runs=(const SimRun*)sim.runs.contents;
        for(uint32_t env=0;env<sim.cfg.n;env++) {
            require(done[env],"witness flight never terminated env "+std::to_string(env));
            const BankEntry& entry=slice[env];
            const WitnessLabelRow& label=slice_labels[env];
            float distance=0;
            for(int axis=0;axis<3;axis++) {
                const float delta=entry.goal_position[axis]-states[env].position[axis];
                distance+=delta*delta;
            }
            distance=std::sqrt(distance);
            const float speed=std::sqrt(states[env].linear_velocity[0]*states[env].linear_velocity[0]+
                                        states[env].linear_velocity[1]*states[env].linear_velocity[1]+
                                        states[env].linear_velocity[2]*states[env].linear_velocity[2]);
            flights<<slot<<','<<env<<','<<entry.family<<','<<entry.scene_seed<<','<<entry.route_class
                   <<','<<label.side<<','<<label.vertical<<','<<(label.has_opening?1:0)
                   <<','<<label.route_points
                   <<','<<velocity_category(entry.start_velocity,
                                            entry.goal_position[0]-entry.start_position[0],
                                            entry.goal_position[1]-entry.start_position[1])
                   <<','<<runs[env].successes<<','<<runs[env].collisions<<','<<runs[env].timeouts
                   <<','<<runs[env].steps<<','<<runs[env].elapsed<<','<<runs[env].path
                   <<','<<runs[env].min_clearance<<','<<entry.initial_distance<<','<<distance<<','<<speed<<'\n';
            total++;successes+=runs[env].successes;collisions+=runs[env].collisions;timeouts+=runs[env].timeouts;
            if(entry.route_class==1u) {blocked_total++;blocked_success+=runs[env].successes;}
            else {direct_total++;direct_success+=runs[env].successes;}
        }
    }
    flights.close();
    std::ofstream summary(output+".summary.json");
    require(bool(summary),"cannot write witness summary");
    summary<<std::setprecision(6)
           <<"{\"schema\":\"local-witness-v1\",\"bank\":\""<<bank_path
           <<"\",\"bank_entry_sha256\":\""<<bank_hash<<"\",\"labels\":\""<<labels_path
           <<"\"\n,\"flights\":"<<total<<",\"successes\":"<<successes
           <<",\"collisions\":"<<collisions<<",\"timeouts\":"<<timeouts
           <<",\"blocked\":{\"n\":"<<blocked_total<<",\"successes\":"<<blocked_success<<"}"
           <<",\"direct\":{\"n\":"<<direct_total<<",\"successes\":"<<direct_success
           <<"}\n,\"script\":\"polyline follow, proportional braking k=1.2, speed cap 1.2, yaw=clamp(4*err), RAPTOR physics\""
           <<",\"wall_s\":"<<seconds()-started<<"}\n";
    summary.close();
    std::cout<<"witness_done flights="<<total<<" successes="<<successes
             <<" collisions="<<collisions<<" timeouts="<<timeouts
             <<" blocked="<<blocked_success<<"/"<<blocked_total
             <<" direct="<<direct_success<<"/"<<direct_total<<"\n";
    return 0;
}

// --------------------------------------------- perception-support tests
//
// Meaningful tests for this mechanism only (no test-volume project):
//  T1 module semantics: measured hit / no-hit / frame expiry on synthetic rings
//  T2 guidance semantics: unknown neutral + capped slow (vs the CORE function
//     on identical inputs: core emits the blind-fast side hint, module does not)
//  T3 GPU/host parity: the override kernel's tail equals a host recomputation
//     from the same buffers (bindings + frame math + distance reconstruction),
//     observation contract: only the3-float prior tail differs flag-off/on
//  T4 frame: sensor_delay>0 with valid_frames=0 reproduces host behaviour
//  T5 cache: the module never consumes memory_clearances (no cache parameter)
// Legacy cross-binary parity (flag off) is run as a shell step in
// provenance/run-ps-tests.sh against the preserved b696 binary.
static int command_ps_test() {
    // ---- T1: ps_memory_clearance semantics
    std::vector<float> range_ring(8*320,12.0f),pose_ring(8*12,0.0f);
    for(uint frame=0;frame<8;frame++) {
        pose_ring[frame*12+3]=1.0f;pose_ring[frame*12+7]=1.0f;pose_ring[frame*12+11]=1.0f;
    }
    const float origin[12]={0,0,0,1,0,0,0,1,0,0,0,1};
    const float front[3]={1,0,0},rear[3]={0,-1,0};
    require(ps_memory_clearance(rear,range_ring.data(),pose_ring.data(),origin,0,0,0.05f,0.75f)==PS_UNKNOWN,
            "T1 no-hit direction must be UNKNOWN (frame expired / valid0)");
    require(ps_memory_clearance(front,range_ring.data(),pose_ring.data(),origin,0,8,0.05f,0.75f)==PS_UNKNOWN,
            "T1 no-hit in-ring direction must be UNKNOWN, never clear");
    range_ring[7*20+9]=2.0f; // pooled bin (row3,col4) sees a2m surface ahead
    const float measured=ps_memory_clearance(front,range_ring.data(),pose_ring.data(),origin,0,8,0.05f,0.75f);
    require(measured>=0.0f&&measured<=2.5f,"T1 a retained hit must read as measured");
    std::fill(range_ring.begin(),range_ring.end(),12.0f);
    range_ring[3*320+7*20+9]=2.0f; // hit parked in a frame excluded by valid2
    require(ps_memory_clearance(front,range_ring.data(),pose_ring.data(),origin,5,2,0.05f,0.75f)==PS_UNKNOWN,
            "T1 an expired frame must not count as measurement");
    std::cout<<"ps_test T1_module_semantics PASS (measured/unknown/frame-expiry)"<<std::endl;
    // Measurement support must follow the capture pose, not the current FOV.
    std::fill(range_ring.begin(),range_ring.end(),12.0f);
    const float back[3]={-1,0,0},side[3]={0,1,0};
    require(ps_measured_free_bound(front,range_ring.data(),pose_ring.data(),origin,0,8,.75f)==3.0f,
            "T1 captured forward max-range supports a forward free bound");
    require(ps_measured_free_bound(back,range_ring.data(),pose_ring.data(),origin,0,8,.75f)==PS_UNKNOWN,
            "T1 forward image must not certify rear space");
    require(ps_measured_free_bound(side,range_ring.data(),pose_ring.data(),origin,0,8,.75f)==PS_UNKNOWN,
            "T1 forward image must not certify lateral space");
    pose_ring[3]=-1;pose_ring[7]=-1; // capture yaw pi, current body yaw zero
    require(ps_measured_free_bound(front,range_ring.data(),pose_ring.data(),origin,0,8,.75f)==PS_UNKNOWN,
            "T1 stale camera facing rear must not certify current forward space");
    require(ps_measured_free_bound(back,range_ring.data(),pose_ring.data(),origin,0,8,.75f)==3.0f,
            "T1 old capture pose must map rear-facing measurements correctly");
    pose_ring[3]=1;pose_ring[7]=1;
    range_ring[7*20+9]=0.0f;
    require(ps_measured_free_bound(front,range_ring.data(),pose_ring.data(),origin,0,8,.75f)==PS_UNKNOWN,
            "T1 missing depth cannot certify free space");
    std::fill(range_ring.begin(),range_ring.end(),12.0f);
    std::cout<<"ps_test T1_capture_support PASS (front/rear/side/delayed-pose/missing)"<<std::endl;


    // ---- T2: guidance semantics vs the CORE function on identical inputs
    std::vector<float> cur(80,12.0f),prev(80,12.0f);
    const float vel[3]={0,0,0};
    std::fill(range_ring.begin(),range_ring.end(),12.0f);
    auto hint_of=[&](const float* goal,float dist){
        float hint[3];
        ps_nav_guidance_memory(cur.data(),prev.data(),goal,dist,vel,0.05f,0.75f,
                               range_ring.data(),pose_ring.data(),origin,0,8,hint);
        return std::array<float,3>{hint[0],hint[1],hint[2]};
    };
    auto core_hint=[&](const float* goal,float dist){
        float hint[3];
        nav_guidance_memory(cur.data(),prev.data(),goal,dist,vel,0.05f,0.75f,
                            range_ring.data(),pose_ring.data(),origin,0,8,hint,nullptr);
        return std::array<float,3>{hint[0],hint[1],hint[2]};
    };
    const auto tanh3=[](const std::array<float,3>& h){
        return std::array<float,3>{std::tanh(h[0]),std::tanh(h[1]),std::tanh(h[2])};
    };
    const auto dot=[](const std::array<float,3>& a,const float* b){
        return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
    };
    // A) open space, goal ahead: module == core (legacy-equivalent where measured)
    {
        const float goal[3]={1,0,0};
        auto mine=hint_of(goal,1.5f),core=core_hint(goal,1.5f);
        float worst=std::fabs(mine[0]-core[0])+std::fabs(mine[1]-core[1])+std::fabs(mine[2]-core[2]);
        require(worst<1e-5f,"T2 open forward must be bit-equivalent to core");
        const auto mine3=tanh3(mine);
        require(mine3[0]>0.55f,"T2 open goal-ahead must command forward at speed");
        std::cout<<"ps_test T2_open_parity PASS delta="<<worst<<std::endl;
    }
    // B) goal BEHIND: unknown is chosen (neutral) but capped slow — and the
    //    CORE already has an outside-goal-view speed cap; do not claim otherwise.
    {
        const float goal[3]={-1,0,0};
        auto mine=tanh3(hint_of(goal,1.5f));
        auto core=tanh3(core_hint(goal,1.5f));
        const float mine_speed=std::sqrt(mine[0]*mine[0]+mine[1]*mine[1]+mine[2]*mine[2]);
        const float core_speed=std::sqrt(core[0]*core[0]+core[1]*core[1]+core[2]*core[2]);
        require(dot(mine,goal)/std::fmax(mine_speed,1e-6f)>0.85f,"T2 goal-behind must still turn toward the goal (inspect preserved)");
        require(mine_speed>0.02f,"T2 unknown must not block motion (backtrack preserved)");
        require(mine_speed<=0.25f+1e-4f,"T2 unknown direction must be capped slow");
        require(core_speed<=0.125f+1e-4f,
                "T2 the legacy core already caps out-of-view goal speed");
        std::cout<<"ps_test T2_unknown_capped PASS module_speed="<<mine_speed
                 <<" core_speed="<<core_speed<<std::endl;
    }
    // C) forward ring-blocked: both pick the side, module slow / core fast
    {
        std::fill(range_ring.begin(),range_ring.end(),12.0f);
        for(float& value:cur)value=0.4f;
        for(float& value:prev)value=0.4f;
        for(float& value:range_ring)value=0.4f;
        const float goal[3]={1,0,0};
        auto mine=tanh3(hint_of(goal,1.5f)),core=tanh3(core_hint(goal,1.5f));
        const float mine_speed=std::sqrt(mine[0]*mine[0]+mine[1]*mine[1]+mine[2]*mine[2]);
        const float core_speed=std::sqrt(core[0]*core[0]+core[1]*core[1]+core[2]*core[2]);
        require(mine[0]<0.5f,"T2 blocked-forward must not keep full-speed at the obstacle");
        require(mine_speed>0.02f,"T2 the alternative direction must stay allowed");
        require(mine_speed<=0.25f+1e-4f,"T2 selected unknown alternative must be capped slow");
        require(core_speed>0.5f,"T2 core behaviour (fast side) must be reproduced as the contrast");
        std::cout<<"ps_test T2_blocked_forward PASS module_speed="<<mine_speed
                 <<" core_speed="<<core_speed<<std::endl;
        std::fill(range_ring.begin(),range_ring.end(),12.0f);
        std::fill(cur.begin(),cur.end(),12.0f);
        std::fill(prev.begin(),prev.end(),12.0f);
    }
    std::cout<<"ps_test T5_cache_independence PASS (module has no cache parameter; "
             <<"memory_clearances never bound by ps_override_prior)"<<std::endl;

    // ---- T3/T4 GPU-vs-host parity and observation/frame contracts
    Metal metal;
    metal.compile(ps_source()+PPO_TRAINER_MSL+kWaypointKernels);
    const BankSpec spec=spec_by_name("open");
    auto bank=build_bank(spec,nullptr);
    BankControl control{spec.period,uint32_t(bank.size())};
    // One identical tick on a flag-off and a flag-on run from the same seed.
    Sim sim_off(metal,[](){SimConfig c;c.n=4;c.mode=17;c.family=0;c.eval=1;c.seed=700001;
        c.speed=1.5f;c.distance=4;c.max_steps=400;c.geometry_memory=1;return c;}(),32);
    Sim sim_on(metal,[](){SimConfig c;c.n=4;c.mode=17;c.family=0;c.eval=1;c.seed=700001;
        c.speed=1.5f;c.distance=4;c.max_steps=400;c.geometry_memory=1;return c;}(),32);
    LocalRun run_off=make_local_run(sim_off,bank,control,sim_off.m.pipeline("waypoint_task_apply"),0.2f,false);
    LocalRun run_on=make_local_run(sim_on,bank,control,sim_on.m.pipeline("waypoint_task_apply"),0.2f,true);
    auto p1=[metal.queue commandBuffer];local_probe(run_off,p1);metal.finish(p1);
    auto p2=[metal.queue commandBuffer];local_probe(run_on,p2);metal.finish(p2);
    // host inputs captured pre-tick (observe runs with runs.steps==0)
    std::vector<float> host_input(size_t(sim_on.cfg.n)*fixed_ppo::actor_obs_dim);
    std::memcpy(host_input.data(),sim_on.obs.contents,host_input.size()*sizeof(float));
    std::vector<SimRun> runs_pre(sim_on.cfg.n);
    std::memcpy(runs_pre.data(),sim_on.runs.contents,runs_pre.size()*sizeof(SimRun));
    std::vector<RLPhysicsState> states_pre(sim_on.cfg.n);
    std::memcpy(states_pre.data(),sim_on.states.contents,states_pre.size()*sizeof(RLPhysicsState));
    const std::vector<float> sensors_pre((const float*)sim_on.sensors.contents,
                                         (const float*)sim_on.sensors.contents+size_t(4)*8*320);
    const std::vector<float> poses_pre((const float*)sim_on.poses.contents,
                                       (const float*)sim_on.poses.contents+size_t(4)*8*12);
    // host recomputation for env0 (frame math from sim_observe)
    {
        const SimConfig& cfg=sim_on.cfg;
        const float sensor_dt=float(cfg.sensor_period)*0.01f*float(cfg.substeps);
        auto host_hint=[&](uint32_t env)->std::array<float,3>{
            const SimRun& run=runs_pre[env];
            const uint32_t available=run.steps/cfg.sensor_period;
            const uint32_t frame=available>cfg.sensor_delay?available-cfg.sensor_delay:0;
            const uint32_t valid=available>=cfg.sensor_delay?std::min(frame+1u,8u-cfg.sensor_delay):0u;
            const uint32_t row=env*fixed_ppo::actor_obs_dim;
            const uint32_t context=row+fixed_ppo::context_offset;
            std::array<float,80> cur{},prev{};
            for(uint k=0;k<80;k++){cur[k]=host_input[row+k]*12.0f;prev[k]=host_input[row+fixed_ppo::pooled_depth_dim+k]*12.0f;}
            // NOTE: pooled_depth_dim==80 for the184 profile (context_offset==160)
            float goal[3],vel[3],hint[3];
            for(uint j=0;j<3;j++){goal[j]=host_input[context+j];vel[j]=host_input[context+4+j]*4.0f;}
            const float distance=host_input[context+3]*10.0f;
            float pose[12];
            cpu_reference::rotation(states_pre[env].orientation_wxyz,pose+3);
            for(uint j=0;j<3;j++)pose[j]=states_pre[env].position[j];
            ps_nav_guidance_memory(cur.data(),prev.data(),goal,distance,vel,sensor_dt,
                                   NAV_SENSOR_ACTIVE_TAN_V,
                                   sensors_pre.data()+size_t(env)*8*320,
                                   poses_pre.data()+size_t(env)*8*12,
                                   pose,frame,valid,hint);
            return {hint[0],hint[1],hint[2]};
        };
        auto tick_on=[metal.queue commandBuffer];
        local_tick(run_on,tick_on,0,false);metal.finish(tick_on);
        auto tick_off=[metal.queue commandBuffer];
        local_tick(run_off,tick_off,0,false);metal.finish(tick_off);
        // observation contract: only the3-float prior tail differs
        const float* obs_on=(const float*)sim_on.obs.contents;
        const float* obs_off=(const float*)sim_off.obs.contents;
        float worst_tail=0,worst_other=0;
        for(uint32_t env=0;env<sim_on.cfg.n;env++) {
            const size_t base=size_t(env)*fixed_ppo::actor_obs_dim;
            for(uint k=0;k<fixed_ppo::actor_obs_dim;k++) {
                const float delta=std::fabs(obs_on[base+k]-obs_off[base+k]);
                const bool is_tail=k>=fixed_ppo::actor_obs_dim-3;
                if(is_tail)worst_tail=std::max(worst_tail,delta);
                else worst_other=std::max(worst_other,delta);
            }
        }
        require(worst_other<1e-6f,
                "T3 observation contract: the override must change ONLY the3-float prior tail");
        bool tail_finite=true;
        for(uint32_t env=0;env<sim_on.cfg.n;env++)
            for(uint k=fixed_ppo::actor_obs_dim-3;k<fixed_ppo::actor_obs_dim;k++)
                tail_finite=tail_finite&&std::isfinite(obs_on[size_t(env)*fixed_ppo::actor_obs_dim+k]);
        require(tail_finite,"T3 prior tail must be finite");
        std::cout<<"ps_test T3_observation_contract PASS non_tail_delta="<<worst_other
                 <<" tail_delta="<<worst_tail<<std::endl;
        // GPU/host parity: env0 tail vs host recomputation (pre-tick inputs,
        // frame0 semantics — observe ran with runs.steps==0)
        const float* tail=(const float*)sim_on.obs.contents;
        auto host=host_hint(0);
        float worst=0;
        for(uint j=0;j<3;j++)
            worst=std::max(worst,std::fabs(tail[size_t(0)*fixed_ppo::actor_obs_dim+fixed_ppo::actor_obs_dim-3+j]-host[j]));
        require(worst<1e-4f,"T3 GPU override tail must match host recomputation");
        std::cout<<"ps_test T3_gpu_host_parity PASS max_delta="<<worst<<std::endl;
        // T4 frame: repeat with sensor_delay=3 (valid=0 at frame0) — host vs GPU
    }
    // T4: sensor delay3 → frame math valid=0 at start; the override must match
    // a host recomputation with valid=0 and the PRE-tick pose.
    {
        Sim sim_delay(metal,[](){SimConfig c;c.n=4;c.mode=17;c.family=0;c.eval=1;c.seed=700001;
            c.speed=1.5f;c.distance=4;c.max_steps=400;c.geometry_memory=1;c.sensor_delay=3;return c;}(),32);
        LocalRun run_delay=make_local_run(sim_delay,bank,control,sim_delay.m.pipeline("waypoint_task_apply"),0.2f,true);
        auto p=[metal.queue commandBuffer];local_probe(run_delay,p);metal.finish(p);
        // pre-tick inputs (observe runs with steps==0)
        std::vector<float> pre_obs(size_t(sim_delay.cfg.n)*fixed_ppo::actor_obs_dim);
        std::memcpy(pre_obs.data(),sim_delay.obs.contents,pre_obs.size()*sizeof(float));
        std::vector<RLPhysicsState> pre_states(sim_delay.cfg.n);
        std::memcpy(pre_states.data(),sim_delay.states.contents,pre_states.size()*sizeof(RLPhysicsState));
        auto t=[metal.queue commandBuffer];local_tick(run_delay,t,0,false);metal.finish(t);
        const float sensor_dt=float(sim_delay.cfg.sensor_period)*0.01f*float(sim_delay.cfg.substeps);
        const uint32_t available=0u; // steps==0 at observe time
        const uint32_t valid=available>=sim_delay.cfg.sensor_delay
            ?std::min(available+1u,8u-sim_delay.cfg.sensor_delay):0u;
        require(valid==0,"T4 sensor_delay=3 at step0 must yield valid_frames=0");
        std::array<float,80> cur{},prevr{};
        for(uint k=0;k<80;k++){cur[k]=pre_obs[k]*12.0f;prevr[k]=pre_obs[fixed_ppo::pooled_depth_dim+k]*12.0f;}
        const uint32_t context=fixed_ppo::context_offset;
        float goal[3],vel[3],hint[3];
        for(uint j=0;j<3;j++){goal[j]=pre_obs[context+j];vel[j]=pre_obs[context+4+j]*4.0f;}
        const float distance=pre_obs[context+3]*10.0f;
        float pose[12];
        cpu_reference::rotation(pre_states[0].orientation_wxyz,pose+3);
        for(uint j=0;j<3;j++)pose[j]=pre_states[0].position[j];
        ps_nav_guidance_memory(cur.data(),prevr.data(),goal,distance,vel,sensor_dt,
                               NAV_SENSOR_ACTIVE_TAN_V,
                               (const float*)sim_delay.sensors.contents,
                               (const float*)sim_delay.poses.contents,
                               pose,0u,valid,hint);
        const float* obs=(const float*)sim_delay.obs.contents;
        float worst=0;
        for(uint j=0;j<3;j++)
            worst=std::max(worst,std::fabs(obs[fixed_ppo::actor_obs_dim-3+j]-hint[j]));
        require(worst<1e-4f,"T4 GPU tail with valid_frames=0 must match host recomputation");
        std::cout<<"ps_test T4_frame_delay PASS valid_frames=0 sensor_delay=3 max_delta="<<worst<<std::endl;
    }
    std::cout<<"ps_test_all PASS module=perception_support.hpp kernel=ps_override_prior"<<std::endl;
    return 0;
}

// ------------------------------------------------------- causal instrument
//
// ps-probe: one full source-bank policy pass (mechanism OFF = the current
// core prior) that records, at every terminal tick, what the prior commanded,
// what the actor executed, and whether those directions had ANY measured
// evidence in the retained8-frame range/pose ring — plus the contact
// geometry's own last-seen class (never / measured timely / aged / off-target).
// Offline source geometry is used ONLY for post-hoc contact attribution
// (never an actor input). Output: per-episode CSV + summary cross-tab that
// answers "are failures commanded/meant into unmeasured space?".
static int command_ps_probe(int argc,char** argv) {
    require(argc>=4,"ps-probe CHECKPOINT OUT_PREFIX [--spec source]");
    const std::string checkpoint=argv[2],output=argv[3];
    std::string spec_name="source";
    for(int index=4;index<argc;index++) {
        const std::string option=argv[index];
        if(option=="--spec")spec_name=option_value(argc,argv,index,option);
        else throw std::runtime_error("unknown ps-probe option "+option);
    }
    const BankSpec spec=spec_by_name(spec_name);
    Metal metal;
    metal.compile(ps_source()+PPO_TRAINER_MSL+kWaypointKernels);
    auto apply=metal.pipeline("waypoint_task_apply");
    BankControl control{};const auto bank=build_bank(spec,nullptr);
    control.period=spec.period;control.count=uint32_t(bank.size());
    SimConfig config;
    config.n=spec.environments;config.mode=17;config.eval=0;config.family=0;config.seed=spec.seed;
    config.speed=1.5f;config.distance=4;config.max_steps=400;config.geometry_memory=1;
    Sim sim(metal,config,32);
    navigation_training::load_actor(sim,checkpoint,false);
    // support=false: this diagnoses the CURRENT system (core prior).
    LocalRun run=make_local_run(sim,bank,control,apply,1.0f,false);
    auto probe=[metal.queue commandBuffer];
    local_probe(run,probe);
    metal.finish(probe);
    verify_reset(run,bank,control,true);

    const uint32_t horizon=sim.horizon,envs=spec.environments,target=spec.period;
    const float sensor_dt=float(config.sensor_period)*0.01f*float(config.substeps);
    const uint32_t context=fixed_ppo::context_offset;
    // per-tick pre-advance snapshots (host-visible buffers)
    std::vector<SimRun> snap_runs(envs);
    std::vector<RLPhysicsState> snap_states(envs);
    std::vector<WWorld> snap_worlds(envs);
    std::vector<float> snap_sensors(size_t(envs)*8*320),snap_poses(size_t(envs)*8*12);
    std::vector<float> snap_row(size_t(envs)*(context+7+3)); // ctx goal(3)+dist+vel(3) then prior(3)
    std::vector<uint32_t> prev_episodes(envs,0),prev_success(envs,0),
                          prev_collisions(envs,0),prev_timeouts(envs,0);
    std::vector<uint8_t> in_cycle(envs,1);
    const std::filesystem::path destination(output);
    if(destination.has_parent_path())std::filesystem::create_directories(destination.parent_path());
    std::ofstream csv(output+".flights.csv");
    require(bool(csv),"cannot write "+output+".flights.csv");
    csv<<std::setprecision(6);
    csv<<"env,slot,family,scene_seed,route_class,success,collision,timeout,time_s,steps,"
          "goal_bearing_deg,goal_body_dist_m,cmd_x,cmd_y,cmd_z,cmd_speed,cmd_unknown,cmd_clearance_m,"
          "hint_x,hint_y,hint_z,hint_fraction,hint_unknown,hint_clearance_m,"
          "contact_geom_id,contact_surface_m,contact_class,contact_age_s\n";
    uint64_t episodes=0,successes=0,contacts=0;
    uint64_t contact_cmd_unknown=0,contact_hint_unknown=0;
    uint64_t success_cmd_unknown=0,success_hint_unknown=0;
    uint64_t class_never=0,class_timely=0,class_aged=0,class_off=0;
    const double started=seconds();
    bool covered=false;uint32_t tick=0;

    // Host-side coverage: same swept-sphere geometry as ps_memory_clearance,
    // returning the frames-ago of the newest qualifying hit.
    const auto coverage=[&](const float* dir_body,const float* sensors_env,const float* poses_env,
                            const float* pose12,uint32_t latest,uint32_t valid)->std::array<float,3>{
        // returns {measured(0/1), nearest, last_back}
        float nearest=3.0f;float measured=0.0f;float last_back=-1.0f;
        for(uint32_t back=0;back<valid&&back<8;back++) {
            const uint32_t frame=(latest+8u-back)%8u;
            const float* old_pose=poses_env+frame*12;
            const float* ranges=sensors_env+frame*320;
            const float age=float(back)*sensor_dt;
            for(uint32_t row=0;row<8;row++)for(uint32_t col=0;col<10;col++) {
                const uint32_t px=row*2*20+col*2;
                uint32_t hit=px;float range=ranges[px];
                const uint32_t pixels[3]={px+1,px+20,px+21};
                for(uint32_t j=0;j<3;j++)if(ranges[pixels[j]]<range){range=ranges[pixels[j]];hit=pixels[j];}
                if(range<=0.01f||range>=11.9f)continue;
                float ry=1.0f-(float(hit%20)+.5f)/10.0f,rz=0.75f*(1.0f-(float(hit/20)+.5f)/8.0f);
                float inv=1.0f/std::sqrt(1+ry*ry+rz*rz);float ray[3]={inv,ry*inv,rz*inv};
                const float wx=old_pose[3]*ray[0]+old_pose[4]*ray[1]+old_pose[5]*ray[2];
                const float wy=old_pose[6]*ray[0]+old_pose[7]*ray[1]+old_pose[8]*ray[2];
                const float wz=old_pose[9]*ray[0]+old_pose[10]*ray[1]+old_pose[11]*ray[2];
                const float dx=old_pose[0]+range*wx-pose12[0];
                const float dy=old_pose[1]+range*wy-pose12[1];
                const float dz=old_pose[2]+range*wz-pose12[2];
                const float x=pose12[3]*dx+pose12[6]*dy+pose12[9]*dz;
                const float y=pose12[4]*dx+pose12[7]*dy+pose12[10]*dz;
                const float z=pose12[5]*dx+pose12[8]*dy+pose12[11]*dz;
                const float along=x*dir_body[0]+y*dir_body[1]+z*dir_body[2];
                const float radius=0.20f+0.015f*range+0.15f*age;
                if(along<=0.0f||along>3.0f+radius)continue;
                const float lateral2=std::max(x*x+y*y+z*z-along*along,0.0f);
                const float radius2=radius*radius;
                if(lateral2<radius2){
                    measured=1.0f;nearest=std::min(nearest,std::max(along-std::sqrt(radius2-lateral2),0.0f));
                    last_back=float(back);
                }
            }
        }
        return {measured,nearest,last_back};
    };
    // Nearest obstacle surface (offline attribution only) — wclearance's
    // per-obstacle branch without the body/wall terms.
    const auto nearest_obstacle=[&](const WWorld& world,WVec pos,float time,int& index,float& surface)->bool {
        float best=1e9f;index=-1;surface=0.0f;
        for(uint32_t i=0;i<world.count;i++) {
            const WObstacle& obstacle=world.obstacles[i];const WVec q=ws(pos,wc(obstacle,time));float distance;
            if(obstacle.kind==1)distance=wl(q)-obstacle.size[0];
            else if(obstacle.kind==2) {
                float a=std::sqrt(q.x*q.x+q.y*q.y)-obstacle.size[0],b=std::fabs(q.z)-obstacle.size[2];
                distance=std::sqrt(std::max(a,0.0f)*std::max(a,0.0f)+std::max(b,0.0f)*std::max(b,0.0f))+std::min(std::max(a,b),0.0f);
            } else {
                WVec k=wv(std::fabs(q.x)-obstacle.size[0],std::fabs(q.y)-obstacle.size[1],std::fabs(q.z)-obstacle.size[2]);
                distance=wl(wv(std::max(k.x,0.0f),std::max(k.y,0.0f),std::max(k.z,0.0f)))+std::min(std::max(k.x,std::max(k.y,k.z)),0.0f);
            }
            if(distance<best){best=distance;index=int(i);surface=distance;}
        }
        return index>=0&&surface<=0.60f;
    };

    for(;tick<=config.max_steps+8;tick++) {
        covered=true;
        for(uint32_t env=0;env<envs;env++)if(prev_episodes[env]<target){covered=false;break;}
        if(covered)break;
        const auto c=sim.configs[tick%horizon];
        // sense + actor + act (mechanism OFF: local_tick semantics minus advance)
        auto sense=[metal.queue commandBuffer];
        if(tick==0)sim.m.dispatch(sense,run.apply_pipeline,envs,
            {sim.states,sim.runs,sim.worlds,sim.task_states,run.bank,run.bank_control,sim.task_control,c},64);
        sim.m.dispatch(sense,sim.depth_p,envs*320,{sim.states,sim.runs,sim.worlds,sim.sensors,sim.physics,c,sim.poses});
        if(sim.cfg.geometry_memory) {
            sim.m.dispatch(sense,sim.memory_points_p,envs*640,
                {sim.states,sim.runs,sim.sensors,sim.poses,sim.memory_points,sim.physics,c});
            sim.m.dispatch(sense,sim.memory_candidates_p,envs*85,
                {sim.states,sim.worlds,sim.memory_points,sim.memory_clearances,c});
        }
        sim.m.dispatch(sense,sim.observe_p,envs,
            {sim.states,sim.runs,sim.worlds,sim.sensors,sim.obs,sim.co,sim.physics,c,sim.poses,sim.memory_clearances},64);
        const size_t row_offset=size_t(tick%horizon)*envs;
        encode(sim.m,sense,sim.simd_actor?"ppo_actor_forward_simd_fused":"ppo_actor_forward",
               sim.simd_actor?((size_t(envs)+7)/8)*256:envs,
               {{sim.obs,row_offset*fixed_ppo::actor_obs_dim*4},{sim.actor,0},{sim.actor_workspace,0},
                {sim.actions,row_offset*fixed_ppo::action_dim*4},{sim.env_count,0}},sim.simd_actor?256:64);
        sim.m.dispatch(sense,sim.act_p,envs,
            {sim.states,sim.runs,sim.worlds,sim.obs,sim.co,sim.actor,sim.critic,sim.actions,sim.logp,
             sim.values,sim.commands,sim.physics,c},64);
        metal.finish(sense);
        // ---- snapshot pre-advance (the terminal tick's evidence)
        std::memcpy(snap_runs.data(),sim.runs.contents,envs*sizeof(SimRun));
        std::memcpy(snap_states.data(),sim.states.contents,envs*sizeof(RLPhysicsState));
        std::memcpy(snap_worlds.data(),sim.worlds.contents,envs*sizeof(WWorld));
        std::memcpy(snap_sensors.data(),sim.sensors.contents,size_t(envs)*8*320*sizeof(float));
        std::memcpy(snap_poses.data(),sim.poses.contents,size_t(envs)*8*12*sizeof(float));
        {
            const float* obs=(const float*)sim.obs.contents;
            for(uint32_t env=0;env<envs;env++) {
                const size_t row=(row_offset+env)*fixed_ppo::actor_obs_dim;
                const size_t out=size_t(env)*10;
                for(uint j=0;j<7;j++)snap_row[out+j]=obs[row+context+j];
                for(uint j=0;j<3;j++)snap_row[out+7+j]=obs[row+fixed_ppo::actor_obs_dim-3+j];
            }
        }
        // advance + install
        auto move=[metal.queue commandBuffer];
        sim.m.dispatch(move,sim.advance_p,envs,
            {sim.states,sim.runs,sim.worlds,sim.sensors,sim.commands,sim.raptor,sim.critic,sim.rewards,
             sim.next_values,sim.terminated,sim.truncated,sim.physics,c,sim.bank_worlds,sim.bank_schedule,
             sim.bank_control,sim.bank_active_ids,sim.bank_transition_ids,sim.environment_physics,
             sim.runtime_control,sim.task_states,sim.task_control,sim.potential_fields,sim.potential_spec,
             sim.potential_control},64);
        sim.m.dispatch(move,run.apply_pipeline,envs,
            {sim.states,sim.runs,sim.worlds,sim.task_states,run.bank,run.bank_control,sim.task_control,c},64);
        metal.finish(move);
        // ---- terminals from the snapshot evidence (snapshot = pre-advance,
        // exactly what observe/act saw on the terminal tick; post-advance
        // counters carry the deltas)
        const SimRun* runs_now=(const SimRun*)sim.runs.contents;
        for(uint32_t env=0;env<envs;env++) {
            if(runs_now[env].episodes==prev_episodes[env])continue;
            require(runs_now[env].episodes==prev_episodes[env]+1,"probe expects one terminal per tick");
            const bool in_cycle=prev_episodes[env]<target;
            const bool out_success=runs_now[env].successes>prev_success[env];
            const bool out_contact=runs_now[env].collisions>prev_collisions[env];
            const bool out_timeout=runs_now[env].timeouts>prev_timeouts[env];
            if(in_cycle&&out_contact&&out_success)
                require(false,"probe terminal cannot be success and contact");
            if(in_cycle) {
                const SimRun& snap=snap_runs[env];
                const RLPhysicsState& state=snap_states[env];
                const float* rowp=snap_row.data()+size_t(env)*10;
                const uint32_t slot=prev_episodes[env]%spec.period;
                const BankEntry& entry=bank[size_t(env)*spec.period+slot];
                const float goal_bearing_deg=std::atan2(rowp[1],rowp[0])*57.29577951308232f;
                const float goal_dist=rowp[3]*10.0f;
                const float prior[3]={rowp[7],rowp[8],rowp[9]};
                float hint_vec[3]={std::tanh(prior[0]),std::tanh(prior[1]),std::tanh(prior[2])};
                const float hint_len=std::sqrt(hint_vec[0]*hint_vec[0]+hint_vec[1]*hint_vec[1]+hint_vec[2]*hint_vec[2]);
                if(hint_len>1e-4f)for(int axis=0;axis<3;axis++)hint_vec[axis]/=hint_len;
                float cmd[3]={snap.previous_nav[0],snap.previous_nav[1],snap.previous_nav[2]};
                const float cmd_speed=std::sqrt(cmd[0]*cmd[0]+cmd[1]*cmd[1]+cmd[2]*cmd[2]);
                float pose[12];
                for(int axis=0;axis<3;axis++)pose[axis]=state.position[axis];
                cpu_reference::rotation(state.orientation_wxyz,pose+3);
                const uint32_t available=snap.steps/config.sensor_period;
                const uint32_t frame=available>config.sensor_delay?available-config.sensor_delay:0;
                const uint32_t valid=available>=config.sensor_delay
                    ?std::min(frame+1u,8u-config.sensor_delay):0u;
                const float* sensors_env=snap_sensors.data()+size_t(env)*8*320;
                const float* poses_env=snap_poses.data()+size_t(env)*8*12;
                std::array<float,3> cmd_cov={-2.0f,3.0f,-1.0f}; // still / nearest / last_back
                if(cmd_speed>1e-4f) {
                    float dir[3]={cmd[0]/cmd_speed,cmd[1]/cmd_speed,cmd[2]/cmd_speed};
                    cmd_cov=coverage(dir,sensors_env,poses_env,pose,frame,valid);
                }
                std::array<float,3> hint_cov={-2.0f,3.0f,-1.0f};
                if(hint_len>1e-4f)
                    hint_cov=coverage(hint_vec,sensors_env,poses_env,pose,frame,valid);
                const bool cmd_unknown=cmd_speed>1e-4f&&cmd_cov[0]==0.0f;
                const bool hint_unknown=hint_len>1e-4f&&hint_cov[0]==0.0f;
                // contact geometry attribution (offline truth, never actor input)
                int geom_id=-1;float surface=0.0f;
                std::string contact_class="none";
                float contact_age=-1.0f;
                if(out_contact) {
                    const bool found=nearest_obstacle(snap_worlds[env],
                        wv(state.position[0],state.position[1],state.position[2]),
                        snap.elapsed,geom_id,surface);
                    if(!found) contact_class="no_near_geometry";
                    else {
                        const WVec center=wc(snap_worlds[env].obstacles[geom_id],snap.elapsed);
                        const WVec world_dir=wn(ws(center,wv(state.position[0],state.position[1],state.position[2])));
                        const float body[3]={
                            pose[3]*world_dir.x+pose[6]*world_dir.y+pose[9]*world_dir.z,
                            pose[4]*world_dir.x+pose[7]*world_dir.y+pose[10]*world_dir.z,
                            pose[5]*world_dir.x+pose[8]*world_dir.y+pose[11]*world_dir.z};
                        const auto cov=coverage(body,sensors_env,poses_env,pose,frame,valid);
                        if(cov[0]==0.0f) contact_class="never";
                        else if(std::fabs(cov[1]-surface)>0.35f) contact_class="offtarget";
                        else contact_class=(cov[2]*sensor_dt<=0.4f)?"timely":"aged";
                        contact_age=cov[2]*sensor_dt;
                    }
                }
                csv<<env<<','<<slot<<','<<entry.family<<','<<entry.scene_seed<<','
                   <<entry.route_class<<','<<(out_success?1:0)<<','<<(out_contact?1:0)
                   <<','<<(out_timeout?1:0)<<','<<snap.elapsed<<','<<(snap.elapsed+sensor_dt)
                   <<','<<snap.steps<<','<<goal_bearing_deg<<','<<goal_dist<<','
                   <<cmd[0]<<','<<cmd[1]<<','<<cmd[2]<<','<<cmd_speed<<','
                   <<(cmd_unknown?1:0)<<','<<cmd_cov[1]<<','<<hint_vec[0]<<','<<hint_vec[1]
                   <<','<<hint_vec[2]<<','<<hint_len<<','<<(hint_unknown?1:0)<<','<<hint_cov[1]
                   <<','<<geom_id<<','<<surface<<','<<contact_class<<','<<contact_age<<'\n';
                episodes++;
                if(out_success)successes++;
                if(out_contact) {
                    contacts++;
                    if(cmd_unknown)contact_cmd_unknown++;
                    if(hint_unknown)contact_hint_unknown++;
                    if(contact_class=="never")class_never++;
                    else if(contact_class=="timely")class_timely++;
                    else if(contact_class=="aged")class_aged++;
                    else if(contact_class=="offtarget")class_off++;
                } else if(out_success) {
                    if(cmd_unknown)success_cmd_unknown++;
                    if(hint_unknown)success_hint_unknown++;
                }
            }
            prev_episodes[env]=runs_now[env].episodes;
            prev_success[env]=runs_now[env].successes;
            prev_collisions[env]=runs_now[env].collisions;
            prev_timeouts[env]=runs_now[env].timeouts;
        }
    }
    csv.close();
    std::ofstream summary(output+".summary.json");
    require(bool(summary),"cannot write "+output+".summary.json");
    summary<<std::setprecision(6)
           <<"{\"schema\":\"ps-probe-v1\",\"mechanism\":\"OFF (core prior = the current system)\""
           <<",\"checkpoint\":\""<<checkpoint<<"\",\"spec\":\""<<spec.name<<"\""
           <<",\"episodes\":"<<episodes<<",\"successes\":"<<successes<<",\"contacts\":"<<contacts
           <<",\"contacts_cmd_unknown\":"<<contact_cmd_unknown
           <<",\"contacts_hint_unknown\":"<<contact_hint_unknown
           <<",\"successes_cmd_unknown\":"<<success_cmd_unknown
           <<",\"successes_hint_unknown\":"<<success_hint_unknown
           <<",\"contact_class\":{\"never\":"<<class_never<<",\"timely\":"<<class_timely
           <<",\"aged\":"<<class_aged<<",\"offtarget\":"<<class_off
           <<",\"no_near_geometry\":"<<(contacts-class_never-class_timely-class_aged-class_off)
           <<"}\n,\"time_note\":\"time_s is pre-advance (terminal tick start); terminal_time_upper_s adds one tick (0.05s)\""
           <<",\"geometry_note\":\"offline source-geometry attribution only; never an actor input\""
           <<",\"wall_s\":"<<seconds()-started<<"}\n";
    summary.close();
    std::cout<<"ps_probe_done episodes="<<episodes<<" successes="<<successes<<" contacts="<<contacts
             <<" contact_cmd_unknown="<<contact_cmd_unknown<<"/"<<contacts
             <<" contact_hint_unknown="<<contact_hint_unknown<<"/"<<contacts
             <<" success_cmd_unknown="<<success_cmd_unknown<<"/"<<successes
             <<" class_never="<<class_never<<" timely="<<class_timely
             <<" aged="<<class_aged<<" off="<<class_off<<"\n";
    return 0;
}

} // namespace waypoint

int main(int argc,char** argv) {@autoreleasepool {try {
    // Training logs must survive a crash and interleaved shell pipes.
    std::setvbuf(stdout,nullptr,_IONBF,0);
    std::setvbuf(stderr,nullptr,_IONBF,0);
    if(argc<2) { waypoint::usage(); return 0; }
    const std::string command=argv[1];
    if(command=="local-bank")return waypoint::command_bank(argc,argv);
    if(command=="local-test")return waypoint::command_test();
    if(command=="local-contract")return waypoint::command_contract(argc,argv);
    if(command=="local-train")return waypoint::command_train(argc,argv);
    if(command=="local-eval")return waypoint::command_eval(argc,argv);
    if(command=="local-trace")return waypoint::command_trace(argc,argv);
    if(command=="local-teacher")return waypoint::command_teacher(argc,argv);
    if(command=="local-bc")return waypoint::command_bc(argc,argv);
    if(command=="local-witness")return waypoint::command_witness(argc,argv);
    if(command=="ps-test")return waypoint::command_ps_test();
    if(command=="ps-probe")return waypoint::command_ps_probe(argc,argv);
    // Production eval/export stay reachable through the reserved main for
    // legacy-family retention and deployment export.
    if(command=="eval"||command=="export")return metal_nav_reserved_main(argc,argv);
    if(command=="--help"||command=="help") { waypoint::usage(); return 0; }
    std::cerr<<"unknown command "<<command<<"\n";waypoint::usage();return 1;
} catch(const std::exception& e) { std::cerr<<"ERROR: "<<e.what()<<"\n"; return 1; }}}
