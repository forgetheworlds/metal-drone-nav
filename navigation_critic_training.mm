// Experimental matched critic-state comparison. Actor, reward and PPO unchanged.
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

struct BankControl { uint32_t period, count; };
static_assert(sizeof(BankControl)==8,"waypoint bank control ABI");

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
struct WaypointBankControl { uint period; uint count; };
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
    const uint index=n*period+(runs[n].episodes%period);
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
    } else if(name=="open") {
        spec.name="open";spec.seed=20261008;spec.families={0};
        spec.environments=128;spec.period=1;spec.route_length_ratio_max=3.0f;
    } else if(name=="clutter") {
        spec.name="clutter";spec.seed=20261009;spec.families={1,2,4,5,14,15,16};
        spec.environments=128;spec.period=1;spec.route_length_ratio_max=3.0f;
    } else if(name=="dev-r1") {
        // Frozen selector bank for the collision-penalty comparison. Same
        // original local generator and distribution as dev-a/dev-b, a new
        // independent seed, no current-view requirement anywhere.
        spec.name="dev-r1";spec.seed=20261101;spec.families={2,4,5,14,15,16};
        spec.environments=128;spec.period=1;spec.route_length_ratio_max=2.5f;
    } else if(name=="dev-r2") {
        // Frozen nonselector bank: never used for checkpoint selection.
        spec.name="dev-r2";spec.seed=20261102;spec.families={2,4,5,14,15,16};
        spec.environments=128;spec.period=1;spec.route_length_ratio_max=2.5f;
    } else {
        throw std::runtime_error("unknown bank spec '"+name+"' (source|dev-a|dev-b|open|clutter|dev-r1|dev-r2)");
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
struct LocalRoute {
    bool found=false;
    float length_m=0.0f;
    float clearance_m=0.0f;
    uint32_t segments=0;
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
        return detour;
    }
    if(lattice_budget>0) {
        lattice_budget--;
        return route_via_waypoint(world,start,goal);
    }
    return LocalRoute{};
}

// Build one deterministic bank. A rejected task is retried with new scene and
// endpoint draws; a persistent failure is a hard error, never a silent filler.
static std::vector<BankEntry> build_bank(const BankSpec& spec,std::ostream* csv) {
    NavigationTaskConfig witness_config{};
    witness_config.minimum_witness_clearance_m=kMinimumWitnessClearanceM;
    witness_config.require_detour=0u;
    HostRng rng(spec.seed);
    const size_t count=size_t(spec.environments)*spec.period;
    std::vector<BankEntry> bank;bank.reserve(count);
    std::vector<uint32_t> route_counts(3,0),family_counts(17,0),class_counts(3,0);
    uint32_t fallbacks=0;
    // class_mode: 0 any draw, 1 the straight line must be blocked (a real
    // local detour), 2 the straight line must already be clear.
    const auto try_accept=[&](const uint32_t family,const int class_mode,BankEntry& entry) {
        uint32_t lattice_budget=8;
        for(uint32_t scene_attempt=1;scene_attempt<=16;scene_attempt++) {
            entry=BankEntry{};
            const uint32_t scene_seed=rng.raw();
            wgenerate(entry.world,scene_seed,family,spec.scene_distance);
            const WVec start=wv(kStartLow[0]+(kStartHigh[0]-kStartLow[0])*rng.uniform(),
                                kStartLow[1]+(kStartHigh[1]-kStartLow[1])*rng.uniform(),
                                kStartLow[2]+(kStartHigh[2]-kStartLow[2])*rng.uniform());
            if(wclearance(entry.world,start,0.0f)<=kEndpointClearanceM)continue;
            for(uint32_t direction_attempt=1;direction_attempt<=64;direction_attempt++) {
                WVec direction=rng.unit();
                // Blocked-class draws aim into the procedural field: the
                // geometry always sits ahead of the start region, so a
                // backward draw almost never produces a detour.
                if(class_mode==1&&direction.x<0.0f&&rng.uniform()<0.75f)direction.x=-direction.x;
                const float distance=spec.goal_min_m+(spec.goal_max_m-spec.goal_min_m)*rng.uniform();
                const WVec goal=wa(start,wm(direction,distance));
                if(!inside_bounds(goal,kGoalLow,kGoalHigh))continue;
                if(wclearance(entry.world,goal,0.0f)<=kEndpointClearanceM)continue;
                const float direct=navigation_task_segment_clearance(entry.world,start,goal);
                const bool blocked=direct<=kMinimumWitnessClearanceM;
                if(class_mode==1&&!blocked)continue;
                if(class_mode==2&&blocked)continue;
                LocalRoute route;
                if(!blocked)route=route_direct(entry.world,start,goal);
                else route=find_local_route(entry.world,start,goal,witness_config,lattice_budget);
                if(!route.found)continue;
                if(route.length_m>spec.route_length_ratio_max*std::max(distance,1e-3f))continue;
                entry.family=family;entry.scene_seed=scene_seed;
                entry.start_position[0]=start.x;entry.start_position[1]=start.y;entry.start_position[2]=start.z;
                entry.goal_position[0]=goal.x;entry.goal_position[1]=goal.y;entry.goal_position[2]=goal.z;
                entry.world.goal[0]=goal.x;entry.world.goal[1]=goal.y;entry.world.goal[2]=goal.z;
                entry.start_yaw=rng.symmetric()*3.14159265359f;
                const float speed=spec.start_velocity_max_mps*rng.uniform();
                entry.start_velocity[0]=direction.x*speed;
                entry.start_velocity[1]=direction.y*speed;
                entry.start_velocity[2]=direction.z*speed;
                entry.clearance_start=wclearance(entry.world,start,0.0f);
                entry.clearance_goal=wclearance(entry.world,goal,0.0f);
                entry.direct_clearance=direct;
                entry.witness_clearance=route.clearance_m;
                entry.witness_length=route.length_m;
                entry.initial_distance=wl(ws(goal,start));
                entry.route_class=blocked?1u:0u;
                entry.attempts=uint32_t(entry.route_class==1u?2u:1u);
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
        BankEntry entry{};
        int achieved_class=preferred;
        bool accepted=try_accept(family,preferred,entry);
        if(!accepted) { achieved_class=0; accepted=try_accept(family,0,entry); fallbacks++; }
        require(accepted,"bank generation exhausted for "+spec.name+" index "+std::to_string(index)+
                " family "+std::to_string(family));
        class_counts[achieved_class]++;
        route_counts[entry.route_class]++;
        if(entry.family<family_counts.size())family_counts[entry.family]++;
        bank.push_back(entry);
        if(csv) {
            const BankEntry& e=entry;
            *csv<<index<<','<<index/spec.period<<','<<index%spec.period<<','<<e.family<<','
                <<family_name(e.family)<<','<<e.scene_seed<<','<<e.route_class<<','<<e.attempts;
            for(int axis=0;axis<3;axis++)*csv<<','<<e.start_position[axis];
            *csv<<','<<e.start_yaw;
            for(int axis=0;axis<3;axis++)*csv<<','<<e.start_velocity[axis];
            for(int axis=0;axis<3;axis++)*csv<<','<<e.goal_position[axis];
            *csv<<','<<e.initial_distance<<','<<e.clearance_start<<','<<e.clearance_goal
                <<','<<e.direct_clearance<<','<<e.witness_clearance<<','<<e.witness_length<<'\n';
        }
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
static NavigationTaskControl local_task_recipe(float time_cost_per_s=0.2f,float contact_penalty=10.0f,
                                               float arrival_bonus=10.0f) {
    NavigationTaskControl control=navigation_training::task_settings(NAV_TASK_STAGE_OPEN_GOAL,0u);
    control.config.goal_distance_min_m=1.0f;
    control.config.goal_distance_max_m=3.0f;
    // 0.2 is the verified navigation-task recipe. Anything else is an explicit
    // diagnostic arm and is recorded in the training provenance.
    control.config.time_cost_per_s=time_cost_per_s;
    // contact_penalty defaults to the source10.0f. A non-default value is the
    // one scalar the collision-penalty comparison varies; it is recorded in
    // the training contract and never silently changed.
    control.config.contact_penalty=contact_penalty;
    // arrival_bonus defaults to the source10.0f. A non-default value is the
    // one scalar decision2 varies (stall-vs-success ordering); recorded too.
    control.config.stable_arrival_bonus=arrival_bonus;
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
};

static LocalRun make_local_run(Sim& sim,const std::vector<BankEntry>& bank,const BankControl& control,
                               id<MTLComputePipelineState> apply_pipeline,float time_cost_per_s=0.2f,
                               float contact_penalty=10.0f,float arrival_bonus=10.0f) {
    enable_task_control(sim,local_task_recipe(time_cost_per_s,contact_penalty,arrival_bonus));
    LocalRun run{sim,nullptr,nullptr,apply_pipeline};
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
                           uint32_t seed,uint32_t mode) {
    (void)split; // Bank spec names the split in each recorded row.
    if(path.empty())return;
    const std::filesystem::path destination(path);
    if(destination.has_parent_path())std::filesystem::create_directories(destination.parent_path());
    std::ofstream file(path);
    require(bool(file),"cannot write evaluation CSV "+path);
    file<<std::setprecision(9);
    file<<"split,env,family,family_name,route_class,scene_seed,seed,mode,sensor_delay,command_delay,"
          "start_x,start_y,start_z,start_yaw,"
          "goal_x,goal_y,goal_z,goal_distance_m,direct_clearance_m,witness_clearance_m,witness_length_m,"
          "route_ratio,success,collision,timeout,time_s,path_m,mean_speed_mps,peak_speed_mps,min_clearance_m,"
          "final_x,final_y,final_z,final_distance_m,final_speed_mps,stable_hold_s\n";
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
            <<','<<tasks[env].stable_time_s<<'\n';
    }
}

// Run one bank split exactly once per environment with the given inference
// mode and return the aggregate score plus an optional per-episode CSV.
static BankScore evaluate_bank(Metal& metal,const float* actor,const BankSpec* spec,
                               const std::vector<BankEntry>& bank,const BankControl& control,
                               uint32_t mode,uint32_t seed,float speed,const std::string& csv_path,
                               const std::string& split,uint32_t environments=128,uint32_t max_steps=400,
                               uint32_t sensor_delay=0,uint32_t command_delay=0,float contact_penalty=10.0f,
                               float arrival_bonus=10.0f) {
    SimConfig config;
    config.n=environments;config.mode=mode;config.family=0;config.eval=1;config.seed=seed;
    config.speed=speed;config.distance=4;config.max_steps=max_steps;config.geometry_memory=1;
    config.sensor_delay=sensor_delay;config.command_delay=command_delay;
    require(sensor_delay<=6&&command_delay<=7,"bank evaluation delay rings support <=6 sensor frames and <=7 command steps");
    require(bank.size()==size_t(environments)*control.period,"evaluation bank does not match the environment count");
    Sim sim(metal,config,32);
    if(actor)std::memcpy(sim.actor.contents,actor,fixed_ppo::actor_param_count*4);
    LocalRun run=make_local_run(sim,bank,control,metal.pipeline("waypoint_task_apply"),0.2f,contact_penalty,
                                arrival_bonus);
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
    write_eval_csv(csv_path,split,spec,bank,control,sim,seed,mode);
    // Per-eval receipt: the exact bank bytes this flight set was graded on.
    const std::string bank_entry_sha=ppo_safeguard_sha256(bank.data(),bank.size()*sizeof(BankEntry));
    std::cout<<"bank_eval split="<<split<<" mode="<<mode<<" seed="<<seed<<" tasks="<<config.n
             <<" success="<<score.success<<" collision="<<score.collision<<" timeout="<<score.timeout
             <<" mean_arrival_s="<<score.mean_time_s<<" mean_path_m="<<score.mean_path_m
             <<" min_clearance_m="<<score.min_clearance_m<<" gpu_s="<<gpu
             <<" contact_penalty="<<contact_penalty
             <<" bank_entry_sha256="<<bank_entry_sha<<"\n";
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
        "              [--time-cost X] [--contact-penalty X] [--arrival-bonus X]\n"
        "  local-eval  CHECKPOINT OUT_CSV [--spec NAME] [--bank PATH] [--mode N] [--seed N]\n"
        "              [--envs N] [--max-steps N] [--sensor-delay N] [--command-delay N] [--selector 0|1]\n"
        "  local-trace CHECKPOINT OUT_CSV [--spec NAME] [--bank PATH] [--mode N] [--seed N] [--envs N]\n"
        "  local-reward-audit CHECKPOINT OUT_CSV [--spec NAME] [--bank PATH] [--mode N] [--seed N]\n"
        "              [--envs N] [--max-steps N] [--time-cost X] [--contact-penalty X] [--arrival-bonus X]\n"
        "              [--zero-actor]\n"
        "specs: source dev-a dev-b open clutter dev-r1 dev-r2\n";
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
    metal.compile(base_source()+PPO_TRAINER_MSL+kWaypointKernels);
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
    for(const char* name:{"source","dev-a","dev-b","open","clutter","dev-r1","dev-r2"}) {
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
    csv<<"index,env,slot,family,family_name,scene_seed,route_class,attempts,"
         "start_x,start_y,start_z,start_yaw,start_velocity_x,start_velocity_y,start_velocity_z,"
         "goal_x,goal_y,goal_z,initial_distance_m,clearance_start_m,clearance_goal_m,"
         "direct_clearance_m,witness_clearance_m,witness_length_m\n";
    const auto bank=build_bank(spec,&csv);
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
    float learning_rate=0,entropy_coef=0,risk_coef=0,value_coef=0,speed=0,time_cost=0,warmstart_log_std=0;
    float contact_penalty=0;
    float arrival_bonus=0;
};

// The exact Metal source text this binary compiles at run time. Pinning only
// the runner file would leave world.hpp, sim.metal, ppo.metal, guidance.hpp and
// the sensor profile free to change under a resumed checkpoint.
static std::string contract_core_sha256() {
    const std::string source=base_source()+PPO_TRAINER_MSL+kWaypointKernels;
    return ppo_safeguard_sha256(source.data(),source.size());
}
static std::string contract_runner_sha256() {
    return challenge_evaluation::sha256_file(std::string(SOURCE_DIR)+"/navigation_critic_training.mm");
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
    require_json_float(text,"contact_penalty",expected.contact_penalty);
    require_json_float(text,"arrival_bonus",expected.arrival_bonus);
    require_json_float(text,"warmstart_log_std",expected.warmstart_log_std);
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
       <<"  \"contact_penalty\":"<<contract.contact_penalty<<",\n"
       <<"  \"arrival_bonus\":"<<contract.arrival_bonus<<",\n"
       <<"  \"warmstart_log_std\":"<<contract.warmstart_log_std<<",\n"
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
        <<"  \"runner_file\":\"navigation_waypoint_training.mm\",\n"
        <<"  \"runner_sha256\":\""<<runner<<"\",\n"
        <<"  \"core_source_sha256\":\""<<core<<"\",\n"
        <<"  \"core_source_bytes\":"<<(base_source()+PPO_TRAINER_MSL+kWaypointKernels).size()<<"\n"
        <<"}\n";
    std::cout<<json.str();
    if(argc>2) {
        std::ofstream out(argv[2]);require(bool(out),"cannot write contract "+std::string(argv[2]));
        out<<json.str();out.flush();require(bool(out),"contract write failed");
        std::cout<<"contract_write "<<argv[2]<<"\n";
    }
    return 0;
}

struct TrainingOptions {
    BankSpec spec;
    std::string bank_path;
    uint32_t seed=20261004,epochs=2,environments=128,horizon=32,eval_every=50;
    float log_std=-1.0f,learning_rate=0.0001f,entropy=0.001f,speed=1.5f,time_cost=0.2f;
    float contact_penalty=10.0f;
    float arrival_bonus=10.0f;
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
        else if(option=="--seed")options.seed=option_uint(argc,argv,index,option);
        else if(option=="--epochs")options.epochs=option_uint(argc,argv,index,option);
        else if(option=="--envs")options.environments=option_uint(argc,argv,index,option);
        else if(option=="--horizon")options.horizon=option_uint(argc,argv,index,option);
        else if(option=="--logstd")options.log_std=option_float(argc,argv,index,option);
        else if(option=="--learning-rate")options.learning_rate=option_float(argc,argv,index,option);
        else if(option=="--entropy")options.entropy=option_float(argc,argv,index,option);
        else if(option=="--speed")options.speed=option_float(argc,argv,index,option);
        else if(option=="--time-cost")options.time_cost=option_float(argc,argv,index,option);
        else if(option=="--contact-penalty")options.contact_penalty=option_float(argc,argv,index,option);
        else if(option=="--arrival-bonus")options.arrival_bonus=option_float(argc,argv,index,option);
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
        bank_csv<<"index,env,slot,family,family_name,scene_seed,route_class,attempts,"
                  "start_x,start_y,start_z,start_yaw,start_velocity_x,start_velocity_y,start_velocity_z,"
                  "goal_x,goal_y,goal_z,initial_distance_m,clearance_start_m,clearance_goal_m,"
                  "direct_clearance_m,witness_clearance_m,witness_length_m\n";
        bank=build_bank(options.spec,&bank_csv);
        bank_sha=ppo_safeguard_sha256(bank.data(),bank.size()*sizeof(BankEntry));
    } else {
        bank=read_bank(options.bank_path,control,bank_sha);
        require(size_t(control.count)==size_t(options.environments)*control.period,
                "bank file does not match --envs/period");
    }

    BankControl eval_control{};
    std::vector<BankEntry> eval_bank;
    std::string eval_bank_sha="none";
    std::ostringstream eval_bank_csv;
    if(options.eval_spec!="none") {
        options.eval_bank_spec=spec_by_name(options.eval_spec);
        const BankSpec& eval_spec=options.eval_bank_spec;
        eval_bank_csv<<std::setprecision(9);
        eval_bank_csv<<"index,env,slot,family,family_name,scene_seed,route_class,attempts,"
                       "start_x,start_y,start_z,start_yaw,start_velocity_x,start_velocity_y,start_velocity_z,"
                       "goal_x,goal_y,goal_z,initial_distance_m,clearance_start_m,clearance_goal_m,"
                       "direct_clearance_m,witness_clearance_m,witness_length_m\n";
        eval_bank=build_bank(eval_spec,&eval_bank_csv);
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
    contract.contact_penalty=options.contact_penalty;
    contract.arrival_bonus=options.arrival_bonus;
    contract.warmstart_log_std=options.log_std;

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
    metal.compile(base_source()+PPO_TRAINER_MSL+kWaypointKernels);
    auto apply_pipeline=metal.pipeline("waypoint_task_apply");

    SimConfig config;
    config.n=options.environments;config.mode=22;config.family=0;config.seed=options.seed;
    config.speed=options.speed;config.distance=4;config.max_steps=400;config.geometry_memory=1;
    config.entropy_coef=options.entropy;config.learning_rate=options.learning_rate;
    Sim sim(metal,config,options.horizon);
    LocalRun run=make_local_run(sim,bank,control,apply_pipeline,options.time_cost,options.contact_penalty,
                                options.arrival_bonus);
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
              "actor_drift_l2,actor_grad_norm_mean,actor_grad_norm_max,actor_clip_frac,"
              "critic_grad_norm_mean,critic_grad_norm_max,critic_clip_frac\n";
    }
    BankScore best;
    best.success=-1;
    if(resume&&std::filesystem::exists(checkpoint+".best")&&!eval_bank.empty()) {
        Sim selected(metal,config,options.horizon);
        navigation_training::load_actor(selected,checkpoint+".best",false);
        best=evaluate_bank(metal,(const float*)selected.actor.contents,&options.eval_bank_spec,eval_bank,eval_control,
                           17,700001,options.speed,"",options.eval_spec,options.environments,400,0,0,
                           options.contact_penalty,options.arrival_bonus);
    }

    std::vector<SimRun> previous_runs(options.environments);
    const double started=seconds();
    const uint32_t finish=trainer.completed_rollouts+rollouts;
    std::cout<<"local_train bank="<<options.spec.name<<" environments="<<options.environments
             <<" horizon="<<options.horizon<<" epochs="<<options.epochs
             <<" lr="<<options.learning_rate<<" entropy="<<options.entropy
             <<" risk="<<config.risk_coef<<" time_cost="<<options.time_cost
             <<" contact_penalty="<<options.contact_penalty
             <<" arrival_bonus="<<options.arrival_bonus
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
        diag<<','<<trainer.actor_drift_l2();
        const PPOTrainer::GradNormSummary grad=trainer.grad_norm_summary();
        diag<<','<<grad.actor_mean<<','<<grad.actor_max<<','<<grad.actor_clip_frac
            <<','<<grad.critic_mean<<','<<grad.critic_max<<','<<grad.critic_clip_frac<<'\n';
        diag.flush();
        if((rollout+1)%options.eval_every==0||rollout+1==finish) {
            trainer.save_checkpoint(checkpoint,config.family,options.seed,rollout+1);
            BankScore score;
            if(!eval_bank.empty())
                score=evaluate_bank(metal,(const float*)sim.actor.contents,&options.eval_bank_spec,eval_bank,eval_control,
                                    17,700001,options.speed,checkpoint+".dev.csv",options.eval_spec,options.environments,
                                    400,0,0,options.contact_penalty,options.arrival_bonus);
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
    bool selector=false,spec_given=false;
    for(int index=4;index<argc;index++) {
        const std::string option=argv[index];
        if(option=="--spec") { spec_name=option_value(argc,argv,index,option); spec_given=true; }
        else if(option=="--bank")bank_path=option_value(argc,argv,index,option);
        else if(option=="--mode")mode=option_uint(argc,argv,index,option);
        else if(option=="--seed")seed=option_uint(argc,argv,index,option);
        else if(option=="--envs")environments=option_uint(argc,argv,index,option);
        else if(option=="--max-steps")max_steps=option_uint(argc,argv,index,option);
        else if(option=="--speed")speed=option_float(argc,argv,index,option);
        else if(option=="--sensor-delay")sensor_delay=option_uint(argc,argv,index,option);
        else if(option=="--command-delay")command_delay=option_uint(argc,argv,index,option);
        else if(option=="--selector")selector=option_uint(argc,argv,index,option)!=0;
        else throw std::runtime_error("unknown local-eval option "+option);
    }
    Metal metal;
    metal.compile(base_source()+PPO_TRAINER_MSL+kWaypointKernels);
    BankControl control{};std::vector<BankEntry> bank;BankSpec spec;
    if(bank_path.empty()) {
        spec=spec_by_name(spec_name);
        bank=build_bank(spec,nullptr);
        control.period=spec.period;control.count=uint32_t(bank.size());
    } else {
        std::string unused_hash;
        bank=read_bank(bank_path,control,unused_hash);
        // An explicit --spec is the human label for a file-backed bank; the
        // path stays the label only when no spec was given.
        spec.name=spec_given?spec_name:bank_path;
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
                  sensor_delay,command_delay);
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
    metal.compile(base_source()+PPO_TRAINER_MSL+kWaypointKernels);
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
    // The critic is part of the checkpoint after the actor; the value column
    // must come from the trained critic, not the constructor's random init.
    require(header.critic_count==fixed_ppo::critic_param_count,"trace checkpoint critic dimensions");
    file.read((char*)sim.critic.contents,sim.critic.length);
    require(bool(file),"trace checkpoint critic read failed");
    LocalRun run=make_local_run(sim,bank,control,metal.pipeline("waypoint_task_apply"));
    const std::filesystem::path destination(output);
    if(destination.has_parent_path())std::filesystem::create_directories(destination.parent_path());
    std::ofstream trace(output);
    require(bool(trace),"cannot write trace CSV "+output);
    trace<<std::setprecision(9);
    trace<<"tick,env,family,scene_seed,time_s,x,y,z,vx,vy,vz,yaw,goal_x,goal_y,goal_z,goal_distance_m,"
           "clearance_m,steps,episodes,terminated,critic_value\n";
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
        const float* values=(const float*)sim.values.contents;
        const size_t value_row=size_t((tick_base+step)%sim.horizon)*environments;
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
                 <<(runs[env].episodes>0?1:0)<<','<<values[value_row+env]<<'\n';
        }
    }
    std::cout<<"trace_write "<<output<<" envs="<<environments<<" steps="<<max_steps<<"\n";
    return 0;
}

// ------------------------------------------------- reward decomposition audit
//
// Records the ACTUAL reward sim_advance wrote and every
// navigation_task_step component on flown trajectories. Evaluation mode gives
// one episode per environment with no in-advance reset, so the task record,
// executed command and reward of a tick are all readable before the next tick.
// The reconstruction is checked against the kernel reward itself; a large
// mismatch means the audit (not the kernel) is wrong.
static int command_reward_audit(int argc,char** argv) {
    require(argc>=4,"local-reward-audit CHECKPOINT OUT_CSV");
    const std::string checkpoint=argv[2],output=argv[3];
    std::string spec_name="dev-r1",bank_path;
    uint32_t mode=17,seed=700001,environments=128,max_steps=400;
    float speed=1.5f,time_cost=1.0f,contact_penalty=10.0f,arrival_bonus=10.0f;
    bool zero_actor=false,spec_given=false;
    for(int index=4;index<argc;index++) {
        const std::string option=argv[index];
        if(option=="--spec") { spec_name=option_value(argc,argv,index,option); spec_given=true; }
        else if(option=="--bank")bank_path=option_value(argc,argv,index,option);
        else if(option=="--mode")mode=option_uint(argc,argv,index,option);
        else if(option=="--seed")seed=option_uint(argc,argv,index,option);
        else if(option=="--envs")environments=option_uint(argc,argv,index,option);
        else if(option=="--max-steps")max_steps=option_uint(argc,argv,index,option);
        else if(option=="--speed")speed=option_float(argc,argv,index,option);
        else if(option=="--time-cost")time_cost=option_float(argc,argv,index,option);
        else if(option=="--contact-penalty")contact_penalty=option_float(argc,argv,index,option);
        else if(option=="--arrival-bonus")arrival_bonus=option_float(argc,argv,index,option);
        else if(option=="--zero-actor")zero_actor=true;
        else throw std::runtime_error("unknown local-reward-audit option "+option);
    }
    Metal metal;
    metal.compile(base_source()+PPO_TRAINER_MSL+kWaypointKernels);
    BankControl control{};std::vector<BankEntry> bank;BankSpec spec;
    if(bank_path.empty()) {
        spec=spec_by_name(spec_name);
        bank=build_bank(spec,nullptr);
        control.period=spec.period;control.count=uint32_t(bank.size());
    } else {
        std::string unused_hash;
        bank=read_bank(bank_path,control,unused_hash);
        spec.name=spec_given?spec_name:bank_path;
    }
    require(control.count==size_t(environments)*control.period,"reward-audit bank does not match --envs");
    SimConfig config;
    config.n=environments;config.mode=mode;config.family=0;config.eval=1;config.seed=seed;
    config.speed=speed;config.distance=4;config.max_steps=max_steps;config.geometry_memory=1;
    Sim sim(metal,config,32);
    if(zero_actor) {
        // All-zero actor in mode 4 executes tanh(0)=0: a flown hover path.
        std::memset(sim.actor.contents,0,sim.actor.length);
    } else {
        std::ifstream file(checkpoint,std::ios::binary);
        const PpoCheckpointHeader header=read_checkpoint_header(file);
        require(header.actor_count==fixed_ppo::actor_param_count,"reward-audit checkpoint dimensions");
        std::vector<float> actor(header.actor_count);
        file.read(reinterpret_cast<char*>(actor.data()),std::streamsize(actor.size()*sizeof(float)));
        require(bool(file),"reward-audit actor read failed");
        std::memcpy(sim.actor.contents,actor.data(),actor.size()*sizeof(float));
    }
    LocalRun run=make_local_run(sim,bank,control,metal.pipeline("waypoint_task_apply"),time_cost,contact_penalty,
                                arrival_bonus);
    // Decompose with the exact reward constants the kernel is using.
    const NavigationTaskConfig& recipe=static_cast<const NavigationTaskControl*>(sim.task_control.contents)->config;
    require(std::fabs(recipe.time_cost_per_s-time_cost)<1e-6f&&
            std::fabs(recipe.contact_penalty-contact_penalty)<1e-6f&&
            std::fabs(recipe.stable_arrival_bonus-arrival_bonus)<1e-6f,
            "reward-audit recipe does not match the requested reward settings");
    const std::filesystem::path destination(output);
    if(destination.has_parent_path())std::filesystem::create_directories(destination.parent_path());
    std::ofstream csv(output);
    require(bool(csv),"cannot write reward audit CSV "+output);
    csv<<std::setprecision(9);
    csv<<"env,family,scene_seed,route_class,step,d_before_m,d_after_m,progress_m,"
          "reward_kernel,reward_progress,reward_time,reward_smooth,reward_contact,reward_bonus,"
          "contact,success,timeout,executed_x,executed_y,executed_z,executed_yaw,"
          "reconstructed,recon_abs_err\n";
    auto probe=[metal.queue commandBuffer];
    local_probe(run,probe);
    metal.finish(probe);
    verify_reset(run,bank,control,true);
    std::vector<float> prev_distance(environments),prev_executed(environments*4);
    std::vector<uint32_t> prev_episodes(environments,0),prev_collisions(environments,0),
                          prev_successes(environments,0),prev_timeouts(environments,0);
    std::vector<uint64_t> episode_rows(environments,0);
    std::vector<double> episode_return(environments,0.0);
    std::vector<uint32_t> episode_contacts(environments,0);
    {
        for(uint32_t env=0;env<environments;env++) {
            // Exactly navigation_task_step's previous_distance_m at episode
            // start, and its last executed command before the first tick.
            const BankEntry& entry=bank[size_t(env)*control.period];
            prev_distance[env]=entry.initial_distance;
            for(int axis=0;axis<3;axis++)prev_executed[env*4+axis]=entry.start_velocity[axis];
            prev_executed[env*4+3]=0.0f;
        }
    }
    const uint32_t tick_base=sim.cfg.tick;
    uint64_t rows_written=0;double max_abs_error=0.0;uint64_t error_rows=0;
    for(uint32_t step=0;step<max_steps;step++) {
        auto commands=[metal.queue commandBuffer];
        local_tick(run,commands,tick_base+step,step==0);
        metal.finish(commands);
        const auto* states=(const RLPhysicsState*)sim.states.contents;
        const auto* runs=(const SimRun*)sim.runs.contents;
        const auto* tasks=(const NavigationTaskState*)sim.task_states.contents;
        const float* rewards=(const float*)sim.rewards.contents;
        const size_t row=size_t((tick_base+step)%sim.horizon)*environments;
        bool active=false;
        for(uint32_t env=0;env<environments;env++) {
            if(prev_episodes[env]!=0)continue;
            active=true;
            const SimRun& flight=runs[env];
            const BankEntry& entry=bank[size_t(env)*control.period];
            const bool terminal_tick=flight.episodes!=0;
            const bool contact=flight.collisions!=prev_collisions[env];
            const bool success=flight.successes!=prev_successes[env];
            const bool timeout=flight.timeouts!=prev_timeouts[env];
            float executed[4]={flight.desired_velocity[0],flight.desired_velocity[1],
                               flight.desired_velocity[2],flight.previous_nav[3]*0.5f};
            // Same float expression as navigation_task_step.
            const float gx=entry.goal_position[0]-states[env].position[0];
            const float gy=entry.goal_position[1]-states[env].position[1];
            const float gz=entry.goal_position[2]-states[env].position[2];
            const float distance_after=std::sqrt(gx*gx+gy*gy+gz*gz);
            const float progress=prev_distance[env]-distance_after;
            float difference=0.0f;
            for(int axis=0;axis<4;axis++) {
                const float d=executed[axis]-prev_executed[env*4+axis];
                difference+=d*d;
            }
            const float reward_progress=recipe.progress_reward_per_m*progress;
            const float reward_time=-recipe.time_cost_per_s*recipe.nav_period_s;
            const float reward_smooth=-recipe.command_smoothness_weight*difference*recipe.nav_period_s;
            const float reward_contact=contact?-recipe.contact_penalty:0.0f;
            const float reward_bonus=(success&&!contact)?recipe.stable_arrival_bonus:0.0f;
            const float reconstructed=reward_progress+reward_time+reward_smooth+reward_contact+reward_bonus;
            const float kernel_reward=rewards[row+env];
            const double error=std::fabs(double(kernel_reward)-double(reconstructed));
            if(!(error<=1e-4)) { error_rows++; }
            if(error>max_abs_error)max_abs_error=error;
            csv<<env<<','<<entry.family<<','<<entry.scene_seed<<','<<entry.route_class<<','
               <<tasks[env].step_count<<','<<prev_distance[env]<<','<<distance_after<<','<<progress<<','
               <<kernel_reward<<','<<reward_progress<<','<<reward_time<<','<<reward_smooth<<','
               <<reward_contact<<','<<reward_bonus<<','
               <<(contact?1:0)<<','<<(success?1:0)<<','<<(timeout?1:0)<<','
               <<executed[0]<<','<<executed[1]<<','<<executed[2]<<','<<executed[3]<<','
               <<reconstructed<<','<<error<<'\n';
            rows_written++;
            episode_rows[env]++;episode_return[env]+=double(kernel_reward);
            episode_contacts[env]+=contact?1u:0u;
            prev_distance[env]=distance_after;
            for(int axis=0;axis<4;axis++)prev_executed[env*4+axis]=executed[axis];
            prev_collisions[env]=flight.collisions;prev_successes[env]=flight.successes;
            prev_timeouts[env]=flight.timeouts;
            if(terminal_tick)prev_episodes[env]=flight.episodes;
        }
        if(!active)break;
    }
    csv.flush();require(bool(csv),"reward audit write failed: "+output);
    uint64_t episodes=0,contacts=0,successes=0,timeouts=0,active_rows=0;
    double return_sum=0.0;
    for(uint32_t env=0;env<environments;env++) {
        if(prev_episodes[env]==0)continue;
        episodes++;contacts+=episode_contacts[env];return_sum+=episode_return[env];
        active_rows+=episode_rows[env];
        successes+=prev_successes[env];timeouts+=prev_timeouts[env];
    }
    std::cout<<"reward_audit output="<<output<<" mode="<<mode<<" seed="<<seed
             <<" time_cost="<<recipe.time_cost_per_s<<" contact_penalty="<<recipe.contact_penalty
             <<" rows="<<rows_written<<" episodes="<<episodes<<" contacts="<<contacts
             <<" successes="<<successes<<" timeouts="<<timeouts
             <<" mean_kernel_return="<<(episodes?return_sum/double(episodes):0.0)
             <<" recon_rows_over_1e-4="<<error_rows<<" recon_max_abs_err="<<max_abs_error<<"\n";
    return 0;
}

static int command_critic_state_test() {
#if FIXED_PPO_CRITIC_OBS_DIM == 64
    static_assert(fixed_ppo::critic_obs_dim==64,"critic-state test uses64 inputs");
    Metal metal;metal.compile(base_source()+PPO_TRAINER_MSL+kWaypointKernels);
    SimConfig cfg;cfg.n=4;cfg.eval=1;cfg.mode=17;cfg.seed=700001;cfg.speed=1.5f;cfg.max_steps=400;cfg.geometry_memory=1;
    Sim sim(metal,cfg,32);const auto spec=spec_by_name("open");BankControl bc{};
    const auto bank=build_bank(spec,nullptr);bc.period=spec.period;bc.count=uint32_t(bank.size());
    LocalRun run=make_local_run(sim,bank,bc,metal.pipeline("waypoint_task_apply"),1.0f,50.0f,10.0f);
    navigation_training::load_actor(sim,std::string(SOURCE_DIR)+"/results/omp-local-capability/runs/bc.bin",true);
    auto cb=[metal.queue commandBuffer];local_probe(run,cb);metal.finish(cb);
    std::vector<float> original_obs(cfg.n*fixed_ppo::actor_obs_dim),original_critic(cfg.n*64);
    std::memcpy(original_obs.data(),sim.obs.contents,original_obs.size()*4);
    std::memcpy(original_critic.data(),sim.co.contents,original_critic.size()*4);
    auto* states=(RLPhysicsState*)sim.states.contents;auto* runs=(SimRun*)sim.runs.contents;
    for(uint n=0;n<cfg.n;n++){
        for(uint j=0;j<4;j++)states[n].rpm[j]=0.8f;
        for(uint j=0;j<16;j++)runs[n].hidden[j]=0.3f;
        runs[n].yaw+=0.8f;
    }
    cb=[metal.queue commandBuffer];sim.m.dispatch(cb,sim.observe_p,cfg.n,
        {sim.states,sim.runs,sim.worlds,sim.sensors,sim.obs,sim.co,sim.physics,sim.configs[0],sim.poses,sim.memory_clearances},64);metal.finish(cb);
    const float* obs=(const float*)sim.obs.contents;const float* co=(const float*)sim.co.contents;
    require(std::memcmp(obs,original_obs.data(),original_obs.size()*4)==0,"actor inputs must not acquire privileged controller state");
    float extra_delta=0;
    for(uint n=0;n<cfg.n;n++){
        require(std::memcmp(co+n*64,original_critic.data()+n*64,32*4)==0,"base32 critic aliases must remain equal");
        for(uint j=32;j<64;j++)extra_delta=std::max(extra_delta,std::fabs(co[n*64+j]-original_critic[n*64+j]));
    }
#ifdef NAV_CRITIC_CONTROL_STATE
    if(NAV_CRITIC_CONTROL_STATE)require(extra_delta>0.1f,"rich critic must distinguish motor/controller states");
    else require(extra_delta==0,"control extra features must be zero");
#endif
    std::cout<<"critic_state_test PASS actor_bytes_equal=1 base32_bytes_equal=1 extra_delta="<<extra_delta
             <<" critic_inputs="<<fixed_ppo::critic_obs_dim<<" actor_inputs="<<fixed_ppo::actor_obs_dim<<"\n";
    return 0;
#else
    throw std::runtime_error("critic-state-test requires a64-input critic build");
#endif
}

} // namespace waypoint

#ifndef WAYPOINT_EMBEDDED

int main(int argc,char** argv) {@autoreleasepool {try {
    // Training logs must survive a crash and interleaved shell pipes.
    std::setvbuf(stdout,nullptr,_IONBF,0);
    std::setvbuf(stderr,nullptr,_IONBF,0);
    if(argc<2) { waypoint::usage(); return 0; }
    const std::string command=argv[1];
    if(command=="local-bank")return waypoint::command_bank(argc,argv);
    if(command=="local-test")return waypoint::command_test();
    if(command=="critic-state-test")return waypoint::command_critic_state_test();
    if(command=="local-contract")return waypoint::command_contract(argc,argv);
    if(command=="local-train")return waypoint::command_train(argc,argv);
    if(command=="local-eval")return waypoint::command_eval(argc,argv);
    if(command=="local-trace")return waypoint::command_trace(argc,argv);
    if(command=="local-reward-audit")return waypoint::command_reward_audit(argc,argv);
    if(command=="--help"||command=="help") { waypoint::usage(); return 0; }
    std::cerr<<"unknown command "<<command<<"\n";waypoint::usage();return 1;
} catch(const std::exception& e) { std::cerr<<"ERROR: "<<e.what()<<"\n"; return 1; }}}

#endif
