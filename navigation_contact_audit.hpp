#pragma once

#include "vehicle_geometry.hpp"

// Read-only audit of the vehicle state saved in a PPO checkpoint. This reports
// one saved snapshot per environment; it does not reconstruct full-flight
// contact outcomes or alter training/evaluation scores.

[[maybe_unused]] static void audit_navigation_checkpoint_contacts(
    Metal& metal,const std::string& checkpoint_path,const std::string& output_path) {
    require(!checkpoint_path.empty()&&!output_path.empty(),"contact audit needs checkpoint and output paths");
    const std::string checkpoint_hash=challenge_evaluation::sha256_file(checkpoint_path);
    std::ifstream checkpoint(checkpoint_path,std::ios::binary);
    require(bool(checkpoint),"cannot open contact-audit checkpoint: "+checkpoint_path);
    const PpoCheckpointHeader header=read_checkpoint_header(checkpoint);
    require(header.version>=3&&header.version<=9,"contact audit supports PPO checkpoint versions 3..9");
    require(header.actor_count==fixed_ppo::actor_param_count&&
            header.critic_count==fixed_ppo::critic_param_count,
            "contact-audit checkpoint parameter dimensions do not match this build");
    require(header.n>0&&header.n<=32768,"contact-audit checkpoint environment count is unsupported");

    const std::streampos payload_start=checkpoint.tellg();
    require(payload_start>=0,"cannot locate contact-audit checkpoint payload");
    const uint64_t optimizer_float_count=3ULL*(uint64_t(header.actor_count)+uint64_t(header.critic_count));
    require(optimizer_float_count<=uint64_t(std::numeric_limits<std::streamoff>::max())/sizeof(float),
            "contact-audit parameter payload size overflow");
    checkpoint.seekg(std::streamoff(optimizer_float_count*sizeof(float)),std::ios::cur);
    require(bool(checkpoint),"checkpoint ends before parameters and Adam moments");

    const size_t count=header.n;
    std::vector<RLPhysicsState> states(count);
    std::vector<SimRun> runs(count);
    std::vector<WWorld> worlds(count);
    auto read_snapshot=[&](void* data,size_t bytes,const char* field) {
        checkpoint.read(static_cast<char*>(data),std::streamsize(bytes));
        require(bool(checkpoint),std::string("truncated contact-audit snapshot: ")+field);
    };
    read_snapshot(states.data(),states.size()*sizeof(RLPhysicsState),"states");
    read_snapshot(runs.data(),runs.size()*sizeof(SimRun),"runs");
    read_snapshot(worlds.data(),worlds.size()*sizeof(WWorld),"worlds");

    std::vector<float> elapsed(count);
    for(size_t n=0;n<count;n++) {
        const float* state_values=reinterpret_cast<const float*>(&states[n]);
        for(size_t j=0;j<17;j++)
            require(std::isfinite(state_values[j]),"non-finite saved physics state at environment "+std::to_string(n));
        require(std::isfinite(runs[n].elapsed)&&runs[n].elapsed>=0.0f,
                "invalid saved elapsed time at environment "+std::to_string(n));
        require(worlds[n].count<=16,"invalid saved obstacle count at environment "+std::to_string(n));
        for(uint32_t j=0;j<worlds[n].count;j++) {
            const WObstacle& obstacle=worlds[n].obstacles[j];
            require(obstacle.kind<=2,"unsupported saved obstacle kind at environment "+std::to_string(n));
            for(uint32_t axis=0;axis<3;axis++)
                require(std::isfinite(obstacle.center[axis])&&std::isfinite(obstacle.size[axis])&&
                        std::isfinite(obstacle.velocity[axis])&&obstacle.size[axis]>=0.0f,
                        "invalid saved obstacle at environment "+std::to_string(n));
        }
        elapsed[n]=runs[n].elapsed;
    }

    const float safety_margin_m=VG_DEFAULT_SAFETY_MARGIN_M;
    const uint32_t emit_legacy_sphere=1;
    std::vector<VGResult> gpu_results(count),cpu_results(count);
    std::vector<float> gpu_legacy_clearance(count),cpu_legacy_clearance(count);

    // The checkpoint has the live state and world but no observation frames are
    // needed for this geometry-only audit. Compile the shared query and its
    // audit kernel alongside the normal simulator source.
    metal.compile(base_source()+read_text(std::string(SOURCE_DIR)+"/vehicle_geometry.hpp")+
                  read_text(std::string(SOURCE_DIR)+"/navigation_contact.metal"));
    auto states_buffer=metal.buffer(states.size()*sizeof(RLPhysicsState),states.data());
    auto worlds_buffer=metal.buffer(worlds.size()*sizeof(WWorld),worlds.data());
    auto elapsed_buffer=metal.buffer(elapsed.size()*sizeof(float),elapsed.data());
    auto margin_buffer=metal.buffer(sizeof(safety_margin_m),&safety_margin_m);
    auto result_buffer=metal.buffer(gpu_results.size()*sizeof(VGResult));
    auto legacy_buffer=metal.buffer(gpu_legacy_clearance.size()*sizeof(float));
    auto legacy_flag_buffer=metal.buffer(sizeof(emit_legacy_sphere),&emit_legacy_sphere);
    auto commands=[metal.queue commandBuffer];
    metal.dispatch(commands,metal.pipeline("navigation_contact_audit"),count,
                   {states_buffer,worlds_buffer,elapsed_buffer,margin_buffer,result_buffer,
                    legacy_buffer,legacy_flag_buffer},64);
    metal.finish(commands);
    std::memcpy(gpu_results.data(),result_buffer.contents,gpu_results.size()*sizeof(VGResult));
    std::memcpy(gpu_legacy_clearance.data(),legacy_buffer.contents,
                gpu_legacy_clearance.size()*sizeof(float));

    constexpr float parity_tolerance_m=2e-4f;
    float max_clearance_error=0.0f,max_legacy_error=0.0f;
    uint64_t declared_contacts=0,margin_violations=0,legacy_sphere_violations=0,ambiguous_results=0;
    for(size_t n=0;n<count;n++) {
        vg_world_contact(states[n].position,states[n].orientation_wxyz,worlds[n],elapsed[n],
                         safety_margin_m,cpu_results[n]);
        cpu_legacy_clearance[n]=wclearance(
            worlds[n],wv(states[n].position[0],states[n].position[1],states[n].position[2]),elapsed[n]);
        const VGResult& gpu=gpu_results[n];
        const VGResult& cpu=cpu_results[n];
        require(gpu.contact<=1&&gpu.ambiguous<=1,"invalid GPU contact flags at environment "+std::to_string(n));
        require(std::isfinite(gpu.physical_clearance_m)&&std::isfinite(gpu.safety_clearance_m)&&
                std::isfinite(gpu_legacy_clearance[n]),
                "non-finite GPU contact metric at environment "+std::to_string(n));
        require(gpu.physical_clearance_m>=0.0f&&
                std::fabs(gpu.safety_clearance_m-(gpu.physical_clearance_m-safety_margin_m))<=1e-5f,
                "inconsistent physical/safety clearance at environment "+std::to_string(n));
        require(!gpu.ambiguous||gpu.contact,"ambiguous query did not fail closed at environment "+std::to_string(n));
        const float clearance_error=std::fabs(gpu.physical_clearance_m-cpu.physical_clearance_m);
        const float legacy_error=std::fabs(gpu_legacy_clearance[n]-cpu_legacy_clearance[n]);
        max_clearance_error=std::max(max_clearance_error,clearance_error);
        max_legacy_error=std::max(max_legacy_error,legacy_error);
        require(clearance_error<=parity_tolerance_m&&legacy_error<=parity_tolerance_m&&
                gpu.contact==cpu.contact&&gpu.ambiguous==cpu.ambiguous,
                "CPU/Metal contact audit mismatch at environment "+std::to_string(n));
        declared_contacts+=gpu.contact;
        margin_violations+=gpu.safety_clearance_m<=0.0f;
        legacy_sphere_violations+=gpu_legacy_clearance[n]<=0.0f;
        ambiguous_results+=gpu.ambiguous;
    }

    const std::string geometry_hash=challenge_evaluation::sha256_file(
        std::string(SOURCE_DIR)+"/vehicle_geometry.hpp");
    const std::string kernel_hash=challenge_evaluation::sha256_file(
        std::string(SOURCE_DIR)+"/navigation_contact.metal");
    const std::filesystem::path destination(output_path);
    if(destination.has_parent_path())std::filesystem::create_directories(destination.parent_path());
    const std::string temporary_output=output_path+".tmp";
    std::ofstream csv(temporary_output,std::ios::binary|std::ios::trunc);
    require(bool(csv),"cannot write contact-audit CSV: "+temporary_output);
    challenge_evaluation::write_csv_row(csv,{
        "checkpoint_sha256","vehicle_geometry_sha256","contact_kernel_sha256","profile",
        "audit_scope","environment","world_family","world_seed","saved_steps","elapsed_s",
        "position_x_m","position_y_m","position_z_m","orientation_w","orientation_x",
        "orientation_y","orientation_z","declared_mechanical_contact","declared_model_clearance_m",
        "declared_model_safety_margin_m","declared_model_safety_clearance_m","declared_safety_margin_violation",
        "geometry_ambiguous","historical_018m_sphere_safety_clearance_m"});
    for(size_t n=0;n<count;n++) {
        const RLPhysicsState& state=states[n];const VGResult& result=gpu_results[n];
        challenge_evaluation::write_csv_row(csv,{
            checkpoint_hash,geometry_hash,kernel_hash,"l2f_declared_7_piece_v1","checkpoint_saved_state",
            challenge_evaluation::uint_cell(n),challenge_evaluation::uint_cell(worlds[n].family),
            challenge_evaluation::uint_cell(worlds[n].seed),challenge_evaluation::uint_cell(runs[n].steps),
            challenge_evaluation::number_cell(runs[n].elapsed),
            challenge_evaluation::number_cell(state.position[0]),challenge_evaluation::number_cell(state.position[1]),
            challenge_evaluation::number_cell(state.position[2]),challenge_evaluation::number_cell(state.orientation_wxyz[0]),
            challenge_evaluation::number_cell(state.orientation_wxyz[1]),challenge_evaluation::number_cell(state.orientation_wxyz[2]),
            challenge_evaluation::number_cell(state.orientation_wxyz[3]),challenge_evaluation::uint_cell(result.contact),
            challenge_evaluation::number_cell(result.physical_clearance_m),
            challenge_evaluation::number_cell(safety_margin_m),challenge_evaluation::number_cell(result.safety_clearance_m),
            challenge_evaluation::uint_cell(result.safety_clearance_m<=0.0f),
            challenge_evaluation::uint_cell(result.ambiguous),
            challenge_evaluation::number_cell(gpu_legacy_clearance[n])});
    }
    csv.flush();require(bool(csv),"contact-audit CSV write failed");csv.close();
    std::error_code rename_error;std::filesystem::rename(temporary_output,destination,rename_error);
    require(!rename_error,"cannot publish contact-audit CSV: "+rename_error.message());
    std::cout<<"contact_audit scope=checkpoint_saved_state environments="<<count
             <<" declared_contacts="<<declared_contacts<<" margin_violations="<<margin_violations
             <<" historical_sphere_violations="<<legacy_sphere_violations
             <<" ambiguous="<<ambiguous_results<<" max_cpu_metal_clearance_error="<<max_clearance_error
             <<" max_legacy_sphere_error="<<max_legacy_error<<" checkpoint_sha256="<<checkpoint_hash
             <<" output="<<output_path<<"\n";
}
