#pragma once
// Optional Metal per-encoder timestamp profiler. Include after <Metal/Metal.h>.
// It uses stage-boundary samples because this M3 does not support dispatch-boundary sampling.
#include <algorithm>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <string>
#include <vector>

struct M3TimestampProfile {
    static constexpr NSUInteger capacity = 2048; // 16 KiB for 8-byte timestamp samples.
    id<MTLDevice> device = nil;
    id<MTLCounterSampleBuffer> samples = nil;
    bool available = false;
    bool armed = false;
    NSUInteger next = 0;
    size_t dropped = 0;
    struct Event { std::string name; NSUInteger first, last; };
    std::vector<Event> events;

    void initialize(id<MTLDevice> d) {
        device = d;
        const bool stage = [device supportsCounterSampling:MTLCounterSamplingPointAtStageBoundary];
        const bool dispatch = [device supportsCounterSampling:MTLCounterSamplingPointAtDispatchBoundary];
        id<MTLCounterSet> timestampSet = nil;
        for (id<MTLCounterSet> set in device.counterSets) {
            std::cout << "M3_PROFILE counter_set=" << set.name.UTF8String << " counters=";
            for (id<MTLCounter> counter in set.counters) std::cout << counter.name.UTF8String << ",";
            std::cout << "\n";
            if ([set.name isEqualToString:MTLCommonCounterSetTimestamp]) timestampSet = set;
        }
        std::cout << "M3_PROFILE device=" << device.name.UTF8String
                  << " stage_boundary=" << stage << " dispatch_boundary=" << dispatch << "\n";
        if (!stage || !timestampSet) return;
        MTLCounterSampleBufferDescriptor* desc = [MTLCounterSampleBufferDescriptor new];
        desc.counterSet = timestampSet;
        desc.sampleCount = capacity;
        desc.storageMode = MTLStorageModeShared;
        desc.label = @"m3-navigation-kernel-times";
        NSError* error = nil;
        samples = [device newCounterSampleBufferWithDescriptor:desc error:&error];
        if (!samples) {
            std::cout << "M3_PROFILE unavailable reason=" << (error ? error.localizedDescription.UTF8String : "no timestamp buffer") << "\n";
            return;
        }
        available = true;
    }

    void arm() {
        armed = available;
        next = 0;
        dropped = 0;
        events.clear();events.reserve(capacity/2);
    }

    id<MTLComputeCommandEncoder> encoder(id<MTLCommandBuffer> cb, const char* kernelName) {
        if (!armed || !available || next + 2 > capacity) {
            if (armed && available) ++dropped;
            return [cb computeCommandEncoder];
        }
        MTLComputePassDescriptor* pass = [MTLComputePassDescriptor computePassDescriptor];
        MTLComputePassSampleBufferAttachmentDescriptor* attachment = pass.sampleBufferAttachments[0];
        attachment.sampleBuffer = samples;
        const NSUInteger first = next++;
        const NSUInteger last = next++;
        attachment.startOfEncoderSampleIndex = first;
        attachment.endOfEncoderSampleIndex = last;
        id<MTLComputeCommandEncoder> result = [cb computeCommandEncoderWithDescriptor:pass];
        if (!result) {
            next -= 2;
            return [cb computeCommandEncoder];
        }
        const char* label = kernelName ? kernelName : "unnamed-kernel";
        result.label = [NSString stringWithUTF8String:label];
        events.push_back({label, first, last});
        return result;
    }

    void report(id<MTLCommandBuffer> cb, MTLTimestamp cpu0, MTLTimestamp gpu0) {
        if (!armed) return;
        armed = false;
        const NSUInteger used = next;
        if (!used) { std::cout << "M3_PROFILE no_samples\n"; return; }
        NSData* data = [samples resolveCounterRange:NSMakeRange(0, used)];
        if (!data || data.length < used * sizeof(MTLCounterResultTimestamp)) {
            std::cout << "M3_PROFILE resolve_failed samples=" << used << "\n";
            return;
        }
        MTLTimestamp cpu1=0, gpu1=0;
        [device sampleTimestamps:&cpu1 gpuTimestamp:&gpu1];
        double gpuToCpuScale = (gpu1 > gpu0 && cpu1 > cpu0)
            ? double(cpu1-cpu0) / double(gpu1-gpu0) : 1.0;
        const auto* timestamps = static_cast<const MTLCounterResultTimestamp*>(data.bytes);
        std::map<std::string, std::vector<double>> times;
        size_t invalid = 0;
        for (const Event& event : events) {
            const uint64_t a = timestamps[event.first].timestamp;
            const uint64_t b = timestamps[event.last].timestamp;
            if (a == MTLCounterErrorValue || b == MTLCounterErrorValue || b < a) { ++invalid; continue; }
            times[event.name].push_back(double(b-a) * gpuToCpuScale / 1000.0); // microseconds
        }
        const double commandGpuMs = cb.GPUEndTime > cb.GPUStartTime ? (cb.GPUEndTime-cb.GPUStartTime)*1000.0 : 0.0;
        std::cout << std::fixed << std::setprecision(4)
                  << "M3_PROFILE gpu_command_ms=" << commandGpuMs
                  << " timestamp_scale=" << gpuToCpuScale
                  << " sampled_dispatches=" << events.size()
                  << " invalid=" << invalid << " capacity_skipped=" << dropped << "\n";
        std::vector<std::pair<std::string,std::vector<double>>> ordered(times.begin(),times.end());
        std::sort(ordered.begin(),ordered.end(),[](const auto& a,const auto& b){
            const double sa=std::accumulate(a.second.begin(),a.second.end(),0.0);
            const double sb=std::accumulate(b.second.begin(),b.second.end(),0.0);
            return sa>sb;
        });
        std::cout << "M3_PROFILE_KERNEL name calls total_ms mean_us p50_us p95_us max_us\n";
        for (auto& item : ordered) {
            auto& v=item.second; std::sort(v.begin(),v.end());
            const double total=std::accumulate(v.begin(),v.end(),0.0);
            const double p50=v[(v.size()-1)*50/100], p95=v[(v.size()-1)*95/100];
            std::cout << "M3_PROFILE_KERNEL " << item.first << " " << v.size() << " " << total/1000.0 << " "
                      << total/v.size() << " " << p50 << " " << p95 << " " << v.back() << "\n";
        }
    }
};
