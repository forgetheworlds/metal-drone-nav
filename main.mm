#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <string>

int main(int argc, char** argv) {
    @autoreleasepool {
        try {
            id<MTLDevice> device = MTLCreateSystemDefaultDevice();
            if (!device) throw std::runtime_error("No Metal device");
            MTLCompileOptions* options = [MTLCompileOptions new];
            options.mathMode = MTLMathModeSafe;
            options.mathFloatingPointFunctions = MTLMathFloatingPointFunctionsPrecise;
            NSError* error = nil;
            NSString* source = @"#include <metal_stdlib>\nusing namespace metal;\nkernel void probe(device float* x [[buffer(0)]], uint i [[thread_position_in_grid]]) { x[i] = float(i)*2.0f+1.0f; }";
            id<MTLLibrary> library = [device newLibraryWithSource:source options:options error:&error];
            if (!library) throw std::runtime_error(error.localizedDescription.UTF8String);
            id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"probe"] error:&error];
            if (!pipeline) throw std::runtime_error(error.localizedDescription.UTF8String);
            id<MTLCommandQueue> queue = [device newCommandQueue];
            constexpr size_t n=4096;
            id<MTLBuffer> buffer = [device newBufferWithLength:n*sizeof(float) options:MTLResourceStorageModeShared];
            id<MTLCommandBuffer> cmd = [queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:pipeline];
            [enc setBuffer:buffer offset:0 atIndex:0];
            [enc dispatchThreads:MTLSizeMake(n,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
            [enc endEncoding]; [cmd commit]; [cmd waitUntilCompleted];
            if(cmd.status==MTLCommandBufferStatusError) throw std::runtime_error(cmd.error.localizedDescription.UTF8String);
            const float* values = static_cast<const float*>(buffer.contents);
            for(size_t i=0;i<n;i++) if(values[i]!=float(i)*2+1) throw std::runtime_error("probe mismatch");
            std::cout << "Metal runtime compiler PASS; device=" << device.name.UTF8String
                      << "; probe values=" << n << "; GPU seconds=" << cmd.GPUEndTime-cmd.GPUStartTime << "\n";
            (void)argc; (void)argv;
            return 0;
        } catch(const std::exception& e) { std::cerr << "ERROR: " << e.what() << "\n"; return 1; }
    }
}
