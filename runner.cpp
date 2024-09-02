/**
 * Test runner for our experiemtns. The runner will initialize the data and invoke the benchmark exactly once.
 * 
 * The runner is compiled to launch one kernel, defined throught the KERNEL_NAME macro.
 * 
 * The kernel must be compiled to accept memrefs as bare pointers.
 * 
 * See the makefile for further instructions.
 */

#include <cstdint>
#include <chrono>
#include <stdlib.h>
#include <iostream>
#include <stdfloat>

typedef std::chrono::steady_clock::time_point timestamp;

typedef struct measurement {
    timestamp start;
    timestamp kernel_start;
    timestamp end;
} Measurement;

Measurement current_measurement;


// kernels we will be running:
// void tpp_baseline(void* a, void* b, void* out);
// void tpp_deduplicated(void* a, void* b, void* out);
// void accfg_deduplicated(void* a, void* b, void* out);
void KERNEL_NAME(void* a, void* b, void* out);


inline void print_time_reasonable(timestamp start, timestamp end) {
    if (std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() > 1000 * 100) {
        std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms";
    } else {
        std::cout << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << "µs";
    }
}

void time_kernel_start() {
    current_measurement.kernel_start = std::chrono::steady_clock::now();
}

inline void bench_preamble() {
    current_measurement.start = std::chrono::steady_clock::now();
}

inline void bench_epilogue() {   
    current_measurement.end = std::chrono::steady_clock::now();

    std::cout << "total duration: ";
    print_time_reasonable(current_measurement.start, current_measurement.end);
    
    std::cout << std::endl << "data shuffling: ";
    print_time_reasonable(current_measurement.start, current_measurement.kernel_start);

    std::cout << std::endl << "kernel with conf: ";
    print_time_reasonable(current_measurement.kernel_start, current_measurement.end);
    std::cout << std::endl;
}

int main() {
    std::cout << " --=== Initialization ===--- " << std::endl;
    bench_preamble();

    // allocate room for 1024*1024 16 bit floats
    void* A = malloc(2 * 1024 * 1024);
    void* B = malloc(2 * 1024 * 1024);
    void* C = malloc(2 * 1024 * 1024);

    time_kernel_start();

    // fill with bfloat16 encoded 1s
    for (int i = 0; i < 1024*1024 / 2; i++) {
        // fill two "1"s encoded as bf16
        ((int*)(A))[i] = 0x3f803f80;
        ((int*)(B))[i] = 0x3f803f80;
        ((int*)(C))[i] = 0x3f803f80;
    }

    bench_epilogue();


    std::cout << " --=== Kernel ===--- " << std::endl;

    bench_preamble();

    KERNEL_NAME(A, B, C);

    bench_epilogue();
    
    return 0;
}
