void time_kernel_start();

void dummy_kernel(void* a, void* b, void* c) {
    int z = 0;
    for (int i = 0; i< 100000000; i++) {
        z += 4;
    }

    time_kernel_start();

    for (int i = 0; i< 100000000; i++) {
        z += 4;
    }

    return;
}