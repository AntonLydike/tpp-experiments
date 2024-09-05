#!/usr/bin/env python

import sys

# insert the timer before this line
TIMER_LINE = '%10 = "xsmm.IntelAMXtileConfig.dispatch"() <{data_type = 2 : i64, flags = [4096, 64], inputs = array<i64: 32, 32, 32, 32, 32, 1024, 1024, 1024>}> : () -> i64'

def process_old():
    """
    insert timer calls for runner.cpp into the IR
    """
    for l in sys.stdin:
        if '"builtin.module"() ({' in l:
            print(l, end="")
            print ("  func.func private @time_kernel_start()")
        elif TIMER_LINE in l:
            print("    func.call @time_kernel_start() : () -> ()")
            print(l, end="")
        else:
            print(l, end="")

def process_new():
    """
    insert timer calls for tpp-runner into the IR
    """
    timer_ended = False
    for l in sys.stdin:
        if TIMER_LINE in l:
            print("    %timer_start = func.call @perf_start_timer() : () -> i64")
        elif 'memref.dealloc' in l and not timer_ended:
            timer_ended = True
            print("    %ttl_time = func.call @perf_stop_timer(%timer_start) : (i64) -> f64")
            print("    vector.print %ttl_time : f64")
        print(l)

if __name__ == '__main__':
    if '--tpp-runner' in sys.argv:
        process_new()
    else:
        process_old()
