#!/usr/bin/env python

import sys

# insert the timer before this line
TIMER_LINE = '%10 = "xsmm.IntelAMXtileConfig.dispatch"() <{data_type = 2 : i64, flags = [4096, 64], inputs = array<i64: 32, 32, 32, 32, 32, 1024, 1024, 1024>}> : () -> i64'

def process():
    for l in sys.stdin:
        if '"builtin.module"() ({' in l:
            print(l, end="")
            print ("  func.func private @time_kernel_start()")
        elif TIMER_LINE in l:
            print("    func.call @time_kernel_start() : () -> ()")
            print(l, end="")
        else:
            print(l, end="")

if __name__ == '__main__':
    process()
