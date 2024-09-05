#!/usr/bin/env python3

import sys

# insert the timer before this line
TIMER_LINE = 'xsmm.brgemm.dispatch'

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

TPP_RUNNER_WRAPPER = """
  "func.func"() <{function_type = () -> (), sym_name = "entry"}> ({
    %35 = "arith.constant"() <{value = 0 : index}> : () -> index
    %36 = "arith.constant"() <{value = 1.000000e+00 : bf16}> : () -> bf16
    %37 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<1024x1024xbf16>
    "linalg.fill"(%36, %37) <{operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%arg19: bf16, %arg20: bf16):
      "linalg.yield"(%arg19) : (bf16) -> ()
    }) : (bf16, memref<1024x1024xbf16>) -> ()
    %38 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<1024x1024xbf16>
    "linalg.fill"(%36, %38) <{operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%arg17: bf16, %arg18: bf16):
      "linalg.yield"(%arg17) : (bf16) -> ()
    }) : (bf16, memref<1024x1024xbf16>) -> ()
    %39 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<1024x1024xbf16>
    %40 = "arith.constant"() <{value = 0.000000e+00 : bf16}> : () -> bf16
    "linalg.fill"(%40, %39) <{operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%arg15: bf16, %arg16: bf16):
      "linalg.yield"(%arg15) : (bf16) -> ()
    }) : (bf16, memref<1024x1024xbf16>) -> ()
    "func.call"(%37, %38, %39) <{callee = @tpp_entrypoint_name}> : (memref<1024x1024xbf16>, memref<1024x1024xbf16>, memref<1024x1024xbf16>) -> ()
    "func.call"(%37, %38, %39) <{callee = @tpp_entrypoint_name}> : (memref<1024x1024xbf16>, memref<1024x1024xbf16>, memref<1024x1024xbf16>) -> ()
    "func.call"(%37, %38, %39) <{callee = @tpp_entrypoint_name}> : (memref<1024x1024xbf16>, memref<1024x1024xbf16>, memref<1024x1024xbf16>) -> ()
    "func.call"(%37, %38, %39) <{callee = @tpp_entrypoint_name}> : (memref<1024x1024xbf16>, memref<1024x1024xbf16>, memref<1024x1024xbf16>) -> ()
    "func.call"(%37, %38, %39) <{callee = @tpp_entrypoint_name}> : (memref<1024x1024xbf16>, memref<1024x1024xbf16>, memref<1024x1024xbf16>) -> ()
    "func.call"(%37, %38, %39) <{callee = @tpp_entrypoint_name}> : (memref<1024x1024xbf16>, memref<1024x1024xbf16>, memref<1024x1024xbf16>) -> ()
    "func.call"(%37, %38, %39) <{callee = @tpp_entrypoint_name}> : (memref<1024x1024xbf16>, memref<1024x1024xbf16>, memref<1024x1024xbf16>) -> ()
    "func.call"(%37, %38, %39) <{callee = @tpp_entrypoint_name}> : (memref<1024x1024xbf16>, memref<1024x1024xbf16>, memref<1024x1024xbf16>) -> ()
    "func.call"(%37, %38, %39) <{callee = @tpp_entrypoint_name}> : (memref<1024x1024xbf16>, memref<1024x1024xbf16>, memref<1024x1024xbf16>) -> ()
    "func.call"(%37, %38, %39) <{callee = @tpp_entrypoint_name}> : (memref<1024x1024xbf16>, memref<1024x1024xbf16>, memref<1024x1024xbf16>) -> ()
    "func.call"(%37, %38, %39) <{callee = @tpp_entrypoint_name}> : (memref<1024x1024xbf16>, memref<1024x1024xbf16>, memref<1024x1024xbf16>) -> ()
    "func.call"(%37, %38, %39) <{callee = @tpp_entrypoint_name}> : (memref<1024x1024xbf16>, memref<1024x1024xbf16>, memref<1024x1024xbf16>) -> ()
    "func.call"(%37, %38, %39) <{callee = @tpp_entrypoint_name}> : (memref<1024x1024xbf16>, memref<1024x1024xbf16>, memref<1024x1024xbf16>) -> ()
    "func.call"(%37, %38, %39) <{callee = @tpp_entrypoint_name}> : (memref<1024x1024xbf16>, memref<1024x1024xbf16>, memref<1024x1024xbf16>) -> ()
    "func.call"(%37, %38, %39) <{callee = @tpp_entrypoint_name}> : (memref<1024x1024xbf16>, memref<1024x1024xbf16>, memref<1024x1024xbf16>) -> ()
    "func.call"(%37, %38, %39) <{callee = @tpp_entrypoint_name}> : (memref<1024x1024xbf16>, memref<1024x1024xbf16>, memref<1024x1024xbf16>) -> ()
    "func.call"(%37, %38, %39) <{callee = @tpp_entrypoint_name}> : (memref<1024x1024xbf16>, memref<1024x1024xbf16>, memref<1024x1024xbf16>) -> ()
    "func.call"(%37, %38, %39) <{callee = @tpp_entrypoint_name}> : (memref<1024x1024xbf16>, memref<1024x1024xbf16>, memref<1024x1024xbf16>) -> ()
    "func.call"(%37, %38, %39) <{callee = @tpp_entrypoint_name}> : (memref<1024x1024xbf16>, memref<1024x1024xbf16>, memref<1024x1024xbf16>) -> ()
    "func.call"(%37, %38, %39) <{callee = @tpp_entrypoint_name}> : (memref<1024x1024xbf16>, memref<1024x1024xbf16>, memref<1024x1024xbf16>) -> ()
    "func.call"(%37, %38, %39) <{callee = @tpp_entrypoint_name}> : (memref<1024x1024xbf16>, memref<1024x1024xbf16>, memref<1024x1024xbf16>) -> ()
    "func.call"(%37, %38, %39) <{callee = @tpp_entrypoint_name}> : (memref<1024x1024xbf16>, memref<1024x1024xbf16>, memref<1024x1024xbf16>) -> ()
    "func.call"(%37, %38, %39) <{callee = @tpp_entrypoint_name}> : (memref<1024x1024xbf16>, memref<1024x1024xbf16>, memref<1024x1024xbf16>) -> ()
    "func.call"(%37, %38, %39) <{callee = @tpp_entrypoint_name}> : (memref<1024x1024xbf16>, memref<1024x1024xbf16>, memref<1024x1024xbf16>) -> ()
    "func.call"(%37, %38, %39) <{callee = @tpp_entrypoint_name}> : (memref<1024x1024xbf16>, memref<1024x1024xbf16>, memref<1024x1024xbf16>) -> ()
    "func.call"(%37, %38, %39) <{callee = @tpp_entrypoint_name}> : (memref<1024x1024xbf16>, memref<1024x1024xbf16>, memref<1024x1024xbf16>) -> ()
    "func.call"(%37, %38, %39) <{callee = @tpp_entrypoint_name}> : (memref<1024x1024xbf16>, memref<1024x1024xbf16>, memref<1024x1024xbf16>) -> ()
    "func.call"(%37, %38, %39) <{callee = @tpp_entrypoint_name}> : (memref<1024x1024xbf16>, memref<1024x1024xbf16>, memref<1024x1024xbf16>) -> ()
    "func.call"(%37, %38, %39) <{callee = @tpp_entrypoint_name}> : (memref<1024x1024xbf16>, memref<1024x1024xbf16>, memref<1024x1024xbf16>) -> ()
    "func.call"(%37, %38, %39) <{callee = @tpp_entrypoint_name}> : (memref<1024x1024xbf16>, memref<1024x1024xbf16>, memref<1024x1024xbf16>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> i64, sym_name = "perf_start_timer", sym_visibility = "private"}> ({
  }) : () -> ()
  "func.func"() <{function_type = (i64) -> f64, sym_name = "perf_stop_timer", sym_visibility = "private"}> ({
  }) : () -> ()
"""

def process_new():
    """
    insert timer calls for tpp-runner into the IR
    """
    timer_ended = False
    for l in sys.stdin:
        if "builtin.module" in l:
            print(l, end="")
            print(TPP_RUNNER_WRAPPER)
        elif TIMER_LINE in l:
            print(l, end="")
            print('    %timer_start = "func.call"() <{callee = @perf_start_timer}> : () -> (i64)')
        elif 'memref.dealloc' in l and not timer_ended:
            timer_ended = True
            print('    %ttl_time = "func.call"(%timer_start) <{callee = @perf_stop_timer}> : (i64) -> f64')
            print('    "vector.print"(%ttl_time) : (f64) -> ()')
        # eat all deallocations because tpp-run doesn't like them
        elif 'memref.dealloc' in l:
            pass
        else:
            print(l, end="")

if __name__ == '__main__':
    if '--tpp-runner' in sys.argv:
        process_new()
    else:
        process_old()
