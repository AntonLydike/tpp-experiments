#!/usr/bin/env python

import sys
import re

REPLACEMENTS = {
    '"scf.reduce"() : () -> ()': '"scf.yield"() : () -> ()',
    '%11 = "xsmm.IntelAMXtileConfig.dispatch"() <{data_type = 2 : i64, flags = [4096, 128], inputs = array<i64: 32, 32, 32, 32, 32, 1024, 1024, 1024>}> : () -> i64': '',
    '%20 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<64xi8>': "",
    '"xsmm.IntelAMXtileConfig"(%10, %20) : (i64, memref<64xi8>) -> ()': 
    '          %state = "accfg.setup"(%10) <{param_names=["conf"], accelerator="amx", operandSegmentSizes = array<i32: 1, 0>}> : (i64) -> !accfg.state<"amx">\n',
    '"xsmm.brgemm"(%12, %15, %18, %19, %5) <{data_type = 2 : i64}> : (i64, memref<32x32x32xbf16, strided<[1024, 32, 1], offset: ?>>, memref<32x16x32x2xbf16, strided<[1024, 64, 2, 1], offset: ?>>, memref<32x32xbf16, strided<[1024, 1], offset: ?>>, i64) -> ()': 
    '          %t = "accfg.launch"(%12, %15, %18, %19, %5, %state) <{"param_names" = ["gemm", "a", "b", "out", "size"], "accelerator" = "amx"}> : (i64, memref<32x32x32xbf16, strided<[1024, 32, 1], offset: ?>>, memref<32x16x32x2xbf16, strided<[1024, 64, 2, 1], offset: ?>>, memref<32x32xbf16, strided<[1024, 1], offset: ?>>, i64, !accfg.state<"amx">) -> (!accfg.token<"amx">)\n',
    '"xsmm.IntelAMXtileConfig"(%11, %20) : (i64, memref<64xi8>) -> ()': 
    '          "accfg.await"(%t) : (!accfg.token<"amx">) -> ()\n',
}
def process(prog: str):
    with open(prog, "r") as f:
        for l in f:
            if '<{overflowFlags = #arith.overflow<none>}>' in l:
                l = l.replace('<{overflowFlags = #arith.overflow<none>}>', '')
            print(REPLACEMENTS.get(l.strip(), l), end="")



REVERSE_REPLACEMENTS = {
    '%24 = "accfg.setup"(%22) <{"param_names" = ["conf"], "accelerator" = "amx", "operandSegmentSizes" = array<i32: 1, 0>}> : (i64) -> !accfg.state<"amx">':
    '      %alloca = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<64xi8>\n      "xsmm.IntelAMXtileConfig"(%22, %alloca) : (i64, memref<64xi8>) -> ()\n',
    '%t = "accfg.launch"(%23, %29, %33, %34, %5, %state) <{"param_names" = ["gemm", "a", "b", "out", "size"], "accelerator" = "amx"}> : (i64, memref<32x32x32xbf16, strided<[1024, 32, 1], offset: ?>>, memref<32x16x32x2xbf16, strided<[1024, 64, 2, 1], offset: ?>>, memref<32x32xbf16, strided<[1024, 1], offset: ?>>, i64, !accfg.state<"amx">) -> !accfg.token<"amx">':
    '          "xsmm.brgemm"(%23, %29, %33, %34, %5) <{data_type = 2 : i64}> : (i64, memref<32x32x32xbf16, strided<[1024, 32, 1], offset: ?>>, memref<32x16x32x2xbf16, strided<[1024, 64, 2, 1], offset: ?>>, memref<32x32xbf16, strided<[1024, 1], offset: ?>>, i64) -> ()\n',
    '"accfg.await"(%t) : (!accfg.token<"amx">) -> ()': '',
    '"accfg.reset"(%25) : (!accfg.state<"amx">) -> ()':
    '      %resetcfg = "xsmm.IntelAMXtileConfig.dispatch"() <{data_type = 2 : i64, flags = [4096, 128], inputs = array<i64: 32, 32, 32, 32, 32, 1024, 1024, 1024>}> : () -> i64\n'
    '      "xsmm.IntelAMXtileConfig"(%resetcfg, %alloca) : (i64, memref<64xi8>) -> ()\n',
    '%25 = "scf.for"(%4, %1, %2, %24) ({': '      "scf.for"(%4, %1, %2) ({',
    '^8(%arg5 : index, %26 : !accfg.state<"amx">):': '      ^8(%arg5 : index):',
    '%30 = "scf.for"(%4, %0, %2, %26) ({': '        "scf.for"(%4, %0, %2) ({',
    '^9(%arg6 : index, %state : !accfg.state<"amx">):': '        ^9(%arg6 : index):',
    '"scf.yield"(%state) : (!accfg.state<"amx">) -> ()': '          "scf.yield"() : () -> ()',
    '}) : (index, index, index, !accfg.state<"amx">) -> !accfg.state<"amx">': '}) : (index, index, index) -> ()',
    '"scf.yield"(%30) : (!accfg.state<"amx">) -> ()': '        "scf.yield"() : () -> ()'
}

def reverse(prog: str):
    outstr = ""
    if prog == '-':
        for l in sys.stdin:
            outstr += REVERSE_REPLACEMENTS.get(l.strip(), l)
    else:
        with open(prog, "r") as f:
            for l in f:
                outstr += REVERSE_REPLACEMENTS.get(l.strip(), l)
    
    outstr = re.sub(r'arith.addi"(\(.*\)) :', r'arith.addi"\1 <{overflowFlags = #arith.overflow<none>}> :', outstr)

    outstr = outstr.replace('      "scf.yield"() : () -> ()\n    }) : (index, index, index, index, index, index) -> ()', '      "scf.reduce"() : () -> ()\n    }) : (index, index, index, index, index, index) -> ()')
    print(outstr)



if __name__ == '__main__':
    if "--reverse" in sys.argv:
        reverse(sys.argv[-1])
    else:
        process(sys.argv[-1])
