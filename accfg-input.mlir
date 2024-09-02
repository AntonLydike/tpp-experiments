#map = affine_map<(d0) -> (d0 * 32)>
"builtin.module"() ({
  func.func private @time_kernel_start()
  "func.func"() <{function_type = (memref<1024x1024xbf16>, memref<1024x1024xbf16>, memref<1024x1024xbf16>) -> (), sym_name = "tpp_entrypoint_name"}> ({
  ^bb0(%arg0: memref<1024x1024xbf16>, %arg1: memref<1024x1024xbf16>, %arg2: memref<1024x1024xbf16>):
    %0 = "arith.constant"() <{value = 8 : index}> : () -> index
    %1 = "arith.constant"() <{value = 4 : index}> : () -> index
    %2 = "arith.constant"() <{value = 1 : index}> : () -> index
    %3 = "arith.constant"() <{value = 32 : index}> : () -> index
    %4 = "arith.constant"() <{value = 0 : index}> : () -> index
    %5 = "arith.constant"() <{value = 32 : i64}> : () -> i64
    %6 = "memref.alloc"() <{alignment = 64 : i64, operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<32x32x32x32xbf16>
    %7 = "xsmm.unary.dispatch"() <{data_type = 2 : i64, flags = [0], inputs = array<i64: 32, 32, 1024, 32>, kind = 1 : i64}> : () -> i64
    "scf.parallel"(%4, %4, %3, %3, %1, %0) <{operandSegmentSizes = array<i32: 2, 2, 2, 0>}> ({
    ^bb0(%arg11: index, %arg12: index):
      "scf.for"(%4, %1, %2) ({
      ^bb0(%arg13: index):
        %27 = "arith.addi"(%arg13, %arg11) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
        %28 = "affine.apply"(%27) <{map = #map}> : (index) -> index
        "scf.for"(%4, %0, %2) ({
        ^bb0(%arg14: index):
          %29 = "arith.addi"(%arg14, %arg12) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
          %30 = "affine.apply"(%29) <{map = #map}> : (index) -> index
          %31 = "memref.subview"(%arg0, %28, %30) <{operandSegmentSizes = array<i32: 1, 2, 0, 0>, static_offsets = array<i64: -9223372036854775808, -9223372036854775808>, static_sizes = array<i64: 32, 32>, static_strides = array<i64: 1, 1>}> : (memref<1024x1024xbf16>, index, index) -> memref<32x32xbf16, strided<[1024, 1], offset: ?>>
          %32 = "memref.subview"(%6, %27, %29) <{operandSegmentSizes = array<i32: 1, 2, 0, 0>, static_offsets = array<i64: -9223372036854775808, -9223372036854775808, 0, 0>, static_sizes = array<i64: 1, 1, 32, 32>, static_strides = array<i64: 1, 1, 1, 1>}> : (memref<32x32x32x32xbf16>, index, index) -> memref<32x32xbf16, strided<[32, 1], offset: ?>>
          "xsmm.unary"(%7, %31, %32) <{callee = 1 : i64, data_type = 2 : i64}> : (i64, memref<32x32xbf16, strided<[1024, 1], offset: ?>>, memref<32x32xbf16, strided<[32, 1], offset: ?>>) -> ()
          "scf.yield"() : () -> ()
        }) : (index, index, index) -> ()
        "scf.yield"() : () -> ()
      }) : (index, index, index) -> ()
      "scf.reduce"() : () -> ()
    }) : (index, index, index, index, index, index) -> ()
    %8 = "memref.alloc"() <{alignment = 64 : i64, operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<32x32x16x32x2xbf16>
    %9 = "xsmm.unary.dispatch"() <{data_type = 2 : i64, flags = [0], inputs = array<i64: 32, 32, 1024, 32>, kind = 28 : i64}> : () -> i64
    "scf.parallel"(%4, %4, %3, %3, %1, %0) <{operandSegmentSizes = array<i32: 2, 2, 2, 0>}> ({
    ^bb0(%arg7: index, %arg8: index):
      "scf.for"(%4, %1, %2) ({
      ^bb0(%arg9: index):
        %21 = "arith.addi"(%arg9, %arg7) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
        %22 = "affine.apply"(%21) <{map = #map}> : (index) -> index
        "scf.for"(%4, %0, %2) ({
        ^bb0(%arg10: index):
          %23 = "arith.addi"(%arg10, %arg8) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
          %24 = "affine.apply"(%23) <{map = #map}> : (index) -> index
          %25 = "memref.subview"(%arg1, %24, %22) <{operandSegmentSizes = array<i32: 1, 2, 0, 0>, static_offsets = array<i64: -9223372036854775808, -9223372036854775808>, static_sizes = array<i64: 32, 32>, static_strides = array<i64: 1, 1>}> : (memref<1024x1024xbf16>, index, index) -> memref<32x32xbf16, strided<[1024, 1], offset: ?>>
          %26 = "memref.subview"(%8, %21, %23) <{operandSegmentSizes = array<i32: 1, 2, 0, 0>, static_offsets = array<i64: -9223372036854775808, -9223372036854775808, 0, 0, 0>, static_sizes = array<i64: 1, 1, 16, 32, 2>, static_strides = array<i64: 1, 1, 1, 1, 1>}> : (memref<32x32x16x32x2xbf16>, index, index) -> memref<16x32x2xbf16, strided<[64, 2, 1], offset: ?>>
          "xsmm.unary"(%9, %25, %26) <{callee = 28 : i64, data_type = 2 : i64}> : (i64, memref<32x32xbf16, strided<[1024, 1], offset: ?>>, memref<16x32x2xbf16, strided<[64, 2, 1], offset: ?>>) -> ()
          "scf.yield"() : () -> ()
        }) : (index, index, index) -> ()
        "scf.yield"() : () -> ()
      }) : (index, index, index) -> ()
      "scf.reduce"() : () -> ()
    }) : (index, index, index, index, index, index) -> ()
    func.call @time_kernel_start() : () -> ()
    %10 = "xsmm.IntelAMXtileConfig.dispatch"() <{data_type = 2 : i64, flags = [4096, 64], inputs = array<i64: 32, 32, 32, 32, 32, 1024, 1024, 1024>}> : () -> i64
    %11 = "xsmm.IntelAMXtileConfig.dispatch"() <{data_type = 2 : i64, flags = [4096, 128], inputs = array<i64: 32, 32, 32, 32, 32, 1024, 1024, 1024>}> : () -> i64
    %12 = "xsmm.brgemm.dispatch"() <{data_type = 2 : i64, flags = [4096, 64, 128], inputs = array<i64: 32, 32, 32, 32, 32, 1024, 1024, 1024>}> : () -> i64
    "scf.parallel"(%4, %4, %3, %3, %1, %0) <{operandSegmentSizes = array<i32: 2, 2, 2, 0>}> ({
    ^bb0(%arg3: index, %arg4: index):
      "scf.for"(%4, %1, %2) ({
      ^bb0(%arg5: index):
        %13 = "arith.addi"(%arg5, %arg3) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
        %14 = "affine.apply"(%13) <{map = #map}> : (index) -> index
        %15 = "memref.subview"(%6, %13) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: -9223372036854775808, 0, 0, 0>, static_sizes = array<i64: 1, 32, 32, 32>, static_strides = array<i64: 1, 1, 1, 1>}> : (memref<32x32x32x32xbf16>, index) -> memref<32x32x32xbf16, strided<[1024, 32, 1], offset: ?>>
        "scf.for"(%4, %0, %2) ({
        ^bb0(%arg6: index):
          %16 = "arith.addi"(%arg6, %arg4) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
          %17 = "affine.apply"(%16) <{map = #map}> : (index) -> index
          %18 = "memref.subview"(%8, %16) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: -9223372036854775808, 0, 0, 0, 0>, static_sizes = array<i64: 1, 32, 16, 32, 2>, static_strides = array<i64: 1, 1, 1, 1, 1>}> : (memref<32x32x16x32x2xbf16>, index) -> memref<32x16x32x2xbf16, strided<[1024, 64, 2, 1], offset: ?>>
          %19 = "memref.subview"(%arg2, %14, %17) <{operandSegmentSizes = array<i32: 1, 2, 0, 0>, static_offsets = array<i64: -9223372036854775808, -9223372036854775808>, static_sizes = array<i64: 32, 32>, static_strides = array<i64: 1, 1>}> : (memref<1024x1024xbf16>, index, index) -> memref<32x32xbf16, strided<[1024, 1], offset: ?>>
          %20 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<64xi8>
          "xsmm.IntelAMXtileConfig"(%10, %20) : (i64, memref<64xi8>) -> ()
          "xsmm.brgemm"(%12, %15, %18, %19, %5) <{data_type = 2 : i64}> : (i64, memref<32x32x32xbf16, strided<[1024, 32, 1], offset: ?>>, memref<32x16x32x2xbf16, strided<[1024, 64, 2, 1], offset: ?>>, memref<32x32xbf16, strided<[1024, 1], offset: ?>>, i64) -> ()
          "xsmm.IntelAMXtileConfig"(%11, %20) : (i64, memref<64xi8>) -> ()
          "scf.yield"() : () -> ()
        }) : (index, index, index) -> ()
        "scf.yield"() : () -> ()
      }) : (index, index, index) -> ()
      "scf.reduce"() : () -> ()
    }) : (index, index, index, index, index, index) -> ()
    "memref.dealloc"(%6) : (memref<32x32x32x32xbf16>) -> ()
    "memref.dealloc"(%8) : (memref<32x32x16x32x2xbf16>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

