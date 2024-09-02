#map = affine_map<(d0) -> (d0 * 32)>
module {
  func.func private @time_kernel_start()
  func.func private @xsmm_intel_amx_tile_config_invoke(i64, i64, !llvm.ptr, index)
  func.func private @xsmm_brgemm_invoke(i64, i64, !llvm.ptr, index, !llvm.ptr, index, !llvm.ptr, index, i64)
  func.func private @xsmm_brgemm_dispatch(i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> i64
  func.func private @xsmm_intel_amx_tile_config_dispatch(i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> i64
  func.func private @xsmm_unary_invoke(i64, i64, !llvm.ptr, index, !llvm.ptr, index)
  func.func private @xsmm_unary_dispatch(i64, i64, i64, i64, i64, i64, i64) -> i64
  func.func @tpp_baseline(%arg0: memref<1024x1024xbf16>, %arg1: memref<1024x1024xbf16>, %arg2: memref<1024x1024xbf16>) {
    %c2240_i64 = arith.constant 2240 : i64
    %c2176_i64 = arith.constant 2176 : i64
    %c2112_i64 = arith.constant 2112 : i64
    %c28_i64 = arith.constant 28 : i64
    %c0_i64 = arith.constant 0 : i64
    %c1024_i64 = arith.constant 1024 : i64
    %c2_i64 = arith.constant 2 : i64
    %c1_i64 = arith.constant 1 : i64
    %c8 = arith.constant 8 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %c0 = arith.constant 0 : index
    %c32_i64 = arith.constant 32 : i64
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<32x32x32x32xbf16>
    %0 = call @xsmm_unary_dispatch(%c1_i64, %c2_i64, %c32_i64, %c32_i64, %c1024_i64, %c32_i64, %c0_i64) : (i64, i64, i64, i64, i64, i64, i64) -> i64
    scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c32, %c32) step (%c4, %c8) {
      scf.for %arg5 = %c0 to %c4 step %c1 {
        %5 = arith.addi %arg5, %arg3 : index
        %6 = affine.apply #map(%5)
        scf.for %arg6 = %c0 to %c8 step %c1 {
          %7 = arith.addi %arg6, %arg4 : index
          %8 = affine.apply #map(%7)
          %subview = memref.subview %arg0[%6, %8] [32, 32] [1, 1] : memref<1024x1024xbf16> to memref<32x32xbf16, strided<[1024, 1], offset: ?>>
          %subview_1 = memref.subview %alloc[%5, %7, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : memref<32x32x32x32xbf16> to memref<32x32xbf16, strided<[32, 1], offset: ?>>
          %base_buffer, %offset, %sizes:2, %strides:2 = memref.extract_strided_metadata %subview : memref<32x32xbf16, strided<[1024, 1], offset: ?>> -> memref<bf16>, index, index, index, index, index
          %intptr = memref.extract_aligned_pointer_as_index %subview : memref<32x32xbf16, strided<[1024, 1], offset: ?>> -> index
          %9 = arith.index_cast %intptr : index to i64
          %10 = llvm.inttoptr %9 : i64 to !llvm.ptr
          %base_buffer_2, %offset_3, %sizes_4:2, %strides_5:2 = memref.extract_strided_metadata %subview_1 : memref<32x32xbf16, strided<[32, 1], offset: ?>> -> memref<bf16>, index, index, index, index, index
          %intptr_6 = memref.extract_aligned_pointer_as_index %subview_1 : memref<32x32xbf16, strided<[32, 1], offset: ?>> -> index
          %11 = arith.index_cast %intptr_6 : index to i64
          %12 = llvm.inttoptr %11 : i64 to !llvm.ptr
          func.call @xsmm_unary_invoke(%c2_i64, %0, %10, %offset, %12, %offset_3) : (i64, i64, !llvm.ptr, index, !llvm.ptr, index) -> ()
        }
      }
      scf.reduce 
    }
    %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<32x32x16x32x2xbf16>
    %1 = call @xsmm_unary_dispatch(%c28_i64, %c2_i64, %c32_i64, %c32_i64, %c1024_i64, %c32_i64, %c0_i64) : (i64, i64, i64, i64, i64, i64, i64) -> i64
    scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c32, %c32) step (%c4, %c8) {
      scf.for %arg5 = %c0 to %c4 step %c1 {
        %5 = arith.addi %arg5, %arg3 : index
        %6 = affine.apply #map(%5)
        scf.for %arg6 = %c0 to %c8 step %c1 {
          %7 = arith.addi %arg6, %arg4 : index
          %8 = affine.apply #map(%7)
          %subview = memref.subview %arg1[%8, %6] [32, 32] [1, 1] : memref<1024x1024xbf16> to memref<32x32xbf16, strided<[1024, 1], offset: ?>>
          %subview_1 = memref.subview %alloc_0[%5, %7, 0, 0, 0] [1, 1, 16, 32, 2] [1, 1, 1, 1, 1] : memref<32x32x16x32x2xbf16> to memref<16x32x2xbf16, strided<[64, 2, 1], offset: ?>>
          %base_buffer, %offset, %sizes:2, %strides:2 = memref.extract_strided_metadata %subview : memref<32x32xbf16, strided<[1024, 1], offset: ?>> -> memref<bf16>, index, index, index, index, index
          %intptr = memref.extract_aligned_pointer_as_index %subview : memref<32x32xbf16, strided<[1024, 1], offset: ?>> -> index
          %9 = arith.index_cast %intptr : index to i64
          %10 = llvm.inttoptr %9 : i64 to !llvm.ptr
          %base_buffer_2, %offset_3, %sizes_4:3, %strides_5:3 = memref.extract_strided_metadata %subview_1 : memref<16x32x2xbf16, strided<[64, 2, 1], offset: ?>> -> memref<bf16>, index, index, index, index, index, index, index
          %intptr_6 = memref.extract_aligned_pointer_as_index %subview_1 : memref<16x32x2xbf16, strided<[64, 2, 1], offset: ?>> -> index
          %11 = arith.index_cast %intptr_6 : index to i64
          %12 = llvm.inttoptr %11 : i64 to !llvm.ptr
          func.call @xsmm_unary_invoke(%c2_i64, %1, %10, %offset, %12, %offset_3) : (i64, i64, !llvm.ptr, index, !llvm.ptr, index) -> ()
        }
      }
      scf.reduce 
    }
    call @time_kernel_start() : () -> ()
    %2 = call @xsmm_intel_amx_tile_config_dispatch(%c2_i64, %c32_i64, %c32_i64, %c32_i64, %c32_i64, %c32_i64, %c1024_i64, %c1024_i64, %c1024_i64, %c2112_i64) : (i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> i64
    %3 = call @xsmm_intel_amx_tile_config_dispatch(%c2_i64, %c32_i64, %c32_i64, %c32_i64, %c32_i64, %c32_i64, %c1024_i64, %c1024_i64, %c1024_i64, %c2176_i64) : (i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> i64
    %4 = call @xsmm_brgemm_dispatch(%c2_i64, %c32_i64, %c32_i64, %c32_i64, %c32_i64, %c32_i64, %c1024_i64, %c1024_i64, %c1024_i64, %c2240_i64) : (i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> i64
    scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c32, %c32) step (%c4, %c8) {
      scf.for %arg5 = %c0 to %c4 step %c1 {
        %5 = arith.addi %arg5, %arg3 : index
        %6 = affine.apply #map(%5)
        %subview = memref.subview %alloc[%5, 0, 0, 0] [1, 32, 32, 32] [1, 1, 1, 1] : memref<32x32x32x32xbf16> to memref<32x32x32xbf16, strided<[1024, 32, 1], offset: ?>>
        scf.for %arg6 = %c0 to %c8 step %c1 {
          %7 = arith.addi %arg6, %arg4 : index
          %8 = affine.apply #map(%7)
          %subview_1 = memref.subview %alloc_0[%7, 0, 0, 0, 0] [1, 32, 16, 32, 2] [1, 1, 1, 1, 1] : memref<32x32x16x32x2xbf16> to memref<32x16x32x2xbf16, strided<[1024, 64, 2, 1], offset: ?>>
          %subview_2 = memref.subview %arg2[%6, %8] [32, 32] [1, 1] : memref<1024x1024xbf16> to memref<32x32xbf16, strided<[1024, 1], offset: ?>>
          %alloca = memref.alloca() : memref<64xi8>
          %intptr = memref.extract_aligned_pointer_as_index %alloca : memref<64xi8> -> index
          %9 = arith.index_cast %intptr : index to i64
          %10 = llvm.inttoptr %9 : i64 to !llvm.ptr
          func.call @xsmm_intel_amx_tile_config_invoke(%c2_i64, %2, %10, %c0) : (i64, i64, !llvm.ptr, index) -> ()
          %base_buffer, %offset, %sizes:3, %strides:3 = memref.extract_strided_metadata %subview : memref<32x32x32xbf16, strided<[1024, 32, 1], offset: ?>> -> memref<bf16>, index, index, index, index, index, index, index
          %intptr_3 = memref.extract_aligned_pointer_as_index %subview : memref<32x32x32xbf16, strided<[1024, 32, 1], offset: ?>> -> index
          %11 = arith.index_cast %intptr_3 : index to i64
          %12 = llvm.inttoptr %11 : i64 to !llvm.ptr
          %base_buffer_4, %offset_5, %sizes_6:4, %strides_7:4 = memref.extract_strided_metadata %subview_1 : memref<32x16x32x2xbf16, strided<[1024, 64, 2, 1], offset: ?>> -> memref<bf16>, index, index, index, index, index, index, index, index, index
          %intptr_8 = memref.extract_aligned_pointer_as_index %subview_1 : memref<32x16x32x2xbf16, strided<[1024, 64, 2, 1], offset: ?>> -> index
          %13 = arith.index_cast %intptr_8 : index to i64
          %14 = llvm.inttoptr %13 : i64 to !llvm.ptr
          %base_buffer_9, %offset_10, %sizes_11:2, %strides_12:2 = memref.extract_strided_metadata %subview_2 : memref<32x32xbf16, strided<[1024, 1], offset: ?>> -> memref<bf16>, index, index, index, index, index
          %intptr_13 = memref.extract_aligned_pointer_as_index %subview_2 : memref<32x32xbf16, strided<[1024, 1], offset: ?>> -> index
          %15 = arith.index_cast %intptr_13 : index to i64
          %16 = llvm.inttoptr %15 : i64 to !llvm.ptr
          func.call @xsmm_brgemm_invoke(%c2_i64, %4, %12, %offset, %14, %offset_5, %16, %offset_10, %c32_i64) : (i64, i64, !llvm.ptr, index, !llvm.ptr, index, !llvm.ptr, index, i64) -> ()
          func.call @xsmm_intel_amx_tile_config_invoke(%c2_i64, %3, %10, %c0) : (i64, i64, !llvm.ptr, index) -> ()
        }
      }
      scf.reduce 
    }
    memref.dealloc %alloc : memref<32x32x32x32xbf16>
    memref.dealloc %alloc_0 : memref<32x32x16x32x2xbf16>
    return
  }
}

