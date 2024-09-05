func.func @tpp_entrypoint_name(%arg0: tensor<1024x1024xbf16>,
                 %arg1: tensor<1024x1024xbf16>,
                 %arg2: tensor<1024x1024xbf16>) -> tensor<1024x1024xbf16> {
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<1024x1024xbf16>, tensor<1024x1024xbf16>)
                     outs(%arg2 : tensor<1024x1024xbf16>) -> tensor<1024x1024xbf16>
  return %0 : tensor<1024x1024xbf16>
}


func.func @entry() {
  %c0 = arith.constant 0 : index
  %f0 = arith.constant 1.0 : bf16
  %da = memref.alloc() :memref<1024x1024xbf16>
  linalg.fill ins(%f0 : bf16) outs(%da : memref<1024x1024xbf16>)
  // Call kernel.
  %0 = memref.alloc() : memref<1024x1024xbf16>
  linalg.fill ins(%f0:bf16) outs (%0: memref<1024x1024xbf16>)
  %D = memref.alloc() : memref<1024x1024xbf16>
  %zero = arith.constant 0.0 : bf16
  linalg.fill ins(%zero : bf16) outs(%D:memref<1024x1024xbf16>)
  call @matmultpp(%da, %0, %D)
       : (memref<1024x1024xbf16>, memref<1024x1024xbf16>, memref<1024x1024xbf16>)->()

  // TODO: check output for correctness?

  return
}

  func.func private @perf_start_timer() -> i64
  func.func private @time_kernel_start(i64) -> f64
