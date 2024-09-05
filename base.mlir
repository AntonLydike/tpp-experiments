func.func @tpp_entrypoint_name(%arg0: tensor<2048x2048xbf16>,
                 %arg1: tensor<2048x2048xbf16>,
                 %arg2: tensor<2048x2048xbf16>) -> tensor<2048x2048xbf16> {
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<2048x2048xbf16>, tensor<2048x2048xbf16>)
                     outs(%arg2 : tensor<2048x2048xbf16>) -> tensor<2048x2048xbf16>
  return %0 : tensor<2048x2048xbf16>
}
