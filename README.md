# TPP-MLIR experiments

## Running the benchmarks:

I did not manage to compile the output of `tpp-opt` to object files, I'd be very glad if someone could take care of that.

The Makefile is set up to produce three actual and one test benchmark

- `tpp_baseline.mlir` is a version where no setup optimisation has happened. A good baseline.
- `tpp_deduplicated.mlir` is compiled with the normal tpp-mlir flow (running individual passes, not the all-encompassing pass)
- `accfg_deduplicated.mlir` is deduplicated using `accfg`. This is our work.

Since I couldn't get the MLIR files to compile, the makefile lacks these steps. Once the MLIR has been compiled to object files, the Makefile expects them to be located at `<name>.o`, so for example `tpp_baseline.o`. With these in place, `make tpp_baseline.out` will produce the final benchmark executable. These are instrumented to record execution time. If you have access to better counters, please let us know. Just execution time is a good enought proxy though. (we measure both total time (includung shuffling) and kernel time only, without data shuffles.).

We have committed both the intermediate steps, as well as the makefile to produce these, to make it as easy as possible.

Steps to run benchmarks:

- compile the three MLIR files `tpp_baseline.mlir`, `tpp_deduplicated.mlir` and `accfg_deduplicated.mlir` to object files.
- run `make tpp_baseline.out tpp_deduplicated.out accfg_deduplicated.out`
- run multiple passes of each benchmark, writing the stdout down.


## Passes

Some notes I made trying to tame the beast:

Starting with a a simple `base.mlir` file:

```mlir
func.func @entry(%arg0: tensor<1024x1024xbf16>,
                 %arg1: tensor<1024x1024xbf16>,
                 %arg2: tensor<1024x1024xbf16>) -> tensor<1024x1024xbf16> {
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<1024x1024xbf16>, tensor<1024x1024xbf16>)
                     outs(%arg2 : tensor<1024x1024xbf16>) -> tensor<1024x1024xbf16>
  return %0 : tensor<1024x1024xbf16>
}
```

- we can run the entire lowering pipeline through the use of `tpp-mlir -default-tpp-passes`
- Otherwise, we need to perform these lowering steps:
  - preprocessing: `-tpp-mapping -lower-packs-unpacks -canonicalize -cse -bufferize`
  - linalg to dispatch: `-linalg-lowering -convert-forall-to-parallel`
  - tiling and such `-scf-parallel-loop-tiling-pass=parallel-loop-tile-sizes=4,8 -canonicalize`
  - insert tilecfg ops: `-intel-amx-tile-config-insertion-pass`
  - LICM: `-loop-invariant-code-motion -intel-amx-tile-config-hoisting-pass`
  - to functions: `-convert-xsmm-to-func -canonicalize -cse`

Note: After this the IR appears to be in the same form as `default-tpp-passes`, but it's actually slightly different!

