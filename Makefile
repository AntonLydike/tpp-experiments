.PHONY: clean celan-venv

all: tpp_baseline.mlir tpp_deduplicated.mlir accfg_deduplicated.mlir

venv:
	python3 -m venv venv
	git clone https://github.com/KULeuven-MICAS/snax-mlir
	bash -c "cd snax-mlir; git checkout 170f122ff3a0732327cdadafe38fcdaf47860f1a"
	bash -c "source venv/bin/activate && pip install ./snax-mlir && pip install git+https://github.com/xdslproject/xdsl.git@566496ddd8a9c5109fd577a230cba73c9ace47f3"

accfg-input.mlir: base.mlir
	tpp-opt -tpp-mapping -lower-packs-unpacks -canonicalize -cse -bufferize -linalg-lowering -convert-forall-to-parallel -scf-parallel-loop-tiling-pass=parallel-loop-tile-sizes=4,8 -canonicalize -intel-amx-tile-config-insertion-pass -loop-invariant-code-motion --mlir-print-op-generic $< | ./insert-func-call-to-timer.py > $@

tpp_baseline.mlir: accfg-input.mlir
	tpp-opt $< -convert-xsmm-to-func -canonicalize -cse | sed 's/@tpp_entrypoint_name/@tpp_baseline/g' > $@

tpp_deduplicated.mlir: accfg-input.mlir
	tpp-opt -intel-amx-tile-config-hoisting-pass -convert-xsmm-to-func -canonicalize -cse $< | sed 's/@tpp_entrypoint_name/@tpp_deduplicated/g' > $@

accfg_deduplicated.mlir: accfg-input.mlir
	./xsmm-to-accfg.py $< | snax-opt --print-op-generic --allow-unregistered-dialect -p "accfg-trace-states,accfg-dedup,accfg-insert-resets" | ./xsmm-to-accfg.py --reverse - | sed 's/@tpp_entrypoint_name/@accfg_deduplicated/g' | tpp-opt -convert-xsmm-to-func -canonicalize -cse > $@


# individual benchmark runners rules:
# these require the object file to be built. I don't know how to do that, so omitted from this file.

tpp_baseline.out: tpp_baseline.o runner.cpp
	clang++ -DKERNEL_NAME=tpp_baseline -O3 $^ -o $@

tpp_deduplicated.out: tpp_deduplicated.o runner.cpp
	clang++ -DKERNEL_NAME=tpp_deduplicated -O3 $^ -o $@

accfg_deduplicated.out: accfg_deduplicated.o runner.cpp
	clang++ -DKERNEL_NAME=accfg_deduplicated -O3 $^ -o $@

dummy_kernel.out: dummy_kernel.o runner.cpp
	clang++ -DKERNEL_NAME=dummy_kernel -O3 $^ -o $@

# dummy runner, for testing:
dummy_kernel.o: dummy_kernel.cpp
	clang++ -c -O0 $< -o $@


clean:
	rm -f accfg-input.mlir tpp_baseline.mlir tpp_deduplicated.mlir accfg_deduplicated.mlir
	rm -f tpp_baseline tpp_deduplicated accfg_deduplicated dummy_kernel tpp_baseline.o tpp_deduplicated.o accfg_deduplicated.o dummy_kernel.o 

clean-venv:
	rm -rf venv snax-mlir
