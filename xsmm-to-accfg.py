#!/usr/bin/env python3

import sys
from typing import Sequence, TextIO
from xdsl.pattern_rewriter import RewritePattern, PatternRewriter, PatternRewriteWalker
from xdsl.passes import ModulePass
from xdsl.xdsl_opt_main import xDSLOptMain
from xdsl.ir import SSAValue, Operation
from xdsl.dialects.builtin import UnregisteredOp
import io
from dataclasses import dataclass

# monkey-patch accfg dialect in xdsl
from compiler.dialects import accfg
from xdsl.dialects import accfg as xdsl_accfg
from xdsl.dialects import memref, builtin, func
xdsl_accfg.ACCFG = accfg.ACCFG

from compiler.transforms import convert_accfg_to_csr

@dataclass
class XSMMToAccfgPattern(RewritePattern):
    state: None | SSAValue = None
    token: None | SSAValue = None

    def match_and_rewrite(self, op, rewriter: PatternRewriter):
        if not isinstance(op, UnregisteredOp):
            return

        if op.op_name.data == "xsmm.IntelAMXtileConfig":
            is_reset = 128 in [x.value.data for x in op.operands[0].owner.properties["flags"]]
            if is_reset:
                buff_op = op.operands[1].owner
                rewriter.replace_matched_op(
                    accfg.AwaitOp(self.token)
                )
            else:
                buff_op = op.operands[1].owner
                rewriter.replace_matched_op(
                    setup_op := accfg.SetupOp(op.operands[0], ["conf"], "amx"),
                    new_results=[],
                )
                self.state = setup_op.out_state

            # we need to erase the buffer alloc op as well
            # but we only delete it after rewriting the last setup op
            if len(buff_op.results[0].uses) == 0:
                rewriter.erase_op(buff_op)

        elif op.op_name.data == "xsmm.brgemm":
            rewriter.replace_matched_op(
                launch := accfg.LaunchOp(op.operands, ["gemm", "a", "b", "out", "size"], self.state),
                new_results=[],
            )
            self.token = launch.token

xsmm_IntelAMXtileConfigOp = UnregisteredOp.with_name("xsmm.IntelAMXtileConfig")
xsmm_brgemmOp = UnregisteredOp.with_name("xsmm.brgemm")
xsmm_IntelAMXtileConfig_dispatchOp = UnregisteredOp.with_name("xsmm.IntelAMXtileConfig.dispatch")

configMemrefType = builtin.MemRefType(builtin.IntegerType(8), [64])

@dataclass
class ACCFGToXSMMPattern(RewritePattern):
    alloced: SSAValue | None = None

    def match_and_rewrite(self, op, rewriter: PatternRewriter):
        if isinstance(op, accfg.SetupOp):
            rewriter.replace_matched_op(
                [
                    alloc := memref.Alloca([], [], configMemrefType),
                    xsmm_IntelAMXtileConfigOp.create(
                        operands=[*op.values, alloc.memref],
                    ),
                ], 
                new_results=[None], 
                safe_erase=False
            )
            self.alloced = alloc.memref
        elif isinstance(op, accfg.LaunchOp):
            rewriter.replace_matched_op(
                xsmm_brgemmOp.create(
                    operands=[*op.operands],
                    properties={"data_type": builtin.IntegerAttr.from_int_and_width(2, 64)}
                ),
                new_results=[None],
                safe_erase=False,
            )
        elif isinstance(op, accfg.AwaitOp):
            rewriter.erase_matched_op()
        elif isinstance(op, accfg.ResetOp):
            reset_cfg = find_reset_conf(op)
            rewriter.replace_matched_op([
                xsmm_IntelAMXtileConfigOp.create(
                    operands=[reset_cfg, self.alloced],
                ),
            ], new_results=[])


def find_reset_conf(op: Operation):
    func_op = op
    while not isinstance(func_op, func.FuncOp):
        func_op = func_op.parent_op()
    for op in func_op.body.block.ops:
        if isinstance(op, UnregisteredOp) and op.op_name.data == 'xsmm.IntelAMXtileConfig.dispatch':
            is_reset = 128 in [x.value.data for x in op.properties["flags"]]
            if is_reset:
                return op.results[0]


class XSMMToAccfgPass(ModulePass):
    name = "xsmm-to-accfg"

    def apply(self, ctx, op):
        PatternRewriteWalker(
            XSMMToAccfgPattern(),
            apply_recursively=False,
        ).rewrite_module(op)


class ACCFGToXSMMPass(ModulePass):
    name = "accfg-to-xsmm"

    def apply(self, ctx, op):
        PatternRewriteWalker(
            ACCFGToXSMMPattern(),
            apply_recursively=False,
        ).rewrite_module(op)
        PatternRewriteWalker(
            convert_accfg_to_csr.DeleteAllStates(),
        ).rewrite_module(op)

        
def process(prog: str):
    obj = StreamingXDSLOptMain(args=[
        "-p", XSMMToAccfgPass.name, '--allow-unregistered-dialect', '--print-op-generic'
    ], input=io.StringIO(prog), passes=(XSMMToAccfgPass,))
    obj.run()
    print(obj.get_written_output())


def reverse(prog: str):
    obj = StreamingXDSLOptMain(args=[
        "-p", ACCFGToXSMMPass.name, '--allow-unregistered-dialect', '--print-op-generic'
    ], input=io.StringIO(prog), passes=(ACCFGToXSMMPass,))
    obj.run()
    print(obj.get_written_output().replace(
        '"scf.yield"() {"was_reduce"}', '"scf.reduce"()'
    ))


class StreamingXDSLOptMain(xDSLOptMain):
    _input_stream: TextIO
    _output_stream: TextIO
    _passes: Sequence[type[ModulePass]]

    def __init__(self, description = "xDSL modular optimizer driver", args = None, input = None, output = None, passes = tuple()):
        self._passes = passes
        super().__init__(description, args)
        self._input_stream = input
        if output is None:
            output = io.StringIO()
        self._output_stream = output
    
    def register_all_passes(self):
        super().register_all_passes()
        for _pass in self._passes:
            self.register_pass(_pass.name, lambda _pass=_pass: _pass)

    def get_written_output(self) -> str | None:
        if isinstance(self._output_stream, io.StringIO):
            return self._output_stream.getvalue()

    def get_input_stream(self):
        return self._input_stream, "mlir"
    
    def prepare_output(self):
        return self._output_stream
        
    def run(self):
        """
        Executes the different steps.
        """
        # we overwrite run to not call close at the end, as to not throw
        # away our StringIO buffers just yet
        chunks, file_extension = self.prepare_input()
        output_stream = self.prepare_output()
        try:
            for i, (chunk, offset) in enumerate(chunks):
                try:
                    if i > 0:
                        output_stream.write("// -----\n")
                    module = self.parse_chunk(chunk, file_extension, offset)

                    if module is not None:
                        if self.apply_passes(module):
                            output_stream.write(self.output_resulting_program(module))
                    output_stream.flush()
                finally:
                    chunk.close()
        finally:
            if output_stream is not sys.stdout and not isinstance(output_stream, io.StringIO):
                output_stream.close()


if __name__ == '__main__':
    inp = sys.argv[-1]
    prog = ""
    if inp == "-":
        prog = []
        for l in sys.stdin:
            prog.append(l)
        prog = "".join(prog)
    else:
        with open(inp, "r") as f:
            prog = f.read()

    if "--reverse" in sys.argv:
        reverse(prog)
    else:
        process(prog.replace(" <{overflowFlags = #arith.overflow<none>}>", "").replace('"scf.reduce"()', '"scf.yield"() {was_reduce}'))
