import subprocess
import os

WARMUP = int(os.getenv("BENCH_WARMUP_RUNS", "3"))
RUNS = int(os.getenv("BENCH_RUNS", 10))


def do_runs():
    print("bench,opt,time")
    for opt in ("O0", "O1", "O2", "O3"):
        for name in ('tpp_baseline.mlir', 'tpp_deduplicated.mlir', 'accfg_deduplicated.mlir'):
            # do three runs without recording time to "warm up"
            for _ in range(WARMUP):
                subprocess.check_output([
                    'tpp-run', f'-{opt}', "-e", "entry", "-entry-point-result=void", name
                ], text=True)
            for _ in range(RUNS):
                res = subprocess.check_output([
                    'tpp-run', f'-{opt}', "-e", "entry", "-entry-point-result=void", name
                ], text=True).strip()
                print(f"{name},{opt},{res}")

if __name__ == '__main__':
    do_runs()
