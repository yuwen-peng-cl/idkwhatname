"""Build k worlds and run bilingual GPU training/evaluation with bounded concurrency.

Use run_windows_gpu.ps1 on Windows, or invoke this file directly with Python.
Each training process uses one CUDA GPU; evaluation currently runs on the CPU.
"""

import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
HEART_COUNTS = "4,6,9,12,16,20,26,35"


def invoke(command, log=None, dry_run=False):
    print("+ " + subprocess.list2cmdline([str(x) for x in command]), flush=True)
    if dry_run:
        return
    if log is None:
        subprocess.run(command, cwd=ROOT, check=True)
    else:
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("w", encoding="utf-8") as stream:
            subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)


def build_worlds(python, ks, dry_run):
    web = ROOT / "worlds" / "web32.json"
    invoke([python, ROOT / "world" / "web.py", "--levels", "8,8,8,8",
            "--max_eat", "2", "--seed", "42", "--out", web], dry_run=dry_run)
    for k in ks:
        invoke([python, ROOT / "world" / "build.py", "--web", web, "--k", str(k),
                "--tags", "--indiv_repeat", "4", "--untested_hearts", HEART_COUNTS,
                "--nests", "untested", "--out_dir", ROOT / "worlds" / f"nestk_k{k}"],
               dry_run=dry_run)


def check_cuda(python):
    code = ("import torch; "
            "assert torch.cuda.is_available(), 'CUDA is unavailable: install a CUDA-enabled PyTorch and check the NVIDIA driver'; "
            "print('CUDA:', torch.cuda.get_device_name(0), 'PyTorch:', torch.__version__, 'CUDA runtime:', torch.version.cuda)")
    invoke([python, "-c", code])


def run_one(python, k, seed, epochs, dry_run):
    world_name = f"nestk_k{k}"
    name = f"model_both_s{seed}" + (f"_e{epochs}" if epochs != 100 else "")
    world = ROOT / "worlds" / world_name
    model = world / name
    results = ROOT / "results" / ("models" if epochs == 100 else "smoke")
    result = results / f"{world_name}__{name}.json"
    log = ROOT / "worlds" / f"{world_name}__{name}.train.log"
    eval_log = ROOT / "worlds" / f"{world_name}__{name}.eval.log"
    print(f"START k={k} seed={seed} epochs={epochs} | log: {log}", flush=True)
    invoke([python, ROOT / "train" / "train.py", "--world_dir", world,
            "--langs", "both", "--seed", str(seed), "--epochs", str(epochs),
            "--require_cuda", "--out_dir", model], log=log, dry_run=dry_run)
    invoke([python, ROOT / "eval" / "evaluate.py", "--world_dir", world,
            "--model_dir", model / "final", "--out", result],
           log=eval_log, dry_run=dry_run)
    print(f"DONE  k={k} seed={seed} | result: {result}", flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ks", nargs="+", type=int, choices=[1, 2, 4, 8], default=[1, 2, 4, 8])
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    p.add_argument("--jobs", type=int, default=2, help="simultaneous training processes on one GPU (default 2)")
    p.add_argument("--epochs", type=int, default=100, help="non-100 runs write to results/smoke")
    p.add_argument("--build-only", action="store_true")
    p.add_argument("--skip-build", action="store_true", help="reuse previously built worlds")
    p.add_argument("--dry-run", action="store_true", help="show commands without executing them")
    args = p.parse_args(argv)
    if args.jobs < 1 or args.epochs < 1:
        p.error("--jobs and --epochs must be positive")
    if args.build_only and args.skip_build:
        p.error("--build-only and --skip-build cannot be combined")
    ks = list(dict.fromkeys(args.ks))
    seeds = list(dict.fromkeys(args.seeds))
    python = Path(sys.executable)

    if not args.build_only and not args.dry_run:
        check_cuda(python)
    if not args.skip_build:
        build_worlds(python, ks, args.dry_run)
    if args.build_only:
        return 0
    if args.skip_build and not args.dry_run:
        for k in ks:
            for filename in ("world.json", "corpus_a.txt", "corpus_b.txt"):
                path = ROOT / "worlds" / f"nestk_k{k}" / filename
                if not path.is_file():
                    p.error(f"missing {path}; run without --skip-build")

    failures = []
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(run_one, python, k, seed, args.epochs, args.dry_run): (k, seed)
                   for k in ks for seed in seeds}
        for future in as_completed(futures):
            try:
                future.result()
            except subprocess.CalledProcessError as exc:
                k, seed = futures[future]
                failures.append((k, seed))
                print(f"FAILED k={k} seed={seed}: exit {exc.returncode}; inspect train/eval log in worlds/", file=sys.stderr)
            except Exception as exc:
                k, seed = futures[future]
                failures.append((k, seed))
                print(f"FAILED k={k} seed={seed}: {exc}", file=sys.stderr)
    if failures:
        print(f"{len(failures)} job(s) failed; summary was not regenerated.", file=sys.stderr)
        return 1
    if args.epochs == 100:
        invoke([python, ROOT / "scripts" / "collect.py"], dry_run=args.dry_run)
    else:
        print("Smoke results saved in results/smoke/; full summary was not changed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
