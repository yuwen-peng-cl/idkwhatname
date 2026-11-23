"""Summarize a complete k experiment across independent world and model seeds.

Per-model CSV has one row per trained model. The summary's mean and sample SD
are over WORLD means (averaged over model seeds within each world). This keeps
training runs on the same synthetic world from masquerading as independent worlds.
"""

import argparse
import csv
import statistics
from pathlib import Path

from collect import ROOT, row_of


METRICS = ("static_tested_exact", "lm_fc", "probe_ab_L4", "probe_b_L4")


def collect(root, world_seeds, ks, model_seeds):
    rows = []
    missing = []
    for world_seed in world_seeds:
        for k in ks:
            for model_seed in model_seeds:
                stem = f"nestk_w{world_seed}_k{k}__model_both_s{model_seed}"
                path = root / "results" / "models" / f"{stem}.json"
                if not path.is_file():
                    missing.append(path)
                    continue
                rows.append({"world_seed": world_seed, "k": k, "model_seed": model_seed,
                             **row_of(path)})
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} result(s), starting with {missing[0]}")
    return rows


def summarize(rows, world_seeds, ks):
    out = []
    for k in ks:
        subset = [r for r in rows if r["k"] == k]
        record = {"k": k, "bound_1_over_k": 1 / k,
                  "n_worlds": len(world_seeds), "n_models": len(subset)}
        for key in METRICS:
            world_means = [statistics.mean(float(r[key]) for r in subset if r["world_seed"] == w)
                           for w in world_seeds]
            record[f"{key}_mean"] = statistics.mean(world_means)
            record[f"{key}_sd_world"] = statistics.stdev(world_means) if len(world_means) > 1 else 0.0
        out.append(record)
    return out


def save_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--world-seeds", nargs="+", type=int, required=True)
    p.add_argument("--ks", nargs="+", type=int, required=True)
    p.add_argument("--seeds", nargs="+", type=int, required=True, help="model seeds")
    args = p.parse_args(argv)
    world_seeds, ks, seeds = (list(dict.fromkeys(values)) for values in
                              (args.world_seeds, args.ks, args.seeds))
    rows = collect(ROOT, world_seeds, ks, seeds)
    summary = summarize(rows, world_seeds, ks)
    detail_path = ROOT / "results" / "multiworld_models.csv"
    summary_path = ROOT / "results" / "multiworld_summary.csv"
    save_csv(detail_path, rows)
    save_csv(summary_path, summary)

    def display(row, key):
        return f"{row[key + '_mean']:.3f} ± {row[key + '_sd_world']:.3f}"

    print("| k | 1/k | worlds | models | exact | LM head | probe A→B | probe B |")
    print("|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in summary:
        print(f"| {r['k']} | {r['bound_1_over_k']:.3f} | {r['n_worlds']} | {r['n_models']} | "
              f"{display(r, 'static_tested_exact')} | {display(r, 'lm_fc')} | "
              f"{display(r, 'probe_ab_L4')} | {display(r, 'probe_b_L4')} |")
    print(f"Detailed results: {detail_path}\nSummary: {summary_path}")


if __name__ == "__main__":
    main()
