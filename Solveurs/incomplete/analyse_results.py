#!/usr/bin/env python3
# stats_json_folder.py
import argparse
import json
import math
from pathlib import Path
from collections import defaultdict, Counter

NUM_KEYS_TOP = [
    "objective", "solutions",
    "solveTime_ms", "flatTime_ms", "nodes", "failures", "propagations",
    "python_total_ms", "timeout_sec", "optimisation_level",
]
NUM_KEYS_SOLUTION = ["objective"]  # dans solution.{...}

CAT_KEYS_TOP = ["status", "solver", "instance"]

def is_number(x):
    return isinstance(x, (int, float)) and not (isinstance(x, float) and (math.isnan(x) or math.isinf(x)))

def quantile(sorted_vals, q):
    # q in [0,1], quantile linéaire (type R-7)
    n = len(sorted_vals)
    if n == 0:
        return None
    if n == 1:
        return float(sorted_vals[0])
    h = (n - 1) * q
    lo = int(math.floor(h))
    hi = int(math.ceil(h))
    if lo == hi:
        return float(sorted_vals[lo])
    return float(sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (h - lo))

def describe(values):
    vals = [v for v in values if is_number(v)]
    vals.sort()
    n = len(vals)
    if n == 0:
        return {"n": 0}
    mean = sum(vals) / n
    # écart-type (échantillon)
    var = sum((v - mean) ** 2 for v in vals) / (n - 1) if n > 1 else 0.0
    return {
        "n": n,
        "min": float(vals[0]),
        "q1": quantile(vals, 0.25),
        "median": quantile(vals, 0.50),
        "q3": quantile(vals, 0.75),
        "max": float(vals[-1]),
        "mean": float(mean),
        "std": float(math.sqrt(var)),
    }

def fmt_desc(d):
    if d.get("n", 0) == 0:
        return "n=0"
    return (
        f"n={d['n']} min={d['min']:.4g} q1={d['q1']:.4g} med={d['median']:.4g} "
        f"q3={d['q3']:.4g} max={d['max']:.4g} mean={d['mean']:.4g} std={d['std']:.4g}"
    )

def main():
    ap = argparse.ArgumentParser(description="Stats descriptives sur un dossier de JSON de résultats.")
    ap.add_argument("folder", type=Path, help="Dossier contenant les fichiers .json")
    ap.add_argument("--recursive", action="store_true", help="Parcourir récursivement les sous-dossiers")
    ap.add_argument("--out-csv", type=Path, default=None, help="Écrire un CSV récapitulatif (1 ligne par JSON)")
    args = ap.parse_args()

    folder = args.folder
    if not folder.exists() or not folder.is_dir():
        raise SystemExit(f"Erreur: dossier invalide: {folder}")

    pattern = "**/*.json" if args.recursive else "*.json"
    files = sorted(folder.glob(pattern))
    if not files:
        raise SystemExit(f"Aucun .json trouvé dans {folder} (recursive={args.recursive})")

    # collecteurs
    num_series = defaultdict(list)   # key -> [values]
    cat_counts = {k: Counter() for k in CAT_KEYS_TOP if k != "instance"}
    solver_status = Counter()

    rows = []  # pour CSV

    bad = 0
    for fp in files:
        try:
            data = json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            bad += 1
            continue

        row = {"_file": str(fp)}

        # top-level numériques
        for k in NUM_KEYS_TOP:
            v = data.get(k, None)
            num_series[k].append(v if is_number(v) else None)
            row[k] = v

        # solution.*
        sol = data.get("solution") or {}
        if isinstance(sol, dict):
            for k in NUM_KEYS_SOLUTION:
                v = sol.get(k, None)
                key = f"solution.{k}"
                num_series[key].append(v if is_number(v) else None)
                row[key] = v

            # exemple: stats simples sur conflict (somme, longueur)
            conflict = sol.get("conflict", None)
            if isinstance(conflict, list) and conflict:
                s = sum(x for x in conflict if is_number(x))
                row["solution.conflict_sum"] = s
                row["solution.conflict_len"] = len(conflict)
                num_series["solution.conflict_sum"].append(s)
                num_series["solution.conflict_len"].append(len(conflict))
            else:
                row["solution.conflict_sum"] = None
                row["solution.conflict_len"] = None
                num_series["solution.conflict_sum"].append(None)
                num_series["solution.conflict_len"].append(None)

            # idem pour w (longueur)
            w = sol.get("w", None)
            if isinstance(w, list) and w:
                row["solution.w_len"] = len(w)
                num_series["solution.w_len"].append(len(w))
            else:
                row["solution.w_len"] = None
                num_series["solution.w_len"].append(None)

        # catégoriels
        status = data.get("status", None)
        solver = data.get("solver", None)
        if status is not None:
            cat_counts["status"][str(status)] += 1
            row["status"] = status
        if solver is not None:
            cat_counts["solver"][str(solver)] += 1
            row["solver"] = solver
        if status is not None and solver is not None:
            solver_status[(str(solver), str(status))] += 1

        rows.append(row)

    # sortie
    print(f"Fichiers .json lus: {len(files)} | OK: {len(rows)} | erreurs lecture/parse: {bad}\n")

    # stats numériques
    print("=== Stats numériques (valeurs None ignorées) ===")
    for k in sorted(num_series.keys()):
        vals = [v for v in num_series[k] if is_number(v)]
        d = describe(vals)
        print(f"- {k:22s} {fmt_desc(d)}")
    print()

    # catégoriels
    for k, c in cat_counts.items():
        print(f"=== Répartition {k} ===")
        total = sum(c.values())
        for val, cnt in c.most_common():
            pct = (cnt / total * 100.0) if total else 0.0
            print(f"- {val}: {cnt} ({pct:.1f}%)")
        print()

    print("=== (solver, status) ===")
    for (solver, status), cnt in solver_status.most_common():
        print(f"- {solver:12s} {status:12s} : {cnt}")
    print()

    # CSV (optionnel)
    if args.out_csv:
        import csv
        # colonnes = union de toutes les keys
        cols = sorted({k for r in rows for k in r.keys()})
        with args.out_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"CSV écrit: {args.out_csv}")

if __name__ == "__main__":
    main()
