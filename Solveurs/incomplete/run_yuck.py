#!/usr/bin/env python3
from pathlib import Path
import minizinc
import json
import time
import sys
from datetime import timedelta

def to_ms(v):
    """Convertit un éventuel timedelta en millisecondes."""
    if v is None:
        return None
    if hasattr(v, "total_seconds"):
        return v.total_seconds() * 1000.0
    return v

def sol_to_dict(sol):
    """Convertit la solution MiniZinc en dict JSON-compatible."""
    if sol is None:
        return None
    out = {}
    for k, v in sol.__dict__.items():
        if k.startswith("_"):
            continue
        if hasattr(v, "tolist"):
            out[k] = v.tolist()
        else:
            out[k] = v
    return out

def run_mzn_api(
    model_path: Path,
    data_path: Path,
    solver_name: str = "yuck",
    optimisation_level: int = 1,
    timeout_sec: int = 30,
):
    model = minizinc.Model()
    model.add_file(str(model_path))

    solver = minizinc.Solver.lookup(solver_name)

    inst = minizinc.Instance(solver, model)
    inst.add_file(str(data_path))

    t0 = time.perf_counter()
    result = inst.solve(
        verbose=True,
        optimisation_level=optimisation_level,
        timeout=timedelta(seconds=timeout_sec),
    )
    t1 = time.perf_counter()

    stats = result.statistics
    metrics = {}

    metrics["instance"] = data_path.name
    metrics["status"] = str(result.status)
    metrics["objective"] = getattr(result, "objective", None)
    metrics["solutions"] = len(result)

    metrics["solveTime_ms"] = to_ms(stats.get("solveTime", None))
    metrics["flatTime_ms"] = to_ms(stats.get("flatTime", None))

    metrics["nodes"] = stats.get("nodes", None)
    metrics["failures"] = stats.get("failures", None)
    metrics["propagations"] = stats.get("propagations", None)

    metrics["python_total_ms"] = (t1 - t0) * 1000.0
    metrics["timeout_sec"] = timeout_sec
    metrics["optimisation_level"] = optimisation_level
    metrics["solver"] = solver_name

    metrics["solution"] = sol_to_dict(result.solution)

    return metrics, result

def results_subdir_name(solver_name: str, optimisation_level: int, timeout_sec: int) -> str:
    # simple, stable, filesystem-friendly
    return f"{solver_name}_opt{optimisation_level}_t{timeout_sec}s"

def save_results_json(metrics: dict, base_results_dir: Path, call_name: str, subdir: str):
    """
    Enregistre le fichier JSON dans:
      base_results_dir / subdir / call_name / <instance>.json
    """
    instance_name = metrics["instance"].replace(".dzn", "")
    outdir = base_results_dir / subdir / call_name
    outdir.mkdir(parents=True, exist_ok=True)

    outfile = outdir / f"{instance_name}.json"
    with outfile.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f">>> Résultats enregistrés dans: {outfile}")

def collect_dzn_paths(root: Path, args) -> list[Path]:
    dzns: list[Path] = []

    if not args:
        dzn = root / ".." / ".." / "instances" / "small" / "fillfactor_0.90" / "small_002.dzn"
        dzns.append(dzn)
        return dzns

    for raw in args:
        p = Path(raw)
        if not p.is_absolute():
            p = (root / p).resolve()

        if p.is_file() and p.suffix == ".dzn":
            dzns.append(p)
        elif p.is_dir():
            dzns.extend(sorted(p.glob("*.dzn")))
        else:
            print(f"[WARN] Ignoré (ni .dzn ni dossier) : {p}")

    return dzns

def main():
    root = Path(__file__).resolve().parent
    model = root / "crossword_yuck_optimisation.mzn"

    dzns = collect_dzn_paths(root, sys.argv[1:])
    if not dzns:
        print("Aucun fichier .dzn à traiter.")
        return

    # paramètres de run (centralisés)
    solver_name = "yuck"
    optimisation_level = 1
    timeout_sec = 30

    print(f">>> Modèle : {model}")
    print(f">>> {len(dzns)} instance(s) à résoudre.\n")

    base_results_dir = root / "results"
    subdir = results_subdir_name(solver_name, optimisation_level, timeout_sec)

    for dzn in dzns:
        print(f"\n=== Instance : {dzn} ===")
        metrics, _ = run_mzn_api(
            model, dzn,
            solver_name=solver_name,
            optimisation_level=optimisation_level,
            timeout_sec=timeout_sec,
        )

        print("\n--- METRICS ---")
        print(json.dumps(metrics, indent=2))

        save_results_json(metrics, base_results_dir, "", subdir)

if __name__ == "__main__":
    main()
