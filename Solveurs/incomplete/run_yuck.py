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


def run_mzn_api(model_path: Path, data_path: Path, solver_name: str = "yuck"):
    model = minizinc.Model()
    model.add_file(str(model_path))

    solver = minizinc.Solver.lookup(solver_name)

    inst = minizinc.Instance(solver, model)
    inst.add_file(str(data_path))

    t0 = time.perf_counter()
    # ⏱ timeout 30 secondes
    result = inst.solve(
        verbose=True,
        optimisation_level=1,
        timeout=timedelta(seconds=30),
    )
    t1 = time.perf_counter()

    stats = result.statistics
    metrics = {}

    metrics["instance"] = data_path.name
    metrics["status"] = str(result.status)          # OPTIMAL_SOLUTION / SATISFIED / UNKNOWN ...
    metrics["objective"] = getattr(result, "objective", None)
    metrics["solutions"] = len(result)

    metrics["solveTime_ms"] = to_ms(stats.get("solveTime", None))
    metrics["flatTime_ms"] = to_ms(stats.get("flatTime", None))

    metrics["nodes"] = stats.get("nodes", None)
    metrics["failures"] = stats.get("failures", None)
    metrics["propagations"] = stats.get("propagations", None)

    metrics["python_total_ms"] = (t1 - t0) * 1000.0
    metrics["timeout_sec"] = 30  # pour savoir avec quel timeout ça a tourné

    metrics["solution"] = sol_to_dict(result.solution)

    return metrics, result


def save_results_json(metrics: dict, results_dir: Path):
    """Enregistre le fichier JSON dans ./results/."""
    results_dir.mkdir(exist_ok=True)
    instance_name = metrics["instance"].replace(".dzn", "")
    outfile = results_dir / f"{instance_name}.json"

    with outfile.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f">>> Résultats enregistrés dans: {outfile}")


def collect_dzn_paths(root: Path, args) -> list[Path]:
    """
    Si args est vide -> un seul .dzn par défaut (small_002).
    Sinon :
      - si arg = fichier .dzn  -> on l'ajoute
      - si arg = dossier       -> on ajoute tous les .dzn dedans (non récursif)
    """
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

    print(f">>> Modèle : {model}")
    print(f">>> {len(dzns)} instance(s) à résoudre.\n")

    results_dir = root / "results"

    for dzn in dzns:
        print(f"\n=== Instance : {dzn} ===")
        metrics, _ = run_mzn_api(model, dzn, solver_name="yuck")

        print("\n--- METRICS ---")
        print(json.dumps(metrics, indent=2))

        save_results_json(metrics, results_dir)


if __name__ == "__main__":
    main()
