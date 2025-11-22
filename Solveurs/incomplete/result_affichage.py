# run_crossword.py
import argparse
from minizinc import Model, Solver, Instance, Status

def main():
    p = argparse.ArgumentParser(description="Solve crossword model and print w.")
    p.add_argument("--mzn", required=True, help="Chemin du modèle .mzn (ex: crossword.mzn)")
    p.add_argument("--dzn", help="Chemin des données .dzn (optionnel)")
    p.add_argument("--solver", default="gecode", help="Solveur MiniZinc (par défaut: gecode)")
    p.add_argument("--time", type=int, default=None, help="Limite de temps en ms (optionnel)")
    args = p.parse_args()

    model = Model(args.mzn)
    solver = Solver.lookup(args.solver)
    inst = Instance(solver, model)
    if args.dzn:
        inst.add_file(args.dzn)
    if args.time:
        result = inst.solve(timeout=args.time)
    else:
        result = inst.solve()

    if result.status in {Status.SATISFIED, Status.OPTIMAL_SOLUTION, Status.ALL_SOLUTIONS}:
        w = result["w"]  # tableau 1..S d'indices choisis
        print("w =", list(w))
    else:
        print(f"Aucune solution (status: {result.status})")

if __name__ == "__main__":
    main()
