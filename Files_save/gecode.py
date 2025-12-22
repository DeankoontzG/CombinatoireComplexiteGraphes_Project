import minizinc

gecode = minizinc.Solver.lookup("gecode")
print(gecode)

"""
def solve_with_gecode(model_path, dzn_path, strategy="first_fail", timeout_ms=60000):
    model = minizinc.Model(model_path)
    gecode = minizinc.Solver.lookup("gecode")
    instance = minizinc.Instance(gecode, model)
    instance.add_file(dzn_path)

    # Tu peux définir ta stratégie de recherche directement dans le .mzn :
    # solve :: int_search(x, first_fail, indomain_min, complete) satisfy;

    result = instance.solve(
        timeout=timeout_ms,
        random_seed=42
    )

    print(f"Résolu avec stratégie {strategy}, statut: {result.status}")
    print("Temps :", result.statistics["solveTime"], "sec")
    print("Solutions trouvées :", len(result), "\n")

    return result
"""