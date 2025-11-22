import json
import time
import random
from ortools.sat.python import cp_model
import pandas as pd

def clean_lexique(char, char_replacement):
    # Charger le fichier parquet existant
    df = pd.read_parquet('../../data/lexique_filtre_cleaned.parquet')

    # Vérifier les premières lignes pour comprendre la structure du DataFrame
    print(df.head())

    # Remplacer les '.' par des espaces dans la colonne des mots (supposons que la colonne s'appelle 'mot')
    df['ortho'] = df['ortho'].str.replace(char, char_replacement, regex=False)

    # Sauvegarder le DataFrame nettoyé dans un nouveau fichier parquet
    df.to_parquet('lexique_filtre_cleaned.parquet')

def charger_grille(fichier_json):
    """Charge une grille depuis un fichier JSON."""
    with open(fichier_json, 'r') as f:
        data = json.load(f)

    grid = data["grid"]
    slots = data["slots"]
    intersections = data["intersections"]
    dictionary = data["dictionary"]

    return grid, slots, intersections, dictionary

def status_name(status_code):
    if status_code == cp_model.OPTIMAL:
        return "Optimal"
    elif status_code == cp_model.FEASIBLE:
        return "Feasible"
    elif status_code == cp_model.INFEASIBLE:
        return "Infeasible"
    elif status_code == cp_model.MODEL_INVALID:
        return "Model Invalid"
    elif status_code == cp_model.UNKNOWN:
        return "Unknown"
    else:
        return "Unknown Status"

def afficher_grille_solution(grid, slots, word_vars, solver):
    """Affiche la grille résolue en utilisant les valeurs du solveur."""
    solved_grid = [list(row) for row in grid]

    print("\n--- GRILLE RÉSOLUE ---")

    for slot in slots:
        slot_id = slot['id']
        start_row = slot['row']
        start_col = slot['col']
        orientation = slot['orientation']

        # On s'assure que le slot est dans word_vars (au cas où il ait été rendu infaisable)
        if slot_id not in word_vars:
            continue

        word_values = [solver.Value(letter_var) for letter_var in word_vars[slot_id]]

        # Convertit les valeurs numériques en lettres
        solved_word = "".join(
            chr(value + ord('a') - 1)
            for value in word_values
        )

        for i, letter in enumerate(solved_word):
            row = start_row + (i if orientation == 'V' else 0)
            col = start_col + (i if orientation == 'H' else 0)

            if 0 <= row < len(solved_grid) and 0 <= col < len(solved_grid[0]):
                 solved_grid[row][col] = letter.upper()

    # --- Affichage Final de la Grille ---
    num_cols = len(solved_grid[0])


    print("      " + " ".join([str(i % 10) for i in range(num_cols)]))
    print("      " + "—" * (2 * num_cols - 1))

    for r, row in enumerate(solved_grid):
        print(f"{r: <4} | " + " ".join(row))

def resolveClassic(model, timeout=60.0):
    start_exec_time = time.perf_counter()
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = timeout
    status = solver.Solve(model)
    end_exec_time = time.perf_counter()
    exec_time = end_exec_time - start_exec_time
    return status, exec_time, solver

def resolveWithHeuristic(model, ordered_vars, timeout=60.0):

    start_exec_time = time.perf_counter()

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = timeout


    model.AddDecisionStrategy(
        ordered_vars,
        cp_model.CHOOSE_FIRST,  # Choisit les variables dans l'ordre fourni
        cp_model.SELECT_MIN_VALUE  # Choisit la plus petite valeur dans le domaine
    )

    status = solver.Solve(model)

    end_exec_time = time.perf_counter()
    exec_time = end_exec_time - start_exec_time

    return status, exec_time, solver

def executeORTools(gridpath, display_grid = False, timeout = 60.0, heuristic = None):

    start_prep_time = time.perf_counter()

    model = cp_model.CpModel()
    grid, slots, intersections, dictionary = charger_grille(gridpath)

    word_vars = {} # Variables de lettres (1-26)
    allowed_tuples_by_slot = {} # "Mots" = tuples de lettres valides

    for slot in slots:
        slot_id = slot['id']
        slot_length = slot['length']

        # 1.a Variables de Lettres
        letter_vars = [model.NewIntVar(1, 26, f"letter_{slot_id}_{i}") for i in range(slot_length)]
        word_vars[slot_id] = letter_vars

        # 1.b Préparation des tuples valides pour ce slot
        allowed_tuples = []
        for word in dictionary:
            if len(word) == slot_length:
                word_values = [(ord(letter) - ord('a') + 1) for letter in word]
                allowed_tuples.append(word_values)

        if not allowed_tuples:
            print(f"ATTENTION: Aucun mot trouvé pour le slot {slot_id} (longueur {slot_length}). Problème infaisable.")
            model.Add(0 == 1) # Rend le modèle impossible
            continue

        allowed_tuples_by_slot[slot_id] = allowed_tuples

    # --- 2. Ajout des Contraintes ---

    # --- 2.a Contrainte de mots existants dans le dictionnaire ---
    for slot in slots:
        slot_id = slot['id']
        if slot_id in allowed_tuples_by_slot:
            model.AddAllowedAssignments(word_vars[slot_id], allowed_tuples_by_slot[slot_id])

    # --- 2.b : Contraintes d'intersection ---
    for intersection in intersections:
        s1, p1, s2, p2 = intersection['s1'], intersection['p1'], intersection['s2'], intersection['p2']
        index_p1 = p1 - 1
        index_p2 = p2 - 1
        if s1 in word_vars and s2 in word_vars:
            model.Add(word_vars[s1][index_p1] == word_vars[s2][index_p2])


    # --- 3. Résolution du Problème ---
    end_prep_time = time.perf_counter()
    prep_time = end_prep_time - start_prep_time


    if heuristic == None:
        status, exec_time, solver = resolveClassic(model, timeout=timeout)
    elif heuristic == "IntersectionsFirst":
        print("Using Intersections First Heuristic")

        # Heuristique 1 : on commence par les variables des intersections.
        slot_to_vars = {slot['id']: word_vars[slot['id']] for slot in slots if slot['id'] in word_vars}
        all_model_vars = [v for slot_id in slot_to_vars for v in slot_to_vars[slot_id]]
        intersection_vars = set()

        for intersection in intersections:
            s1, p1, s2, p2 = intersection['s1'], intersection['p1'], intersection['s2'], intersection['p2']
            if s1 in word_vars and s2 in word_vars:
                intersection_vars.add(word_vars[s1][p1 - 1])
                intersection_vars.add(word_vars[s2][p2 - 1])

        ordered_vars_1 = list(intersection_vars)
        # On ajoute les variables restantes
        ordered_vars_1.extend([var for var in all_model_vars if var not in intersection_vars])

        status, exec_time, solver = resolveWithHeuristic(model, ordered_vars_1 ,timeout=timeout)

    elif heuristic == "LongestWordsFirst":
        print("Using Longest Words First Heuristic")

        slot_to_vars = {slot['id']: word_vars[slot['id']] for slot in slots if slot['id'] in word_vars}

        # Heuristique 2 : on commence par les mots les plus longs
        slots_by_length = sorted(slots, key=lambda s: s['length'], reverse=True)
        ordered_vars_2 = [var for slot in slots_by_length for var in slot_to_vars.get(slot['id'], [])]

        status, exec_time, solver = resolveWithHeuristic(model, ordered_vars_2 ,timeout=timeout)

    elif heuristic == "MostIntersectionsFirst":
        print("Using Most Intersections Words First Heuristic")

        slot_to_vars = {slot['id']: word_vars[slot['id']] for slot in slots if slot['id'] in word_vars}

        # Heuristique 3 : mots ayant le plus d'intersections
        intersections_count = {slot['id']: 0 for slot in slots}
        for intersection in intersections:
            intersections_count[intersection['s1']] += 1
            intersections_count[intersection['s2']] += 1

        slots_by_intersections = sorted(slots, key=lambda s: intersections_count[s['id']], reverse=True)
        ordered_vars_3 = [var for slot in slots_by_intersections for var in slot_to_vars.get(slot['id'], [])]

        status, exec_time, solver = resolveWithHeuristic(model, ordered_vars_3 ,timeout=timeout)
    else :
        print(f"Wrong heuristic name : {heuristic}, using classic resolution")
        status, exec_time, solver = resolveClassic(model, timeout=timeout)


    # Affichage du statut et des temps
    print(f"{gridpath[-15:-5]} | status : {status_name(status)}; Time Exec : {exec_time:.4f}; Time Prep : {prep_time:.4f}")

    # OPTIONNEL : Afficher la solution trouvée
    if (status == cp_model.OPTIMAL or status == cp_model.FEASIBLE) and display_grid:
        print("\n--- SOLUTION ---")
        afficher_grille_solution(grid, slots, word_vars, solver)

    return status, exec_time, prep_time
