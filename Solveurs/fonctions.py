import json
import time
import random
from ortools.sat.python import cp_model
import pandas as pd

# ------------------------------------------
# F1 : Nettoyage du lexique (remplacement de caractères inutilies)
# ------------------------------------------
def clean_lexique(char, char_replacement):
    # Charger le fichier parquet existant
    df = pd.read_parquet('../../data/lexique_filtre_cleaned.parquet')

    print(df.head())

    # Remplacer les '.' par des espaces dans la colonne des mots
    df['ortho'] = df['ortho'].str.replace(char, char_replacement, regex=False)

    # Sauvegarder le DataFrame nettoyé dans un nouveau fichier parquet
    df.to_parquet('lexique_filtre_cleaned.parquet')

# ------------------------------------------
# F2 : Charger une grille depuis un JSON
# ------------------------------------------

def charger_grille(fichier_json):
    """Charge une grille depuis un fichier JSON."""
    with open(fichier_json, 'r') as f:
        data = json.load(f)
        
    grid = data["grid"]
    slots = data["slots"]
    intersections = data["intersections"]
    dictionary = data["dictionary"]
    
    return grid, slots, intersections, dictionary

# ------------------------------------------
# F3 : Transformer le code du statut OR-Tools en texte 
# ------------------------------------------

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


# ------------------------------------------
# F4 : Permet de visualiser une grille résolue
# ------------------------------------------

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

# ------------------------------------------
# F5 : Fonctions de résolution avec différentes stratégies (OR-Tools)
# ------------------------------------------

# ------------------------------------------
# F5.a : appelle la résolution via OR-Tools (heuristique par défaut)
# ------------------------------------------
def resolveClassic(model, timeout=60.0):
    start_exec_time = time.perf_counter()
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = timeout 
    status = solver.Solve(model)
    end_exec_time = time.perf_counter()
    exec_time = end_exec_time - start_exec_time
    return status, exec_time, solver

# ------------------------------------------
# F5.b : appelle la résolution via OR-Tools - heuristiques customisées
# ------------------------------------------
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

# -------------------------------------------------------------------------------------------------------
# F5.c : modélisation du problème, des différentes heuristiques, et appel de la résolution (F5.a ou F5.b)
# -------------------------------------------------------------------------------------------------------

def executeORTools(gridpath, display_grid = False, timeout = 60.0, heuristic = None):

    start_prep_time = time.perf_counter()

    model = cp_model.CpModel()
    grid, slots, intersections, dictionary = charger_grille(gridpath)

    # --- 1. Définition des variables ---

    word_vars = {} # Variables de lettres (1-26)
    allowed_tuples_by_slot = {} # "Mots" = tuples de lettres valides

    SCRABBLE_POINTS = [0, 1, 3, 3, 2, 1, 4, 2, 4, 1, 8, 10, 1, 2, 1, 1, 3, 8, 1, 1, 1, 1, 4, 10, 10, 10, 10]
    total_score_vars = []

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

        # 1.c Ajout de la variable de score pour chaque lettre
        for i, letter_var in enumerate(letter_vars):
            score_var = model.NewIntVar(0, 10, f"score_{slot_id}_{i}")
            # Contrainte AddElement : score_var = SCRABBLE_POINTS[letter_var]
            model.AddElement(letter_var, SCRABBLE_POINTS, score_var)
            total_score_vars.append(score_var)

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

    # Création de la fonciton objectif
    total_score = model.NewIntVar(0, 10000, "total_scrabble_score")
    model.Add(total_score == sum(total_score_vars))
    model.Maximize(total_score)

    if heuristic == None:
        status, exec_time, solver = resolveClassic(model, timeout=timeout)

    # Heuristique 1 : on commence par les variables des intersections.
    elif heuristic == "IntersectionsFirst": 
        print("Using Intersections First Heuristic")

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

    # Heuristique 2 : on commence par les mots les plus longs
    elif heuristic == "LongestWordsFirst":
        print("Using Longest Words First Heuristic")

        slot_to_vars = {slot['id']: word_vars[slot['id']] for slot in slots if slot['id'] in word_vars}

        slots_by_length = sorted(slots, key=lambda s: s['length'], reverse=True)
        ordered_vars_2 = [var for slot in slots_by_length for var in slot_to_vars.get(slot['id'], [])]

        status, exec_time, solver = resolveWithHeuristic(model, ordered_vars_2 ,timeout=timeout)

    # Heuristique 3 : mots ayant le plus d'intersections
    elif heuristic == "MostIntersectionsFirst": 
        print("Using Most Intersections Words First Heuristic")

        slot_to_vars = {slot['id']: word_vars[slot['id']] for slot in slots if slot['id'] in word_vars}
    
        intersections_count = {slot['id']: 0 for slot in slots}
        for intersection in intersections:
            intersections_count[intersection['s1']] += 1
            intersections_count[intersection['s2']] += 1

        slots_by_intersections = sorted(slots, key=lambda s: intersections_count[s['id']], reverse=True)
        ordered_vars_3 = [var for slot in slots_by_intersections for var in slot_to_vars.get(slot['id'], [])]

        status, exec_time, solver = resolveWithHeuristic(model, ordered_vars_3 ,timeout=timeout)

    #  Heuristique 4 : On commence par slot le plus contraint, puis on explore tous les mots qui le croisent. etc
    elif heuristic == "TopologicalOrder":
        print("Using Topological Order Heuristic (BFS on Most Constrained Slot)")

        slot_to_vars = {slot['id']: word_vars[slot['id']] for slot in slots if slot['id'] in word_vars}
        
        intersections_count = {slot['id']: 0 for slot in slots}
        for intersection in intersections:
            intersections_count[intersection['s1']] += 1
            intersections_count[intersection['s2']] += 1

        start_slot_id = max(intersections_count, key=intersections_count.get)
        

        adj = {slot['id']: set() for slot in slots}
        for intersection in intersections:
            adj[intersection['s1']].add(intersection['s2'])
            adj[intersection['s2']].add(intersection['s1'])

        # On ordonne les mots à explorer
        queue = [start_slot_id]
        visited = {start_slot_id}
        ordered_slots = []
        
        while queue:
            s_id = queue.pop(0)
            ordered_slots.append(s_id)
            for neighbor_id in sorted(list(adj[s_id]), key=lambda n: intersections_count.get(n, 0), reverse=True):
                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    queue.append(neighbor_id)

        # On en extrait les lettres (variables) ordonnées qui en découlent
        ordered_vars_4 = [
            var for slot_id in ordered_slots 
            for var in (slot_to_vars.get(slot_id) or [])
        ]
        
        # Sécurité : on ajoute les éventuels slots isolés
        all_slot_ids = {slot['id'] for slot in slots}
        remaining_slots = all_slot_ids - visited
        for slot_id in remaining_slots:
            ordered_vars_4.extend(slot_to_vars.get(slot_id) or [])


        status, exec_time, solver = resolveWithHeuristic(model, ordered_vars_4, timeout=timeout)

    else : 
        print(f"Wrong heuristic name : {heuristic}, using classic resolution")
        status, exec_time, solver = resolveClassic(model, timeout=timeout)
        

    # Affichage du statut et des temps
    print(f"{gridpath[-15:-5]} | status : {status_name(status)}; Score : {solver.ObjectiveValue():.0f}; Time Exec : {exec_time:.4f}; Time Prep : {prep_time:.4f}")
    # Optionel : Afficher la solution trouvée
    if (status == cp_model.OPTIMAL or status == cp_model.FEASIBLE) and display_grid:
        print("\n--- SOLUTION ---")
        afficher_grille_solution(grid, slots, word_vars, solver)

    return status, exec_time, prep_time, solver