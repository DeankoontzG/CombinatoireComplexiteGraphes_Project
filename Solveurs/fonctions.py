import json
import time
import random
from ortools.sat.python import cp_model
import pandas as pd


def charger_grille(fichier_json):
    """Charge une grille depuis un fichier JSON."""
    with open(fichier_json, 'r') as f:
        data = json.load(f)
        
    # Exemple de ce que tu pourrais vouloir retourner :
    grid = data["grid"]
    slots = data["slots"]
    intersections = data["intersections"]
    dictionary = data["dictionary"]
    
    return grid, slots, intersections, dictionary

def clean_lexique(char, char_replacement):
    # Charger le fichier parquet existant
    df = pd.read_parquet('../../data/lexique_filtre_cleaned.parquet')

    # Vérifier les premières lignes pour comprendre la structure du DataFrame
    print(df.head())

    # Remplacer les '.' par des espaces dans la colonne des mots (supposons que la colonne s'appelle 'mot')
    df['ortho'] = df['ortho'].str.replace(char, char_replacement, regex=False)

    # Sauvegarder le DataFrame nettoyé dans un nouveau fichier parquet
    df.to_parquet('lexique_filtre_cleaned.parquet')

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

def executeORTools(gridpath, display_grid = False, timeout = 60.0, heuristic = None):

    start_prep_time = time.perf_counter()

    model = cp_model.CpModel()

    grid, slots, intersections, dictionary = charger_grille(gridpath)

    # --- 1. Préparation des Données et Variables ---

    # 1.a Préparer le Dictionnaire par Longueur et Valeurs Numériques
    # Structure: { 3: (['mot', 'nom', ...], [[13, 15, 20], [14, 15, 13], ...]) }
    dictionary_data = {}
    for word in dictionary:
        length = len(word)
        word_values = [(ord(letter) - ord('a') + 1) for letter in word] # 'a' -> 1, 'b' -> 2, ...
        
        if length not in dictionary_data:
            dictionary_data[length] = {'words': [], 'values': []}
            
        dictionary_data[length]['words'].append(word)
        dictionary_data[length]['values'].append(word_values)
        
    # Dico pour stocker les variables de lettres (word_vars) et d'index (index_vars)
    word_vars = {}
    index_vars = {} # Nouvelle structure pour stocker les variables d'index

    for slot in slots:
        slot_id = slot['id']
        slot_length = slot['length']
        
        # 1.b Variables de Lettres
        # Une variable pour chaque lettre dans le slot. Domaine de 1 à 26 (a à z).
        letter_vars = [
            model.NewIntVar(1, 26, f"letter_{slot_id}_{i}") 
            for i in range(slot_length)
        ] 
        word_vars[slot_id] = letter_vars 
        
        # 1.c Variable d'Index de Mot
        # L'index du mot choisi parmi les mots de la bonne longueur.
        
        possible_words_data = dictionary_data.get(slot_length)
        if not possible_words_data:
            print(f"ATTENTION: Aucun mot trouvé pour le slot {slot_id} (longueur {slot_length}). Problème infaisable.")
            model.Add(0 == 1) # Rend le modèle impossible
            continue

        num_words = len(possible_words_data['words'])
        word_index_var = model.NewIntVar(0, num_words - 1, f"word_index_{slot_id}") 
        index_vars[slot_id] = word_index_var

    # --- 2. Ajout des Contraintes ---

    # --- 2.a Contrainte de liaison (AddElement) (REMPLACE AddAllowedAssignments) ---
    
    for slot in slots:
        slot_id = slot['id']
        slot_length = slot['length']
        letter_vars = word_vars[slot_id]
        
        # S'il n'y a pas de données, on passe au slot suivant (déjà rendu infaisable plus haut)
        if slot_length not in dictionary_data:
            continue
            
        word_index_var = index_vars[slot_id]
        all_word_values = dictionary_data[slot_length]['values']
        
        # Transposer les valeurs des mots pour obtenir les colonnes (lettre par position)
        # Ex: Mots [['m', 'o', 't'], ['n', 'o', 'm']] -> Transposé [[m, n], [o, o], [t, m]]
        # Pour une grille typique, la longueur est petite (ex: < 15), la transposition est rapide.
        # Si la longueur est très grande, on pourrait optimiser cette partie.
        transposed_values = list(zip(*all_word_values))

        for j in range(slot_length):
            # La liste des j-ièmes lettres de TOUS les mots possibles (colonne j de la matrice)
            jth_letters_values = transposed_values[j]
            
            # Contrainte: letter_vars[j] doit être la valeur à l'index word_index_var dans la liste jth_letters_values
            # i.e., letter_vars[j] = jth_letters_values[word_index_var]
            model.AddElement(word_index_var, jth_letters_values, letter_vars[j])


    # --- 2.b : Contraintes d'intersection (Identique à avant, car elles lient les letter_vars)
    for intersection in intersections:
        s1, p1, s2, p2 = intersection['s1'], intersection['p1'], intersection['s2'], intersection['p2']
        index_p1 = p1 - 1
        index_p2 = p2 - 1
        # Assurez-vous que les slots existent avant d'accéder aux variables
        if s1 in word_vars and s2 in word_vars:
            model.Add(word_vars[s1][index_p1] == word_vars[s2][index_p2])
        # Note: Si un slot manque (problème infaisable), cette contrainte est ignorée.

    # --- 3. Préparation des heuristiques de recherche ---

    slot_to_vars = {slot['id']: word_vars[slot['id']] for slot in slots}

    # Heuristique 1 : on commence par les variables des intersections.
    all_model_vars = [v for slot_id in slot_to_vars for v in slot_to_vars[slot_id]]
    intersection_vars = set()
    for intersection in intersections:
        s1, p1, s2, p2 = intersection['s1'], intersection['p1'], intersection['s2'], intersection['p2']
        intersection_vars.add(word_vars[s1][p1 - 1])
        intersection_vars.add(word_vars[s2][p2 - 1])

    ordered_vars_1 = list(intersection_vars) 

    ## On ajouter les variables restantes une par une
    for var in all_model_vars:
        if var not in intersection_vars:
            ordered_vars_1.append(var)

    # Heuristique 2 : on commence par les mots les plus longs
    slots_by_length = sorted(slots, key=lambda s: s['length'], reverse=True)
    
    ordered_vars_2 = []
    for slot in slots_by_length:
        ordered_vars_2.extend(slot_to_vars[slot['id']]) # On priorise toutes les lettres de ce slot
    
    # Heuristique 3 : mots ayant le plus d'intersections
    intersections_count = {slot['id']: 0 for slot in slots}
    for intersection in intersections:
        intersections_count[intersection['s1']] += 1
        intersections_count[intersection['s2']] += 1

    slots_by_intersections = sorted(slots, key=lambda s: intersections_count[s['id']], reverse=True)

    ordered_vars_3 = []
    for slot in slots_by_intersections:
        ordered_vars_3.extend(slot_to_vars[slot['id']]) # On priorise toutes les lettres de ce slot

    # --- 3. Résolution du Problème ---
    end_prep_time = time.perf_counter()
    prep_time = end_prep_time - start_prep_time

    if heuristic == None:
        status, exec_time, solver = resolveClassic(model, timeout=timeout)
    elif heuristic == "IntersectionsFirst": 
        print("Using Intersections First Heuristic")
        status, exec_time, solver = resolveWithHeuristic(model, ordered_vars_1 ,timeout=timeout)
    elif heuristic == "LongestWordsFirst":
        print("Using Longest Words First Heuristic")
        status, exec_time, solver = resolveWithHeuristic(model, ordered_vars_2 ,timeout=timeout)
    elif heuristic == "MostIntersectionsFirst": 
        print("Using Most Intersections Words First Heuristic")
        status, exec_time, solver = resolveWithHeuristic(model, ordered_vars_3 ,timeout=timeout)
    else : 
        print(f"Wrong heuristic name : {heuristic}, using classic resolution")
        status, exec_time, solver = resolveClassic(model, timeout=timeout)
        

    # Affichage du statut et des temps
    print(f"{gridpath[-15:-5]} | status : {status_name(status)}; Time Exec : {exec_time:.4f}; Time Prep : {prep_time:.4f}")

    # OPTIONNEL : Afficher la solution trouvée
    if (status == cp_model.OPTIMAL or status == cp_model.FEASIBLE) and display_grid:
        print("\n--- SOLUTION ---")
        for slot in slots:
            slot_id = slot['id']
            slot_length = slot['length']
            
            # S'assurer que le slot a des variables (non rendu infaisable au début)
            if slot_length not in dictionary_data:
                print(f"Slot {slot_id}: Pas de mot possible.")
                continue
                
            # 1. Obtenir l'index du mot choisi
            word_index_var = index_vars[slot_id]
            chosen_index = solver.Value(word_index_var)
            
            # 2. Récupérer le mot à partir de l'index et des données préparées
            possible_words = dictionary_data[slot_length]['words']
            solved_word = possible_words[chosen_index]

            # 3. Vérification (optionnelle) par les variables de lettres
            # solved_word_check = "".join(
            #     chr(solver.Value(letter_var) + ord('a') - 1)
            #     for letter_var in word_vars[slot_id]
            # )
            
            print(f"Slot {slot_id} (Length {slot_length}): **{solved_word}** (Index: {chosen_index})")

    return status, exec_time, prep_time


def resolveClassic(model, timeout=60.0):
    start_exec_time = time.perf_counter()

    solver = cp_model.CpSolver()
    solver.parameters.num_workers = 4 # Utilisation de 4 threads
    solver.parameters.max_time_in_seconds = timeout # Notre Timeout

    status = solver.Solve(model)

    end_exec_time = time.perf_counter()
    exec_time = end_exec_time - start_exec_time

    return(status, exec_time, solver)

def resolveWithHeuristic(model, ordered_vars, timeout=60.0):

    start_exec_time = time.perf_counter()

    solver = cp_model.CpSolver()
    solver.parameters.num_workers = 4 # Utilisation de 4 threads
    solver.parameters.max_time_in_seconds = timeout # Notre Timeout

    # On fixe aléatoirement une lettre à toutes les vairables odronnées.
    # Si ça viole une contrainte, le solveur tentera d'autres lettres pour cette variable.
    hint_values = []
    for _ in ordered_vars:
        rdm_lettre = random.randint(1, 26)
        hint_values.append(rdm_lettre)
 
    model.AddHint(ordered_vars, hint_values)
    solver.parameters.search_branching = cp_model.FIXED_SEARCH #Pour le forcer à suivre notre heuristique

    status = solver.Solve(model)

    end_exec_time = time.perf_counter()
    exec_time = end_exec_time - start_exec_time

    return(status, exec_time, solver)
