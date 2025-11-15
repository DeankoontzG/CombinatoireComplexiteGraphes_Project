import json
import time
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

def executeORTools(gridpath,  display_grid = False, timeout = 60.0):

    start_prep_time = time.perf_counter()

    model = cp_model.CpModel()

    grid, slots, intersections, dictionary = charger_grille(gridpath)

    # Dico pour stocker les variables de lettres pour chaque slot
    word_vars = {}
    # Doci pour stocker les variables booléennes (mot choisi)
    bool_vars = {}

    for slot in slots:
        slot_id = slot['id']
        slot_length = slot['length']
        
        # Une variable pour chaque lettre dans le slot. On définit le domaine, aka les valeurs que peuvent prendre ces variables.
        letter_vars = [model.NewIntVar(1, 26, f"letter_{slot_id}_{i}") for i in range(slot_length)] 
        word_vars[slot_id] = letter_vars 

        bool_vars[slot_id] = [] 
        
        # On créé une variable booléenne pour chaque mot du dico de la bonne longueur
        for word in dictionary:
            if len(word) == slot_length:
                bool_var = model.NewBoolVar(f"word_selected_{slot_id}_{word}")
                bool_vars[slot_id].append((word, bool_var))

    # --- 2. Ajout des Contraintes ---

    # --- 2.a Les mots choisis doivent exister dans le Dico (AddAllowedAssignments) ---
    
    for slot in slots:
        slot_id = slot['id']
        slot_length = slot['length']
        letter_vars = word_vars[slot_id]
    
        allowed_tuples = []
        
        for word in dictionary:
            if len(word) == slot_length:
                word_values = [(ord(letter) - ord('a') + 1) for letter in word]
                allowed_tuples.append(word_values)
        
        if not allowed_tuples:
            # Sécurité : S'il n'y a aucun mot de la bonne longueur, le problème est infaisable.
            print(f"ATTENTION: Aucun mot trouvé pour le slot {slot_id}. Problème infaisable.")
            model.Add(0 == 1) 
            continue

        model.AddAllowedAssignments(letter_vars, allowed_tuples)


    # --- 2.b : Contraintes d'intersection (les lettres doivent être identiques)
    for intersection in intersections:
        s1, p1, s2, p2 = intersection['s1'], intersection['p1'], intersection['s2'], intersection['p2']
        index_p1 = p1 - 1
        index_p2 = p2 - 1
        model.Add(word_vars[s1][index_p1] == word_vars[s2][index_p2])

    # --- 3. Résolution du Problème ---
    end_prep_time = time.perf_counter()
    start_exec_time = time.perf_counter()

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = timeout # Notre Timeout

    status = solver.Solve(model)

    end_exec_time = time.perf_counter()
    exec_time = end_exec_time - start_exec_time
    prep_time = end_prep_time - start_prep_time

    # Affichage du statut et des temps
    print(f"{gridpath[-15:-5]} | status : {status_name(status)}; Time Exec : {exec_time}; Time Prep : {prep_time}")

    # OPTIONNEL : Afficher la solution trouvée
    if (status == cp_model.OPTIMAL or status == cp_model.FEASIBLE) and display_grid:
        print("\n--- SOLUTION ---")
        for slot in slots:
            slot_id = slot['id']
            # Trouver le mot qui a été choisi
            for word, bool_var in bool_vars[slot_id]:
                if solver.Value(bool_var):
                    # print(f"Slot {slot_id}: {word}")
                    
                    # On peut aussi reconstruire le mot à partir des variables de lettres (vérification)
                    solved_word = "".join(
                        chr(solver.Value(letter_var) + ord('a') - 1)
                        for letter_var in word_vars[slot_id]
                    )
                    print(f"Slot {slot_id} (Length {slot['length']}): {solved_word}")
                    break

    return status, exec_time, prep_time


