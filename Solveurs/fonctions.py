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

def executeORTools(gridpath):

    start_prep_time = time.perf_counter()

    def word_to_int(word):
        """Convertit un mot en deux entiers en le découpant en deux parts égales."""
        mid = len(word) // 2  # Position du milieu du mot
        part1 = word[:mid]  # Première moitié du mot
        part2 = word[mid:]  # Deuxième moitié du mot

        # Convertir chaque moitié en un entier, en concaténant les valeurs des lettres
        part1_int = 0
        for letter in part1:
            part1_int = part1_int * 100 + (ord(letter) - ord('a') + 1)  # Multiplie par 100 pour ajouter la nouvelle valeur

        part2_int = 0
        for letter in part2:
            part2_int = part2_int * 100 + (ord(letter) - ord('a') + 1)  # Multiplie par 100 pour ajouter la nouvelle valeur

        return part1_int, part2_int

    model = cp_model.CpModel()

    grid, slots, intersections, dictionary = charger_grille(gridpath)

    # Convertir les mots du dictionnaire en deux entiers par découpe
    word_to_int_dict = {word: word_to_int(word) for word in dictionary}

    # Créer des variables pour chaque moitié de chaque mot dans chaque slot
    word_vars = {}
    bool_vars = {}

    for slot in slots:
        word_vars[slot['id']] = []  # Liste des variables pour ce slot
        bool_vars[slot['id']] = []  # Liste des booléens pour chaque mot dans ce slot

        # Découpe du slot en 2 parties
        mid = slot['length'] // 2
        part1_length = mid
        part2_length = slot['length'] - part1_length

        # Créer des variables pour chaque lettre dans la partie 1
        part1_vars = [model.NewIntVar(1, 26, f"part1_{slot['id']}_{i}") for i in range(part1_length)]
        part2_vars = [model.NewIntVar(1, 26, f"part2_{slot['id']}_{i}") for i in range(part2_length)]

        word_vars[slot['id']] = [part1_vars, part2_vars]


        # Créer des variables booléennes pour chaque mot dans le dictionnaire
        for word, (part1_values, part2_values) in word_to_int_dict.items():
            if len(word) == slot['length']:  # Si la longueur du mot correspond à celle du slot
                bool_var = model.NewBoolVar(f"word_{slot['id']}_{word}")
                bool_vars[slot['id']].append((word, bool_var))  # Ajouter un tuple (mot, bool_var)

    # Ajouter des contraintes pour chaque slot
    for slot in slots:
        word_length = slot['length']
        part1_vars, part2_vars = word_vars[slot['id']]  # Séparer les deux parties
        bools = bool_vars[slot['id']]

        # Ajouter la contrainte qu'il faut exactement une variable booléenne vraie pour chaque slot
        model.Add(sum(bools[i][1] for i in range(len(bools))) == 1)

        # Pour chaque mot du dictionnaire, découper le mot et ajouter les contraintes d'égalité
        for word, (part1_values, part2_values) in word_to_int_dict.items():
            if len(word) == word_length:
                
                model.Add(part1_varsf"part1_{slot['id']}_{i}") == part1_values).OnlyEnforceIf(bool_vars[slot['id']][[w for w, bv in bools].index(word)][1])
                model.Add(part1_vars == part2_values).OnlyEnforceIf(bool_vars[slot['id']][[w for w, bv in bools].index(word)][1])

    # Ajouter les contraintes d'intersection
    for intersection in intersections:
        s1, p1, s2, p2 = intersection['s1'], intersection['p1'], intersection['s2'], intersection['p2']
        model.Add(word_vars[s1][p1 - 1] == word_vars[s2][p2 - 1])

    # Résoudre le problème
    end_prep_time = time.perf_counter()
    start_exec_time = time.perf_counter()

    solver = cp_model.CpSolver()

    # Durée d'exécution
    end_exec_time = time.perf_counter()
    exec_time = end_exec_time - start_exec_time
    prep_time = end_prep_time - start_prep_time

    status = solver.Solve(model)
    print(f"{gridpath[-15:-5]} | status : {status_name(status)}; Time Exec : {exec_time}; Time Prep : {prep_time}")

    return status, exec_time, prep_time

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
