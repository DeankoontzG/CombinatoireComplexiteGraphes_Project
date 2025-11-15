import pandas as pd
import numpy as np
import json
import os
import random
import shutil
from pathlib import Path
from collections import Counter
import xml.etree.ElementTree as ET
from collections import Counter 

# ---------------------
# PARAMÈTRES GLOBAUX
# ---------------------
LEXIQUE_PATH = "data/lexique_filtre_cleaned.parquet"
OUTPUT_DIR = Path("instances")

# Nombre d’instances à générer par catégorie
N_INSTANCES = {
    "small": 10,
    "medium": 10,
    "large": 10
}

# ---------------------
# 1. Chargement du lexique
# ---------------------
lexique = pd.read_parquet(LEXIQUE_PATH)
lexique = lexique.dropna(subset=["longueur"])
lexique["longueur"] = lexique["longueur"].astype(int)

# ---------------------
# 2. Génération d’un dictionnaire filtré
# ---------------------
def get_dictionary(category):
    """Retourne un dictionnaire filtré selon la catégorie."""
    if category == "small":
        max_length = 5
    elif category == "medium":
        max_length = 10
    else:  # category == "large" ou autre
        max_length = 12
        
    filtered_df = lexique[lexique["longueur"] <= max_length].copy()
    
    words = filtered_df["mot"].dropna().drop_duplicates().tolist()

    return words

# ---------------------
# 3. Génération d’une grille vide aléatoire
# ---------------------
def sample_slot_length(mu=7, sigma=2, min_len=2, max_len=15):
    """
    Tire une longueur de slot selon une loi normale tronquée.
    """
    attempts = 0
    while True:
        length = int(np.random.normal(mu, sigma))
        if min_len <= length <= max_len:
            return length
        attempts += 1
        if attempts > 100:  # sécurité pour éviter boucle infinie
            return min_len

def generate_empty_grid(grid_size, mu=7, sigma=2, min_len=2, max_len=15, fill_factor=0.8):
    """
    Génère une grille remplie de slots ('.') et de '#' pour le reste,
    en ciblant un pourcentage de remplissage via une stratégie de croisement.
    """
    grid = [["#" for _ in range(grid_size)] for _ in range(grid_size)]
    slots = []

    # Cible : Nombre de cellules à remplir
    target_cells = int(grid_size * grid_size * fill_factor)
    current_cells = 0

    attempts = 0
    max_attempts = grid_size * grid_size * 5  # Limite de sécurité

    # Compteurs d'échecs pour basculer en mode de placement libre
    consecutive_cross_failures = 0
    MAX_CONSECUTIVE_FAILURES = 100 

    while current_cells < target_cells and attempts < max_attempts:
        
        placement_success = False
        
        # --- Stratégie 1 : Tenter le croisement (Priorité) ---
        if len(slots) > 0 and consecutive_cross_failures < MAX_CONSECUTIVE_FAILURES:
            slot_to_cross = random.choice(slots)
            placement_success, new_slot, filled_cells = attempt_cross_placement(
                grid, grid_size, slot_to_cross, mu, sigma, min_len, max_len
            )
            
            if placement_success:
                consecutive_cross_failures = 0
            else:
                consecutive_cross_failures += 1
                
        # --- Stratégie 2 : Tenter le placement libre (Si échecs croisement ou grille vide) ---
        if not placement_success:
            # Correction ici : Passer TOUS les paramètres nécessaires
            placement_success, new_slot, filled_cells = attempt_free_placement(
                grid, grid_size, mu, sigma, min_len, max_len
            )
        
        # --- Mise à jour de la grille et des slots ---
        if placement_success:
            slots.append(new_slot)
            current_cells += filled_cells
            attempts = 0 # Réinitialise les tentatives si placement réussi
        
        attempts += 1 # Incrémente toujours attempts si la boucle tourne

    # Finalisation : Assurez-vous qu'il n'y a plus de marques temporaires
    # (Votre logique de grille ne génère pas de marques temporaires ici, mais bonne pratique de l'enlever)
    return grid

# --- Fonctions auxiliaires à ajouter ---

def is_valid_segment(grid, row, col, length, orientation, grid_size):
    """Vérifie si un segment est valide pour un nouveau slot (pas de chevauchement H-H ou V-V)."""
    
    # Vérification des limites
    if (orientation == "H" and col + length > grid_size) or \
       (orientation == "V" and row + length > grid_size):
        return False

    for i in range(length):
        r = row + i if orientation == "V" else row
        c = col + i if orientation == "H" else col
        
        # Si la case est déjà un '.' et que l'orientation est la même, c'est un chevauchement non autorisé.
        # Sinon, la case doit être un '#'. (Le croisement H-V est géré implicitement par le placement)
        if grid[r][c] == '.':
            # Cette vérification est simplifiée par rapport à votre logique initiale,
            # mais elle est souvent suffisante dans une stratégie de croisement.
            return False 
            
    # Vérification des délimiteurs (doit être bord de grille ou '#')
    if orientation == "H":
        if (col > 0 and grid[row][col - 1] == '.') or \
           (col + length < grid_size and grid[row][col + length] == '.'):
            return False
    else:
        if (row > 0 and grid[row - 1][col] == '.') or \
           (row + length < grid_size and grid[row + length][col] == '.'):
            return False
            
    return True


# --- Fonctions auxiliaires corrigées (elles doivent recevoir les paramètres mu, sigma, min_len, max_len) ---

def attempt_free_placement(grid, grid_size, mu, sigma, min_len, max_len):
    """Tente de placer un slot aléatoirement dans un espace libre (méthode initiale)."""
    
    orientation = random.choice(["H", "V"])
    length = sample_slot_length(mu, sigma, min_len, max_len) # Tirage au sort de la longueur

    if orientation == "H":
        max_col = grid_size - length
        if max_col < 0: return False, None, 0

        for _ in range(grid_size * grid_size):
            row = random.randint(0, grid_size - 1)
            col = random.randint(0, max_col)
            
            if is_valid_segment(grid, row, col, length, orientation, grid_size):
                for c_idx in range(col, col + length):
                    grid[row][c_idx] = "."
                return True, {"orientation": orientation, "row": row, "col": col, "length": length}, length
        
    else: # Orientation V
        max_row = grid_size - length
        if max_row < 0: return False, None, 0
        
        # ... (le reste de la logique V) ...
        for _ in range(grid_size * grid_size):
            row = random.randint(0, max_row)
            col = random.randint(0, grid_size - 1)
            
            if is_valid_segment(grid, row, col, length, orientation, grid_size):
                for r_idx in range(row, row + length):
                    grid[r_idx][col] = "."
                return True, {"orientation": orientation, "row": row, "col": col, "length": length}, length

    return False, None, 0


def attempt_cross_placement(grid, grid_size, existing_slot, mu, sigma, min_len, max_len):
    """Tente de placer un slot perpendiculaire à un slot existant."""
    
    s = existing_slot
    new_orientation = "V" if s["orientation"] == "H" else "H"
    
    # 1. Identifier les points de croisement potentiels
    potential_cross_points = []
    for p_exist in range(s["length"]):
        r_cross = s["row"] + (p_exist if s["orientation"] == "V" else 0)
        c_cross = s["col"] + (p_exist if s["orientation"] == "H" else 0)
        
        # Un point de croisement doit être un '.'
        if grid[r_cross][c_cross] == '.':
            
            # Vérifier si la position perpendiculaire est actuellement un '#' (vide)
            # Vérification très rapide : si la case est déjà croisée, on l'ignore.
            is_already_crossed = False
            if new_orientation == "H": # Croisement V sur un slot H
                 # Regarder si (r_cross, c_cross-1) ou (r_cross, c_cross+1) sont des '.'
                 if (c_cross > 0 and grid[r_cross][c_cross-1] == '.') or \
                    (c_cross < grid_size - 1 and grid[r_cross][c_cross+1] == '.'):
                    is_already_crossed = True
            else: # Croisement H sur un slot V
                 # Regarder si (r_cross-1, c_cross) ou (r_cross+1, c_cross) sont des '.'
                 if (r_cross > 0 and grid[r_cross-1][c_cross] == '.') or \
                    (r_cross < grid_size - 1 and grid[r_cross+1][c_cross] == '.'):
                    is_already_crossed = True
                    
            if not is_already_crossed:
                potential_cross_points.append((r_cross, c_cross))
    
    if not potential_cross_points:
        return False, None, 0

    # 2. Tenter de placer un slot à partir d'un point de croisement
    random.shuffle(potential_cross_points)
    
    for r_cross, c_cross in potential_cross_points:
        # Longueur aléatoire pour le nouveau slot
        length = sample_slot_length(mu, sigma, min_len, max_len)
        length = min(length, grid_size)
        
        # Calculer les positions de départ possibles (doit contenir le point de croisement)
        
        if new_orientation == "H":
            start_col_min = max(0, c_cross - length + 1)
            start_col_max = min(grid_size - length, c_cross)
            
            for start_col in range(start_col_min, start_col_max + 1):
                # Vérifier la validité du segment (r_cross, start_col)
                if is_valid_segment_for_cross(grid, r_cross, start_col, length, new_orientation, grid_size):
                    # Placement du slot
                    for c_idx in range(start_col, start_col + length):
                         # On ne compte pas la cellule déjà remplie par le slot existant
                         if grid[r_cross][c_idx] == '#':
                             grid[r_cross][c_idx] = "."
                             
                    # Déterminer les cellules effectivement remplies (celles qui étaient des '#')
                    filled_cells = length - 1 # Le point de croisement était déjà un '.'
                    return True, {"orientation": new_orientation, "row": r_cross, "col": start_col, "length": length}, filled_cells
        else: # Orientation V
            start_row_min = max(0, r_cross - length + 1)
            start_row_max = min(grid_size - length, r_cross)

            for start_row in range(start_row_min, start_row_max + 1):
                if is_valid_segment_for_cross(grid, start_row, c_cross, length, new_orientation, grid_size):
                    # Placement du slot
                    for r_idx in range(start_row, start_row + length):
                         if grid[r_idx][c_cross] == '#':
                             grid[r_idx][c_cross] = "."
                             
                    filled_cells = length - 1 
                    return True, {"orientation": new_orientation, "row": start_row, "col": c_cross, "length": length}, filled_cells

    return False, None, 0


def is_valid_segment_for_cross(grid, row, col, length, orientation, grid_size):
    """
    Vérifie la validité d'un segment destiné à croiser un slot existant.
    Autorise le segment à passer par UN SEUL '.' (le point de croisement).
    """
    
    # Vérification des limites
    if (orientation == "H" and col + length > grid_size) or \
       (orientation == "V" and row + length > grid_size):
        return False
        
    dot_count = 0
    
    for i in range(length):
        r = row + i if orientation == "V" else row
        c = col + i if orientation == "H" else col
        
        # Si la case est occupée par un slot non perpendiculaire, ou si elle est hors grille
        if r >= grid_size or c >= grid_size: 
            return False

        if grid[r][c] == '.':
            dot_count += 1
            # Si le segment chevauche plus d'un slot existant, il n'est pas valide
            if dot_count > 1:
                return False 
        
    # Le segment doit croiser exactement 1 slot existant pour être un croisement valide
    if dot_count != 1:
        return False 
        
    # Vérification des délimiteurs (# ou bord de grille)
    if orientation == "H":
        if (col > 0 and grid[row][col - 1] == '.') or \
           (col + length < grid_size and grid[row][col + length] == '.'):
            return False
    else:
        if (row > 0 and grid[row - 1][col] == '.') or \
           (row + length < grid_size and grid[row + length][col] == '.'):
            return False
            
    return True


def split_long_words(grid, max_len=13):
    """
    Parcourt la grille et coupe les mots trop longs en insérant un '#' au milieu.
    """
    size = len(grid)

    # Horizontaux
    for i in range(size):
        j = 0
        while j < size:
            if grid[i][j] != "#":
                start = j
                while j < size and grid[i][j] != "#":
                    j += 1
                length = j - start
                if length > max_len:
                    # On coupe au milieu
                    cut_pos = start + length // 2
                    grid[i][cut_pos] = "#"
            else:
                j += 1

    # Verticaux
    for j in range(size):
        i = 0
        while i < size:
            if grid[i][j] != "#":
                start = i
                while i < size and grid[i][j] != "#":
                    i += 1
                length = i - start
                if length > max_len:
                    # On coupe au milieu
                    cut_pos = start + length // 2
                    grid[cut_pos][j] = "#"
            else:
                i += 1

    return grid

# ---------------------
# 4. Détermination des slots
# ---------------------

def find_word_slots(grid):
    """
    Détecte tous les slots horizontaux et verticaux (>=2 lettres).
    Retourne une liste de dictionnaires avec orientation, position et longueur.
    """
    slots = []
    size = len(grid)
    slot_id = 1  # pour MiniZinc (1-based)

    # Horizontaux
    for i in range(size):
        j = 0
        while j < size:
            if grid[i][j] != "#":
                start = j
                while j < size and grid[i][j] != "#":
                    j += 1
                length = j - start
                if length >= 2:
                    slots.append({
                        "id": slot_id,
                        "orientation": "H",
                        "row": i,
                        "col": start,
                        "length": length
                    })
                    slot_id += 1
            else:
                j += 1

    # Verticaux
    for j in range(size):
        i = 0
        while i < size:
            if grid[i][j] != "#":
                start = i
                while i < size and grid[i][j] != "#":
                    i += 1
                length = i - start
                if length >= 2:
                    slots.append({
                        "id": slot_id,
                        "orientation": "V",
                        "row": start,
                        "col": j,
                        "length": length
                    })
                    slot_id += 1
            else:
                i += 1

    return slots

# ---------------------
# 5. Calcul des intersections
# ---------------------

def compute_intersections(slots):
    intersections = []
    for s1 in slots:
        for s2 in slots:
            if s1["id"] < s2["id"]:  # éviter doublons
                if s1["orientation"] != s2["orientation"]:
                    for p1 in range(s1["length"]):
                        for p2 in range(s2["length"]):
                            r1 = s1["row"] + (p1 if s1["orientation"] == "V" else 0)
                            c1 = s1["col"] + (p1 if s1["orientation"] == "H" else 0)
                            r2 = s2["row"] + (p2 if s2["orientation"] == "V" else 0)
                            c2 = s2["col"] + (p2 if s2["orientation"] == "H" else 0)
                            if r1 == r2 and c1 == c2:
                                intersections.append({
                                    "s1": s1["id"],
                                    "p1": p1 + 1,  # MiniZinc → 1-based
                                    "s2": s2["id"],
                                    "p2": p2 + 1
                                })
    return intersections

# ---------------------
# 6. Génération d’une instance complète
# ---------------------

def generate_instance(category, grid_size, fill_factor=0.8):
    dictionary = get_dictionary(category)

    if category == "small" :
        grid = generate_empty_grid(grid_size, mu=4, sigma=1, max_len=5, fill_factor=fill_factor)
    elif category == "medium" :
        grid = generate_empty_grid(grid_size, mu=6, sigma=1, max_len=10, fill_factor=fill_factor)
    else :
        grid = generate_empty_grid(grid_size, mu=7, sigma=2, max_len=15, fill_factor=fill_factor)
        grid = split_long_words(grid, max_len=12)
        
    slots = find_word_slots(grid)
    intersections = compute_intersections(slots)
    
    instance = {
        "category": category,
        "grid_size": grid_size,
        "grid": grid,
        "slots": slots,
        "intersections": intersections,
        "dictionary": dictionary
    }
    return instance

# ---------------------
# 7. Mise en forme des données (MiniZinc)
# ---------------------

def generate_dzn(instance, dzn_path):
    """
    Génère un fichier .dzn à partir d'une instance de grille de mots fléchés.
    """
    slots = instance["slots"]
    intersections = instance["intersections"]
    dictionary = instance["dictionary"]

    # --- Extraction des infos principales ---
    n_slots = len(slots)
    lengths = [slot["length"] for slot in slots]

    n_intersections = len(intersections)
    slot1 = [i["s1"] for i in intersections]
    pos1 = [i["p1"] for i in intersections]
    slot2 = [i["s2"] for i in intersections]
    pos2 = [i["p2"] for i in intersections]

    n_dict = len(dictionary)

    # --- Construction du contenu .dzn ---
    lines = [
        f"N_SLOTS = {n_slots};",
        f"LENGTHS = {lengths};",
        f"N_INTERSECTIONS = {n_intersections};",
        f"SLOT1 = {slot1};",
        f"POS1 = {pos1};",
        f"SLOT2 = {slot2};",
        f"POS2 = {pos2};",
        f"N_DICT = {n_dict};",
        'DICT = [' + ', '.join(f'"{w}"' for w in dictionary) + '];'
    ]

    # --- Écriture dans le fichier .dzn ---
    with open(dzn_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))



# ---------------------
# 8. Génération et sauvegarde du benchmark
# ---------------------
def clean_category_directory(category_path: Path):
    """Supprime récursivement un répertoire s'il existe et le recrée."""
    
    if category_path.exists():
        print(f"  [Nettoyage] Suppression du contenu de {category_path}...")
        
        # Supprime récursivement le dossier et tout son contenu
        shutil.rmtree(category_path)
    
    # Recrée le dossier vide
    category_path.mkdir(exist_ok=True)
    print(f"  [Nettoyage] Dossier {category_path.name} prêt.")


def generate_benchmark():
    OUTPUT_DIR.mkdir(exist_ok=True)
    

    configs = {
        "small": {"size": 5},
        "medium": {"size": 10},
        "large": {"size": 15},
    }

    all_lengths = {cat: {} for cat in configs}
    num_ff = 5

    for category in ["small", "medium", "large"]:
        category_path = OUTPUT_DIR / category
        category_path.mkdir(exist_ok=True)
        clean_category_directory(category_path)
        print(f"→ Génération de la catégorie '{category}'...")

        cfg = configs[category]
        num_instances = N_INSTANCES[category]
        total_saved_instances = 0

        for fill_factor in [0.8, 0.85, 0.90, 0.95, 1.0]:
            fill_factor_str = f"{fill_factor:.2f}"
                
            sub_folder_name = f"fillfactor_{fill_factor_str}"
            out_path = category_path / sub_folder_name 
            out_path.mkdir(exist_ok=True)
                
            print(f"  > Fill Factor: {fill_factor_str} ({num_instances} instances)")
            all_lengths[category][fill_factor] = []

            for i in range(N_INSTANCES[category]):
                instance = generate_instance(category, cfg["size"], fill_factor=fill_factor)
                
                # Sauvegarde au format Json
                filename_json = out_path / f"{category}_{i+1:03d}.json"
                with open(filename_json, "w", encoding="utf-8") as f:
                    json.dump(instance, f, ensure_ascii=False, indent=2)

                # Sauvegarde au format MiniZinc
                filename_dzn = out_path / f"{category}_{i+1:03d}.dzn"
                generate_dzn(instance, filename_dzn)

                # Collecte des longueurs
                slots = find_word_slots(instance["grid"])
                lengths = [s["length"] for s in slots]
                all_lengths[category][fill_factor].extend(lengths)
                
                total_saved_instances += 1

        print(f"  ✓ {total_saved_instances} instances (pour {num_ff} FFs) sauvegardées sous {category_path}/")


    # ----------------------------------------------------------------------
    # --- AFFICHAGE DES STATISTIQUES (Utilise la nouvelle structure) ---
    # ----------------------------------------------------------------------
    print("\n--- Répartition moyenne des tailles par Catégorie et Fill Factor ---\n")

    for category, fill_factors_data in all_lengths.items():
        print(f"[{category.upper()}]")

        for fill_factor, lengths in sorted(fill_factors_data.items()):
            fill_factor_str = f"{fill_factor:.2f}"
        
            if not lengths:
                print(f"  [FF {fill_factor_str}] Pas de slots détectés.")
                continue
                
            total = len(lengths)
            count_lengths = Counter(lengths)
            
            print(f"  [FF {fill_factor_str}] ({total} slots au total):")
            
            # Afficher la répartition par taille
            for size in sorted(count_lengths):
                pct = count_lengths[size] / total * 100
                print(f"    Taille {size} : {count_lengths[size]} slots ({pct:.1f}%)")
            print("    " + "-" * 30)
        print("\n" + "=" * 50 + "\n")
