import pandas as pd
import numpy as np
import json
import os
import random
from pathlib import Path
from collections import Counter
import xml.etree.ElementTree as ET

# ---------------------
# PARAMÈTRES GLOBAUX
# ---------------------
LEXIQUE_PATH = "data/lexique_filtre.parquet"
OUTPUT_DIR = Path("instances")

# Nombre d’instances à générer par catégorie
N_INSTANCES = {
    "small": 20,
    "medium": 20,
    "large": 20
}

# ---------------------
# 1. Chargement du lexique
# ---------------------
lexique = pd.read_parquet(LEXIQUE_PATH)
lexique = lexique.dropna(subset=["nblettres"])
lexique["nblettres"] = lexique["nblettres"].astype(int)

# ---------------------
# 2. Génération d’un dictionnaire filtré
# ---------------------
def get_dictionary(category):
    """Retourne un dictionnaire filtré selon la catégorie."""
    if category == "small":
        max_words = 500
        filtered = lexique[lexique["nblettres"] <= 5]
    elif category == "medium":
        max_words = 50000
        filtered = lexique[lexique["nblettres"] <= 10]
    else:  # large
        max_words = 100000
        filtered = lexique[lexique["nblettres"] <= 12]

    words = filtered["ortho"].dropna().drop_duplicates().sample(
        n=min(max_words, len(filtered)), random_state=42
    ).tolist()

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

def expected_n_slots(grid_size, mu=7, sigma=2, fill_factor=0.8):
    """
    Estime le nombre moyen de slots pour une grille de taille grid_size.
    """
    avg_len = mu 
    total_slots_h = (grid_size * grid_size) / avg_len * fill_factor
    total_slots_v = (grid_size * grid_size) / avg_len * fill_factor
    total_slots = int(total_slots_h + total_slots_v)

    return total_slots


def generate_empty_grid(grid_size, mu=7, sigma=2, min_len=2, max_len=15):
    """
    Génère une grille simple remplie de slots ('.') et de '#' pour le reste.
    La logique est simplifiée, on ne gère plus de croisements particuliers.
    """
    grid = [["#" for _ in range(grid_size)] for _ in range(grid_size)]
    slots = []
    n_slots = expected_n_slots(grid_size, mu, sigma)

    attempts = 0
    max_attempts = n_slots * 30  # limite pour éviter boucle infinie

    while len(slots) < n_slots and attempts < max_attempts:
        length = sample_slot_length(mu, sigma, min_len, max_len)
        length = min(length, grid_size)

        orientation = random.choice(["H", "V"])

        # Calculer le maximum de départ pour réserver une case pour le hashtag
        if orientation == "H":
            max_col = grid_size - length
            if max_col < grid_size - 1:
                max_col -= 1
            if max_col < 0:
                attempts += 1
                continue
            row = random.randint(0, grid_size - 1)
            col = random.randint(0, max_col)
        else:
            max_row = grid_size - length
            if max_row < grid_size - 1:
                max_row -= 1
            if max_row < 0:
                attempts += 1
                continue
            row = random.randint(0, max_row)
            col = random.randint(0, grid_size - 1)

        overlap_ok = True

        # Vérification H–H / V–V et croisements H–V autorisés
        for i in range(length):
            r = row + i if orientation == "V" else row
            c = col + i if orientation == "H" else col

            if grid[r][c] == ".":  # case déjà occupée
                for slot in slots:
                    sr, sc, sl, so = slot["row"], slot["col"], slot["length"], slot["orientation"]
                    if so == orientation:
                        if orientation == "H":
                            if sr == r and sc <= c < sc + sl:
                                overlap_ok = False
                                break
                        else:
                            if sc == c and sr <= r < sr + sl:
                                overlap_ok = False
                                break
            if not overlap_ok:
                break

        # Vérification de la case avant et après pour le hashtag
        if overlap_ok:
            if orientation == "H":
                if col > 0 and grid[row][col - 1] not in ("#", "$"):
                    overlap_ok = False
                if col + length < grid_size and grid[row][col + length] not in ("#", "$"):
                    overlap_ok = False
            else:
                if row > 0 and grid[row - 1][col] not in ("#", "$"):
                    overlap_ok = False
                if row + length < grid_size and grid[row + length][col] not in ("#", "$"):
                    overlap_ok = False

        if overlap_ok:
            # placer le slot
            for i in range(length):
                r = row + i if orientation == "V" else row
                c = col + i if orientation == "H" else col
                grid[r][c] = "."

            # placer hashtags protégés
            if orientation == "H":
                if col > 0 and grid[row][col - 1] == "#":
                    grid[row][col - 1] = "$"
                if col + length < grid_size and grid[row][col + length] == "#":
                    grid[row][col + length] = "$"
            else:
                if row > 0 and grid[row - 1][col] == "#":
                    grid[row - 1][col] = "$"
                if row + length < grid_size and grid[row + length][col] == "#":
                    grid[row + length][col] = "$"

            slots.append({"orientation": orientation, "row": row, "col": col, "length": length})

        attempts += 1

    # Transformation finale : tous les $ deviennent #
    for r in range(grid_size):
        for c in range(grid_size):
            if grid[r][c] == "$":
                grid[r][c] = "#"

    return grid


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
def generate_instance(category, grid_size):
    dictionary = get_dictionary(category)

    if category == "small" : 
        grid = generate_empty_grid(grid_size, mu=4, sigma = 1, max_len = 5)
    elif category == "medium" : 
        grid = generate_empty_grid(grid_size, mu=6, sigma = 1, max_len = 10)
    else :
        grid = generate_empty_grid(grid_size, mu=7, sigma = 2, max_len = 15) 
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
def generate_benchmark():
    OUTPUT_DIR.mkdir(exist_ok=True)

    configs = {
        "small": {"size": 5},
        "medium": {"size": 10},
        "large": {"size": 15},
    }

    all_lengths = {cat: [] for cat in configs}

    for category in ["small", "medium", "large"]:
        out_path = OUTPUT_DIR / category
        out_path.mkdir(exist_ok=True)
        print(f"→ Génération {category}...")

        for i in range(N_INSTANCES[category]):
            cfg = configs[category]
            instance = generate_instance(category, cfg["size"])

            # Json
            filename = out_path / f"{category}_{i+1:03d}.json"
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(instance, f, ensure_ascii=False, indent=2)

            # MiniZinc
            filename = out_path / f"{category}_{i+1:03d}.dzn"
            generate_dzn(instance, filename)

            slots = find_word_slots(instance["grid"])
            lengths = [s["length"] for s in slots]
            all_lengths[category].extend(lengths)

        print(f"  ✓ {N_INSTANCES[category]} instances sauvegardées dans {out_path}/")

    
    print("Répartition moyenne des tailles par catégorie :\n")
    for category, lengths in all_lengths.items():
        if not lengths:
            print(f"[{category}] Pas de slots détectés.")
            continue
        total = len(lengths)
        count_lengths = Counter(lengths)
        print(f"[{category}]")
        for size in sorted(count_lengths):
            pct = count_lengths[size] / total * 100
            print(f"  Taille {size} : {count_lengths[size]} slots ({pct:.1f}%)")
        print()