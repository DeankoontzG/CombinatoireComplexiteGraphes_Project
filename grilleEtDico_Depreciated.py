import random
import pandas as pd

# -------------------------
# 1. Gestion du dictionnaire
# -------------------------
def load_dictionary_parquet(path, min_len=1, max_len=15):
    """
    Charge un dictionnaire depuis un fichier Parquet contenant une colonne 'ortho'.
    Filtre les mots selon leur longueur.
    """
    # Lire le fichier Parquet
    df = pd.read_parquet(path)
    
    # Extraire la colonne des mots et mettre en minuscules
    words = df['ortho'].str.lower().tolist()
    
    # Filtrer par longueur
    return [w for w in words if min_len <= len(w) <= max_len]


# -------------------------
# 2. Définition de grilles
# -------------------------
def grid_5x5():
    """
    Exemple simple de grille 5x5 :
    - '#' = case bloquée
    - '.' = case à remplir
    """
    return [
        list("....."),
        list("..#.."),
        list("....."),
        list("..#.."),
        list("....."),
    ]

def grid_7x7():
    return [
        list("......."),
        list("..#.#.."),
        list("......."),
        list("#.....#"),
        list("......."),
        list("..#.#.."),
        list("......."),
    ]

def grid_crossword_10x10():
    return [
        list("...#.....#"),
        list("..#...##.."),
        list(".....#...."),
        list("#....#...."),
        list("..###....."),
        list("....#....#"),
        list("..#......."),
        list("....#..#.."),
        list("#.....##.."),
        list("...#......"),
    ]

def grid_crossword_15x10():
    return [
        list(".....#.....#...."),
        list("..#......#......"),
        list("......#........."),
        list("#....###........"),
        list("........#......."),
        list("..#.......#....."),
        list("......##........"),
        list("....#......#...."),
        list("#.........#....."),
        list(".......#......#."),
    ]

def grid_crossword_20x10():
    return [
        list("....#.......#......."),  # mots courts et moyens
        list("..#.....###.......#"),
        list(".........#.........."),
        list("##........#....#...."),
        list(".....###..........."),
        list("......#.......#...."),
        list("...#..............#"),
        list("..........##......."),
        list("#.......#........#."),
        list(".......#.......##.."),
    ]

# -------------------------
# 3. Extraction des mots (slots)
# -------------------------
def extract_slots(grid):
    """
    Extrait les "emplacements" (horizontaux + verticaux) à remplir dans la grille.
    Retourne une liste de (coords, longueur, orientation).
    """
    slots = []

    # Horizontal
    for i, row in enumerate(grid):
        j = 0
        while j < len(row):
            if row[j] == '.':
                start = j
                while j < len(row) and row[j] == '.':
                    j += 1
                length = j - start
                if length > 1:  # au moins 2 lettres
                    coords = [(i, k) for k in range(start, j)]
                    slots.append(("H", coords))
            else:
                j += 1

    # Vertical
    for j in range(len(grid[0])):
        i = 0
        while i < len(grid):
            if grid[i][j] == '.':
                start = i
                while i < len(grid) and grid[i][j] == '.':
                    i += 1
                length = i - start
                if length > 1:
                    coords = [(k, j) for k in range(start, i)]
                    slots.append(("V", coords))
            else:
                i += 1

    return slots


# -------------------------
# 4. Exemple d’utilisation
# -------------------------
if __name__ == "__main__":
    # Charger un dictionnaire
    dico = load_dictionary("mots.txt", min_len=2, max_len=10)
    print(f"{len(dico)} mots chargés")

    # Charger une grille
    grid = grid_5x5()

    # Extraire les slots
    slots = extract_slots(grid)
    print("Slots trouvés :")
    for s in slots:
        print(s)
