#!/usr/bin/env python3
import json
import re
import sys
from pathlib import Path

def parse_int(name, text):
    m = re.search(rf"\b{name}\s*=\s*(\d+)\s*;", text)
    return int(m.group(1)) if m else None

def parse_array_1d(name, text):
    m = re.search(rf"\b{name}\s*=\s*\[(.*?)\]\s*;", text, re.S)
    if not m:
        return None
    raw = m.group(1).replace("\n", " ").split(",")
    return [int(x.strip()) for x in raw if x.strip()]

def parse_array3d_letter(name, text, S, maxC, maxL):
    m = re.search(
        rf"\b{name}\s*=\s*array3d\s*\(.*?\[(.*?)\]\s*\)\s*;",
        text,
        re.S
    )
    if not m:
        raise ValueError(f"Impossible de trouver array3d pour {name} dans le .dzn")
    data = [int(x.strip()) for x in m.group(1).replace("\n", " ").split(",") if x.strip()]
    if len(data) != S * maxC * maxL:
        raise ValueError(f"Taille de {name} inattendue: {len(data)} valeurs pour {S}*{maxC}*{maxL}")
    letter = [[[0] * maxL for _ in range(maxC)] for __ in range(S)]
    idx = 0
    for s in range(S):
        for c in range(maxC):
            for p in range(maxL):
                letter[s][c][p] = data[idx]
                idx += 1
    return letter

def load_dzn(path: Path):
    text = path.read_text()

    S    = parse_int("S", text)
    maxC = parse_int("maxC", text)
    maxL = parse_int("maxL", text)
    if None in (S, maxC, maxL):
        raise ValueError("Impossible de parser S/maxC/maxL dans le .dzn")

    L = parse_array_1d("L", text)
    if L is None or len(L) != S:
        raise ValueError("Impossible de parser L correctement dans le .dzn")

    letter = parse_array3d_letter("letter", text, S, maxC, maxL)

    return {
        "S": S,
        "maxC": maxC,
        "maxL": maxL,
        "L": L,
        "letter": letter,
    }

def find_instance_json(instances_root: Path, dzn_name: str) -> Path:
    json_name = Path(dzn_name).with_suffix(".json").name
    candidates = list(instances_root.rglob(json_name))
    if not candidates:
        raise FileNotFoundError(f"Aucun .json nommé {json_name} trouvé sous {instances_root}")
    return candidates[0]

def build_words_from_dzn_letters(dzn_data, w):
    S      = dzn_data["S"]
    L      = dzn_data["L"]
    letter = dzn_data["letter"]

    assert len(w) == S

    words = []
    for s in range(S):
        k0 = w[s] - 1  # 1-based -> 0-based
        length = L[s]
        chars = []
        for p in range(length):
            code = letter[s][k0][p]
            chars.append(chr(ord("A") + code) if code >= 0 else "_")
        words.append("".join(chars))
    return words

def build_grid_from_json_and_slots(json_data, slots, words, check_conflicts=True):
    """
    Reconstruit la grille en utilisant:
      - json_data["grid_size"] + json_data["grid"] (masque #/.)
      - slots (row/col/orientation/length)
      - words (un mot par slot, déjà reconstruit depuis le DZN)
    Hypothèse: row/col dans slots sont 0-based (cohérent avec ton rendu précédent).
    Si c'est 1-based chez toi, ajoute -1 sur row0/col0.
    """
    grid_size = json_data["grid_size"]
    base = json_data["grid"]  # liste de listes de "#" / "."

    if len(base) != grid_size or any(len(r) != grid_size for r in base):
        raise ValueError("grid_size / grid incohérents dans le JSON")

    # Copie de travail : '#' reste '#', '.' devient '.'
    grid = [row[:] for row in base]

    # Place les mots
    for slot, word in zip(slots, words):
        row0 = slot["row"]
        col0 = slot["col"]
        orient = slot["orientation"]  # 0 horiz, 1 vert
        length = slot["length"]

        for i in range(length):
            r = row0 + (orient == 1) * i
            c = col0 + (orient == 0) * i

            if not (0 <= r < grid_size and 0 <= c < grid_size):
                raise ValueError(f"Slot sort de la grille: (r={r}, c={c})")

            if grid[r][c] == "#":
                # slot qui passe dans une case noire => données incohérentes
                raise ValueError(f"Placement sur case noire (#) en (r={r}, c={c}) pour slot={slot}")

            ch = word[i]

            if grid[r][c] == ".":
                grid[r][c] = ch
            else:
                # déjà une lettre
                if check_conflicts and grid[r][c] != ch:
                    print(f"[WARN] conflit (r={r},c={c}): {grid[r][c]} vs {ch}")

    return grid

def is_vertical(o):
    """Interprète orientation slot: 0/1, 'H'/'V', 'across'/'down', etc."""
    if isinstance(o, bool):
        return bool(o)
    if isinstance(o, int):
        return o == 1
    if isinstance(o, str):
        oo = o.strip().lower()
        return oo in ("v", "vert", "vertical", "down", "d")
    raise ValueError(f"Orientation inconnue: {o!r}")

def detect_offset(slots, grid_size):
    """
    Détecte si row/col sont 0-based ou 1-based.
    - si tout est dans [0, grid_size-1] -> offset=0
    - si tout est dans [1, grid_size]     -> offset=1
    """
    rows = [s["row"] for s in slots]
    cols = [s["col"] for s in slots]
    mn = min(rows + cols)
    mx = max(rows + cols)
    if 0 <= mn and mx <= grid_size - 1:
        return 0
    if 1 <= mn and mx <= grid_size:
        return 1
    # fallback: si présence de 0 on suppose 0-based
    return 0

def build_grid_from_json_and_slots(json_data, slots, words, check_conflicts=True):
    grid_size = json_data["grid_size"]
    base = json_data["grid"]

    # copie de la grille de base
    grid = [row[:] for row in base]

    offset = detect_offset(slots, grid_size)

    for slot, word in zip(slots, words):
        row0 = slot["row"] - offset
        col0 = slot["col"] - offset
        vert = is_vertical(slot["orientation"])
        length = len(word)

        for i in range(length):
            r = row0 + (1 if vert else 0) * i
            c = col0 + (0 if vert else 1) * i

            if not (0 <= r < grid_size and 0 <= c < grid_size):
                raise ValueError(f"Slot sort de la grille: slot={slot}, (r={r}, c={c}), offset={offset}")

            if grid[r][c] == "#":
                raise ValueError(f"Placement sur case noire (#) en (r={r}, c={c}) pour slot={slot}")

            ch = word[i].upper()

            if grid[r][c] == ".":
                grid[r][c] = ch
            else:
                if check_conflicts and grid[r][c] != ch:
                    print(f"[WARN] conflit (r={r},c={c}): {grid[r][c]} vs {ch}")

    return grid


def main():
    root = Path(__file__).resolve().parent

    if len(sys.argv) > 1:
        result_path = Path(sys.argv[1])
    else:
        result_path = root / "results" / "small_002.json"

    metrics = json.loads(result_path.read_text(encoding="utf-8"))

    if metrics.get("solution") is None:
        print("Aucune solution dans ce result.json")
        return

    w = metrics["solution"]["w"]
    instance_dzn = metrics["instance"]

    instances_root = root / ".." / ".." / "instances"

    # retrouver le .dzn
    dzn_path = None
    for p in instances_root.rglob(instance_dzn):
        dzn_path = p
        break
    if dzn_path is None:
        dzn_path = instances_root / instance_dzn

    if not dzn_path.exists():
        raise FileNotFoundError(f"Impossible de trouver le .dzn {instance_dzn} sous {instances_root}")

    instance_json = find_instance_json(instances_root, instance_dzn)

    print(f"Instance JSON utilisée : {instance_json}")
    print(f"Instance DZN utilisée  : {dzn_path}")

    dzn_data = load_dzn(dzn_path)
    words = build_words_from_dzn_letters(dzn_data, w)

    json_data = json.loads(instance_json.read_text(encoding="utf-8"))
    slots = json_data["slots"]

    grid = build_grid_from_json_and_slots(json_data, slots, words, check_conflicts=True)

    print("\n=== GRILLE RECONSTRUITE ===\n")
    for row in grid:
        # Affichage: remplace '.' par espace pour lisibilité
        print(" ".join(" " if ch == "." else ch for ch in row))
    print("\n=== FIN ===")

if __name__ == "__main__":
    main()
