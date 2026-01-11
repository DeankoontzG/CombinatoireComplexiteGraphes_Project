#!/usr/bin/env python3
import json
import re
import sys
from pathlib import Path

# --- ANSI (terminal) ---
RED = "\x1b[31;1m"
RESET = "\x1b[0m"


# -------------------------
# Parsing DZN
# -------------------------
def parse_int(name: str, text: str):
    m = re.search(rf"\b{name}\s*=\s*(\d+)\s*;", text)
    return int(m.group(1)) if m else None


def parse_array_1d(name: str, text: str):
    m = re.search(rf"\b{name}\s*=\s*\[(.*?)\]\s*;", text, re.S)
    if not m:
        return None
    raw = m.group(1).replace("\n", " ").split(",")
    return [int(x.strip()) for x in raw if x.strip()]


def parse_array3d_letter(name: str, text: str, S: int, maxC: int, maxL: int):
    m = re.search(
        rf"\b{name}\s*=\s*array3d\s*\(.*?\[(.*?)\]\s*\)\s*;",
        text,
        re.S,
    )
    if not m:
        raise ValueError(f"Impossible de trouver array3d pour {name} dans le .dzn")

    data = [int(x.strip()) for x in m.group(1).replace("\n", " ").split(",") if x.strip()]
    expected = S * maxC * maxL
    if len(data) != expected:
        raise ValueError(f"Taille de {name} inattendue: {len(data)} valeurs (attendu {expected}={S}*{maxC}*{maxL})")

    letter = [[[0] * maxL for _ in range(maxC)] for __ in range(S)]
    idx = 0
    for s in range(S):
        for c in range(maxC):
            for p in range(maxL):
                letter[s][c][p] = data[idx]
                idx += 1
    return letter


def load_dzn(path: Path):
    text = path.read_text(encoding="utf-8", errors="replace")

    S = parse_int("S", text)
    maxC = parse_int("maxC", text)
    maxL = parse_int("maxL", text)
    if None in (S, maxC, maxL):
        raise ValueError("Impossible de parser S/maxC/maxL dans le .dzn")

    L = parse_array_1d("L", text)
    if L is None or len(L) != S:
        raise ValueError("Impossible de parser L correctement dans le .dzn")

    letter = parse_array3d_letter("letter", text, S, maxC, maxL)

    return {"S": S, "maxC": maxC, "maxL": maxL, "L": L, "letter": letter}


# -------------------------
# Helpers instances
# -------------------------
def find_instance_json(instances_root: Path, dzn_name: str) -> Path:
    json_name = Path(dzn_name).with_suffix(".json").name
    candidates = list(instances_root.rglob(json_name))
    if not candidates:
        raise FileNotFoundError(f"Aucun .json nommé {json_name} trouvé sous {instances_root}")
    return candidates[0]


def find_instance_dzn(instances_root: Path, dzn_name: str) -> Path:
    candidates = list(instances_root.rglob(dzn_name))
    if candidates:
        return candidates[0]
    p = instances_root / dzn_name
    if p.exists():
        return p
    raise FileNotFoundError(f"Impossible de trouver le .dzn {dzn_name} sous {instances_root}")


# -------------------------
# Reconstruction mots/grille
# -------------------------
def build_words_from_dzn_letters(dzn_data, w):
    S = dzn_data["S"]
    L = dzn_data["L"]
    letter = dzn_data["letter"]

    if len(w) != S:
        raise ValueError(f"Taille w incorrecte: len(w)={len(w)} attendu S={S}")

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


def detect_offset(slots, grid_size: int) -> int:
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
    return 0


def build_grid_from_json_and_slots(json_data, slots, words, check_conflicts=True):
    grid_size = json_data["grid_size"]
    base = json_data["grid"]

    if len(base) != grid_size or any(len(r) != grid_size for r in base):
        raise ValueError("grid_size / grid incohérents dans le JSON")

    grid = [row[:] for row in base]
    conflicts = set()

    offset = detect_offset(slots, grid_size)

    for slot, word in zip(slots, words):
        row0 = slot["row"] - offset
        col0 = slot["col"] - offset
        vert = is_vertical(slot["orientation"])
        length = slot.get("length", len(word))  # fallback si absent

        if len(word) != length:
            # si jamais le JSON dit length != len(word), on s'aligne sur le mot
            length = min(length, len(word))

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
                    conflicts.add((r, c))
                    print(f"[WARN] conflit (r={r},c={c}): {grid[r][c]} vs {ch}")

    return grid, conflicts


def print_grid(grid, conflicts):
    for r, row in enumerate(grid):
        out = []
        for c, ch in enumerate(row):
            disp = " " if ch == "." else ch
            if (r, c) in conflicts and disp != " ":
                disp = f"{RED}{disp}{RESET}"
            out.append(disp)
        print(" ".join(out))


# -------------------------
# Main
# -------------------------
def main():
    root = Path(__file__).resolve().parent

    if len(sys.argv) > 1:
        result_path = Path(sys.argv[1]).expanduser()
        if not result_path.is_absolute():
            result_path = (root / result_path).resolve()
    else:
        result_path = root / "results/yuck_opt1_t30s/small" / "small_005.json"

    if not result_path.exists():
        raise FileNotFoundError(f"Result JSON introuvable: {result_path}")

    metrics = json.loads(result_path.read_text(encoding="utf-8"))

    sol = metrics.get("solution")
    if sol is None:
        print("Aucune solution dans ce result.json")
        return

    if "w" not in sol:
        raise KeyError("La solution ne contient pas la clé 'w'")

    w = sol["w"]
    instance_dzn = metrics["instance"]

    instances_root = (root / ".." / ".." / "instances").resolve()

    dzn_path = find_instance_dzn(instances_root, instance_dzn)
    instance_json = find_instance_json(instances_root, instance_dzn)

    print(f"Instance JSON utilisée : {instance_json}")
    print(f"Instance DZN utilisée  : {dzn_path}")

    dzn_data = load_dzn(dzn_path)
    words = build_words_from_dzn_letters(dzn_data, w)

    json_data = json.loads(instance_json.read_text(encoding="utf-8"))
    slots = json_data["slots"]

    grid, conflicts = build_grid_from_json_and_slots(json_data, slots, words, check_conflicts=True)

    print("\n=== GRILLE RECONSTRUITE ===\n")
    print_grid(grid, conflicts)
    print("\n=== FIN ===")


if __name__ == "__main__":
    main()
