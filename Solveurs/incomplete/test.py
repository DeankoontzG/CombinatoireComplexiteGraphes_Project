#!/usr/bin/env python3
from pathlib import Path
import json, re
from collections import defaultdict

root = Path(__file__).resolve().parent

# Chemins
result_path = root / "results" / "small_021.json"
json_path   = root / ".." / ".." / "instances" / "small" / "fillfactor_0.90" / "small_021.json"
dzn_path    = root / ".." / ".." / "instances" / "small" / "fillfactor_0.90" / "small_021.dzn"

def clean(w: str) -> str:
    w = w.strip().lower()
    return "".join(ch for ch in w if 'a' <= ch <= 'z')

# --- Charge result.json ---
metrics = json.loads(result_path.read_text(encoding="utf-8"))
w = metrics["solution"]["w"]            # indices 1..maxC

# --- Charge JSON origine ---
data_json = json.loads(json_path.read_text(encoding="utf-8"))
slots = data_json["slots"]
dictionary = data_json["dictionary"]

DICT = [clean(x) for x in dictionary if clean(x)]
by_len = defaultdict(list)
for ww in DICT:
    by_len[len(ww)].append(ww)

# --- Charge dzn ---
txt = dzn_path.read_text()

def parse_int(name, text):
    m = re.search(rf"\b{name}\s*=\s*(\d+)\s*;", text)
    return int(m.group(1)) if m else None

def parse_array_1d(name, text):
    m = re.search(rf"\b{name}\s*=\s*\[(.*?)\]\s*;", text, re.S)
    raw = m.group(1).replace("\n", " ").split(",")
    return [int(x.strip()) for x in raw if x.strip()]

def parse_array3d_letter(text, S, maxC, maxL):
    m = re.search(r"letter\s*=\s*array3d\(.*?\[(.*?)\]\);", text, re.S)
    vals = [int(x.strip()) for x in m.group(1).replace("\n", " ").split(",") if x.strip()]
    letter = [[[0]*maxL for _ in range(maxC)] for __ in range(S)]
    idx = 0
    for s in range(S):
        for c in range(maxC):
            for p in range(maxL):
                letter[s][c][p] = vals[idx]
                idx += 1
    return letter

S    = parse_int("S", txt)
maxC = parse_int("maxC", txt)
maxL = parse_int("maxL", txt)
L    = parse_array_1d("L", txt)
letter = parse_array3d_letter(txt, S, maxC, maxL)

print(f"S = {S}, maxC = {maxC}, maxL = {maxL}")
print(f"len(slots) = {len(slots)}, len(L) = {len(L)}, len(w) = {len(w)}")

# --- Vérification slot par slot ---
ok_global = True

for s in range(S):
    length = L[s]
    k0 = w[s] - 1      # 0-based
    # mot reconstruit depuis letter
    codes = [letter[s][k0][p] for p in range(length)]
    word_dzn = "".join(chr(ord("a")+c) for c in codes)

    # pool attendu depuis JSON
    pool = by_len[length]  # liste de mots nettoyés
    in_pool = word_dzn in pool

    if not in_pool:
        ok_global = False
        print(f"[PROBLEME] Slot {s+1}, L={length}: mot_dzn='{word_dzn}' PAS dans pool by_len[{length}]")
    else:
        print(f"[OK] Slot {s+1}, L={length}: mot_dzn='{word_dzn}' est dans pool (taille {len(pool)})")

print("\n=== RESULTAT GLOBAL ===")
print("COHERENT" if ok_global else "INCOHERENT")
