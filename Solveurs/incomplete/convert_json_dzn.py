# json_to_dzn_recursive.py
import json, string
from collections import defaultdict
from pathlib import Path

def to_code(c):
    return string.ascii_lowercase.index(c)

def clean(w):
    w = w.strip().lower()
    return "".join(ch for ch in w if 'a' <= ch <= 'z')

def json_to_rect_dzn(path_json: Path) -> str:
    with path_json.open('r', encoding='utf-8') as f:
        data = json.load(f)

    slots = data["slots"]                # [{id,row,col,orientation,length}]
    inters = data["intersections"]       # [{s1,p1,s2,p2}]
    DICT   = [clean(w) for w in data["dictionary"] if clean(w)]

    by_len = defaultdict(list)
    for w in DICT:
        by_len[len(w)].append(w)

    S = len(slots)
    L = [s["length"] for s in slots]
    maxL = max(L) if L else 0
    pools = [by_len[L[sidx]] for sidx in range(S)]
    maxC = max((len(p) for p in pools), default=0)

    # valid & letter
    valid = []
    letter = []
    for sidx in range(S):
        pool = pools[sidx]
        valid += [True]*len(pool) + [False]*(maxC - len(pool))
        for k in range(maxC):
            if k < len(pool):
                w = pool[k]
                row = [to_code(w[i]) for i in range(len(w))] + [-1]*(maxL - len(w))
            else:
                row = [-1]*maxL
            letter += row

    X  = len(inters)
    s1 = [x["s1"] for x in inters]
    p1 = [x["p1"] for x in inters]
    s2 = [x["s2"] for x in inters]
    p2 = [x["p2"] for x in inters]

    def arr1(name, arr):
        return f'{name} = [ {", ".join(map(str, arr))} ];\n'

    dzn = []
    dzn.append(f"S = {S};\n")
    dzn.append(arr1("L", L))
    dzn.append(f"maxC = {maxC};\n")
    dzn.append(f"maxL = {maxL};\n")

    valid_str = ", ".join("true" if b else "false" for b in valid)
    dzn.append(f"valid = array2d(1..{S},1..{maxC}, [ {valid_str} ]);\n")

    letter_str = ", ".join(map(str, letter))
    dzn.append(f"letter = array3d(1..{S},1..{maxC},1..{maxL}, [ {letter_str} ]);\n")

    dzn.append(f"X = {X};\n")
    dzn.append(arr1("s1", s1))
    dzn.append(arr1("p1", p1))
    dzn.append(arr1("s2", s2))
    dzn.append(arr1("p2", p2))
    return "".join(dzn)

# === Conversion récursive de tous les JSON d’un dossier racine ===
input_root  = Path("../../instances")         # dossier source (avec sous-dossiers)
output_root = Path("../../instances")         # dossier cible (arborescence répliquée)
output_root.mkdir(parents=True, exist_ok=True)

json_files = sorted(input_root.rglob("*.json"))
if not json_files:
    print(f"Aucun .json trouvé sous {input_root.resolve()}")
else:
    print(f"{len(json_files)} fichiers .json trouvés sous {input_root}")

for json_path in json_files:
    # chemin relatif pour recréer l’arborescence
    rel = json_path.relative_to(input_root)
    dzn_path = (output_root / rel).with_suffix(".dzn")
    dzn_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Conversion : {rel} → {dzn_path.relative_to(output_root)}")
    dzn_text = json_to_rect_dzn(json_path)
    dzn_path.write_text(dzn_text, encoding="utf-8")

print("Terminé.")
