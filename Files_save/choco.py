# save as build_dzn.py ; usage: python build_dzn.py input.dzn output.dzn
import re, sys, ast, json
from collections import defaultdict

def parse_array(s):
    # parse "X = [a, b, c];" to list
    return ast.literal_eval(re.search(r'\[(.*)\]', s, re.S).group(0).replace(';',''))

def parse_string_array(s):
    return ast.literal_eval(re.search(r'\[(.*)\]', s, re.S).group(0).replace(';',''))

def load_dzn(paths):
    if isinstance(paths, str):
        paths = [paths]
    txt = ""
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            txt += f.read() + "\n"

    def get(name, is_str=False, required=True):
        pat = rf'^\s*{name}\s*=\s*(.*?);'
        m = re.search(pat, txt, re.S | re.M)
        if not m:
            if required:
                raise ValueError(f"Missing {name} in provided .dzn files")
            return None
        val = m.group(0)
        if is_str:
            m2 = re.search(r'\[(.*)\]', val, re.S)
            if not m2:
                raise ValueError(f"{name} is not an array of strings")
            return ast.literal_eval(m2.group(0))
        if '[' in val:
            return ast.literal_eval(re.search(r'\[(.*)\]', val, re.S).group(0))
        return int(re.search(r'=\s*([0-9]+)\s*;', val).group(1))

    return {
        'N_SLOTS': get('N_SLOTS'),
        'LENGTHS': get('LENGTHS'),
        'N_INTERSECTIONS': get('N_INTERSECTIONS'),
        'SLOT1': get('SLOT1'),
        'POS1': get('POS1'),
        'SLOT2': get('SLOT2'),
        'POS2': get('POS2'),
        'N_DICT': get('N_DICT'),
        'DICT': get('DICT', is_str=True),
    }


def write_dzn(base, out_path, NO_REUSE=1):
    N_SLOTS = base['N_SLOTS']
    LENGTHS = base['LENGTHS']
    DICT = base['DICT']
    # alphabet
    alpha = sorted({ch for w in DICT for ch in w})
    ch2i = {c:i for i,c in enumerate(alpha)}
    # candidats par slot (par longueur)
    by_len = defaultdict(list)
    for idx, w in enumerate(DICT, start=1):  # 1-based for DICT
        by_len[len(w)].append((idx, w))

    max_len = max(LENGTHS)
    ccounts = []
    slot_words = []
    slot_ids = []
    max_cand = 0
    for s in range(N_SLOTS):
        L = LENGTHS[s]
        cands = by_len.get(L, [])
        if not cands:
            raise RuntimeError(f"Aucun mot de longueur {L} pour le slot {s+1}")
        ccounts.append(len(cands))
        max_cand = max(max_cand, len(cands))
        slot_ids.append([idx for idx,_ in cands])
        slot_words.append([[ch2i[c] for c in w] + [0]*(max_len - L) for _,w in cands])

    # pad à MAX_CAND
    for s in range(N_SLOTS):
        need = max_cand - len(slot_words[s])
        if need>0:
            L = LENGTHS[s]
            pad_row = [0]*max_len
            slot_words[s].extend([pad_row]*need)
            slot_ids[s].extend([slot_ids[s][0]]*need)  # id bidon, jamais choisi si choice<=CANDS_COUNT

    # écrire .dzn
    def arr(lst): return "[" + ", ".join(map(str,lst)) + "]"
    def arr_str(lst): return "[" + ", ".join('"' + x + '"' for x in lst) + "]"

    with open(out_path, 'w', encoding='utf-8') as f:
        # recopier données d’origine
        f.write(f"N_SLOTS = {base['N_SLOTS']};\n")
        f.write(f"LENGTHS = {arr(base['LENGTHS'])};\n")
        f.write(f"N_INTERSECTIONS = {base['N_INTERSECTIONS']};\n")
        f.write(f"SLOT1 = {arr(base['SLOT1'])};\n")
        f.write(f"POS1 = {arr(base['POS1'])};\n")
        f.write(f"SLOT2 = {arr(base['SLOT2'])};\n")
        f.write(f"POS2 = {arr(base['POS2'])};\n")
        f.write(f"N_DICT = {base['N_DICT']};\n")
        f.write(f"DICT = {arr_str(base['DICT'])};\n")
        # nouveaux blocs
        f.write(f"NO_REUSE = {NO_REUSE};\n")
        f.write(f"ALPHABET_SIZE = {len(alpha)};\n")
        f.write(f"ALPHABET = {arr_str(alpha)};\n")
        f.write(f"MAX_LEN = {max_len};\n")
        f.write(f"MAX_CAND = {max_cand};\n")
        f.write(f"CANDS_COUNT = {arr(ccounts)};\n")

        # CAND_IDS
        f.write("CAND_IDS = array2d(1..N_SLOTS, 1..MAX_CAND, [\n")
        for s in range(N_SLOTS):
            f.write(", ".join(map(str, slot_ids[s])) + (",\n" if s < N_SLOTS-1 else "\n"))
        f.write("]);\n")

        # WORDS (3D aplati en array3d MiniZinc)
        f.write("WORDS = array3d(1..N_SLOTS, 1..MAX_CAND, 1..MAX_LEN, [\n")
        for s in range(N_SLOTS):
            for k in range(max_cand):
                f.write(", ".join(map(str, slot_words[s][k])) + ",\n")
        f.seek(f.tell()-2)  # retire la dernière virgule
        f.write("\n]);\n")

def write_mzn(out_path="crossword.mzn"):
    content = """\
int: N_SLOTS;
array[1..N_SLOTS] of int: LENGTHS;

int: N_INTERSECTIONS;
array[1..N_INTERSECTIONS] of int: SLOT1;
array[1..N_INTERSECTIONS] of int: POS1;
array[1..N_INTERSECTIONS] of int: SLOT2;
array[1..N_INTERSECTIONS] of int: POS2;

int: NO_REUSE;

int: N_DICT;
array[1..N_DICT] of string: DICT;

int: ALPHABET_SIZE;
array[0..ALPHABET_SIZE-1] of string: ALPHABET;

int: MAX_LEN;
int: MAX_CAND;
array[1..N_SLOTS] of int: CANDS_COUNT;
array[1..N_SLOTS, 1..MAX_CAND, 1..MAX_LEN] of 0..ALPHABET_SIZE-1: WORDS;
array[1..N_SLOTS, 1..MAX_CAND] of 1..N_DICT: CAND_IDS;

array[1..N_SLOTS] of var 1..MAX_CAND: choice;
array[1..N_SLOTS, 1..MAX_LEN] of var 0..ALPHABET_SIZE-1: letter;

constraint
  forall(s in 1..N_SLOTS)(
    choice[s] <= CANDS_COUNT[s] /\\
    forall(p in 1..LENGTHS[s]) (
      letter[s,p] = WORDS[s, choice[s], p]
    )
  );

constraint
  forall(i in 1..N_INTERSECTIONS)(
    letter[SLOT1[i], POS1[i]] = letter[SLOT2[i], POS2[i]]
  );

include "alldifferent.mzn";
constraint
  if NO_REUSE = 1 then
    alldifferent([ CAND_IDS[s, choice[s]] | s in 1..N_SLOTS ])
  else
    true
  endif;

output [
  "slot " ++ show(s) ++ " (len=" ++ show(LENGTHS[s]) ++ "): " ++ DICT[CAND_IDS[s, fix(choice[s])]] ++ "\n"
  | s in 1..N_SLOTS
];

"""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"✅ Modèle MiniZinc écrit dans {out_path}")


if __name__ == "__main__":
    if len(sys.argv)<3:
        print("Usage: python build_dzn.py input.dzn output.dzn")
        sys.exit(1)
    base = load_dzn(sys.argv[1])
    write_dzn(base, sys.argv[2], NO_REUSE=1)
    write_mzn("crossword.mzn")
