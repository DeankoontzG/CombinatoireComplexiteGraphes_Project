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
array[1..ALPHABET_SIZE] of string: ALPHABET;
array[0..ALPHABET_SIZE-1] of string: ALPHABET0 = array1d(0..ALPHABET_SIZE-1, ALPHABET);

int: MAX_LEN;
int: MAX_CAND;
array[1..N_SLOTS] of int: CANDS_COUNT;
array[1..N_SLOTS, 1..MAX_CAND, 1..MAX_LEN] of 0..ALPHABET_SIZE-1: WORDS;
array[1..N_SLOTS, 1..MAX_CAND] of 1..N_DICT: CAND_IDS;

array[1..N_SLOTS] of var 1..MAX_CAND: choice;
array[1..N_SLOTS, 1..MAX_LEN] of var 0..ALPHABET_SIZE-1: letter;

constraint
  forall(s in 1..N_SLOTS)(
    choice[s] <= CANDS_COUNT[s] /\
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

solve satisfy;

output [
  "slot " ++ show(s) ++ " (len=" ++ show(LENGTHS[s]) ++ "): " ++
  DICT[CAND_IDS[s, fix(choice[s])]] ++
  (if s < N_SLOTS then "\n" else "" endif)
  | s in 1..N_SLOTS
];
