%%%% crossword.mzn %%%%

% --- Données d'entrée ---
int: N_SLOTS;                         % nombre de slots à remplir
array[1..N_SLOTS] of int: SLOT_LEN;   % longueur de chaque slot
set of int: DICT;                     % indices des mots disponibles
array[DICT] of string: WORDS;         % le dictionnaire
array[1..N_SLOTS] of set of int: POSSIBLE_WORDS;  % mots compatibles (par taille)

% --- Contraintes d’intersection ---
int: N_INTERSECTIONS;
array[1..N_INTERSECTIONS] of int: SLOT1;
array[1..N_INTERSECTIONS] of int: POS1;
array[1..N_INTERSECTIONS] of int: SLOT2;
array[1..N_INTERSECTIONS] of int: POS2;

% --- Variables ---
array[1..N_SLOTS] of var DICT: word_index;   % mot choisi pour chaque slot

% --- Contraintes : tailles compatibles ---
constraint forall(i in 1..N_SLOTS) (
    word_index[i] in POSSIBLE_WORDS[i]
);

% --- Contraintes : intersections cohérentes ---
constraint forall(k in 1..N_INTERSECTIONS) (
    let {
        var string: w1 = WORDS[word_index[SLOT1[k]]],
        var string: w2 = WORDS[word_index[SLOT2[k]]]
    } in
        substr(w1, POS1[k], 1) = substr(w2, POS2[k], 1)
);

% --- Pas de doublons (optionnel) ---
constraint all_different(word_index);

solve satisfy;
