% --- Dimensions ---
int: S;                               % nb de slots
array[1..S] of int: L;                % longueur de chaque slot
int: maxC;                            % nb max de candidats par slot
int: maxL;                            % longueur max d’un slot

% --- Candidats pré-calculés (padding rectangulaire) ---
array[1..S, 1..maxC] of bool: valid;                 % valid[s,k] = vrai si le candidat k du slot s existe
array[1..S, 1..maxC, 1..maxL] of int: letter;        % code lettre (0..25) ou -1 pour padding

% --- Intersections (1-based) ---
int: X;                                                % nb d’intersections
array[1..X] of int: s1;  array[1..X] of int: p1;      % slot s1, position p1
array[1..X] of int: s2;  array[1..X] of int: p2;      % slot s2, position p2

% --- Décisions ---
array[1..S] of var 1..maxC: w;        % pour chaque slot, index du mot choisi

% --- Faisabilité des choix ---
constraint forall(s in 1..S)( valid[s, w[s]] );

% (optionnel) Sécurité des positions si besoin :
% constraint forall(x in 1..X)( p1[x] <= L[s1[x]] /\ p2[x] <= L[s2[x]] );

% --- Conflits aux intersections ---
array[1..X] of var 0..1: conflict;

constraint forall(x in 1..X)(
  % 1 si les lettres sont différentes, 0 si elles sont égales
  conflict[x] = bool2int(
    letter[s1[x], w[s1[x]], p1[x]]
    !=
    letter[s2[x], w[s2[x]], p2[x]]
  )
);

% --- Objectif : minimiser le nombre total de conflits ---
solve minimize sum(x in 1..X)( conflict[x] );

% --- Sortie : indices choisis + score ---
output [
  "w = ", show([ w[s] | s in 1..S ]), "\n",
  "total_conflicts = ", show(sum(x in 1..X)( conflict[x] )), "\n"
];
