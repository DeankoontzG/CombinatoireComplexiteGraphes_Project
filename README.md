# Résolution de Grilles de Mots Fléchés par Contraintes (CSP)

Ce projet a été réalisé dans le cadre du module de résolution de problèmes par contraintes. L'objectif est de modéliser et résoudre un problème **NP-Complet** : le remplissage de grilles de mots fléchés.

## Présentation du Problème
Le problème consiste à déterminer si une grille de mots fléchés peut être remplie de manière cohérente en utilisant un dictionnaire de mots donné. 
* **Réponse binaire :** Le programme indique si la grille est faisable ou non.
* **Résultat :** Si la grille est faisable, le solveur l'indique et le code peut renvoyer la grille complétée (en option).

## Architecture du Projet

Le projet est structuré de la manière suivante :

- **`ROOT`**
    - `generate_benchmark.py` : Fonctions permettant de générer des instances de test de tailles variées.
    - `main.py` : Script principal pour lancer la génération du benchmark.
    - **`Solveur/`** : Cœur algorithmique du projet.
        - `fonctions.py` : Fonctions utilitaires (chargement JSON/DZN, parsing).
        - **`Complete/`** : Approche exacte via **OR-Tools**.
            - `OR-Tools.py` : Modélisation CP-SAT, définition des paramètres et heuristiques.
            - `plots/` : Graphiques de performances (temps de calcul, succès).
        - **`Incomplete/`** : Approche par recherche locale via le solveur **Yuck**.
    - **`instances/`** : Benchmark généré (formats JSON et DZN), divisé en catégories `small`, `medium`, et `large`.
    - **`data/`** : Ressources linguistiques.
        - `dataset_cleaning.py` : Script de nettoyage du dictionnaire.
        - Fichiers de vocabulaire.
    - **`Files_save/`** : Sauvegardes et versions alternatives des fichiers.

## Méthodologies de Résolution

Nous avons implémenté et comparé deux types d'approches :

### 1. Méthodes Complètes (OR-Tools)
Nous utilisons le solveur **CP-SAT** de Google OR-Tools.
* **Heuristique de recherche :** Comparaison entre le choix automatique du solveur et plusieurs stratégies de branchement spécifique sur les variables à domaines réduits.

### 2. Méthodes Incomplètes (Yuck / CBLS)
Utilisation du solveur **Yuck** basé sur la recherche locale (Constraint-Based Local Search). 


## Utilisation

### Génération des données
Pour nettoyer le dataset de mots :
``` python
 data/dataset_cleaning.py
```

### Génération du Benchmark
Cette étape crée les instances de test (les grilles vides et les domaines de mots) au format JSON et DZN. Les instances sont classées par difficulté dans le dossier instances/.
``` python
main.py
```

### Appel de la résolution complète
Cette étape lance la résolution des instances spécifiées au début de la fonction. Vous pouvez y modifier au début du fichier **OR-Tools.py** : 
- *base_instance_folder*, pour choisir entre des grilles small, medium, ou large
- *num_instances* pour choisir le nombre d'instances à résoudre (entre 1 et 50)
Pour exécuter la fonction :
``` python
cd Solveur/Complete
OR-Tools.py
```

## 👥 Auteurs
* **Guilhem Dupuy & Artus Bleton**

---
*Note : Ce projet utilise des formats standards (JSON/DZN) pour assurer la compatibilité entre les différents outils de la suite de résolution.*

## Conversion de DZN + MZN en FZN

minizinc --compile-only -o sortie.fzn modele.mzn donnees.dzn

