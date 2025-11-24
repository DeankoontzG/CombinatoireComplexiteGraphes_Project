from numpy import average
import pandas as pd
import numpy as np
from ortools.sat.python import cp_model
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from Solveurs.fonctions import * 

print("OR-Tools est installé correctement !")

fill_factors = ["0.85"]
base_instance_folder = "medium"
num_instances = 3

STRATEGIES = {
    "Standard": None, 
    "IntersectionsFirst": "IntersectionsFirst",
    "LongestWordsFirst": "LongestWordsFirst",
    "MostIntersectionsFirst": "MostIntersectionsFirst",
    "TopologicalOrder": "TopologicalOrder"
}

results = {
    'fill_factor': [],
    'avg_prep_time': []
}

METRICS = ['exec_time', 'branches', 'conflicts', 'success_rate']

for metric in METRICS:
    for name in STRATEGIES:
        key = f"avg_{metric}_{name.replace(' ', '')}"
        results[key] = []


script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.abspath(os.path.join(script_dir, '../../'))


for fill_factor in fill_factors:
    print(f"\n--- Traitement du Facteur de Remplissage: {fill_factor} ---")

    current_data = {
        name: {'exec_time': [], 'branches': [], 'conflicts': [], 'success_count': 0, 'prep_time': []}
        for name in STRATEGIES
    }
    
    instance_subdir = os.path.join(base_dir, f"instances/{base_instance_folder}/fillfactor_{fill_factor}")

    for i in range(1, num_instances + 1):
        instance_filename = f"{base_instance_folder}_{i:03d}.json"
        instance_path = os.path.join(instance_subdir, instance_filename)

        if not os.path.exists(instance_path):
            print(f"ATTENTION: Instance non trouvée : {instance_path}")
            continue

        display = (i == num_instances) # Afficher seulement la dernière grille

        # BOUCLE SUR LES STRATÉGIES
        for name, heuristic_key in STRATEGIES.items():
            
            
            status, exec_time, prep_time, solver = executeORTools(instance_path, display_grid=display, timeout=120, heuristic=heuristic_key)
            
            current_data[name]['exec_time'].append(solver.UserTime())
            current_data[name]['branches'].append(solver.NumBranches())
            current_data[name]['conflicts'].append(solver.NumConflicts())
            current_data[name]['prep_time'].append(prep_time)
            
            if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE: 
                current_data[name]['success_count'] += 1
            
            
            print(f"Strat {name}, Instance {i}/{num_instances} - Temps exec: {exec_time:.4f}s, Statut: {status_name(status)}")
    
    # Calcul et enregistrement des moyennes pour ce fill_factor
    num_tested = num_instances
    avg_prep_time = sum(current_data["Standard"]['prep_time']) / num_tested
    results['fill_factor'].append(float(fill_factor))
    results['avg_prep_time'].append(avg_prep_time)

    for name in STRATEGIES:
        key_suffix = name.replace(' ', '')
        
        # Calculer les moyennes
        avg_exec_time = sum(current_data[name]['exec_time']) / num_tested
        avg_branches = sum(current_data[name]['branches']) / num_tested
        avg_conflicts = sum(current_data[name]['conflicts']) / num_tested
        success_rate = current_data[name]['success_count'] / num_tested
        
        # Sauvegarde dans le dictionnaire results
        results[f"avg_exec_time_{key_suffix}"].append(avg_exec_time)
        results[f"avg_branches_{key_suffix}"].append(avg_branches)
        results[f"avg_conflicts_{key_suffix}"].append(avg_conflicts)
        results[f"avg_success_rate_{key_suffix}"].append(success_rate * 100)

    print(f"Résultats pour {fill_factor}: Taux de succès (Std): {results['avg_success_rate_Standard'][-1]:.2f} %; Temps exec moyen (Std): {results['avg_exec_time_Standard'][-1]:.4f}s")


# =================================
# --- AFFICHAGE ET TRACÉ  ---
# =================================

strategies_for_plot = [
    ("Standard", 'black', 'o'),
    ("IntersectionsFirst", 'red', 's'),
    ("LongestWordsFirst", 'blue', '^'),
    ("MostIntersectionsFirst", 'green', 'd'),
    ("TopologicalOrder", 'purple', 'x')
]

# Les noms des stratégies (utilisés pour les boucles et la table)
strategy_names = [name for name, _, _ in strategies_for_plot]

# Métriques à afficher sur l'axe des X (discrètes)
METRICS_FOR_PLOT = ['Temps (s)', 'Branches', 'Conflits']
x_labels = METRICS_FOR_PLOT
x_indices = np.arange(len(x_labels)) # [0, 1, 2]

# --- Initialisation Unique des valeurs de Fill Factor ---
ff_value = str(results['fill_factor'][0]).replace('.', '_')
fill_factor_val = results['fill_factor'][0]

# --- INITIALISATION de data_metrics ---
data_metrics = {}
for name, _, _ in strategies_for_plot:
    key_suffix = name.replace(' ', '')
    data_metrics[name] = {
        'exec_time': results[f"avg_exec_time_{key_suffix}"][0],
        'branches': results[f"avg_branches_{key_suffix}"][0],
        'conflicts': results[f"avg_conflicts_{key_suffix}"][0],
        'success_rate': results[f"avg_success_rate_{key_suffix}"][0]
    }

# 1. Calcul des maximums pour la normalisation (Min-Max Scaling)
max_values = {
    'exec_time': np.max([data_metrics[name]['exec_time'] for name, _, _ in strategies_for_plot]),
    'branches': np.max([data_metrics[name]['branches'] for name, _, _ in strategies_for_plot]),
    'conflicts': np.max([data_metrics[name]['conflicts'] for name, _, _ in strategies_for_plot])
}

# 2. Création des données normalisées (de 0 à 1)
data_normalized = {}
for name, _, _ in strategies_for_plot:
    data_normalized[name] = [
        data_metrics[name]['exec_time'] / max_values['exec_time'],
        data_metrics[name]['branches'] / max_values['branches'],
        data_metrics[name]['conflicts'] / max_values['conflicts']
    ]
    
# Max des échelles réelles (pour les axes auxiliaires)
max_time = max_values['exec_time']
max_branches = max_values['branches']
max_conflicts = max_values['conflicts']


# =================================
# --- GÉNÉRATION DU GRAPHIQUE ---
# =================================

fig, ax1 = plt.subplots(figsize=(14, 8))

# --- AXE X : Métriques ---
ax1.set_xlabel('Métrique de Performance Mesurée', fontsize=12)
ax1.set_xticks(x_indices)
ax1.set_xticklabels(x_labels, rotation=0, ha="center")
ax1.set_xlim(-0.5, len(x_indices) - 0.5)
ax1.grid(True, linestyle='--', alpha=0.6, axis='x')

# --- AXE Y1 (Principal) : NORMALISÉ (de 0 à 1) ---
color_normalized = 'black'
ax1.set_ylabel('Performance Normalisée (Max = 1)', color=color_normalized, fontsize=12, weight='bold')
ax1.set_ylim(0, 1.1)
ax1.tick_params(axis='y', labelcolor=color_normalized) # CORRECTION : Utiliser la couleur black/normalized

# --- AXE Y2 (Conflits, Rouge) ---
ax2 = ax1.twinx()
color_conflicts = 'tab:red'
ax2.set_ylabel(f'Conflits (Max: {max_conflicts:,.0f})', color=color_conflicts, fontsize=10)
ax2.tick_params(axis='y', labelcolor=color_conflicts)
ax2.set_ylim(0, max_conflicts * 1.1)

# --- AXE Y3 (Branches, Vert) ---
ax3 = ax1.twinx()
ax3.spines['right'].set_position(('outward', 60)) 
color_branches = 'tab:green'
ax3.set_ylabel(f'Branches (Max: {max_branches:,.0f})', color=color_branches, fontsize=10)
ax3.tick_params(axis='y', labelcolor=color_branches)
ax3.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
ax3.set_ylim(0, max_branches * 1.1)

# --- AXE Y4 (Temps, Bleu) ---
ax4 = ax1.twinx()
ax4.spines['right'].set_position(('outward', 120))
color_time = 'tab:blue'
ax4.set_ylabel(f'Temps (s) (Max: {max_time:.4f})', color=color_time, fontsize=10)
ax4.tick_params(axis='y', labelcolor=color_time)
ax4.set_ylim(0, max_time * 1.1)


# --- TRACÉ DES DONNÉES NORMALISÉES (LIGNES CONTINUES sur AX1) ---
legend_handles = []
legend_labels = []

for name, color, marker in strategies_for_plot:
    
    values_real = data_metrics[name] # Valeurs réelles pour l'annotation

    # Tracé de la ligne normalisée sur l'axe principal (ax1)
    line_strat, = ax1.plot(
        x_indices, 
        data_normalized[name], 
        color=color, 
        marker='o', 
        linestyle='-', 
        linewidth=2, 
        label=name
    )
    legend_handles.append(line_strat)
    legend_labels.append(name)
    
    # Annotation des valeurs réelles pour chaque point
    # Temps (indice 0)
    ax4.text(x_indices[0] + 0.05, values_real['exec_time'], f'{values_real["exec_time"]:.4f}', 
             ha='left', va='center', fontsize=8, color=color)
    
    # Branches (indice 1)
    ax3.text(x_indices[1] + 0.05, values_real['branches'], f'{values_real["branches"]:,.0f}', 
             ha='left', va='center', fontsize=8, color=color)
    
    # Conflits (indice 2)
    ax2.text(x_indices[2] + 0.05, values_real['conflicts'], f'{values_real["conflicts"]:,.0f}', 
             ha='left', va='center', fontsize=8, color=color)


# --- LÉGENDE FINALE ---
# Créer les handles des clés des marqueurs
metric_handles = [
    Line2D([0], [0], color='black', marker='o', linestyle='-', label='Temps (s)'),
    Line2D([0], [0], color='black', marker='s', linestyle='-', label='Branches'),
    Line2D([0], [0], color='black', marker='^', linestyle='-', label='Conflits'),
]
metric_labels = [h.get_label() for h in metric_handles]

# Affichage de la légende combinée (Stratégies + Clés des marqueurs)
ax1.legend(legend_handles + metric_handles, legend_labels + metric_labels, 
           loc='upper left', title="Stratégie / Métrique", ncol=2, fontsize='small')


# --- FINALISATION ET SAUVEGARDE ---
ax1.set_title(f'Performance Normalisée des Heuristiques (Fill Factor {fill_factor_val})', 
              fontsize=14, fontweight='bold')
fig.tight_layout()

filename = f"HeuristicsPerfsComparison_{base_instance_folder}_FF{ff_value}_Normalized.png"
plt.savefig(filename)
print(f"Graphique Normalisé (Triple-Axe) sauvegardé : {filename}")
plt.show()


# =================================
# --- Sauvegarde du Tableau Récapitulatif ---
# =================================

print("\n" + "="*50)
print("💾 Sauvegarde du Tableau Récapitulatif des Performances (valeurs) 💾")
print("="*50)

df_data = []
for name in strategy_names:
    row = {
        'Stratégie': name,
        'Fill Factor': results['fill_factor'][0],
        'Temps (s)': data_metrics[name]['exec_time'],
        'Branches': data_metrics[name]['branches'],
        'Conflits': data_metrics[name]['conflicts'],
        'Taux de Succès (%)': data_metrics[name]['success_rate']
    }
    df_data.append(row)

df_results = pd.DataFrame(df_data)

df_results['Temps (s)'] = df_results['Temps (s)'].round(4)
df_results['Taux de Succès (%)'] = df_results['Taux de Succès (%)'].round(2)
df_results['Branches'] = df_results['Branches'].round(0).astype(int)
df_results['Conflits'] = df_results['Conflits'].round(0).astype(int)

csv_filename = f"HeuristicsPerfsComparison_{base_instance_folder}_FF{ff_value}_Summary.csv"
df_results.to_csv(csv_filename, index=False, sep=';', decimal=',')

print(f"Tableau récapitulatif sauvegardé : {csv_filename}")
print("\n--- Tableau récapitulatif (Aperçu) ---")
print(df_results)