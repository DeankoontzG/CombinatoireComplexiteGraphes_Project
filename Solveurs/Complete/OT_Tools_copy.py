from numpy import average
from ortools.sat.python import cp_model
import matplotlib.pyplot as plt

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from Solveurs.fonctions import * 

print("OR-Tools est installé correctement !")

fill_factors = ["0.80"]
base_instance_folder = "medium"
num_instances = 1

results = {
    'fill_factor': [],
    'avg_exec_time': [],
    'avg_exec_time_a': [],
    'avg_exec_time_b': [],
    'avg_exec_time_c': [],
    'avg_exec_time_d': [],
    'avg_prep_time': [],
    'success_rate': []
}

testpaths = []
average_exec_time = 0
average_exec_time_a = 0
average_exec_time_b = 0
average_exec_time_c = 0
average_exec_time_d = 0
average_prep_time = 0
succescount = 0
script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.abspath(os.path.join(script_dir, '../../'))

for fill_factor in fill_factors:
    print(f"\n--- Traitement du Facteur de Remplissage: {fill_factor} ---")

    current_exec_times = []
    current_exec_times_a = []
    current_exec_times_b = []
    current_exec_times_c = []
    current_exec_times_d = []
    current_prep_times = []
    succescount = 0

    instance_subdir = os.path.join(base_dir, f"instances/{base_instance_folder}/fillfactor_{fill_factor}")

    for i in range(1, num_instances +1) : 

        instance_filename = f"{base_instance_folder}_{i:03d}.json"
        instance_path = os.path.join(instance_subdir, instance_filename)

        if not os.path.exists(instance_path):
            print(f"ATTENTION: Instance non trouvée : {instance_path}")
            continue

        if i == num_instances: # Affiche la grille uniquement pour la dernière instance du sous-dossier
             status, exec_time, prep_time, solver = executeORTools(instance_path, display_grid=True, timeout=120)
            #  status_a, exec_time_a, prep_time_a, solver_a = 0,0,0,0
            #  status_b, exec_time_b, prep_time_b, solver_b = 0,0,0,0
            #  status_c, exec_time_c, prep_time_c, solver_c = 0,0,0,0
            #  status_d, exec_time_d, prep_time_d, solver_d = 0,0,0,0
             status_a, exec_time_a, prep_time_a, solver_a = executeORTools(instance_path, display_grid=True, heuristic="IntersectionsFirst")
             status_b, exec_time_b, prep_time_b, solver_b = executeORTools(instance_path, display_grid=True, heuristic="LongestWordsFirst")
             status_c, exec_time_c, prep_time_c, solver_c = executeORTools(instance_path, display_grid=True, heuristic="MostIntersectionsFirst")
             status_d, exec_time_d, prep_time_d, solver_d = executeORTools(instance_path, display_grid=True, heuristic="TopologicalOrder")

            
        else:
             status, exec_time, prep_time, solver = executeORTools(instance_path, display_grid=False)
            #  status_a, exec_time_a, prep_time_a, solver_a = 0,0,0,0
            #  status_b, exec_time_b, prep_time_b, solver_b = 0,0,0,0
            #  status_c, exec_time_c, prep_time_c, solver_c = 0,0,0,0
            #  status_d, exec_time_d, prep_time_d, solver_d = 0,0,0,0
             status_a, exec_time_a, prep_time_a, solver_a = executeORTools(instance_path, display_grid=True, heuristic="IntersectionsFirst")
             status_b, exec_time_b, prep_time_b, solver_b = executeORTools(instance_path, display_grid=True, heuristic="LongestWordsFirst")
             status_c, exec_time_c, prep_time_c, solver_c = executeORTools(instance_path, display_grid=True, heuristic="MostIntersectionsFirst")
             status_d, exec_time_d, prep_time_d, solver_d = executeORTools(instance_path, display_grid=True, heuristic="TopologicalOrder")

        current_exec_times.append(solver.UserTime())
        current_exec_times_a.append(solver_a.UserTime())
        current_exec_times_b.append(solver_b.UserTime())
        current_exec_times_c.append(solver_c.UserTime())
        current_exec_times_d.append(solver_d.UserTime())
        current_prep_times.append(prep_time)

        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE: 
            succescount += 1
            
        print(f"Instance {i}/{num_instances} - Temps exec: {exec_time:.4f}s, Statut: {status}")

    if current_exec_times:
        avg_exec_time = sum(current_exec_times) / len(current_exec_times)
        avg_exec_time_a = sum(current_exec_times_a) / len(current_exec_times_a)
        avg_exec_time_b = sum(current_exec_times_b) / len(current_exec_times_b)
        avg_exec_time_c = sum(current_exec_times_c) / len(current_exec_times_c)
        avg_exec_time_d = sum(current_exec_times_d) / len(current_exec_times_d)
        avg_prep_time = sum(current_prep_times) / len(current_prep_times)
        success_rate = succescount / len(current_exec_times)
    else:
        # Cas où aucune instance n'a été traitée
        avg_exec_time, avg_exec_time_a, avg_exec_time_b, avg_exec_time_c, avg_exec_time_d, avg_prep_time, success_rate = 0, 0, 0, 0, 0, 0, 0

    # Sauvegarde des résultats
    results['fill_factor'].append(float(fill_factor))
    results['avg_exec_time'].append(avg_exec_time)
    results['avg_exec_time_a'].append(avg_exec_time_a)
    results['avg_exec_time_b'].append(avg_exec_time_b)
    results['avg_exec_time_c'].append(avg_exec_time_c)
    results['avg_exec_time_d'].append(avg_exec_time_d)
    results['avg_prep_time'].append(avg_prep_time)
    results['success_rate'].append(success_rate * 100) # En pourcentage
    
    print(f"Résultats pour {fill_factor}: Taux de succès: {success_rate*100:.2f} %; Temps exec moyen: {avg_exec_time:.4f}s")

# ----------------------------------------------------------------------
# --- Affichage des résultats finaux et Courbes de Performance ---
# ----------------------------------------------------------------------

print("\n" + "="*50)
print("✨ Résumé des Performances par Facteur de Remplissage ✨")
print("="*50)

# Affichage des données brutes
for i, ff in enumerate(results['fill_factor']):
    print(f"Fill Factor {ff:.2f}: Succès: {results['success_rate'][i]:.2f} % | Exec Time: {results['avg_exec_time'][i]:.4f}s")


# --- Traçage des courbes avec Matplotlib ---
# 2 sous-graphiques (un pour le temps, un pour le taux de succès)
fig, ax1 = plt.subplots(figsize=(12, 7))

strategies = [
    ('avg_exec_time', 'Standard', 'black', 'o'),
    ('avg_exec_time_a', 'IntersectionsFirst', 'red', 's'),
    ('avg_exec_time_b', 'LongestWordsFirst', 'blue', '^'),
    ('avg_exec_time_c', 'MostIntersectionsFirst', 'green', 'd'),
    ('avg_exec_time_d', 'TopologicalOrder', 'yellow', 'x')
]

ax1.set_xlabel('Facteur de Remplissage (Fill Factor)')
ax1.set_ylabel('Temps d\'Exécution Moyen (s)', color='darkslategray') # Couleur neutre
ax1.tick_params(axis='y', labelcolor='darkslategray')
ax1.grid(True, linestyle='--', alpha=0.6, axis='y') # Grille pour l'axe Y des temps

# Tracé de chaque stratégie
for key, label, color, marker in strategies:
    ax1.plot(
        results['fill_factor'], 
        results[key], 
        color=color, 
        marker=marker, 
        linestyle='-', 
        label=f'Temps - {label}'
    )

ax1.legend(loc='upper left', frameon=True, title="Temps d'Exécution")

# second axe Y pour le Taux de Succès, partageant l'axe X
ax2 = ax1.twinx()  
color_success = 'tab:purple'
ax2.set_ylabel('Taux de Succès (%)', color=color_success, fontweight='bold')  
ax2.plot(
    results['fill_factor'], 
    results['success_rate'], 
    color=color_success, 
    marker='X', 
    linestyle='--', 
    linewidth=2,
    label='Taux de Succès Global' # Note : on suppose que le taux de succès est le même pour toutes les heuristiques si basé sur 'status'
)
ax2.tick_params(axis='y', labelcolor=color_success)
ax2.set_ylim(0, 105) # S'assurer que l'échelle est en pourcentage (0 à 100%)
ax2.legend(loc='upper right', frameon=True, title="Taux de Succès")


plt.title(f'Performance du Solveur OR-Tools ({base_instance_folder})', fontsize=14, fontweight='bold')
fig.tight_layout() 

# --- Sauvegarde du graphique dans un fichier PNG ---
plt.savefig('performances_plot.png') 
print("\nGraphique sauvegardé sous : performances_plot.png")

plt.show()
