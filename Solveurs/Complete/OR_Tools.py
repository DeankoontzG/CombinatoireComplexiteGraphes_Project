from numpy import average
from ortools.sat.python import cp_model
import matplotlib.pyplot as plt

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from Solveurs.fonctions import * 

print("OR-Tools est installé correctement !")

fill_factors = ["0.80", "0.85", "0.90", "0.95", "1.00"]
base_instance_folder = "medium"
num_instances = 10

results = {
    'fill_factor': [],
    'avg_exec_time': [],
    'avg_prep_time': [],
    'success_rate': []
}

testpaths = []
average_exec_time = 0
average_prep_time = 0
succescount = 0
script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.abspath(os.path.join(script_dir, '../../'))

for fill_factor in fill_factors:
    print(f"\n--- Traitement du Facteur de Remplissage: {fill_factor} ---")

    current_exec_times = []
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
             status, exec_time, prep_time = executeORTools(instance_path, display_grid=True)
        else:
             status, exec_time, prep_time = executeORTools(instance_path, display_grid=False)

        current_exec_times.append(exec_time)
        current_prep_times.append(prep_time)

        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE: 
            succescount += 1
            
        print(f"Instance {i}/{num_instances} - Temps exec: {exec_time:.4f}s, Statut: {status}")

    if current_exec_times:
        avg_exec_time = sum(current_exec_times) / len(current_exec_times)
        avg_prep_time = sum(current_prep_times) / len(current_prep_times)
        success_rate = succescount / len(current_exec_times)
    else:
        # Cas où aucune instance n'a été traitée
        avg_exec_time, avg_prep_time, success_rate = 0, 0, 0

    # Sauvegarde des résultats
    results['fill_factor'].append(float(fill_factor))
    results['avg_exec_time'].append(avg_exec_time)
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
fig, ax1 = plt.subplots(figsize=(10, 6))


# 1. Courbe du Temps d'Exécution Moyen
color_time = 'tab:red'
ax1.set_xlabel('Facteur de Remplissage (Fill Factor)')
ax1.set_ylabel('Temps d\'Exécution Moyen (s)', color=color_time)
ax1.plot(results['fill_factor'], results['avg_exec_time'], color=color_time, marker='o', label='Temps d\'Exécution')
ax1.tick_params(axis='y', labelcolor=color_time)
ax1.grid(True, linestyle='--', alpha=0.6)

# second axe Y pour le Taux de Succès, partageant l'axe X
ax2 = ax1.twinx()  
color_success = 'tab:blue'
ax2.set_ylabel('Taux de Succès (%)', color=color_success)  
ax2.plot(results['fill_factor'], results['success_rate'], color=color_success, marker='x', linestyle='--', label='Taux de Succès')
ax2.tick_params(axis='y', labelcolor=color_success)

plt.title(f'Performance du Solveur OR-Tools pour les instances {base_instance_folder} en fonction du Fill Factor')
fig.tight_layout() 

# --- Sauvegarde du graphique dans un fichier PNG ---
plt.savefig('performances_plot.png') 
print("\nGraphique sauvegardé sous : performances_plot.png")

plt.show() 
