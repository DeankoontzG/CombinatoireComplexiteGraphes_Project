from numpy import average
from ortools.sat.python import cp_model
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from Solveurs.fonctions import * 

print("OR-Tools est installé correctement !")

testpaths = []
average_exec_time = 0
average_prep_time = 0
succescount = 0

for i in range(1) : 

    testpaths.append(f"../../instances/small/small_{i+1:03d}.json")
    status, exec_time, prep_time = executeORTools(testpaths[i])
    average_exec_time += (average_exec_time - exec_time)*(1/(i+1))
    average_prep_time += (average_prep_time - prep_time)*(1/(i+1))

    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE : 
        succescount += 1

succesrate = succescount / 20

print(f"Succès rate : {succesrate*100} %; Avg exec time : {average_exec_time}; Avg prep time : {average_prep_time}")
