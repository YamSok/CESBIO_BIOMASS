import os
import numpy as np
from croscor import *



axis0, axis1 = 0,0
seuil = 3
distance = 2 * seuil
k = 0
while distance > seuil:
    command = f"mpiexec -n 4 python xpparallel.py {axis0} {axis1}"
    print(f"Itération {k} : axis0 = {axis0} | axis1 = {axis1}")
    os.system(command)
    c = choiceSimple(folder = '../decoup',all = False,first = True)
    tab = np.load("../decoup/" + c)
    distance, xdist, ydist = countCorrect(tab,seuil, verbose=False)
    axis0 += int(np.floor(ydist))
    axis1 += int(np.floor(xdist))
    k += 1

print(f"Convergence in {k} itération for a threshold of {seuil} px \nBest choices :  axis0 = {axis0} | axis1 = {axis1}")