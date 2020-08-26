import os
import numpy as np
from croscor import *


'''
Ce qu'il faut faire : 
    * Itérer tant qu'on touche les bords de la région qu'on s'est fixé
        * Si accu < seuil (ex : 70%)
    * Si on touche le bord on integre un petit shift dans la direction moyenne
    * On itère jusqu'à avoir un vecteur déplacement de norme inférieur au seuil de tolérence
'''


# Images 

band1 = "../data/band1.npy"
band2 = "../data/band2.npy"
# band1 = '../data/afri_band1.npy'
# band2 = '../data/afri_band1.npy'

# Initialisation du décalage
axis0, axis1 = 0,0
seuil = 3 #px
distance = 2 * seuil
k = 0
while distance > seuil:
    command = f"mpiexec -n 4 python xpparallel.py {band1} {band2} {axis0} {axis1}"
    print(f"\nItération {k} : axis0 = {axis0} | axis1 = {axis1}")
    os.system(command)
    c = choiceSimple(folder = '../decoup/afri/',all = False,first = True)
    tab = np.load("../decoup/afri/" + c)
    distance, xdist, ydist = countCorrect(tab,seuil, verbose=False)
    # print(f"Mean x-displacement : {xdist} | Mean y-displacement : {ydist}")
    axis0 += int(np.floor(ydist))
    axis1 += int(np.floor(xdist))
    print(f'ajout de {np.floor(xdist)} en x | ajout de {np.floor(xdist)} en y ')
    k += 1

print(f"Convergence in {k} itération for a threshold of {seuil} px \nBest choices :  axis0 = {axis0} | axis1 = {axis1}")

