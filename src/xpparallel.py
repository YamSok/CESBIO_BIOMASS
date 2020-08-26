'''
Main program for estimating biomass quantity from satellite and DEM frames.
'''

import mpi4py.MPI as mpi
import numpy as np
from croscor import *
import time
import warnings
warnings.filterwarnings("ignore")
import sys

def main(band1, band2, axis0,axis1,bs,f,seuil):

    '''
    Loads satellite and DEM frames, and computes the mean shift. Saves the results for post processings.
    '''

    b1 = np.load(band1)
    b2 = np.load(band2)
    # b1,b2 = shiftSelec(band1,band2,axis0,axis1)

    ### Distribution des blocs sur les process
    n,m = np.shape(b2)
    if f == 1:
        nb = (n // bs) * (m // bs)
    else :
        ncol = int(m // (bs/f) - (f - 1))
        nrow = int(n // (bs/f) - (f - 1))
        # nb = (f*(n // bs) - (f-1)) * (f*(m // bs) - (f-1)) # Nombre de blocs dans l'image
        nb = nrow * ncol
    nd = nb // size # Nombre de blocs à traiter par process
    start =  rank * nd + rank * ((rank - 1) < (nb % size)) + (rank)*((rank - 1) >= (nb % size))
    end = (rank + 1) * nd + (rank + 1) * ((rank - 1) < (nb % size)) + (rank + 1)*((rank - 1) >= (nb % size))
    if (rank == size - 1): # Le dernier process va jusqu'au bout au cas où nb % size != 0
        end = nb
        nd = end - start

    # print("Nombre de blocs à traiter : " + str(nb))
    # print("rank : " + str(rank) + " | start : " + str(start) + " | end : " + str(end))


    tabx,taby,count = decoupageSuperpose(b2,b1,bs,f,start,end)

    mpi.COMM_WORLD.barrier()  # Attente de tous les processus

    c = mpi.COMM_WORLD.allreduce(sendobj = count, op = mpi.SUM) # additions des compteurs des blocs corrects de chaque processus

    # Regroupement des données calculées par chaque processus
    tabx = mpi.COMM_WORLD.allgather(tabx)
    taby = mpi.COMM_WORLD.allgather(taby)

    # Correction du format renvoyé par la fonction allgather
    # Passage de 2 liste à 1 matrice avec 2 lignes

    if rank == 0:
        accu = int(c / nb * 100)  # calcul de l'accuracy avec le nombre de blocs corrects
        print(str(c)+" blocs corrects/ "+str(nb) + " | " + str(accu) + "% de précision")
        tab = np.zeros((2,nb))
        for k in range(size):
            for i in range(len(tabx[k])):
                tab[0][k * len(tabx[0]) + i] = tabx[k][i]
                tab[1][k * len(taby[0]) + i] = taby[k][i]
        filename = "../decoup/afri/" + str(f) + "f_" + str(bs) + "bs" + "_"+str(axis0) + "sx_" + str(axis1) + "sy_" + str(seuil) + "seuil_" + str(accu) + "accu.npy"
        np.save(filename, tab)  # Enregistrement des résultats pour post traitement (visualisation et correction des données pour calcul du coef de Pearson)


"""
Programme actuellement en mode automatique
Convergence vers la configuration produisant le plus petit déplacement moyen
entre band2 et band1 ----> voir script convergence_test.py
"""

rank = mpi.COMM_WORLD.Get_rank() #  Numéro du process
size = mpi.COMM_WORLD.Get_size() # Nombre de process

# Bricolage pour récupérer les arguments de lancement du script en calcul parallèle

if rank == 0:
    # axis0 = 15 #input("Axis 0 : ")
    # axis1 = 15 #input("Axis 1 : ")
    seuil = 10 # Sert à rien ??
    # bs = 128
    print("Processing ...")
    # f = int(input("Entrez le facteur de recouvrement : "))
    # bs = int(input("Entrez le block size : "))
    f = 2
    bs = 128
    # axis0 = int(input("Entrez le shift de la band1 sur l'axis0 (vertical)"))
    # axis1 = int(input("Entrez le shift de la band1 sur l'axis1 (horizontal)"))
    band1 = sys.argv[1]
    band2 = sys.argv[2]
    axis0 = int(sys.argv[3])
    axis1 = int(sys.argv[4])
    data = [band1, band2, axis0, axis1, seuil, bs, f]
    t0 = time.time()
else:
    data = []

mpi.COMM_WORLD.barrier()
data = mpi.COMM_WORLD.bcast(data, root=0)
band1, band2, axis0, axis1, seuil, bs, f = data
main(band1, band2, axis0, axis1, bs, f, seuil)
mpi.COMM_WORLD.barrier()

if rank == 0:
    t1 = time.time()
    print("Temps d'exec : " + str((t1 - t0)//60) + "min" + str("%.2f" % ((t1 - t0)%60)))


# Notes : 
# * Seuil sert à rien (?)
# * 
