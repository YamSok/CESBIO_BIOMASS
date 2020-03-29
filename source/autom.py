#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 17:03:26 2020

@author: yamsok
"""

import os
import numpy as np
        
def choice():
    os.chdir('../decoup')
    rez = os.popen('ls -t').read()
    a = rez.split()
    rez2 = [str(i) + ' - ' + a[i] for i in range(len(a)) ]
    print("Liste des résulats (du plus récent au moins récent)")
    for i in range(len(a)):
        print(rez2[i])
    
    cin = input("Entre le numéro résulat à visuliser : ")
    
    return a[int(cin)]

def ExtractFeatures(filename):
    #test = "256bs_15sx_15sy_25r_15seuil_0count.png"
    liste = filename.split('_')   
    features = ['f', 'bs', 'ax0', 'ax1', 'seuil', 'count']
    objectFeatures = {}
    for i in range(len(features)):
        objectFeatures[features[i]] = "".join([liste[i][s] for s in range(len(liste[i])) if liste[i][s].isdigit()])
    return objectFeatures
c = choice()
ff = ExtractFeatures(c)
print(ff)
#npy = choice()

#visualize(npy)
# b1,b2,tab,bs,axis0,axis1,r,f,seuil
def visualizeSuperpose(ff): # file features
#    if f == None:
#        bs = input("Block size ? :")
#        axis0 = input("Décalage selon l'axe 0 :")
#        axis1 = input("Décalage selon l'axe 1 :")
    bs = ff["bs"]
    ax0 = ff["ax0"]
    ax1 = ff["ax1"]
    seuil = ff["seuil"]
    count = ff["count"]
    
    n,m = np.shape(b2)
    fig,ax = plt.subplots(1,2,figsize=(10,10))
    ax[0].imshow(b2)
    ax[1].imshow(b1)
    count = 0
    for i in range(f * (n//bs) - (f-1)) :
        for j in range(f * (m//bs) - (f-1)) :

            if np.sqrt(tab[0][i * (f * (m//bs) - (f-1)) + j]**2 + tab[1][i *(f * (m//bs) - (f-1)) + j]**2) == r :
                c =  'k'
                l = 1
            elif np.sqrt(tab[0][i * (f * (m//bs) - (f-1)) + j]**2 + tab[1][i * (f * (m//bs) - (f-1)) + j]**2)  <= seuil:
                c = 'm'
                l = 1
                count +=1
            else:
                c = 'r'
                l = 1
            rect = patches.Rectangle( (int(j/f) * bs, int((i/f) * bs)) ,bs,bs,linewidth=l,edgecolor='k',facecolor='none')
            rect2 = patches.Rectangle( (int(j/f) * bs, int((i/f) * bs)) ,bs,bs,linewidth=l,edgecolor='k',facecolor='none')

            arrow = patches.Arrow( int((j/f) * bs + bs // 2 ) , int((i/f) *bs + bs // 2) ,tab[0][i * (f * (m//bs) - (f-1)) + j],tab[1][i * (f * (m//bs) - (f-1)) + j], width=0.7,edgecolor=c,facecolor='none')
            ax[1].add_patch(arrow)
            ax[0].add_patch(rect)
            ax[1].add_patch(rect2)
    plt.tight_layout()

    #plt.savefig("b2")
    accu = (count / nb * 100)
    plt.savefig("../results/"+ str(f) + "f_" + str(bs) + "bs_" + str(axis0) + "ax1_" + str(axis1) + "ax1_" + str(seuil) + "seuil_" + str(accu) + "accu..png")
    
    #plt.savefig("../results/sup_"+str(bs) + "x" + str(bs)+"_"+str(axis0) + "ax0_"+str(axis1)+"ax1_"+str(r)+"r_"+str(seuil)+"seuil_"+str(count)+ "count.png")
    # plt.savefig("results/"+str(bs) + "x" + str(bs)+"_"+str(axis0) + "ax0_"+str(axis1)+"ax1_"+str(r)+"r_"+str(seuil)+"seuil_"+str(count)+ "count.png")
