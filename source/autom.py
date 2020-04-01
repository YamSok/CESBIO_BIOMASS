#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 17:03:26 2020

@author: yamsok
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm
from matplotlib.ticker import MultipleLocator
from matplotlib.pyplot import figure
import matplotlib.patches as patches
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300
import warnings
warnings.filterwarnings("ignore")

def choice():
    os.chdir('../decoup')
    rez = os.popen('ls -t').read()
    a = rez.split()
    rez2 = [str(i) + ' - ' + a[i] for i in range(len(a)) ]
    print("Liste des résulats (du plus récent au moins récent)")
    for i in range(len(a)):
        print(rez2[i])

    cin = input("Entre le numéro résulat à visualiser : ")

    return a[int(cin)]

def ExtractFeatures(filename):
    #test = "256bs_15sx_15sy_25r_15seuil_0count.png"
    liste = filename.split('_')
    features = ['f', 'bs', 'ax0', 'ax1', 'seuil', 'count']
    objectFeatures = {}
    for i in range(len(features)):
        objectFeatures[features[i]] = "".join([liste[i][s] for s in range(len(liste[i])) if liste[i][s].isdigit()])
    return objectFeatures

def visualizeSuperpose(ff,tab): # file features
#    if f == None:
#        bs = input("Block size ? :")
#        axis0 = input("Décalage selon l'axe 0 :")
#        axis1 = input("Décalage selon l'axe 1 :")
    f = int(ff["f"])
    bs = int(ff["bs"])
    ax0 = int(ff["ax0"])
    ax1 = int(ff["ax1"])
    seuil = int(ff["seuil"])
    count = int(ff["count"])
    b1 = np.load("../data/band1.npy")
    b2 = np.load("../data/band2.npy")
    b1, b2 = shiftSelec(b1,b2,ax0,ax1)
    r = 25
    n,m = np.shape(b2)
    nb = (f*(n // bs) - (f-1)) * (f*(m // bs) - (f-1)) # Nombre de blocs dans l'image
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
            #rect = patches.Rectangle( (int(j/f) * bs, int((i/f) * bs)) ,bs,bs,linewidth=l,edgecolor='k',facecolor='none')
            #rect2 = patches.Rectangle( (int(j/f) * bs, int((i/f) * bs)) ,bs,bs,linewidth=l,edgecolor='k',facecolor='none')

            arrow = patches.Arrow( int((j/f) * bs + bs // 2 ) , int((i/f) *bs + bs // 2) ,tab[0][i * (f * (m//bs) - (f-1)) + j],tab[1][i * (f * (m//bs) - (f-1)) + j], width=0.7,edgecolor=c,facecolor='none')
            ax[1].add_patch(arrow)
            #ax[0].add_patch(rect)
            #ax[1].add_patch(rect2)
    plt.tight_layout()

    #plt.savefig("b2")
    accu = round((count / nb * 100))
    plt.savefig("../results/"+ str(f) + "f_" + str(bs) + "bs_" + str(ax0) + "sx_" + str(ax0) + "sy_" + str(seuil) + "seuil_" + str(accu) + "accu..png")

    #plt.savefig("../results/sup_"+str(bs) + "x" + str(bs)+"_"+str(axis0) + "ax0_"+str(axis1)+"ax1_"+str(r)+"r_"+str(seuil)+"seuil_"+str(count)+ "count.png")
    # plt.savefig("results/"+str(bs) + "x" + str(bs)+"_"+str(axis0) + "ax0_"+str(axis1)+"ax1_"+str(r)+"r_"+str(seuil)+"seuil_"+str(count)+ "count.png")

# SELECTION DE DIMENSIONS EN PUISSANCE DE 2 | SHIFT
# def shiftSelec(im1,im2,axis0,axis1):
#     band2_s = np.roll(np.roll(im2,axis0,axis=0),axis1,axis=1)
#     #b2 = selection(band2_s,115,1651,30,1054)
#     b2 = selection(10*np.log(band2_s),115,1651,30,1054)
#     b1 = selection(im1,115,1651,30,1054)
#     return b1,b2

def shiftSelec(im1,im2,axis0,axis1):
    band2_s = np.roll(np.roll(im2,axis0,axis=0),axis1,axis=1)
    #b2 = selection(band2_s,115,1651,30,1054)
    b2 = selection(10*np.log(band2_s),115 + 2 * 256,1651,30 + 256,1054)
    b1 = selection(im1,115 + 2 *256,1651,30 + 256 ,1054)
    return b1,b2


def selection(img,x0,x1,y0,y1):
    h = abs(x0 - x1)
    w = abs(y0 - y1)
    return img[x0:x0+h,y0:y0+w]
c = choice()
ff = ExtractFeatures(c)
print(ff)
print(c)
tab = np.load('../decoup/' + c)
visualizeSuperpose(ff,tab)
#npy = choice()
