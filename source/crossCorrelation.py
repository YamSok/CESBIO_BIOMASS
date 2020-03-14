#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 14:30:12 2020

@author: Daxter
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm
from matplotlib.ticker import MultipleLocator
from matplotlib.pyplot import figure
import matplotlib.patches as patches

from scipy import signal
from scipy import misc

#import cv2   #Librairie à installer

import warnings
warnings.filterwarnings("ignore")



#Fonctions de pré traiement des images (nécéssite le package OpenCV 2)

def gaussianFilter(im1,factor):
    kernel = np.ones((factor,factor),np.float32)/(factor**2)
    target = cv2.filter2D(im1,-1,kernel)
    return target

def sharpenFilter(im):
    kernel = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
    im = cv2.filter2D(im, -1, kernel)
    return im


def sharpenFilter2(im):
    kernel = np.array([[1,0,-1], [0,0,0], [-1,0,1]])
    im = cv2.filter2D(im, -1, kernel)
    return im

def selection(img,x0,x1,y0,y1):
    h = abs(x0 - x1)
    w = abs(y0 - y1)
    fig,ax = plt.subplots(1, 2,figsize=(16, 8))
    im1 = ax[0].imshow(img)
    rect = patches.Rectangle((y0,x0),w,h,linewidth=1,edgecolor='r',facecolor='none')
    ax[0].add_patch(rect)
    ax[0].set_title("Image originale")
    fig.colorbar(im1,ax=ax[0])
    im2 = ax[1].imshow(img[x0:x0+h,y0:y0+w])
    ax[1].set_title("Selection")
    fig.colorbar(im2,ax=ax[1])

    plt.show()
    return img[x0:x0+h,y0:y0+w]





# Fonctions principales de calcul des corrélations croisées

def displayImg(original,template,corr,x,y):
    n,m = np.shape(original)
    fig, (ax_orig, ax_template, ax_corr) = plt.subplots(1, 3,figsize=(10, 20))
    ax_orig.imshow(original)
    ax_orig.set_title('Original')
    
    ax_template.imshow(template)
    ax_template.set_title('Template')
    
    ax_corr.imshow(corr)
    ax_corr.set_title('Cross-correlation')
    
    ax_orig.plot(x, y, 'ro')
    ax_orig.plot(n/2,n/2, 'rx')
    #ax_template.plot(x, y, 'ro')
    fig.show()
    
    print("(x,y) = ("+str(x)+','+str(y)+')' )

def decalageBloc(original, template):
    orig = np.copy(original)  #prévenir pbs de pointeurs python
    temp = np.copy(template)

    orig -= original.mean()
    orig = orig/np.std(orig)
    temp -= template.mean()
    temp = temp/np.std(temp)

    corr = signal.correlate2d(orig, temp, boundary='symm', mode='same')
    y, x = np.unravel_index(np.argmax(corr), corr.shape)  # find the match
    return orig, temp, corr, x, y

def countCorrect(tabx,taby,seuil, verbose=False):
    count = 0
    dist = []
    for i in range(len(tabx)):
        distance = np.sqrt(tabx[i]**2 + taby[i]**2)
        if verbose :
            print("Décalage du block " +str(i)+ " : %.2f" % (np.sqrt(tabx[i]**2 + taby[i]**2)*5) + " m.")
        if distance < seuil:  #distance inférieure à 50 px (c'est beaucoup)
            count +=1
        dist.append(distance)    
    if verbose:
        print(str(count)+" corrects sur "+ str(len(tabx)) + " avec une marge de " + str(seuil * 5) +" m.")
    return count, np.mean(distance*5)

def decoupage(b2,b1,bs,v=False): #bs= blocksize
    n,m = np.shape(b2)
    fig,ax = plt.subplots(1,2,figsize=(10,10))
    ax[0].imshow(b2)
    ax[1].imshow(b1)
    tabx=[]
    taby=[]
    count = 0
    for i in range(n//bs):
    #i = 0 # pour les tests
        for j in range(m//bs):
            band2Block = np.copy(b2[i*bs:(i+1)*bs,j*bs:(j+1)*bs])
            band1Block = np.copy(b1[i*bs:(i+1)*bs,j*bs:(j+1)*bs])
            #print("bloc " + str(i*(m//bs) + j) + " | Var : " + "%.2f" % np.std(band1Block))
            templateBlock = np.copy(band1Block[10:bs-10,10:bs-10])
            orig,temp,corr,x,y = decalageBloc(band2Block,templateBlock)
            xm = x-bs/2
            ym = y-bs/2
            tabx.append(xm)
            taby.append(ym)
            if np.sqrt(xm**2 + ym**2) > 25 : 
                rect = patches.Rectangle((j*bs,i*bs),bs,bs,linewidth=2,edgecolor='r',facecolor='none')
                rect2 = patches.Rectangle((j*bs,i*bs),bs,bs,linewidth=2,edgecolor='r',facecolor='none')
            else : 
                count += 1
                rect = patches.Rectangle((j*bs,i*bs),bs,bs,linewidth=1,edgecolor='m',facecolor='none')
                rect2 = patches.Rectangle((j*bs,i*bs),bs,bs,linewidth=1,edgecolor='m',facecolor='none')

            arrow = patches.Arrow(j*bs + bs//2,i*bs + bs//2 ,xm,ym, width=1.0,edgecolor='r',facecolor='none')    
            ax[1].add_patch(arrow)
            ax[0].add_patch(rect)
            ax[1].add_patch(rect2)

            if v:
                print('itération : '+str(j))
                displayImg(orig,temp,corr,x,y)
    
    plt.savefig("207.png")
    plt.show()
    print(str(count)+" blocs corrects/ "+str((n//bs)*(m//bs)))
    return tabx,taby


#problème : les bords ... decalageBlock renvoit des 'orig' et des 'corr' blancs
	


def main():
		# Importation et affichage des données
	band1 = np.loadtxt("band1.txt")
	band2 = np.loadtxt("band2.txt")
	
	fig,ax = plt.subplots(1,2, figsize=(15,8))
	
	im1 = ax[0].imshow(band1, vmin = 0, vmax = 5)
	ax[0].set_title("BAND 1 - Relevé topographique ")
	fig.colorbar(im1,ax=ax[0])
	
	im2 = ax[1].imshow(10*np.log(band2),vmin=-40,vmax=0)
	ax[1].set_title("BAND 2 - Image satellite")
	fig.colorbar(im2,ax=ax[1])
	
	plt.tight_layout()
	plt.savefig("images_radar.png")
	plt.show()
	b2 = selection(10*np.log(band2),115,1840,10,1045)  #taille choisie pour avoir peu de bords blancs et divisible par 115
	b1 = selection(band1,115,1840,10,1045)
	t1,t2 = decoupage(b2,b1,207)
	
	
main()
	
	