# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 16:10:40 2019

@author: Maximilien SOVICHE
"""


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
#import scipy as sc

import pywt
import pywt.data


from skimage import color

#img = mpimg.imread('route.jpg')
#imgtab = np.asarray(img)

#print(imgtab)
#imgplot = plt.imshow(imgtab)

#print("nombre de lignes :")
#print(len(imgtab))

#print("nombre de colonnes")
#print(len(imgtab[1]))

route = color.rgb2gray(mpimg.imread("C:/Users/Maximilien SOVICHE/Documents/3INSA/BE_TRAITEMENT_IMAGES/route.jpg"))
routab = np.asarray(route)
#
#print(routab)
#plt.figure(0)
#routplot = plt.imshow(routab, cmap = "gray")
#
#np.shape(routab)

# Ondelette de Haar

def Haar (t):
    return 0 + ((t>0)*(t<1/2)) - ((t<1)*(t>=1/2))

# Ondelette discrète
  
def Ondelette_D (n):
    N = 800
    Psi = np.zeros(N)
    #a_pj = []
    for j in range(N):
        Psi[n] = (1/(2**j))*Haar(n/2**j)
    return Psi

def Dilat_translat(t,j,k):
    return (1/np.sqrt(2**j))*Haar((t-k*2**j)/(2**j))
# 
###Transformée rapide        
#
##convolve 2D
#Ondelette = np.zeros((800,800))
#for n in range(800) :
#    Ondelette[n] = Ondelette_D(n);
#
#sc.signal.convolve2d(routab,Ondelette)

# QUESTION 11 

# tracé de l'ondelette de Haar 
    
t = np.linspace(-10, 10, 1000)
#
#plt.figure(1)
#plt.plot(t,Haar(t),'-b')
#plt.title("Tracé de l'ondelette de Haar sur l'intervalle [-4,4]")
#plt.grid()
#plt.xlabel("t")
#plt.ylabel("Haar(t)")
#plt.savefig("Tracé de l'ondelette de Haar sur l'intervalle [-4,4]")

# tracé des dilatations et translations de Haar avec différentes valeurs de j et de k

#val_j = [0, -0.00001,0.5,1,2,3,4,5]
#val_k = [0, -0.00001,0.5,1,2,3,4,5]
#
#for j in val_j:
#    for k in val_k:
#        plt.figure()
#        plt.plot(t,Dilat_translat(t,j,k),'-b')
#        plt.title("Haar_dilatation_translation_j = "+str(j)+"_k = "+str(k))
#        plt.savefig("Haar_dilatation_translation_j = "+str(j)+"_k = "+str(k)+"_.png")

# plus les valeurs de j et k sont proches de 0, plus on se rapproche de l'ondelette de Haar,
# comme prévu théoriquement.

# QUESTION 14 - application de la transformée en ondelette à un image - pywavelets
        
# Load image
        
#original = pywt.data.camera()

# Wavelet transform of image, and plot approximation and details
titles = ['Approximation', ' Horizontal detail',
          'Vertical detail', 'Diagonal detail']
coeffs2 = pywt.dwt2(routab, 'bior1.3')
LL, (LH, HL, HH) = coeffs2
fig = plt.figure(figsize=(12, 3))
for i, a in enumerate([LL, LH, HL, HH]):
    ax = fig.add_subplot(1, 4, i + 1)
    ax.imshow(a, interpolation="nearest", cmap=plt.cm.gray)
    ax.set_title(titles[i], fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])

fig.tight_layout()
plt.show()
#        










