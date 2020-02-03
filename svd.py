import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy.linalg as npl

from skimage import data
from skimage import color



# IMPLEMENTATION SVD
def SVD(A):
    # Arguments en entrée:
    # A : la matrice dont on veut la décomposition
    # Arguments en sortie:
    # U,S,V : U et V étant les matrices orthogonales
    # des vecteurs singuliers à gauche/droite et S
    # est une matrice de même taille que A dont la 
    # diagonale consistuée des valeurs singulières
    # par ordre décroissant
    
    #nombre de lignes de la matrice A
    m = np.shape(A)[0]
    
    #calcul du rang, des val. propres et des vect. propres 
    r = npl.matrix_rank(A)
    v,V = npl.eig(np.dot((A.T),A))
    
    #on classe les éléments propres par ordre décroissant
    ind = np.argsort(-np.real(v))
    v=v[ind]
    V=V[:,ind]
    
    #on ne garde que les r premiers éléments (on élimine les éléments propres négligeables)
    v = v[:r]
    V = V[:,:r]
    S = np.diag(np.sqrt(v))
    
    #construction de U
    
    U = np.zeros((m,r))
    
    for i in range(r):
        U[:,i] = (1/S[i,i])*(np.dot(A,V[:,i]))
    
    return U,S,V

# AFFICHAGE IMAGE 
A = np.array(data.camera(),float)
plt.imshow(A,cmap = 'gray')
plt.show()

# SVD : REDUCTION DE RANG DIRECTE
def SVD_r(A,r):
    U,S,V = SVD(A)
    Ar = np.dot(U[:,:r],S[:r,:r]).dot(V[:,:r].T)
    return Ar

A = np.array(data.camera(),float)
Ar = SVD_r(A,512)

plt.imshow(A,cmap = 'gray')
plt.show()

plt.imshow(Ar,cmap = 'gray')
plt.show()

# le cout de calcul = (nb ligne + nb collones)*(nb d'imagettes qu'on somme)

# SVD : REDUCTION DE RANG SUR DES SOUS MATRICES 

# on suppose ici que la matrice de base A est carrée, et que n est pair.

def SVD_sqr(A,r,n):
    
    l = np.shape(A)[0]
    #on a donc n² matrices de taille (l/n,l/n)
    
    t = int(l/n) #taille de la sous matrice 
    Asqr = np.zeros((np.shape(A)))
    
    for i in range(n):
        for j in range(n):
            Sa = A[i*t:(i+1)*t,j*t:(j+1)*t]
            Asqr[i*t:(i+1)*t,j*t:(j+1)*t] = SVD_r(Sa,r)
    return Asqr

A = np.array(data.camera(),float)
Asq = SVD_sqr(A,8,8)

plt.imshow(A,cmap = 'gray')
plt.show()

plt.imshow(Asq,cmap = 'gray')
plt.show()

# LINEARISATION DE MATRICE 

def mat_linear(A):
    n = np.shape(A)[0]
    #m = np.shape(A)[1]
    A_linear = []
    for i in range(n):
        A_linear += A[i]
    return A_linear;

A = [[1,2,3],
     [6,5,4],
     [9,8,7]]

print(mat_linear(A))
        
baby_duck = color.rgb2gray(mpimg.imread("C:/Users/Maximilien SOVICHE/Documents/3INSA/BE_TRAITEMENT_IMAGES/images/babyduck.png"))
bear = color.rgb2gray(mpimg.imread("C:/Users/Maximilien SOVICHE/Documents/3INSA/BE_TRAITEMENT_IMAGES/images/bear.png"))
minecraft = color.rgb2gray(mpimg.imread("C:/Users/Maximilien SOVICHE/Documents/3INSA/BE_TRAITEMENT_IMAGES/images/minecraft.png"))
quad = color.rgb2gray(mpimg.imread("C:/Users/Maximilien SOVICHE/Documents/3INSA/BE_TRAITEMENT_IMAGES/images/quad.png"))

babyduck_tab = np.array(baby_duck, float)
bear_tab = np.array(bear, float)
minecraft_tab = np.array(minecraft, float)
quad_tab = np.array(quad, float)

plt.figure(0)
plt.imshow(babyduck_tab, cmap = 'gray')
plt.figure(1)
plt.imshow(bear_tab, cmap = 'gray')
plt.figure(2)
plt.imshow(minecraft_tab, cmap = 'gray')
plt.figure(3)
plt.imshow(quad_tab, cmap = 'gray')










