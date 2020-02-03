
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy as sc
from scipy import fftpack
import time

from skimage import color

#img = mpimg.imread("/home/soviche/3MICPROJET/voitures/image_0001.jpg")
#imgtab = np.asarray(img)

#print(imgtab)
#imgplot = plt.imshow(imgtab)

#print("nombre de lignes :")
#print(len(imgtab))

#print("nombre de colonnes")
#print(len(imgtab[1]))

route = color.rgb2gray(mpimg.imread("/home/soviche/3MICPROJET/route.jpg"))
routab = np.asarray(route)

print(routab)
plt.figure(0)
routplot = plt.imshow(routab, cmap = "gray")

np.shape(routab)

 #img = color.rgb2gray(io.imread('image.png'))
dct_rout = sc.fftpack.dct(routab)

np.shape(dct_rout)

plt.figure(1)
plt.imshow(dct_rout)

idct_rout = sc.fftpack.idct(dct_rout)

plt.figure(2)
plt.imshow(idct_rout,cmap = "gray")

plt.show()

v=dct_rout.reshape(800*800)
plt.plot(np.sort(v))
plt.show()

#On met la matrice en norme
# =============================================================================
# mat = abs(dct_rout)
# plt.show()
# plt.plot(np.sort(mat.reshape(800*800)))
# 
# print(np.quantile(mat,0.8))
# 
# =============================================================================
#for lin in range(len(dct_rout)):
 #   for col in range(len(dct_rout[1])):
  #      if abs(dct_rout[lin][col]) < 1.5:
   #         dct_rout[lin][col] = 0
    
   #np.quantile(abs(dct_rout),0.8)
print(np.quantile(abs(dct_rout),0.8))
dct_rout[abs(dct_rout) < np.quantile(abs(dct_rout),0.8)] = 0
idct_rout = sc.fftpack.idct(dct_rout)
plt.figure(3)
plt.imshow(idct_rout,cmap = "gray")



# =============================================================================
#Question 6
# =============================================================================
tmps1 = time.clock()


m,n =np.shape(route)
c= round(m/8)

route_bloc = np.zeros(np.shape(route))

T = [
     [16, 11, 10, 16, 24, 40, 51, 61],
     [12, 12, 14, 19, 26, 58, 60, 55],
     [14, 13, 16, 24, 40, 57, 69, 56],
     [14, 17, 22, 29, 51, 87, 80, 62],
     [18, 22, 37, 56, 68, 109, 103, 77],
     [24, 35, 55, 64, 81, 104, 113, 92],
     [49, 64, 78, 87, 103, 121, 120, 101],
     [72, 92, 95, 98, 112, 100, 103, 99]
     ]

Q = 50
Tq = T


if Q > 50:
    Tq = round(50*T/Q)
elif Q < 50:
    Tq = round(T*(100-Q)/50)



for i in range(c):
    for j in range(c):        
        s_route = routab[i*8:(i+1)*8, j*8:(j+1)*8]
        s_route = sc.fftpack.dct(s_route)
        s_route = s_route/Tq
        
        route_bloc[i*8:(i+1)*8, j*8:(j+1)*8] = s_route

plt.figure(4)
plt.imshow(route_bloc)

idct_route_bloc = np.zeros(np.shape(route))
for i in range(c):
    for j in range(c):
        s_route_inv = route_bloc[i*8:(i+1)*8, j*8:(j+1)*8]
        s_route_inv = s_route_inv*Tq
        s_route_inv = sc.fftpack.idct(s_route_inv)
        
        idct_route_bloc[i*8:(i+1)*8, j*8:(j+1)*8] = s_route_inv

plt.figure(5)
plt.imshow(idct_route_bloc, cmap = "gray")

tmps2 = time.clock()
temps_exec = tmps2-tmps1
print("temps d'éxécution = ",temps_exec) 

# on transforme chaque bloc de 8x8 en une liste grâce à la fonction qui suit 

def serpentin(T):
    list = []
    monter=False

    maxi = len(T)-1
    maxj = len(T[0])-1

    i=0
    j=0
    while not(i==maxi and j==maxj) :
        list += [T[i][j]]
        
        if (j==0 or i==maxi):
            monter = True    
        elif (i==0 or j==maxj):
            monter = False
        if (i==0 or i==maxi) and (j % 2 == 0):
            j = j+1
        elif (j==0 or j==maxj) and (i % 2 == 1):
            i = i+1
        else:
            if monter == True:
                i=i-1
                j=j+1
            elif monter == False:
                i=i+1
                j=j-1
                                    
    list += [T[maxi][maxj]]
    return list

l = serpentin(T)
print(l)

# on crée deux listes :
# - la première contient les premiers éléments de chaque tableau 8x8
# - la deuxième contient le reste des éléments 

List_premiers = []
List_reste = []

for i in range(c):
    for j in range(c): 
        List_premiers += [routab[i*8:(i+1)*8, j*8:(j+1)*8][1,1]]
        List_reste += [serpentin(routab[i*8:(i+1)*8, j*8:(j+1)*8][1:64])]

print(List_premiers)
print(List_reste)


# on effectue le codage RLE en 0:
# chaque série de 0 est remplacée par :
# un 0 puis le nombre d'occurences de 0 



































