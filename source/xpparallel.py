import mpi4py.MPI as mpi
import numpy as np
from scipy import signal
from scipy import misc
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm
from matplotlib.ticker import MultipleLocator
from matplotlib.pyplot import figure
import matplotlib.patches as patches
import matplotlib as mpl
import time
mpl.rcParams['figure.dpi'] = 300
import warnings
warnings.filterwarnings("ignore")

from datetime import datetime as dt

def afficherlol():
    a = np.array([1,2])
    print("lol")

# SELECTION DE DIMENSIONS EN PUISSANCE DE 2 | SHIFT
def shiftSelec(im1,im2,axis0,axis1):
    band2_s = np.roll(np.roll(im2,axis0,axis=0),axis1,axis=1)
    #b2 = selection(band2_s,115,1651,30,1054)
    b2 = selection(10*np.log(band2_s),115,1651,30,1054)
    b1 = selection(im1,115,1651,30,1054)
    return b1,b2

def selection(img,x0,x1,y0,y1):
    h = abs(x0 - x1)
    w = abs(y0 - y1)
    return img[x0:x0+h,y0:y0+w]

# AFFICHAGE DE 4 SUBPLOTS | ( original, tamplate, cross correlation, zoom de cross correlation )
def displayImg(original,template,corr,x,y,r):
    n,m = np.shape(original)
    fig, (ax_orig, ax_template, ax_corr, ax_corr2) = plt.subplots(1, 4,figsize=(10, 20))
    ax_orig.imshow(original)
    ax_orig.set_title('Original')

    ax_template.imshow(template)
    ax_template.set_title('Template')

    ax_corr.imshow(corr)
    nn , mm = np.shape(corr)
    nc = nn // 2
    mc = mm // 2
    rect = patches.Rectangle((nc - r,mc - r),2 * r,2 * r,linewidth=1,edgecolor='r',facecolor='none')
    ax_corr.add_patch(rect)
    ax_corr.set_title('Cross-correlation')

    rect2 = patches.Rectangle((nc - r,mc - r),2 * r,2 * r,linewidth=1,edgecolor='r',facecolor='none')
    ax_orig.add_patch(rect2)

    ax_orig.plot(x, y, 'ro')
    ax_orig.plot(n/2,n/2, 'rx')
    #ax_template.plot(x, y, 'ro')

    ax_corr2.imshow(corr[nc - r:nc + r, mc - r:mc + r])
    ax_corr2.set_title('Cross-correlation [' + str(r) + 'x' + str(r) + "]")
    ax_corr2.plot(x - nc + r, y - mc + r, 'ro')
    fig.show()

    print("(x,y) = ("+str(x)+','+str(y)+')' )

# CALCUL DE LA CORRELATION CROISEE ENTRE original ET template
def decalageBloc(original, template, r):
    orig = np.copy(original)  #prévenir pbs de pointeurs python
    temp = np.copy(template)

    orig -= original.mean()
    orig = orig/np.std(orig)
    temp -= template.mean()
    temp = temp/np.std(temp)

    corr = signal.correlate2d(orig, temp, boundary='symm', mode='same')
    n,m = np.shape(corr)
    nc = n // 2
    mc = m // 2
    y, x = np.unravel_index(np.argmax(corr[nc - r:nc + r, mc - r:mc + r]), corr[nc - r:nc + r, mc - r:mc + r].shape)  # find the match
    #print("Shape: " + str(np.shape(corr[nc - r:nc + r, mc - r:mc + r])))
    y = y + mc - r
    x = x + nc - r

    return orig, temp, corr, x, y

# APPLICATION CORRELATION A UNE IMAGE DECOUPEE EN BLOCS
def decoupage(b2,b1,bs,r,start,end):
    n,m = np.shape(b2)
    # VARIABLES
    tabx=[] # stockage décalage x
    taby=[] # stockage décalage y
    count = 0 # compte des blocs corrects

    for i in range(n//bs):
    #i = 0 # pour les tests
        for j in range(m//bs):
            if i * (m//bs) + j  >= start and i * (m//bs) + j < end:
                #print(i * (m//bs) + j)
                #print("rank : " + str(rank) + " | bloc #" + str(i * (m//bs) + j))
                band2Block = np.copy(b2[i*bs:(i+1)*bs,j*bs:(j+1)*bs])
                band1Block = np.copy(b1[i*bs:(i+1)*bs,j*bs:(j+1)*bs])
                templateBlock = np.copy(band1Block[5:bs-5,5:bs-5])
                orig,temp,corr,x,y = decalageBloc(band2Block,templateBlock,r)
                xm = x-bs/2
                ym = y-bs/2
                tabx.append(xm)
                taby.append(ym)
                if np.sqrt(xm**2 + ym**2) < 25 :
                    count += 1
                # tabx.append(i * (m//bs) + j)
    #print("rank : " + str(rank) + " | count : " + str(count))
    return tabx,taby,count

# APPLICATION CORRELATION CROISEE SUR DES BLOCS SUPERPOSES
def decoupageSuperpose(b2,b1,bs,r,f,start,end): # f = factor
    n,m = np.shape(b2)
    # VARIABLES
    tabx=[] # stockage décalage x
    taby=[] # stockage décalage y
    count = 0 # compte des blocs corrects

    for i in range(f * (n//bs) - 1):
    #i = 0 # pour les tests
        for j in range(f * (m//bs)-1):
            if i * (f * (m // bs) - 1) + j  >= start and i * (f * (m // bs) - 1) + j < end:
                band2Block = np.copy(b2[int((i / f) * bs) : int((i / f) * bs + bs) , int((j / f) * bs) : int((j / f) * bs + bs)])
                band1Block = np.copy(b1[int((i / f) * bs) : int((i / f) * bs + bs) , int((j / f) * bs) : int((j / f) * bs + bs)])
                templateBlock = np.copy(band1Block[5:bs-5,5:bs-5])
                orig,temp,corr,x,y = decalageBloc(band2Block,templateBlock,r)
                xm = x-bs/2
                ym = y-bs/2
                tabx.append(xm)
                taby.append(ym)
                if np.sqrt(xm**2 + ym**2) < 25 :
                    count += 1
    return tabx,taby,count

# AFFICHAGE DES RESULTATS DU DECOUPAGE
def visualize(b1,b2,tabx,taby,bs,axis0,axis1,r,seuil):
    n,m = np.shape(b2)
    fig,ax = plt.subplots(1,2,figsize=(10,10))
    ax[0].imshow(b2)
    ax[1].imshow(b1)
    count = 0
    for i in range(n//bs) :
        for j in range(m//bs) :
            if np.sqrt(tab[0][i * (m//bs) + j]**2 + tab[1][i * (m//bs) + j]**2) == r :
                c =  'k'
                l = 2
            elif np.sqrt(tab[0][i * (m//bs) + j]**2 + tab[1][i * (m//bs) + j]**2)  <= seuil:
                c = 'm'
                l = 1
                count +=1
            else:
                c = 'r'
                l = 2

            rect = patches.Rectangle((j*bs,i*bs),bs,bs,linewidth=l,edgecolor=c,facecolor='none')
            rect2 = patches.Rectangle((j*bs,i*bs),bs,bs,linewidth=l,edgecolor=c,facecolor='none')
            arrow = patches.Arrow(j*bs + bs//2,i*bs + bs//2 ,tabx[i * (m//bs) + j],taby[i * (m//bs) + j], width=0.7,edgecolor='r',facecolor='none')
            ax[1].add_patch(arrow)
            ax[0].add_patch(rect)
            ax[1].add_patch(rect2)
    plt.tight_layout()
    plt.savefig("results/"+str(bs) + "x" + str(bs)+"_"+str(axis0) + "ax0_"+str(axis1)+"ax1_"+str(r)+"r_"+str(seuil)+"seuil_"+str(count)+ "count.png")
    print(str(count)+" blocs corrects/ "+str((n//bs)*(m//bs)))


#AFFICHAGE DES RESULTATS DU DECOUPAGE SUPERPOSE
def visualizeSuperpose(b1,b2,tab,bs,axis0,axis1,r,f,seuil):
    n,m = np.shape(b2)
    fig,ax = plt.subplots(1,2,figsize=(10,10))
    ax[0].imshow(b2)
    ax[1].imshow(b1)
    count = 0
    print(f * (n//bs) - 1)
    print(f * (m//bs) - 1)
    for i in range(f * (n//bs)-1) :
        for j in range(f * (m//bs)-1) :
            print('bloc # ' + str(i * (f * (m // bs) - 1) + j))

            if np.sqrt(tab[0][i * (f * (m // bs) - 1) + j]**2 + tab[1][i *(f * (m // bs) - 1) + j]**2) == r :
                c =  'k'
                l = 2
            elif np.sqrt(tab[0][i * (f * (m // bs) - 1) + j]**2 + tab[1][i * (f * (m // bs) - 1) + j]**2)  <= seuil:
                c = 'm'
                l = 1
                count +=1
            else:
                c = 'r'
                l = 2
            # rect = patches.Rectangle((j*bs,i*bs),bs,bs,linewidth=l,edgecolor=c,facecolor='none')
            # rect2 = patches.Rectangle((j*bs,i*bs),bs,bs,linewidth=l,edgecolor=c,facecolor='none')
            arrow = patches.Arrow( int((j/f) * bs + bs // 2 ) , int((i/f) *bs + bs // 2) ,tab[0][i * (f * (m // bs) - 1) + j],tab[1][i * (f * (m // bs) - 1) + j], width=0.7,edgecolor=c,facecolor='none')
            ax[1].add_patch(arrow)
            # ax[0].add_patch(rect)
            # ax[1].add_patch(rect2)
    plt.tight_layout()
    plt.savefig("verif2")
    #plt.savefig("../results/sup_"+str(bs) + "x" + str(bs)+"_"+str(axis0) + "ax0_"+str(axis1)+"ax1_"+str(r)+"r_"+str(seuil)+"seuil_"+str(count)+ "count.png")
    # plt.savefig("results/"+str(bs) + "x" + str(bs)+"_"+str(axis0) + "ax0_"+str(axis1)+"ax1_"+str(r)+"r_"+str(seuil)+"seuil_"+str(count)+ "count.png")
    print(str(count)+" blocs corrects/ "+str((n//bs)*(m//bs)))

# DONNE LE NOMBRE DE BLOCS AVEC DECALGE < SEUIL
def countCorrect(tab,seuil,nb, verbose=False):
    count = 0
    dist = []
    for i in range(nb):
        distance = np.sqrt(tab[0][i]**2 + tab[1][i]**2)
        if verbose :
            print("Décalage du block " +str(i)+ " : %.2f" % (np.sqrt(tab[0][i]**2 + tab[1][i]**2)*5) + " m.")
        if distance < seuil:  #distance inférieure à 50 px (c'est beaucoup)
            count +=1
        dist.append(distance)
    if verbose:
        print(str(count)+" corrects sur "+ str(nb) + " avec une marge de " + str(seuil * 5) +" m.")
    print("Moyenne des déplacements : " + str(np.mean(distance * 5)))
    return count, np.mean(distance*5)

# PROGRAMME PRINCIPAL
def main(axis0,axis1,bs,f,seuil):
    band1 = np.load("../data/band1.npy")
    band2 = np.load("../data/band2.npy")
    b1,b2 = shiftSelec(band1,band2,axis0,axis1)
    # Block size
    r = 25
    # Distribution des blocs sur les processes
    n,m = np.shape(b2)
    nb = (2*(n // bs)-1) * (2*(m // bs)-1) # Nombre de blocs dans l'image
    nd = nb // size # Nombre de blocs à traiter par process
    start =  rank * nd
    end = (rank + 1) * nd
    if (rank == size - 1): # Le dernier process va jusqu'au bout au cas où nb % size != 0
        end = nb
        nd = end - start

    print("Nombre de blocs à traiter : " + str(nb))
    print("rank : " + str(rank) + " | start : " + str(start) + " | end : " + str(end))
    tabx,taby,count = decoupageSuperpose(b2,b1,bs,r,f,start,end)
    #print(str(count)+" BLOCS CORRECTS")
    mpi.COMM_WORLD.barrier()

    #c = mpi.COMM_WORLD.allreduce(sendobj = count, op = mpi.SUM)
    tabx = mpi.COMM_WORLD.allgather(tabx)
    taby = mpi.COMM_WORLD.allgather(taby)

    if rank == 0:
        tab = np.zeros((2,nb))
        # tx = np.zeros(nb)
        # ty = np.zeros(nb)
        for k in range(size):
            for i in range(len(tabx[0])):
                tab[0][k * len(tabx[0]) + i] = tabx[k][i]
                tab[1][k * len(taby[0]) + i] = tabx[k][i]
                #tx[k * len(tabx[0]) + i] = tabx[k][i]
                #ty[k * len(taby[0]) + i] = taby[k][i]
        np.save("../decoup/tab_superpose2.npy", tab)
        #tab = np.load("../decoup/tab.npy")
        visualizeSuperpose(b1,b2,tab,bs,axis0,axis1,r,f,seuil)
rank = mpi.COMM_WORLD.Get_rank() #  Numéro du process
size = mpi.COMM_WORLD.Get_size() # Nombre de process"
#print("rank : " + str(rank) + " | size : " + str(size))
if rank == 0:
    t0 = time.time()

axis0 = 15
axis1 = 15
seuil = 15
bs = 256
f = 2
r = 25
# band1 = np.load("../data/band1.npy")
# band2 = np.load("../data/band2.npy")
# b1,b2 = shiftSelec(band1,band2,axis0,axis1)
# tab = np.load("../decoup/tab_superpose.npy")
    #print(np.shape(tab))
    # visualizeSuperpose(b1,b2,tab,bs,axis0,axis1,r,f,seuil)
main(axis0,axis1,bs,f,seuil)
mpi.COMM_WORLD.barrier()

if rank == 0:
    t1 = time.time()
    print("Temps d'exec : " + str((t1 - t0)//60) + "min" + str((t1 - t0)%60))
