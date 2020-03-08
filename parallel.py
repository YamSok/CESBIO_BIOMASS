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
mpl.rcParams['figure.dpi'] = 300
import warnings
warnings.filterwarnings("ignore")

def shiftSelec(im1,im2,axis0,axis1):
    band2_s = np.roll(np.roll(im2,axis0,axis=0),axis1,axis=1)
    b2 = selection(10*np.log(band2_s),115,1840,30,1065)
    b1 = selection(im1,115,1840,30,1065)
    return b1,b2

def selection(img,x0,x1,y0,y1):
    h = abs(x0 - x1)
    w = abs(y0 - y1)
    return img[x0:x0+h,y0:y0+w]

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

def decoupage(b2,b1,bs,start,end):
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
                orig,temp,corr,x,y = decalageBloc(band2Block,templateBlock)
                xm = x-bs/2
                ym = y-bs/2
                tabx.append(xm)
                taby.append(ym)
                if np.sqrt(xm**2 + ym**2) < 25 :
                    count += 1
                # tabx.append(i * (m//bs) + j)
    #print("rank : " + str(rank) + " | count : " + str(count))
    return tabx,taby,count

def visualize(b1,b2,tabx,taby,bs):
    n,m = np.shape(b2)
    fig,ax = plt.subplots(1,2,figsize=(10,10))
    ax[0].imshow(b2)
    ax[1].imshow(b1)
    count = 0
    for i in range(n//bs) :
        for j in range(m//bs) :
            if np.sqrt(tabx[i * (m//bs) + j]**2 + taby[i * (m//bs) + j]**2) > 25 :
                rect = patches.Rectangle((j*bs,i*bs),bs,bs,linewidth=2,edgecolor='r',facecolor='none')
                rect2 = patches.Rectangle((j*bs,i*bs),bs,bs,linewidth=2,edgecolor='r',facecolor='none')
            else :
                count += 1
                rect = patches.Rectangle((j*bs,i*bs),bs,bs,linewidth=1,edgecolor='m',facecolor='none')
                rect2 = patches.Rectangle((j*bs,i*bs),bs,bs,linewidth=1,edgecolor='m',facecolor='none')

            arrow = patches.Arrow(j*bs + bs//2,i*bs + bs//2 ,tabx[i * (m//bs) + j],taby[i * (m//bs) + j], width=1.0,edgecolor='r',facecolor='none')
            ax[1].add_patch(arrow)
            ax[0].add_patch(rect)
            ax[1].add_patch(rect2)
    plt.savefig("test_reussi.png")
    print(str(count)+" blocs corrects/ "+str((n//bs)*(m//bs)))

def main():

    band1 = np.loadtxt("band1.txt")
    band2 = np.loadtxt("band2.txt")

    # DECALAGE "GROSSIER" de BAND 2 par rapport à BAND 1
    axis0 = 3
    axis1 = 2
    b1,b2 = shiftSelec(band1,band2,axis0,axis1)
    bs = 207 # Block size
    # Distribution des blocs sur les processes
    n,m = np.shape(b2)
    nb = (n // bs) * (m // bs) # Nombre de blocs dans l'image
    nd = nb // size # Nombre de blocs à traiter par process
    start =  rank * nd
    end = (rank + 1) * nd
    if (rank == size - 1): # Le dernier process va jusqu'au bout au cas où nb % size != 0
        end = nb
        nd = end - start

    # Parcours des blocks
    # if rank == 0 :
    #     print("Nombre de blocs : " + str(nb))
    #
    print("rank : " + str(rank) + " | start : " + str(start) + " | end : " + str(end))
    tabx,taby,count = decoupage(b2,b1,bs,start,end)
    mpi.COMM_WORLD.barrier()
    #c = mpi.COMM_WORLD.allreduce(sendobj = count, op = mpi.SUM)
    tabx = mpi.COMM_WORLD.allgather(tabx)
    taby = mpi.COMM_WORLD.allgather(taby)
    if rank == 0:
        tx = np.zeros(nb)
        ty = np.zeros(nb)
        for k in range(size):
            for i in range(len(tabx[0])):
                tx[k * len(tabx[0]) + i] = tabx[k][i]
                ty[k * len(taby[0]) + i] = taby[k][i]

        # np.savetxt("tx.txt", tx)
        # np.savetxt("ty.txt", ty)
        # tx = np.loadtxt("tx.txt")
        # ty = np.loadtxt("ty.txt")
        visualize(b1,b2,tx,ty,bs)

rank = mpi.COMM_WORLD.Get_rank() #  Numéro du process
size = mpi.COMM_WORLD.Get_size() # Nombre de process
main()
