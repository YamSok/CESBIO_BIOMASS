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
import os


# def shiftSelec(im1,im2,axis0,axis1):
#     band2_s = np.roll(np.roll(im2,axis0,axis=0),axis1,axis=1)
#     #b2 = selection(band2_s,115,1651,30,1054)
#     b2 = selection(10*np.log(band2_s),115,1651,30,1054)
#     b1 = selection(im1,115,1651,30,1054)
#     return b1,b2

################################################################################
############################# Prétraitement ####################################
################################################################################

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

################################################################################
################################ Calcul ########################################
################################################################################

# CALCUL DE LA CORRELATION CROISEE ENTRE original ET template
def decalageBloc(original, template):
    r = 25
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
    y = y + mc - r
    x = x + nc - r

    return orig, temp, corr, x, y

# APPLICATION CORRELATION CROISEE SUR DES BLOCS SUPERPOSES
def decoupageSuperpose(b2,b1,bs,r,f,start,end): # f = factor
    n,m = np.shape(b2)
    # VARIABLES
    tabx=[] # stockage décalage x
    taby=[] # stockage décalage y
    count = 0 # compte des blocs corrects

    for i in range(f * (n//bs) - (f-1)): # Parcours des blocs superposés (incertain)
        for j in range(f * (m//bs)- (f-1)):
            if i * (f * (m // bs) - (f-1)) + j  >= start and i * (f * (m // bs) - (f-1)) + j < end: # Vérification que le processus doit bien traiter ce bloc
                band2Block = np.copy(b2[int((i / f) * bs) : int((i / f) * bs + bs) , int((j / f) * bs) : int((j / f) * bs + bs)])  # Selection des blocs sur band 1 et 2
                band1Block = np.copy(b1[int((i / f) * bs) : int((i / f) * bs + bs) , int((j / f) * bs) : int((j / f) * bs + bs)])
                templateBlock = np.copy(band1Block[5:bs-5,5:bs-5])  # Selection du sous bloc
                orig,temp,corr,x,y = decalageBloc(band2Block,templateBlock) # Calcul du déplacement
                xm = x-bs/2
                ym = y-bs/2
                tabx.append(xm)
                taby.append(ym)
                if np.sqrt(xm**2 + ym**2) < 10 :
                    count += 1
    return tabx,taby,count

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


################################################################################
################################ Affichage #####################################
################################################################################

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
    accu = int(ff["accu"])
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
    # accu = round((count / nb * 100))
    plt.savefig("../results/"+ str(f) + "f_" + str(bs) + "bs_" + str(ax0) + "sx_" + str(ax0) + "sy_" + str(seuil) + "seuil_" + str(accu) + "accu.png")

################################################################################
################################ Outils ########################################
################################################################################

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
    features = ['f', 'bs', 'ax0', 'ax1', 'seuil', 'accu']
    objectFeatures = {}
    for i in range(len(features)):
        objectFeatures[features[i]] = "".join([liste[i][s] for s in range(len(liste[i])) if liste[i][s].isdigit()])
    return objectFeatures

################################################################################
################################ Notebook #####################################
################################################################################

def miseEnBouche(band1,band2):
    fig,ax = plt.subplots(1,2, figsize=(15,8))
    im1 = ax[0].imshow(band1, vmin = 0, vmax = 5)
    ax[0].set_title("BAND 1 - Relevé topographique ")
    fig.colorbar(im1,ax=ax[0])

    im2 = ax[1].imshow(10*np.log(band2),vmin=-40,vmax=0)
    ax[1].set_title("BAND 2 - Image satellite")
    fig.colorbar(im2,ax=ax[1])

    plt.tight_layout()
    #plt.savefig("images_radar.png")
    plt.show()
# AFFICHAGE DE 4 SUBPLOTS | ( original, tamplate, cross correlation, zoom de cross correlation )
def displayImg(original,template,corr,x,y):
    n,m = np.shape(original)
    r = 25
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

################################################################################
################################ old ###########################################
################################################################################


## Vieilles fonctions, pour le rapport peut êtrye

# APPLICATION CORRELATION A UNE IMAGE DECOUPEE EN BLOCS
# def decoupage(b2,b1,bs,r,start,end):
#     n,m = np.shape(b2)
#     # VARIABLES
#     tabx=[] # stockage décalage x
#     taby=[] # stockage décalage y
#     count = 0 # compte des blocs corrects
#
#     for i in range(n//bs):
#     #i = 0 # pour les tests
#         for j in range(m//bs):
#             if i * (m//bs) + j  >= start and i * (m//bs) + j < end:
#                 #print(i * (m//bs) + j)
#                 #print("rank : " + str(rank) + " | bloc #" + str(i * (m//bs) + j))
#                 band2Block = np.copy(b2[i*bs:(i+1)*bs,j*bs:(j+1)*bs])
#                 band1Block = np.copy(b1[i*bs:(i+1)*bs,j*bs:(j+1)*bs])
#                 templateBlock = np.copy(band1Block[5:bs-5,5:bs-5])
#                 orig,temp,corr,x,y = decalageBloc(band2Block,templateBlock,r)
#                 xm = x-bs/2
#                 ym = y-bs/2
#                 tabx.append(xm)
#                 taby.append(ym)
#                 if np.sqrt(xm**2 + ym**2) < 25 :
#                     count += 1
#                 # tabx.append(i * (m//bs) + j)
#     #print("rank : " + str(rank) + " | count : " + str(count))
#     return tabx,taby,count
#
# # AFFICHAGE DES RESULTATS DU DECOUPAGE
# def visualize(b1,b2,tabx,taby,bs,axis0,axis1,r,seuil):
#     n,m = np.shape(b2)
#     fig,ax = plt.subplots(1,2,figsize=(10,10))
#     ax[0].imshow(b2)
#     ax[1].imshow(b1)
#     count = 0
#     for i in range(n//bs) :
#         for j in range(m//bs) :
#             if np.sqrt(tab[0][i * (m//bs) + j]**2 + tab[1][i * (m//bs) + j]**2) == r :
#                 c =  'k' # couleur noire
#                 l = 2 # épaisseur du trait du vecteur
#             elif np.sqrt(tab[0][i * (m//bs) + j]**2 + tab[1][i * (m//bs) + j]**2)  <= seuil: # calcul de la
#                 c = 'm' # magenta
#                 l = 1
#                 count +=1
#             else:
#                 c = 'r'
#                 l = 2
#             rect = patches.Rectangle((j*bs,i*bs),bs,bs,linewidth=l,edgecolor=c,facecolor='none')
#             rect2 = patches.Rectangle((j*bs,i*bs),bs,bs,linewidth=l,edgecolor=c,facecolor='none')
#             arrow = patches.Arrow(j*bs + bs//2,i*bs + bs//2 ,tabx[i * (m//bs) + j],taby[i * (m//bs) + j], width=0.7,edgecolor='r',facecolor='none')
#             ax[1].add_patch(arrow)
#             ax[0].add_patch(rect)
#             ax[1].add_patch(rect2)
#     plt.tight_layout()
#     plt.savefig("results/"+str(bs) + "x" + str(bs)+"_"+str(axis0) + "ax0_"+str(axis1)+"ax1_"+str(r)+"r_"+str(seuil)+"seuil_"+str(count)+ "count.png")
#     print(str(count)+" blocs corrects/ "+str((n//bs)*(m//bs)))


## Fonctions notebook
#
def decoupage(band2,band1,bs,i,j,axis0=0,axis1=0,v=False): #bs= blocksize

    # DECALAGE "GROSSIER" de BAND 2 par rapport à BAND 1
    b1,b2 = shiftSelec(band1,band2,axis0,axis1)
    r = 25
    n,m = np.shape(b2)

    fig,ax = plt.subplots(1,2,figsize=(10,10))
    ax[0].imshow(b2)
    ax[1].imshow(b1)

    # VARIABLES
    # tabx=[] # stockage décalage x
    # taby=[] # stockage décalage y
    nb = (n // bs) * (m // bs)
    tab = np.zeros((2,nb))
    count = 0 # compte des blocs corrects

    #for i in range(n//bs):
    #for i in range(3): # pour les tests
    # i = 2
    # j = 3
    #for j in range(m//bs):
    band2Block = np.copy(b2[i*bs:(i+1)*bs,j*bs:(j+1)*bs])
    band1Block = np.copy(b1[i*bs:(i+1)*bs,j*bs:(j+1)*bs])
    #print("bloc " + str(i*(m//bs) + j) + " | Var : " + "%.2f" % np.std(band1Block))
    templateBlock = np.copy(band1Block[5:bs-5,5:bs-5])
    orig,temp,corr,x,y = decalageBloc(band2Block,templateBlock)
    xm = x-bs//2
    ym = y-bs//2
    tab[0][i * (m//bs) + j] = xm
    tab[1][i * (m//bs) + j] = ym
    if np.sqrt(xm**2 + ym**2) == r :
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

    # SAUVEGARDE + NOM | AFFICHAGE
    #plt.savefig("results/"+str(bs) + "x" + str(bs)+"_"+str(axis0) + "ax0" + "_"+str(axis1)+"ax1"+".png")
    plt.show()

    # AFFICHAGE BLOC CORRECTS : err < 25 pixels
    # print(str(count)+" blocs corrects/ "+str((n//bs)*(m//bs)))
    return orig,temp,corr,x,y,tab
    #return tabx,taby

def compareImg(original,shift,template,bloc):
    n,m = np.shape(original)
    fig, (ax_orig, ax_shift, ax_template) = plt.subplots(1, 3,figsize=(10, 20))


    arrowx1 = patches.Arrow(40,0 ,0,207, width=1.0,edgecolor='r',facecolor='none')
    arrowx2 = patches.Arrow(40,0 ,0,207, width=1.0,edgecolor='r',facecolor='none')
    arrowx3 = patches.Arrow(40,0 ,0,207, width=1.0,edgecolor='r',facecolor='none')

    arrowy1 = patches.Arrow(0,120 ,207,0, width=1.0,edgecolor='r',facecolor='none')
    arrowy2 = patches.Arrow(0,120 ,207,0, width=1.0,edgecolor='r',facecolor='none')
    arrowy3 = patches.Arrow(0,120 ,207,0, width=1.0,edgecolor='r',facecolor='none')

    ax_shift.add_patch(arrowx3)
    ax_shift.add_patch(arrowy3)
    ax_shift.imshow(shift)
    ax_shift.set_title('Shift')

    ax_template.add_patch(arrowx1)
    ax_template.add_patch(arrowy1)
    ax_template.imshow(template)
    ax_template.set_title('Template ' + str(bloc))

    ax_orig.imshow(original)
    ax_orig.set_title('Original')
    ax_orig.add_patch(arrowx2)
    ax_orig.add_patch(arrowy2)

    fig.show()


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

def gaussianFilter(im1,factor):
    kernel = np.ones((factor,factor),np.float32)/(factor**2)
    target = cv2.filter2D(im1,-1,kernel)
    return target
