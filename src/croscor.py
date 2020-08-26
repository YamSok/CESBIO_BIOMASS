import numpy as np
import numpy.linalg as npl
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
import scipy.stats as scpstats
mpl.rcParams['figure.dpi'] = 100

# def shiftSelec(im1,im2,axis0,axis1):
#     band2_s = np.roll(np.roll(im2,axis0,axis=0),axis1,axis=1)
#     #b2 = selection(band2_s,115,1651,30,1054)
#     b2 = selection(10*np.log(band2_s),115,1651,30,1054)
#     b1 = selection(im1,115,1651,30,1054)
#     return b1,b2

################################################################################
############################# Prétraitement ####################################
################################################################################

# def shiftSelec(im1,im2,axis0,axis1):
#     band2_s = np.roll(np.roll(im2,axis0,axis=0),axis1,axis=1)
#     #b2 = selection(band2_s,115,1651,30,1054)
#     b2 = selection(10*np.log(band2_s),115 + 3 * 256,1651,30 + 2 * 256,1054)
#     b1 = selection(im1,115 + 3 *256,1651,30 + 2* 256 ,1054)
#     return b1,b2
#
# def selection(img,x0,x1,y0,y1):
#     h = abs(x0 - x1)
#     w = abs(y0 - y1)
#     return img[x0:x0+h,y0:y0+w]


def shiftSelec(im1,im2,axis0,axis1):
    #band2_s = np.roll(np.roll(im2,axis0,axis=0),axis1,axis=1)
    band1_s = np.roll(np.roll(im1,axis0,axis=0),axis1,axis=1)
    #b2 = selection(band2_s,115,1651,30,1054)
    margin = 64
    (x0,x1,y0,y1) = (30 + 2 * 256 - margin, 1054 + margin, 215 + 3 * 256, 1751 + margin)
    #(x0,x1,y0,y1) = (30 + 2 * 256, 1054, 215 + 3 * 256, 1751)
    coord = (x0,x1,y0,y1)
    b2 = selection(10*np.log(im2),coord)
    b1 = selection(band1_s,coord)
    return b1,b2


def shiftSelec_new(im1,im2,axis0,axis1): # Pour nouvelle résolution
    #band2_s = np.roll(np.roll(im2,axis0,axis=0),axis1,axis=1)
    band1_s = np.roll(np.roll(im1,axis0,axis=0),axis1,axis=1)
    #b2 = selection(band2_s,115,1651,30,1054)
    margin = 0
    (x0, x1, y0, y1) = (200, 520, 480, 864)
    #(x0,x1,y0,y1) = (30 + 2 * 128 - margin, 1054 + margin, 215 + 3 * 256, 1751 + margin)
    #(x0,x1,y0,y1) = (30 + 2 * 256, 1054, 215 + 3 * 256, 1751)
    coord = (x0,x1,y0,y1)
    b2 = selection(10*np.log(im2),coord)
    b1 = selection(band1_s,coord)
    return b1,b2



def selection(img,coord, output = False):

    """
    * Retourne l'image donnée en argument crop selon les coordonnées données en argument
    * Plot les ROI si output = True
    """

    (x0,x1,y0,y1) = coord
    w = abs(x0 - x1)
    h = abs(y0 - y1)
    if output:
        fig, ax = plt.subplots(figsize=(10,15))
        parcels = loadParcels()
        #bms = loadBiomass(16)
        # for i in range(len(parcels)):
        #     x = [p[0] for p in parcels[i]]
        #     y = [p[1] for p in parcels[i]]
        #     if i == 0:
        #         plt.scatter(x,y, 0.1, color="black")
        #     else :
        #         plt.scatter(x,y, 0.1)
        #arrow = patches.Arrow(0,1369,1000,0,edgecolor='r')
        #arrow2 = patches.Arrow(662,0,0,1800,edgecolor='r')

        rect = patches.Rectangle((x0,y0),w,h,linewidth=3,edgecolor='r',facecolor='none')
        ax.imshow(10*np.log(img),vmin=-40,vmax=0)
        #ax.add_patch(arrow)
        #ax.add_patch(arrow2)

        ax.add_patch(rect)
        plt.tight_layout()
        plt.savefig("../misc/roi_corrected")
        plt.show()
    return img[y0:y0+h,x0:x0+w]


def selection2(img,coord, output = False):
    (x0,x1,y0,y1) = coord
    w = abs(x0 - x1)
    h = abs(y0 - y1)
    if output:
        fig, ax = plt.subplots(figsize=(10,15))
        # parcels = loadParcels()
        # #bms = loadBiomass(16)
        # for i in range(len(parcels)):
        #     x = [p[0] for p in parcels[i]]
        #     y = [p[1] for p in parcels[i]]
        #     if i == 0:
        #         plt.scatter(x,y, 0.1, color="black")
        #     else :
        #         plt.scatter(x,y, 0.1)
        #arrow = patches.Arrow(0,1369,1000,0,edgecolor='r')
        #arrow2 = patches.Arrow(662,0,0,1800,edgecolor='r')

        rect = patches.Rectangle((x0,y0),w,h,linewidth=3,edgecolor='r',facecolor='none')
        ax.imshow(img,vmin=0,vmax=5)
        #ax.add_patch(arrow)
        #ax.add_patch(arrow2)

        ax.add_patch(rect)
        plt.tight_layout()
        plt.savefig("../misc/dem")
        plt.show()
    return img[y0:y0+h,x0:x0+w]

################################################################
# CHARGEMENT DES ROI - PARCELLES
################################################################

"""
Ancienne version, pour être sûr que l'import se passe bien
"""
# def loadParcels(num = None):
#     if num == None: # On charge toutes les parcelles dans une liste
#         parcels = []
#         for i in range(1,17):
#             #parcels.append(np.loadtxt("../data/16ROI/indcsROI_PAR" +"{:02d}".format(i)+ ".dat"))
#             parcels.append(np.loadtxt("../data/16ROI/indcsROI_PAR" +"{:02d}".format(i)+ ".dat"))
#         return [x.astype(int) for x in parcels]
#     else: #  On charge uniquemnent la parcelle numéro "num"
#         #parcel = np.loadtxt("../data/16ROI/indcsROI_PAR" +"{:02d}".format(num)+ ".dat")
#         parcel = np.loadtxt("../data/16ROI/indcsROI_PAR" +"{:02d}".format(num)+ ".dat")
#         return parcel.astype(int)

"""
Version plus simple
"""
def loadParcels(num = 16):
    filenames = choiceSimple("../data/"+str(num)+"ROI/",all=True)
    parcels = []
    for filename in filenames:
        parcels.append(np.loadtxt("../data/"+str(num)+"ROI/"+filename))
    return [x.astype(int) for x in parcels]

################################################################
# CHARGEMENT DES ROI - BIOMASSE
################################################################

"""
Ancienne version, pour être sûr que l'import se passe bien
"""
# def loadBiomass(num = None):
#     if num == None :
#         bmssList = np.loadtxt("../data/16insituAGB.dat")
#     else:
#         l = np.loadtxt("../data/16insituAGB.dat")
#         bmssList = l[num - 1]
#     return bmssList

"""
Version plus simple
"""
def loadBiomass(num = 85):
    return np.loadtxt("../data/"+ str(num)+"insituAGB.dat")

################################################################
# AFFICHAGE DES ROI
################################################################

"""
Fonction standalone pour plot les ROI sur band 2
"""
def plotParcels(num = None):
    band2 = np.load("../data/band2.npy")
    #band2x = 10 * np.log(band2)
    plt.figure(1)
    plt.imshow(band2)
    if num == None :
        Parcels = loadParcels()
        for i in range(16):
            X = Parcels[i]
            plt.scatter(X[:,0], X[:,1])
            plt.savefig("parcels.png")
    else:
        X = loadParcels(num)
        print(np.shape(X))
        plt.scatter(X[:,0], X[:,1])
        plt.savefig("parcel.png")

################################################################
# VALEUR DES INTENSITES - IMG RAPPORT BAND2 / BAND1SHIFTEE
################################################################

def Intensities(band1shiftee,band2):
    band2corr = band2 / band1shiftee
    return band2corr

################################################################
# INTENSITES D'UNE ZONE PARTICULIERE
################################################################

"""
POTENTIELLE SOURCE D'ERREUR
Après test, retourner Intmean ou 10 * np.log(Intmean) ne change pas le résultat
"""

def IntensityZone(X,img):
    IntTab = []
    n,m = np.shape(X)
    for i in range(n):
        IntTab.append(img[X[i][1],X[i][0]])
    Intmean = np.mean(np.array(IntTab))
    #print(IntTab)
    return 10*np.log10(Intmean), IntTab

################################################################
# TRIAGE DES COUPLES BIOMASSE - INTENSITE
################################################################

"""
Correspondance entre valeurs de biomasse in situ et valeurs d'intensités
"""
def sortBiomInt(BiomassData,IntensityData):
    dataList = []
    finalList = []

    for i in range(len(BiomassData)):
        dataList.append( ( BiomassData[i] , IntensityData[i] ) )
    sortedList = sorted(dataList)

    for i in range(len(sortedList)):
        finalList.append( [ sortedList[i][0] , sortedList[i][1] ] )

    return np.array(finalList)

################################################################################
################################ Calcul ########################################
################################################################################

# CALCUL DE LA CORRELATION CROISEE ENTRE original ET template
def decalageBloc(original, template, padding):
    p = padding
    orig = np.copy(original)  #prévenir pbs de pointeurs python
    temp = np.copy(template)

    # Normalisation des données
    orig -= original.mean()
    orig = orig/np.std(orig)
    temp -= template.mean()
    temp = temp/np.std(temp)

    corr = signal.correlate2d(orig, temp, boundary='symm', mode='same')
    n,m = np.shape(corr)
    n_middle = n // 2
    m_middle = m // 2

    # Selection du domaine admissible pour la recherche du déplacement, correpondant au carré de coté 2 x padding
    corr_admissible = corr[n_middle - p:n_middle + p, m_middle - p:m_middle + p]
    y, x = np.unravel_index(np.argmax(corr_admissible), corr_admissible.shape)  # find the match (max of correlation)
    y = y + m_middle - p
    x = x + n_middle - p

    return orig, temp, corr, x, y

# APPLICATION CORRELATION CROISEE SUR DES BLOCS SUPERPOSES
# def decoupageSuperposeOld(b2,b1,bs,r,f,start,end): # f = factor
#     n,m = np.shape(b2)
#     # VARIABLES
#     tabx=[] # stockage décalage x
#     taby=[] # stockage décalage y
#     count = 0 # compte des blocs corrects
#     padding = 10

#     for i in range(f * (n//bs) - (f-1)): # Parcours des blocs superposés (incertain)
#         for j in range(f * (m//bs)- (f-1)):
#             if i * (f * (m // bs) - (f-1)) + j  >= start and i * (f * (m // bs) - (f-1)) + j < end: # Vérification que le processus doit bien traiter ce bloc
#                 band2Block = np.copy(b2[int((i / f) * bs) : int((i / f) * bs + bs) , int((j / f) * bs) : int((j / f) * bs + bs)])  # Selection des blocs sur band 1 et 2
#                 band1Block = np.copy(b1[int((i / f) * bs) : int((i / f) * bs + bs) , int((j / f) * bs) : int((j / f) * bs + bs)])
#                 templateBlock = np.copy(band1Block[padding:bs - padding, padding:bs - padding])  # Selection du sous bloc
#                 orig,temp,corr,x,y = decalageBloc(band2Block,templateBlock, padding) # Calcul du déplacement
#                 xm = x-bs/2 # Normalisation
#                 ym = y-bs/2
#                 tabx.append(xm)
#                 taby.append(ym)
#                 # if np.sqrt(xm**2 + ym**2) < 10 :
#                 if npl.norm([x,y], np.inf)
#                     count += 1
                
#     return tabx,taby,count

def decoupageSuperpose(b2,b1,bs,f,start,end): # f = facteur de recouvrement
    """
    Calcul du déplacement pour chaque bloc de l'image, sachant le facteur de recouvrement f
    """
    n,m = np.shape(b2)
    # VARIABLES
    tabx=[] # stockage décalage x
    taby=[] # stockage décalage y
    count = 0 # compte des blocs corrects
    padding = 10
    ncol = int(m // (bs/f) - (f - 1))
    nrow = int(n // (bs/f) - (f - 1))
    for i in range(nrow):
        for j in range(ncol):
            if i * ncol + j  >= start and i * ncol + j < end: # Vérification que le processus doit bien traiter ce bloc
                band2Block = np.copy(b2[int((i / f) * bs) : int((i / f) * bs + bs) , int((j / f) * bs) : int((j / f) * bs + bs)])  # Selection des blocs sur band 1 et 2
                band1Block = np.copy(b1[int((i / f) * bs) : int((i / f) * bs + bs) , int((j / f) * bs) : int((j / f) * bs + bs)])
                templateBlock = np.copy(band1Block[padding:bs - padding, padding:bs - padding])  # Selection du sous bloc
                orig, temp, corr, x, y = decalageBloc(band2Block, templateBlock, padding) # Calcul du déplacement
                xm = x - bs / 2
                ym = y - bs / 2
                tabx.append(xm)
                taby.append(ym)
                # if np.sqrt(xm**2 + ym**2) < 10 :
                if npl.norm([xm,ym], np.inf) < 10:
                    count += 1
    return tabx,taby,count

# DONNE LE NOMBRE DE BLOCS AVEC DECALGE < SEUIL
def countCorrect(tab, seuil, verbose=False):

    for i in range(len(tab[0])):
        # distance = np.sqrt(tab[0][i]**2 + tab[1][i]**2)
        distance = npl.norm([tab[0][i],tab[1][i]], np.inf)
        if verbose :
            print("Décalage du block " +str(i)+ " : %.2f" % (distance) + " m.")
        # if distance < seuil:  #distance inférieure à 50 px (c'est beaucoup)
        #     dist.append(distance)
    # if verbose:
    #     print(str(count)+" corrects sur "+ str(nb) + " avec une marge de " + str(seuil * 5) +" m.")
    xdist = np.mean(tab[0])
    ydist = np.mean(tab[1])
    dist = npl.norm([xdist,ydist], np.inf)

    # dist = np.sqrt(xdist**2 + ydist**2)
    # print("\n")
    print("Moyenne des déplacements : " + str(dist))
    print("Moyenne des en x : " + str(xdist))
    print("Moyenne des en y : " + str(ydist))

    return dist, xdist, ydist


################################################################################
################################ Affichage #####################################
################################################################################
"""
Fonction qui permet de visualiser un champ de vecteur déjà calculé sur band1
"""

def visualizeSuperpose(ff,tab): # ff = file features, tab = tableau des déplacement (x,y)
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
    b1 = np.load("../data/afri_band1.npy")
    b2 = np.load("../data/afri_band2.npy")
    # b1, b2 = shiftSelec(b1,b2,ax0,ax1)
    r = 7
    n,m = np.shape(b2)
    fig,ax = plt.subplots(1,1,figsize=(10,15))
    ax.imshow(b1)
    # ax.imshow(b1,vmin=0,vmax=5)
    # ax[1].imshow(b1)
    count = 0
    ncol = int(m // (bs/f) - (f - 1))
    nrow = int(n // (bs/f) - (f - 1))
    nb = nrow * ncol      # Nombre de blocs dans l'image
    for i in range(nrow) :
        for j in range(ncol) :
            # dist = np.sqrt(tab[0][i * ncol + j]**2 + tab[1][i * ncol + j]**2)
            dist = npl.norm([tab[0][i * ncol + j],tab[1][i * ncol + j]],np.inf)
            if dist == r :

                c =  'k'
                l = 1

                plt.plot(int((j/f) * bs + bs // 2 ) , int((i/f) *bs + bs // 2), color = c, marker = '.')

                arrow = patches.Arrow( int((j/f) * bs + bs // 2 ) , int((i/f) *bs + bs // 2) ,tab[0][i * ncol + j],tab[1][i * ncol + j], width=0.1,edgecolor=c,facecolor='none')
                ax.add_patch(arrow)

                # Q1 = ax[1].quiver(int((j/f) * bs + bs // 2 ) , int((i/f) *bs + bs // 2) ,tab[0][i * (f * (m//bs) - (f-1)) + j],tab[1][i * (f * (m//bs) - (f-1)) + j], angles='xy', color = c, units='width')#, headlength = 0.1, headwidth = 0.1)
                # qk = ax[1].quiverkey(Q1, 0.9, 0.9, 2, r'$2 \frac{m}{s}$', labelpos='E',
                #    coordinates='figure')

            elif dist  <= seuil:
                c = 'w'
                l = 1
                count +=1

                arrow = patches.FancyArrow( int((j/f) * bs + bs // 2 ) , int((i/f) *bs + bs // 2) ,tab[0][i * ncol + j],tab[1][i * ncol + j], width=0.1,head_width=3,head_length=6,edgecolor=c,facecolor='none')
                ax.add_patch(arrow)

                # Q2 = ax[1].quiver(int((j/f) * bs + bs // 2 ) , int((i/f) *bs + bs // 2) ,tab[0][i * (f * (m//bs) - (f-1)) + j],tab[1][i * (f * (m//bs) - (f-1)) + j], angles='xy',  color = c, units='width')#, headlength = 0.1, headwidth = 0.1)
                # qk = ax[1].quiverkey(Q2, 0.9, 0.9, 2, r'$2 mise\frac{m}{s}$', labelpos='E',
                #    coordinates='figure')
            else:
                c = 'w'
                l = 1

                arrow = patches.FancyArrow( int((j/f) * bs + bs // 2 ) , int((i/f) *bs + bs // 2) ,tab[0][i * ncol + j],tab[1][i * ncol + j], width=0.1,head_width=3,head_length=6,edgecolor=c,facecolor='none')
                ax.add_patch(arrow)
                #plt.scatter(int((j/f) * bs + bs // 2 ) , int((i/f) *bs + bs // 2), color = c, )



            #rect = patches.Rectangle( (int(j/f) * bs, int((i/f) * bs)) ,bs,bs,linewidth=l,edgecolor='k',facecolor='none')
            #rect2 = patches.Rectangle( (int(j/f) * bs, int((i/f) * bs)) ,bs,bs,linewidth=l,edgecolor='k',facecolor='none')

            # affichage de fleches
            #ax[1].quiver(int((j/f) * bs + bs // 2 ) , int((i/f) *bs + bs // 2) ,tab[0][i * (f * (m//bs) - (f-1)) + j],tab[1][i * (f * (m//bs) - (f-1)) + j], angles='xy', scale_units='xy', color = c, headlength = 0.1, headwidth = 0.1)

            # affichage de traits
            # arrow = patches.Arrow( int((j/f) * bs + bs // 2 ) , int((i/f) *bs + bs // 2) ,tab[0][i * (f * (m//bs) - (f-1)) + j],tab[1][i * (f * (m//bs) - (f-1)) + j], width=0.1,edgecolor=c,facecolor='none')
            # ax[1].add_patch(arrow)

            #ax[0].add_patch(rect)
            #ax[1].add_patch(rect2)
    plt.tight_layout()

    #plt.savefig("b2")
    # accu = round((count / nb * 100))
    plt.savefig("../results/"+ str(f) + "f_" + str(bs) + "bs_" + str(ax0) + "sx_" + str(ax0) + "sy_" + str(seuil) + "seuil_" + str(accu) + "accu.png")
    plt.show()

################################################################################
################################ Outils ########################################
################################################################################

"""
Fonction permettant de selectionner le nom d'un fichier présent dans le dossier "folder"
"""

def choiceSimple(folder = '../decoup',all = False,first = False):
    cwd = os.getcwd()
    os.chdir(folder)
    rez = os.popen('ls -t').read()
    os.chdir(cwd)

    a = rez.split()
    rez2 = [str(i) + ' - ' + a[i] for i in range(len(a)) ]
    if all == False and first == False :
        print("Liste des résulats disponibles \n")
        for i in range(len(a)):
            print(rez2[i])
        cin = input("Selection : ")
        print(a[int(cin)])
        return a[int(cin)]
    elif first == True :
        return a[0]
    else:
        return a

"""
Ancienne fonction
"""

def choice():
    os.chdir('../decoup/afri/')
    rez = os.popen('ls -t').read()
    a = rez.split()
    rez2 = [str(i) + ' - ' + a[i] for i in range(len(a)) ]
    print("Liste des résulats disponibles \n")
    for i in range(len(a)):
        print(rez2[i])
    inp = []
    ok = True
    print("\n ## Pour mettre fin à la sélection, appuyer sur entré sans entrer de numéro ##\n")
    while ok:
        cin = input("Selection : ")
        if cin == '':
            ok = False
        else :
            print(a[int(cin)])
            inp.append(cin)
    tabnames = [a[int(x)] for x in inp]
    print(tabnames)
    return tabnames

"""
Permet de récupérer les attributs d'un test déjà effectué à partir du nom du fichier enregistré
"""
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
    ax[1].set_title("BAND 2 - Image aéroportée")
    fig.colorbar(im2,ax=ax[1])

    plt.tight_layout()
    plt.savefig("../misc/images_radar.png")
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
    r = 15
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
    templateBlock = np.copy(band1Block[10:bs-10,10:bs-10])
    orig,temp,corr,x,y = decalageBloc(band2Block,templateBlock)
    xm = x-bs//2
    ym = y-bs//2
    print((xm,ym))
    tab[0][i * (m//bs) + j] = xm
    tab[1][i * (m//bs) + j] = ym
    if np.sqrt(xm**2 + ym**2) > 10 :
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
