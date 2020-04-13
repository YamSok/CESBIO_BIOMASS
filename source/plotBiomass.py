
################################################################
# BIBLIOTHEQUES
################################################################

import numpy as np
import matplotlib.pyplot as plt

################################################################
# CHARGEMENT DES ROI - PARCELLES
################################################################

def loadParcels(num = None):
    if num == None: # On charge toutes les parcelles dans une liste
        parcels = []
        for i in range(1,17):
            #parcels.append(np.loadtxt("../data/16ROI/indcsROI_PAR" +"{:02d}".format(i)+ ".dat"))
            parcels.append(np.loadtxt("../data/16ROI/indcsROI_PAR" +"{:02d}".format(i)+ ".dat"))
        return [x.astype(int) for x in parcels]
    else: #  On charge uniquemnent la parcelle numéro "num"
        #parcel = np.loadtxt("../data/16ROI/indcsROI_PAR" +"{:02d}".format(num)+ ".dat")
        parcel = np.loadtxt("../data/16ROI/indcsROI_PAR" +"{:02d}".format(num)+ ".dat")
        return parcel.astype(int)

################################################################
# CHARGEMENT DES ROI - BIOMASSE
################################################################

def loadBiomass(num = None):
    if num == None :
        bmssList = np.loadtxt("../data/16insituAGB.dat")
    else:
        l = np.loadtxt("../data/16insituAGB.dat")
        bmssList = l[num - 1]
    return bmssList

################################################################
# AFFICHAGE DES ROI
################################################################

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

def IntensityZone(X,img): # programme juste
    IntTab = []
    n,m = np.shape(X)
    for i in range(n):
        IntTab.append(img[X[i][1],X[i][0]])
    Intmean = np.mean(np.array(IntTab))
    return Intmean, IntTab

################################################################
# TRIAGE DES COUPLES BIOMASSE - INTENSITE
################################################################

def sortBiomInt(BiomassData,IntensityData):
    dataList = []
    finalList = []

    for i in range(len(BiomassData)):
        dataList.append( ( BiomassData[i] , IntensityData[i] ) )
    sortedList = sorted(dataList)

    for i in range(len(sortedList)):
        finalList.append( [ sortedList[i][0] , sortedList[i][1] ] )

    return np.array(finalList)


################################################################
#################### PROGRAMME PRINCIPAL #######################
################################################################

def main(img): #img est l'image corrigée - rapport band2/band1-corrigee
    IntensityData = []
    BiomassData = loadBiomass()

    Parcels = loadParcels()
    for X in Parcels:
        IntensityZone_X = IntensityZone(X,img)
        IntensityData.append(IntensityZone_X[0])

    sortedData = sortBiomInt(BiomassData,IntensityData)
    print(sortedData)
    print("---------------------------------------------")

    plt.figure(1)
    plt.scatter(sortedData[:,0],sortedData[:,1])
    plt.title("Intensité image en fonction de la biomasse sur 16 ROI de forêt")
    plt.xlabel("Qté de biomasse de 16 parcelles (Ordre croissant de qté)")
    plt.ylabel("Intensité image parcelle")
    plt.savefig("../results/plotMaternelle.png")
    plt.show()

################################################################

img = np.load("../decoup/band1_new.npy")
main(img)


################################################################
#################### PROGRAMME TERMINE #########################
################################################################
