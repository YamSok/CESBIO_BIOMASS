'''
This library permits to load and visualize biomass regions of interest.
'''

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as scpstats
from croscor import choiceSimple

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

def loadParcels(num = None):

    '''
    Loads forest regions of interest from
    '''

    filenames = choiceSimple("../data/16ROI/")
    parcels = []
    for filename in filenames:
        parcels.append(np.loadtxt(filename))
    return [x.astype(int) for x in parcels]

################################################################
# CHARGEMENT DES ROI - BIOMASSE
################################################################

# def loadBiomass(num = None):
#     if num == None :
#         bmssList = np.loadtxt("../data/16insituAGB.dat")
#     else:
#         l = np.loadtxt("../data/16insituAGB.dat")
#         bmssList = l[num - 1]
#     return bmssList

def loadBiomass(num = 85):
    '''
    Loads data from regions of interest from the file.
    '''
    return np.loadtxt("../data/"+ str(num)+"insituAGB.dat")

def plotParcels(num = None):

    '''
    Displays regions of interest on 'band2' frame.
    'num' is the number of the region of interest to be displayed.
    If 'num' is not given, all regions of interest are displayed.
    '''

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
    '''
    Returns image from which biomass can be estimated.
    '''
    band2corr = band2 / band1shiftee
    return band2corr

################################################################
# INTENSITES D'UNE ZONE PARTICULIERE
################################################################

def IntensityZone(X,img):

    '''
    Takes an array of points from a given region of interest and
    returns mean intensity value associated to this region.

    'X' is a 2D array containing all 'x' and 'y' coordinates from point of the region.
    'Intmean' is ...
    'IntTab' is ...
    '''

    IntTab = []
    n,m = np.shape(X)
    for i in range(n):
        IntTab.append(img[X[i][1],X[i][0]])
    Intmean = np.mean(np.array(IntTab))
    return Intmean, IntTab

def sortBiomInt(BiomassData,IntensityData):

    '''
    Sorts biomass and intensity couples. ???

    'BiomassData' is ...
    'IntensityData' is ...
    '''

    dataList = []
    finalList = []

    for i in range(len(BiomassData)):
        dataList.append( ( BiomassData[i] , IntensityData[i] ) )
    sortedList = sorted(dataList)

    for i in range(len(sortedList)):
        finalList.append( [ sortedList[i][0] , sortedList[i][1] ] )

    return np.array(finalList)
