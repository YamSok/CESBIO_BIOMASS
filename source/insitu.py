import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['figure.dpi'] = 300

band1 = np.load("../data/band1.npy")
band2 = np.load("../data/band2.npy")

def loadParcels(num = None):

    if num == None: # On charge toutes les parcelles dans une liste
        print("oui")
        parcels = []
        for i in range(1,17):
            parcels.append(np.loadtxt("../data/16ROI/indcsROI_PAR" +"{:02d}".format(i)+ ".dat"))
        return [x.astype(int) for x in parcels]
    else: #  On charge uniquemnent la parcelle numéro "num"
        parcel = np.loadtxt("../data/16ROI/indcsROI_PAR" +"{:02d}".format(num)+ ".dat")
        return parcel.astype(int)

def loadAGB():
    return np.loadtxt("../data/16insituAGB.dat")


plt.imshow(10*np.log(band2),vmin=-40,vmax=0)
plt.title("BAND 2 - Image aéroportée")
plt.colorbar()
parcels = loadParcels()
for i in range(len(parcels)):
    for j in range(len(parcels[i])):
        plt.scatter(parcels[i][j][0],parcels[i][j][1] , 0.1)
plt.show()
plt.savefig("band2_insitu")
