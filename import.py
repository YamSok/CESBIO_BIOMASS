# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 20:09:08 2020

@author: Mikael
"""

#from osgeo import gdal
import numpy as np
#import rasterio
#from rasterio.plot import shows
import matplotlib.pyplot as plt
### From stackoverflow, to make imshow subplot with individuals colorbars
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm
from matplotlib.ticker import MultipleLocator
####
#
#filepath = r"../Data/img_tif.tif"
#raster = rasterio.open(filepath)

#show((raster, 1), cmap='Reds')
#show((raster, 2), cmap='inferno')

#fig, (ax1, ax2) = plt.subplots(1,2)

#band1 = raster.read(1)
#band2 = raster.read(2)
#np.savetxt("band2.txt",band2)
band1 = np.loadtxt("band1.txt")
band2 = np.loadtxt("band2.txt")

plt.imshow(10*np.log(band2))

plt.savefig("test.png")

plt.imshow(band1)

#plt.savefig("band1_big.png")
#plt.savefig("band1test.png")
#plt.imshow(a)
#plt.savefig("a.png")
#im1 = ax1.imshow(raster.read(2)[:400,800:])
#im2 = ax2.imshow(raster.read(1)[:400,800:])
#plt.savefig("sample1")
#plt.show()


##
##raster = gdal.Open(filepath)
##
####print(type(raster))
##
##print(raster.RasterCount)
##
##
##
##band = raster.GetRasterBand(1)
##
##rasterArray = raster.ReadAsArray()
##
##print(rasterArray)
