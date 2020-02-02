# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 20:09:08 2020

@author: Mikael
"""

#from osgeo import gdal
import numpy as np
import rasterio
from rasterio.plot import show
import matplotlib.pyplot as plt
### From stackoverflow, to make imshow subplot with individuals colorbars
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm
from matplotlib.ticker import MultipleLocator
####

filepath = r"../Data/img_tif.tif"
raster = rasterio.open(filepath)

#show((raster, 1), cmap='Reds')
#show((raster, 2), cmap='inferno')

fig, (ax1, ax2) = plt.subplots(1,2)


im1 = ax1.imshow(raster.read(2)[:400,800:])
im2 = ax2.imshow(raster.read(1)[:400,800:])
plt.show()
plt.savefig("sample1")

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
