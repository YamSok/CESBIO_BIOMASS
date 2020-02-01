# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 20:09:08 2020

@author: Mikael
"""

from osgeo import gdal
import numpy as np
import rasterio
from rasterio.plot import show

filepath = r"../Data/img_tif.tif"
raster = rasterio.open(filepath)

show((raster,2))
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
