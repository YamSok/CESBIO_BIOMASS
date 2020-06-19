# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 20:09:08 2020

@author: Mikael
"""

import numpy as np
import rasterio
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm
from matplotlib.ticker import MultipleLocator
import time as t


filepath = r"../data/AfriSAR_Lope/geo10Md3N0-iHV_lk3-15-t9.tif"
raster = rasterio.open(filepath)

band1 = raster.read(1)
band2 = raster.read(2)
np.save("../data/afri_band1.npy",band1)
np.save("../data/afri_band2.npy",band2)
