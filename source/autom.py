#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 17:03:26 2020

@author: yamsok
"""

# import os
# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.axes_grid1 import make_axes_locatable
# from matplotlib.colors import LogNorm
# from matplotlib.ticker import MultipleLocator
# from matplotlib.pyplot import figure
# import matplotlib.patches as patches
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300
import warnings
warnings.filterwarnings("ignore")

from croscor import *


c = choice()
ff = ExtractFeatures(c)
print(ff)
print(c)
tab = np.load('../decoup/' + c)
visualizeSuperpose(ff,tab)
#npy = choice()
