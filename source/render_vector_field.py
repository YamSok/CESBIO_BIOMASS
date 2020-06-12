#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 17:03:26 2020

@author: yamsok
"""

import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 100
import warnings
warnings.filterwarnings("ignore")

from croscor import *

c = choiceSimple()
ff = ExtractFeatures(c)
print(ff)
print(c)
tab = np.load('../decoup/' + c)
visualizeSuperpose(ff,tab)
#npy = choice()
