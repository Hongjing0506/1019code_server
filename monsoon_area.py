'''
Author: ChenHJ
Date: 2021-11-22 16:33:19
LastEditors: ChenHJ
LastEditTime: 2021-11-22 16:33:19
FilePath: /ys17-23/chenhj/1019code/monsoon_area.py
Aim: 
Mission: 
'''
# %%
import numpy as np
import xarray as xr
import os
import re
from cdo import Cdo
import shutil

cdo = Cdo()

# for plot
import proplot as pplt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter
from cartopy.mpl.ticker import LatitudeFormatter
from matplotlib.ticker import MultipleLocator
import matplotlib.pyplot as plt
# %%

