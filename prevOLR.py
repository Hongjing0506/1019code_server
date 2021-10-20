'''
Author: ChenHJ
Date: 2021-10-20 17:46:10
LastEditors: ChenHJ
LastEditTime: 2021-10-20 18:51:14
FilePath: /chenhj/1019code/prevOLR.py
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

def pick_year(srcPath, dstPath, fromyear, toyear):
    g = os.walk(srcPath)
    for path, dir_list, file_list in g:
        for filename in file_list:
            inputfile = os.path.join(path, filename)
            outputfile = os.path.join(dstPath, filename[:-12] + str(fromyear) + "-" + str(toyear) + ".nc")
            cdo.selyear(str(fromyear) + r"/" + str(toyear), input = inputfile, output = outputfile)

srcPath = "/home/ys17-23/chenhj/monsoon/HadISST/"
dstPath = "/home/ys17-23/chenhj/monsoon/pyear/"
fromyear = 1975
toyear = 2020
pick_year(srcPath, dstPath, fromyear, toyear)
# %%