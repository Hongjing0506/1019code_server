'''
Author: ChenHJ
Date: 2021-10-19 16:08:44
LastEditors: ChenHJ
LastEditTime: 2021-10-19 18:03:56
FilePath: /1019coderepositories/predealing.py
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

def predealing(srcPath, tmpPath, dstPath, name):
    if os.path.isdir(dstPath):
        shutil.rmtree(dstPath)
    if os.path.isdir(tmpPath):
        shutil.rmtree(tmpPath)
    os.mkdir(dstPath)
    os.mkdir(tmpPath)
    g = os.walk(srcPath)
    for path, dir_list, file_list in g:
        inputString = ""
        if len(file_list) == 1:
            inputpath = os.path.join(path, file_list[0])
            outputpath = os.path.join(dstPath, name)
            cdo.copy(input = "-remapbil,r144x72 " + inputpath, output = outputpath)
        else:
            for filename in file_list:
                inputpath = os.path.join(path, filename)
                tmppath = os.path.join(tmpPath, filename)
                inputString += tmppath + " "
                cdo.remapbil("r144x72", input = inputpath, output = tmppath)
            outputpath = os.path.join(dstPath, name)
            cdo.mergetime(input = inputString, output = outputpath)
str1 = "/mnt/e/"
str2 = "monsoon/"
str3 = "modified/"
str4 = "ERSSTv5/"

srcPath = str1 + str2 + str4
tmpPath = str1 + str2 + "tmp/"
dstPath = str1 + str3 + str4
name = "ERSSTv5_r144x72_1870-2020.nc"
predealing(srcPath, tmpPath, dstPath, name)
# %%