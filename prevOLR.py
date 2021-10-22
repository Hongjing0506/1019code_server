'''
Author: ChenHJ
Date: 2021-10-20 17:46:10
LastEditors: ChenHJ
LastEditTime: 2021-10-22 21:23:42
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
import numpy as np
import xarray as xr
import os
import re
from cdo import Cdo
import shutil
cdo = Cdo()

folr = xr.open_dataset("/home/ys17-23/chenhj/monsoon/pyear/OLR_r144x72_1975-2020.nc")
olr = folr["olr"]

def p_time(data,mon_s,mon_end):
    time = data["time"]
    n_year = int(len(time)/12)
    print(n_year)
    n_mon = mon_end - mon_s + 1
    plist = np.zeros(n_mon * n_year,dtype=np.int64)
    for i in np.arange(0,n_mon):
        print(plist[n_mon * i : n_mon + i])
        plist[n_mon * i : n_mon + i] = np.arange(mon_s-1, mon_end)
        plist[n_mon * i : n_mon + i] += 12 * i
    print(plist)
p_time(olr, 6, 9)


# %%
import numpy as np
plist = np.zeros(8)
plist[0:4] = np.arange(5,9)
print(plist)
# %%
