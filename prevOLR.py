'''
Author: ChenHJ
Date: 2021-10-20 17:46:10
LastEditors: ChenHJ
LastEditTime: 2021-11-04 21:50:41
FilePath: /ys17-23/chenhj/1019code/prevOLR.py
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

ch = ""
def pick_year(srcPath, dstPath, fromyear, toyear):
    g = os.walk(srcPath)
    for path, dir_list, file_list in g:
        for filename in file_list:
            inputfile = os.path.join(path, filename)
            outputfile = os.path.join(dstPath, filename[:-12] + str(fromyear) + "-" + str(toyear) + ".nc")
            cdo.selyear(str(fromyear) + r"/" + str(toyear), input = inputfile, output = outputfile)

srcPath = ch + "/home/ys17-23/chenhj/monsoon/HadISST/"
dstPath = ch + "/home/ys17-23/chenhj/monsoon/pyear/"
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
import proplot as pplt
import cartopy
cdo = Cdo()



def p_time(data, mon_s, mon_end, meanon):
    time = data["time"]
    n_year = int(len(time)/12)
    n_mon = mon_end - mon_s + 1
    plist = np.zeros(n_mon * n_year,dtype=np.int64)
    for i in np.arange(0,n_year):
        plist[n_mon * i : n_mon * (i + 1)] = np.arange(mon_s-1, mon_end, dtype=np.int64)
        plist[n_mon * i : n_mon * (i + 1)] += 12 * i
    n_data = data.sel(time=time[plist],method=None)
    # print(n_data)
    if meanon == True :
        n_data_mean = n_data.coarsen(time=n_mon).mean()
        return(n_data_mean)
    elif meanon == False :
        return(n_data)
    else:
        print("Bad argument: meanon")

# ch = "/mnt/e"
ch = ""

folr = xr.open_dataset(ch + "/home/ys17-23/chenhj/monsoon/pyear/OLR_r144x72_1975-2020.nc")
olr = folr["olr"]
olr69 = p_time(olr, 6, 9, True)

fersst = xr.open_dataset(ch + "/home/ys17-23/chenhj/monsoon/pyear/ERSSTv5_r144x72_1975-2020.nc")
ersst = fersst["sst"]
ersst69 = p_time(ersst, 6, 9, True)
ersst69mean = ersst69.mean(dim=["time","lev"],skipna=True)
# print(ersst69mean)

fhadisst = xr.open_dataset(ch + "/home/ys17-23/chenhj/monsoon/pyear/HadISST_r144x72_1975-2020.nc")
hadisst = fhadisst["sst"]
hadisst69 = p_time(hadisst, 6, 9, True)


pplt.rc.reso = 'med'
array = [[1,1,2,2],[0,3,3,0]]
fig = pplt.figure(refwidth=1.8)
axs = fig.subplots(array, proj='cyl')
axs.format(coast=True, latlines=30, lonlines=60, abc=True, abcloc='ul', suptitle='SST & OLR',  lonlim=(40,180), latlim=(-50,40))
m = axs[0].contourf(ersst69mean, cmap='ColdHot')
fig.colorbar(m, loc='b', label='degree')
fig.format(abc='a)', abcloc='ul', abcborder=True, suptitle='SST & OLR')




# %%

# %%
