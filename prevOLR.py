"""
Author: ChenHJ
Date: 2021-10-20 17:46:10
LastEditors: ChenHJ
LastEditTime: 2021-11-06 14:55:39
FilePath: /chenhj/1019code/prevOLR.py
Aim: 
Mission: 
"""
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


"""
description: 
    (已废弃，请使用p_month)
    该函数用于选取数据中每一年的相应月份，并可选地计算每一年这些相应月份的平均值；例如：p_time(data, 6, 8, False)将选取数据中每一年的JJA，但不计算每年JJA的平均值；p_time(data, 6, 8, True)则在选取数据中每一年的JJA后计算每年JJA的平均值；
param {*} data
    the data should be xarray dataset or dataarray
param {float} mon_s
    the start months
param {float} mon_end
    the end months
param {bool} meanon
    whether to calculate the seasonal mean for every year
return {*}
    xarray dataarray
"""


def p_time(data, mon_s, mon_end, meanon):
    time = data["time"]
    n_data = data.sel(time=(data.time.dt.month<=mon_end)*(data.time.dt.month>=mon_s))
    n_mon = mon_end - mon_s + 1
    if meanon == True:
        n_data_mean = n_data.coarsen(time=n_mon).mean()
        return n_data_mean
    elif meanon == False:
        return n_data
    else:
        print("Bad argument: meanon")
        
        
'''
description: 
    本函数用于将需要的月份挑选出来，并存储为月份x年份xlatxlon的形式
param {*} data
param {*} mon_s
param {*} mon_e
return {*}
'''        
def p_month(data, mon_s, mon_e):
    import pandas as pd
    import xarray as xr
    time = data["time"]
    data.transpose("time",...)
    year_s = pd.to_datetime(time).year[1]
    year_e = pd.to_datetime(time).year[-1]
    nyear = pd.date_range(str(year_s), str(year_e), freq = "AS")
    m_ind = data.groupby("time.month").groups[mon_s]
    res = data[m_ind]
    res['time'] = nyear
    for i in np.arange(mon_s + 1, mon_e + 1):
        m_ind = data.groupby("time.month").groups[i]
        tmp = data[m_ind]
        tmp['time'] = nyear
        res = xr.concat([res, tmp], "month")
        
    month = np.arange(mon_s, mon_e + 1)
    res["month"] = month
    return(res)
    



'''
description: 计算去除气候平均态后的月距平值(计算年循环)
param {*} data
return {*}
'''
def cal_annual_a(data):
    data_m = data.mean(dim = ["time"])
    data_ac = data.groupby("time.month").mean(dim = "time") - data_m
    return data_ac
    
    

# %%
ch = ""


def pick_year(srcPath, dstPath, fromyear, toyear):
    g = os.walk(srcPath)
    for path, dir_list, file_list in g:
        for filename in file_list:
            inputfile = os.path.join(path, filename)
            outputfile = os.path.join(
                dstPath, filename[:-12] + str(fromyear) + "-" + str(toyear) + ".nc"
            )
            cdo.selyear(
                str(fromyear) + r"/" + str(toyear), input=inputfile, output=outputfile
            )


srcPath = ch + "/home/ys17-23/chenhj/monsoon/ERSSTv5/"
dstPath = ch + "/home/ys17-23/chenhj/monsoon/pyear/"
fromyear = 1979
toyear = 2020
pick_year(srcPath, dstPath, fromyear, toyear)
# %%


# ch = "/mnt/e"
ch = ""

folr = xr.open_dataset(
    ch + "/home/ys17-23/chenhj/monsoon/pyear/OLR_r144x72_1979-2020.nc"
)

startmon = 6
endmon = startmon

olr = folr["olr"]
olr69 = p_time(olr, startmon, endmon, True)
olr69mean = olr69.mean(dim="time", skipna=True)


fpre = xr.open_dataset(
    ch + "/home/ys17-23/chenhj/monsoon/pyear/GPCC_r144x72_1979-2020.nc"
)
pre = fpre["precip"]
pre69 = p_time(pre, startmon, endmon, True)
pre69mean = pre69.mean(dim="time", skipna=True)


fersst = xr.open_dataset(
    ch + "/home/ys17-23/chenhj/monsoon/pyear/ERSSTv5_r144x72_1979-2020.nc"
)
ersst = fersst["sst"]
ersst69 = p_time(ersst, 6, 9, True)
ersst69mean = ersst69.mean(dim=["time", "lev"], skipna=True)
# print(ersst69mean)

fhadisst = xr.open_dataset(
    ch + "/home/ys17-23/chenhj/monsoon/pyear/HadISST_r144x72_1979-2020.nc"
)
hadisst = fhadisst["sst"]
hadisst69 = p_time(hadisst, 6, 9, True)
hadisst69mean = hadisst69.mean(dim=["time"])


# %%
pplt.rc.grid = False
pplt.rc.reso = "lo"


# array = [1, 1, 2, 2]
fig = pplt.figure(refwidth=1.8)

# 以下为地理图的坐标轴设置
proj = pplt.PlateCarree()
axs = fig.subplots(ncols=2, nrows=1, proj=proj, wspace=3)
xticks = np.array([60, 90, 120, 150, 180])
yticks = np.array([-30, 0, 30])
axs.format(coast=True, coastlinewidth=0.8, lonlim=(40, 180), latlim=(-50, 40))
axs.set_xticks(xticks)
axs.set_yticks(yticks)
lon_formatter = LongitudeFormatter(zero_direction_label=True)
lat_formatter = LatitudeFormatter()
axs.minorticks_on()
xminorLocator = MultipleLocator(10)
yminorLocator = MultipleLocator(10)
for ax in axs:
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    ax.xaxis.set_minor_locator(xminorLocator)
    ax.yaxis.set_minor_locator(yminorLocator)
    ax.outline_patch.set_linewidth(1.0)
axs.tick_params(
    axis="both",
    which="major",
    labelsize=8,
    direction="out",
    length=3,
    width=0.8,
    pad=0.2,
    top=True,
    right=True,
)
axs.tick_params(
    axis="both",
    which="minor",
    direction="out",
    length=2,
    width=0.8,
    top=True,
    right=True,
)
axs.format(abc=True, abcloc="ul", suptitle="SST & OLR")


m = axs[0].contourf(olr69mean, cmap="ColdHot", extend="both", vmin=180, vmax=300)
fig.colorbar(m, loc="b", span=1, label="W/m^2", width=0.11, ticklen=0, ticklabelsize=5)
m = axs[1].contourf(pre69mean, cmap="ColdHot", extend="both", vmin=0, vmax=15)
fig.colorbar(m, loc="b", span=2, label="mm/day", width=0.11, ticklen=0, ticklabelsize=5)
# fig.colorbar(m, loc="r", span=1, label="degree", width=0.11, ticklen=0, ticklabelsize=5)
fig.format(abc="a)", abcloc="ul", abcborder=True, suptitle="SST & OLR")


pplt.rc.reset()

# %%
# 选取阿拉伯海的若干区域并画降水年循环
# startmon = 1
# endmon = 12

ch = ""

folr = xr.open_dataset(
    ch + "/home/ys17-23/chenhj/monsoon/pyear/OLR_r144x72_1979-2020.nc"
)
olr = folr["olr"]
olr_ac = cal_annual_a(olr)
#area_a: 5°N-20°N, 60°E-75°E
olr_ac_a = olr_ac.loc[:,5:20,60:75].mean(dim=["lat","lon"])

#area_b: 5°N-20°N, 50°E-60°E
olr_ac_b = olr_ac.loc[:,5:20,50:60].mean(dim=["lat","lon"])
# print(olr_ac_a)
# olr_p = p_time(olr, startmon, endmon, True)

fpre = xr.open_dataset(
    ch + "/home/ys17-23/chenhj/monsoon/pyear/GPCC_r144x72_1979-2020.nc"
)
pre = fpre["precip"]
pre_ac = cal_annual_a(pre)
pre_ac_a = pre_ac.loc[:,5:20,60:75].mean(dim=["lat","lon"])
pre_ac_b = pre_ac.loc[:,5:20,50:60].mean(dim=["lat","lon"])
# print(pre_ac)


# pre_p = p_time(pre, startmon, endmon, True)


# %%
fig = pplt.figure(refwidth = 1.8, span = False, share = False)
axs = fig.subplots(ncols = 2, nrows = 2)
axs[0].plot(olr_ac_a, zorder = 0)
axs[0].scatter(olr_ac_a, marker = 'x', zorder = 1)
axs[0].format(ylim = (-35,35), ylocator = 10, title = 'OLR', urtitle = '5°N-20°N\n60°E-75°E', ylabel = 'W/m^2')
axs[1].plot(pre_ac_a, zorder = 0)
axs[1].scatter(pre_ac_a, marker = 'x', zorder = 1)
axs[1].format(ylim = (-4,4), ylocator = 1, title = 'Precip', urtitle = '5°N-20°N\n60°E-75°E', ylabel = 'mm/day')

axs[2].plot(olr_ac_b, zorder = 0)
axs[2].scatter(olr_ac_b, marker = 'x', zorder = 1)
axs[2].format(ylim = (-35,35), ylocator = 10, title = 'OLR', urtitle = '5°N-20°N\n50°E-60°E', ylabel = 'W/m^2')
axs[3].plot(pre_ac_b, zorder = 0)
axs[3].scatter(pre_ac_b, marker = 'x', zorder = 1)
axs[3].format(ylim = (-4,4), ylocator = 1, title = 'Precip', urtitle = '5°N-20°N\n50°E-60°E', ylabel = 'mm/day')

axs.format(xlocator=np.arange(1,13), grid = False, tickminor = False, abc = 'a)', abcloc = 'ul')


# %%
ch = ""
fpre = xr.open_dataset(
    ch + "/home/ys17-23/chenhj/monsoon/pyear/GPCC_r144x72_1979-2020.nc"
)
pre = fpre["precip"]
pre_m = pre.groupby("time.month").mean()
premaxidx = pre_m.idxmax(dim = "month", skipna = True)
preminidx = pre_m.idxmin(dim = "month", skipna = True)

premaxn1 = premaxidx - 1
premaxn1 = xr.where(premaxn1 == 0, 12, premaxn1)
premaxn3 = premaxidx + 1
premaxn3 = xr.where(premaxn3 == 13, 1, premaxn3)

premax_sum = pre_m.sel(month = premaxn1) + pre_m.sel(month = premaxidx) + pre_m.sel(month = premaxn3)

preminn1 = preminidx - 1
preminn1 = xr.where(preminn1 == 0, 12, preminn1)
preminn3 = preminidx + 1
preminn3 = xr.where(preminn3 == 13, 1, preminn3)

premin_sum = pre_m.sel(month = preminn1) + pre_m.sel(month = preminidx) + pre_m.sel(month = preminn3)

pre_ar = premax_sum - premin_sum
pre_ar = pre_ar / 3

pre_DJF = pre_m[0] + pre_m[1] + pre_m[11]
pre_JJA = pre_m[5] + pre_m[6] + pre_m[7]

pre_arWD = pre_JJA - pre_DJF
pre_arWD = xr.where(pre_arWD.lat < 0, (-1 * pre_arWD), pre_arWD)
pre_arWD = pre_arWD / 3.0


# %%
pplt.rc.grid = False
pplt.rc.reso = "lo"


# array = [1, 1, 2, 2]
fig = pplt.figure(refwidth=1.8)

# 以下为地理图的坐标轴设置
proj = pplt.PlateCarree()
axs = fig.subplots(ncols=2, nrows=2, proj=proj, wspace=3)
xticks = np.array([40, 55, 70, 85, 100, 115])
yticks = np.array([-10, 0, 10, 20, 30])
axs.format(coast=True, coastlinewidth=0.8, lonlim=(40, 120), latlim=(-10, 30))
axs.set_xticks(xticks)
axs.set_yticks(yticks)
lon_formatter = LongitudeFormatter(zero_direction_label=True)
lat_formatter = LatitudeFormatter()
axs.minorticks_on()
xminorLocator = MultipleLocator(5)
yminorLocator = MultipleLocator(5)
for ax in axs:
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    ax.xaxis.set_minor_locator(xminorLocator)
    ax.yaxis.set_minor_locator(yminorLocator)
    ax.outline_patch.set_linewidth(1.0)
axs.tick_params(
    axis="both",
    which="major",
    labelsize=8,
    direction="out",
    length=3,
    width=0.8,
    pad=0.2,
    top=True,
    right=True,
)
axs.tick_params(
    axis="both",
    which="minor",
    direction="out",
    length=2,
    width=0.8,
    top=True,
    right=True,
)
axs.format(abc=True, abcloc="ul", suptitle="SST & OLR")


m = axs[0].pcolormesh(premaxidx, cmap="ColdHot", vmin = 1, vmax = 12)
m = axs[1].pcolormesh(preminidx, cmap="ColdHot", vmin = 1, vmax = 12)
fig.colorbar(m, loc="r", span=1, label="month", width=0.11, ticklen=0, ticklabelsize=5)
m = axs[2].contourf(pre_ar, cmap = "ColdHot", vmin = -12,vmax = 12, extend = "both")
m = axs[3].contourf(pre_arWD, cmap = "ColdHot", vmin = -12, vmax = 12, extend = "both")
fig.colorbar(m, loc="r", span=2, label="mm/day", width=0.11, ticklen=0, ticklabelsize=5)
# fig.colorbar(m, loc="b", span=2, label="mm/day", width=0.11, ticklen=0, ticklabelsize=5)
# fig.colorbar(m, loc="r", span=1, label="degree", width=0.11, ticklen=0, ticklabelsize=5)
fig.format(abc="a)", abcloc="ul", abcborder=True, suptitle="annual range")


pplt.rc.reset()
# %%
