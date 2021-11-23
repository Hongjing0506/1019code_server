'''
Author: ChenHJ
Date: 2021-11-22 16:33:19
LastEditors: ChenHJ
LastEditTime: 2021-11-23 17:41:18
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
    
# %%
# 读取数据

ch = ""
fpre = xr.open_dataset(
    ch + "/home/ys17-23/chenhj/monsoon/pyear/GPCC_r144x72_1979-2020.nc"
)
pre = fpre["precip"]

# %%
#   计算Total Rainfall May to Sep(mm/day)
pre_59sum = p_time(pre, 5, 9, False).sum(dim = "time") / 42.0 / 5.0


# %%
#   计算Annual range (mm/day)
pre_ac = p_month(pre, 1, 12).mean(dim = "time")
pre_ar = pre_ac.max(dim = "month", skipna = True) - pre_ac.min(dim = "month", skipna = True)


# %%
#   计算Rainfall ratio of (May to Sep)/year )(%)
pre_y = pre_ac.sum(dim = "month", skipna = True)
pre_ratio = pre_59sum / pre_y

# %%
#   计算RR index = R_i - R_J
pre_Jan = pre.groupby("time.month")[1].mean(dim = "time", skipna = True)
pre_max = pre_ac.max(dim = "month", skipna = True)
pre_RR = pre_max - pre_Jan
# %%
#   画图
pplt.rc.grid = False
pplt.rc.reso = "lo"


# array = [1, 1, 2, 2]
fig = pplt.figure()

# 以下为地理图的坐标轴设置
proj = pplt.PlateCarree()
axs = fig.subplots(ncols=2, nrows=2, proj=proj, wspace=4.0, hspace = 4.0)
xticks = np.array([40, 60, 80, 100, 120, 140, 160, 180])
yticks = np.array([0, 10, 20, 30, 40, 50])
axs.format(coast=True, coastlinewidth=0.8, lonlim=(40, 180), latlim=(0, 50))
axs.set_xticks(xticks)
axs.set_yticks(yticks)
lon_formatter = LongitudeFormatter(zero_direction_label=True)
lat_formatter = LatitudeFormatter()
axs.minorticks_on()
xminorLocator = MultipleLocator(5)
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
    length=4.0,
    width=0.8,
    pad=2.0,
    top=False,
    right=False,
)
axs.tick_params(
    axis="both",
    which="minor",
    direction="out",
    length=3.0,
    width=0.8,
    top=False,
    right=False,
)


axs[0].contourf(pre_59sum, cmap="Greys", colorbar = "b", colorbar_kw = {"ticklen": 0, "ticklabelsize": 5, "width": 0.11, "label": ""}, extend = "both", vmin = 2.0, vmax = 17.0, levels = np.arange(2.0, 18.0, 3.0))
axs[0].format(title = "Total Rainfall May to Sep", titleloc = 'l', rtitle = "mm/day")
axs[0].contour(pre_59sum, c = "black", vmin = 2.0, vmax = 2.0, lw = 1.0)
# axs.colorbar(m, ticklen = 0, ticklabelsize = 5)
axs[1].contourf(pre_ar, cmap="Greys", vmin = 5, vmax = 17, colorbar = "b", colorbar_kw = {"ticklen": 0, "ticklabelsize": 5, "width": 0.11, "label": ""}, extend = "both", levels = np.arange(5.0, 18.0, 2.0))
axs[1].format(title = "Annual range", titleloc = 'l', rtitle = "mm/day")
# axs.colorbar(m, ticklen = 0, ticklabelsize = 5)
# fig.colorbar(m, loc="b", span=1, label="month", width=0.11, ticklen=0, ticklabelsize=5)
axs[2].contourf(pre_ratio, cmap = "Greys", vmin = 0, vmax = 1, extend = "both", colorbar = "b", colorbar_kw = {"ticklen": 0, "ticklabelsize": 5, "width": 0.11, "label": ""})
axs[2].contour(pre_ratio, c = "black", vmin = 0.55, vmax = 0.55, lw = 1.0)
axs[2].format(title = "Rainfall ratio of (May to Sep)/year", titleloc = 'l', rtitle = "%")
# axs.colorbar(m, ticklen = 0, ticklabelsize = 5)
axs[3].contourf(pre_RR, cmap = "Greys", vmin = 0, vmax = 18, values = np.arange(0.0, 19.0, 1.0), extend = "both", colorbar = "b", colorbar_kw = {"ticklen": 0, "ticklabelsize": 5, "width": 0.11, "label": ""})
axs[3].contour(pre_RR, c = "black", vmin = 5, vmax = 5, lw = 1.0)
axs[3].format(title = "Monsoon Annual Range", titleloc = 'l', rtitle = "mm/day")
# axs.colorbar(m, ticklen = 0, ticklabelsize = 5)
# fig.colorbar(m, loc="b", span=2, label="mm/day", width=0.11, ticklen=0, ticklabelsize=5)
# fig.colorbar(m, loc="b", span=2, label="mm/day", width=0.11, ticklen=0, ticklabelsize=5)
# fig.colorbar(m, loc="r", span=1, label="degree", width=0.11, ticklen=0, ticklabelsize=5)
fig.format(abc="a)", abcloc="l", abcborder=True, suptitle = "Monsoon Annual Range")


pplt.rc.reset()
# %%
#   Reading the ERA5 data
ch = ""
ferapre = xr.open_dataset(ch + "/home/ys17-23/chenhj/monsoon/pyear/pensyn_tp.197901-201412.nc")
era = ferapre["tp"]
era = era * 1000.0 * 24.0

# %%
