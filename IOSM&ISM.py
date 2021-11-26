"""
Author: ChenHJ
Date: 2021-11-25 00:38:16
LastEditors: ChenHJ
LastEditTime: 2021-11-25 17:29:42
FilePath: /ys17-23/chenhj/1019code/IOSM&ISM.py
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
    n_data = data.sel(
        time=(data.time.dt.month <= mon_end) * (data.time.dt.month >= mon_s)
    )
    n_mon = mon_end - mon_s + 1
    if meanon == True:
        n_data_mean = n_data.coarsen(time=n_mon).mean()
        return n_data_mean
    elif meanon == False:
        return n_data
    else:
        print("Bad argument: meanon")


"""
description: 
    本函数用于将需要的月份挑选出来，并存储为月份x年份xlatxlon的形式
param {*} data
param {*} mon_s
param {*} mon_e
return {*}
"""


def p_month(data, mon_s, mon_e):
    import pandas as pd
    import xarray as xr

    time = data["time"]
    data.transpose("time", ...)
    year_s = pd.to_datetime(time).year[1]
    year_e = pd.to_datetime(time).year[-1]
    nyear = pd.date_range(str(year_s), str(year_e), freq="AS")
    m_ind = data.groupby("time.month").groups[mon_s]
    res = data[m_ind]
    res["time"] = nyear
    for i in np.arange(mon_s + 1, mon_e + 1):
        m_ind = data.groupby("time.month").groups[i]
        tmp = data[m_ind]
        tmp["time"] = nyear
        res = xr.concat([res, tmp], "month")

    month = np.arange(mon_s, mon_e + 1)
    res["month"] = month
    return res


def lsmask(ds, lsdir, label):
    with xr.open_dataset(lsdir) as f:
        da = f["mask"][0]
    landsea = filplonlat(da)
    ds.coords["mask"] = (("lat", "lon"), landsea.values)
    if label == "land":
        ds = ds.where(ds.mask < 1)
    elif label == "ocean":
        ds = ds.where(ds.mask > 0)
    del ds["mask"]
    return ds


# %%
# 读取数据

ch = ""
fpre = xr.open_dataset(
    ch + "/home/ys17-23/chenhj/monsoon/pyear/GPCC_r144x72_1979-2020.nc"
)
pre = fpre["precip"]

lmask = ch + "/home/ys17-23/chenhj/monsoon/pyear/lsmask.nc"

# %%
#   calculate monsoon area
pre_ac = p_month(pre, 1, 12).mean(dim="time")
pre_Jan = pre.groupby("time.month")[1].mean(dim="time", skipna=True)
pre_max = pre_ac.max(dim="month", skipna=True)
pre_RR = pre_max - pre_Jan


# %%
#   Indian Ocean monsoon area
IOpre = pre.loc[:, 5:20, 65:75]


#   calculate annual cycle
IOac = p_month(IOpre, 1, 12).mean(dim=["time", "lat", "lon"], skipna=True)


# %%
#   plot the annual cycle
fig = pplt.figure(span=False, share=False)
axs = fig.subplots(ncols=2, nrows=2, wspace=4.0, hspace=4.0)
axs[0].plot(IOac, zorder=0)
axs[0].scatter(IOac, marker="x", zorder=2)
axs[0].format(
    ylim=(0, 10),
    ylocator=1,
    title="IOSM annual cycle",
    urtitle="5°N-20°N\n65°E-75°E",
    ylabel="mm/day",
    xlim=(1, 12),
    xlocator=1,
    grid=False,
    tickminor=False,
)
# %%
