'''
Author: ChenHJ
Date: 2021-11-22 16:33:19
LastEditors: ChenHJ
LastEditTime: 2021-11-22 16:58:36
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
pre_59sum = p_time(pre, 5, 9, False).sum(dim = "time", skipna = True) / 42.0
pre_59sum_in = pre_59sum.loc[0:20, 40:180]

# %%
