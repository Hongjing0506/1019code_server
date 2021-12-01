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
from cartopy.util import add_cyclic_point
from matplotlib.ticker import MultipleLocator
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
from scipy import stats

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


def filplonlat(ds):
    # To facilitate data subsetting
    # print(da.attrs)
    """
    print(
        f'\n\nBefore flip, lon range is [{ds["lon"].min().data}, {ds["lon"].max().data}].'
    )
    ds["lon"] = ((ds["lon"] + 180) % 360) - 180
    # Sort lons, so that subset operations end up being simpler.
    ds = ds.sortby("lon")
    """
    ds = ds.sortby("lat", ascending=True)
    # print(ds.attrs)
    print('\n\nAfter sorting lat values, ds["lat"] is:')
    print(ds["lat"])
    return ds


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


def mapart(ax, extents):
    proj = ccrs.PlateCarree()
    ax.coastlines(color="k", lw=1.5)
    ax.add_feature(cfeature.LAND, edgecolor="black", facecolor="white")
    ax.set_extent(extents, crs=proj)
    xticks = np.arange(extents[0], extents[1] + 1, 20)
    yticks = np.arange(extents[2], extents[3] + 1, 10)
    # 这里的间隔需要根据自己实际调整
    ax.set_xticks(xticks, crs=proj)
    ax.set_yticks(yticks, crs=proj)
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    xminorLocator = MultipleLocator(5)
    yminorLocator = MultipleLocator(10)
    ax.yaxis.set_major_formatter(lat_formatter)
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_minor_locator(xminorLocator)
    ax.xaxis.set_minor_locator(yminorLocator)
    ax.tick_params(
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
    # 为了便于在不同的场景中使用，这里使用了一个全局变量gl_font
    ax.minorticks_on()
    ax.tick_params(
        axis="both",
        which="minor",
        direction="out",
        length=3.0,
        width=0.8,
        top=False,
        right=False,
    )
    ax.outline_patch.set_linewidth(1.0)


def detrend_dim(da, dim, deg, trend):
    # detrend along a single dimension
    p = da.polyfit(dim=dim, deg=1, skipna=True)
    fit = xr.polyval(da[dim], p.polyfit_coefficients)
    if trend == False:
        return da - fit
    elif trend == True:
        return fit


def dim_linregress(x, y):
    # returns: slope,intercept,rvalue,pvalue,hypothesis
    return xr.apply_ufunc(
        stats.linregress,
        x,
        y,
        input_core_dims=[["time"], ["time"]],
        output_core_dims=[[], [], [], [], []],
        vectorize=True,
        dask="parallelized",
    )


def plt_sig(da, ax, n, area):
    da_cyc, lon_cyc = add_cyclic_point(da[::n, ::n], coord=da.lon[::n])
    nx, ny = np.meshgrid(lon_cyc, da.lat[::n])
    sig = ax.scatter(
        nx[area], ny[area], marker=".", s=9, c="black", alpha=0.6, transform=proj
    )


# %%
# 读取数据

ch = ""
fpre = xr.open_dataset(
    ch + "/home/ys17-23/chenhj/monsoon/pyear/GPCC_r144x72_1979-2020.nc"
)
pre = fpre["precip"]

lmask = ch + "/home/ys17-23/chenhj/monsoon/pyear/lsmask72x144.nc"

fu = xr.open_dataset(
    ch + "/home/ys17-23/chenhj/monsoon/pyear/ERA5u_r144x72_1979-2020.nc"
)
u = fu["u"]
u850 = u.loc[:, 850, :, :]
u200 = u.loc[:, 200, :, :]

fv = xr.open_dataset(
    ch + "/home/ys17-23/chenhj/monsoon/pyear/ERA5v_r144x72_1979-2020.nc"
)
v = fv["v"]
v850 = v.loc[:, 850, :, :]
v200 = v.loc[:, 200, :, :]

fhgt = xr.open_dataset(
    ch + "/home/ys17-23/chenhj/monsoon/pyear/ERA5hgt_r144x72_1979-2020.nc"
)
hgt = fhgt["z"]
hgt = hgt / 9.80665
hgt850 = hgt.loc[:, 850, :, :]
hgt200 = hgt.loc[:, 200, :, :]

fhadisst = xr.open_dataset(
    ch + "/home/ys17-23/chenhj/monsoon/pyear/HadISST_r144x72_1979-2020.nc"
)
hadisst = fhadisst["sst"]

fersst = xr.open_dataset(
    ch + "/home/ys17-23/chenhj/monsoon/pyear/ERSSTv5_r144x72_1979-2020.nc"
)
ersst = fersst["sst"].loc[:, 0.0, :, :]
ersst["time"] = hadisst["time"]
print(ersst)
# %%
#   calculate monsoon area
pre_ac = p_month(pre, 1, 12).mean(dim="time")
pre_Jan = pre.groupby("time.month")[1].mean(dim="time", skipna=True)
pre_max = pre_ac.max(dim="month", skipna=True)
pre_RR = pre_max - pre_Jan


# %%
#   Indian Ocean monsoon area
ma = pre.where(pre_RR > 5.00)
IOSM_pre = lsmask(ma, lmask, "ocean").loc[:, 0:30, 60:80]
# IOSM_pre = lsmask(ma, lmask, "ocean").loc[:, 0:25, 65:75]
ISM_pre = lsmask(ma, lmask, "land").loc[:, 0:30, 70:87]

uma = u850.where(pre_RR > 5.00)
IOSM_u = lsmask(uma, lmask, "ocean").loc[:, 0:30, 60:80]
ISM_u = lsmask(uma, lmask, "land").loc[:, 0:30, 70:87]


# %%
#   calculate annual cycle
IOSMac = p_month(IOSM_pre, 1, 12).mean(dim=["time", "lat", "lon"], skipna=True)
# IOac = p_month(ma.loc[:, 0:25, 65:75], 1, 12).mean(
#     dim=["time", "lat", "lon"], skipna=True
# )
ISMac = p_month(ISM_pre, 1, 12).mean(dim=["time", "lat", "lon"], skipna=True)

IOSM_uac = p_month(IOSM_u, 1, 12).mean(dim=["time", "lat", "lon"], skipna=True)
IOSM_uac = IOSM_uac - IOSM_uac.mean(skipna=True)
ISM_uac = p_month(ISM_u, 1, 12).mean(dim=["time", "lat", "lon"], skipna=True)
ISM_uac = ISM_uac - ISM_uac.mean(skipna=True)

# %%
#   different month for hgt & uv
hgt850_month = p_month(hgt850, 5, 9).mean(dim="time", skipna=True)
# hgt850_month = hgt850_month - hgt850.mean(dim = "time")
u850_month = p_month(u850, 5, 9).mean(dim="time", skipna=True)
# u850_month = u850_month - u850.mean(dim = "time")
v850_month = p_month(v850, 5, 9).mean(dim="time", skipna=True)
# v850_month = v850_month - v850.mean(dim = "time")

hgt200_month = p_month(hgt200, 5, 9).mean(dim="time", skipna=True)
u200_month = p_month(u200, 5, 9).mean(dim="time", skipna=True)
v200_month = p_month(v200, 5, 9).mean(dim="time", skipna=True)
# %%
#   plot the annual cycle
pplt.rc.grid = False
pplt.rc.reso = "lo"

proj = pplt.PlateCarree()
widths = 2
heights = [2, 3]
fig = pplt.figure(span=False, share=False, refwidth=4.0)
axs = fig.subplots(
    ncols=1,
    nrows=2,
    wspace=4.0,
    hspace=4.0,
    proj=[proj, None],
    wratios=widths,
    hratios=heights,
)

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


axs[0].xaxis.set_major_formatter(lon_formatter)
axs[0].yaxis.set_major_formatter(lat_formatter)
axs[0].xaxis.set_minor_locator(xminorLocator)
axs[0].yaxis.set_minor_locator(yminorLocator)
axs[0].outline_patch.set_linewidth(1.0)
axs[0].tick_params(
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
axs[0].tick_params(
    axis="both",
    which="minor",
    direction="out",
    length=3.0,
    width=0.8,
    top=False,
    right=False,
)

axs[0].contour(pre_RR, c="black", vmin=5, vmax=5, lw=1.0)
axs[0].pcolormesh(IOSM_pre.mean(dim=["time"], skipna=True), extend="both", color="red")
axs[0].pcolormesh(ISM_pre.mean(dim=["time"], skipna=True), extend="both", color="blue")
axs[0].format(title="monsoon area", titleloc="l")


m1 = axs[1].plot(IOSMac, color="red", marker="o", zorder=1, markersize=3.0)
m2 = axs[1].plot(ISMac, color="blue", marker="o", zorder=2, markersize=3.0)
axs[1].axhline(5, color="black", linewidth=0.8, zorder=0)
ox = axs[1].alty(color="black", label="m/s", linewidth=1)
m3 = ox.line(
    IOSM_uac, color="red", marker="o", zorder=1, markersize=3.0, linestyle="--"
)
m4 = ox.line(
    ISM_uac, color="blue", marker="o", zorder=2, markersize=3.0, linestyle="--"
)
ox.format(ylim=(-15, 15), ylocator=3, tickminor=False)
axs[1].format(
    ylim=(0, 10),
    ylocator=2,
    ylabel="mm/day",
    xlim=(1, 12),
    xlocator=1,
    grid=False,
    ytickminor=True,
    xtickminor=False,
    titleloc="l",
    title="annual cycle",
)
axs[1].legend(
    handles=[m1, m2, m3, m4],
    labels=["IOSM_pre", "ISM_pre", "IOSM_u850", "ISM_u850"],
    lw=0.6,
    loc="ur",
    ncols=1,
    markersize=2.5,
    fontsize=0.5,
    frame=False,
    center=None,
)
fig.format(abcloc="l", abc="a)")

# %%
# plot the hgt & u, v
pplt.rc.grid = False
pplt.rc.reso = "lo"

proj = pplt.PlateCarree()

array = [[1, 1, 2, 2], [3, 3, 4, 4], [0, 5, 5, 0]]
fig = pplt.figure(span=False, share=False)
axs = fig.subplots(array, proj=proj, wspace=4.0, hspace=4.0)

xticks = np.array([40, 60, 80, 100, 120])
yticks = np.array([0, 10, 20, 30, 40])
axs.format(
    coast=True, coastlinewidth=0.8, lonlim=(40, 120), latlim=(0, 40), coastzorder=1
)
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
    ax.tick_params(
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
    ax.tick_params(
        axis="both",
        which="minor",
        direction="out",
        length=3.0,
        width=0.8,
        top=False,
        right=False,
    )

w, h = 0.12, 0.14
for i, ax in enumerate(axs):
    rect = Rectangle(
        (1 - w, 0), w, h, transform=ax.transAxes, fc="white", ec="k", lw=0.5, zorder=1.1
    )
    con = ax.contourf(
        hgt850_month[i, :, :],
        cmap="ColdHot",
        extend="both",
        vmin=1400,
        vmax=1560,
        levels=np.arange(1400, 1570, 20),
    )
    m = ax.quiver(
        u850_month[i, :, :],
        v850_month[i, :, :],
        zorder=1,
        headwidth=4,
        scale_units="xy",
        scale=3,
        pivot="mid",
        minlength=1.0,
    )
    ax.add_patch(rect)
    qk = ax.quiverkey(
        m,
        X=1 - w / 2,
        Y=0.7 * h,
        U=8,
        label="8 m/s",
        labelpos="S",
        labelsep=0.02,
        fontproperties={"size": 5},
        zorder=3.1,
    )
    title = ["MAY", "JUN", "JUL", "AUG", "SEP"]
    ax.format(ltitle=title[i])
fig.colorbar(con, loc="b", label="m")
fig.format(suptitle="hgt & wind in 850hPa", abcloc="l", abc="a)")


# %%
#   plot the 200hPa hgt & u,v wind
pplt.rc.grid = False
pplt.rc.reso = "lo"

proj = pplt.PlateCarree()

array = [[1, 1, 2, 2], [3, 3, 4, 4], [0, 5, 5, 0]]
fig = pplt.figure(span=False, share=False)
axs = fig.subplots(array, proj=proj, wspace=4.0, hspace=4.0)

xticks = np.array([40, 60, 80, 100, 120])
yticks = np.array([0, 10, 20, 30, 40])
axs.format(
    coast=True, coastlinewidth=0.8, lonlim=(40, 120), latlim=(0, 40), coastzorder=1
)
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
    ax.tick_params(
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
    ax.tick_params(
        axis="both",
        which="minor",
        direction="out",
        length=3.0,
        width=0.8,
        top=False,
        right=False,
    )

w, h = 0.12, 0.14
for i, ax in enumerate(axs):
    rect = Rectangle(
        (1 - w, 0), w, h, transform=ax.transAxes, fc="white", ec="k", lw=0.5, zorder=1.1
    )
    con = ax.contourf(
        hgt200_month[i, :, :],
        cmap="ColdHot",
        extend="both",
        vmin=12100,
        vmax=12600,
        levels=np.arange(12100, 12600, 20),
    )
    ax.contour(
        hgt200_month[i, :, :], color="black", levels=np.arange(12500, 12541, 20), lw=0.7
    )
    m = ax.quiver(
        u200_month[i, :, :],
        v200_month[i, :, :],
        zorder=1,
        headwidth=4,
        scale_units="xy",
        scale=4,
        pivot="mid",
        minlength=1.0,
    )
    ax.add_patch(rect)
    qk = ax.quiverkey(
        m,
        X=1 - w / 2,
        Y=0.7 * h,
        U=8,
        label="8 m/s",
        labelpos="S",
        labelsep=0.02,
        fontproperties={"size": 5},
        zorder=3.1,
    )
    title = ["MAY", "JUN", "JUL", "AUG", "SEP"]
    ax.format(ltitle=title[i])
fig.colorbar(con, loc="b", label="m")
fig.format(suptitle="hgt & wind in 200hPa", abcloc="l", abc="a)")


# %%
#  calculate interannual variation and linear trend
IOSMiv = p_month(IOSM_pre, 6, 7).mean(dim=["month", "lat", "lon"], skipna=True)
IOSMivstd = IOSMiv.std()
ISMiv = p_month(ISM_pre, 6, 9).mean(dim=["month", "lat", "lon"], skipna=True)
ISMivstd = ISMiv.std()
year = np.arange(1979, 2021, 1)
st = np.std(year)
IOSMiv.coords["time"] = year
ISMiv.coords["time"] = year
IOSMivtrend = IOSMiv.polyfit(
    dim="time", deg=1, skipna=True, full=True
).polyfit_coefficients.loc[1]
IOSMivb = IOSMiv.polyfit(
    dim="time", deg=1, skipna=True, full=True
).polyfit_coefficients.loc[0]

IOSMivr = IOSMivtrend * st / IOSMivstd

ISMivtrend = ISMiv.polyfit(
    dim="time", deg=1, skipna=True, full=True
).polyfit_coefficients.loc[1]
ISMivb = ISMiv.polyfit(
    dim="time", deg=1, skipna=True, full=True
).polyfit_coefficients.loc[0]

ISMivr = ISMivtrend * st / ISMivstd

print("IOSMivtrend = ", IOSMivtrend, "\n ISMivtrend = ", ISMivtrend)
print("IOSMivr = ", IOSMivr, "\n ISMivr = ", ISMivr)

#   another way to calculate linear trend
# IOSMivtrend1 = dim_linregress(np.arange(1979, 2021, 1), IOSMiv)
# print(IOSMivtrend1[0])
# %%
#   plot the interannual variation and linear trend
fig3 = pplt.figure(span=False, share=False, refwidth=4.0)
axs = fig3.subplots(ncols=1, nrows=1)
axs[0].plot(IOSMiv, color="red", zorder=1)
axs[0].plot(ISMiv, color="blue", zorder=2)
axs[0].format(
    ylim=(4, 10),
    ylabel="mm/day",
    grid=False,
    ytickminor=False,
    xtickminor=True,
    titleloc="l",
    title="interannual variability",
)

# %%
#   calculate interannual linear trend map
# r_lim in 95% is 0.3042
startmon = 5
endmon = 9
preiv = p_month(pre, startmon, endmon).mean(dim="month", skipna=True)
year = np.arange(1979, 2021, 1)
preiv.coords["time"] = year
# print(preiv)

#   another calculation way: xarray.polyfit
# preivtrend = preiv.polyfit(dim="time", deg=1, skipna=True, full=True)

# preivslope = dim_linregress(np.arange(1979, 2021, 1), preiv)
# print(preivslope[0])
preivsl, preivin, preivrv, preivpv, preivhy = dim_linregress(year, preiv)
print(preivpv)

#  calculate hgt & wind linear trend
hgt850iv = p_month(hgt850, startmon, endmon).mean(dim="month", skipna=True)
hgt850iv.coords["time"] = year
hgt850ivsl, hgt850ivin, hgt850ivrv, hgt850ivpv, hgt850ivhy = dim_linregress(
    year, hgt850iv
)

u850iv = p_month(u850, startmon, endmon).mean(dim="month", skipna=True)
u850iv.coords["time"] = year
u850ivsl, u850ivin, u850ivrv, u850ivpv, u850ivhy = dim_linregress(year, u850iv)

v850iv = p_month(v850, startmon, endmon).mean(dim="month", skipna=True)
v850iv.coords["time"] = year
v850ivsl, v850ivin, v850ivrv, v850ivpv, v850ivhy = dim_linregress(year, v850iv)

hgt200iv = p_month(hgt200, startmon, endmon).mean(dim="month", skipna=True)
hgt200iv.coords["time"] = year
hgt200ivsl, hgt200ivin, hgt200ivrv, hgt200ivpv, hgt200ivhy = dim_linregress(
    year, hgt200iv
)

u200iv = p_month(u200, startmon, endmon).mean(dim="month", skipna=True)
u200iv.coords["time"] = year
u200ivsl, u200ivin, u200ivrv, u200ivpv, u200ivhy = dim_linregress(year, u200iv)

v200iv = p_month(v200, startmon, endmon).mean(dim="month", skipna=True)
v200iv.coords["time"] = year
v200ivsl, v200ivin, v200ivrv, v200ivpv, v200ivhy = dim_linregress(year, v200iv)
# %%
#   plot linear trend map
pplt.rc.grid = False
pplt.rc.reso = "lo"

proj = pplt.PlateCarree()

# array = [[1, 1, 2, 2], [3, 3, 4, 4], [0, 5, 5, 0]]
fig4 = pplt.figure(span=False, share=False)
axs = fig4.subplots(ncols=2, nrows=2, proj=proj, wspace=4.0, hspace=4.0)

xticks = np.array([40, 60, 80, 100, 120])
yticks = np.array([0, 10, 20, 30, 40])
axs.format(
    coast=True, coastlinewidth=0.8, lonlim=(40, 120), latlim=(0, 40), coastzorder=1
)
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
    ax.tick_params(
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
    ax.tick_params(
        axis="both",
        which="minor",
        direction="out",
        length=3.0,
        width=0.8,
        top=False,
        right=False,
    )

m = axs[0].contourf(preivrv, cmap="ColdHot", levels=np.arange(-1.0, 1.1, 0.1))
axs[0].contour(pre_RR, c="black", vmin=5, vmax=5, lw=1.0)
n = 1
plt_sig(preivpv, axs[0], n, np.where(preivpv[::n, ::] <= 0.05))
axs[0].format(ltitle = "precip")

w, h = 0.12, 0.14
rect = Rectangle(
    (1 - w, 0), w, h, transform=axs[1].transAxes, fc="white", ec="k", lw=0.5, zorder=1.1
)
axs[1].contourf(hgt850ivrv, cmap="ColdHot", levels=np.arange(-1.0, 1.1, 0.1), zorder = 0)
axs[1].contour(
    hgt850ivpv, color="green", vmin=0.05, vmax=0.05, zorder=0.5, linewidth=0.5
)
qu = axs[1].quiver(
    u850ivrv.where(u850ivpv <= 0.05),
    v850ivrv.where(v850ivpv <= 0.05),
    zorder=1,
    headwidth=4,
    scale_units="xy",
    scale=0.3,
    pivot="mid",
    minlength=1.0
)
axs[1].quiver(
    u850ivrv,
    v850ivrv,
    zorder=0.8,
    headwidth=4,
    scale_units="xy",
    scale=0.3,
    pivot="mid",
    minlength=1.0,
    color = "grey"
)
axs[1].add_patch(rect)
qk = axs[1].quiverkey(
    qu,
    X=1 - w / 2,
    Y=0.7 * h,
    U=1,
    label="1 m/s",
    labelpos="S",
    labelsep=0.02,
    fontproperties={"size": 5},
    zorder=3.1,
)
axs[1].format(ltitle = "hgt and wind", rtitle = "850hPa")

fig4.colorbar(m, loc="b", ticklen=0, ticklabelsize=5, width=0.11, label="")
fig4.format(suptitle="linear trend", abcloc="l", abc="a)")
# %%
#   calculate SST linear tendency in different month
SSTiv = p_month(ersst, 1, 12)
year = np.arange(1979, 2021, 1)
SSTiv.coords["time"] = year
sstivsl, sstivin, sstivrv, sstivpv, sstivhy = dim_linregress(year, SSTiv)
print(sstivrv)
# %%
#   plot the SST linear tendency
pplt.rc.grid = False
pplt.rc.reso = "lo"

proj = pplt.PlateCarree()

# array = [[1, 1, 2, 2], [3, 3, 4, 4], [0, 5, 5, 0]]
fig5 = pplt.figure(span=False, share=False)
axs = fig5.subplots(ncols=3, nrows=4, proj=proj, wspace=4.0, hspace=4.0)

xticks = np.array([40, 60, 80, 100, 120])
yticks = np.array([0, 10, 20, 30, 40])
axs.format(
    coast=True, coastlinewidth=0.8, lonlim=(40, 120), latlim=(0, 40), coastzorder=1
)
axs.set_xticks(xticks)
axs.set_yticks(yticks)
lon_formatter = LongitudeFormatter(zero_direction_label=True)
lat_formatter = LatitudeFormatter()
axs.minorticks_on()
xminorLocator = MultipleLocator(5)
yminorLocator = MultipleLocator(10)

month = [
    "JAN",
    "FEB",
    "MAR",
    "APR",
    "MAY",
    "JUN",
    "JUL",
    "AUG",
    "SEP",
    "OBT",
    "NOV",
    "DEC",
]
for i, ax in enumerate(axs):
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    ax.xaxis.set_minor_locator(xminorLocator)
    ax.yaxis.set_minor_locator(yminorLocator)
    ax.outline_patch.set_linewidth(1.0)
    ax.tick_params(
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
    ax.tick_params(
        axis="both",
        which="minor",
        direction="out",
        length=3.0,
        width=0.8,
        top=False,
        right=False,
    )
    m = ax.contourf(sstivrv[i, :, :], cmap="ColdHot", levels=np.arange(-1.0, 1.1, 0.1))
    n = 1
    plt_sig(sstivpv, ax, n, np.where(sstivpv[i, ::n, ::] < 0.05))
    ax.format(titleloc="l", title=month[i])
w, h = 0.12, 0.14


# axs[0].contour(pre_RR, c = "black", vmin = 5, vmax = 5, lw = 1.0)
fig5.colorbar(m, ticklen=0, ticklabelsize=5, width=0.11, label="", loc="b")
fig5.format(suptitle="SST linear trend", abcloc="l", abc="a)")
# %%
#   calculate hgt, wind linear trend
preiv = p_month(pre, 6, 7).mean(dim="month", skipna=True)
year = np.arange(1979, 2021, 1)
preiv.coords["time"] = year
# print(preiv)
preivtrend = preiv.polyfit(dim="time", deg=1, skipna=True, full=True)

print(preivtrend.polyfit_coefficients)
# preivslope = dim_linregress(np.arange(1979, 2021, 1), preiv)
preivsl, preivin, preivrv, preivpv, preivhy = dim_linregress(year, preiv)
print(preivpv)
