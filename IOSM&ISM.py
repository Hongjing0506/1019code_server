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
from matplotlib.patches import Rectangle
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
    ax.coastlines(color='k', lw=1.5)
    ax.add_feature(cfeature.LAND, edgecolor='black', facecolor='white')
    ax.set_extent(extents, crs=proj)
    xticks = np.arange(extents[0], extents[1]+1, 20)
    yticks = np.arange(extents[2], extents[3]+1, 10)
    #这里的间隔需要根据自己实际调整
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
    ax.tick_params(axis='both', which='major', labelsize=8, direction='out',
                    length=4.0, width=0.8, pad=2.0, top=False, right=False)
    #为了便于在不同的场景中使用，这里使用了一个全局变量gl_font
    ax.minorticks_on()
    ax.tick_params(axis='both', which='minor', direction='out', length=3.0,
                    width=0.8, top=False, right=False)
    ax.outline_patch.set_linewidth(1.0)
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

fv = xr.open_dataset(
    ch + "/home/ys17-23/chenhj/monsoon/pyear/ERA5v_r144x72_1979-2020.nc"
)
v = fv["v"]
v850 = v.loc[:, 850, :, :]

fhgt = xr.open_dataset(
    ch + "/home/ys17-23/chenhj/monsoon/pyear/ERA5hgt_r144x72_1979-2020.nc"
)
hgt = fhgt["z"]
hgt = hgt / 9.80665
hgt850 = hgt.loc[:, 850, :, :]

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
ISM_uac = p_month(ISM_u, 1, 12).mean(dim=["time", "lat", "lon"], skipna=True)

# %%
#   different month for hgt & uv
hgt850_month = p_month(hgt850, 5, 9).mean(dim = "time", skipna = True)
# hgt850_month = hgt850_month - hgt850.mean(dim = "time")
u850_month = p_month(u850, 5, 9).mean(dim = "time", skipna = True)
u850_month = u850_month - u850.mean(dim = "time")
v850_month = p_month(v850, 5, 9).mean(dim = "time", skipna = True)
v850_month = v850_month - v850.mean(dim = "time")

# %%
#   plot the annual cycle
pplt.rc.grid = False
pplt.rc.reso = "lo"

proj = pplt.PlateCarree()

fig = pplt.figure(span=False, share=False)
axs = fig.subplots(ncols=2, nrows=1, wspace=4.0, hspace=4.0, proj=[proj, None])

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


axs[1].plot(IOSMac, color="red", marker="o", zorder=1, markersize=3.0)
axs[1].plot(ISMac, color="blue", marker="o", zorder=2, markersize=3.0)
axs[1].axhline(5, color="black", linewidth=0.8, zorder=0)

axs[1].format(
    ylim=(0, 13),
    ylocator=2,
    ylabel="mm/day",
    xlim=(1, 12),
    xlocator=1,
    grid=False,
    tickminor=False,
    titleloc="l",
    title="annual cycle",
)
axs[1].legend(
    labels=["IOSM", "ISM"],
    lw=1.0,
    loc="ur",
    ncols=1,
    markersize=3.0,
    fontsize=0.8,
    frame=False,
)

# %%
# plot the hgt & u, v
pplt.rc.grid = False
pplt.rc.reso = "lo"

proj = pplt.PlateCarree()

array = [[1,1,2,2],[3,3,4,4],[0,5,5,0]]
fig = pplt.figure(span=False, share=False)
axs = fig.subplots(array, proj=proj, wspace=4.0, hspace=4.0)

xticks = np.array([40, 60, 80, 100, 120])
yticks = np.array([0, 10, 20, 30, 40])
axs.format(coast=True, coastlinewidth=0.8, lonlim=(40, 120), latlim=(0, 40), coastzorder = 1)
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
for i,ax in enumerate(axs):
    rect = Rectangle(
        (1 - w, 0), w, h, transform=ax.transAxes,
        fc='white', ec='k', lw=0.5, zorder=1.1
    )
    con = ax.contourf(hgt850_month[i, :, :], cmap = "ColdHot", extend = "both", vmin = 1400, vmax = 1560, levels = np.arange(1400, 1570, 20))
    m = ax.quiver(u850_month[i, :, :], v850_month[i, :, :], zorder = 1, headwidth = 4, scale_units = 'xy', scale = 3, pivot = 'mid', minlength = 1.0)
    ax.add_patch(rect)
    qk = ax.quiverkey(
        m, X=1-w/2, Y=0.7*h, U=8,
        label='8 m/s', labelpos='S', labelsep=0.02,
        fontproperties={'size': 5}, zorder = 3.1
    )
fig.colorbar(con, loc = 'b', label = 'm')


# %%
# # plot hgt & UV with pyplot
# proj = ccrs.ccrs.PlateCarree()
# fig = plt.figure()
# axs = fig.add_subplot(111 , projection=proj)
# extents = [40,120,0,35]
# for ax in axs:
#     mapart1(ax, extents)
#     #填色图
#     im = ax.contourf(
#         lon, lat, da, transform=proj, extend='neither',cmap=cmaps.ViBlGrWhYeOrRe, 
#         alpha=0.8, levels=np.arange(-1., 1.09, 0.1), zorder=0
#     )
#     ax.set_title('left_title', fontdict=font, loc='left')
#     ax.set_title('right_title', fontdict=font, loc='right')
#     cbposition = fig.add_axes([0.45, 0.12, 0.015, 0.55])
#     # 设置色标位置。第一个调整左右位置，值越大越往右；第二个调整上下位置，值越大越高；第三个调整长度，左边位置不变，值越大越长
#     cb1 = fig.colorbar(im, cax=cbposition, orientation='vertical',
#                     spacing='proportional', format='%.1f', extend='both')
#     cbart(cb1)
#     # 添加显著性检验
#     area = np.where(s[ ::n, ::].data < 0.05)
#     plt_sig(s, ax, n, area)
#     #矢量图
#     Q = ax.quiver(
#         u.lon[::6].data, u.lat[::6].data,
#         u[ ::6, ::6].data, v[ ::6, ::6].data, transform=proj, zorder=2,
#         units='xy', angles='uv', scale=0.1, scale_units='xy',
#         width=0.8
#     )
#     #添加指示箭头
#     w, h = 0.12, 0.12
#     rect = Rectangle(
#         (1 - w, 0), w, h, transform=ax2.transAxes,
#         fc='white', ec='k', lw=0.5, zorder=1.1
#     )
#     ax2.add_patch(rect)
#     # 添加quiverkey.
#     # U指定风箭头对应的速度.
#     qk = ax.quiverkey(
#         Q, X=1-w/2, Y=0.7*h, U=1,
#         label=f'{1}', labelpos='S', labelsep=0.02,
#         fontproperties={'size': 4}
#     )
    
# %%

# %%
