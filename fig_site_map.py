from pc_data_tools import load_pc_dataset, get_avi_pcs_no_water_sites
import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import cartopy.feature as cfeature
import cartopy.crs as ccrs
from cartopy.io import shapereader
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.io.img_tiles as cimgt
from scalebar import scale_bar
import cartopy.mpl.geoaxes
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from cartopy.feature import ShapelyFeature
import os
import matplotlib.patches as mpatches

_, all_sites, _, all_pcs = load_pc_dataset('pc_data_parsed_raw_audioset_feats_300s')

chosen_pcs = get_avi_pcs_no_water_sites(all_pcs, all_sites)

chosen_sites = np.asarray([pc.site for pc in chosen_pcs])
_, unq_idx = np.unique([s.name for s in chosen_sites], return_index=True)
unq_sites = chosen_sites[unq_idx]


request_satellite = cimgt.GoogleTiles(style='satellite')


fig = plt.figure(figsize=(9, 9))
ax = plt.axes(projection=ccrs.PlateCarree())
gl = ax.gridlines(draw_labels=True, alpha=0.2)
gl.xlabels_top = gl.ylabels_right = False
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER

plt.title('Sampling sites')

min_lat = min([s.lat for s in unq_sites])
max_lat = max([s.lat for s in unq_sites])
min_long = min([s.long for s in unq_sites])
max_long = max([s.long for s in unq_sites])
lat_offs = 0.02
long_offs = 0.04
ax.set_extent([min_long-long_offs, max_long+long_offs, min_lat-lat_offs, max_lat+lat_offs], crs=ccrs.PlateCarree())

ax.add_image(request_satellite, 13)

for s in unq_sites:
    sz = 5 + (0.05 * s.get_agb())

    abbrv = s.get_abbrv_name()
    if abbrv == 'OG': c = '#00FF00'
    elif abbrv == 'LF': c = '#00FFFF'
    elif abbrv == 'OP': c = '#FFFF00'
    elif abbrv == 'SL': c = '#FF7F2A'

    plt.plot(s.long, s.lat, transform=ccrs.PlateCarree(), markersize=sz, marker='o', color=c)
    plt.text(s.long, s.lat, '{} ({})'.format(s.get_abbrv_name(), s.name), transform=ccrs.PlateCarree(), color=c, ha='center', va='top', fontsize=13)
    print('{}: {} {} (AGB = {})'.format(s.name, s.long, s.lat, s.get_agb()))

# need a font that support enough Unicode to draw up arrow. need space after Unicode to allow wide char to be drawm?
# plt.text(max_long+0.01, min_lat-0.01, u'\u25B2 \nN ', transform=ccrs.PlateCarree(), va='center', color='yellow', fontsize=15, family='Arial')

scale_bar(ax, (0.9, 0.05), 1, color='white')

axins = inset_axes(ax, width="50%", height="50%", loc="upper left",
                   axes_class=cartopy.mpl.geoaxes.GeoAxes, borderpad=0,
                   axes_kwargs=dict(map_projection=cartopy.crs.PlateCarree()))

axins.add_feature(cfeature.LAND)
axins.add_feature(cfeature.COASTLINE)


'''
lat_corners = np.array([min_lat, min_lat, max_lat, max_lat])
lon_corners = np.array([min_long, max_long, max_long, min_long])
poly_corners = np.zeros((len(lat_corners), 2), np.float64)
poly_corners[:,0] = lon_corners
poly_corners[:,1] = lat_corners
poly = mpatches.Polygon(poly_corners, closed=True, ec='r', fill=False, lw=1, fc=None, transform=ccrs.PlateCarree())
axins.add_patch(poly)
'''

axins.set_extent([94, 127.8, -7.2, 10], crs=ccrs.PlateCarree())

scale_bar(axins, (0.5, 0.87), 100, color='black')
axins.plot(min_long, min_lat, transform=ccrs.PlateCarree(), markersize=10, marker='*', color='red')


plt.savefig(os.path.join('figs','site_map.svg'))
plt.show()
