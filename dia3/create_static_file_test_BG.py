"""
 This file reads the DEM of the study site and the shapefile and creates the needed static.nc
"""
import sys
import os
import xarray as xr
import numpy as np
from itertools import product
import richdem as rd

static_folder = ''

tile = False
aggregate = False

### input digital elevation model (DEM)
dem_path_tif = static_folder + '../data/DEM/Gapless_BEL_GLA_250m.tif'
### input shape of glacier or study area, e.g. from the Randolph glacier inventory
shape_path = static_folder + '../data/Shapefiles/BEL_GLA.shp'
### path were the static.nc file is saved
output_path = static_folder + 'BG_static_250m.nc'


### intermediate files, will be removed afterwards
dem_path_tif_temp = static_folder + 'DEM_temp.tif'
dem_path_tif_temp2 = static_folder + 'DEM_temp2.tif'
dem_path_tif_temp3 = static_folder + 'DEM_temp3.tif'
dem_path = static_folder + 'dem.nc'
dem_path_fill = static_folder + 'dem_fill.nc'
aspect_path = static_folder + 'aspect.nc'
mask_path = static_folder + 'mask.nc'
slope_path = static_folder + 'slope.nc'


### convert DEM from tif to NetCDF
os.system('gdal_translate -of NETCDF ' + dem_path_tif  + ' ' + dem_path)

dem = xr.open_dataset(dem_path)
array = dem.Band1.values
array[np.isnan(array)] = 0
array[array < 0] = 0
dem['Band1'][:] = array
dem.to_netcdf(dem_path_fill)

os.system('gdal_translate -of GTiff ' + dem_path_tif+' '+ dem_path_tif_temp3)

### calculate slope as NetCDF from DEM
os.system('gdaldem slope -of NETCDF ' + dem_path_fill + ' ' + slope_path + ' -s 111120')

### calculate aspect as NetCDF from DEM
os.system('gdaldem aspect -of NETCDF ' + dem_path_fill + ' ' + aspect_path)

### calculate mask as NetCDF with DEM and shapefile
os.system('gdalwarp -of NETCDF -tr 0.0026 -0.00270 -tap  --config GDALWARP_IGNORE_BAD_CUTLINE YES -cutline ' + shape_path + ' ' + dem_path_tif  + ' ' + mask_path)

### open intermediate netcdf files
dem = xr.open_dataset(dem_path_fill)
aspect = xr.open_dataset(aspect_path)
mask = xr.open_dataset(mask_path)
slope = xr.open_dataset(slope_path)

### set NaNs in mask to -9999 and elevation within the shape to 1
mask=mask.Band1.values
mask[np.isnan(mask)]=-9999
mask[mask>0]=1
mask[mask<0]=-9999
print('Mask')
print(mask.shape)
print('Dem')
print(dem)

#breakpoint()

ds = xr.Dataset()

ds.coords['lon'] = dem.lon.values
ds.lon.attrs['standard_name'] = 'lon'
ds.lon.attrs['long_name'] = 'longitude'
ds.lon.attrs['units'] = 'degrees_east'

ds.coords['lat'] = dem.lat.values
ds.lat.attrs['standard_name'] = 'lat'
ds.lat.attrs['long_name'] = 'latitude'
ds.lat.attrs['units'] = 'degrees_north'

### function to insert variables to dataset
def insert_var(ds, var, name, units, long_name):
    ds[name] = (('lat','lon'), var)
    ds[name].attrs['units'] = units
    ds[name].attrs['long_name'] = long_name
    ds[name].attrs['_FillValue'] = -9999

### insert needed static variables
insert_var(ds, dem.Band1.values,'HGT','meters','meter above sea level')
insert_var(ds, aspect.Band1.values,'ASPECT','degrees','Aspect of slope')
insert_var(ds, slope.Band1.values,'SLOPE','degrees','Terrain slope')
insert_var(ds, mask,'MASK','boolean','Glacier mask')

### save combined static file, delete intermediate files and print number of glacier grid points
ds.to_netcdf(output_path)
os.system('rm '+ dem_path + ' ' + aspect_path + ' ' + mask_path + ' ' + slope_path + ' ' + dem_path_tif_temp + ' '+ dem_path_tif_temp2+ ' '+ dem_path_tif_temp3+ ' '+ dem_path_fill)
print("Study area consists of ", np.nansum(mask[mask==1]), " glacier points")


