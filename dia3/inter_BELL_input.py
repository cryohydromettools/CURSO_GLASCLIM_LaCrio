import sys
import numpy as np
import xarray as xr
from scipy.interpolate import griddata

sys.path.append('/home/christian/COSIPY/cosipy/')

from cosipy.modules.radCor import *


def add_variable_along_timelatlon(ds, var, name, units, long_name):
    """ This function adds missing variables to the DATA class """
    ds[name] = (('time','lat','lon'), var)
    ds[name].attrs['units'] = units
    ds[name].attrs['long_name'] = long_name
    return ds


T0 = 273.16 # K
a1 = 611.21 # Pa
a3 = 17.502 # K
a4 = 32.19  # K
R_dry = 287.0597 # Kg^-1 K^-1
R_vap = 461.5250 # Kg^-1 K^-1
g       = 9.80665

lapse_T  = (1/100)    # K
lapse_D  = (0.9/100)  # K
lapse_SF = (0.572/100) # m w.e.

## https://doi.pangaea.de/10.1594/PANGAEA.773308

# Load DEM data
file = 'BG_static_250m.nc'

# Load 10 km precipitation data in NetCDF format

#YEAR = np.arange(1969,2021,1)
YEAR = [2013]

zeni_thld = 89.0 # If you do not know the exact value for your location, set value to 89.0
timezone_lon = 90.0 # Longitude of station

## 

lat_g = -62.166
lon_g = -58.889

print(YEAR)

for n_files in range(0, len(YEAR)):
    print((YEAR[n_files]))

    file_met = '../data/SSI_ERA5/era5_'+str(YEAR[n_files])+'.nc'

    dem = xr.open_dataset(file)  # Replace with your actual DEM file
    ds = xr.open_dataset(file_met).sel(time=slice('20130101','20130831'))  # Replace with your actual NetCDF file
    ds = ds.sortby(ds.latitude)
    ds = ds.rename({'latitude':'lat','longitude':'lon'})
    ds_SWin = ds['ssrd'].sel(lat=lat_g, lon=lon_g, method = "nearest").resample(time='6H').mean('time')
    ds_LWin = ds['strd'].sel(lat=lat_g, lon=lon_g, method = "nearest").resample(time='6H').mean('time')
    ds_t2   = ds['t2m'].sel(lat=lat_g, lon=lon_g, method = "nearest").resample(time='6H').mean('time')
    ds_d2   = ds['d2m'].sel(lat=lat_g, lon=lon_g, method = "nearest").resample(time='6H').mean('time')
    ds_pres = ds['sp'].sel(lat=lat_g, lon=lon_g, method = "nearest").resample(time='6H').mean('time')
    ds_U10   = ds['u10'].sel(lat=lat_g, lon=lon_g, method = "nearest").resample(time='6H').mean('time')
    ds_V10   = ds['v10'].sel(lat=lat_g, lon=lon_g, method = "nearest").resample(time='6H').mean('time')

    ds_sf   = ds['sf'].sel(lat=lat_g, lon=lon_g, method = "nearest").resample(time='6H').sum('time')
    ds_tp   = ds['tp'].sel(lat=lat_g, lon=lon_g, method = "nearest").resample(time='6H').sum('time')

    #breakpoint()

    dso = dem   
    dso.coords['time'] = ds_t2.time.values

    dem_int = (ds['z'][0].sel(lat=lat_g, lon=lon_g, method = "nearest"))/g

    G_interp = np.full([len(dso.time), len(dso.lat), len(dso.lon)], np.nan)
    LWin_interp = np.full([len(dso.time), len(dso.lat), len(dso.lon)], np.nan)
    T_interp = np.full([len(dso.time), len(dso.lat), len(dso.lon)], np.nan)
    PRES_interp = np.full([len(dso.time), len(dso.lat), len(dso.lon)], np.nan)
    RH2_interp = np.full([len(dso.time), len(dso.lat), len(dso.lon)], np.nan)
    WS_interp = np.full([len(dso.time), len(dso.lat), len(dso.lon)], np.nan)

    SF_interp = np.full([len(dso.time), len(dso.lat), len(dso.lon)], np.nan)
    RRR_interp = np.full([len(dso.time), len(dso.lat), len(dso.lon)], np.nan)


    for t in range(len(dso.time)):

        t2_int = ds_t2[t]

        d2_int = ds_d2[t]

        pres_int = ds_pres[t]

        U10_int = ds_U10[t]

        V10_int = ds_V10[t]

        LWin_int = ds_LWin[t]

        sf_int = ds_sf[t]

        tp_int = ds_tp[t]

        t2_dow = ((t2_int.values) - (lapse_T * (dem.HGT.values-dem_int.values)))
        d2_dow = ((d2_int.values) - (lapse_D * (dem.HGT.values-dem_int.values)))

        SLP = (pres_int.values/100) / np.power((1 - (0.0065 * dem_int.values) / (288.15)), 5.255)
        pres_dow = SLP * np.power((1 - (0.0065 * dem.HGT.values)/(288.15)), 5.22) 


        T  = t2_dow
        Td = d2_dow
        P  = pres_dow
        T_e_sat = a1 * np.exp(a3* ((T - T0)/(T - a4)))
        T_q_sat = ((R_dry/R_vap)*T_e_sat)/(P - (1 - (R_dry/R_vap)) * T_e_sat)
        Td_e_sat = a1 * np.exp(a3* ((Td - T0)/(Td - a4)))
        Td_q_sat = ((R_dry/R_vap)*Td_e_sat)/(P - (1 - (R_dry/R_vap)) * Td_e_sat)
        rh2_dow    = 100 * Td_e_sat/T_e_sat

        WS10 = np.sqrt((U10_int.values)**2 + (V10_int.values)**2)
        WS2_dow = WS10 * (np.log(2/(2.12*1000))/np.log(10/(2.12*1000)))

        T_interp[t,:,:]  = t2_dow
        PRES_interp[t,:,:]  = pres_dow
        RH2_interp[t,:,:]  = rh2_dow
        RH2_interp[RH2_interp > 100]  = 100.0
        RH2_interp[RH2_interp <   0]  = 0.0

        WS_interp[t,:,:]  = WS2_dow

        SF_interp[t,:,:] = ((sf_int.values) * (1 + lapse_SF * (dem.HGT.values-dem_int.values)))
        SF_interp[SF_interp < 0] = 0

        RRR_interp[t,:,:] = ((tp_int.values) * (1 + lapse_SF * (dem.HGT.values-dem_int.values)))
        RRR_interp[RRR_interp < 0] = 0

        LWin_interp[t,:,:] = LWin_int.values/3600

    mask = dem.MASK.values
    #print(mask)
    #print(dem.MASK.values)
    hgt = dem.HGT.values
    slope = dem.SLOPE.values
    aspect = dem['ASPECT'].values - 180.0
    lats = dem.lat.values
    lons = dem.lon.values

    for t in range(len(dso.time)):
        print(t)
        doy = ds_SWin.time[t].dt.dayofyear
        hour = ds_SWin.time[t].dt.hour

        G_int  = ds_SWin[t]/3600

        for i in range(len(dem.lat)):
            for j in range(len(dem.lon)):
                if (mask[i, j] == 1):
                    G_interp[t, i, j] = np.maximum(0.0, correctRadiation(lats[i], lons[j], timezone_lon, doy, hour, slope[i, j], aspect[i, j], G_int, zeni_thld))
    print('Radiation finished')
    #breakpoint()

    add_variable_along_timelatlon(dso, T_interp, 'T2', 'K', 'Temperature at 2 m')
    add_variable_along_timelatlon(dso, RH2_interp, 'RH2', '%', 'Relative humidity at 2 m')
    add_variable_along_timelatlon(dso, WS_interp, 'U2', 'm s\u207b\xb9', 'Wind velocity at 2 m')
    add_variable_along_timelatlon(dso, G_interp, 'G', 'W m\u207b\xb2', 'Incoming shortwave radiation')
    add_variable_along_timelatlon(dso, PRES_interp, 'PRES', 'hPa', 'Atmospheric Pressure')

    add_variable_along_timelatlon(dso, SF_interp, 'SNOWFALL', 'm w.e.', 'Snowfall')
    add_variable_along_timelatlon(dso, RRR_interp * 1000, 'RRR', 'mm', 'Total precipitation (liquid+solid)')
    add_variable_along_timelatlon(dso, LWin_interp, 'LWin', 'W m\u207b\xb2', 'Incoming longwave radiation')
    
    dso.to_netcdf('data_input/ERA5_down_'+str(YEAR[n_files])+'.nc')

