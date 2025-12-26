# -*- coding: utf-8 -*-
"""
Analysis of Haruki Climate model data
"""


from os import chdir, listdir, path, environ
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
import pandas as pd


regions = {'Arctic': ((60, 80), (0, 360)),
           'NA': ((30, 50), (290, 360)),
           'NP': ((30, 50), (170, 240)),
           'NWA': ((0, 30), (290, 335)),
           'NI': ((0, 30), (45, 100)),
           'NWP': ((0, 30), (120, 170)),
           'NEP': ((0, 30), (210, 250)),
           'SEP': ((-30, 0), (110, 290)),
           'SEA': ((-30, 0), (335, 375)),
           'SWP': ((-30, 0), (150, 190)),
           'IndOcn': ((-40, -10), (75, 120)),
           'SA': ((-50, -30), (305, 375)),
           'SP': ((-50, -30), (190, 270)),
           'SI': ((-50, -30), (30, 100))}


chdir('C:/Users/aakas/Documents/MCB-Zonal/')


def load_atmos_data(fpath):
    """
    loads data from atmospheric component of models, all stored in a single
    filepath. Returns dictionary of xarray datasets with each dict entry
    being a different MCB experiment
    """
    brightened = dict()
    regions = listdir(fpath)
    for loc in regions:
        data_files = listdir(path.join(fpath, loc))
        data = []
        for file in data_files:
            data.append(xr.open_dataset(path.join(fpath, loc, file),
                                        engine='zarr'))
        # merge all data vars into a single dataarray
        # skip U/V for now
        # main = data.pop()
        main = data[0]
        data = data[1:]
        #global var_data
        for var_data in data:
            # hack way of getting the only variable with shape
            # time, lat, lon
            name = [x for x in var_data.data_vars if
                    len(var_data[x].shape) == 3]
            if name:
                name = name[0]
                # skipping the 3D wind fields for now
                main[name] = var_data[name]
        # add to dict
        brightened[loc] = main
    # add albedo
    for key, value in brightened.items():
        brightened[key]['albedo'] = value['FSUTOA'] / value['SOLIN']
    # redo keys for R1-R5
    rename = {'R1': 'NEP', 'R2': 'SEP', 'R3': 'SEA', 'R4': 'NP', 'R5': 'SP'}
    for key, value in rename.items():
        brightened[value] = brightened.pop(key)
    return brightened


def weighted_mean(data, var, weights, min_lat, max_lat, 
                  min_lon, max_lon, topo, ocean=True):
    """
    Returns weighted mean using xarray interface
    """
    # replace all land grid cells with NaN
    if ocean:
        data = data.where(topo.LANDFRAC <= 0.5)
    
    if max_lon <= 360:
        # normal process
        weights = weights.sel(lat=slice(min_lat, max_lat),
                              lon=slice(min_lon, max_lon))
        data = data.sel(lat=slice(min_lat, max_lat),
                        lon=slice(min_lon, max_lon))    
        data_w = data[var].weighted(weights)
        weight_mean = data_w.mean()
    else:
        # if the box crosses the prime meridian
        weights_west = weights.sel(lat=slice(min_lat, max_lat),
                                   lon=slice(min_lon, 360))
        weights_east = weights.sel(lat=slice(min_lat, max_lat),
                                   lon=slice(0, max_lon - 360))
        data_west = data.sel(lat=slice(min_lat, max_lat),
                             lon=slice(min_lon, 360))
        data_east = data.sel(lat=slice(min_lat, max_lat),
                             lon=slice(0, max_lon - 360))
        data_west_w = data_west[var].weighted(weights_west).mean()
        data_east_w = data_east[var].weighted(weights_east).mean()
        # finally weight by areas across the meridian
        weights_west = weights_west.sum()
        weights_east = weights_east.sum()
        weight_mean = (weights_west * data_west_w + weights_east * data_east_w) /\
            (weights_east + weights_west)
    
    return weight_mean


def calc_vars(data_dict, topo_data):
    """
    Calculates meriodional difference in albedo, zonal difference
    in surface temperature, central pacific wind speeds
    averaged over last five years of simulation
    """
    final_df = dict()
    for key, value in data_dict.items():
        # remove weird duplicates
        value = value.drop_duplicates(dim='time')
        # weight based on weights in model
        if 'time' in list(topo_data.AREA.coords):
            weights = topo_data.AREA.mean(dim='time')
        else:
            weights = topo_data.AREA
        # retain last five years of simulation
        last_five = value.time[-240:]
        last_five = value.isel(time=slice(-240, None)).mean(dim='time')
        # pacific albedo gradient
        np_albedo = weighted_mean(last_five, 'albedo', weights,
                                  8, 65, 130, 280, topo=topo_data)
        sp_albedo = weighted_mean(last_five, 'albedo', weights,
                                  -65, -8, 130, 280, topo=topo_data)
        # weight average albedo by area of ocean pixels
        npac_area = topo_data['AREA'].where(topo_data.LANDFRAC <= 0.5).\
            sel(lat=slice(8, 65), lon=slice(130, 280)).sum()
        spac_area = topo_data['AREA'].where(topo_data.LANDFRAC <= 0.5).\
            sel(lat=slice(-65, -8), lon=slice(130, 280)).sum()
        pac_ex_albedo = (npac_area * np_albedo + spac_area * sp_albedo) /\
            (npac_area + spac_area)
        pac_eq_albedo = weighted_mean(last_five, 'albedo', weights,
                                  -8, 8, 130, 280, topo=topo_data)
        pac_albedo_grad = pac_ex_albedo - pac_eq_albedo
        # east-west pacific SST gradient
        west_pac = weighted_mean(last_five, 'TS', weights,
                                  -8, 8, 130, 205, topo=topo_data)
        east_pac = weighted_mean(last_five, 'TS', weights,
                                  -8, 8, 205, 280, topo=topo_data)
        sst_grad = west_pac - east_pac
        # central pacific low level winds
        winds = weighted_mean(last_five, 'TS', weights,
                                  -8, 8, 180, 240, topo=topo_data)
        # create dict
        summary = {'pac_albedo_grad': float(pac_albedo_grad),
                   'sst_grad': float(sst_grad),
                   'cp_winds': float(winds)}
        final_df[key] = summary
    return final_df        


def scatter_plot(grad_info, var1, var2, name1, name2, burls=False):
    """
    plots the scatter of var1 vs. var2 from grad_info, potentially including
    the burls+federov trendline
    """
    icons = ['o', '.', ',', 'x', '+', 'v', '>', '<', 's', '^', 'd']
    icons = cycle(icons)
    min_alb = np.inf
    max_alb = -np.inf
    plt.figure()
    for key, value in grad_info.items():
        plt.scatter(value[var1], value[var2], label=key,
                    marker=next(icons))
        if burls:
            if value[var1] < min_alb:
                min_alb = value[var1]
            elif value[var1] > max_alb:
                max_alb = value[var1]
    plt.legend(bbox_to_anchor=(1.05, 1))
    plt.xlabel(name1)
    plt.ylabel(name2)
    if burls:
        line = np.linspace(min_alb, max_alb, 100)
        plt.plot(line, 16.2 * line + 1.7)
        # plt.plot(line, 19 * line, color='red')
    plt.grid()
    plt.show()    
    

def load_reference(fpath, name='cesm'):
    """
    Loads the reference runs, returning each ensemble
    """
    files = listdir(fpath)
    data_list = []
    cases = []
    curr_case = None
    for file in files:
        data = xr.load_dataset(path.join(fpath, file), engine='netcdf4')
        if data.time.min().dt.year >= 2065:
            # we only want data until 2065
            continue
        if curr_case == data.case:
            cases.append(data)
        else:
            # group ensemble members together
            curr_case = data.case
            if cases:
                # only make list of lists if cases is populated
                data_list.append(cases)
            # create new list and add new mismatched data element
            cases = []
            cases.append(data)
    # append last case        
    data_list.append(cases)
    # merge files
    data_merge = {f'{name}_{n+1}': xr.merge(x, compat='override') for n, x in 
                  enumerate(data_list)}
    # add albedo
    for key, value in data_merge.items():
        data_merge[key]['albedo'] = value['FSUTOA'] / value['SOLIN']
    return data_merge


def gradient_obs(ceres_data, ersst_data, ceres_grid):
    """
    Calculates sst and albedo gradient for observational data
    """
    last_ten = ceres_data.time[-240:]
    ceres_mean = ceres_data.sel(time=last_ten).mean(dim='time', skipna=True)
    ersst_mean = ersst_data.sel(time=last_ten).mean(dim='time')
    # filter out land for ceres albedo
    ceres_mean = ceres_mean.where(ceres_grid.aux_ocean_mon >= 50)
    # get meriodional albedo differences
    np_albedo = ceres_mean['toa_alb_all_mon'].sel(lat=slice(8, 65),
                                         lon=slice(130, 280)).\
        mean(dim=['lat', 'lon'])
    sp_albedo = ceres_mean['toa_alb_all_mon'].sel(lat=slice(-65, -8),
                                         lon=slice(130, 280)).\
        mean(dim=['lat', 'lon'], skipna=True)
    exp_albedo = (np_albedo + sp_albedo) / 2
    eq_albedo = ceres_mean['toa_alb_all_mon'].sel(lat=slice(-8, 8),
                                         lon=slice(130, 280)).\
        mean(dim=['lat', 'lon'], skipna=True)
    albedo_grad = exp_albedo - eq_albedo
    # get zonal sst differences
    # get land mask from where SST is zero
    wp_sst = ersst_mean['sst'].where(ersst_mean['sst'] != 0 ).\
        sel(lat=slice(-8, 8), lon=slice(130, 205)).\
        mean(dim=['lat', 'lon'])
    ep_sst = ersst_mean['sst'].where(ersst_mean['sst'] != 0 ).\
        sel(lat=slice(-8, 8), lon=slice(205, 280)).\
        mean(dim=['lat', 'lon'])
    sst_grad = wp_sst - ep_sst
    return {'pac_albedo_grad': float(albedo_grad), 'sst_grad': float(sst_grad)}


def calc_heat_trans(cesm_dict, topo_map):
    """
    Calculates total heat transport using the TOA integration approach of 
    Singh et. al. 2022
    
    mean over last 20 years of simulation
    """
    # this will hold the final heat transport
    poleward_data = {}
    for key, value in cesm_dict.items():
        # remove weird duplicates
        value = value.drop_duplicates(dim='time')
        last_twenty = value.time[-240:]
        rel_data = value.sel(time=last_twenty).mean(dim='time')
        rel_data['toa_energy_bal'] = rel_data['FSNTOA'] - rel_data['FLNT']
        # mean heat uptake
        energy_imbal = rel_data['toa_energy_bal'].\
            weighted(topo_map.AREA).mean()
        # subtract mean heat uptake from field
        rel_data['toa_energy_bal'] -= float(energy_imbal)
        # start integration- thanks chatGPT for code corrections
        # --- 1. Zonal integration (gives W per latitude band)
        band_flux = rel_data['toa_energy_bal'].\
            mean(dim='lon')
        # --- 2. Latitude spacing (in radians)
        dphi = float(np.radians(band_flux.lat[1] - band_flux.lat[0]))
        coslat = np.cos(np.radians(band_flux.lat))
        # --- 3. Cumulative φ-integration
        Hphi = (coslat.data * band_flux * dphi).cumsum(dim='lat')
        # --- 4. Multiply by spherical geometry factor and convert to PW
        a = 6371000
        Hphi = Hphi * (2 * np.pi * a**2) / 1e15
        # save info for a given run
        poleward_data[key] = Hphi.data
    poleward_data['lat'] = topo_map.lat
    return pd.DataFrame(poleward_data)


def calc_ocean_heat_trans(cesm_dict, topo_map):
    """
    Calculates oceanic heat transport using the surface enrgy budget
    integration approach of Singh et. al. 2022
    """
    poleward_data = {}
    for key, value in cesm_dict.items():
        # remove weird duplicates
        value = value.drop_duplicates(dim='time')
        last_twenty = value.time[-240:]
        rel_data = value.sel(time=last_twenty).mean(dim='time')
        # surface energy budget- I think the sign convention is correct
        rel_data['sfc_energy_bal'] = rel_data['FSNS'] - rel_data['LHFLX'] -\
            rel_data['SHFLX'] - rel_data['FLNS']
        # mask land area now
        # rel_data = rel_data.where(topo_map.LANDFRAC < 0.5)
        # Subtract energy imbalance
        energy_imbal = rel_data['sfc_energy_bal'].\
            weighted(topo_map.AREA).mean()
        # subtract mean heat uptake from field
        rel_data['sfc_energy_bal'] -= float(energy_imbal)
        # start integration- thanks chatGPT for code corrections
        # --- 1. Zonal integration (gives W per latitude band)
        band_flux = rel_data['sfc_energy_bal'].\
            mean(dim='lon')
        # --- 2. Latitude spacing (in radians)
        dphi = float(np.radians(band_flux.lat[1] - band_flux.lat[0]))
        coslat = np.cos(np.radians(band_flux.lat))
        # --- 3. Cumulative φ-integration
        Hphi = (coslat.data * band_flux * dphi).cumsum(dim='lat')
        # --- 4. Multiply by spherical geometry factor and convert to PW
        a = 6371000
        Hphi = Hphi * (2 * np.pi * a**2) / 1e15
        # save info for a given run
        poleward_data[key] = Hphi.data
    poleward_data['lat'] = topo_map.lat
    return pd.DataFrame(poleward_data)
 

def toa_rad_anoms(cesm_dict, cesm_topo, regions):
    """
    Returns radiative anomalies across different simulations
    calculating the average over the last twenty years of simulation
    and subtracting the reference simulations
    
    dict format: global rad anomaly, global LW anomaly,
    global SW anomaly, and the same for within the patch regions
    """
    anomalies = pd.DataFrame(columns=['RadNet_global', 'RadNet_box',
                                      'T_global', 'T_box', 'Area'])
    # calculate means and add radiation field
    for key, value in cesm_dict.items():
        value = value.drop_duplicates(dim='time')
        last_twenty = value.time[-240:]
        rel_data = value.sel(time=last_twenty).mean(dim='time')
        rel_data['RadNet'] = rel_data['FSNTOA'] - rel_data['FLNT']
        # isolate regional averages
        coords = regions[key]
        global_tanom = weighted_mean(rel_data.copy(), 'TS', cesm_topo.AREA,
                                     min_lat=-90, max_lat=90,
                                     min_lon=0, max_lon=360,
                                     topo=cesm_topo, ocean=False)
        global_radanom = weighted_mean(rel_data.copy(), 'RadNet', cesm_topo.AREA,
                                     min_lat=-90, max_lat=90,
                                     min_lon=0, max_lon=360,
                                     topo=cesm_topo, ocean=False)
        region_tanom = weighted_mean(rel_data.copy(), 'TS', cesm_topo.AREA,
                                     min_lat=coords[0][0], max_lat=coords[0][1],
                                     min_lon=coords[1][0], max_lon=coords[1][1],
                                     topo=cesm_topo, ocean=False)
        region_radanom = weighted_mean(rel_data.copy(), 'RadNet', cesm_topo.AREA,
                                     min_lat=coords[0][0], max_lat=coords[0][1],
                                     min_lon=coords[1][0], max_lon=coords[1][1],
                                     topo=cesm_topo, ocean=False)
        area = weighted_mean(cesm_topo.copy(), 'AREA', cesm_topo.AREA,
                            min_lat=coords[0][0], max_lat=coords[0][1],
                            min_lon=coords[1][0], max_lon=coords[1][1],
                            topo=cesm_topo, ocean=False)
        data = {'RadNet_global': float(global_radanom),
                'RadNet_box': float(region_radanom),
                'T_global': float(global_tanom),
                'T_box': float(region_tanom),
                'Area': float(area)}
        anomalies.loc[key] = data
    
    return anomalies


def ref_rad_anoms(cesm_ref, cesm_topo, regions):
    """
    Calculates the radiative anomalies and temperature anomalies within
    each box similar to the toa_rad_anoms() function
    
    Difference- averaging across all three ensemble members
    """
    anomalies = pd.DataFrame(columns=['RadNet_global', 'RadNet_box',
                                      'T_global', 'T_box', 'Area'])
    # calculate means of radiation field
    for region, coords in regions.items():
        global_tanom = []
        global_radanom = []
        region_tanom = []
        region_radanom = []
        area = []
        for key, value in cesm_ref.items():
            value = value.drop_duplicates(dim='time')
            last_twenty = value.time[-240:]
            rel_data = value.sel(time=last_twenty).mean(dim='time')
            rel_data['RadNet'] = rel_data['FSNTOA'] - rel_data['FLNT']
            # isolate regional averages
            global_tanom.append(weighted_mean(rel_data.copy(), 'TS', cesm_topo.AREA,
                                min_lat=-90, max_lat=90,
                                min_lon=0, max_lon=360,
                                topo=cesm_topo, ocean=False))
            global_radanom.append(weighted_mean(rel_data.copy(), 'RadNet', cesm_topo.AREA,
                                min_lat=-90, max_lat=90,
                                min_lon=0, max_lon=360,
                                topo=cesm_topo, ocean=False))
            region_tanom.append(weighted_mean(rel_data.copy(), 'TS', cesm_topo.AREA,
                                min_lat=coords[0][0], max_lat=coords[0][1],
                                min_lon=coords[1][0], max_lon=coords[1][1],
                                topo=cesm_topo, ocean=False))
            region_radanom.append(weighted_mean(rel_data.copy(), 'RadNet', cesm_topo.AREA,
                                min_lat=coords[0][0], max_lat=coords[0][1],
                                min_lon=coords[1][0], max_lon=coords[1][1],
                                topo=cesm_topo, ocean=False))
            area.append(weighted_mean(cesm_topo.copy(), 'AREA', cesm_topo.AREA,
                        min_lat=coords[0][0], max_lat=coords[0][1],
                        min_lon=coords[1][0], max_lon=coords[1][1],
                        topo=cesm_topo, ocean=False))
        global_tanom = np.mean(global_tanom)
        global_radanom = np.mean(global_radanom)
        region_tanom = np.mean(region_tanom)
        region_radanom = np.mean(region_radanom)
        area = np.mean(area)
        data = {'RadNet_global': float(global_radanom),
                'RadNet_box': float(region_radanom),
                'T_global': float(global_tanom),
                'T_box': float(region_tanom),
                'Area': float(area)}
        anomalies.loc[region] = data
        
    return anomalies        
   

def main():
    global pole_anoms
    global anomalies, reference
    data_dict = load_atmos_data('mcb_runs/CESM')
    # load simulation data
    cesm_dict = load_reference('reference_runs/CESM', 'cesm')
    cesm_grid = xr.load_dataset('gcm_grid_info/cam_gridinfo.nc').\
        mean(dim='time')
    grad_info = calc_vars(data_dict, topo_data=cesm_grid)
    cesm_info = calc_vars(cesm_dict, topo_data=cesm_grid)
    
    # load observational data
    ceres_syn = xr.concat([xr.load_dataset(path.join('obs_data/CERES-SYN', x))
                          for x in 
                          listdir('obs_data/CERES-SYN')], dim='time')
    ceres_grid = xr.load_dataset('gcm_grid_info/ceres_ocean_area.nc').\
        mean(dim='time')
    # replace NaNs/??
    # ceres_syn = ceres_syn.where(ceres_syn == -999)
    # first of month indexing
    ceres_syn['time'] = ceres_syn['time'] - pd.Timedelta(14, 'days')
    ersst_data = xr.load_dataset('obs_data/ERSST/ERSST_merged.nc')
    obs = gradient_obs(ceres_syn, ersst_data, ceres_grid)
    grad_info['obs'] = obs
    # plot results
    merged = grad_info | cesm_info
    scatter_plot(merged, 'pac_albedo_grad', 'sst_grad', 
                 'Pacific Albedo Gradient', 'Zonal Pacific SST Gradient', True)
    
    # poleward heat transport
    cesm_pole = calc_heat_trans(data_dict, cesm_grid)
    cesm_pole_ref = calc_heat_trans(cesm_dict, cesm_grid)
    cesm_pole_ref['mean'] = cesm_pole_ref[['cesm_1', 'cesm_2', 'cesm_3']].\
        mean(axis=1)
        
    cesm_ocean = calc_ocean_heat_trans(data_dict, cesm_grid)     
    cesm_ocean_ref = calc_ocean_heat_trans(cesm_dict, cesm_grid)                          
    cesm_ocean_ref['mean'] = cesm_ocean_ref[['cesm_1', 'cesm_2', 'cesm_3']].\
        mean(axis=1)
    # atmosphere now
    cesm_atms = cesm_pole - cesm_ocean 
    #cesm_atms['lat'] = cesm_pole['lat']
    cesm_atms_ref = cesm_pole_ref - cesm_ocean_ref
    #cesm_atms_ref['lat'] = cesm_pole_ref['lat']
    
    anomalies = toa_rad_anoms(data_dict, cesm_grid, regions)
    reference = ref_rad_anoms(cesm_dict, cesm_grid, regions)
           
    true_anoms = anomalies - reference
    
if __name__ == '__main__':
    main()
