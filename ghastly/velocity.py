import numpy as np
import numpy.polynomial.polynomial as nppoly
import math
import openmc
from ghastly import read_input
from ghastly import pebble
from ghastly import data
from jinja2 import Environment, PackageLoader
from os import path
import glob
import h5py
import sys
import matplotlib.pyplot as plt
from matplotlib import colormaps



def velocity_profiler(d_h5, r_h5, zone_bounds = [0, 10, 15, 20],
                      n_layers = 15, cycle_days = 183, name_key='',
                      verbose=True):
    '''
    given input info, return indices (aka UIDs) of pebbles that create the
    initial 'wave front', used to determine flow regions and select
    depleting pebbles
    '''  

    with h5py.File(d_h5, mode='r') as d5f, h5py.File(r_h5, mode='r') as r5f:
        N_r =  r5f.attrs['Nsteps']
        r_steps = np.arange(N_r)
        n_r = list(r5f['n_recirc_step'])
        n_r_calc = np.array(list(map(int, n_r)))
        n_r_cumul = list(r5f['n_recirc_cumul'])
        n_r_avg = sum(n_r_calc)/len(n_r_calc) 
        N_d = d5f.attrs['Nsteps']
        d_steps = np.arange(N_d)
        dt_r = d5f.attrs['n_recirc']
        dt_d = d5f.attrs['n_dump']
        i_settled = [d_steps[np.argmax(dt_d*d_steps == (i+1)*dt_r - dt_d)] 
                   for i in r_steps[:-1]]
        if verbose:
            n_settled = len(i_settled)
            i_yap = int((n_settled+1)/10)
        peb_R = d5f.attrs['peb_r']
        peb_D = 2*peb_R
        init_xyz = d5f['XYZ'][0]
        init_z = [xyz[2] for xyz in init_xyz]
        peb_inv = len(init_z)
        p_per_d = peb_inv/cycle_days
        init_layer = d5f['layer'][0]
        bed_top = max(init_z)
        safe_zmax = bed_top - 3.5*peb_D
        safe_zmin = 0
        layers = np.flip(np.linspace(safe_zmin, safe_zmax, n_layers+1)) 
        zones = peb_D*np.array(zone_bounds)
        n_zones = int(len(zones)) - 1
        
        if verbose:
            prnt_tot = int(n_r_cumul[-1])
            prnt_avg = int(100*n_r_avg)/100
            prnt_pd = int(100*p_per_d)/100
            prnt_cycd = int(100*(n_r_avg/p_per_d))/100
            print(f'{N_r} recirculation cycles in database.')
            print(f'Total pebbles cycled: {prnt_tot}.')
            print(f'{prnt_avg} pebbles are recirculated each cycle on average.')
            print(f'There are {prnt_pd} pebbles cycled per day.')
            print(f'Therefore, one cycle is approx. {prnt_cycd} days.')
            print()
        
        old_z = init_z
        old_recirc = np.zeros(peb_inv)
        v_d = [[np.zeros(n_zones) for _ in range(n_layers)]
               for _ in range(peb_inv)]
        n_v = [[np.zeros(n_zones) for _ in range(n_layers)]
               for _ in range(peb_inv)]
        for i_cyc, i_set in enumerate(i_settled):
            if verbose and i_cyc%i_yap == 0:
                progress = 100*(((i_cyc//i_yap)*i_yap)/n_settled)
                print(f'{int(progress)}% complete')
            
            set_xyz = d5f['XYZ'][i_set]
            set_z = [xyz[2] for xyz in set_xyz]
            set_recirc = d5f['recirc_n'][i_set]
            for i, z in enumerate(set_z):
                if z > safe_zmin and z <= safe_zmax: 
                    if set_recirc[i] == old_recirc[i]:
                        r = sum(set_xyz[i][0:2]**2)**0.5
                        for il, lb in list(enumerate(layers))[:-1]:
                            if z > layers[il+1] and z <= lb:
                                for ir, rb in list(enumerate(zones))[1:]:
                                    if r > zones[ir-1] and r <= rb:
                                        n_v[i][il][ir-1] += 1
                                        disp = set_z[i] - old_z[i]
                                        days = n_r[i_cyc]/p_per_d
                                        v1 = (disp/days)
                                        n1 = n_v[i][il][ir-1]
                                        v0 = v_d[i][il][ir-1]
                                        n0 = n1-1
                                        v_d[i][il][ir-1] = ((v0*n0 + v1)/n1)
                                        break
                                    else:
                                        pass
                                break
                            else:
                                pass
                    else:
                        old_recirc[i] = set_recirc[i]
                else:
                    pass
            old_z= set_z
        vzr = [np.zeros(n_zones) for _ in range(n_layers)]
        nvzr = [np.zeros(n_zones) for _ in range(n_layers)]
        for ip, vpeb in enumerate(v_d):
            for il, vl in enumerate(vpeb):
                for ir, vr in enumerate(vl):
                    if n_v[ip][il][ir] != 0:
                        nvzr[il][ir] += 1
                        vzr[il][ir] = ((vzr[il][ir]*(nvzr[il][ir]-1) + vr)
                                        /nvzr[il][ir])
        vr_profile = np.array(sum(vzr)/len(vzr))
        if verbose:
            print('Radial vz profile for each axial layer in cm/day,')
            print('Layers from top to bottom, zones from center to outside')
            for vlayer in vzr:
                print(vlayer)
            print()
            print('Axial-layer averaged  radial vz profile in cm/day.')
            print(vr_profile)
            print()
        velcsv = f'{n_layers}axl_{n_zones}radz_velprof{name_key}.csv'
        vr_csv = np.concatenate((vzr, [np.empty(n_zones)], [vr_profile]), 
                                axis=0)
        with open(velcsv, mode='w') as vcsv:
            np.savetxt(velcsv, vr_csv, delimiter = ',')
 

def velocity_plotter(d_csv, peb_D, zone_bounds = [0, 10, 15, 20],
                      n_layers = 15, cycle_days = 183, name_key='',
                      verbose=True, n_rgb=100, c0 = 0.1, c1 = 0.9, 
                     cmap='magma_r'):
    '''
    plot velocity profiles using the given csv with post-processed data from
    velocity_profiler()
    '''
    plt_c = np.linspace(c0, c1, n_rgb)
    plt_rgb = colormaps[cmap](plt_c)
    latt_zmax = 9.2035e+02
    zones = peb_D*np.array([zone_bounds])
    zone_rgb = int(n_rgb/(n_zones+2))*np.arange(n_zones+2)[1:-1]
    zoneticks = [int(zone) for zone in zones]
    midzone = np.zeros(n_zones)
    for ir, rb in list(enumerate(zones))[1:]:
        midzone[ir-1] = (rb - zones[ir-1])/2 + zones[ir-1]


    n_prog = 13
    prog_rgb = int(n_rgb/(n_prog+2))*np.arange(n_prog+2)[1:-1] 
    peb_prog = [latt_zmax + 15*i*vr_profile for i in np.arange(n_prog)] 

    for i in range(n_prog):
        plt.plot(midzone, peb_prog[i], 
                     linestyle='-', marker='o',label = f'Day {i*15}',
                     color = plt_rgb[prog_rgb[i]])
        
    for zone in zones:
        plt.vlines(zone, -100, latt_zmax+100, 
                       label = '_zonemarker', color = plt_rgb[-1])
    plt.hlines(latt_zmax, 0, 120, label= '_coremarker',
                   linestyle = '--', color=plt_rgb[-1])
    plt.hlines(safe_zmin, 0, 120, label= '_coremarker',
                   linestyle = '--', color=plt_rgb[-1])
    plt.legend(loc = 'center left')
    plt.title('Axial progression of pebbles over 180 days')
    plt.ylim(-100, latt_zmax+100)
    plt.xlim(0, 120)
    plt.ylabel('Z [cm]')
    plt.xlabel('Middle of Radial Zone [cm]') 
    plt.xticks(zoneticks)
    progpng = f'{n_layers}axl_{n_zones}radz_pebprog{name_key}.png'
    plt.savefig(progpng)
    plt.close()

    for i in range(n_zones):
        r_line = np.linspace(zones[i], zones[i+1], 10)
        vz_line = vr_profile[i]*np.ones(10)
        plt.plot(r_line, vz_line, lw=3.0,
                     color = plt_rgb[zone_rgb[i]], label = f'Zone {i+1}')
            
        plt.vlines(zones[i+1], 1.5*min(vr_profile), 0.5*max(vr_profile),
                       label = '_zonemarker', color = plt_rgb[-1])
        
    plt.title('Radial vz profile in main core region')
    plt.ylim(1.5*min(vr_profile), 0.5*max(vr_profile))
    plt.xlim(0, 120)
    plt.xlabel('r [cm]')
    plt.ylabel('vz [cm/day]')
    plt.xticks(zoneticks)
    velprofpng = f'{n_layers}axl_{n_zones}radz_velprof{name_key}.png'
    plt.savefig(velprofpng)
    plt.close()



def transit_profiler(d_h5, r_h5, cycle_days = 183, name_key='',
                     verbose = True, n_rgb=100, 
                     c0 = 0.1, c1 = 0.9, cmap='magma_r'):
    '''
    given data h5 and recirc h5, select a sample of pebbles at the top of the
    core and calculate their tracklength, residence time, and transit number
    '''
    plt_c = np.linspace(c0, c1, n_rgb)
    plt_rgb = colormaps[cmap](plt_c)

    with h5py.File(d_h5, mode='r') as d5f, h5py.File(r_h5, mode='r') as r5f:
        N_r =  r5f.attrs['Nsteps']
        r_steps = np.arange(N_r)
        n_r = list(r5f['n_recirc_step'])
        n_r_cumul = list(r5f['n_recirc_cumul']) 
        N_d = d5f.attrs['Nsteps']
        d_steps = np.arange(N_d)
        dt_r = d5f.attrs['n_recirc']
        dt_d = d5f.attrs['n_dump']
        i_settled = [d_steps[np.argmax(dt_d*d_steps == (i+1)*dt_r - dt_d)] 
                   for i in r_steps[:-1]]
        if verbose:
            n_settled = len(i_settled)
            i_yap = int((n_settled+1)/10)

        peb_R = d5f.attrs['peb_r']
        peb_D = 2*peb_R
        init_xyz = d5f['XYZ'][0]
        init_r = [sum(xyz[0:2]**2)**0.5 for xyz in init_xyz]
        peb_inv = len(init_r)
        p_per_d = peb_inv/cycle_days
        bed_zmax = max([xyz[2] for xyz in init_xyz])
        safe_zmax = bed_zmax - 4.0*peb_D
        margin = 0.5*peb_R
        enough = False
        while not enough:
            sample = [i for i, xyz in enumerate(init_xyz) 
                       if xyz[2] <= (safe_zmax + margin) 
                       and xyz[2] >= (safe_zmax - margin)]
            if len(sample) >= 1000:
                enough = True
            else:
                margin += 0.05*peb_R
        N_s = len(sample)
        sample_r = [init_r[i] for i in sample]
        old_xyz = [init_xyz[i] for i in sample]
        tracklength = np.zeros(len(sample))
        pebcycled = np.zeros(len(sample))
        for i_cyc, i_set in enumerate(i_settled):
            set_xyz = d5f['XYZ'][i_set]
            sample_xyz = [set_xyz[i] for i in sample]
            set_recircn = d5f['recirc_n'][i_set]
            sample_recircn = [set_recircn[i] for i in sample]
            if sum(sample_recircn) == len(sample_recircn):
                print('All pebbles have completed their transit')
                break
            for i in range(N_s):
                if sample_recircn[i] != 0:
                    pass
                else:
                    pebcycled[i] = n_r_cumul[i_cyc+1]
                    distance = sum((old_xyz[i] - sample_xyz[i])**2)**0.5
                    tracklength[i] += distance
            old_xyz=sample_xyz

        transit_num = pebcycled/peb_inv
        res_time = pebcycled/p_per_d
        
        n_plots = 3
        transit_rgb = int(n_rgb/(n_plots+2))*np.arange(n_plots+2)[1:-1]
        plt.plot(sample_r, tracklength, color = plt_rgb[transit_rgb[0]],
                 linestyle = '', marker = '.')
        plt.xlim(0, 120)
        plt.xlabel('Initial relative radial position [r/R]')
        plt.ylabel('Total tracklength [cm]')
        plt.title('Tracklength vs relative radius; sample of {n_t} pebbles')
        trackpng = f'{name_key}tracklength.png'
        plt.savefig(trackpng)
        plt.close()

        plt.plot(sample_r, transit_num, color = plt_rgb[transit_rgb[1]],
                 linestyle = '', marker = '.')
        plt.xlim(0, 120)
        plt.xlabel('Initial relative radial position [r/R]')
        plt.ylabel('Transit number [-]')
        plt.title('Transit number vs relative radius; sample of {n_t} pebbles')
        transitpng = f'{name_key}transitnumber.png'
        plt.savefig(transitpng)
        plt.close()

        plt.plot(sample_r, res_time, color = plt_rgb[transit_rgb[2]],
                 linestyle = '', marker = '.')
        plt.xlim(0, 120)
        plt.xlabel('Initial relative radial position [r/R]')
        plt.ylabel('Residence time [days]')
        plt.title('Residence time vs relative radius; sample of {n_t} pebbles')
        restimepng = f'{name_key}restime.png'
        plt.savefig(restimepng)
        plt.close()


def deadzone(d_h5, r_h5, target_vmag = 4.8, frac = 0.10, 
             zone_bounds = [0, 10, 15, 20], n_layers = 15, cycle_days = 183, 
             name_key='', verbose=True, n_rgb=1000, 
             c0 = 0.1, c1 = 0.9, cmap='magma_r'):
    '''
    given data h5 and recirc h5, characterize the deadzone
    '''
    plt_c = np.linspace(c0, c1, n_rgb)
    plt_rgb = colormaps[cmap](plt_c)
    
    with h5py.File(d_h5, mode='r') as d5f, h5py.File(r_h5, mode='r') as r5f:
        N_r =  r5f.attrs['Nsteps']
        r_steps = np.arange(N_r)
        n_r = list(r5f['n_recirc_step'])
        n_r_cumul = list(r5f['n_recirc_cumul'])
        n_r_calc = np.array(list(map(int, n_r)))
        n_r_avg = sum(n_r_calc)/len(n_r_calc) 
        N_d = d5f.attrs['Nsteps']
        d_steps = np.arange(N_d)
        dt_r = d5f.attrs['n_recirc']
        dt_d = d5f.attrs['n_dump']
        i_settled = [d_steps[np.argmax(dt_d*d_steps == (i+1)*dt_r - dt_d)] 
                   for i in r_steps[:-1]]
        
        if verbose:
            n_settled = len(i_settled)
            i_yap = int((n_settled+1)/10)

        peb_R = d5f.attrs['peb_r']
        peb_D = 2*peb_R
        init_xyz = d5f['XYZ'][0]
        init_z = [xyz[2] for xyz in init_xyz] 
        zmax = 200
        zmin = 0
        peb_inv = len(init_xyz)
        p_per_d = peb_inv/cycle_days
        frac = 1/10
        dead_threshold = frac*abs((target_vmag/p_per_d)*n_r_avg)
        print(n_r_avg, dead_threshold)
        deadzone = {i_set:[] for i_set in i_settled}
        xyz0 = init_xyz
        recircn0 = d5f['recirc_n'][0]
        for i_cyc, i_set in enumerate(i_settled):
            set_xyz = d5f['XYZ'][i_set]
            set_recircn = d5f['recirc_n'][i_set]
            
            for i, xyz1 in enumerate(set_xyz):
                if set_recircn[i] != recircn0[i]:
                    recircn0[i] == set_recircn[i]
                else:
                    if xyz1[2] > zmin and xyz1[2] < zmax:
                        disp = sum((xyz0[i] - xyz1)**2)**0.5
                        if disp < dead_threshold:
                            deadzone[i_set].append(i)
            
            xyz0 = set_xyz
        
        dead_inv = np.zeros(peb_inv)
        for i_set, deadkey in deadzone.items():
            i_rgb = int(n_rgb*(i_set/i_settled[-1]))
            set_xyz = d5f['XYZ'][i_set]
            set_r = [sum(xyz[0:2]**2)**0.5 for xyz in set_xyz]
            set_z = [xyz[2] for xyz in set_xyz]
            dead_r = [set_r[dkey] for dkey in deadkey]
            dead_z = [set_z[dkey] for dkey in deadkey]
            plt.plot(dead_r, dead_z, linestyle='', marker='.', 
                     color = plt_rgb[i_rgb], alpha = 0.3)
            for dkey in deadkey:
                dead_inv[dkey] += 1
         
        plt.xlim(0, 120)
        plt.ylim(0, 100)
        plt.xlabel('r [cm]')
        plt.ylabel('z [cm]')
        titlestr = f'threshold {dead_threshold} [cm], {n_r_avg} pebbles/cycle.'
        plt.title('Dead pebble centroids: ' + titlestr)
        plt.savefig(f'{name_key}deadtest_{frac}.png')
        plt.close()
        print(max(dead_inv))















        








