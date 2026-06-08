import numpy as np
import h5py
import sys
import matplotlib.pyplot as plt
from matplotlib import colormaps



def velocity_profiler(d_h5, r_h5, zone_bounds = [0, 10, 15, 20], 
                      chute_zmin = 0.0, chute_zmax = 54.0, 
                      chute_R = 24.0, core_R = 120.0,
                      cycle_days = 183, name_key='', verbose=True):
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
        i_settled = [d_steps[np.argmax(dt_d*d_steps == (i+1)*dt_r)] 
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
        safe_zmin = chute_zmin
        layer_H = chute_zmax-chute_zmin
        n_layers = int(bed_top/layer_H)
        safe_zmax = n_layers*layer_H
        layers = np.flip(np.linspace(safe_zmin, safe_zmax, n_layers+1))
        zones = peb_D*np.array(zone_bounds)
        n_zones = int(len(zones)) - 1
        core_vol = layer_H*np.array([ri**2 - zones[i-1]**2 
                                    for i, ri in list(enumerate(zones))[1:]])
        chute_m = (layer_H)/(core_R - chute_R)
        chute_vol = np.zeros(n_zones)
        for i, r1 in list(enumerate(zones))[1:]:
            r0 = zones[i-1]
            if r1 <= chute_R:
                vol_i = layer_H*(r1**2 - r0**2)
            else:
                r1_H = chute_m*(r1)
                r0_H = chute_m*(r0)
                vol_i = layer_H*r1**2 - (layer_H + r1_H - r0_H)*r0**2
            
            chute_vol[i-1] = vol_i
        chute_wt = chute_vol/core_vol
        print(chute_wt)

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


        vr_profile = (sum(vzr[0:-1]) + chute_wt*vzr[-1])/n_layers
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
                      n_layers = 17, cycle_days = 183, 
                     name_key='', verbose=True,
                     progymin = -1000, progymax = 1000,
                     vzymin = -11, vzymax = -3,
                     n_rgb=100, c0 = 0.1, c1 = 0.9, cmap='magma_r'):
    '''
    plot velocity profiles using the given csv with post-processed data from
    velocity_profiler()
    '''

    with open(d_csv, mode='r') as d:
        vr_profile = np.loadtxt(d, delimiter=',')[-1]
    

    plt_c = np.linspace(c0, c1, n_rgb)
    plt_rgb = colormaps[cmap](plt_c)
    latt_zmax = 9.2035e+02
    latt_zmin = 0
    zones = peb_D*np.array(zone_bounds)
    n_zones = len(zones) - 1
    zone_rgb = int(n_rgb/(n_zones+2))*np.arange(n_zones+2)[1:-1]
    zoneticks = np.array([int(zone) for zone in zones])

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
        
    plt.vlines(zones, progymin, progymax, 
               label = '_zonemarker', color = plt_rgb[-1])
    plt.hlines(latt_zmax, 0, 120, label= '_coremarker',
                   linestyle = '--', color=plt_rgb[-1])
    plt.hlines(latt_zmin, 0, 120, label= '_coremarker',
                   linestyle = '--', color=plt_rgb[-1])
    plt.legend(bbox_to_anchor=(1.02, 1.0))
    if name_key == '':
        fancyname = ''
    elif name_key == 'f2.5':
        fancyname = 'big-bulk model'
    elif name_key == 'f2':
        fancyname = 'small-bulk model'
    else:
        fancyname = ''
    plt.title(f'Axial progression of pebbles over 180 days {fancyname}')
    plt.ylim(progymin, progymax)
    plt.xlim(0, 120)
    plt.ylabel('Z [cm]')
    plt.xlabel('Middle of Radial Zone [cm]') 
    plt.xticks(zoneticks)
    progpng = f'{n_layers}axl_{n_zones}radz_pebprog{name_key}.png'
    plt.savefig(progpng, bbox_inches = 'tight', dpi = 600)
    plt.close()

    for i in range(n_zones):
        r_line = np.linspace(zones[i], zones[i+1], 10)
        vz_line = vr_profile[i]*np.ones(10)
        plt.plot(r_line, vz_line, lw=3.0,
                     color = plt_rgb[int(n_rgb/2)], label = f'Zone {i+1}')
            
        plt.vlines(zones[i+1], vzymin, vzymax,
                       label = '_zonemarker', color = plt_rgb[-1])
        
    plt.title(f'Radial vz profile in main core region {fancyname}')
    plt.ylim(vzymin, vzymax)
    plt.xlim(0, 120)
    plt.xlabel('r [cm]')
    plt.ylabel('vz [cm/day]')
    plt.xticks(zoneticks)
    velprofpng = f'{n_layers}axl_{n_zones}radz_velprof{name_key}.png'
    plt.savefig(velprofpng, bbox_inches = 'tight', dpi = 600)
    plt.close()



    for i in range(n_prog):
        plt.plot(zones[1:]**2, peb_prog[i], 
                     linestyle='-', marker='o',label = f'Day {i*15}',
                     color = plt_rgb[prog_rgb[i]])
        
    plt.vlines(zones**2, progymin, progymax, 
                       label = '_zonemarker', color = plt_rgb[-1])
    plt.hlines(latt_zmax, 0, 120**2, label= '_coremarker',
                   linestyle = '--', color=plt_rgb[-1])
    plt.hlines(latt_zmin, 0, 120**2, label= '_coremarker',
                   linestyle = '--', color=plt_rgb[-1])
    plt.legend(bbox_to_anchor=(1.01, 1.0))
    if name_key == '':
        fancyname = ''
    elif name_key == 'f2.5':
        fancyname = 'big-bulk model'
    elif name_key == 'f2':
        fancyname = 'small-bulk model'
    else:
        fancyname = ''
    plt.title(f'Axial progression of pebbles over 180 days {fancyname}')
    plt.ylim(progymin, progymax)
    plt.xlim(0, 120**2)
    plt.ylabel('Z [cm]')
    plt.xlabel('Radius squared [cm^2]') 
    if len(zones) > 7:
        plt.xticks(zones**2, rotation = 55, ha='right')
    else:
        plt.xticks(zones**2)
    progpng = f'{n_layers}axl_{n_zones}radz_pebprogcumul{name_key}.png'
    plt.savefig(progpng, bbox_inches = 'tight', dpi = 600)
    plt.close()

    for i in range(n_zones):
        r2_line = np.linspace(zones[i]**2, zones[i+1]**2, 10)
        vz_line = vr_profile[i]*np.ones(10)
        plt.plot(r2_line, vz_line, lw = 3.0, color = plt_rgb[int(n_rgb/2)])
    plt.vlines(zones**2, vzymin, vzymax,
                       label = '_zonemarker', color = plt_rgb[-1])
    plt.title(f'Radial vz profile in main core region {fancyname}')
    plt.ylim(vzymin, vzymax)
    plt.xlim(0, 120**2)
    plt.xlabel('Radius squared [cm^2]')
    plt.ylabel('vz [cm/day]')
    if len(zones) > 7:
        plt.xticks(zoneticks**2, rotation=55, ha='right')
    else:
        plt.xticks(zoneticks**2)
    velprofpng = f'{n_layers}axl_{n_zones}radz_velprofcumul{name_key}.png'
    plt.savefig(velprofpng, bbox_inches = 'tight', dpi = 600)
    plt.close()


def velocity_plottog(dcsv_a, name_a, dcsv_b, name_b, peb_D, 
                     zone_bounds = [0, 10, 15, 20], n_layers = 17, 
                     cycle_days = 183, name_key='', verbose=True, 
                     progymin = -1000, progymax = 1000, 
                     vzymin = -11, vzymax = -3, 
                     latt_zmax = 920.35, latt_zmin = 0,
                     n_rgb=100, c0 = 0.1, c1 = 0.9, cmap='magma_r'):
    '''
    given csvs with corewise velocity data a and velocity data b with the same
    radial zone boundaries, plot them together with a using a -r axis so they 
    meet at 0
    '''

    with open(dcsv_a, mode='r') as da, open(dcsv_b, mode='r') as db:
        vr_a = np.loadtxt(da, delimiter=',')
        vr_b = np.loadtxt(db, delimiter=',')
        vrprof_a = vr_a[-1]
        vrprof_b = vr_b[-1]
        zonevr_a = vr_a[:(n_layers-1)]
        zonevr_b = vr_b[:(n_layers-1)]
    

    plt_c = np.linspace(c0, c1, n_rgb)
    plt_rgb = colormaps[cmap](plt_c)
    
    zones = peb_D*np.array(zone_bounds)
    zones = np.array(zone_bounds)
    n_zones = len(zones) - 1
    zone_rgb = int(n_rgb/(n_zones+2))*np.arange(n_zones+2)[1:-1]
    rticksright =  np.array([int(zone) for zone in zones])/zones[-1]
    rticks = np.concatenate((np.flip(-1*rticksright[1:]), rticksright))
    r2ticksright = rticksright**2
    r2ticks = np.concatenate((np.flip(-1*r2ticksright[1:]), r2ticksright))

    n_prog = 13
    prog_rgb = int(n_rgb/(n_prog+2))*np.arange(n_prog+2)[1:-1] 
    pebprog_a = [latt_zmax + 15*i*vrprof_a for i in np.arange(n_prog)]
    pebprog_b = [latt_zmax + 15*i*vrprof_b for i in np.arange(n_prog)]
    prog_r = np.array([0.5*(rt - rticksright[i-1]) + rticksright[i-1] 
                             for i, rt in list(enumerate(rticksright))[1:]]) 
    
    prog_r2 = np.array([0.5*(rt2 - r2ticksright[i-1]) + r2ticksright[i-1]
               for i, rt2 in list(enumerate(r2ticksright))[1:]])

    for i in range(n_prog):
        plt.plot(-1*prog_r, pebprog_a[i], 
                     linestyle='-', marker='o',label = f'Day {i*15}',
                     color = plt_rgb[prog_rgb[i]])
        plt.plot(prog_r, pebprog_b[i], 
                     linestyle='-', marker='o',label = f'Day {i*15}',
                     color = plt_rgb[prog_rgb[i]])
        
    plt.vlines(rticks, progymin, progymax, 
               label = '_zonemarker', color = plt_rgb[-1])
    plt.hlines(latt_zmax, -1, 1, label= '_coremarker',
                   linestyle = '--', color=plt_rgb[-1])
    plt.hlines(latt_zmin, -1, 1, label= '_coremarker',
                   linestyle = '--', color=plt_rgb[-1])
    plt.legend(bbox_to_anchor=(1.01, 1.0))

    plt.suptitle('Z coordinate of pebbles over time six months')  
    plt.title(f'left: {name_a}, right: {name_b}')
    plt.ylim(progymin, progymax)
    plt.xlim(-1, 1)
    plt.ylabel('Z [cm]')
    plt.xlabel('Middle of Radial Zone [cm]') 
    plt.xticks(rticks, fontsize = 7, rotation = 90, ha='center')
    progpng = f'{n_layers}axl_{n_zones}radz_pebprog{name_key}.png'
    plt.savefig(progpng, bbox_inches = 'tight', dpi = 900)
    plt.close()


    for i in range(n_zones):
        rline_a = -1*np.linspace(zones[i], zones[i+1], 10)/zones[-1]
        rline_b = np.linspace(zones[i], zones[i+1], 10)/zones[-1]
        vzline_a = vrprof_a[i]*np.ones(10)
        vzline_b = vrprof_b[i]*np.ones(10)
        plt.plot(rline_a, vzline_a, lw=2.0,
                     color = plt_rgb[int(n_rgb/3)], label = f'{name_a}')
        plt.plot(rline_b, vzline_b, lw=2.0,
                     color = plt_rgb[int((2*n_rgb)/3)], label = f'{name_b}')
            
    plt.vlines(rticks, vzymin, vzymax, 
               label = '_zonemarker', color = plt_rgb[-1])
    plt.suptitle('Radial vz profile in main core region')
    plt.title(f'left: {name_a}, right: {name_b}')
    plt.ylim(vzymin, vzymax)
    plt.xlim(-1, 1)
    plt.xlabel('r [cm]')
    plt.ylabel('Z-component of velocity [cm/day]')
    plt.xticks(rticks, fontsize = 10, rotation = 90, ha = 'center')

    velprofpng = f'{n_layers}axl_{n_zones}radz_velprof{name_key}.png'
    plt.savefig(velprofpng, bbox_inches = 'tight', dpi = 900)
    plt.close()


    for i in range(n_prog):
        plt.plot(-1*prog_r2, pebprog_a[i], 
                     linestyle='-', marker='o',label = f'Day {i*15}',
                     color = plt_rgb[prog_rgb[i]])
        plt.plot(prog_r2, pebprog_b[i], 
                     linestyle='-', marker='o',label = f'Day {i*15}',
                     color = plt_rgb[prog_rgb[i]])
        
    plt.vlines(r2ticks, progymin, progymax, lw=0.5, 
               label = '_zonemarker', color = plt_rgb[-1])
    plt.hlines(latt_zmax, -1, 1, label= '_coremarker',
                   linestyle = '--', color=plt_rgb[-1])
    plt.hlines(latt_zmin, -1, 1, label= '_coremarker',
                   linestyle = '--', color=plt_rgb[-1])
    plt.legend(bbox_to_anchor=(1.01, 1.0))

    plt.suptitle('Z coordinate of pebbles over time six months')  
    plt.title(f'left: {name_a}, right: {name_b}')
    plt.ylim(progymin, progymax)
    plt.xlim(-1, 1)
    plt.ylabel('Z [cm]')
    plt.xlabel('Middle of Radial Zone [cm]') 
    if len(zones) > 7:
        r2ticks_less = np.concatenate((r2ticks[:n_zones-1], 
                                       [r2ticks[n_zones]], 
                                       r2ticks[n_zones+2:]))
        plt.xticks(r2ticks_less, fontsize = 8, rotation = 90, ha = 'center')
    else:
        plt.xticks(r2ticks, fontsize = 8, rotation = 90, ha = 'center')
    progpng = f'{n_layers}axl_{n_zones}radz_pebprogcumul{name_key}.png'
    plt.savefig(progpng, bbox_inches = 'tight', dpi = 900)
    plt.close()


    for i in range(n_zones):
        rline_a = -1*np.linspace(zones[i]**2, zones[i+1]**2, 10)/zones[-1]**2
        rline_b = np.linspace(zones[i]**2, zones[i+1]**2, 10)/zones[-1]**2
        vzline_a = vrprof_a[i]*np.ones(10)
        vzline_b = vrprof_b[i]*np.ones(10)
        plt.plot(rline_a, vzline_a, lw=2.0,
                     color = plt_rgb[int(n_rgb/3)], label = f'{name_a}')
        plt.plot(rline_b, vzline_b, lw=2.0,
                     color = plt_rgb[int((2*n_rgb)/3)], label = f'{name_b}')
            
    plt.vlines(r2ticks, vzymin, vzymax, lw=0.5,
               label = '_zonemarker', color = plt_rgb[-1])
    plt.suptitle('Radial vz profile in main core region')
    plt.title(f'left: {name_a}, right: {name_b}')
    plt.ylim(vzymin, vzymax)
    plt.xlim(-1, 1)
    plt.xlabel('r^2 [cm]')
    plt.ylabel('Z-component of velocity [cm/day]')
    if len(zones) > 7:
        r2ticks_less = np.concatenate((r2ticks[:n_zones-1], 
                                       [r2ticks[n_zones]], 
                                       r2ticks[n_zones+2:]))
        plt.xticks(r2ticks_less, fontsize = 8, rotation = 90, ha = 'center')
    else:
        plt.xticks(r2ticks, fontsize = 8, rotation=90, ha='center')
    velprofpng = f'{n_layers}axl_{n_zones}radz_velprofcumul{name_key}.png'
    plt.savefig(velprofpng, bbox_inches = 'tight', dpi = 900)
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
        i_settled = [d_steps[np.argmax(dt_d*d_steps == (i+1)*dt_r)] 
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


def deadzone(d_h5, r_h5, target_vmag = 5.0, frac = 0.10, 
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
        i_settled = [d_steps[np.argmax(dt_d*d_steps == (i+1)*dt_r)] 
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
        dead_threshold = frac*target_vmag
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
                        v = (disp/n_r[i_cyc])*p_per_d
                        if v < dead_threshold:
                            deadzone[i_set].append(i)
            
            xyz0 = set_xyz
        
        dead_inv = np.zeros(peb_inv)
        for i_set, deadkey in deadzone.items():
            i_rgb = int(n_rgb*(i_set/i_settled[-1])-1)
            set_xyz = d5f['XYZ'][i_set]
            set_r = [sum(xyz[0:2]**2)**0.5 for xyz in set_xyz]
            set_z = [xyz[2] for xyz in set_xyz]
            dead_r = [set_r[dkey] for dkey in deadkey]
            dead_z = [set_z[dkey] for dkey in deadkey]
            plt.plot(dead_r, dead_z, linestyle='', marker='.', 
                     color = plt_rgb[i_rgb], alpha = 0.3)
            for dkey in deadkey:
                dead_inv[dkey] += 1
        plt.axline((24, 0), (120, 54), linestyle = '--', color=plt_rgb[-1]) 
        plt.xlim(0, 120)
        plt.ylim(0, 100)
        plt.xlabel('r [cm]')
        plt.ylabel('z [cm]')
        threshstr = f'threshold {dead_threshold:.2f} [cm/day]. '
        recircstr = f'{int(n_r_avg)} pebbles/cycle: '
        titlestr = recircstr + threshstr
        plt.title('Dead pebble centroids, ' + titlestr)
        plt.savefig(f'{name_key}deadtest_{frac}.png')
        plt.close()
        print(max(dead_inv))


def _find_common_steps(Nr_a, nr_cumul_a, Nr_b, nr_cumul_b):
    '''
    given the cumulative recirc tally for two distinct simulations, 
    find steps with a similar number of total recirculated pebbles
    '''

    com_ind = np.zeros(Nr_a-1, dtype=int)
    for icyc_a in range(Nr_a-1):
        for icyc_b in range(Nr_b):
            if nr_cumul_a[icyc_a] <= nr_cumul_b[icyc_b]:
                com_ind[icyc_a] = icyc_b
                break


    return com_ind


def plot_paratog(name_a, d_h5a, r_h5a, name_b, d_h5b, r_h5b, 
                 para_name = 'para_tog', offset_mag = 130, offset_ax = 0, 
                 peb_inv = 229959, cycle_days = 183, name_key='', 
                 verbose=True, n_rgb=100, c0 = 0.1, c1 = 0.9, cmap='magma_r'):
    '''
    plot two sims together.  The total number of pebbles Sim A cycled must be
    less than or equal to the number Sim B cycled
    '''
    plt_c = np.linspace(c0, c1, n_rgb)
    plt_rgb = colormaps[cmap](plt_c)
    offset = np.zeros(3)
    offset[offset_ax] = offset_mag
    

    with (h5py.File(d_h5a, mode='r') as d5fa, 
          h5py.File(r_h5a, mode='r') as r5fa,
          h5py.File(d_h5b, mode='r') as d5fb,
          h5py.File(r_h5b, mode='r') as r5fb):

        Nr_a = r5fa.attrs['Nsteps']
        rsteps_a = np.arange(Nr_a)
        Nd_a = d5fa.attrs['Nsteps']
        dsteps_a = np.arange(Nd_a)
        dtr_a = d5fa.attrs['n_recirc']
        dtd_a = d5fa.attrs['n_dump']
        isettled_a = [dsteps_a[np.argmax(dtd_a*dsteps_a == (i+1)*dtr_a)] 
                   for i in rsteps_a[:-1]]

        Nr_b = r5fb.attrs['Nsteps']
        rsteps_b = np.arange(Nr_b)
        Nd_b = d5fb.attrs['Nsteps']
        dsteps_b = np.arange(Nd_b)
        dtr_b = d5fb.attrs['n_recirc']
        dtd_b = d5fb.attrs['n_dump']
        isettled_b = [dsteps_b[np.argmax(dtd_b*dsteps_b == (i+1)*dtr_b)] 
                   for i in rsteps_b[:-1]]


        nr_cumul_a = list(r5fa['n_recirc_cumul'])
        nr_cumul_b = list(r5fb['n_recirc_cumul'])
        nrtot_a = nr_cumul_a[-1]
        nrtot_b = nr_cumul_b[-1]
        
        assert nrtot_a <= nrtot_b, "Args in wrong order, switch and try again"
        com_ind = _find_common_steps(Nr_a, nr_cumul_a,
                                     Nr_b, nr_cumul_b)

        xyz0a = d5fa['XYZ'][0]
        bed_zmax = max([xyz[2] for xyz in xyz0a])
        layer_H = 54 # chute height
        n_layer = int(bed_zmax/layer_H)
        safe_zmax = n_layer*layer_H
        margin = 0.15
        enough = False
        while not enough:
            line0a = [i for i, xyz in enumerate(xyz0a) 
                      if xyz[2] <= safe_zmax + margin 
                      and xyz[2] >= safe_zmax - margin]
            if len(line0a) >= 1000:
                enough = True
            else:
                margin += 0.05
        xyz0b = d5fb['XYZ'][0]
        ib_list = np.arange(peb_inv)
        
        uid_ab_key = np.zeros(len(line0a), dtype = object)
        for i, ia in enumerate(line0a):
            for i_list, ib in enumerate(ib_list):
                if sum(xyz0a[ia] - xyz0b[ib]) == 0:
                    uid_ab_key[i] = (ia, ib)
                    ib_list = np.delete(ib_list, i_list)
                    break

        for ia, ib in enumerate(com_ind):
            setxyz_a = d5fa['XYZ'][isettled_a[ia]]
            setxyz_b = d5fb['XYZ'][isettled_b[ib]]
            paraxyz_a = []
            paraxyz_b = []
            for i, ipeb in enumerate(uid_ab_key):
                offset_a = setxyz_a[ipeb[0]] - offset
                offset_a = np.concatenate((offset_a, [i]))
                offset_b = setxyz_b[ipeb[1]] + offset
                offset_b = np.concatenate((offset_b, [i]))
                paraxyz_a.append(offset_a)
                paraxyz_b.append(offset_b)
                break
            
            paraxyz = np.array(paraxyz_a + paraxyz_b)
            pad = 4 - len(str(ia))
            csvstr = f'sharedstep_{pad*'0'}{ia}.csv'
            with open(csvstr, mode = 'w') as paracsv:
                np.savetxt(paracsv, paraxyz)
                


def find_i_day(d_h5, r_h5, day, 
                  cycle_days = 183, name_key='', verbose=True):
    '''
    given database, total elapsed days at the desired dep,
    return the recirc index that most closely matches the next dep step
    '''
    
    with h5py.File(d_h5, mode='r') as d5f, h5py.File(r_h5, mode='r') as r5f:

        Nr = r5f.attrs['Nsteps']
        r_steps = np.arange(Nr)
        Nrcumul = list(r5f['n_recirc_cumul'])
        peb_inv = d5f['XYZ'][0].shape[0]
        p_per_d = peb_inv/cycle_days
        peb_sofar = day*p_per_d
        
        search = abs(Nrcumul/peb_sofar - 1)
        i_day = np.argmax(search == min(search))
        return i_day
