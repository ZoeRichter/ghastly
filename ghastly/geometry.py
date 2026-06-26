import numpy as np
import h5py
import sys
import matplotlib.pyplot as plt
from ghastly import read_input
from matplotlib import colormaps


def _next_latt_z(latt_D, layer_a_z, peb_vz, days_cumul):
    '''
    determine the number of index positions between the depleting pebble's
    current lattice z-coordinate and the one for the new step.
    '''
    peb_vz = abs(peb_vz) # just in case 
    z0 = max(layer_a_z)
    true_z = z0 - peb_vz*(days_cumul)

    if true_z < 0:
        print('Pebble has left the core')
        nearest = None
    else:
        search = abs(layer_a_z/true_z - 1)
        nearest = np.argmax(search == min(search))

    return nearest

def shift_lattice(dep_z, latt_D, layer_a_z, peb_vz, days_cumul):
    '''
    given dep pebble's current position in the layer z array as an index,
    the cumulative days passed in the next step, cycle days, and lattice
    spacing, and the 
    '''

    peb_vz = abs(peb_vz)
    current = np.argmax(layer_a_z == dep_z)
    nearest = _next_latt_z(latt_D, layer_a_z, peb_vz, days_cumul)
    if nearest == None:
        shift = None
    else:
        shift = int(nearest - current)

    return shift


def _latt_z_arrays(latt_D, active_R, peb_R, n_pebs, pf):
    '''
    given input args, calculates and returns z-axis lattice spacing
    for the layer configuration that can have dep-pebbles
    '''

    bed_zmax = (4*n_pebs*peb_R**3)/(3*pf*active_R**2)

    a0 = latt_D*np.array([0, 0, 0.5])
    layer_z_offset = latt_D*np.array([0, 0, 3**0.5])
    layer_b_offset = latt_D*np.array([0.5, 1/(2*3**0.5), (2/3)**0.5])

    N_layer_a = int((bed_zmax - latt_D)/layer_z_offset[2])
    N_layer_b = int((bed_zmax - layer_b_offset[2] - 0.5*latt_D)/
                    layer_z_offset[2])
    
    layer_a_z = ([a0[2]] + [a0[2] + (i+1)*layer_z_offset[2] 
                            for i in range(N_layer_a)])
    layer_b_z = ([a0[2]+layer_b_offset[2]] + 
                 [a0[2]+layer_b_offset[2] + (i+1)*layer_z_offset[2] 
                  for i in range(N_layer_b)])

    return layer_a_z, layer_b_z

def gen_periph_geom(input_file):
    '''
    gen openmc periph
    '''

def insanity(d_h5, r_h5, peb_inv = 223000, cycle_days = 183, 
             name_key='', verbose = True):
    '''
    attempt to piece together a smaller number of recirculations into
    one continuous set of snapshots.
    this way lies madness
    '''

    with h5py.File(d_h5, mode='r') as d5f, h5py.File(r_h5, mode='r') as r5f:
        '''
        plan (???): take layer at the top (1) at t0 as your reference pebs 
        (you will pick dep pebs from here later).  get their positions at the 
        end of your data set (t-1).  save those somewhere (??? probably????).  
        return to t0.  find the pebbles that are closest to the postions of 
        ref. pebs at t-1.  log those.  new ref pebs (hooray)  find where they
        are at t-1 (pet bunger), return to t0, etc etc repeat until you 
        have something useful to make the psychic damage worth it.
        '''
        peb_R = d5f.attrs['peb_r']
        peb_D = 2*peb_R
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

        t0_layer = d5f['layer'][0]
        t0_layer1 = [i for i, l in enumerate(t0_layer) if l == 1]
        t0_xyz = d5f['XYZ'][0]
        layer1_zmax = max([t0_xyz[i_l1][2] for i_l1 in t0_layer1])
        layer1_zmin = min([t0_xyz[i_l1][2] for i_l1 in t0_layer1])
        layer1_zmid = layer1_zmin + 0.5*(layer1_zmax - layer1_zmin)
        z_cutoff = layer1_zmid
        too_low = True
        while too_low:
            thickness = layer1_zmax - z_cutoff
            if thickness <= 1.5*peb_D:
                too_low = False
            else:
                z_cutoff += 0.1

        ref0 = [i_l1 for i_l1 in t0_layer1 if t0_xyz[i_l1][2] >= z_cutoff]
        tend_xyz = d5f['XYZ'][i_settled[-1]]
        ref0_tend = [tend_xyz[i] for i in ref0]
        ref0_tend_r = [sum(t0_xyz[i][0:2]**2)**0.5 for i in ref0]
        r_asc_order = sorted(enumerate(ref0_tend_r), key=lambda x: x[1])
        print(z_cutoff, len(r_asc_order))
        
        ref1 = np.zeros(len(ref0), dtype=int)
        checklist = np.arange(len(t0_xyz))
        ticks = 0
        for r_asc in r_asc_order:
            ticks +=1
            i_asc = r_asc[0]
            R = [sum((ref0_tend[i_asc] - t0_xyz[i_peb])**2)**0.5 
                 for i_peb in checklist]
            R_sort = sorted(enumerate(R), key=lambda x:x[1])
            close3 = R_sort[0:3]
            close3_zdiff = abs(np.array([t0_xyz[i_3[0]][2] 
                                          for i_3 in close3]) 
                                         - ref0_tend[i_asc][2])
            i_match = R_sort[np.argmax(close3_zdiff == min(close3_zdiff))][0]
            ref1[i_asc] = checklist[i_match]
            checklist = np.delete(checklist, i_match)
            if ticks%(int(len(ref0)/10)) == 0:
                print(ticks)

        diff = abs(np.array([t0_xyz[i_r1] for i_r1 in ref1]) - ref0_tend)
        avg_diff = sum(diff)/len(diff)
        print(avg_diff)
