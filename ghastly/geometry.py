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
        shift == None
    else:
        shift = int(nearest - current)

    return shift


def _latt_z_array(latt_D, active_R, peb_R, n_pebs, pf):
    '''
    given input args, calculates and returns z-axis lattice spacing
    for the layer configuration that can have dep-pebbles
    '''

    bed_zmax = (4*n_pebs*peb_R**3)/(3*pf*active_R**2)

    a0 = latt_D*np.array([0, 0, 0.5])
    layer_z_offset = latt_D*np.array([0, 0, 3**0.5])
    N_layer_a = int((bed_zmax - latt_D)/layer_z_offset[2])
    layer_a_z = ([a0[2]] + [a0[2] + (i+1)*layer_z_offset[2] 
                            for i in range(N_layer_a)])

    return layer_a_z

def gen_periph_geom(input_file):
    '''
    gen openmc periph
    '''


