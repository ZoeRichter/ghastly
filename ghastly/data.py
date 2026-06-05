import numpy as np
from ghastly import read_input
from ghastly import pebble
import os
import glob
import string
import h5py

rng = np.random.default_rng()

def create_recirc_hdf5(recircpath, sort_int=-19, 
                      data_name = 'recirc_data', recreate=False):
    '''
    pull lammps recirc data given recirc directory, return P = N pebs recirc'd
    each recirc
    '''
    rpath = os.path.expanduser(recircpath)
    unsorted_r_fnames = glob.glob(os.path.join(rpath, "*.bin"))
    r_fnames = sorted(unsorted_r_fnames, key=lambda x:x[sort_int:])
    P = [float(read_input.read_lammps_bin(rfile, rpath, skiprows=3, 
                                          max_rows=1)) for rfile in r_fnames]
    P_cumul = np.cumsum(P)
    P_uids = [[int(uid) for uid in group] 
              for group in [read_input.read_lammps_bin(rfile, 
                                                       rpath, 
                                                       skiprows=9) 
                            for rfile in r_fnames]]

    data_file = data_name + ".h5"

    # delete the old h5 and create the new one from scratch.
    if recreate == True:
        try:
            os.remove(data_file)
        except OSError:
            pass

    with h5py.File(data_file, mode='a') as h5f:
            #initialize h5 if needed:
            if (recreate == True):
                h5f.attrs['Nsteps'] = len(P)
                h5f.create_dataset('n_recirc_step',  data=P, dtype=np.uint16,
                       chunks=True, maxshape=(None))
                h5f.create_dataset('n_recirc_cumul',  data=P_cumul, 
                                   dtype=np.uint32,
                       chunks=True, maxshape=(None))
            #append if not:
            else:
                for i, p in enumerate(P):
                    new_size = h5f['n_recirc_step'].shape[0] + 1
                    h5f.attrs['Nsteps'] = new_size
                    h5f['n_recirc_step'].resize(new_size, axis=0)
                    h5f['n_recirc_step'][-1] = p
                    h5f['n_recirc_cumul'].resize(new_size, axis=0)
                    h5f['n_recirc_cumul'][-1] = P_cumul[i]

    


def create_data_hdf5(inputfile, coordpath,
                     data_name="pebble_data", pad=15, sort_int=-19, n_skip=1, 
                     delimiter=' ', skiprows=9, n_recirc=2500000, 
                     n_dump=250000, 
                     pattern=[0, 9, 10, 11, 2, 3, 4, 5, 6, 7, 8], units = 'm',
                     recreate=False):
    '''
    create peb coord and reactor-time converted velocity hdf5 w/ extra data 
    needed for paraview plotting. values in cm for later openmc use. 
    also time step value csv
    '''

    inp_path = os.path.expanduser(inputfile)
    inp_block = read_input.InputBlock(inp_path)
    sim_block = inp_block.create_obj()
    

    cpath = os.path.expanduser(coordpath)
    unsorted_c_fnames = glob.glob(os.path.join(cpath, "*.bin"))
    c_fnames = sorted(unsorted_c_fnames, key=lambda x:x[sort_int:])
    n_tsteps = len(c_fnames)
    match units:
        case 'm':
            c = 100
        case 'cm':
            c = 1
        case 'in':
            c = 2.54
        case 'ft':
            c = 30.48

    data_file = data_name + ".h5"

    # delete the old h5 and create the new one from scratch.
    if recreate == True:
        try:
            os.remove(data_file)
        except OSError:
            pass

    
    for tstep in range(n_tsteps): 
        data = read_input.read_lammps_bin(c_fnames[tstep], cpath)
        uid = []
        coord = []
        v, vmag = [], []
        zone = []
        layer = []
        recirc_n = []

        for d in data:
            uid.append(int(d[pattern[0]]))
            coord.append([c*d[pattern[1]], c*d[pattern[2]], c*d[pattern[3]]])
            zone.append(int(d[pattern[4]]))
            layer.append(int(d[pattern[5]]))
            recirc_n.append(int(d[pattern[7]]))
            vi = [c*d[pattern[8]], c*d[pattern[9]], c*d[pattern[10]]]
            v.append(vi)
            vmag.append(sum([vi[0]**2 + vi[1]**2 + vi[2]**2])**0.5)
        
        uid_key = sorted(enumerate(uid), key=lambda x: x[1])
        coord = [coord[i] for i, _ in uid_key]
        zone = [zone[i] for i, _ in uid_key]
        layer = [layer[i] for i, _ in uid_key]
        recirc_n = [recirc_n[i] for i, _ in uid_key]
        v = [v[i] for i, _ in uid_key]
        vmag = [vmag[i] for i, _ in uid_key]
        shape_n = len(coord)


        with h5py.File(data_file, mode='a') as h5f:
            #initialize h5 if needed:
            if (recreate == True) and tstep == 0:
                h5f.attrs['peb_r'] = c*sim_block.r_pebble
                h5f.attrs['n_dump'] = n_dump
                h5f.attrs['n_recirc'] = n_recirc
                _init_data_h5(h5f, coord, v, vmag, zone, layer, recirc_n)
            #append if not:
            else:
                new_size = h5f['XYZ'].shape[0] + 1
                h5f.attrs['Nsteps'] = new_size 
                
                h5f['XYZ'].resize(new_size, axis=0)
                h5f['XYZ'][-1] = coord

                h5f['v'].resize(new_size, axis=0)
                h5f['v'][-1] = v
                

                h5f['vmag'].resize(new_size, axis=0)
                h5f['vmag'][-1] = vmag
                

                h5f['zone'].resize(new_size, axis=0)
                h5f['zone'][-1] = zone


                h5f['layer'].resize(new_size, axis=0)
                h5f['layer'][-1] = layer
                 
                
                h5f['recirc_n'].resize(new_size, axis=0)
                h5f['recirc_n'][-1] = recirc_n




def _init_data_h5(h5f, coord, v, vmag, zone, layer, recirc_n):
    '''
    init h5, including the root vtk group
    '''
    shape_n = len(coord)
    
    
    h5f.create_dataset('XYZ', data=[coord], maxshape=(None, shape_n, 3), dtype='f')

    h5f.create_dataset('v', data=[v], dtype='f',
                       chunks=True, maxshape=(None, shape_n, 3))
    h5f.create_dataset('vmag', data=[vmag], dtype='f',
                       chunks=True, maxshape=(None, shape_n))

    h5f.create_dataset('zone',  data=[zone], dtype=np.uint8,
                       chunks=True, maxshape=(None, shape_n))
    h5f.create_dataset('layer', data=[layer], dtype=np.uint8,
                       chunks=True, maxshape=(None, shape_n)) 
    h5f.create_dataset('recirc_n', data=[recirc_n], dtype=np.uint8,
                       chunks=True, maxshape=(None, shape_n))
    h5f.attrs['Nsteps'] = 1
    

def create_paraview_hdf5(data_path, data_name="para_plot", 
                         sort_int=-19, n_skip=1, 
                         delimiter=' ', skiprows=9):
    '''
    create peb coord and reactor-time converted velocity hdf5 w/ extra data 
    needed for paraview plotting. values in cm for later openmc use. 
    also time step value csv
    ''' 

    data_file = data_name + ".h5"
    xmf_file = data_name + ".xmf"

    # delete the old h5 and create the new one from scratch. 
    try:
        os.remove(data_file)
    except OSError:
        pass
    
    # xmf is always recreated to make sure it reflects the updated h5.
    try:
        os.remove(xmf_file)
    except OSError:
        pass

    with h5py.File(data_path, mode='r') as d5f:
        t_steps = list(range(d5f.attrs['Nsteps']))

        for i in t_steps:
            pad = 6 - len(str(i))
            group_name = pad*'0'+str(i)
            with h5py.File(data_file, mode='a') as h5f: 
                group = h5f.require_group(group_name)
                group.create_dataset('xyz', data=d5f['XYZ'][i])
                group.create_dataset('vmag', data=d5f['vmag'][i])
                group.create_dataset('zone', data=d5f['zone'][i])
                group.create_dataset('layer', data=d5f['layer'][i])
                group.create_dataset('recirc_n', data=d5f['recirc_n'][i]) 

    _write_data_xmf(data_file, xmf_file)

def _write_data_xmf(data_file, xmf_file):
    """
    Write an XMF file to accompany the HDF5 data file.
    """

    with h5py.File(data_file, mode='r') as h5f:
        h5keys = list(h5f.keys())
        shape_n = len(h5f[h5keys[0]]['xyz'])
        n_dset = len(h5f)

    with open(xmf_file, mode='w') as xmf:
        xmf.write('<?xml version="1.0" ?>\n')
        xmf.write('<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>\n')
        xmf.write('<Xdmf Version="3.0">\n')
        xmf.write('  <Domain>\n')
        # CollectionType="Temporal" for paraview
        xmf.write('    <Grid Name="Pebbles" GridType="Collection" CollectionType="Temporal">\n')
        for i, h5k in enumerate(h5keys):
            xmf.write('      <Grid Name="Pebbles" GridType="Uniform">\n')
            xmf.write(f'        <Time Value="{h5k}" />\n')

            xmf.write(f'        <Topology TopologyType="Polyvertex" '
                      'NumberOfElements="{shape_n}" />\n')
            xmf.write('        <Geometry GeometryType="XYZ">\n')
            xmf.write(f'          <DataItem Format="HDF" Dimensions="{shape_n} 3">\n')
            xmf.write(f'            {data_file}:/{h5k}/xyz\n')
            xmf.write('          </DataItem>\n')
            xmf.write('        </Geometry>\n')

            xmf.write('        <Attribute Name="V Magnitude" AttributeType="Scalar" Center="Node">\n')
            xmf.write(f'          <DataItem Format="HDF" Dimensions="{shape_n}">\n')
            xmf.write(f'            {data_file}:/{h5k}/vmag\n')
            xmf.write('          </DataItem>\n')
            xmf.write('        </Attribute>\n')

            xmf.write('        <Attribute Name="Zone" AttributeType="Scalar" Center="Node">\n')
            xmf.write(f'          <DataItem Format="HDF" Dimensions="{shape_n}">\n')
            xmf.write(f'            {data_file}:/{h5k}/zone\n')
            xmf.write('          </DataItem>\n')
            xmf.write('        </Attribute>\n')

            xmf.write('        <Attribute Name="Layer" AttributeType="Scalar" Center="Node">\n')
            xmf.write(f'          <DataItem Format="HDF" Dimensions="{shape_n}">\n')
            xmf.write(f'            {data_file}:/{h5k}/layer\n')
            xmf.write('          </DataItem>\n')
            xmf.write('        </Attribute>\n')

            xmf.write('        <Attribute Name="Number of Recirculations" AttributeType="Scalar" Center="Node">\n')
            xmf.write(f'          <DataItem Format="HDF" Dimensions="{shape_n}">\n')
            xmf.write(f'            {data_file}:/{h5k}/recirc_n\n')
            xmf.write('          </DataItem>\n')
            xmf.write('        </Attribute>\n')

            xmf.write('      </Grid>\n')
        
        
        xmf.write('    </Grid>\n')
        xmf.write('  </Domain>\n')
        xmf.write('</Xdmf>\n')
        



