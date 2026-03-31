import numpy as np
from ghastly import read_input
from ghastly import pebble
import os
import glob
import string
import h5py

rng = np.random.default_rng()

def _recirc_data(recircpath, sort_int):
    '''
    pull lammps recirc data given recirc directory, return P = N pebs recirc'd
    each recirc
    '''
    rpath = os.path.expanduser(recircpath)
    unsorted_r_fnames = glob.glob(os.path.join(rpath, "*.bin"))
    r_fnames = sorted(unsorted_r_fnames, key=lambda x:x[sort_int:])
    P = [float(read_input.read_lammps_bin(rfile, rpath, skiprows=3, 
                                          max_rows=1)) for rfile in r_fnames]
    P_uids = [[int(uid) for uid in group] 
              for group in [read_input.read_lammps_bin(rfile, 
                                                       rpath, 
                                                       skiprows=9) 
                            for rfile in r_fnames]]
    return P, P_uids


def create_data_hdf5(inputfile, coordpath, recircpath, n0=0, 
                     data_name="pebble_data", pad=15, sort_int=-19, n_skip=1, 
                     delimiter=' ', skiprows=9, n_recirc=2500000, 
                     n_dump=250000, dt=2.4790e-07, recirc_hz=0.014, 
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
    P, _ = _recirc_data(recircpath, sort_int)
    

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
    xmf_file = data_name + ".xmf"

    # delete the old h5 and create the new one from scratch.
    if recreate == True:
        try:
            os.remove(data_file)
        except OSError:
            pass
    
    # xmf is always recreated to make sure it reflects the updated h5.
    try:
        os.remove(xmf_file)
    except OSError:
        pass

    
    for tstep in range(n_tsteps):
        sim_step = int(n0 + tstep*n_dump)
        i_r = int((tstep*n_dump)//n_recirc)
        t_scale = P[i_r]/(n_recirc*dt*recirc_hz)
        sim_time = sim_step*dt
        reactor_time = sim_time*t_scale
        data = read_input.read_lammps_bin(c_fnames[tstep], cpath)
        uid = []
        coord = []
        vx, vy, vz, vmag = [], [], [], []
        zone = []
        layer = []
        pass_n = []
        recirc_n = []

        for d in data:
            uid.append(int(d[pattern[0]]))
            coord.append([c*d[pattern[1]], c*d[pattern[2]], c*d[pattern[3]]])
            zone.append(int(d[pattern[4]]))
            layer.append(int(d[pattern[5]]))
            pass_n.append(int(d[pattern[6]]))
            recirc_n.append(int(d[pattern[7]]))
            vxi, vyi, vzi = c*d[pattern[8]], c*d[pattern[9]], c*d[pattern[10]]
            vx.append(vxi)
            vy.append(vyi)
            vz.append(vzi)
            vmag.append(sum([vxi**2 + vyi**2 + vzi**2])**0.5)
            

        with h5py.File(data_file, mode='a') as h5f:
            #initialize h5 if needed:
            if (n0 == 0 or recreate == True) and tstep == 0:
                h5f.attrs['peb_r'] = c*sim_block.r_pebble
                h5f.attrs['dt'] = dt
                h5f.attrs['recirc_hz'] = recirc_hz
                h5f.attrs['n_dump'] = n_dump
                h5f.attrs['n_recirc'] = n_recirc

                _init_data_h5(h5f, coord, vx, vy, vz, vmag, uid, zone, layer, 
                              pass_n, recirc_n, reactor_time, t_scale)
            #append if not:
            else:
                new_size = h5f['xyz'].shape[0] + 1
                
                h5f['xyz'].resize(new_size, axis=0)
                h5f['xyz'][-1] = coord

                h5f['vx'].resize(new_size, axis=0)
                h5f['vx'][-1] = vx
                h5f['vy'].resize(new_size, axis=0)
                h5f['vy'][-1] = vy
                h5f['vz'].resize(new_size, axis=0)
                h5f['vz'][-1] = vz
                h5f['vmag'].resize(new_size, axis=0)
                h5f['vmag'][-1] = vmag

                h5f['uid'].resize(new_size, axis=0)
                h5f['uid'][-1] = uid
                h5f['zone'].resize(new_size, axis=0)
                h5f['zone'][-1] = zone
                h5f['layer'].resize(new_size, axis=0)
                h5f['layer'][-1] = layer
                h5f['pass_n'].resize(new_size, axis=0)
                h5f['pass_n'][-1] = pass_n
                h5f['recirc_n'].resize(new_size, axis=0)
                h5f['recirc_n'][-1] = recirc_n
                h5f['reactor_time'].resize(new_size, axis=0)
                h5f['reactor_time'][-1] = reactor_time
                h5f['time_scale'].resize(new_size, axis=0)
                h5f['time_scale'][-1] = t_scale


    _write_data_xmf(data_file, xmf_file)


def _init_data_h5(h5f, coord, vx, vy, vz, vmag, uid, zone, layer, 
                  pass_n, recirc_n, reactor_time, t_scale):
    '''
    christ:wq

    '''
    shape_n = len(coord)
    h5f.create_dataset('xyz', data=[coord], dtype=np.single,
                       chunks=True, maxshape=(None, shape_n, 3))

    h5f.create_dataset('vx', data=[vx], dtype=np.single,
                       chunks=True, maxshape=(None, shape_n))
    h5f.create_dataset('vy', data=[vy], dtype=np.single,
                       chunks=True, maxshape=(None, shape_n))
    h5f.create_dataset('vz', data=[vz], dtype=np.single,
                       chunks=True, maxshape=(None, shape_n))
    h5f.create_dataset('vmag', data=[vmag], dtype=np.single,
                       chunks=True, maxshape=(None, shape_n))

    h5f.create_dataset('uid', data=[uid], dtype=np.uintc,
                       chunks=True, maxshape=(None, shape_n))
    h5f.create_dataset('zone',  data=[zone], dtype=np.ubyte,
                       chunks=True, maxshape=(None, shape_n))
    h5f.create_dataset('layer', data=[layer], dtype=np.ubyte,
                       chunks=True, maxshape=(None, shape_n))
    h5f.create_dataset('pass_n', data=[pass_n], dtype=np.ubyte,
                       chunks=True, maxshape=(None, shape_n))
    h5f.create_dataset('recirc_n', data=[recirc_n], dtype=np.ubyte,
                       chunks=True, maxshape=(None, shape_n))
    h5f.create_dataset('reactor_time', data=[reactor_time], dtype=np.double,
                       chunks=True, maxshape=(None,))
    h5f.create_dataset('time_scale', data=[t_scale], dtype=np.single,
                       chunks=True, maxshape=(None,))


def _write_data_xmf(data_file, xmf_file):
    """
    Write an XMF file to accompany the HDF5 data file.
    """

    with h5py.File(data_file, mode='r') as h5f:
        tsteps = h5f['xyz'].shape[0]
        shape_n = h5f['xyz'].shape[1]
        n_dump = h5f.attrs['n_dump']

    with open(xmf_file, mode='w') as xmf:
        xmf.write('<?xml version="1.0" ?>\n')
        xmf.write('<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>\n')
        xmf.write('<Xdmf Version="3.0">\n')
        xmf.write('  <Domain>\n')
        # CollectionType="Temporal" for paraview
        xmf.write('    <Grid Name="Pebbles" GridType="Collection" CollectionType="Temporal">\n')
        for tstep in range(tsteps):
            xmf.write('      <Grid Name="Pebbles" GridType="Uniform">\n')
            xmf.write(f'        <Time Type="Single" Value="{tstep*n_dump}" />\n')
            xmf.write(f'        <Topology TopologyType="Polyvertex NumberOfElements="{shape_n}" />\n')
            xmf.write('        <Geometry GeometryType="XYZ">\n')
            xmf.write(f'          <DataItem Format="HDF" Dimensions="{shape_n} 3">\n')
            xmf.write(f'            {data_file}:/xyz\n')
            xmf.write('          </DataItem>\n')
            xmf.write('        </Geometry>\n')

            xmf.write('        <Attribute Name="Vx [cm/s]" AttributeType="Scalar" Center="Node">\n')
            xmf.write(f'          <DataItem Format="HDF" Dimensions="{shape_n}">\n')
            xmf.write(f'            {data_file}:/vx\n')
            xmf.write('          </DataItem>\n')
            xmf.write('        </Attribute>\n')
            xmf.write('        <Attribute Name="Vy [cm/s]" AttributeType="Scalar" Center="Node">\n')
            xmf.write(f'          <DataItem Format="HDF" Dimensions="{shape_n}">\n')
            xmf.write(f'            {data_file}:/vy\n')
            xmf.write('          </DataItem>\n')
            xmf.write('        </Attribute>\n')
            xmf.write('        <Attribute Name="Vz [cm/s]" AttributeType="Scalar" Center="Node">\n')
            xmf.write(f'          <DataItem Format="HDF" Dimensions="{shape_n}">\n')
            xmf.write(f'            {data_file}:/vz\n')
            xmf.write('          </DataItem>\n')
            xmf.write('        </Attribute>\n')

            xmf.write('        <Attribute Name="V Magnitude [cm/s]" AttributeType="Scalar" Center="Node">\n')
            xmf.write(f'          <DataItem Format="HDF" Dimensions="{shape_n}">\n')
            xmf.write(f'            {data_file}:/vmag\n')
            xmf.write('          </DataItem>\n')
            xmf.write('        </Attribute>\n')

            xmf.write('        <Attribute Name="Zone" AttributeType="Scalar" Center="Node">\n')
            xmf.write(f'          <DataItem Format="HDF" Dimensions="{shape_n}">\n')
            xmf.write(f'            {data_file}:/zone\n')
            xmf.write('          </DataItem>\n')
            xmf.write('        </Attribute>\n')

            xmf.write('        <Attribute Name="Layer" AttributeType="Scalar" Center="Node">\n')
            xmf.write(f'          <DataItem Format="HDF" Dimensions="{shape_n}">\n')
            xmf.write(f'            {data_file}:/layer\n')
            xmf.write('          </DataItem>\n')
            xmf.write('        </Attribute>\n')

            xmf.write('        <Attribute Name="Pass Number" AttributeType="Scalar" Center="Node">\n')
            xmf.write(f'          <DataItem Format="HDF" Dimensions="{shape_n}">\n')
            xmf.write(f'            {data_file}:/pass_n\n')
            xmf.write('          </DataItem>\n')
            xmf.write('        </Attribute>\n')

            xmf.write('        <Attribute Name="Number of Recirculations" AttributeType="Scalar" Center="Node">\n')
            xmf.write(f'          <DataItem Format="HDF" Dimensions="{shape_n}">\n')
            xmf.write(f'            {data_file}:/recirc_n\n')
            xmf.write('          </DataItem>\n')
            xmf.write('        </Attribute>\n')

            xmf.write('      </Grid>\n')
        
        xmf.write('    </Grid>\n')
        xmf.write('  </Domain>\n')
        xmf.write('</Xdmf>\n')



