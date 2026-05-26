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

    data_file = data_name + ".hdf"
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
        data = read_input.read_lammps_bin(c_fnames[tstep], cpath)
        uid = []
        coord = []
        v, vmag = [], []
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
            vi = [c*d[pattern[8]], c*d[pattern[9]], c*d[pattern[10]]]
            v.append(vi)
            vmag.append(sum([vi[0]**2 + vi[1]**2 + vi[2]**2])**0.5)
        
        uid_key = sorted(enumerate(uid), key=lambda x: x[1])
        coord = [coord[i] for i, _ in uid_key]
        zone = [zone[i] for i, _ in uid_key]
        layer = [layer[i] for i, _ in uid_key]
        pass_n = [pass_n[i] for i, _ in uid_key]
        recirc_n = [recirc_n[i] for i, _ in uid_key]
        v = [v[i] for i, _ in uid_key]
        vmag = [vmag[i] for i, _ in uid_key]
        shape_n = len(coord)


        with h5py.File(data_file, mode='a') as h5f:
            #initialize h5 if needed:
            if (n0 == 0 or recreate == True) and tstep == 0:
                h5f.attrs['peb_r'] = c*sim_block.r_pebble
                h5f.attrs['dt'] = dt
                h5f.attrs['recirc_hz'] = recirc_hz
                h5f.attrs['n_dump'] = n_dump
                h5f.attrs['n_recirc'] = n_recirc
                root = h5f.create_group("VTKHDF")
                _init_data_h5(root, coord, v, vmag, zone, layer, 
                              pass_n, recirc_n, t_scale)
            #append if not:
            else:
                new_size = h5f["VTKHDF"]['Points'].shape[0] + 1
                h5f["VTKHDF"]['Steps'].attrs['Nsteps'] = new_size
                h5f["VTKHDF"]['Steps']['Values'].resize(new_size, axis=0)
                h5f["VTKHDF"]['Steps']['Values'][-1] = new_size-1
                h5f["VTKHDF"]['Steps']['PartOffsets'].resize(new_size, axis=0)
                h5f["VTKHDF"]['Steps']['PartOffsets'][-1] = new_size-1
                
                h5f["VTKHDF"]['Points'].resize(new_size, axis=0)
                h5f["VTKHDF"]['Points'][-1] = coord
                h5f["VTKHDF"]['Steps']['PointOffsets'].resize(new_size, axis=0)
                h5f["VTKHDF"]['Steps']['PointOffsets'][-1] = new_size-1
                h5f["VTKHDF"]['NumberofPoints'].resize(new_size, axis=0)
                h5f["VTKHDF"]['NumberofPoints'][-1] = shape_n

                h5f["VTKHDF"]['PointData']['v'].resize(new_size, axis=0)
                h5f["VTKHDF"]['PointData']['v'][-1] = v
                h5f["VTKHDF"]['Steps']['PointDataOffsets']['v'].resize(new_size, axis=0)
                h5f["VTKHDF"]['Steps']['PointDataOffsets']['v'][-1] = new_size-1

                h5f["VTKHDF"]['PointData']['vmag'].resize(new_size, axis=0)
                h5f["VTKHDF"]['PointData']['vmag'][-1] = vmag
                h5f["VTKHDF"]['Steps']['PointDataOffsets']['vmag'].resize(new_size, axis=0)
                h5f["VTKHDF"]['Steps']['PointDataOffsets']['vmag'][-1] = new_size-1

                h5f["VTKHDF"]['PointData']['zone'].resize(new_size, axis=0)
                h5f["VTKHDF"]['PointData']['zone'][-1] = zone
                h5f["VTKHDF"]['Steps']['PointDataOffsets']['zone'].resize(new_size, axis=0)
                h5f["VTKHDF"]['Steps']['PointDataOffsets']['zone'][-1] = new_size-1

                h5f["VTKHDF"]['PointData']['layer'].resize(new_size, axis=0)
                h5f["VTKHDF"]['PointData']['layer'][-1] = layer
                h5f["VTKHDF"]['Steps']['PointDataOffsets']['layer'].resize(new_size, axis=0)
                h5f["VTKHDF"]['Steps']['PointDataOffsets']['layer'][-1] = new_size-1
                
                h5f["VTKHDF"]['PointData']['pass_n'].resize(new_size, axis=0)
                h5f["VTKHDF"]['PointData']['pass_n'][-1] = pass_n
                h5f["VTKHDF"]['Steps']['PointDataOffsets']['pass_n'].resize(new_size, axis=0)
                h5f["VTKHDF"]['Steps']['PointDataOffsets']['pass_n'][-1] = new_size-1
                
                h5f["VTKHDF"]['PointData']['recirc_n'].resize(new_size, axis=0)
                h5f["VTKHDF"]['PointData']['recirc_n'][-1] = recirc_n
                h5f["VTKHDF"]['Steps']['PointDataOffsets']['recirc_n'].resize(new_size, axis=0)
                h5f["VTKHDF"]['Steps']['PointDataOffsets']['recirc_n'][-1] = new_size-1

                h5f["VTKHDF"]['PointData']['time_scale'].resize(new_size, axis=0)
                h5f["VTKHDF"]['PointData']['time_scale'][-1] = t_scale
                h5f["VTKHDF"]['Steps']['PointDataOffsets']['time_scale'].resize(new_size, axis=0)
                h5f["VTKHDF"]['Steps']['PointDataOffsets']['time_scale'][-1] = new_size-1


                h5f["VTKHDF"]['Vertices']['Connectivity'].resize(new_size, axis=0)
                h5f["VTKHDF"]['Vertices']['Connectivity'][-1] =  np.arange(shape_n)
                h5f["VTKHDF"]['Vertices']['Offsets'].resize(new_size, axis=0)
                h5f["VTKHDF"]['Vertices']['Offsets'][-1] = np.arange(1, shape_n+1)
                h5f["VTKHDF"]['Steps']['CellOffsets'].resize(new_size, axis=0)
                h5f["VTKHDF"]['Steps']['CellOffsets'][-1] = [0,0,0,0]
                h5f["VTKHDF"]['Steps']['ConnectivityIdOffsets'].resize(new_size, axis=0)
                h5f["VTKHDF"]['Steps']['ConnectivityIdOffsets'][-1] = [0,0,0,0]




def _init_data_h5(root, coord, v, vmag, zone, layer, 
                  pass_n, recirc_n, t_scale):
    '''
    init h5, including the root vtk group
    '''
    shape_n = len(coord)
    root.attrs["Version"] = (2, 0)
    type_ASCII = 'vtkPolyData'.encode('ascii')
    root.attrs.create('Type', type_ASCII, 
                      dtype=h5py.string_dtype('ascii', len(type_ASCII)))
    root.create_dataset('NumberofPoints', data=[shape_n], maxshape=(None,), 
                        dtype=np.uint32)
    root.create_dataset('Points', data=[coord], maxshape=(None, shape_n, 3), dtype='f')

    vertices = root.create_group('Vertices')
    vertices.create_dataset('Connectivity', data=[np.arange(shape_n)], 
                            dtype=np.int32, maxshape=(None, shape_n))
    vertices.create_dataset('Offsets', data=[np.arange(1, shape_n+1)],
                            dtype=np.int32, maxshape=(None, shape_n))

    point_data = root.create_group('PointData')

    point_data.create_dataset('v', data=[v], dtype='f',
                       chunks=True, maxshape=(None, shape_n, 3))
    point_data.create_dataset('vmag', data=[vmag], dtype='f',
                       chunks=True, maxshape=(None, shape_n))

    point_data.create_dataset('zone',  data=[zone], dtype=np.uint8,
                       chunks=True, maxshape=(None, shape_n))
    point_data.create_dataset('layer', data=[layer], dtype=np.uint8,
                       chunks=True, maxshape=(None, shape_n))
    point_data.create_dataset('pass_n', data=[pass_n], dtype=np.uint8,
                       chunks=True, maxshape=(None, shape_n))
    point_data.create_dataset('recirc_n', data=[recirc_n], dtype=np.uint8,
                       chunks=True, maxshape=(None, shape_n))
    point_data.create_dataset('time_scale', data=[t_scale], dtype='f',
                       chunks=True, maxshape=(None,))

    steps = root.create_group('Steps')
    steps.attrs['NSteps'] = 1
    steps.create_dataset('Values', data=[0], maxshape=(None,), dtype=np.uint32)
    steps.create_dataset('PartOffsets', data=[0], maxshape=(None,), dtype=np.int32)
    #steps.create_dataset('NumberOfParts', data=[0], maxshape=(None,), dtype='i8')
    steps.create_dataset('PointOffsets', data=[0], maxshape=(None,), dtype=np.int32)
    steps.create_dataset('CellOffsets', data=[[0,0,0,0]], maxshape=(None,4), dtype=np.int32)
    steps.create_dataset('ConnectivityIdOffsets', data=[[0,0,0,0]], 
                         maxshape=(None,4), dtype=np.int32)
    point_data_offsets = steps.create_group('PointDataOffsets')
    point_data_offsets.create_dataset('v', data=[0], 
                                      maxshape=(None,), dtype=np.int32)
    point_data_offsets.create_dataset('vmag', data=[0],
                                      maxshape=(None,), dtype=np.int32)
    point_data_offsets.create_dataset('zone', data=[0],
                                      maxshape=(None,), dtype=np.int32)
    point_data_offsets.create_dataset('layer', data=[0],
                                      maxshape=(None,), dtype=np.int32)
    point_data_offsets.create_dataset('pass_n', data=[0],
                                      maxshape=(None,), dtype=np.int32)
    point_data_offsets.create_dataset('recirc_n', data=[0],
                                      maxshape=(None,), dtype=np.int32)
    point_data_offsets.create_dataset('time_scale', data=[0], maxshape=(None,), 
                                      dtype=np.int32)

def create_paraview_hdf5(inputfile, coordpath, recircpath, 
                     data_name="pebble_data", sort_int=-19, n_skip=1, 
                     delimiter=' ', skiprows=9, n_recirc=2500000, 
                     n_dump=250000, dt=2.4790e-07, recirc_hz=0.014, 
                     pattern=[0, 9, 10, 11, 2, 3, 4, 5, 6, 7, 8], units = 'm'):
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

    
    for tstep in range(n_tsteps):
        data = read_input.read_lammps_bin(c_fnames[tstep], cpath)
        uid = []
        coord = []
        vmag = []
        zone = []
        layer = []

        for d in data:
            uid.append(int(d[pattern[0]]))
            coord.append([c*d[pattern[1]], c*d[pattern[2]], c*d[pattern[3]]])
            zone.append(int(d[pattern[4]]))
            layer.append(int(d[pattern[5]])) 
            vi = [c*d[pattern[8]], c*d[pattern[9]], c*d[pattern[10]]] 
            vmag.append(sum([vi[0]**2 + vi[1]**2 + vi[2]**2])**0.5)
        
        uid_key = sorted(enumerate(uid), key=lambda x: x[1])
        coord = [coord[i] for i, _ in uid_key]
        zone = [zone[i] for i, _ in uid_key]
        layer = [layer[i] for i, _ in uid_key] 
        vmag = [vmag[i] for i, _ in uid_key]
        shape_n = len(coord)


        pad = 10 - len(str(tstep))
        group_name = pad*'0'+str(tstep)
        with h5py.File(data_file, mode='a') as h5f: 
            group = h5f.require_group(group_name)

            group.create_dataset('xyz', data=coord)
 
            group.create_dataset('vmag', data=vmag)
            
            group.create_dataset('zone',
                                 data=zone, dtype=np.uint8)
            group.create_dataset('layer', 
                                 data=layer, dtype=np.uint8) 

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

            xmf.write('      </Grid>\n')
        
        
        xmf.write('    </Grid>\n')
        xmf.write('  </Domain>\n')
        xmf.write('</Xdmf>\n')
        



