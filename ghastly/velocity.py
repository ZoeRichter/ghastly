import numpy as np
import openmc
from ghastly import read_input
from ghastly import pebble
from ghastly import data
from jinja2 import Environment, PackageLoader
from os import path
import glob
import h5py
import matplotlib.pyplot as plt
import matplotlib as mpl
from cycler import cycler
import pandas as pd
import ast

default_cmap = mpl.colormaps['magma']
default_cycler = (cycler(color=default_cmap(np.linspace(0,1,6))))

plt.rc('axes', prop_cycle=default_cycler)

def vel_profiler(inputfile, csvpath, recircpath, radial_zones,
                 sort_int=-19, n_skip=1, delimiter = ' ', skiprows = 9,
                 n_dump = 250000, n_recirc = 2500000,
                 dt=2.4790e-07, recirc_hz=0.014):
    '''
    generate velocity profile plots
    '''

    inp_path = path.expanduser(inputfile)
    inp_block = read_input.InputBlock(inp_path)
    sim_block = inp_block.create_obj()

    peb_d = 2*100*sim_block.r_pebble
    P, _ = data._recirc_data(recircpath, sort_int)

    cpath = path.expanduser(csvpath)
    unsorted_filenames = glob.glob(path.join(cpath, "*_data.csv"))
    csvfiles = sorted(unsorted_filenames, key=lambda x:x[:sort_int])
    peb_data = {}
    count = 0
    names = ['uid', 'x', 'y', 'z', 'vx', 'vy', 'vz', 
             'vx_adj', 'vy_adj', 'vz_adj', 'peb_r', 
             'zone', 'layer', 'n_pass', 'n_recirc']
    for csvfile in csvfiles:
        csvcols = pd.read_csv(csvfile, names=names, 
                              index_col=False).columns.tolist()
        converter = {col:ast.literal_eval for col in csvcols}
        df = pd.read_csv(csvfile, names=names, 
                         index_col=False, converters=converter)
        if count == 1:
            print(df)
        count += 1
    

def find_wavefront(inputfile, h5_file, 
                   zone_bounds = [10, 15, 20], depth_factor=1.5):
    '''
    given input info, return indices (aka UIDs) of pebbles that create the
    initial 'wave front', used to determine flow regions and select
    depleting pebbles
    '''
    inp_path = path.expanduser(inputfile)
    inp_block = read_input.InputBlock(inp_path)
    sim_block = inp_block.create_obj()

    peb_R = 100*sim_block.r_pebble
    peb_D = 2*peb_R
    wavefront = []
    with h5py.File(h5_file, mode='r') as h5f:
        init_xyz = h5f["VTKHDF"]["Points"][0]
        peb_zmax = max([xyz[2] for xyz in init_xyz])
        for i, xyz in enumerate(init_xyz):
            if xyz[2] >= peb_zmax - depth_factor*peb_D:
                wavefront.append(i)
        n_steps = list(range(h5f["VTKHDF"]['Steps'].attrs['Nsteps']))
        print(n_steps)

### anything below this line is in development/for testing/scratch/unfinished
def vel_profilerdeletelater(directory, velpath, recircpath,
                 inputfile, vel_widths, sort_int=-19,
                 n_skip=1, delimiter = ' ', skiprows = 9,
                 n_dump = 372366, n_recirc = 2500000,
                 dt=2.6855e-07, recirc_hz=0.014):
    '''
    generate a velocity profile plot given a path to a top level velocity
    output directory, a list of the velocity subdirectories within it 
    (no slashes), the associated ghastly inputfile, and the widths of the
    v_regs, as a function of pebble diameter
    (so a region of width 2*d_peb is 2)
    '''

    inp_path = path.expanduser(inputfile)
    inp_block = read_input.InputBlock(inp_path)
    sim_block = inp_block.create_obj()

    peb_d = 2*100*sim_block.r_pebble
    P, _ = data._recirc_data(recircpath, sort_int)
    
    vel_all_time = {}
    for i in range(n_files):
        i_r = int((i*(n_dump))//n_recirc)
        t_scale = P[i_r]/(n_recirc*dt*recirc_hz)
        if i%n_skip==0:
            vel_map = {}
            for j, vpath in enumerate(velpath):
                bin_dir = path.join(vel_dir, vpath)
                bin_fname = vel_fnames[vpath][i]
                data = read_input.read_lammps_bin(bin_fname,bin_dir)
            
                raw_data = {}
                for d in data:
                    v_z = 100*d[-1]/t_scale
                    z = 100*d[1]
                    q = int(z//peb_d)
                    if q in vel_map:
                        pass
                    else:
                        vel_map[q] = r_regs*[float("nan")]

                    if q in raw_data:
                        raw_data[q].append(v_z)
                    else:
                        raw_data[q] = [v_z]

                for bin_k, bin_v in raw_data.items():
                    n_pebs = len(bin_v)
                    tot_vel = sum(bin_v)
                    avg = tot_vel/n_pebs
                    vel_map[bin_k][j] = avg

        
            ticks = []
            for j, _ in enumerate(vel_widths):
                ticks.append(sum(vel_widths[:j]))
            r = peb_d*np.array(ticks+[ticks[-1]+vel_widths[-1]])
        
            sorted_vel = dict(sorted(vel_map.items(), reverse=True))

            y = peb_d*(np.array(list(sorted_vel.keys())+[-23.0]))
            z_vel = list(sorted_vel.values())
        
            plt.pcolormesh(r, y, z_vel, vmin=-100, vmax=100, shading='flat', 
                           norm='symlog')
            plt.set_cmap('managua')
            plt.gca().set_aspect('equal')
            plt.xticks([0,40,80,120])
            plt.xlabel('Radius [cm]')
            plt.ylabel('Height[cm]')
            plt.ylim(-138.0,200.0)
            plt.title("Radial and Axial Velocity Profile")
            plt.colorbar()
            pad = 15 - len(str(i))
            fname = "vel_"+pad*"0"+str(i)+".png"
            plt.savefig(fname, bbox_inches='tight', dpi = 600)
            plt.close()

def _vel_setup(directory, velpath, inputfile, sort_int):
    '''
    initialize/read in what vel functions need to run (sans the vel_fnames)
    '''
    dpath = path.expanduser(directory)
    
    vel_fnames = {}
    for vpath in velpath:
        unsorted = glob.glob(path.join(dpath, vpath, "*.bin"))
        sorted_fnames = sorted(unsorted, key=lambda x:x[sort_int:])
        vel_fnames[vpath] = sorted_fnames


    inp_path = path.expanduser(inputfile)
    inp_block = read_input.InputBlock(inp_path)
    sim_block = inp_block.create_obj()

    return sim_block, vel_fnames, dpath


def vel_pebbles(directory, velpath, coordpath, recircpath, inputfile,
                sort_int=-19, n_skip=1,
                delimiter=' ', skiprows=9,
                n_recirc=2500000, n_dump=372366, 
                dt=2.6855e-07, recirc_hz=0.014):


    '''
    get per-pebble velocity data, scaled to real time, to use with
    pebble bed xs plotting. (note to self, the pebble class has a
    velocity attribute you use there, so you need to pass the right
    velocity data as a function of uid
    '''

    sim_block, vel_fnames, vel_dir = _vel_setup(directory,
                                                velpath,
                                                inputfile,
                                                sort_int)

    peb_d = 2*100*sim_block.r_pebble
    n_files = min([len(vel_fnames[velpath[n]]) for n in range(len(velpath))])
    
    rpath = path.expanduser(recircpath)
    unsorted_r_fnames = glob.glob(path.join(rpath, "*.bin"))
    r_fnames = sorted(unsorted_r_fnames, key=lambda x:x[sort_int:])
    P = [float(read_input.read_lammps_bin(rfile, 
                                          rpath, 
                                          skiprows=3, 
                                          max_rows=1)) for rfile in r_fnames]
    vel_all_time = {}
    for i in range(n_files):
        i_r = int((i*(n_dump))//n_recirc)
        t_scale = P[i_r]/(n_recirc*dt*recirc_hz)
        if i%n_skip==0:
            vel_map = {}
            sort_data = []
            raw_data = {}
            for vpath in velpath:
                bin_dir = path.join(vel_dir, vpath)
                bin_fname = vel_fnames[vpath][i]
                data = read_input.read_lammps_bin(bin_fname,bin_dir)

                for d in data:
                    uid = int(d[0])
                    v_z = 100*d[-1]/t_scale
                    vel_map[uid] = v_z
            
            vel_all_time[i] = vel_map
    
    return vel_all_time

def interp_coordsremakelater(directory, coordpath, recircpath,
                     sort_int=-19, delimiter=' ', skiprows=9,
                     n_recirc=2500000, n_dump=372366, 
                     dt=2.6855e-07, recirc_hz=0.014):
    '''
    '''
    P, P_uids = _recirc_data(recircpath, sort_int)
    
    recirc_rtime = [(n_recirc*i*dt)*(p/(n_recirc*dt*recirc_hz)) 
                     for i, p in enumerate(P)]

    cpath = path.expanduser(coordpath)
    unsorted_c_fnames = glob.glob(path.join(cpath, "*.bin"))
    c_fnames = sorted(unsorted_c_fnames, key=lambda x:x[sort_int:])
    n_files = len(c_fnames)

    adj_scale = P[0]/(n_recirc*dt*recirc_hz)
    adj_simtime = n_dump*dt
    adj_dt = (adj_simtime*adj_scale)//1
    for i, starting_uids in enumerate(recirc_uids[0:-1]):
        uids = starting_uids
        recirc = {}
        raw_points = {u:[] for u in uids}
        raw_rts = []
        range_start = int(i + (0.5*n_recirc)//n_dump)

        init_simt = range_start*n_dump*dt
        init_i_r = int((range_start*n_dump)//n_recirc)
        init_tscale = P[init_i_r]/(n_recirc*dt*recirc_hz)
        init_rt = init_simt*init_tscale
        for tstep in range(n_files)[range_start:]:
            if len(uids) == 0:
                break
            i_r = int((tstep*(n_dump))//n_recirc)
            t_scale = P[i_r]/(n_recirc*dt*recirc_hz)
            
            if tstep == range_start:
                reactor_time = init_rt
            else:
                reactor_time += (n_dump*dt*t_scale)
            if reactor_time < recirc_rtime[i+1]:
                pass
            elif reactor_time >= recirc_rtime[i+1]:
                raw_rts.append(reactor_time)
                data = read_input.read_lammps_bin(c_fnames[tstep], cpath)
                for d in data:
                    uid = int(d[0])
                    recirc_n = int(d[5])
                    if uid in uids:
                        if len(raw_points[uid]) == 0:
                            recirc[uid] = recirc_n
                            x = d[-3]
                            y = d[-2]
                            r = 100*(x**2+y**2)**0.5
                            z = 100*d[-1]
                            raw_points[uid] = [np.array([r, z])]
                        else:
                            if recirc_n != recirc[uid]:
                                uids.remove(uid)
                            elif recirc_n == recirc[uid]:
                                x = d[-3]
                                y = d[-2]
                                r = 100*(x**2+y**2)**0.5
                                z = 100*d[-1]
                                raw_points[uid].append(np.array([r, z]))
                                next_rt = raw_rts[-1] + (n_dump*dt*t_scale)
        points = {}
        for uid, raw_pts in raw_points.items():
            points[uid] = []
            for j, pt in enumerate(raw_pts):
                if j == 0:
                    points[uid] = [[float(pt[0]), float(pt[1])]]
                else:
                    adj_t = raw_rts[0] + j*adj_dt
                    for t_index, rt in enumerate(raw_rts):
                        if rt > adj_t:
                            rt0 = raw_rts[t_index-1]
                            rt1 = rt
                            break

                    adj_r = (points[uid][-1][0]*((rt1-adj_t)/(rt1-rt0)) + 
                              pt[0]*((adj_t-rt0)/(rt1-rt0)))
                    adj_z = (points[uid][-1][1]*((rt1-adj_t)/(rt1-rt0)) + 
                              pt[1]*((adj_t-rt0)/(rt1-rt0)))
                    points[uid].append([float(adj_r), float(adj_z)])
        pad = 7 - len(str(i))
        fname = pad*'0'+str(i)+'-transit.csv'
        fields = [uidkey for uidkey in points.keys()]
        with open(fname, "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(points.keys())
            writer.writerows(zip(*points.values()))
                

    print(adj_dt)
    return adj_dt
    

def __NOT_IMPLEMENTED_track_avalanches(directory, velpath, 
                                       coorddir, inputfile,
                                       sort_int=-19, n_skip = 1,
                                       delimiter = ' ', skiprows = 9):
    '''

    NOT IMPLEMENTED
    use velocity data to look at the KE of the pebs in the main core
    and try to find avalanches, assumes all pebbles identical, skips 1/2m
    factor in KE
    '''

    sim_block, vel_fnames, vel_dir = _vel_setup(directory,
                                                velpath,
                                                inputfile,
                                                sort_int)

    peb_d = 2*100*sim_block.r_pebble
    n_files = len(vel_fnames[velpath[0]])

    core_zmax = 100*max([core.z_max for core in sim_block.core_main.values()])
    core_zmin = 100*max([core.z_min for core in sim_block.core_main.values()])
    
    kinetic_energy = {}
    impulses = {}
    for i in range(n_files):
        if i%n_skip == 0:
            ke_map = {}
            for vpath in velpath:
                bin_dir = path.join(vel_dir, vpath)
                bin_fname = vel_fnames[vpath][i]
                data = read_input.read_lammps_bin(bin_fname,bin_dir)

                for d in data:
                    if 100*d[1] <= core_zmax and 100*d[1] >= core_zmin:
                        ke_map[d[0]] = sum((100*d[:2])**2)
            kinetic_energy[i] = ke_map

            avg_ke = sum(list(ke_map.values()))/len(ke_map.values())
            impulses[i] = {}
            for uid, ke in ke_map.items():
                if ke/avg_ke >=2.45:
                    impulses[i][uid] = {}
                    impulses[i][uid]['ke'] = ke

    for impulse in impulses.values():
        print(len(impulse))
    print()
    print()

    cdir =  path.expanduser(coorddir)
    unsorted_c_fnames = glob.glob(path.join(cdir, "*.bin"))
    c_fnames = sorted(unsorted_c_fnames, key=lambda x:x[sort_int:])

    impulse_coords = {}
    for i, c_f in enumerate(c_fnames):
        data = read_input.read_lammps_bin(c_f, cdir)
        xs = []
        ys = []
        zs = []
        for d in data:
            if d[0] in impulses[i]:
                coord = 100*d[-3:]
                impulses[i][d[0]]['coord'] = coord
                xs.append(coord[0])
                ys.append(coord[1])
                zs.append(coord[2])

        #note to self for next time: try having a reference map of coords
        #for each time step to compare to.  Then, when you notice what might 
        #be an impulse based on KE, you can add the uid/coord to the pebble at
        #the current step, i, but also at the i+/-n ones.  maybe only put a
        #list of uiuds in for each timestep (check if uid already in there
        #first), then when it comes time to plot, pull coords out based on uid

        #maybe also try to do some sort of thing where you break ke up into
        #regions, and compare against local avg_ke?  could
        #also try adding the criterion that plotted pebbles need to be within 
        # a little more than a pebble diameter of another pebble?
        
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(xs, ys, zs, marker='o')
        ax.set_xlabel('X [cm]')
        ax.set_ylabel('Y [cm]')
        ax.set_zlabel('Z [cm]')
        ax.set_zlim(54, 200)
        ax.set_xlim(-120,120)
        ax.set_ylim(-120,120)
        pad = 15 - len(str(i))
        filename = "impulse_"+pad*"0"+str(i)+".png"
        plt.savefig(filename)
        plt.close(fig)



    









    
        

        







