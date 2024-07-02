#imports
import numpy as np
import math
import csv
from tqdm import tqdm
from collections import defaultdict
from didymus import core
from didymus import pebble
rng = np.random.default_rng(7357)



def pebble_packing(active_core, n_pebbles=0,n_mat_ids=0,pf=0,pf_mat_ids=0,k=10**(-2),perturb_amp=1,read_coords = False, read_file=None):
    '''
    Function to pack pebbles into a cylindrical core (see the CylCore
    Class) using the Jodrey-Tory method.  Users must either define
    n_pebbles or pf, but not both, and describe the corresponding mat_id
    input argument (see below).  Packing function has an upper limit
    on packing fraction of 60%.

    Parameters
    ----------
    active_core : didymus CylCore object
        CylCore object defining the active core shape and flow
    n_pebbles : int
        Number of pebbles to be packed in core.  Either
        n_pebbles or pf must be defined, but not both
    n_mat_ids : numpy array
        numpy array containing the mat_ids of the pebbles.
        Only used if n_pebbles is defined.  Length must be
        equal to n_pebbles.
    pf : float
        Packing fraction of pebbles, in decimal form.  Either
        n_pebbles or pf must be defined, but not both.
    pf_mat_ids : dict
        Dictionary containing mat_id:weight key:value pairs,
        to be used with pf.  Weight describes the fraction of
        total pebbles that will have the associated mat_id.  If
        the weights do not perfectly divide among the number of
        generated pebbles, the choice of mat_id favors the pebbles
        with greatest weight.
    k : float
        Contraction rate, to be used with Jodrey-Tory
        Algorithm

    Returns:
    ----------
    pebbles : list
        A list of length n_pebbles (or the equivalent number of pebbles,
        if packing fraction was provided, with each element containing an instance of
        a :class:`didymus.Pebble`

    '''

    assert n_pebbles != 0 or pf != 0, "n_pebbles or pf must be provided"
    assert not(n_pebbles != 0 and pf != 0), "only provide one of n_pebbles or pf"

    #add check to enforce a hard upper limit that pf <= 0.60
    assert pf <= 0.6, "pf must be less than or equal to 0.6"
    assert type(active_core) == core.CylCore, "Only CylCore is currently supported"

    if n_pebbles != 0 and pf == 0:
        assert type(n_mat_ids) == np.ndarray,"n_mat_ids must be a numpy array with length equal to n_pebbles"
        assert len(n_mat_ids) == n_pebbles,"n_mat_ids must be a numpy array with length equal to n_pebbles"
        pf = n_to_pf(active_core,active_core.pebble_radius,n_pebbles)
        print("Equivalent packing fraction is " + str(pf))
        using_pf = False
        assert pf <= 0.6, "pf must be less than or equal to 0.6.  n_pebbles is too high."

    elif pf != 0 and n_pebbles == 0:
        assert pf_mat_ids != 0, "pf_mat_ids must be defined if using pf"
        assert type(pf_mat_ids) == dict, "pf_mat_ids must be a dictionary"
        n_pebbles = pf_to_n(active_core, active_core.pebble_radius, pf)
        print("Equivalent number of pebbles is " + str(n_pebbles))
        using_pf = True

    #find the starting coords
    if read_coords:
        init_coords = np.loadtxt(read_file)
    else:
        init_coords = find_start_coords(active_core, n_pebbles)

    #now we actually get into the Jodrey-Tory algorithm, with the added
    #help of Rabin-Lipton method of probalisticaly solving the
    #nearest neighbor problem

    final_coords,output_list = jt_algorithm(active_core,
                                init_coords,
                                n_pebbles,
                                pf,
                                k,
                                perturb_amp)

    #generate the list of didymus pebbles.  the specific method changes with
    #whether the user originally gave pf or N
    if using_pf == False:
        pebbles = [pebble.Pebble(coord, active_core.pebble_radius, n_mat_ids[i],i) for i,
         coord in enumerate(final_coords)]
    #if pf was given, we first need to determine the number of whole pebbles
    #of each provided material id (trying to stay close to what the user provided,
    #except you can only have whole pebbles)
    else:
        pebble_split = {}
        normalize = sum(f for f in pf_mat_ids.values())
        for i, key in enumerate(pf_mat_ids):
            if i<len(pf_mat_ids)-1:
                n = np.round(n_pebbles*(pf_mat_ids[key]/normalize))
                pebble_split[key] = [pf_mat_ids[key], int(n)]
            else:
                current_total = sum(p[1] for p in pebble_split.values())
                pebble_split[i] = [pf_mat_ids[key],int(n_pebbles-current_total)]
        #then, once you know the pebble split, actually generate the pebbles
        pebbles =[]
        counter = 0
        for key in pebble_split:
            for _ in range(pebble_split[key][1]):
                pebbles.append(pebble.Pebble(final_coords[counter], active_core.pebble_radius, key ,counter))
                counter+=1
    #replace this later with something better - preferably an output class
    output_name = str(k)+"_"+str(perturb_amp)+"_output.csv"

    with open(output_name, 'w', newline='') as csvfile:
        fieldnames = ['i', 'd_out','d_in_i','d_in_avg']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for row in output_list:
            writer.writerow(row)


    return pebbles, final_coords


def pf_to_n(active_core,radius, pf):
    '''
    Converts packing fraction, pf, to number of pebbles, n_pebbles.  Uses
    floor function to round to nearest integer.

    Parameters
    ----------
    active_core : didymus CylCore object
        didymus CylCore object defining pebble-filled region of the core.
    pf : float
        Packing fraction, given in decimal format

    Returns
    ----------
    n_pebbles : int
        Number of pebbles in active core region

    '''
    if type(active_core) == core.CylCore:
        core_vol = (np.pi*(active_core.core_radius**2))*active_core.core_height
        p_vol_tot = pf*core_vol
        p_vol = (4/3)*np.pi*radius**3
        n_pebbles = int(np.floor(p_vol_tot/p_vol))
        return n_pebbles


def n_to_pf(active_core,radius, n_pebbles):
    '''
    Converts number of pebbles, n_pebbles, to packing fraction, pf

    Parameters
    ----------
    active_core : didymus CylCore object
        didymus CylCore object defining pebble-filled region of the core.
    n_pebbles : int
        Number of pebbles in active core region

    Returns
    ----------
    pf : float
        Packing fraction, given in decimal format
    '''
    if type(active_core) == core.CylCore:
        core_vol = (np.pi*(active_core.core_radius**2))*active_core.core_height
        p_vol = (4/3)*np.pi*radius**3
        p_vol_tot = p_vol*n_pebbles
        pf = p_vol_tot/core_vol
        return pf


def find_start_coords(active_core, n_pebbles):
    '''
    Generates an array of starting center coordinates for pebble
    packing

    Parameters
    ----------
    active_core : didymus CylCore object
        didymus CylCore object defining pebble-filled region of the core.
    bounds : numpy array
        Upper and lower bounds on possible pebble center coordinates within the
        active core, in the form of [maximum radius, minimum Z, maximum Z]
    n_pebbles : int
        Number of pebbles in active core region

    Returns
    ----------
    coords : float
        Numpy array of length n_pebbles, where each element is the centroid
        of a pebble
    '''

    coords = np.empty(n_pebbles,dtype=np.ndarray)
    for i in range(n_pebbles):
        f = rng.random()
        theta = rng.uniform(0,2*np.pi)
        x = active_core.origin[0] + f*active_core.bounds[0]*np.cos(theta)
        y = active_core.origin[1] + f*active_core.bounds[0]*np.sin(theta)
        z = rng.uniform(active_core.bounds[1],active_core.bounds[2])
        coords[i] = np.array([x,y,z])


    return coords

def jt_algorithm(active_core,coords,n_pebbles,pf,k,perturb_amp):
    '''
    Performs the Jodrey-Tory algorithm to remove overlap from given
    pebble coordinates and return non-overlapping coordinates
    (see: DOI:https://doi.org/10.1103/PhysRevA.32.2347)

    Parameters
    ----------
    active_core : didymus CylCore object
        didymus CylCore object defining pebble-filled region of the core.
    bounds : numpy array
        Upper and lower bounds on possible pebble center coordinates within the
        active core, in the form of [maximum radius, minimum Z, maximum Z]
    coords : numpy array
        Numpy array of length n_pebbles, where each element is the centroid
        of a pebble.  This is pre-Jodrey-Tory.
    n_pebbles : int
        Number of pebbles in active core region
    k : float
        Contraction rate, used to determine the rate at which the outer,
        or nominal, diameter decreases in each iteration of the Jodrey-Tory
        algorithm.

    Returns
    ----------
    coords : float
        Numpy array of length n_pebbles, where each element is the centroid
        of a pebble.  This is post-Jodrey-Tory.
    '''

    #step 1: find initial d_out, which is d such that pf = 1
    #core_vol should be a class attribute, with how often you use it
    if type(active_core) == core.CylCore:
        core_vol = (np.pi*(active_core.core_radius**2))*active_core.core_height
    d_out_0 = 2*np.cbrt((3*core_vol)/(4*np.pi*n_pebbles))
    d_out = d_out_0
    sum_d_in=0
    sum_i = 0

    #step 2: probabilistic nearest neighbor search
    #to find worst overlap (shortest rod)
    rod_queue =  nearest_neighbor(active_core,coords,n_pebbles)
    d_in_key = min(rod_queue, key = rod_queue.get)
    d_in = rod_queue[d_in_key]
    counter = 0
    d_in_last = 0.0
    output_list = []
    max_attempts = 10**3
    for i in tqdm(range(max_attempts), miniters = (max_attempts/100)):
        for i in range(len(rod_queue)):
            if not rod_queue:
                break
            else:
                rod = min(rod_queue, key = rod_queue.get)
                coords[rod[0]],coords[rod[1]] = fix_overlap(active_core,
                                                coords,
                                                rod,
                                                d_out)
                del rod_queue[rod]
        if d_in < d_in_last:
            counter +=1
            if counter == 2:
                    coords = perturb(active_core,coords,perturb_amp)

        rod_queue =  nearest_neighbor(active_core,coords,n_pebbles)

        if i%100==0:
            avg_d_in = sum_d_in/sum_i
            ith_dict = {'i':i, 'd_out':d_out,'d_in_i':d_in,'d_in_avg':avg_d_in}
            output_list.append(ith_dict)
            sum_d_in = 0
            sum_i = 0
        if not rod_queue:
            print("did it!")
            break
        else:
            d_in_key = min(rod_queue, key = rod_queue.get)
            d_in = rod_queue[d_in_key]
            sum_d_in+=d_in
            sum_i+=1
            if d_out <= 2*active_core.pebble_radius:
                d_out = 2*active_core.pebble_radius
            else:
                del_pf = abs(n_to_pf(active_core,d_out/2,n_pebbles)-
                    n_to_pf(active_core,d_in/2,n_pebbles))
                j = int(np.floor(-np.log10(del_pf)))
                d_out = d_out - (0.5**j)*(k/n_pebbles)*d_out_0
            
        if i == (max_attempts - 1):
            print("Did not reach packing fraction")
            print("Maximum possible pebble diameter is currently ", d_in)
            np.savetxt("restart.csv", coords)
            
        d_in_last = d_in
    return coords, output_list

def nearest_neighbor(active_core, coords,n_pebbles):
    '''
    Performs Lipton-modified Rabin algorithm for the
    nearest neighbor search problem

    Parameters
    ----------
    active_core : didymus CylCore object
        didymus CylCore object defining pebble-filled region of the core.
    coords : numpy array
        Numpy array of length n_pebbles, where each element is the centroid
        of a pebble.
    n_pebbles : int
        Number of pebbles in active core region

    Returns
    ----------
    rods : dict
        Dictionary with key:value pairs containing information on overlapping
        rods, to use with the Jodrey-Tory algorthim.  Each key is 2-element tuple
        in which each value is the index of the corresponding point in coords.
        The value is the distance (the length of the rod) between the two points
        defined by the key.
    '''
    init_pairs = {}
    for i in range(n_pebbles):
        p1, p2 = select_pair(n_pebbles)
        while (p1,p2) in init_pairs:
            p1, p2 = select_pair(n_pebbles)
        #frobenius norm is default
        init_pairs[(p1,p2)] = (sum((coords[p1]-coords[p2])**2))**(0.5)
    delta = min(init_pairs.values())

    mesh_id= mesh_grid(active_core,coords,n_pebbles,delta)
    #now, for each grid square with at least one point (each element of mesh_id)
    #I make rods between each point in
    #and all points in the moores neighborhood of that square
    #(ix+/-1, iy+/-1, iz+/-1)
    rods = {}
    for i, msqr in enumerate(mesh_id.keys()):
        neighbors = []
        for sqr in list(mesh_id.keys())[i:]:
            if sqr[0]<= msqr[0]+1 and sqr[0]>=msqr[0]-1:
                if sqr[1] <= msqr[1]+1 and sqr[1] >= msqr[1]-1:
                    if sqr[2] <= msqr[2]+1 and sqr[2] >= msqr[2]-1:
                        neighbors += mesh_id[sqr]
        
        #now, the list neighbors should include all points in
        #msqr, plus all points in squares adjacent to msqr -
        #but should skip over squares that would have been included
        # in a previous neighborhood
        #go through all points, brute-force calculate all rods
        #add rods to rods dict, then filter at very end
        for i, p1 in enumerate(neighbors):
            if i == n_pebbles-1:
                pass
            else:
                for p2 in neighbors[(i+1):]:
                    if p1<p2:
                        rods[(p1,p2)] = (sum((coords[p1]-coords[p2])**2))**(0.5)
                    else:
                        rods[(p2,p1)] = (sum((coords[p1]-coords[p2])**2))**(0.5)
    #now, we should have the unfiltered rod list.  but we don't really
    #need all of these
    # we can immediately drop any rod longer than the diameter of a pebble
    #(these pebs aren't actually touching):
    rods = {k:v for (k,v) in rods.items() if v < 2*active_core.pebble_radius}
    #we also only move a given point relative to exactly one other point,
    #prioritizing the worst overlap (ie, the shortest rod)
    for p in range(n_pebbles):
        temp={}
        pairs = list(rods.keys())
        for pair in pairs:
            if pair[0] ==  p or pair[1] == p:
                temp[pair] = rods[pair]
        if not temp:
            pass
        else:
            temp_keys = list(temp.keys())
            for tkey in temp_keys:
                if temp[tkey] != min(temp.values()):
                    del rods[tkey]
    return rods

def select_pair(n_pebbles):
    '''
    select random pair of points from list of coords

    Parameters
    ----------
    coords : numpy array
        Numpy array of length n_pebbles, where each element is the centroid
        of a pebble.  This is pre-Jodrey-Tory.
    n_pebbles : int
        Total number of pebbles.

    Returns
    ----------
    p1, p2 : int
        Integers corresponding to the index of a point in coords,
        where p1 < p2.
    '''
    p1 = rng.integers(0,n_pebbles) #open on the upper end
    p2 = p1
    while p2 == p1:
        p2 = rng.integers(0,n_pebbles)

    if p1 > p2:
            p1, p2 = p2, p1
    return int(p1), int(p2)

def mesh_grid(active_core,coords,n_pebbles,delta):
    '''
    Determines what grid square in the Gamma lattice (from jt-algorithm)
    each point in the given coordinates is in.
    Parameters
    ----------
    active_core : didymus CylCore object
        didymus CylCore object defining pebble-filled region of the core.
    coords : numpy array
        Numpy array of length n_pebbles, where each element is the centroid
        of a pebble.  This is pre-Jodrey-Tory.
    n_pebbles : int
        Total number of pebbles
    delta : float
        From Rabin-Lipton nearest neighbor algorithm.  Delta is defined as
        the smallest distance between any of the initial, randomly-sampled
        pairs, and defines the side-length of the mesh grid square in
        a lattice (Gamma) encompassing the active core region.

    Returns
    ----------
    mesh_index : dict
        Dictionary containg key:value pairs in which each key is a 3-element
        tuple containing the (ix, iy, iz) code for a grid square in the gamma
        lattice, and the value is a list containing the specific points from
        coords that lie inside that grid square.

    '''


    Mx = int(np.ceil(active_core.core_radius*2/delta) - 1)
    x_min = active_core.origin[0] - active_core.core_radius
    My = int(np.ceil(active_core.core_radius*2/delta) - 1)
    y_min = active_core.origin[1] - active_core.core_radius
    Mz = int(np.ceil(active_core.core_height/delta) -1)
    z_min = active_core.origin[2] - active_core.core_height/2
    fild_sqrs = np.empty(n_pebbles, dtype=object)
    for i, p in enumerate(coords):
        ix= int(np.floor((p[0]-x_min)/delta))
        if ix == Mx+1:
            ix=Mx
        iy= int(np.floor((p[1]-y_min)/delta))
        if iy == My+1:
            iy=My
        iz= int(np.floor((p[2]-z_min)/delta))
        if iz==Mz+1:
            iz=Mz
        fild_sqrs[i] = (ix,iy,iz)
    mesh_id = defaultdict(list)
    for i, v in enumerate(fild_sqrs):
        mesh_id[v].append(i)


    return mesh_id
        
    

def fix_overlap(active_core, coords, pair, d_out):
    '''
    Moves the two points in rod an equal and opposite distance such that
    they are d_out apart

    Parameters
    ----------
    active_core : didymus CylCore object
        didymus CylCore object defining the active core region
    bounds : float
        Pebble radius, in units matching those in the core definition
    coords : numpy array
        Numpy array of length n_pebbles, where each element is the centroid
        of a pebble.  This is pre-Jodrey-Tory.
    pair : tuple
        A 2-element tuple made of integers, where each value corresponds
        to a point in coords
    rod :
    n_pebbles : int
        Total number of pebbles.

    Returns
    ----------
    p1, p2 : int
        Integers corresponding to the index of a point in coords,
        where p1 < p2.
    '''
    
    not_apart = True
    j = 0
    p1, p2 = coords[pair[0]], coords[pair[1]]
    
    while not_apart:
        normp1p2 = (sum((p1-p2)**2))**(0.5)
        up1p2 = (p1-p2)/normp1p2
        l = (d_out-normp1p2)/2
        
        p1, p2 = p1+up1p2*l,p2-up1p2*l
        
        p1,p2 = pebble_bounds(active_core,p1), pebble_bounds(active_core,p2)
                
        normp1p2 = (sum((p1-p2)**2))**(0.5)
        if math.isclose(normp1p2,d_out) or normp1p2>d_out:
            not_apart = False
            
        if j>100:
            print("still not apart")
            not_apart = False
        j+=1


    return p1,p2

def perturb(active_core,coords,perturb_amp):
    '''
    randomly perturbs pebble centers
    
    Parameters
    ----------
    active_core : didymus CylCore object
        didymus CylCore object defining the active core region
    coords : numpy array
        Numpy array of length n_pebbles, where each element is the centroid
        of a pebble.  This is pre-perturbation.
    perturb_amp : float
        Fraction of pebble radius a pebble centroid will move when perturbed

    Returns
    ----------
    coords : numpy array
        Numpy array of length n_pebbles, where each element is the centroid
        of a pebble.  This is post-perturbation.
    '''
    l = active_core.pebble_radius*perturb_amp
    for p in coords:
        #ux = rng.choice([-1,1])*rng.random()
        #uy = rng.choice([-1,1])*rng.random()
        #uz = rng.choice([-1,1])*rng.random()
        #unorm = np.linalg.norm([ux,uy,uz])
        unit_dir = rng.choice([-1,1],3)*rng.random(3)
        u_norm = (sum((unit_dir)**2))**(0.5)
        uvector = np.array([ux/unorm,uy/unorm,uz/unorm])
        p += uvector*l
    
    for p in coords:
        p = pebble_bounds(active_core,p)
                
            
    return coords
    
def pebble_bounds(active_core,p):
    '''
    checks if a pebble is in core boundaries, and moves it inside if not
    
    Parameters
    ----------
    active_core : didymus CylCore object
        didymus CylCore object defining the active core region
    p : numpy array
        Three-element numpy array giving the x,y, and z coordinates of
        the centroid of a pebble. This is pre-enforcing core boundaries.

    Returns
    ----------
    p : numpy array
        Three-element numpy array giving the x,y, and z coordinates of
        the centroid of a pebble.  This is post-enforcing core boundaries.
    '''
    p_to_origin = (sum((p[:2]-active_core.origin[:2])**2))**(0.5)
    if p_to_origin > active_core.bounds[0]:
        l_out = p_to_origin-active_core.bounds[0]
        unit_po = (active_core.origin[:2]-p[:2])/p_to_origin
        p[:2] += unit_po*l_out

    if p[2] > active_core.bounds[2]:
        p[2] = active_core.bounds[2]
            
    if p[2] < active_core.bounds[1]:
        p[2] = active_core.bounds[1]
        
    return p
    

