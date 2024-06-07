#imports
import numpy as np
from collections import defaultdict
import didymus as di
from didymus import core
from didymus import pebble
rng = np.random.default_rng()

#rods can be calculated in a more straightforward way via the euclidean
#distance (see scipy spatial) which is defined as the distance between two
#two points



def pebble_packing(active_core, pebble_radius, n_pebbles=0,n_mat_ids=0,pf=0,pf_mat_ids=0,k=10**(-3)):
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
    pebble_radius : float
        Pebble radius.  Units must match those in core
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
    assert type(active_core) == di.core.CylCore, "Only CylCore is currently supported"

    if n_pebbles != 0 and pf == 0:
        assert type(n_mat_ids) == np.ndarray,"n_mat_ids must be a numpy array with length equal to n_pebbles"
        assert len(n_mat_ids) == n_pebbles,"n_mat_ids must be a numpy array with length equal to n_pebbles"
        pf = n_to_pf(active_core, pebble_radius,n_pebbles)
        print("Equivalent packing fraction is " + str(pf))
        using_pf = False
        assert pf <= 0.6, "pf must be less than or equal to 0.6.  n_pebbles is too high."

    elif pf != 0 and n_pebbles == 0:
        assert pf_mat_ids != 0, "pf_mat_ids must be defined if using pf"
        assert type(pf_mat_ids) == dict, "pf_mat_ids must be a dictionary"
        n_pebbles = pf_to_n(active_core, pebble_radius, pf)
        print("Equivalent number of pebbles is " + str(n_pebbles))
        using_pf = True

    #find the starting coords guess
    init_coords = find_start_coords(active_core, pebble_radius, n_pebbles)

    #now we actually get into the Jodrey-Tory algorithm, with the added
    #help of Rabin-Lipton method of probalisticaly solving the
    #nearest neighbor problem

    final_coords = jt_algorithm(active_core, pebble_radius,init_coords,n_pebbles,k)

    #generate the list of didymus pebbles.  the specific method changes with
    #whether the user originally gave pf or N
    if using_pf == False:
        pebbles = [pebble.Pebble(coord, pebble_radius, n_mat_ids[i],i) for i,
         coord in enumerate(final_coords)]
    #if pf was given, we first need to determine the number of whole pebbles
    #of each provided material id (trying to stay close to what the user provided,
    #except you can only have whole pebbles
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
                pebbles.append(pebble.Pebble(final_coords[counter], pebble_radius, key ,counter))
                counter+=1


        #maybe open discussion for how you're implementing this specific part^ next meeting
    #tests:
    #because assigning coords to pebbles goes through final_coords in order, the origin
    #of a pebbles[i] should match final_coords[i]
    # pebble.pebble_radius should match pebble_radius
    #if gave n_pebs:len(pebbles) should match n_pebs
    #if gave pf: vol of pebbles/core volume should approximately be equal to pf

    return pebbles, final_coords


def pf_to_n(active_core, pebble_radius, pf):
    '''
    Converts packing fraction, pf, to number of pebbles, n_pebbles.  Uses
    floor function to round to nearest integer.

    Parameters
    ----------
    active_core : didymus CylCore object
        didymus CylCore object defining pebble-filled region of the core.
    pebble_radius : float
        Radius of a single pebble, with units matching those
        used to create center coordinates.
    pf : float
        Packing fraction, given in decimal format

    Returns
    ----------
    n_pebbles : int
        Number of pebbles in active core region

    '''
    if type(active_core) == di.core.CylCore:
        core_vol = (np.pi*(active_core.core_radius**2))*active_core.core_height
        p_vol_tot = pf*core_vol
        p_vol = (4/3)*np.pi*pebble_radius**3
        n_pebbles = int(np.floor(p_vol_tot/p_vol))
        return n_pebbles


def n_to_pf(active_core, pebble_radius, n_pebbles):
    '''
    Converts number of pebbles, n_pebbles, to packing fraction, pf

    Parameters
    ----------
    active_core : didymus CylCore object
        didymus CylCore object defining pebble-filled region of the core.
    pebble_radius : float
        Radius of a single pebble, with units matching those
        used to create center coordinates.
    n_pebbles : int
        Number of pebbles in active core region

    Returns
    ----------
    pf : float
        Packing fraction, given in decimal format
    '''
    if type(active_core) == di.core.CylCore:
        core_vol = (np.pi*(active_core.core_radius**2))*active_core.core_height
        p_vol = (4/3)*np.pi*pebble_radius**3
        p_vol_tot = p_vol*n_pebbles
        pf = p_vol_tot/core_vol
        return pf


def find_start_coords(active_core, pebble_radius, n_pebbles):
    '''
    Generates an array of starting center coordinates for pebble
    packing

    Parameters
    ----------
    active_core : didymus CylCore object
        didymus CylCore object defining pebble-filled region of the core.
    pebble_radius : float
        Radius of a single pebble, with units matching those
        used to create center coordinates.
    n_pebbles : int
        Number of pebbles in active core region

    Returns
    ----------
    coords : float
        Numpy array of length n_pebbles, where each element is the centroid
        of a pebble
    '''

    #determine dimension upper and lower bounds
    z_up = (active_core.origin[2] + 0.5*active_core.core_height - pebble_radius 
        -active_core.buff)
    z_low = (active_core.origin[2] -0.5*active_core.core_height + pebble_radius 
        + active_core.buff)
    r_up = active_core.core_radius - pebble_radius -active_core.buff

    coords = np.empty(n_pebbles,dtype=np.ndarray)
    for i in range(n_pebbles):
        f = rng.random()
        theta = rng.uniform(0,2*np.pi)
        x = active_core.origin[0] + f*r_up*np.cos(theta)
        y = active_core.origin[1] + f*r_up*np.sin(theta)
        z = rng.uniform(z_low,z_up)
        coords[i] = np.array([x,y,z])


    return coords

def jt_algorithm(active_core, pebble_radius,coords,n_pebbles,k):
    '''
    Performs the Jodrey-Tory algorithm to remove overlap from given
    pebble coordinates and return non-overlapping coordinates
    (see: DOI:https://doi.org/10.1103/PhysRevA.32.2347)

    Parameters
    ----------
    active_core : didymus CylCore object
        didymus CylCore object defining pebble-filled region of the core.
    pebble_radius : float
        Radius of a single pebble, with units matching those
        used to create center coordinates.
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
    if type(active_core) == di.core.CylCore:
        core_vol = (np.pi*(active_core.core_radius**2))*active_core.core_height
    d_out_0 = 2*np.cbrt((3*core_vol)/(4*np.pi*n_pebbles))
    d_out = d_out_0

    #step 2: probabilistic nearest neighbor search
    #to find worst overlap (shortest rod)

    #we can get the starting rod queue (and nearest neighbor)
    # with the nearest_neighbor function
    overlap = True
    i = 0
    while overlap:
        rod_queue = nearest_neighbor(active_core,pebble_radius,coords,n_pebbles)
        if not rod_queue:
            overlap = False
            break
        else:
            for rod in rod_queue:
                d_in = min(rod_queue.values())
                p1 = rod[0]
                p2 = rod[1]
                coords[p1],coords[p2] = move(active_core,
                                        pebble_radius,
                                        coords,
                                        rod,
                                        rod_queue[rod],
                                        d_out)
                if d_out < d_in:
                    print('''Outer and inner diameter converged too quickly.
                    Try again with a smaller contraction rate.''')
                    print("Maximum possible diameter with current packing:",
                         d_in)
                    overlap = False
                    break
                del_pf = (n_to_pf(active_core,d_out/2,n_pebbles)-
                        n_to_pf(active_core,d_in/2,n_pebbles))
                j = int(np.floor(-np.log10(abs(del_pf))))
                d_out = d_out - (0.5**j)*(k/n_pebbles)*d_out_0
                i += 1
        if i > 10**8:
            overlap = False
            print("Did not reach packing fraction")
            print("Maximum possible pebble diameter is currently ", d_in)

    print(i)
    return coords

def nearest_neighbor(active_core, pebble_radius, coords,n_pebbles):
    '''
    Performs Lipton-modified Rabin algorithm for the
    nearest neighbor search problem

    Parameters
    ----------
    active_core : didymus CylCore object
        didymus CylCore object defining pebble-filled region of the core.
    pebble_radius : float
        Radius of a single pebble, with units matching those
        used to create center coordinates.
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
        p1, p2 = select_pair(coords,n_pebbles)
        while (p1,p2) in init_pairs:
            p1, p2 = select_pair(coords,n_pebbles)
        #frobenius norm is default
        init_pairs[(p1,p2)] = np.linalg.norm(coords[p1]-coords[p2])
    delta = min(init_pairs.values())

    mesh_id= mesh_grid(active_core,coords,n_pebbles,delta)
    #now, for each grid square with at least one point (each element of mesh_id)
    #I make rods between each point in
    #and all points in the moores neighborhood of that square 
    #(ix+/-1, iy+/-1, iz+/-1)
    rods = {}
    for i, msqr in enumerate(mesh_id.keys()):
        #checking x index:
        x_dict = defaultdict(list)
        for sqr in list(mesh_id.keys())[i:]:
            if sqr[0]<= msqr[0]+1 and sqr[0]>=msqr[0]-1:
                x_dict[sqr] = mesh_id[sqr]

        #now that we have all the potential grid spaces
        #with an x index in range (that weren't already caught in
        #a previous pass) we can use this subset and search for applicable y
        #we know x_dict can't be empty, because it at least as the
        #central mesh grid square in it (msqr)
        y_dict = defaultdict(list)
        for xsqr in list(x_dict.keys()):
            if xsqr[1] <= msqr[1]+1 and xsqr[1] >= msqr[1]-1:
                y_dict[xsqr] = mesh_id[xsqr]

        #repeat for z, using y_dict
        neighbors = []
        for ysqr in list(x_dict.keys()):
            if ysqr[2] <= msqr[2]+1 and ysqr[2] >= msqr[2]-1:
                neighbors += mesh_id[ysqr]

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
                        rods[(p1,p2)] = np.linalg.norm(coords[p1]-coords[p2])
                    else:
                        rods[(p2,p1)] = np.linalg.norm(coords[p1]-coords[p2])
    #now, we should have the unfiltered rod list.  but we don't really 
    #need all of these
    # we can immediately drop any rod longer than the diameter of a pebble 
    #(these pebs aren't actually touching):
    pairs = list(rods.keys())
    for pair in pairs:
        if rods[pair] > 2*pebble_radius:
            del rods[pair]
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

def select_pair(coords,n_pebbles):
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


    Mx = int(np.ceil(active_core.core_radius*2/delta))
    x_min = active_core.origin[0] - active_core.core_radius
    My = int(np.ceil(active_core.core_radius*2/delta))
    y_min = active_core.origin[1] - active_core.core_radius
    Mz = int(np.ceil(active_core.core_height/delta))
    z_min = active_core.origin[2] - active_core.core_height/2
    fild_sqrs = np.empty(n_pebbles, dtype=object)
    for i, p in enumerate(coords):
        ix,iy,iz = None, None, None
        for j in range(Mx):
            if p[0] > (x_min + j*delta) and p[0] <= (x_min + (j+1)*delta):
                ix = j
                break
            else:
                if j == Mx-1:
                    if not ix:
                        ix = j
                    else:
                        pass
        for k in range(My):
            if p[1] > (y_min + k*delta) and p[1] <= (y_min + (k+1)*delta):
                iy = k
                break
            else:
                if k == My-1:
                    if not iy:
                        iy = k
                    else:
                        pass
        for l in range(Mz):
            if p[2] > (z_min + l*delta) and p[2] <= (z_min + (l+1)*delta):
                iz = l
                break
            else:
                if l == Mz-1:
                    if not iz:
                        iz = l
                    else:
                        pass
        fild_sqrs[i] = (ix,iy,iz)
    mesh_id = defaultdict(list)
    for i, v in enumerate(fild_sqrs):
        mesh_id[v].append(i)


    return mesh_id

def move(active_core,pebble_radius, coords, pair, rod, d_out):
    '''
    Moves the two points in rod an equal and opposite distance such that
    they are d_out apart

    Parameters
    ----------
    active_core : didymus CylCore object
        didymus CylCore object defining the active core region
    pebble_radius : float
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
    l = abs(d_out-rod)/2
    p1, p2 = coords[pair[0]], coords[pair[1]]
    #calculate limits once per simulation
    z_up = (active_core.origin[2] + 0.5*active_core.core_height
        - pebble_radius -active_core.buff)
    z_low = (active_core.origin[2] -0.5*active_core.core_height
        + pebble_radius +active_core.buff)
    r_up = active_core.core_radius - pebble_radius -active_core.buff
    #make unit vector function, run tests scratch that look at scipy spatial
    ux, uy, uz = (p1[0]-p2[0])/rod,(p1[1]-p2[1])/rod,(p1[2]-p2[2])/rod
    #clearer name for below
    up1p2 = np.array([ux,uy,uz])
    for i, p in enumerate([p1,p2]):
        if i ==0:
            p += up1p2*l
        else:
            p += -up1p2*l
    for p in [p1,p2]:
        p_to_center = np.linalg.norm(p[:2]-active_core.origin[:2])
        if p_to_center > r_up:
            l_out = abs(p_to_center-r_up)
            ux_p_to_center = (active_core.origin[0]-p[0])/p_to_center
            uy_p_to_center = (active_core.origin[1]-p[1])/p_to_center
            p[0] += ux_p_to_center*l_out
            p[1] += uy_p_to_center*l_out

        if p[2] > z_up:
            p[2] = z_up

        if p[2] < z_low:
            p[2] = z_low


    return p1,p2

