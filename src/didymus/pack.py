#imports
import numpy as np
import math
from collections import defaultdict
import didymus as di
from didymus import core
from didymus import pebble
rng = np.random.default_rng()

def pebble_packing(core, peb_r, n_pebs=0,n_mat_ids=0,pf=0,pf_mat_ids=0,k=10**(-3)):
	'''
	Function to pack pebbles into a cylindrical core (see the CylCore
	Class) using the Jodrey-Tory method.  Users must either define
	n_pebs or pf, but not both, and describe the corresponding mat_id
	input argument (see below).  Packing function has an upper limit
	on packing fraction of 60%.
	
	Parameters
    ----------
    core : didymus CylCore object
		CylCore object defining the core shape and flow
	peb_r : float
		Pebble radius.  Units must match those in core
	n_pebs : int
		Number of pebbles to be packed in core.  Either
		n_pebs or pf must be defined, but not both
	n_mat_ids : numpy array
		numpy array containing the mat_ids of the pebbles.
		Only used if n_pebs is defined.  Length must be
		equal to n_pebs.
	pf : float
		Packing fraction of pebbles, in decimal form.  Either
		n_pebs or pf must be defined, but not both.
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
    
	'''
	
	assert n_pebs != 0 or pf != 0, "n_pebs or pf must be provided"
	assert not(n_pebs != 0 and pf != 0), "only provide one of n_pebs or pf"
	
	#add check to enforce a hard upper limit that pf <= 0.60
	assert pf <= 0.6, "pf must be less than or equal to 0.6"
	assert type(core) == di.core.CylCore, "Only CylCore is currently supported"
	
	if n_pebs != 0 and pf == 0:
		assert type(n_mat_ids) == np.ndarray,"n_mat_ids must be a numpy array with length equal to n_pebs"
		assert len(n_mat_ids) == n_pebs,"n_mat_ids must be a numpy array with length equal to n_pebs"
		pf = n_to_pf(core, peb_r,n_pebs)
		print("Equivalent packing fraction is " + str(pf))
		assert pf <= 0.6, "pf must be less than or equal to 0.6.  n_pebs is too high."
		
	elif pf != 0 and n_pebs == 0:
		assert pf_mat_ids != 0, "pf_mat_ids must be defined if using pf"
		assert type(pf_mat_ids) == dict, "pf_mat_ids must be a dictionary"
		n_pebs = pf_to_n(core, peb_r, pf)
		print("Equivalent number of pebbles is " + str(n_pebs))
		
	#after assertions, just move forward with n_pebs when you
	#find the starting coords
	init_coords = find_start_coords(core, peb_r, n_pebs)
	
	#now we actually get into the jodrey-tory algo, with the added
	#help of Rabin-Lipton method of probalisticaly solving the
	#nearest neighbor problem, so we're not brute-force searching for
	#the nearest neighbor.
	
	final_coords = jt_algo(core, peb_r,init_coords,n_pebs,k)
	
	return final_coords
	
		
def pf_to_n(core, peb_r, pf):
	'''
	Converts packing fraction to number of pebbles
	'''
	if type(core) == di.core.CylCore:
		core_vol = (np.pi*(core.core_r**2))*core.core_h
		p_vol_tot = pf*core_vol
		p_vol = (4/3)*np.pi*peb_r**3
		n_pebs = int(np.floor(p_vol_tot/p_vol))
		return n_pebs
		

def n_to_pf(core, peb_r, n_pebs):
	'''
	Converts number of pebbles to packing fraction
	'''
	if type(core) == di.core.CylCore:
		core_vol = (np.pi*(core.core_r**2))*core.core_h
		p_vol = (4/3)*np.pi*peb_r**3
		p_vol_tot = p_vol*n_pebs
		pf = p_vol_tot/core_vol
		return pf


def find_start_coords(core, peb_r, n_pebs):
	'''
	Generates an array of starting center coordinates for pebble
	packing
	'''
	
	#determine dimension upper and lower bounds
	z_up = core.origin[2] + 0.5*core.core_h - peb_r -core.buff
	z_low = core.origin[2] -0.5*core.core_h + peb_r + core.buff
	r_up = core.core_r - peb_r -core.buff
	
	coords = np.empty(n_pebs,dtype=np.ndarray)
	for i in range(n_pebs):
		f = rng.random()
		theta = rng.uniform(0,2*np.pi)
		x = core.origin[0] + f*r_up*np.cos(theta)
		y = core.origin[1] + f*r_up*np.sin(theta)
		z = rng.uniform(z_low,z_up)
		coords[i] = np.array([x,y,z])
		
		
	return coords
	
def jt_algo(core, peb_r,coords,n_pebs,k):
	'''
	Performs the Jodrey-Tory algorithm (see: ***PUT DOI HERE***)
	to remove overlap from given pebble coords and return
	updated coords
	'''
	
	#step 1: find initial d_out, which is d such that pf = 1
	if type(core) == di.core.CylCore:
		core_vol = (np.pi*(core.core_r**2))*core.core_h     
	d_out_0 = 2*np.cbrt((3*core_vol)/(4*np.pi*n_pebs))
	d_out = d_out_0
	
	#step 2: probabilistic nearest neighbor search
	#to find worst overlap (shortest rod) (god help us all)
	
	#we can get the starting rod queue (and nearest neighbor)
	# with the nearneigh function
	overlap = True
	i = 0
	while overlap:
		rod_queue = nearneigh(core,peb_r,coords)
		if not rod_queue:
			overlap = False
			break
		else:
			for rod in rod_queue:
				d_in = min(rod_queue.values())
				p1 = rod[0]
				p2 = rod[1]
				coords[p1],coords[p2] = move(core,
										peb_r,
										coords,
										rod,
										rod_queue[rod],
										d_out)
				del_pf = n_to_pf(core,d_out/2,n_pebs)-n_to_pf(core,d_in/2,n_pebs)
				if d_out < d_in:
					print('''Outer diameter and inner diameter converged too quickly.
					Try again with a smaller contraction rate.''')
					print("Maximum possible diameter with current packing:", d_in)
					overlap = False
					break
				j = math.floor(-np.log10(abs(del_pf)))
				d_out = d_out - (0.5**j)*(k/n_pebs)*d_out_0
				i += 1
		if i > 10**8:
			overlap = False
			print("Did not reach packing fraction")
			print("Maximum possible pebble diameter with current packing is ", d_in)
	
	print(i)
	return coords
	
def nearneigh(core, peb_r, coords):
	'''
	Performs Lipton-modified Rabin algorithm for the
	nearest neighbor search problem
	'''
	
	N = len(coords)
	init_pairs = {}
	for i in range(N):
		p1, p2 = selectpair(coords,N)
		while (p1,p2) in init_pairs:
			p1, p2 = selectpair(coords,N)
		#frobenius norm is default
		init_pairs[(p1,p2)] = np.linalg.norm(coords[p1]-coords[p2])
	delta = min(init_pairs.values())
	
	meshind= meshgrid(core,coords,N,delta)
	#now, for each grid square with at least one point (each element of meshind)
	#I make rods between each point in 
	#and all points in the moores neighborhood of that square (ix+/-1, iy+/-1, iz+/- 1)
	rods = {}
	for i, msqr in enumerate(meshind.keys()):
		#checking x index:
		x_dict = defaultdict(list)
		for sqr in list(meshind.keys())[i:]:
			if sqr[0]<= msqr[0]+1 and sqr[0]>=msqr[0]-1:
				x_dict[sqr] = meshind[sqr]

		#now that we have all the potential grid spaces
		#with an x index in range (that weren't already caught in
		#a previous pass) we can use this subset and search for applicable y
		#we know x_dict can't be empty, because it at least as the
		#central mesh grid square in it (msqr)
		y_dict = defaultdict(list)
		for xsqr in list(x_dict.keys()):
			if xsqr[1] <= msqr[1]+1 and xsqr[1] >= msqr[1]-1:
				y_dict[xsqr] = meshind[xsqr]

		#repeat for z, using y_dict
		neighbors = []
		for ysqr in list(x_dict.keys()):
			if ysqr[2] <= msqr[2]+1 and ysqr[2] >= msqr[2]-1:
				neighbors += meshind[ysqr]
				
		#now, the list neighbors should include all points in
		#msqr, plus all points in squares adjacent to msqr -
		#but should skip over squares that would have been included
		# in a previous neighborhood
		#go through all points, brute-force calculate all rods
		#add rods to unfltrd_rods dict, then fix at very end
		for i, p1 in enumerate(neighbors):
			if i == N-1:
				pass
			else:
				for p2 in neighbors[(i+1):]:
					if p1<p2:
						rods[(p1,p2)] = np.linalg.norm(coords[p1]-coords[p2])
					else:
						rods[(p2,p1)] = np.linalg.norm(coords[p1]-coords[p2])
	#now, we should have the unfiltered rod list.  but we don't really need all of these
	# we can immediately drop any rod longer than the diameter of a pebble (these pebs
	#aren't actually touching):
	pairs = list(rods.keys())
	for pair in pairs:
		if rods[pair] > 2*peb_r:
			del rods[pair]
	#we also only move a given point relative to exactly one other point, prioritizing
	#the worst overlap (ie, the shortest rod)
	for p in range(N):
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
		
def selectpair(coords,N):
	'''
	select random pair of points from list of coords
	'''
	
	p1 = rng.integers(0,N) #open on the upper end
	p2 = p1
	while p2 == p1:
		p2 = rng.integers(0,N)
		
	if p1 > p2:
			p1, p2 = p2, p1
	return int(p1), int(p2)
	
def meshgrid(core,coords,N,delta):
	'''
	determines what gamma lattice grid square each point in
	coords is in
	'''
	
	
	Mx = math.ceil(core.core_r*2/delta)
	x_min = core.origin[0] - core.core_r
	My = math.ceil(core.core_r*2/delta)
	y_min = core.origin[1] - core.core_r
	Mz = math.ceil(core.core_h/delta)
	z_min = core.origin[2] - core.core_h/2
	fild_sqrs = np.empty(N, dtype=object)
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
	meshind = defaultdict(list)
	for i, v in enumerate(fild_sqrs):
		meshind[v].append(i)
		
		
	return meshind

def move(core,peb_r, coords, pair, rod, d_out):
	'''
	moves the two points in rod so they are d_out apart
	'''
	l = (d_out-rod)/2
	p1 = coords[pair[0]]
	p2 = coords[pair[1]]
	ux, uy, uz = (p1[0]-p2[0])/rod,(p1[1]-p2[1])/rod,(p1[2]-p2[2])/rod
	up1p2 = np.array([ux,uy,uz])
	
	z_up = core.origin[2] + 0.5*core.core_h - peb_r -core.buff
	z_low = core.origin[2] -0.5*core.core_h + peb_r +core.buff
	r_up = core.core_r - peb_r -core.buff
	
	for i, p in enumerate(p1):
		p1[i] = p + up1p2[i]*l
		if i == 0: #x
			if abs(p1[i]) > core.origin[0]+r_up:
				theta = np.arctan(ux/uy)
				p1[i] = core.origin[0]+ r_up*np.cos(theta)
		if i == 1: #y
			if abs(p1[i]) > core.origin[1]+r_up:
				theta = np.arctan(ux/uy)
				p1[i] = core.origin[1]+ r_up*np.sin(theta)	
		else: #z
			if p1[i] > z_up:
				p1[i] = z_up
			elif p1[i] < z_low:
				p1[i] = z_low
				
	for i, p in enumerate(p2):
		p2[i] = p - up1p2[i]*l
		if i == 0: #x
			if abs(p2[i]) > r_up:
				theta = np.arctan(ux/uy)
				p2[i] = core.origin[0]-r_up*np.cos(theta)
		if i == 1: #y
			if abs(p2[i]) > r_up:
				theta = np.arctan(ux/uy)
				p2[i] = core.origin[1]-r_up*np.sin(theta)	
		else: #z
			if p2[i] > z_up:
				p2[i] = z_up
			elif p2[i] < z_low:
				p2[i] = z_low
		
	
	
	return p1,p2
	
