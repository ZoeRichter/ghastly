import pytest as pyt
import didymus as ddm

#test_core = di.core.CylCore(10.0,50.0,np.array([3.0, 6.0, 9.0]))

def test_pebble_packing():
	'''
	Tests the pebble_packing function in pack.py
	'''
	
	return
	
def test_pf_to_n():
	'''
    Tests pack.pf_to_n()
	'''
    core_h = 1
    core_r = 0.5
    pebble_r = 0.5
    pf = 2/3
    core_vol = np.pi*(core_r**2)*core_h
    pebble_vol = (4/3)*np.pi*(pebble_r**3)
    N = (pf*core_vol)/pebble_vol
    test_core = ddm.core.CylCore(core_r,core_h,pebble_r)
    assert ddm.pack.pf_to_n(test_core, pebble_r, pf) == pyt.approx(N)

def test_n_to_pf():
	'''
    Tests pack.n_to_pf()
	'''
	core_h = 1
    core_r = 0.5
    pebble_r = 0.5
    N = 1
    core_vol = np.pi*(core_r**2)*core_h
    pebble_vol = (4/3)*np.pi*(pebble_r**3)
    pf = (pebble_vol*N)/core_vol
    test_core = ddm.core.CylCore(core_r,core_h,pebble_r)
    assert ddm.pack.n_to_pf(test_core, pebble_r, N) == pyt.approx(pf)
	
def test_find_start_coords():
	'''
	'''
	#make sure length of starting coords is N
    #make sure points in core
    #currently function assumes you won't get repeats - enough of the factors
    #are randomly generated that I think that's fair
    #could, using the 7357 seed, find the first starting coord as function
    #does, and make sure it matches the first item in start_coords?
    #but if you don't reset the generator, I think it wouldn't be, so
    #find_start_coords
    #test_rng = 7357
    #find first coord same way find start coords does
    #make sure they are equal
    #^but is this a useful test
    #could you check for unifrom randomness by calculating the centroid of the
    #start coords point cloud, and making sure it is in the area of the core
    #origin?

def test_jt_algorithm():
	'''
	'''
	pass
	return
	
def test_nearest_neighbor():
	'''
	'''
	pass
	return
	
def test_select_pair():
	'''
	'''
	# check order of points (1, 2), not (2,1), 
    #check you get two points in a tuple
    #check that the point you get is among the points possible

def test_mesh_grid():
	'''
    Tests the didymus pack.mesh_grid() function
	'''
    core_r = 5.0
    core_h = 10.0
    pebble_r = 1.0
    test_core = ddm.core.CylCore(core_r,core_h,pebble_r)
    N = 10
    delta = 2.5
    test_coords = np.array([np.array([0,0,0]),
                            np.array([5,5,5]),
                            np.array([-5,-5,-5]),
                            np.array([-5,1,5]),
                            np.array([-1,3,4]),
                            np.array([-3,-1,2]),
                            np.array([-1,0,3]),
                            np.array([-1,1,3]),
                            np.array([-3,1,4]),
                            np.array([5,-3,0])])
    ans_key = {(2,2,2):[0],
            (3,3,3):[1],
            (0,0,0):[2],
            (0,2,3):[3,8],
            (1,3,3):[4],
            (0,1,2):[5],
            (1,2,3):[6,7],
            (3,0,2):[9]}
    test_mesh_grid = ddm.pack.mesh_grid(test_core,test_coords,N,delta)
    assert test_mesh_grid[(2,2,2)] == ans_key[(2,2,2)]
    assert test_mesh_grid[(3,3,3)] == ans_key[(3,3,3)]
    assert test_mesh_grid[(0,0,0)] == ans_key[(0,0,0)]
    assert test_mesh_grid[(0,2,3)][0] == ans_key[(0,2,3)][0]
    assert test_mesh_grid[(0,2,3)][1] == ans_key[(0,2,3)][1]
    assert test_mesh_grid[(1,3,3)] == ans_key[(1,3,3)]
    assert test_mesh_grid[(0,1,2)] == ans_key[(0,1,2)]
    assert test_mesh_grid[(1,2,3)][0] == ans_key[(1,2,3)][0]
    assert test_mesh_grid[(1,2,3)][1] == ans_key[(1,2,3)][1]
    assert test_mesh_grid[(3,0,2)] == ans_key[(3,0,2)]
	
def test_fix_overlap():
	'''
	'''
	#if you give a specific core, two points that you define explicitly, 
    #and a simple d_out, you should be able to confirm that the function moves
    #them apart and keeps them in the core
    #do one test to just move, do one test that you know will move them out of
    #bounds, and check that it 1) moves them back in bounds
    # 2) repeats move until it is d_out apart (if you have the pebbles line up
    #on an axis, you should be able to hand determine exactly what it will
    #do, down to the number of iterations
    
    pebble_r = 1.0
    core_r = 5.0
    core_h = 10.0
    test_core = ddm.core.CylCore(core_r,core_h, pebble_r, buff = 0.0)
    test_coords = np.array([np.array([-2,0,0]),
                            np.array([1,0,0]),
                            np.array([-4,0,0]),
                            np.array([-1,0,0]),
                            np.array([0,0,4]),
                            np.array([0,0,2]),
                            np.array([0,0,-4]),
                            np.array([0,0,-2])])
    test_pairs = [(0,1),(2,3),(4,5),(6,7)]
    d_out = 5.0
    for pair in test_pairs:
        test_coords[pair[0]],test_coords[pair[1]] = fix_overlap(test_core,
                                        test_coords,
                                        pair,
                                        d_out)
        norm = np.linalg.norm(test_coords[pair[0]]-test_coords[pair[1]])
        assert norm == pyt.approx(d_out,rel=1e-09)
        for p in pair:
            p_to_center = np.linalg.norm(test_coord[p][:2]-
                                        test_core.origin[:2])
            assert p_to_center <= test_core.bounds[0]
            assert test_coords[p][2] >= test_core.bounds[1]
            assert test_coords[p][2] <= test_core.bounds[2]
    
    
