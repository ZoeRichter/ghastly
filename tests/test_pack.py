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
    
    core_r = 5.0
    core_h = 20.0
    pebble_r = 0.5
    test_core = ddm.core.CylCore(core_r,core_h,pebble_r)
    N = 1500
    test_coords = ddm.pack.find_start_coords(test_core,N)
    for p in test_coords:
        p_radius = np.linalg.norm(p[:2])
        assert (p_radius < test_core.bounds[0] or
                p_radius == pyt.approx(test_core.bounds[0]))
        assert p[2] >= test_core.bounds[1] and p[2] <= test_core.bounds[2]
    
    avg = np.sum(test_coords, axis=0)/N
    assert (np.linalg.norm(avg) < pebble_r or
            np.linalg.norm(avg) == pyt.approx(pebble_r))

def test_jt_algorithm():
	'''
	'''
    #could test looowwww packing fraction, confirm things like pebbles
    #being in the core, d_in being at least 2*pebble_radius,
    
	pass
	return
	
def test_nearest_neighbor():
	'''
	'''
	#feed nearest neightbor a selection of points where you already
    #know the answer to the worst overlap, then confirm that nearest
    #neighbor does indeed find that pair
	
def test_select_pair():
	'''
	'''
    
    N = 10
    for i in range(N):
        p1,p2 = ddm.pack.select_pair(N)
        assert p1 < p2
        assert p1 >= 0 and p1 < N
        assert p2 >= 0 and p2 < N
    

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
    
def test_perturb():
    '''
    '''
    
    
    
    
def test_wrangle_pebble():
    '''
    '''
