import pytest as pyt
import didymus as ddm

def test_pebble_packing():
	'''
	Tests the pebble_packing function in pack.py
	'''
	#add optional flag in pebble packing to supress output - you don't need it
    #in tests.  in future, can make it part of simulation class, then pass sim 
    #to function
    
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
    
    core_r = 5.0
    core_h = 10.0
    pebble_r = 0.5
    test_core = ddm.core.CylCore(core_r,core_h,pebble_r)
    N = 150
    pf = 0.1
    test_coords = ddm.pack.find_start_coords(test_core,N)
    
    test_results, _ = ddm.pack.jt_algorithm(test_core, 
                                            test_coords,
                                            N, 
                                            pf, 
                                            k = 0.001,
                                            perturb_amp = 0)
                                            
    x_min = min(test_results[:,0])
    x_max = max(test_results[:,0])
    
    y_min = min(test_results[:,1])
    y_min = max(test_results[:,1])
    
    z_min = min(test_results[:,2])
    z_min = max(test_results[:,2])
    
    assert x_min >= -test_core.bounds[0] and x_max <= test_core.bounds[0]
    assert y_min >= -test_core.bounds[0] and y_max <= test_core.bounds[0]
    assert z_min >= test_core.bounds[1] and z_max <= test_core.bounds[2]
	
def test_nearest_neighbor():
	'''
	'''
	#feed nearest neightbor a selection of points where you already
    #know the answer to the worst overlap, then confirm that nearest
    #neighbor does indeed find that pair
    pebble_r = 1.0
    core_r = 5.0
    core_h = 10.0
    test_core = ddm.core.CylCore(core_r,core_h, pebble_r)
    N = 10
    test_coords = np.array([np.array([0.0,0.0,0.0]),
                            np.array([0.0,0.1,0.0]),
                            np.array([1.0,1.0,1.0]),
                            np.array([1.5,1.5,-1.5]),
                            np.array([2.0,-2.0,2.0]),
                            np.array([2.5,-2.5,-2.5]),
                            np.array([-1.0,-1.0,-1.0]),
                            np.array([-1.5,-1.5,1.5]),
                            np.array([-2.0,2.0,-2.0]),
                            np.array([-2.5,2.5,2.5])])
    worst_overlap = ddm.pack.nearest_neighbor(test_core, test_coords, N)
    
    assert worst_overlap[0] == 0
    assert worst_overlap[1] == 1
    
    
	
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
    test_core = ddm.core.CylCore(core_r,core_h, pebble_r)
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
    pebble_r = 1.0
    core_r = 5.0
    core_h = 10.0
    test_core = ddm.core.CylCore(core_r,core_h, pebble_r)
    
    #maybe make sure centroid after isn't super far off from centroid before
    
    test_coords = np.array([np.array([]),
                            np.array([]),
                            np.array([]),
                            np.array([]),
                            np.array([]),
                            np.array([]),
                            np.array([]),
                            np.array([]),
                            np.array([]),
                            np.array([])])
    centroid_before = np.sum(test_coords, axis=0)/10
    after = ddm.pack.perturb(test_core,test_coords,perturb_amp = 0.1)
    centroid_after = np.sum(after, axis=0)/10
    assert np.linalg.norm(centroid_before-centroid_after) < test_core.bounds[0]
    
    
    
def test_wrangle_pebble():
    '''
    '''
    pebble_r = 1.0
    core_r = 5.0
    core_h = 10.0
    test_core = ddm.core.CylCore(core_r,core_h, pebble_r)
    
    test_coords = np.array([np.array([0,0,0]),
                            np.array([10,0,0]),
                            np.array([-10,0,0]),
                            np.array([0,10,0]),
                            np.array([0,-10,0]),
                            np.array([0,0,10]),
                            np.array([0,0,-10]),
                            np.array([10,10,10]),
                            np.array([-10,-10,-10]))
    after = ddm.pack.wrangle_pebble(test_core,test_coords)
    
    np.testing.assert_array_equal(after[0],test_coords[0])
    assert after[1][0] == pyt.approx(test_core.bounds[0])
    assert after[2][0] == pyt.approx(-test_core.bounds[0])
    assert after[3][1] == pyt.approx(test_coords.bounds[0])
    assert after[4][1] == pyt.approx(-test_core.bounds[0])
    assert after[5][2] == pyt.approx(test_core.bounds[2])
    assert after[6][2] == pyt.approx(test_core.bounds[1])
    coord_at_45 = np.array([np.cos(np.pi/4),np.sin(np.pi/4)])*test_core.bounds[0]
    np.testing.assert_array_equal(after[7][:2],coord_at_45)
    assert after[7][2] == pyt.approx(test_core.bounds[2])
    np.testing.assert_array_equal(after[8][:2],-1*coord_at_45)
    assert after[8][2] == pyt.approx(test_core.bounds[1])
    
