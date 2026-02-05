import pytest
import numpy as np
from ghastly import main
from ghastly import read_input


def test_pack_core():
    '''
    Tests the main function pack_cyl.
    '''
    test_input = read_input.InputBlock("sample_input.json")
    test_sim = test_input.create_obj()
    test_cyl = test_sim.core_main["main_cyl"]
    test_coords = main._pack_core(test_sim, 
                                  test_cyl,
                                  rough_pf=0.2, 
                                  crp = False, 
                                  openmc = False)
    stack = np.vstack(test_coords)
    rlim = test_cyl.r - test_sim.r_pebble
    zlim_min = test_cyl.zmin + test_sim.r_pebble
    zlim_max = test_cyl.zmax - test_sim.r_pebble
    assert stack.min(axis=0)[0] > test_cyl.x_c - rlim
    assert stack.min(axis=0)[1] > test_cyl.y_c - rlim
    assert stack.min(axis=0)[2] > zlim_min
    assert stack.max(axis=0)[0] < test_cyl.x_c + rlim
    assert stack.max(axis=0)[1] < test_cyl.y_c + rlim
    assert stack.max(axis=0)[2] < zlim_max


def test_find_box_bounds():
    '''
    Tests the find_box_bounds function in main.py.
    '''
    test_input = read_input.InputBlock("sample_input.json")
    test_sim = test_input.create_obj()
    bound_limits = main.find_box_bounds(test_sim)

    assert bound_limits['xb_min'] == pytest.approx(0.05 -0.5 - 5*0.03)
    assert bound_limits['yb_max'] == pytest.approx(0.5 + 5*0.03)
    assert bound_limits['zb_min'] == pytest.approx(-0.1 - 5*0.03)
    assert bound_limits['zb_max'] == pytest.approx(1.2 + 5*0.03)
