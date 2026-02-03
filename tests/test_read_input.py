import pytest
from ghastly import read_input
from ghastly import simulation
from ghastly import core


def test_InputBlock():
    '''
    Test InputBlock class.
    '''
    test_input = read_input.InputBlock("sample_input.json")

    assert test_input.sim_var["r_pebble"] == 0.03
    assert test_input.core_inlet_var["inlet_cone"]["zmax"] == 1.1
    assert test_input.core_main_var["main_cyl"]["type"] == "cylinder"
    assert test_input.core_outlet_var["outlet_cyl"]["r"] == 0.1
    assert test_input.lammps_var["density_rho"] == 1700


def test_create_obj():
    '''
    Test InputBlock's create_obj method.
    '''
    test_input = read_input.InputBlock("sample_input.json")
    test_simblock = test_input.create_obj()

    assert type(test_simblock) == simulation.Sim


def test_create_core_zone():
    '''
    Test InputBlock's create_core_zone method.
    '''
    test_input = read_input.InputBlock("sample_input.json")
    test_main = test_input.create_core_zone(test_input.core_main_var)
    assert len(test_main) == 2
    assert type(test_main) == dict
    assert type(test_main["main_cyl"]) == core.CylCore
    assert test_main["main_cyl"].r == 0.5
    assert test_main["main_cyl"].zmax == 1.0
    assert type(test_main["main_cone"]) == core.ConeCore
    assert test_main["main_cone"].r_minor == 0.1
    assert test_main["main_cone"].x_c == 0.05


def create_sim_block():
    '''
    Tests InputBlock's create_sim_block method.
    '''

    test_input = read_input.InputBlock("sample_input.json")
    test_inlet = test_input.create_core_zone(test_input.core_inlet_var)
    test_main = test_input.create_core_zone(test_input.core_main_var)
    test_outlet = test_input.create_core_zone(test_input.core_outlet_var)
    test_sim_block = test_input.create_cim_block(test_inlet,
                                                 test_main,
                                                 test_outlet)
    assert test_sim_block.pf == 0.60
    assert test_sim_block.r_pebble == 0.03
