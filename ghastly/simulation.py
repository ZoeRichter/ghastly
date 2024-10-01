import numpy as np
import openmc
from ghastly import core
from jinja2 import Environment, FileSystemLoader

environment = Environment(loader=FileSystemLoader("templates/"))

class Sim:
    '''
    Class for containing simulation-wide parameters and methods.
    '''

    def __init__(self, r_pebble, t_final, pf, k_rate = 0.001, 
                 down_flow=True):
        '''
        Initializes the Sim class.

        Parameters
        ----------
        r_pebble : float
            Radius of the pebbles in the simulation [m].
        t_final : float
            Total reactor-time that is being simulated [s].
        pf : float
            Target packing fraction in core region.  Determines total number
            of pebbles, but packing fraction may not match this value in
            all areas due to settling.
        down_flow : bool
            The direction of flow in the reactor core.  True means it is
            flowing downward, False means it is flowing upward.  Default is
            down.
        '''
        self.r_pebble = r_pebble
        self.pebble_volume = (4/3)*np.pi*(r_pebble**3)
        self.t_final = t_final
        self.pf = pf
        self.down_flow = down_flow

    def run_sim(self):
        '''
        Run a ghastly simulation
        '''

        pass

    def pack_cyl(self, element):
        '''
        packs a cylindrical core element
        '''
        sides = openmc.ZCylinder(r=element.r)
        top = openmc.ZPlane(z0=element.z_max)
        bottom = openmc.ZPlane(z0=element.z_min)
        region_bounds = -sides & -top & +bottom
        
        coords = openmc.model.pack_spheres(self.r_pebble, 
                                           region = region_bounds,
                                           pf = self.pf,
                                           contraction_rate = self.k_rate)

        return coords

    def pack_annulus(self,element):
        '''
        packs an annular core element
        '''
        sides = openmc.ZCylinder(r=element.r_outer)
        top = openmc.ZPlane(z0=self.z_max)
        bottom = openmc.ZPlace(z0=self.z_min)
        region_bounds = -sides & -top & +bottom

        coords = opemmc.model.pack_spheres(self.r_pebble,
                                           region = region_bounds,
                                           pf = self.pf,
                                           contraction_rate = self.k_rate)
        center = [element.x_c, element.y_c]
        coords = [list(coord) for coord in coords 
                  if (sum((coord[:2]-center)**2))**(0.5) <= element.r_inner]
        return coords

    def fake_dump_file(self, coords, dump_filename, bound_conds,
                       x_b, y_b, z_b):
        '''
        given coord array, create a "fake" dump file that can be imported into
        LAMMPS
        '''

        peb_list = [{"id":i, "x":v[0], "y":v[1], "z":v[2]} 
                    for i, v in enumerate(coords)]

        dump_template = environment.get_template("dump_template.txt")
        dump_text = dump_template.render(n_rough_atoms = len(coords),
                                         bound_conds = bound_conds,
                                         x_b = x_b,
                                         y_b = y_b
                                         z_b = z_b,
                                         peb_list = peb_list)

        with open(dump_filename, mode='w') as f:
            f.write(dump_text)

    def find_box_bounds(self, core_intake = [], core_main = [], 
                        core_outtake = []):
        '''
        given lists of core component elements, determine the appropriate size
        of the lammps bounding box
        '''

        core_list = core_intake+core_main+core_outtake
        x_list = []
        y_list = []
        z_list = []
        for element in core_list:
            z_list += [element.z_min, element.z_max]
            if type(element) == ghastly.core.CylCore:
                x_list += [-element.r, element.r]
                y_list += [-element.r, element.r]
            elif type(element) == ghastly.core.ConeCore:
                x_list += [-element.r_upper, element.r_upper, 
                           -element.r_lower, element.r_lower]
                y_list += [-element.r_upper, element.r_upper,
                           -element.r_lower, element.r_lower]
        x_b = {"low": (min(x_list) - 0.05*min(x_list)), 
               "up": (max(x_list) + 0.05*max(x_list))}

        y_b = {"low": (min(y_list) - 0.05*min(y_list)), 
               "up": (max(y_list) + 0.05*max(y_list))}

        z_b = {"low": (min(z_list) - 0.05*min(z_list)), 
               "up": (max(z_list) + 0.05*max(z_list))}

        return x_b, y_b, z_b


    def pack_core(self, core_main, core_outtake):
        '''
        initial pack for core.  openmc packs cylindrical/annular regions,
        then passes to LAMMPS to fill the rest needed.
        '''
        rough_pack = []
        core_volume = 0
        for element in core_main:
            core_volume += element.volume
            if type(element) == ghastly.core.CylCore:
                coords = self.pack_cyl(element)
                rough_pack += coords
            else:
                pass

        n_pebbles = int((self.pf*core_volume)/self.pebble_volume)

        pebbles_left = n_pebbles - len(rough_pack)

        x_b, y_b, z_b = self.find_box_bounds(core_main = core_main,
                                             core_outtake = core_outtake)


        self.fake_dump_file(rough_pack, "rough-pack.txt", "ff ff ff",
                            x_b, y_b, z_b)

        #next: variable block
        #for each variable, you'll need the name and the value
        #you will either need to escape the ${ for use in lammps templates
        # OR you will have to add a lammps-formatted name as a separate
        #entry from name
        






