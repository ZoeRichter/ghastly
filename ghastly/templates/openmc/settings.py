import openmc
import numpy as np

univ_seed = {{seed}}
rng = np.random.default_rng(seed=univ_seed)


rpv_rout = {{rpv_Rout}}
rpv_zmax = {{rpv_zmax}}
rpv_zmin = {{rpv_zmin}}

###############------------------- MATERIALS ------------------###############

min_temp = {{min_temp}} #K
max_temp = {{max_temp}} #K
default_temp = {{default_temp}} #K


shannon_mesh = openmc.RegularMesh()
shannon_mesh.lower_left = (-rpv_rout, -rpv_rout, rpv_zmin)
shannon_mesh.upper_right = (rpv_rout, rpv_rout, rpv_zmax)
shannon_mesh.dimension= {{shannon_mesh_shape}}

settings = openmc.Settings()
settings.temperature={'method':{{temp_method}},
                      'default' : default_temp,
                      'tolerance' : {{temp_tolerance}},
                      'range':(min_temp, max_temp)}
settings.output = {'tallies' : False}
settings.verbosity={{openmc_verbosity}}
settings.particles=({{n_particles}})
settings.generations_per_batch = {{n_gen_per_batch}}
settings.batches = {{n_batches}}
settings.inactive = {{n_inactive}}
settings.entropy_mesh = shannon_mesh
settings.seed = univ_seed
settings.export_to_xml({{settings_file}})

