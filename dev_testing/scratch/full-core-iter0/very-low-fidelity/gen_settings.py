import openmc


univ_seed = 123123123
rpv_rout = 230
rpv_zmax = 1120
rpv_zmin = -154
###############------------------- MATERIALS ------------------###############

min_temp = 600 #K
max_temp = 1200 #K
default_temp = 900 #K

###############---------------- XML and MODEL ----------------###############


shannon_mesh = openmc.RegularMesh()
shannon_mesh.lower_left = (-rpv_rout, -rpv_rout, rpv_zmin)
shannon_mesh.upper_right = (rpv_rout, rpv_rout, rpv_zmax)
shannon_mesh.dimension=(5, 5, 10)
settings = openmc.Settings()
settings.temperature={'method':'interpolation',
                      'default' : default_temp,
                      'tolerance' : 100.0,
                      'range':(min_temp, max_temp)}
settings.output = {'summary': False,
                   'tallies' : False}
settings.verbosity=7
settings.particles=(10000)
settings.generations_per_batch = 2
settings.batches = 150
settings.inactive = 20
settings.entropy_mesh = shannon_mesh
settings.seed = univ_seed
settings.export_to_xml('standard_settings.xml')

