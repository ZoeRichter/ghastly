import openmc
import numpy as np

x_c = 0
y_c = 0
peb_R = 2.983
peb_D = 2*peb_R
fueled_R = 2.5
triso_R = 0.04275
n_triso = 19000
seed = 999999999

fueled_zone = openmc.Sphere(r=fueled_R)
fueled_reg = -fueled_zone
triso_centers = openmc.model.pack_spheres(radius=triso_R,
                                          region=fueled_reg,
                                          num_spheres=n_triso,
                                          seed=seed)

filename = str(n_triso) + '_Ntriso_' + str(seed) + '.csv'
with open(filename, mode='w') as f:
    np.savetxt(f, triso_centers)
