import openmc
import numpy as np
import math
from itertools import product
from matplotlib import colormaps

rng = np.random.default_rng()

# variable key:
# skirt = grey skirt
# refl = reflector
# coolch = coolant channel
# rpv = reactor pressure vessel (and core barrel, add them together)

# outlet and coolant channels span the reflector height.
# gray shirt bottom lines up with the bottom of the chute

x_c = 0
y_c = 0
active_r = 120
active_zmax = 1010
active_zmin = 54
chute_r = 24
chute_zmin = 0
chute_apex_z = (active_r*chute_zmin - chute_r*active_zmin)/(active_r - chute_r)
chute_r2 = (chute_r/(chute_zmin - chute_apex_z))**2
skirt_zmax = 1030
refl_rin = 140
refl_rout = 210
refl_zmax = 1100
refl_zmin = -134
n_coolch = 18
coolch_r = 4
coolch_R = 165
rpv_rout = 230
rpv_zmax = 1120
rpv_zmin = -154

peb_R = 3.0
fueled_R = 2.5
triso_R = [0.02125, 0.04275]
n_triso = 19000
n_pass = 6
nd_comps = ["ndep"+str(int(i+1)) for i in range(n_pass)]

###############------------------- MATERIALS ------------------###############

#this is just for testing, in the template, we will read in a materials xml we
#make beforehand
mat1 = openmc.Material(name='mat1')
mat1.add_element('C', 1.0)
mat2 = openmc.Material(name='mat2')
mat2.add_element('C', 1.0)
mat3 = openmc.Material(name='mat3')
mat3.add_element('C', 1.0)
mat4 = openmc.Material(name='mat4')
mat4.add_element('C', 1.0)
mat5 = openmc.Material(name='mat5')
mat5.add_element('C', 1.0)
mat6 = openmc.Material(name='mat6')
mat6.add_element('C', 1.0)
peb_mats = [openmc.Material(name='ndep1'),
            openmc.Material(name='ndep2'),
            openmc.Material(name='ndep3'),
            openmc.Material(name='ndep4'),
            openmc.Material(name='ndep5'),
            openmc.Material(name='ndep6')]

for pmat in peb_mats:
    pmat.add_element('C', 1.0)

dep_uco = []
for i in range(24):
    dep_name = 'dep'+str(int(i+229936))
    dep_uco.append(openmc.Material(name=dep_name))
    dep_uco[-1].add_element('C', 1.0)

mats = openmc.Materials([mat1, mat2, mat3, mat4, mat5, mat6]+peb_mats+dep_uco)
mats.export_to_xml()

# replace mats bit above this line w/ importing a preexisting material xml

matnames = np.array([mat.name for mat in mats])

#--- determine material names ---#

graph = 'mat1'
bgraph = 'mat2'
depgraph = 'mat3'
triso_layer = 'mat4'
he = 'mat5'
ss = 'mat6'

dep_uids = list(map(int,np.loadtxt('dep_pebs.csv', delimiter=',')[:,0]))
dep_names = ["dep"+str(uid) for uid in dep_uids]

#--- peripheral material indices and material colors ---#

n_dep = len(dep_names)
n_periph = 5

dep_c = np.linspace(0.10, 0.90, n_dep)
ndep_c = np.linspace(0.05, 0.75, n_pass)
periph_c = np.linspace(0.10, 1.0, n_periph)

periph_rgb = 255*colormaps['magma'](periph_c)[:, 0:-1]
nd_rgb = 255*colormaps['bone'](ndep_c)[:, 0:-1]
dep_rgb = 255*colormaps['RdPu'](dep_c)[:, 0:-1]
dpeb_color = (125, 225, 225)

i_graph = np.argmax(matnames==graph)
i_bgraph = np.argmax(matnames==bgraph)
i_depgraph = np.argmax(matnames==depgraph)
i_layer = np.argmax(matnames==triso_layer)
i_he = np.argmax(matnames==he)
i_ss = np.argmax(matnames==ss)

colors = {mats[i_graph] : tuple(map(int, periph_rgb[1])),
          mats[i_bgraph] : tuple(map(int, periph_rgb[2])),
          mats[i_depgraph] : dpeb_color,
          mats[i_layer] : tuple(map(int, periph_rgb[3])),
          mats[i_he] : tuple(map(int, periph_rgb[4])),
          mats[i_ss] : tuple(map(int, periph_rgb[0]))}

for i, nd_comp in enumerate(nd_comps):
    i_uco = np.argmax(matnames == nd_comp)
    colors[mats[i_uco]] = tuple(map(int, nd_rgb[i]))

for i, dep_name in enumerate(dep_names):
    i_uco = np.argmax(matnames == dep_name)
    colors[mats[i_uco]] = tuple(map(int, dep_rgb[i]))

###############------------------- GEOMETRY -------------------###############

#--- pebbles ---#

ndep_xyz = 100*np.loadtxt('nd_pebs.csv', delimiter=',')
dep_xyz = 100*np.loadtxt('dep_pebs.csv', delimiter=',')[:,1:]


# used by all pebbles
uco_bounds = openmc.Sphere(r=triso_R[0])
fueled_zone = openmc.Sphere(r=fueled_R)
unfueled_zone = openmc.Sphere(r=peb_R)
fueled_reg = -fueled_zone
unfueled_reg = +fueled_zone & -unfueled_zone
triso_centers = openmc.model.pack_spheres(radius=triso_R[1],
                                          region=fueled_reg,
                                          num_spheres=19000,
                                          seed = 16541846)
sphere = openmc.Cell(region=fueled_reg)
ll_peb, ur_peb = sphere.region.bounding_box
shape_peb = (2, 2, 2)
pitch_peb = (ur_peb - ll_peb)/shape_peb

# non-depleting
ndep_triso_univs = []
for nd_comp in nd_comps:
    i_uco = np.argmax(matnames == nd_comp)
    cells = [openmc.Cell(fill=mats[i_uco], region=-uco_bounds),
             openmc.Cell(fill=mats[i_layer], region=+uco_bounds)]
    ndep_triso_univs.append(openmc.Universe(cells=cells))

ndep_univs = []
for u in ndep_triso_univs:
    trisos = [openmc.model.TRISO(triso_R[1], u, c) for c in triso_centers]
    lattice = openmc.model.create_triso_lattice(trisos,
                                                ll_peb,
                                                pitch_peb,
                                                shape_peb,
                                                mats[i_graph])
    cells = [openmc.Cell(fill=lattice, region = fueled_reg),
             openmc.Cell(fill=mats[i_graph], region = unfueled_reg)]
    univ = openmc.Universe(cells=cells)
    ndep_univs.append(univ)

pebbles = []
for xyz in ndep_xyz:
    i_pass = rng.integers(0, n_pass)
    pebbles.append(openmc.model.TRISO(peb_R,
                                      ndep_univs[i_pass],
                                      xyz))

# depleting
dep_triso_univs = []
for dep_name in dep_names:
    i_uco = np.argmax(matnames == dep_name)
    cells = [openmc.Cell(fill=mats[i_uco], region=-uco_bounds),
             openmc.Cell(fill=mats[i_layer], region=+uco_bounds)]
    dep_triso_univs.append(openmc.Universe(cells=cells)) 

dep_univs = []
for u in dep_triso_univs:
    trisos = [openmc.model.TRISO(triso_R[1], u, c) for c in triso_centers]
    lattice = openmc.model.create_triso_lattice(trisos,
                                                ll_peb,
                                                pitch_peb,
                                                shape_peb,
                                                mats[i_depgraph])
    cells = [openmc.Cell(fill=lattice, region = fueled_reg),
             openmc.Cell(fill=mats[i_depgraph], region = unfueled_reg)]
    univ = openmc.Universe(cells=cells)
    dep_univs.append(univ)

for i, xyz in enumerate(dep_xyz):
    pebbles.append(openmc.model.TRISO(peb_R, dep_univs[i], xyz))

ll_active = np.array([-active_r, -active_r, refl_zmin])
ur_active = np.array([active_r, active_r, active_zmax])
shape_active = (4, 4, 8)
pitch_active = (ur_active - ll_active)/shape_active
active_lattice = openmc.model.create_triso_lattice(pebbles,
                                                   ll_active,
                                                   pitch_active,
                                                   shape_active,
                                                   mats[i_he])


#--- core periphery ---#

active_out = openmc.ZCylinder(x0=x_c, y0=y_c, r=active_r)
chute_wall = openmc.ZCone(x0=x_c, y0=y_c, z0=chute_apex_z, 
                          r2 = chute_r2)
outlet_wall = openmc.ZCylinder(x0=x_c, y0=y_c, r=chute_r)
refl_in = openmc.ZCylinder(x0=x_c, y0=y_c, r=refl_rin)
refl_out = openmc.ZCylinder(x0=x_c, y0=y_c, r=refl_rout)
rpv_out = openmc.ZCylinder(x0=x_c, y0=y_c, r=rpv_rout, 
                           boundary_type='vacuum')
active_top = openmc.ZPlane(z0=active_zmax)
active_bot = openmc.ZPlane(z0=active_zmin)
chute_bot = openmc.ZPlane(z0=chute_zmin)
skirt_top = openmc.ZPlane(z0=skirt_zmax)
refl_top = openmc.ZPlane(z0=refl_zmax)
refl_bot = openmc.ZPlane(z0=refl_zmin)
rpv_top = openmc.ZPlane(z0=rpv_zmax, boundary_type='vacuum')
rpv_bot = openmc.ZPlane(z0=rpv_zmin, boundary_type='vacuum')
coolch_rad = (2*np.pi)/n_coolch
coolch_xy = [(coolch_R*math.cos(i*coolch_rad), 
               coolch_R*math.sin(i*coolch_rad)) for i in range(n_coolch)]
coolch_voids = [openmc.ZCylinder(x0=xy[0], y0=xy[1], 
                                 r=coolch_r) 
                for i, xy in enumerate(coolch_xy)]

main_core_reg = -active_out & +active_bot & -active_top
chute_reg = -chute_wall & +chute_bot & -active_bot
outlet_reg = -outlet_wall & +refl_bot & -chute_bot
active_reg = main_core_reg | chute_reg | outlet_reg
skirt_side_reg = +active_out & -refl_in & +active_bot & -active_top
skirt_top_reg = -refl_in & +active_top & -skirt_top
skirt_bot_reg = +chute_wall & -refl_in & -active_bot & +chute_bot
refl_side_reg = +refl_in & -refl_out & +chute_bot & -skirt_top
refl_top_reg = -refl_out & +skirt_top & -refl_top
refl_bot_reg = +outlet_wall & -refl_out & -chute_bot & +refl_bot

coolch_regs = []
for i, ch in enumerate(coolch_voids):
    coolch_regs.append(-ch & +refl_bot & -refl_top)
    skirt_side_reg.append(+ch)
    skirt_top_reg.append(+ch)
    skirt_bot_reg.append(+ch)
    skirt_side_reg.append(+ch)
    refl_side_reg.append(+ch)
    refl_top_reg.append(+ch)
    refl_bot_reg.append(+ch)

skirt_reg = skirt_side_reg | skirt_top_reg | skirt_bot_reg
refl_reg = refl_side_reg | refl_top_reg | refl_bot_reg
rpv_side_reg = +refl_out & -rpv_out & +refl_bot & -refl_top
rpv_top_reg = -rpv_out & +refl_top & -rpv_top
rpv_bot_reg = -rpv_out & -refl_bot & +rpv_bot
rpv_reg = rpv_side_reg | rpv_top_reg | rpv_bot_reg

active = openmc.Cell(fill=active_lattice,
                     region=active_reg)
skirt = openmc.Cell(fill=mats[i_bgraph],
                    region=skirt_reg)
refl = openmc.Cell(fill=mats[i_graph],
                   region=refl_reg)
coolch = [openmc.Cell(fill=mats[i_he], 
                      region=ch_reg) for ch_reg in coolch_regs]
rpv = openmc.Cell(fill=mats[i_ss], 
                  region=rpv_reg)

cells = [active, skirt, refl, rpv] + coolch


###############---------------- XML and MODEL ----------------###############

universe = openmc.Universe(cells = cells)
geometry = openmc.Geometry(universe)
geometry.export_to_xml()


settings = openmc.Settings()
settings.run_mode = 'plot'
settings.export_to_xml()

xyplot = openmc.SlicePlot()
xyplot.basis='xy'
xyplot.origin = (0, 0, rpv_zmin+(rpv_zmax-rpv_zmin)/2)
xyplot.width = (500, 500)
xyplot.pixels = (1000, 1000)
xyplot.color_by = 'material'
xyplot.colors = colors

xyplotzoom = openmc.SlicePlot()
xyplotzoom.basis='xy'
xyplotzoom.origin = (0, 0, rpv_zmin+(rpv_zmax-rpv_zmin)/2)
xyplotzoom.width = (20, 20)
xyplotzoom.pixels = (2400, 2400)
xyplotzoom.color_by = 'material'
xyplotzoom.colors = colors

xyplotdep = openmc.SlicePlot()
xyplotdep.basis='xy'
xyplotdep.origin = (dep_xyz[0]+np.array([0, 0, 0.05]))
xyplotdep.width = (10, 10)
xyplotdep.pixels = (1000, 1000)
xyplotdep.color_by = 'material'
xyplotdep.colors = colors

xzplot = openmc.SlicePlot()
xzplot.basis='xz'
xzplot.origin = (0, 0, rpv_zmin+(rpv_zmax-rpv_zmin)/2)
xzplot.width = (500, 1300)
xzplot.pixels = (1000, 2600)
xzplot.color_by = 'material'
xzplot.colors = colors

plots = openmc.Plots([xyplot, xyplotzoom, xyplotdep, xzplot])
plots.export_to_xml()

openmc.run()



