import openmc
import openmc.deplete
import numpy as np
import math
import json
import sys
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

n_peb = 223000
pf_target = 0.6
peb_R = 3.0
peb_D = 2*peb_R
fueled_R = 2.5
triso_R = [0.02125, 0.04275]
n_triso = 19000
n_pass = 6
nd_comps = ["ndep"+str(int(i+1)) for i in range(n_pass)]

x_c = 0
y_c = 0
active_r = 120
active_zmax = 1010
bed_zmax = (4*n_peb*peb_R**3)/(3*pf_target*active_r**2)
active_zmin = 0
refl_rin = 120
skirt_rin = 125
skirt_rout = 126
refl_rout = 210
refl_zmax = 1100
refl_zmin = -134
n_coolch = 18
coolch_r = 4
coolch_R = 165
rpv_rout = 230
rpv_zmax = 1120
rpv_zmin = -154

zones = np.array([10, 15, 20])*peb_D

bed_sides = openmc.ZCylinder(x0=x_c, y0=y_c, r=active_r)
bed_top = openmc.ZPlane(z0=bed_zmax)
bed_bot = openmc.ZPlane(z0=active_zmin)
bed_reg = -bed_sides & -bed_top & +bed_bot
peb_xyz = openmc.model.pack_spheres(radius=peb_R,
                                     region=bed_reg,
                                     num_spheres=n_peb,
                                     seed = int(rng.integers(10000,1000000)))
peb_xyz_file = "peb_xyz_" + str(rng.integers(1000,10000)) + ".csv"
with open(peb_xyz_file, mode='w') as f:
    np.savetxt(f, peb_xyz)

peb_zmax = max([xyz[2] for xyz in peb_xyz])
top_zone1 = []
top_zone2 = []
top_zone3 = []
for i, xyz in enumerate(peb_xyz):
    if xyz[2] >= peb_zmax - 3*peb_R:
        r = (xyz[0]**2 + xyz[1]**2)**0.5
        if r <= zones[0]:
            top_zone1.append(i)
        elif r > zones[0] and r <= zones[1]:
            top_zone2.append(i)
        elif r > zones[1]:
            top_zone3.append(i)
assert len(top_zone1) > 3**(n_pass-1)
assert len(top_zone2) > 3**(n_pass-1)
assert len(top_zone3) > 3**(n_pass-1)

zone1_i = rng.choice(top_zone1, 3**(n_pass-1), replace=False)
zone2_i = rng.choice(top_zone2, 3**(n_pass-1), replace=False)
zone3_i = rng.choice(top_zone3, 3**(n_pass-1), replace=False)

dep_i = np.concatenate((zone1_i, zone2_i, zone3_i))

mask = np.ones(len(peb_xyz), dtype=bool)
mask[dep_i] = False
ndep_i = np.arange(len(peb_xyz))[mask]

dep_log = {int(i) : {'history' : "hist_",
                'i_zone' : 0} for i in dep_i}
for i in dep_i:
    xi = peb_xyz[i][0]
    yi = peb_xyz[i][1]
    zi = peb_xyz[i][2]
    r = (xi**2 + yi**2)**0.5
    if r <= zones[0]:
        dep_log[i]['history'] += '1'
        dep_log[i]['i_zone'] = 0
    elif r > zones[0] and r <= zones[1]:
        dep_log[i]['history'] += '2'
        dep_log[i]['i_zone'] = 1
    elif r > zones[1]:
        dep_log[i]['history'] += '3'
        dep_log[i]['i_zone'] = 2

with open('dep_log0.json', mode='w') as f:
    json.dump(dep_log, f, indent=4)

###############------------------- MATERIALS ------------------###############

dep_res_mats = openmc.Materials.from_xml('dep-res-mats.xml')
dep_file = 'i4-dep-res.h5'
res = openmc.deplete.Results(dep_file)
dep_t = res.get_times()

f = open('nuc_list.txt', 'r')
nuc_w_data = []
for n in f.readlines():
    nuc_w_data.append(n.strip())

nuc_in_res = list(res[0].index_nuc.keys())

nuclides = sorted(list(set(nuc_w_data) & set(nuc_in_res)))

p_bins = [(0,19),(19,26),(26,31),(31,36),(36,41),(41,None)]
p_comps = {'p01':{},
           'p12':{},
           'p23':{},
           'p34':{},
           'p45':{},
           'p56':{}}
p_t = [sum(dep_t[0:19]), 
       sum(dep_t[19:26]), 
       sum(dep_t[26:31]), 
       sum(dep_t[31:36]),
       sum(dep_t[36:41]), 
       sum(dep_t[41:])]
m_uco = dep_res_mats[0].volume*dep_res_mats[0].density

for nuc in nuclides:
    time, mass = res.get_mass('1', nuc, mass_units='g', time_units='d')
    for i, k in enumerate(list(p_comps.keys())):
        j = p_bins[i]
        m = sum(mass[j[0]:j[1]]*(time[j[0]:j[1]]/p_t[i]))/m_uco
        if m > 0:
            p_comps[k][nuc] = m
        else:
            pass

ndep1= openmc.Material(name=nd_comps[0])
ndep1.set_density('g/cm3', 10.4)
ndep1.add_components(p_comps['p01'], percent_type = 'wo')
ndep1.add_s_alpha_beta('c_Graphite')
ndep1.depletable = False
ndep1.temperature = 1088.15 #K

ndep2= openmc.Material(name=nd_comps[1])
ndep2.set_density('g/cm3', 10.4)
ndep2.add_components(p_comps['p12'], percent_type = 'wo')
ndep2.add_s_alpha_beta('c_Graphite')
ndep2.depletable = False
ndep2.temperature = 1088.15 #K

ndep3= openmc.Material(name=nd_comps[2])
ndep3.set_density('g/cm3', 10.4)
ndep3.add_components(p_comps['p23'], percent_type = 'wo')
ndep3.add_s_alpha_beta('c_Graphite')
ndep3.depletable = False
ndep3.temperature = 1088.15 #K

ndep4= openmc.Material(name=nd_comps[3])
ndep4.set_density('g/cm3', 10.4)
ndep4.add_components(p_comps['p34'], percent_type = 'wo')
ndep4.add_s_alpha_beta('c_Graphite')
ndep4.depletable = False
ndep4.temperature = 1088.15 #K

ndep5= openmc.Material(name=nd_comps[4])
ndep5.set_density('g/cm3', 10.4)
ndep5.add_components(p_comps['p45'], percent_type = 'wo')
ndep5.add_s_alpha_beta('c_Graphite')
ndep5.depletable = False
ndep5.temperature = 1088.15 #K

ndep6= openmc.Material(name=nd_comps[5])
ndep6.set_density('g/cm3', 10.4)
ndep6.add_components(p_comps['p56'], percent_type = 'wo')
ndep6.add_s_alpha_beta('c_Graphite')
ndep6.depletable = False
ndep6.temperature = 1088.15 #K

uco_vol = (3**(n_pass-1))*n_triso*(4/3)*np.pi*triso_R[0]**3
dep_mats = []
for i in range(len(zones)):
    mat_name = 'hist_' + str(i+1)
    dep_mats.append(openmc.Material(name=mat_name))
    dep_mats[-1].set_density('g/cm3', 10.4)
    dep_mats[-1].add_nuclide("U235", 0.1386, percent_type='wo')
    dep_mats[-1].add_nuclide("U238",0.7559, percent_type='wo')
    dep_mats[-1].add_element("O", 0.06025, percent_type='wo')
    dep_mats[-1].add_element('C', 0.04523, percent_type='wo')
    dep_mats[-1].add_s_alpha_beta('c_Graphite')
    dep_mats[-1].temperature = 1088.15 #K
    dep_mats[-1].volume = uco_vol
    dep_mats[-1].depletable = True

buffer = openmc.Material(name='buffer')
buffer.set_density('g/cm3', 1.05)
buffer.add_element('C', 0.9999987, percent_type='wo')
buffer.add_element('B', 1.3*10**(-6), percent_type='wo')


pyc = openmc.Material(name='PyC')
pyc.set_density('g/cm3', 1.9)
pyc.add_element('C', 0.9999987, percent_type='wo')
pyc.add_element('B', 1.3*10**(-6), percent_type='wo')


sic = openmc.Material(name='SiC')
sic.set_density('g/cm3', 3.2)
sic.add_element('C', 0.5, percent_type='ao')
sic.add_element('Si', 0.5, percent_type='ao')


layer_R = [0.02125, 0.03125, 0.03525, 0.03875, 0.04275]
layer_vol = (4/3)*np.pi*(layer_R[4]**3 - layer_R[0]**3)
buffer_vol = (4/3)*np.pi*(layer_R[1]**3 - layer_R[0]**3)
ipyc_vol = (4/3)*np.pi*(layer_R[2]**3 - layer_R[1]**3)
sic_vol = (4/3)*np.pi*(layer_R[3]**3 - layer_R[2]**3)
opyc_vol = (4/3)*np.pi*(layer_R[4]**3 - layer_R[3]**3)

triso_layer_mat = openmc.Material.mix_materials([buffer, pyc, sic],
                                                [buffer_vol/layer_vol, 
                                                 (ipyc_vol+opyc_vol)/layer_vol,
                                                 sic_vol/layer_vol],
                                                'vo')
triso_layer_mat.add_s_alpha_beta('c_Graphite')
triso_layer_mat.temperature = 1088.15 #K
triso_layer_mat.name = 'triso_layer'
triso_layer_mat.depletable = False


graphite = openmc.Material(name='graphite')
graphite.set_density('g/cm3', 1.8)
graphite.temperature = 778.15 #K
graphite.add_element('C', 0.9999985, percent_type='wo')
graphite.add_element('B', 1.5*10**(-6), percent_type='wo')
graphite.add_s_alpha_beta('c_Graphite')
graphite.depletable = False

pebgraphite = openmc.Material(name='pebgraphite')
pebgraphite.set_density('g/cm3', 1.74)
pebgraphite.temperature = 1088.15 #K 
pebgraphite.add_element('C', 0.9999987, percent_type='wo')
pebgraphite.add_element('B', 1.3*10**(-6), percent_type='wo')
pebgraphite.add_s_alpha_beta('c_Graphite')
pebgraphite.depletable = False

mixgraph = openmc.Material(name='mixgraph')
mixgraph.set_density('g/cm3', 1.8)
mixgraph.temperature = 778.15 #K
mixgraph.add_element('C', 0.9999985, percent_type='wo')
mixgraph.add_element('B', 1.5*10**(-6), percent_type='wo')

b4c = openmc.Material(name='b4c')
b4c.set_density('g/cm3', 2.2)
b4c.add_nuclide('B10', 0.1592, percent_type='ao')
b4c.add_nuclide('B11', 0.6408, percent_type='ao')
b4c.add_element('C', 0.2, percent_type='ao')

b4c_wo = 0.0004
graph_wo = 1 - b4c_wo

bgraphite = openmc.Material.mix_materials([mixgraph, b4c],
                                          [graph_wo, 
                                           b4c_wo], 'wo')

bgraphite.add_s_alpha_beta('c_Graphite')
bgraphite.temperature = 778.15 #K
bgraphite.name = 'bgraphite'
bgraphite.depletable = False


he = openmc.Material(name='He')
he.set_density('atom/b-cm', 0.0006)
he.add_element('He', 1.0, percent_type='ao')
he.temperature = 778.15 #K
he.depletable = False

ss_iron = openmc.Material(name='ss_fe')
ss_iron.add_element('Fe', 1.0, 'ao')
ss_iron.set_density('g/cm3', 7.8)
ss_iron.temperature = 513.5
ss_iron.depletable = False

mats = openmc.Materials(dep_mats + [ndep1, ndep2, ndep3, ndep4, ndep5, ndep6,
                         triso_layer_mat, graphite, pebgraphite, 
                         bgraphite, he, ss_iron])
openmc.Materials(mats).export_to_xml()



# replace mats bit above this line w/ importing a preexisting material xml

matnames = np.array([mat.name for mat in mats])

#--- determine material names ---#

graph = 'graphite'
bgraph = 'bgraphite'
pebgraph = 'pebgraphite'
triso_layer = 'triso_layer'
he = 'He'
ss = 'ss_fe'


#--- peripheral material indices and material colors ---#

n_periph = 6
n_dep = len(zones)

dep_c = np.linspace(0.65, 0.95, n_dep)
ndep_c = np.linspace(0.30, 0.80, n_pass)
periph_c = np.linspace(0.0, 0.95, n_periph)

periph_rgb = 255*colormaps['magma'](periph_c)[:, 0:-1]
nd_rgb = 255*colormaps['gray_r'](ndep_c)[:, 0:-1]
dep_rgb = 255*colormaps['rainbow_r'](dep_c)[:, 0:-1]

i_graph = np.argmax(matnames==graph)
i_bgraph = np.argmax(matnames==bgraph)
i_pebgraph = np.argmax(matnames==pebgraph)
i_layer = np.argmax(matnames==triso_layer)
i_he = np.argmax(matnames==he)
i_ss = np.argmax(matnames==ss)

colors = {mats[i_graph] : tuple(map(int, periph_rgb[1])),
          mats[i_bgraph] : tuple(map(int, periph_rgb[3])),
          mats[i_pebgraph] : tuple(map(int, periph_rgb[2])),
          mats[i_layer] : tuple(map(int, periph_rgb[4])),
          mats[i_he] : tuple(map(int, periph_rgb[5])),
          mats[i_ss] : tuple(map(int, periph_rgb[0]))}

for i, nd_comp in enumerate(nd_comps):
    i_uco = np.argmax(matnames == nd_comp)
    colors[mats[i_uco]] = tuple(map(int, nd_rgb[i]))

for i in range(len(zones)):
    uco_name = 'hist_' + str(i+1)
    i_uco = np.argmax(matnames == uco_name)
    colors[mats[i_uco]] = tuple(map(int, dep_rgb[i]))



###############------------------- GEOMETRY -------------------###############

#--- pebbles ---# 


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
                                                mats[i_pebgraph])
    cells = [openmc.Cell(fill=lattice, region = fueled_reg),
             openmc.Cell(fill=mats[i_pebgraph], region = unfueled_reg)]
    univ = openmc.Universe(cells=cells)
    ndep_univs.append(univ)

pebbles = []
for xyz in ndep_xyz:
    i_pass = rng.integers(0, n_pass)
    pebbles.append(openmc.model.TRISO(peb_R,
                                      ndep_univs[i_pass],
                                      xyz))
ndep_log = {}
pebbles = []
for i in ndep_i:
    i_pass = rng.integers(0, n_pass)
    ndep_log[int(i)] = int(i_pass)
    pebbles.append(openmc.model.TRISO(peb_R,
                                      ndep_univs[i_pass],
                                      peb_xyz[i]))
with open('ndep_log0.json', mode='w') as f:
    json.dump(ndep_log, f, indent=4)

# depleting
dep_triso_univs = []
for i in range(len(zones)):
    dep_name = 'hist_' + str(i)
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
                                                mats[i_pebgraph])
    cells = [openmc.Cell(fill=lattice, region = fueled_reg),
             openmc.Cell(fill=mats[i_pebgraph], region = unfueled_reg)]
    univ = openmc.Universe(cells=cells)
    dep_univs.append(univ)

for i, info in dep_log.items():
    pebbles.append(openmc.model.TRISO(peb_R, 
                                      dep_univs[info['i_zone']], 
                                      peb_xyz[i]))

#--- core periphery ---#

active_out = openmc.ZCylinder(x0=x_c, y0=y_c, r=active_r)
skirt_in = openmc.ZCylinder(x0=x_c, y0 = y_c, r=skirt_rin)
skirt_out = openmc.ZCylinder(x0=x_c, y0=y_c, r= skirt_rout)
refl_in = openmc.ZCylinder(x0=x_c, y0=y_c, r=refl_rin)
refl_out = openmc.ZCylinder(x0=x_c, y0=y_c, r=refl_rout)
rpv_out = openmc.ZCylinder(x0=x_c, y0=y_c, r=rpv_rout, 
                           boundary_type='vacuum')
active_top = openmc.ZPlane(z0=active_zmax)
active_bot = openmc.ZPlane(z0=active_zmin)
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

active_reg = -active_out & +active_bot & -active_top

subskirt_reg = +refl_in & -skirt_in & +active_bot & -active_top
skirt_reg = +skirt_in & -skirt_out & +active_bot & -active_top
refl_side_reg = +skirt_out & -refl_out & +active_bot & -active_top
refl_top_reg = -refl_out & +active_top & -refl_top
refl_bot_reg = -refl_out & -active_bot & +refl_bot

coolch_regs = []
for i, ch in enumerate(coolch_voids):
    coolch_regs.append(-ch & +refl_bot & -refl_top)
    refl_side_reg.append(+ch)
    refl_top_reg.append(+ch)
    refl_bot_reg.append(+ch)

refl_reg = refl_side_reg | refl_top_reg | refl_bot_reg
rpv_side_reg = +refl_out & -rpv_out & +refl_bot & -refl_top
rpv_top_reg = -rpv_out & +refl_top & -rpv_top
rpv_bot_reg = -rpv_out & -refl_bot & +rpv_bot
rpv_reg = rpv_side_reg | rpv_top_reg | rpv_bot_reg

active = openmc.Cell(region=active_reg)
ll_active, ur_active = active.region.bounding_box
shape_active = (4, 4, 6)
pitch_active = (ur_active - ll_active)/shape_active
active_lattice = openmc.model.create_triso_lattice(pebbles,
                                                   ll_active,
                                                   pitch_active,
                                                   shape_active,
                                                   mats[i_he])
active.fill = active_lattice

subskirt = openmc.Cell(fill=mats[i_graph],
                       region=subskirt_reg)
skirt = openmc.Cell(fill=mats[i_bgraph],
                    region=skirt_reg)
refl = openmc.Cell(fill=mats[i_graph],
                   region=refl_reg)
coolch = [openmc.Cell(fill=mats[i_he], 
                      region=ch_reg) for ch_reg in coolch_regs]
rpv = openmc.Cell(fill=mats[i_ss], 
                  region=rpv_reg)

cells = [active, subskirt, skirt, refl, rpv] + coolch


###############---------------- XML and MODEL ----------------###############

universe = openmc.Universe(cells = cells)
geometry = openmc.Geometry(universe)
geometry.export_to_xml()

settings = openmc.Settings()
settings.temperature={'method':'interpolation', 'range':(294, 1200)}
settings.output = {'summary': False}
settings.verbosity=7
settings.particles=(15000)
settings.generations_per_batch = 3
settings.batches = 150
settings.inactive = 25
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


xzplot = openmc.SlicePlot()
xzplot.basis='xz'
xzplot.origin = (0, 0, rpv_zmin+(rpv_zmax-rpv_zmin)/2)
xzplot.width = (500, 1300)
xzplot.pixels = (1000, 2600)
xzplot.color_by = 'material'
xzplot.colors = colors

plots = openmc.Plots([xyplot, xyplotzoom, xzplot])
plots.export_to_xml()

egroup_81_edges = np.loadtxt('81-group-HTGR.csv', skiprows=1, delimiter=' ')[:,0]
egroup_81 = openmc.mgxs.EnergyGroups(egroup_81_edges)
power = 165*(10**6) # 165 MW in Watts
timesteps = [1] # days
model = openmc.model.Model(geometry=geometry, 
                           settings=settings, 
                           materials=mats)
model.convert_to_multigroup(groups = egroup_81)
op = openmc.deplete.CoupledOperator(model)
openmc.deplete.CECMIntegrator(op, 
                              timesteps, 
                              power, 
                              timestep_units='d').integrate()



