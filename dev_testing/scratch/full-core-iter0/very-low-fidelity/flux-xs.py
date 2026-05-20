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
pf = 0.6
pf_hcp = 0.74
peb_R = 3.0
peb_D = peb_R*2
fueled_R = 2.5
triso_R = [0.02125, 0.04275]
n_triso = 19000
n_pass = 6
nd_comps = ["ndep"+str(int(i+1)) for i in range(n_pass)]

x_c = 0
y_c = 0
active_r = 120
active_zmax = (4*n_peb*peb_R**3)/(3*pf*active_r**2)
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

zones = [10, 15, 20]

N_latt = int((active_r-peb_R)/peb_D)
latt_R = int(100*0.5*(active_r/(N_latt+0.5)))/100
latt_R = 3.098
latt_D = 2*latt_R
print(f"Lattice pitch = {latt_D}")

a0 = latt_D*np.array([0, 0, 0.5])
b0 = latt_D*np.array([0.5, 0.5*3**0.5, 0.5])
row_x_offset = latt_D*np.array([1, 0, 0])
row_y_offset = latt_D*np.array([0, 3**0.5, 0])
layer_b0_offset = latt_D*np.array([0.5, 1/(2*3**0.5), (2/3)**0.5])
layer_z_offset = latt_D*np.array([0, 0, 3**0.5])
N_a = int((active_r-latt_R)/latt_D)+3
N_b = int((active_r-latt_D)/latt_D)+3

row_a = ([a0] + 
         [a0 + (i+1)*row_x_offset for i in range(N_a)] + 
         [a0 - (i+1)*row_x_offset for i in range(N_a)])

row_b = ([b0] + 
         [b0 + (i+1)*row_x_offset for i in range(N_b)] + 
         [b0-row_x_offset - (i)*row_x_offset for i in range(N_b+1)])
N_row_a = int((active_r - latt_R)/row_y_offset[1]) + 3
N_row_b = int((active_r - latt_R - latt_D*b0[1])/row_y_offset[1]) + 3

layer_a0 = np.concatenate((row_a, row_b, row_b - row_y_offset))
for i in range(N_row_a):
    layer_a0 = np.concatenate((layer_a0, 
                               row_a + (i+1)*row_y_offset, 
                               row_a - (i+1)*row_y_offset))
for i in range(N_row_b):
    layer_a0 = np.concatenate((layer_a0,
                               row_b + (i+1)*row_y_offset,
                               row_b-row_y_offset - (i+1)*row_y_offset))
layer_b0 = layer_a0 + layer_b0_offset
layer_a0 = [ai for ai in layer_a0 
            if ((ai[0]**2 + ai[1]**2)**0.5 + peb_R) < active_r]
layer_b0 = [bi for bi in layer_b0 
            if ((bi[0]**2 + bi[1]**2)**0.5 + peb_R) < active_r]
N_layer_a = int((active_zmax - latt_D)/layer_z_offset[2])
N_layer_b = int((active_zmax - layer_b0_offset[2] - latt_R)/layer_z_offset[2])

ndep_xyz = np.concatenate((layer_a0, layer_b0))
for i in range(N_layer_a):
    ndep_xyz = np.concatenate((ndep_xyz, 
                               layer_a0 + (i+1)*layer_z_offset))
for i in range(N_layer_b):
    ndep_xyz = np.concatenate((ndep_xyz, 
                               layer_b0 + (i+1)*layer_z_offset))
print(f"{len(ndep_xyz)} pebbles in lattice.")
print(f"True pf = {(len(ndep_xyz)*4/3*peb_R**3)/(active_zmax*active_r**2)}.")

# select starting depleting pebbles to pull from lattice
dep_z = a0[2] + N_layer_a*layer_z_offset[2]
dep_layer = [ai for ai in ndep_xyz if ai[2] == dep_z]
ndep_xyz = [ai for ai in ndep_xyz if ai[2] != dep_z]

dep_zone1 = [dep for dep in dep_layer 
             if (dep[0]**2 + dep[1]**2)**0.5 <= zones[0]*peb_D]
dep_zone2 = [dep for dep in dep_layer
             if (dep[0]**2 + dep[1]**2)**0.5 > zones[0]*peb_D 
             and (dep[0]**2 + dep[1]**2)**0.5 <= zones[1]*peb_D]
dep_zone3 = [dep for dep in dep_layer
             if (dep[0]**2 + dep[1]**2)**0.5 > zones[1]*peb_D]

i_zone1 = rng.choice(range(len(dep_zone1)), 
                     size = int(3**(n_pass-1)), replace=False)
i_zone2 = rng.choice(range(len(dep_zone2)), 
                     size = int(3**(n_pass-1)), replace=False)
i_zone3 = rng.choice(range(len(dep_zone3)), 
                     size = int(3**(n_pass-1)), replace=False)
dep_zone1 = [dep_zone1[i] for i in i_zone1]
dep_zone2 = [dep_zone2[i] for i in i_zone2]
dep_zone3 = [dep_zone3[i] for i in i_zone3]
dep_xyz = np.concatenate((dep_zone1, dep_zone2, dep_zone3))
with open("dep_latt.csv", mode = 'w') as f:
    np.savetxt(f, dep_xyz)

for ai in dep_layer:
    if np.any(np.all(dep_xyz == ai, axis = 1)):
        pass
    else:
        ndep_xyz.append(ai)
with open('ndep_latt.csv', mode='w') as f:
    np.savetxt(f, ndep_xyz)

dep_log = {i : {'history' : "hist_",
                'i_zone' : 0} for i in range(len(dep_xyz))}
for i, dep in enumerate(dep_xyz):
    r = (dep[0]**2 + dep[1]**2)**0.5
    if r <= zones[0]*peb_D:
        dep_log[i]['history'] += '1'
        dep_log[i]['i_zone'] = 0
    if r > zones[0]*peb_D and r <= zones[1]*peb_D:
        dep_log[i]['history'] += '2'
        dep_log[i]['i_zone'] = 1
    elif r > zones[1]*peb_D:
        dep_log[i]['history'] += '3'
        dep_log[i]['i_zone'] = 2

with open('dep_log.json', mode='w') as f:
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


graphite = openmc.Material(name='graphite')
graphite.set_density('g/cm3', 1.8)
graphite.temperature = 778.15 #K
graphite.add_element('C', 0.9999985, percent_type='wo')
graphite.add_element('B', 1.5*10**(-6), percent_type='wo')
graphite.add_s_alpha_beta('c_Graphite')

ndepgraphite = openmc.Material(name='ndepgraphite')
ndepgraphite.set_density('g/cm3', 1.74)
ndepgraphite.temperature = 1088.15 #K 
ndepgraphite.add_element('C', 0.9999987, percent_type='wo')
ndepgraphite.add_element('B', 1.3*10**(-6), percent_type='wo')
ndepgraphite.add_s_alpha_beta('c_Graphite')

depgraphite = openmc.Material(name='depgraphite')
depgraphite.set_density('g/cm3', 1.74)
depgraphite.temperature = 1088.15 #K 
depgraphite.add_element('C', 0.9999987, percent_type='wo')
depgraphite.add_element('B', 1.3*10**(-6), percent_type='wo')
depgraphite.add_s_alpha_beta('c_Graphite')

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

rcs_vol = 9*80*np.pi*(5**2 - 4.6**2)
skirt_vol = (active_zmax - active_zmin)*np.pi*(skirt_rout**2 - skirt_rin**2)

bgraphite = openmc.Material.mix_materials([mixgraph, b4c],
                                          [1-(rcs_vol/skirt_vol), 
                                           rcs_vol/skirt_vol], 'vo')

bgraphite.add_s_alpha_beta('c_Graphite')
bgraphite.temperature = 778.15 #K
bgraphite.name = 'bgraphite'


he = openmc.Material(name='He')
he.set_density('atom/b-cm', 0.0006)
he.add_element('He', 1.0, percent_type='ao')
he.temperature = 778.15 #K

ss_iron = openmc.Material(name='ss_fe')
ss_iron.add_element('Fe', 1.0, 'ao')
ss_iron.set_density('g/cm3', 7.8)
ss_iron.temperature = 513.5

mats = openmc.Materials(dep_mats + [ndep1, ndep2, ndep3, ndep4, ndep5, ndep6,
                         triso_layer_mat, graphite, ndepgraphite, depgraphite, 
                         bgraphite, he, ss_iron])
openmc.Materials(mats).export_to_xml()



# replace mats bit above this line w/ importing a preexisting material xml

matnames = np.array([mat.name for mat in mats])

#--- determine material names ---#

graph = 'graphite'
bgraph = 'bgraphite'
ndepgraph = 'ndepgraphite'
depgraph = 'depgraphite'
triso_layer = 'triso_layer'
he = 'He'
ss = 'ss_fe'


#--- peripheral material indices and material colors ---#

n_periph = 6
n_dep = len(zones)

dep_c = np.linspace(0.10, 0.90, n_dep)
ndep_c = np.linspace(0.05, 0.75, n_pass)
periph_c = np.linspace(0.10, 1.0, n_periph)

periph_rgb = 255*colormaps['magma'](periph_c)[:, 0:-1]
nd_rgb = 255*colormaps['bone_r'](ndep_c)[:, 0:-1]
dep_rgb = 255*colormaps['RdPu'](dep_c)[:, 0:-1]
dpeb_color = (125, 225, 225)


i_graph = np.argmax(matnames==graph)
i_bgraph = np.argmax(matnames==bgraph)
i_ndepgraph = np.argmax(matnames==ndepgraph)
i_depgraph = np.argmax(matnames==depgraph)
i_layer = np.argmax(matnames==triso_layer)
i_he = np.argmax(matnames==he)
i_ss = np.argmax(matnames==ss)

colors = {mats[i_graph] : tuple(map(int, periph_rgb[1])),
          mats[i_bgraph] : tuple(map(int, periph_rgb[3])),
          mats[i_ndepgraph] : tuple(map(int, periph_rgb[2])),
          mats[i_depgraph] : dpeb_color,
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
                                                mats[i_ndepgraph])
    cells = [openmc.Cell(fill=lattice, region = fueled_reg),
             openmc.Cell(fill=mats[i_ndepgraph], region = unfueled_reg)]
    univ = openmc.Universe(cells=cells)
    ndep_univs.append(univ)


ndep_log = []
pebbles = []
for xyz in ndep_xyz:
    i_pass = rng.integers(0, n_pass)
    ndep_log.append(i_pass)
    pebbles.append(openmc.model.TRISO(peb_R,
                                      ndep_univs[i_pass],
                                      xyz))
with open('ndep_log.csv', mode='w') as f:
    np.savetxt(f, ndep_log)
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
                                                mats[i_depgraph])
    cells = [openmc.Cell(fill=lattice, region = fueled_reg),
             openmc.Cell(fill=mats[i_depgraph], region = unfueled_reg)]
    univ = openmc.Universe(cells=cells)
    dep_univs.append(univ)

for i, info in dep_log.items():
    pebbles.append(openmc.model.TRISO(peb_R, 
                                      dep_univs[info['i_zone']], 
                                      dep_xyz[i]))

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
settings.temperature={'method':'interpolation'}
settings.verbosity=7
settings.particles=(25000)
settings.generations_per_batch = 3
settings.batches = 150
settings.inactive = 25
settings.export_to_xml()

xyplot = openmc.SlicePlot()
xyplot.basis='xy'
xyplot.origin = (0, 0, dep_z)
xyplot.width = (500, 500)
xyplot.pixels = (5000, 5000)
xyplot.color_by = 'material'
xyplot.colors = colors

xyplotzoom = openmc.SlicePlot()
xyplotzoom.basis='xy'
xyplotzoom.origin = (0, 0, dep_z)
xyplotzoom.width = (12, 12)
xyplotzoom.pixels = (1800, 1800)
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



