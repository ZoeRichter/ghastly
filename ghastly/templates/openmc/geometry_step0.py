import openmc
import openmc.deplete
import numpy as np
import math
import json

univ_seed = {{seed}}
rng = np.random.default_rng(seed=univ_seed)

# variable key:
# skirt = grey skirt
# refl = reflector
# coolch = coolant channel
# rpv = reactor pressure vessel (and core barrel, add them together)

# outlet and coolant channels span the reflector height.
# gray shirt bottom lines up with the bottom of the chute

n_peb = {{n_peb}}
pf = {{pf}}
peb_R = {{peb_R}}
peb_D = peb_R*2
fueled_R = {{fueled_R}}
triso_R = {{triso_R}}
n_triso = {{n_triso}}
triso_file = f"../trisos/{n_triso}_Ntriso_{univ_seed}.csv"
n_pass = {{n_pass}}
nd_comps = ["ndep"+str(int(i+1)) for i in range(n_pass)]

x_c = {{x_c}}
y_c = {{y_c}}
active_r = {{active_R}}
active_zmax = {{active_zmax}}
bed_zmax = (4*n_peb*peb_R**3)/(3*pf*active_r**2)
active_zmin = {{active_zmin}}
refl_rin = {{refl_Rin}}
skirt_rin = {{skirt_Rin}}
skirt_rout = {{skirt_Rout}}
refl_rout = {{refl_Rout}}
refl_zmax = {{refl_zmax}}
refl_zmin = {{refl_zmin}}
n_coolch = {{n_coolch}}
coolch_r = {{coolch_r}}
coolch_R = {{coolch_R}}
rpv_rout = {{rpv_Rout}}
rpv_zmax = {{rpv_zmax}}
rpv_zmin = {{rpv_zmin}}

zones = np.array({{zone_bounds}})*peb_D

latt_R = {{latt_R}}
latt_D = 2*latt_R

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
N_layer_a = int((bed_zmax - latt_D)/layer_z_offset[2])
N_layer_b = int((bed_zmax - layer_b0_offset[2] - latt_R)/layer_z_offset[2])
layer_a_z = ([a0[2]] + 
             [a0[2] + (i+1)*layer_z_offset[2] for i in range(N_layer_a)])
layer_b_z = ([a0[2]+layer_b0_offset[2]] + 
             [a0[2]+layer_b0_offset[2] + (i+1)*layer_z_offset[2] 
              for i in range(N_layer_b)])


peb_xyz = np.concatenate((layer_a0, layer_b0))
for i in range(N_layer_a):
    peb_xyz = np.concatenate((peb_xyz, 
                               layer_a0 + (i+1)*layer_z_offset))
for i in range(N_layer_b):
    peb_xyz = np.concatenate((peb_xyz, 
                               layer_b0 + (i+1)*layer_z_offset))
print(f"{len(peb_xyz)} pebbles in lattice.")
print(f"True pf = {(len(peb_xyz)*4/3*peb_R**3)/(bed_zmax*active_r**2)}.")

core_len = bed_zmax - active_zmin # cm
transit_time = 6*30.5 # days
peb_vel = core_len/transit_time # cm/days
layer_t = round(layer_z_offset[2]/peb_vel) # days
print(f"Pebble velocity is {peb_vel} cm/day.")
print(f"It takes approx. {layer_t} days to move layer positions.")

# select starting depleting pebbles to pull from lattice
peb_zmax = a0[2] + N_layer_a*layer_z_offset[2]
top_zone1 = []
top_zone2 = []
top_zone3 = []
for i, xyz in enumerate(peb_xyz):
    if xyz[2] == peb_zmax:
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

with open("peb_latt.csv", mode = 'w') as f:
    np.savetxt(f, peb_xyz)


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

#for i, z in enumerate(layer_a_z):
    #print(z, layer_a_z[i-int(dep_step/layer_t)])

###############------------------- MATERIALS ------------------###############

mats = openmc.Materials.from_xml({{material_file}})

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


i_graph = np.argmax(matnames==graph)
i_bgraph = np.argmax(matnames==bgraph)
i_pebgraph = np.argmax(matnames==pebgraph)
i_layer = np.argmax(matnames==triso_layer)
i_he = np.argmax(matnames==he)
i_ss = np.argmax(matnames==ss)


###############------------------- GEOMETRY -------------------###############

#--- pebbles ---# 


# used by all pebbles
uco_bounds = openmc.Sphere(r=triso_R[0])
fueled_zone = openmc.Sphere(r=fueled_R)
unfueled_zone = openmc.Sphere(r=peb_R)
fueled_reg = -fueled_zone
unfueled_reg = +fueled_zone & -unfueled_zone
triso_centers = np.loadtxt(triso_file)
sphere = openmc.Cell(region=fueled_reg)
ll_peb, ur_peb = sphere.region.bounding_box
shape_peb = {{triso_latt_shape}}
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
shape_active = {{peb_latt_shape}}
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

geometry.export_to_xml({{geometry_file}})
