import openmc
import numpy as np
import math

univ_seed = 123123123
rng = np.random.default_rng(seed=univ_seed)

n_pass = 6
nd_comps = ["ndep"+str(int(i+1)) for i in range(n_pass)]



###############------------------- MATERIALS ------------------###############

fuel_temp = 1088.15 #K
periph_temp = 778.15 #K
min_temp = 600 #K
max_temp = 1200 #K
default_temp = 900 #K

dep_res_mats = openmc.Materials.from_xml('BCC-mats.xml')
dep_file = 'BCC-dep-res.h5'
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
ndep1.temperature = fuel_temp #K

ndep2= openmc.Material(name=nd_comps[1])
ndep2.set_density('g/cm3', 10.4)
ndep2.add_components(p_comps['p12'], percent_type = 'wo')
ndep2.add_s_alpha_beta('c_Graphite')
ndep2.depletable = False
ndep2.temperature = fuel_temp #K

ndep3= openmc.Material(name=nd_comps[2])
ndep3.set_density('g/cm3', 10.4)
ndep3.add_components(p_comps['p23'], percent_type = 'wo')
ndep3.add_s_alpha_beta('c_Graphite')
ndep3.depletable = False
ndep3.temperature = fuel_temp #K

ndep4= openmc.Material(name=nd_comps[3])
ndep4.set_density('g/cm3', 10.4)
ndep4.add_components(p_comps['p34'], percent_type = 'wo')
ndep4.add_s_alpha_beta('c_Graphite')
ndep4.depletable = False
ndep4.temperature = fuel_temp #K

ndep5= openmc.Material(name=nd_comps[4])
ndep5.set_density('g/cm3', 10.4)
ndep5.add_components(p_comps['p45'], percent_type = 'wo')
ndep5.add_s_alpha_beta('c_Graphite')
ndep5.depletable = False
ndep5.temperature = fuel_temp #K

ndep6= openmc.Material(name=nd_comps[5])
ndep6.set_density('g/cm3', 10.4)
ndep6.add_components(p_comps['p56'], percent_type = 'wo')
ndep6.add_s_alpha_beta('c_Graphite')
ndep6.depletable = False
ndep6.temperature = fuel_temp #K

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
triso_layer_mat.temperature = fuel_temp
triso_layer_mat.name = 'triso_layer'

pebgraphite = openmc.Material(name='pebgraphite')
pebgraphite.set_density('g/cm3', 1.74)
pebgraphite.temperature = fuel_temp
pebgraphite.add_element('C', 0.9999987, percent_type='wo')
pebgraphite.add_element('B', 1.3*10**(-6), percent_type='wo')
pebgraphite.add_s_alpha_beta('c_Graphite')

graphite = openmc.Material(name='graphite')
graphite.set_density('g/cm3', 1.8)
graphite.temperature = periph_temp
graphite.add_element('C', 0.9999985, percent_type='wo')
graphite.add_element('B', 1.5*10**(-6), percent_type='wo')
graphite.add_s_alpha_beta('c_Graphite')

mixgraph = openmc.Material(name='mixgraph')
mixgraph.set_density('g/cm3', 1.8)
mixgraph.temperature = periph_temp
mixgraph.add_element('C', 0.9999985, percent_type='wo')
mixgraph.add_element('B', 1.5*10**(-6), percent_type='wo')

b4c = openmc.Material(name='b4c')
b4c.set_density('g/cm3', 2.2)
b4c.add_nuclide('B10', 0.1592, percent_type='ao')
b4c.add_nuclide('B11', 0.6408, percent_type='ao')
b4c.add_element('C', 0.2, percent_type='ao')

b4c_frac = 0.0005
graph_frac = 1-b4c_frac

bgraphite = openmc.Material.mix_materials([mixgraph, b4c],
                                          [graph_frac, 
                                           b4c_frac], 'wo')

bgraphite.add_s_alpha_beta('c_Graphite')
bgraphite.temperature = periph_temp
bgraphite.name = 'bgraphite'


he = openmc.Material(name='He')
he.set_density('atom/b-cm', 0.0006)
he.add_element('He', 1.0, percent_type='ao')
he.temperature = periph_temp

ss_iron = openmc.Material(name='ss_fe')
ss_iron.add_element('Fe', 1.0, 'ao')
ss_iron.set_density('g/cm3', 7.8)
ss_iron.temperature = min_temp

mats = openmc.Materials([ndep1, ndep2, ndep3, ndep4, ndep5, ndep6,
                         triso_layer_mat, graphite, pebgraphite, 
                         bgraphite, he, ss_iron])

openmc.Materials(mats).export_to_xml('ndep_mats.xml')
