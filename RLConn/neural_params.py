# coding: utf-8

###########################################
#NEURAL PARAMETERS FOR NETWORK SIMULATIONS#
###########################################

"""
N: Number of Neurons
Gc: Cell membrane conductance (pS)
C: Cell Membrane Capacitance
ggap: Gap Junctions scaler (Electrical, 279*279)
gsyn: Synaptic connections scaler (Chemical, 279*279)
Ec: Leakage potential (mV) 
ar: Synaptic activity's rise time
ad: Synaptic activity's decay time
B: Width of the sigmoid (mv^-1)
rate: Rate for continuous stimuli transition
offset: Offset for continuous stimuli transition
init_fdb: Timepoint in seconds in which feedback initiates
t_delay: Time delay in seconds for the feedback 
"""

""" Standard values for fdb_init and t_delay """

# V2 connectome

# BWD = 1.48, 0.54 (with AVD)
# BWD = 1.55, 0.54 (transition to BWD from FWD)
# FWD = 1.13, 0.54
# FWD = 1.2, 0.54
# Waveforce FWD - 1.38, 0.53
# harsh touch FWD - 1.2, 0.54
# harsh touch BWD - 1.2, 0.54

# V3 connectome

# gentle_touch_posterior - PLM = 0.35, scaler = (0.5 ,2), fdb configs = (1.2, 0.54)
# gentle_touch_anterior - ALM = 0.68, AVM = 0.3, scaler = (0.5 ,2), fdb configs = (1.48, 0.54)
# harsh_touch_posterior - PVD, PVE = 0.65, scaler = (0.25 ,2), fdb configs = (1.2, 0.54)
# harsh_touch_anterior - TBA

import os
import numpy as np

from RLConn import sys_paths as paths

##################################
### PARAMETERS / CONFIGURATION ###
##################################

os.chdir(paths.connectome_data_dir)

""" Gap junctions (Chemical, 3*3) """
Gg_Static = np.load('artificial_Gg_sample.npy')

""" Synaptic connections (Chemical, 3*3) """
Gs_Static = np.load('artificial_Gs_sample.npy')

""" Directionality (279*1) """
#E = np.load('emask_v1.npy')
EMat_mask = np.load('artificial_E_sample.npy')

os.chdir(paths.default_dir)

os.chdir(paths.target_modes_dir)

m1_target = np.load('m1_target.npy')
m2_target = np.load('m2_target.npy')

m1_gt = np.load('m1_gt.npy')
m2_gt = np.load('m2_gt.npy')

os.chdir(paths.default_dir)

default = {

    "Gc" : 0.1,
    "C" : 0.015,
    "ggap" : 1.0,
    "gsyn" : 1.0,
    "Ec" : -35.0,
    "E_rev": -48.0, 
    "ar" : 1.0/1.5,
    "ad" : 5.0/1.5,
    "B" : 0.125,
    "rate" : 0.025,
    "offset" : 0.15,
    "iext" : 100000.,
    "init_key_counts" : 13

    }