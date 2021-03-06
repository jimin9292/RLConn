
# coding: utf-8

############################################################
#PATHS FOR WINDOWS AND UNIX BASED OPERATING SYSTEMS#########
############################################################

import os
import platform

platform = platform.system()
default_dir = os.getcwd()

if platform == 'Windows':

    connectome_data_dir = 'RLConn\connectome_data'
    target_modes_dir = 'RLConn\modes_target'

else:

    connectome_data_dir = 'RLConn/connectome_data'
    target_modes_dir = 'RLConn/modes_target'


