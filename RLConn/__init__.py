# coding: utf-8

# FUNCTION DOCUMENTATION GUIDELINES ######################################################################
# Each function follows following order of documentation:                                                #
##### 1. Brief description of what function does                                                         #
##### 2. Description on each argument and output                                                         # 
##### 3. If function consists of sevral steps with distinct roles, mark each step with brief explanation #
########################################################################################################## 

__author__ = 'Jimin Kim - jk55@u.washington.edu'
__version__ = '0.1.00_alpha'

from RLConn import sys_paths 
from RLConn import neural_params    
from RLConn import network_sim 
from RLConn import control_dqn
from RLConn import connectome_utils
from RLConn import utils
from RLConn import problem_definitions
from RLConn import plotting

# TODO: Integrate initial enviornment in global level