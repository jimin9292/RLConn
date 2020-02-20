
# coding: utf-8

import os
import platform

platform = platform.system()

default_dir = os.getcwd()

if platform == 'Windows':

	scripts_dir = default_dir + '\scripts'
	connectome_data_dir = default_dir + '\connectome_data'
	data_4_analysis_dir = default_dir + '\data_4_analysis'

else:

	scripts_dir = default_dir + '/scripts'
	connectome_data_dir = default_dir + '/connectome_data'
	data_4_analysis_dir = default_dir + '/data_4_analysis'

