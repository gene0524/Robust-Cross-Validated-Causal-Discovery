import os
import sys

# add current working directory to python path
current_dir = os.getcwd()
tcdf_dir = os.path.join(current_dir, 'src/models/TCDF_master')
sys.path.append(tcdf_dir)
