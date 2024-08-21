#!/usr/bin/env python3

# Copyright 2019 Jean-Luc Vay, Maxence Thevenet, Remi Lehe
#
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

import os
import re
import sys

import numpy as np
import scipy.constants as scc
import yt

yt.funcs.mylog.setLevel(0)
sys.path.insert(1, "../../../../warpx/Regression/Checksum/")
import checksumAPI

filename = sys.argv[1]

galilean = True if re.search("galilean", filename) else False

# Initial laser energy (at iteration 50)
if galilean:
    filename_init = "pml_x_galilean_plt000050"
    energy_start = 4.439376199524034e-08
else:
    filename_init = "pml_x_psatd_plt000050"
    energy_start = 7.282940107273505e-08

# Check consistency of field energy diagnostics with initial energy above
ds = yt.load(filename_init)
all_data_level_0 = ds.covering_grid(
    level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions
)
Bx = all_data_level_0["boxlib", "Bx"].v.squeeze()
By = all_data_level_0["boxlib", "By"].v.squeeze()
Bz = all_data_level_0["boxlib", "Bz"].v.squeeze()
Ex = all_data_level_0["boxlib", "Ex"].v.squeeze()
Ey = all_data_level_0["boxlib", "Ey"].v.squeeze()
Ez = all_data_level_0["boxlib", "Ez"].v.squeeze()
energyE = np.sum(0.5 * scc.epsilon_0 * (Ex**2 + Ey**2 + Ez**2))
energyB = np.sum(0.5 / scc.mu_0 * (Bx**2 + By**2 + Bz**2))
energy_start_diags = energyE + energyB
error = abs(energy_start - energy_start_diags) / energy_start
tolerance = 1e-14
print("energy_start expected = " + str(energy_start))
print("energy_start_diags    = " + str(energy_start_diags))
print("relative error    = " + str(error))
assert error < tolerance

# Final laser energy
ds = yt.load(filename)
all_data_level_0 = ds.covering_grid(
    level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions
)
Bx = all_data_level_0["boxlib", "Bx"].v.squeeze()
By = all_data_level_0["boxlib", "By"].v.squeeze()
Bz = all_data_level_0["boxlib", "Bz"].v.squeeze()
Ex = all_data_level_0["boxlib", "Ex"].v.squeeze()
Ey = all_data_level_0["boxlib", "Ey"].v.squeeze()
Ez = all_data_level_0["boxlib", "Ez"].v.squeeze()
energyE = np.sum(0.5 * scc.epsilon_0 * (Ex**2 + Ey**2 + Ez**2))
energyB = np.sum(0.5 / scc.mu_0 * (Bx**2 + By**2 + Bz**2))
energy_end = energyE + energyB

reflectivity = energy_end / energy_start_diags
reflectivity_max = 1e-6

print("reflectivity     = " + str(reflectivity))
print("reflectivity_max = " + str(reflectivity_max))

assert reflectivity < reflectivity_max

# Check restart data v. original data
sys.path.insert(0, "../../../../warpx/Examples/")
from analysis_default_restart import check_restart

if not galilean:
    check_restart(filename)

test_name = os.path.split(os.getcwd())[1]
checksumAPI.evaluate_checksum(test_name, filename)
