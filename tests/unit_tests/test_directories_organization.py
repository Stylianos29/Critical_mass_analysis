import sys
import unittest

import pytest


sys.path.append('../')
import custom_library.directories_organization as directories_organization


if __name__ == "__main__":

#   unittest.main()
    data_files_directory="/nvme/h/cy22sg1/Plots/data_files/Pion_correlator_data_files"
    plotting_directory="/nvme/h/cy22sg1/Plots/Critical_mass_analysis/plots"
    
    plotting_subdirectories_paths_list = directories_organization.replicate_directory_structure(data_files_directory, plotting_directory)
