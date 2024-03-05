# import unittest
import os
import sys

import pytest


sys.path.append('../..')
import Critical_mass_analysis.custom_library.fit_functions as fit_functions


if __name__ == "__main__":

    # unittest.main()
    

    data_files_directory = '/nvme/h/cy22sg1/Plots/Critical_mass_analysis/data_files'

    data_files_extension = ".dat"

    
    data_files_list = [data_file for data_file in os.listdir(
            data_files_directory) if data_file.endswith(data_files_extension) ]

    for data_file in data_files_list:
        fit_functions.extract_info(data_file)
        



