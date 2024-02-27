import os
import re


def operator_type_extraction(operator_type_label):
    '''Extracts pieces of information from the working data file or subdirectory name. It returns the operator type ("Standard" or "Brillouin"), the operator method ("Chebyshev" or "KL" - or empty) and the a operator method "enumeration" (if an operator method was implemented), namely either the number of terms N for "Chebyshev" operator method, or the diagonal KL iteration\ for "KL".'''
 
    # Operator type
    if "Standard" in operator_type_label:
        operator_type = "Standard"
    else:
        operator_type = "Brillouin"
    
    # Operator method
    if 'Chebyshev' in operator_type_label:
        operator_method = 'Chebyshev'
    elif 'KL' in operator_type_label:
        operator_method = 'KL'
    else:
        operator_method = ""

    # Operator method enumeration
    operator_method_enumeration = ""
    # Check if operator method string is not empty
    if bool(operator_method):
        match = re.search(r'=(\d+)', operator_type_label)
        if match:
            operator_method_enumeration = int(match.group(1))
    
    return operator_type, operator_method, operator_method_enumeration


class AnalyzeDataset:

    def __init__(self, dataset_file_path, general_operator_type_info: tuple):

        self.dataset_file_path = dataset_file_path

        self.extract_info_from_filename()

        # Additional safety on whether the dataset operator type info is the same as the one extracted from the subdirectory name
        if (general_operator_type_info != self.operator_type_info):
            # TODO: Add action!
            pass
            
        self.extract_content_from_file()

        self.temporal_direction_lattice_size = len(self.contents_array)

    def extract_info_from_filename(self):

        self.operator_type_info = operator_type_extraction(self.dataset_file_path)

        # kappa value
        match = re.search(r'_kappa_(\d+\.\d+).dat', self.dataset_file_path)
        self.kappa = float(match.group(1))
            
        # configuration label
        match = re.search(r'_(\d+)_N_', self.dataset_file_path)
        if match:
            self.configuration_label = match.group(1)
        else:
            match = re.search(r'out_(\d+)_', self.dataset_file_path)
            self.configuration_label = match.group(1)

    def extract_content_from_file(self):

        # Pass the file content in a list as the dictionary value
        self.contents_array = list()
        with open(self.dataset_file_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                self.contents_array.append(float(line.strip("\n")))
