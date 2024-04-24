'''
Given a data files directory, this scripts looks for any files of a specified data files extension, ".dat" by default. Each of these data files are checked for their format and if verified, their relative path is passed to the dataframe.

'data_files_records_log.txt' logs any issues

'''

import os
import sys
import re

import pandas as pd
import numpy as np
import click


sys.path.append('../')
import custom_library.data_files_checks as data_files_checks
import custom_library.directories_organization as directories_organization
import custom_library.info_extraction as info_extraction


HOME_DIRECTORY = os.environ['HOME']


@click.command()
@click.option("--data_files_directory", "data_files_directory", "-data_dir", default=HOME_DIRECTORY+"/Data_analysis/data_files/Pion_correlator_data_files", help="N/A")
@click.option("--data_files_extension", "data_files_extension", "-data_ext", default=".dat", help="N/A")
@click.option("--output_files_directory", "output_files_directory", "-out_dir", default=HOME_DIRECTORY+"/Data_analysis/Critical_mass_analysis/analysis/output_files", help="N/A")
@click.option("--output_csv_file_name", "output_csv_file_name", "-out_csv", default="data_files_record.csv", help="N/A")
@click.option("--output_log_file_name", "output_log_file_name", "-out_log", default="data_files_records_log.txt", help="N/A")


def main(data_files_directory, data_files_extension, output_files_directory, output_csv_file_name, output_log_file_name):

    datasets_characteristics_dictionary = {
        'Data file relative path': list(),
        'Operator category': list(),
        'Operator method': list(),
        'Operator type': list(),
        'Operator enumeration': list(),
        'Bare mass': list(),
        'Optimization factor': list(),
        'Gauge configuration label': list(),
        'Beta value': list(),
        'Temporal lattice size': list(),
        'Spatial lattice size': list(),
        'APE smearing iterations': list(),
        'APE smearing alpha': list(),
        'Kappa value': list(),
        'Clover parameter': list(),
        'CG epsilon': list(),
        'Parallel geometry': list(),
        'Plaquette values': list(),
        'List of CG iterations for every vector inversion': list(),
        'List of times for every vector inversion': list(),
        'Total elapsed time': list()
        }

    # Any issue while analyzing the data files will logged in the 'data_files_records_log.txt' log file
    output_log_file_full_path = os.path.join(output_files_directory, output_log_file_name)
    number_of_issues_to_log = 0
    with open(output_log_file_full_path, 'w') as log_file:
        log_file.write(f'Working on data files...:\n')

        '''Loop over all subdirectories of any depth inside the data files directory looking for data files of the given extension. This way the search for the data files is independent of structure of subdirectories of the data files directory.'''
        for root, _, files in os.walk(data_files_directory):
            for file in files:
                # It is taken for granted though that only the data files from inversions end with the given extension
                if file.endswith(data_files_extension):
                    data_file_full_path = os.path.join(root, file)
                    # Extract data files relative path w.r.t. the data files directory
                    data_file_relative_path = directories_organization.extract_relative_path(data_file_full_path, data_files_directory)

                    # EXTRACT INFO FROM DATA FILE'S CONTENT

                    # Check if data file is empty. Ignore if empty
                    if os.path.getsize(data_file_full_path) == 0:
                        log_file.write(f' - Skipping. File "{data_file_relative_path}" is empty.\n')
                        number_of_issues_to_log += 1
                        continue

                    # Check if the data file has the expected format. Ignore if not
                    if not data_files_checks.data_file_format_validation(data_file_full_path):
                        log_file.write(f' - Skipping. Contents of data file "{data_file_relative_path}" are not of the expected format.\n')
                        number_of_issues_to_log += 1
                        continue

                    # The number of non-empty lines must be proportional to the lattice temporal direction size, T.
                    # NOTE: T value is later cross-checked using extracted info from the log file corresponding to this data file
                    lines_count_check = data_files_checks.data_file_non_empty_lines_count_validation(data_file_full_path)
                    if not lines_count_check: # Failure
                        log_file.write(f' - Skipping. File "{data_file_relative_path}" contains a non-consistent number of lines.\n')
                        number_of_issues_to_log += 1
                        continue
                    else:
                        # otherwise use the output of the function
                        temporal_lattice_size = lines_count_check

                    # Add to dataframe only verified data files
                    datasets_characteristics_dictionary['Data file relative path'].append(data_file_relative_path.lstrip('/'))
                    # TODO: Cross-check T first and then ass to dataframe
                    datasets_characteristics_dictionary['Temporal lattice size'].append(temporal_lattice_size)

                    # EXTRACT INFO FROM DATA FILE NAME

                    '''The assumption here is that the data filename contains useful information, at minimum the operator type, the bare mass value and the configuration label. For numerical values the eventuality of the use of "=" or "_" signs has been taken into consideration.'''
                    data_filename = os.path.basename(data_file_relative_path)

                    # Extract operator classification consisting of operator method, type and enumeration
                    operator_method_from_data_filename, operator_type_from_data_filename, operator_enumeration_from_data_filename = data_files_checks.extract_operator_classification(data_filename)

                    # The process stops if at least the operator type cannot be determined
                    if (operator_method_from_data_filename, operator_type_from_data_filename, operator_enumeration_from_data_filename) == (None, None, None):
                        log_file.write(f' - Skipping. Cannot extract information from the filename of the data file: "{data_file_relative_path}".\n')
                        number_of_issues_to_log += 1
                        continue
                    # Otherwise store the operator classification to the dataframe
                    datasets_characteristics_dictionary['Operator method'].append(operator_method_from_data_filename)
                    datasets_characteristics_dictionary['Operator type'].append(operator_type_from_data_filename)
                    datasets_characteristics_dictionary['Operator enumeration'].append(operator_enumeration_from_data_filename)
                    
                    # Extract bare mass value
                    bare_mass_value_from_data_filename = data_files_checks.extract_bare_mass_value(data_filename)
                    if bare_mass_value_from_data_filename is not None:
                        datasets_characteristics_dictionary['Bare mass'].append(bare_mass_value_from_data_filename)
                    else:
                        # Instead of 'None' pass 'NaN' to the dataframe
                        datasets_characteristics_dictionary['Bare mass'].append('NaN')
                        log_file.write(f' - Cannot extract "bare mass value" from data file: "{data_file_relative_path}".\n')
                        number_of_issues_to_log += 1

                    # Extract configuration label
                    configuration_label_from_data_filename = data_files_checks.extract_configuration_label(data_filename)
                    if configuration_label_from_data_filename is not None:
                        datasets_characteristics_dictionary['Gauge configuration label'].append(configuration_label_from_data_filename)
                    else:
                        # Pass 'None' to the dataframe
                        datasets_characteristics_dictionary['Gauge configuration label'].append(configuration_label_from_data_filename)
                        log_file.write(f' - Cannot extract "gauge configuration label" from data file: "{data_file_relative_path}".\n')
                        number_of_issues_to_log += 1

                    # Extract optimization factor, if it appears in the filename
                    optimization_factor_from_data_filename = data_files_checks.extract_optimization_factor(data_filename)
                    if optimization_factor_from_data_filename is not None:
                        datasets_characteristics_dictionary['Optimization factor'].append(optimization_factor_from_data_filename)
                    else:
                        # Pass 'None' to the dataframe
                        datasets_characteristics_dictionary['Optimization factor'].append(optimization_factor_from_data_filename)

                    # EXTRACT INFO FROM DATA FILE DIRECTORY PATH

                    ''' Next a similar attempt will be made to extract pieces of information from the relative path of the data file's subdirectory. However, this time it is mostly for verification purposes, while it is not even expected that this subdirectory differs from the root data files directory. If the extracted pieces of information do not coincide with the already extracted ones, a warning is issued.'''
                    data_file_relative_subdirectory = os.path.dirname(data_file_relative_path)
                    data_file_relative_subdirectory = data_file_relative_subdirectory.replace("/original_data_files", "")

                    # Check if the specific data file's subdirectory coincides with the overall data files directory. Ignore ff it does.
                    if data_file_relative_subdirectory == '/':
                        continue

                    # Extract operator classification
                    operator_method_from_data_file_subdirectory, operator_type_from_data_file_subdirectory, operator_enumeration_from_data_file_subdirectory = data_files_checks.extract_operator_classification(data_file_relative_subdirectory)

                    # If the operator classification can indeed be determined from the data file's subdirectory, then compare with corresponding pieces of info from filename
                    if (operator_method_from_data_file_subdirectory, operator_type_from_data_file_subdirectory, operator_enumeration_from_data_file_subdirectory) != (None, None, None):
                        # Cross-check operator method
                        if operator_method_from_data_file_subdirectory != operator_method_from_data_filename:
                            log_file.write(f' - Warning: Inconsistencies between filename and subdirectory path for "operator method" of data file "{data_file_relative_path}".\n')
                            number_of_issues_to_log += 1
                        # NOTE: If no inconsistencies arise, then store the operator method along with any accompanying substring set by the user to distinguish datasets, as 'operator category'. This is format specific and can be removed if necessary
                        else:
                            delimiter1 = f'/{operator_method_from_data_file_subdirectory}'
                            delimiter2 = '/'
                            pattern = re.compile(rf'{re.escape(delimiter1)}(.*?){re.escape(delimiter2)}')
                            match = pattern.search(data_file_relative_subdirectory)
                            if match:
                                operator_category_from_data_file_subdirectory = f'{operator_method_from_data_file_subdirectory}' + match.group(1)
                            else:
                                operator_category_from_data_file_subdirectory = None
                            
                            datasets_characteristics_dictionary['Operator category'].append(operator_category_from_data_file_subdirectory)
                        
                        # Cross-check operator type
                        if operator_type_from_data_file_subdirectory != operator_type_from_data_filename:
                            log_file.write(f' - Warning: Inconsistencies between filename and subdirectory path for "operator type" of data file "{data_file_relative_path}".\n')
                            number_of_issues_to_log += 1
                        
                        # Cross-check operator enumeration
                        if operator_enumeration_from_data_file_subdirectory != operator_enumeration_from_data_filename:
                            log_file.write(f' - Warning: Inconsistencies between filename and subdirectory path for "operator enumeration" of data file "{data_file_relative_path}".\n')
                            number_of_issues_to_log += 1

                    # Cross-check bare mass value
                    bare_mass_value_from_data_file_subdirectory = data_files_checks.extract_bare_mass_value(data_file_relative_subdirectory)
                    if bare_mass_value_from_data_file_subdirectory is not None:
                        if bare_mass_value_from_data_file_subdirectory != bare_mass_value_from_data_filename:
                            log_file.write(f' - Warning: Inconsistencies between filename and subdirectory path for "bare mass value" of data file "{data_file_relative_path}".\n')
                            number_of_issues_to_log += 1

                    # Cross-check optimization factor
                    optimization_factor_from_data_data_file_subdirectory = data_files_checks.extract_optimization_factor(data_file_relative_subdirectory)
                    if optimization_factor_from_data_data_file_subdirectory is not None:
                        if optimization_factor_from_data_data_file_subdirectory != optimization_factor_from_data_filename:
                            log_file.write(f' - Warning: Inconsistencies between filename and subdirectory path for "optimization factor" of data file "{data_file_relative_path}".\n')
                            number_of_issues_to_log += 1

                    # EXTRACT INFO FROM CORRESPONDING LOG FILE

                    '''Firstly, look for a corresponding log file to the data file. The assumption is here that it is located in a corresponding "log_files" leaf subdirectory, and that it has the same filename as the data file, apart from instead a ".txt" extension and lacking a preceding "out_" substring.'''
                    log_file_full_path = data_file_full_path.replace("original_data_files", "log_files")
                    log_file_full_path = log_file_full_path.replace("out_", "")
                    log_file_full_path = log_file_full_path.replace(data_files_extension, ".txt")
                    # Check if it exists. If it doesn't then simply ignore                     
                    if not os.path.exists(log_file_full_path):
                        datasets_characteristics_dictionary['Spatial lattice size'].append(None)
                        datasets_characteristics_dictionary['Parallel geometry'].append(None)
                        datasets_characteristics_dictionary['Beta value'].append(None)
                        datasets_characteristics_dictionary['APE smearing iterations'].append(None)
                        datasets_characteristics_dictionary['APE smearing alpha'].append(None)
                        datasets_characteristics_dictionary['Kappa value'].append(None)
                        datasets_characteristics_dictionary['Clover parameter'].append(None)
                        datasets_characteristics_dictionary['CG epsilon'].append(None)
                        datasets_characteristics_dictionary['Plaquette values'].append(None)
                        datasets_characteristics_dictionary['List of CG iterations for every vector inversion'].append(None)
                        datasets_characteristics_dictionary['List of times for every vector inversion'].append(None)
                        datasets_characteristics_dictionary['Total elapsed time'].append(None)

                        continue

                    invert_log_file_object = info_extraction.invert_log_file(log_file_full_path)

                    # Cross-check operator type
                    if operator_type_from_data_filename != invert_log_file_object.operator_type:
                        log_file.write(f' - Warning: Inconsistencies between filename and log file for "operator type" of data file "{data_file_relative_path}".\n')
                        number_of_issues_to_log += 1

                    # Cross-check T
                    if temporal_lattice_size != invert_log_file_object.lattice_dimensions[0]:
                        log_file.write(f' - Warning: Inconsistencies between filename and log file for "temporal lattice size" of data file "{data_file_relative_path}".\n')
                        number_of_issues_to_log += 1

                    datasets_characteristics_dictionary['Spatial lattice size'].append(invert_log_file_object.lattice_dimensions[1])
                    datasets_characteristics_dictionary['Parallel geometry'].append(invert_log_file_object.parallel_geometry)
                    datasets_characteristics_dictionary['Beta value'].append(invert_log_file_object.beta_value)
                    datasets_characteristics_dictionary['APE smearing iterations'].append(invert_log_file_object.APE_smearing_iterations)
                    datasets_characteristics_dictionary['APE smearing alpha'].append(invert_log_file_object.APE_smearing_alpha)
                    datasets_characteristics_dictionary['Kappa value'].append(invert_log_file_object.kappa)
                    datasets_characteristics_dictionary['Clover parameter'].append(invert_log_file_object.clover_parameter)
                    datasets_characteristics_dictionary['CG epsilon'].append(invert_log_file_object.CG_epsilon)
                    datasets_characteristics_dictionary['Plaquette values'].append(invert_log_file_object.plaquette)
                    datasets_characteristics_dictionary['List of CG iterations for every vector inversion'].append(invert_log_file_object.CG_iterations_list)
                    datasets_characteristics_dictionary['List of times for every vector inversion'].append(invert_log_file_object.time_per_vector_inversion_list)
                    datasets_characteristics_dictionary['Total elapsed time'].append(invert_log_file_object.total_time)

        if (number_of_issues_to_log == 0):
            log_file.write(f'...\n* No issues to report.\n')
        else:
            log_file.write(f'* A total of {number_of_issues_to_log} issues need attention with the registered data files.\n')

    df = pd.DataFrame(datasets_characteristics_dictionary)

    # TOTAL NUMBERS OF DATA FILES

    '''Next, a breakdown of the total number of files per 'Operator category', 'Operator type', 'Operator enumeration', and 'Bare mass' is exported in a text file.'''
    counts = df.groupby(['Operator category', 'Operator type', 'Operator enumeration', 'Bare mass']).size().reset_index(name='Count')

    pd.set_option('display.max_rows', None)

    counts_breakdown_file_full_path = os.path.join(output_files_directory, 'breakdown.txt')
    counts.to_csv(counts_breakdown_file_full_path, sep='\t', index=False)

    with open(counts_breakdown_file_full_path, 'r') as file:
        lines = file.readlines()

    with open(counts_breakdown_file_full_path, 'r+') as file:
        # Write the header line
        file.write(lines[0]+'  ')

        # Determine the maximum length of each column
        max_lengths = [max(len(str(value))+2 for value in counts[column]) for column in counts.columns]

        # Write the data rows with aligned columns
        for line in lines[1:]:
            parts = line.split('\t')
            aligned_line = '\t'.join(part.ljust(length) for part, length in zip(parts, max_lengths))
            file.write(aligned_line)

    # FILTER ROWS IN DATAFRAME

    '''If the number of data files per 'Operator category', 'Operator type', 'Operator enumeration', and 'Bare mass' are smaller than 2, then they need to be removed from the dataframe. Furthermore, the same if the number of unique 'Bare mass  values per 'Operator category', 'Operator type', and 'Operator enumeration' are smaller than 3.'''
    # Filter out groups where the maximum count of 'Gauge configuration label' values per 'Bare mass' is smaller than 3
    df = df.groupby(['Operator category', 'Operator type', 'Operator enumeration', 'Bare mass']).filter(lambda x: x['Gauge configuration label'].nunique() >= 2)

    # # Group by the specified columns and count the unique 'Gauge configuration label' values
    # counts_per_group = df.groupby(['Operator category', 'Operator type', 'Operator enumeration'])['Bare mass'].nunique()

    # Filter out groups where the maximum count of 'Gauge configuration label' values per 'Bare mass' is smaller than 3
    df = df.groupby(['Operator category', 'Operator type', 'Operator enumeration']).filter(lambda x: x['Bare mass'].nunique() >= 3)

    # # Group by the specified columns and count the unique 'Gauge configuration label' values
    counts_per_group = df.groupby(['Operator category', 'Operator type', 'Operator enumeration', 'Bare mass'])['Gauge configuration label'].nunique()

    # Find the minimum count of 'Gauge configuration label' values per 'Bare mass' for each group
    min_counts_per_group = counts_per_group.groupby(level=[0, 1, 2]).min()

    for (category, typ, enumeration), group in df.groupby(['Operator category', 'Operator type', 'Operator enumeration']):
        min_count = min_counts_per_group[(category, typ, enumeration)]
        for bare_mass_value, bare_mass_group in group.groupby('Bare mass'):
            if len(bare_mass_group) > min_count:
                indices = bare_mass_group.index.tolist()
                np.random.shuffle(indices)
                indices_to_remove = indices[min_count:]
                df = df.drop(indices_to_remove)

    # EXTRACT TO CSV FILE

    csv_file_full_path = os.path.join(output_files_directory, output_csv_file_name)

    df.to_csv(csv_file_full_path, index=False)

    # BREAKDOWN OF COUNTS FOR FILTERED DATAFRAME

    with open('./output_files/output.txt', 'w') as log_file:
        for operator_category, operator_category_group in df.groupby('Operator category'):
            log_file.write(f'{operator_category}:\n')
            for operator_type, operator_type_group in operator_category_group.groupby('Operator type'):
                log_file.write(f' *{operator_type}:\n')
                for operator_enumeration, operator_enumeration_group in operator_type_group.groupby('Operator enumeration'):
                    if (operator_enumeration != ''):
                        log_file.write(f'  **{operator_enumeration}:\n')
                    for bare_mass_value, bare_mass_value_group in operator_enumeration_group.groupby('Bare mass'):
                        bare_mass_value_elements = bare_mass_value_group['Gauge configuration label']
                        if bare_mass_value < 0:
                            log_file.write(f'   {bare_mass_value:.3f}: {bare_mass_value_elements.count()}\n')
                        else:
                            log_file.write(f'    {bare_mass_value:.3f}: {bare_mass_value_elements.count()}\n')

            log_file.write(f'\n')


if __name__ == "__main__":

    main()
